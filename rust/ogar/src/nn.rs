//! ONNX inference for the vision agent (docs/vision-bot.md, M3): one session per model file per process, shared by
//! all agents; callers batch the frames of every vision agent of a world into one `run`.
//!
//! Model I/O (rl/vision/export.py): inputs `view` u8[B,2,256,256], `wide` u8[B,2,128,128]; outputs `heat`
//! f32[B,64,64] logits, `split`, `eject`, `value` f32[B]. Execution provider: spec `"ep"` or env `OGAR_VISION_EP`
//! (`cpu` default, `cuda`); intra-op threads: spec `"threads"` or env `OGAR_VISION_THREADS` (default 1).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use ort::session::Session;
use ort::value::Tensor;

/// Process-wide inference counters: (forward calls, frames, total ns inside `forward`).
pub static STATS: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

pub fn stats() -> (u64, u64, u64) {
    (STATS[0].load(Ordering::Relaxed), STATS[1].load(Ordering::Relaxed), STATS[2].load(Ordering::Relaxed))
}

pub const PX: usize = 256;
pub const WPX: usize = PX / 2;
pub const HEAT: usize = 64;

/// What the net says for one frame stack.
#[derive(Clone, Copy, Debug)]
pub struct NetOut {
    /// cursor target in view-frame pixels (u, v), 0..PX
    pub target: (f32, f32),
    pub split_logit: f32,
    pub eject_logit: f32,
}

pub struct Net {
    session: Mutex<Session>,
    pub path: String,
    pub ep: String,
    /// split threshold stored next to the model by export.py (`<model>.json`), if any
    pub split_thr: Option<f32>,
}

/// rl/vision/model.py `decode_target` for one heat map [HEAT*HEAT]: argmax cell refined by a 3x3 soft-argmax.
pub fn decode_target(heat: &[f32]) -> (f32, f32) {
    let h = HEAT;
    let (mut best, mut idx) = (f32::NEG_INFINITY, 0usize);
    for (i, &v) in heat.iter().enumerate() {
        if v > best {
            best = v;
            idx = i;
        }
    }
    let (iy, ix) = ((idx / h) as i32, (idx % h) as i32);
    let at = |y: i32, x: i32| -> f32 {
        if y < 0 || x < 0 || y >= h as i32 || x >= h as i32 {
            -1e4
        } else {
            heat[y as usize * h + x as usize]
        }
    };
    let mut patch = [0f32; 9];
    for (j, p) in patch.iter_mut().enumerate() {
        *p = at(iy + (j / 3) as i32 - 1, ix + (j % 3) as i32 - 1);
    }
    let m = patch.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut e = [0f32; 9];
    let mut sum = 0f32;
    for j in 0..9 {
        e[j] = (patch[j] - m).exp();
        sum += e[j];
    }
    let (mut dx, mut dy) = (0f32, 0f32);
    for j in 0..9 {
        let w = e[j] / sum;
        dy += w * ((j / 3) as f32 - 1.0);
        dx += w * ((j % 3) as f32 - 1.0);
    }
    let cell = PX as f32 / h as f32;
    (((ix as f32 + 0.5 + dx) * cell).clamp(0.0, PX as f32), ((iy as f32 + 0.5 + dy) * cell).clamp(0.0, PX as f32))
}

fn cache() -> &'static Mutex<HashMap<String, Arc<Net>>> {
    static C: OnceLock<Mutex<HashMap<String, Arc<Net>>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Load the onnxruntime shared library once: `ORT_DYLIB_PATH`, else `onnxruntime.dll` next to the exe, else
/// `runs/vision/ort/` (relative to the cwd, e.g. the repo root).
fn init_runtime() -> Result<(), String> {
    static INIT: OnceLock<Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| {
        let lib = if cfg!(windows) { "onnxruntime.dll" } else { "libonnxruntime.so" };
        let mut cands: Vec<std::path::PathBuf> = Vec::new();
        if let Ok(p) = std::env::var("ORT_DYLIB_PATH") {
            cands.push(p.into());
        }
        if let Some(d) = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.to_path_buf())) {
            cands.push(d.join(lib));
        }
        cands.push(std::path::Path::new("runs/vision/ort").join(lib));
        let path = cands.iter().find(|p| p.exists()).ok_or(format!("{lib} not found (set ORT_DYLIB_PATH); tried {cands:?}"))?;
        let path = path.canonicalize().map_err(|e| e.to_string())?;
        ort::init_from(&path).map_err(|e| format!("{}: {e}", path.display()))?.commit();
        Ok(())
    })
    .clone()
}

impl Net {
    /// The shared session for (model, ep, threads); loaded on first use.
    pub fn load(path: &str, ep: &str, threads: usize) -> Result<Arc<Net>, String> {
        let key = format!("{path}|{ep}|{threads}");
        let mut c = cache().lock().unwrap();
        if let Some(n) = c.get(&key) {
            return Ok(n.clone());
        }
        init_runtime()?;
        let e = |e: &dyn std::fmt::Display| format!("{path}: {e}");
        let mut b = Session::builder().map_err(|x| e(&x))?.with_intra_threads(threads).map_err(|x| e(&x))?;
        if ep == "cuda" {
            b = b
                .with_execution_providers([ort::ep::CUDA::default()
                    .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
                    .build()
                    .error_on_failure()])
                .map_err(|x| e(&x))?;
        } else if ep != "cpu" {
            return Err(format!("unknown execution provider {ep:?} (cpu | cuda)"));
        }
        let session = b.commit_from_file(path).map_err(|x| e(&x))?;
        let split_thr = std::fs::read_to_string(std::path::Path::new(path).with_extension("json"))
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .and_then(|v| v["split_thr"].as_f64())
            .map(|v| v as f32);
        let n = Arc::new(Net { session: Mutex::new(session), path: path.to_string(), ep: ep.to_string(), split_thr });
        c.insert(key, n.clone());
        Ok(n)
    }

    /// Raw forward pass for `b` frame stacks: view [b,2,PX,PX], wide [b,2,WPX,WPX] -> heat [b,HEAT*HEAT], split, eject.
    pub fn forward(&self, b: usize, view: &[u8], wide: &[u8]) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        assert_eq!(view.len(), b * 2 * PX * PX);
        assert_eq!(wide.len(), b * 2 * WPX * WPX);
        let e = |e: &dyn std::fmt::Display| format!("{}: {e}", self.path);
        let t0 = Instant::now();
        let tv = Tensor::from_array(([b, 2, PX, PX], view.to_vec())).map_err(|x| e(&x))?;
        let tw = Tensor::from_array(([b, 2, WPX, WPX], wide.to_vec())).map_err(|x| e(&x))?;
        let mut s = self.session.lock().unwrap();
        let out = s.run(ort::inputs!["view" => tv, "wide" => tw]).map_err(|x| e(&x))?;
        let get = |name: &str| -> Result<Vec<f32>, String> {
            let (_, d) = out[name].try_extract_tensor::<f32>().map_err(|x| e(&x))?;
            Ok(d.to_vec())
        };
        let r = (get("heat")?, get("split")?, get("eject")?);
        STATS[0].fetch_add(1, Ordering::Relaxed);
        STATS[1].fetch_add(b as u64, Ordering::Relaxed);
        STATS[2].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(r)
    }

    pub fn infer(&self, b: usize, view: &[u8], wide: &[u8]) -> Result<Vec<NetOut>, String> {
        let (heat, split, eject) = self.forward(b, view, wide)?;
        Ok((0..b)
            .map(|i| NetOut {
                target: decode_target(&heat[i * HEAT * HEAT..(i + 1) * HEAT * HEAT]),
                split_logit: split[i],
                eject_logit: eject[i],
            })
            .collect())
    }
}
