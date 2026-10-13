//! ONNX inference for the vision agent (docs/vision-bot.md, M3): one session per model file per process, shared by
//! all agents; callers batch the frames of every vision agent of a world into one `run`.
//!
//! Model I/O (rl/vision/export.py): inputs `view` u8[B,2,256,256], `wide` u8[B,2,128,128]; outputs `heat`
//! f32[B,64,64] logits, `split`, `eject`, `value` f32[B]. A recurrent model (M5, `"gru": N` in its json) has one more input
//! `h` f32[B,N] (zeros for a new life) and one more output `h_out` f32[B,N]; the caller keeps the state per agent.
//!
//! Execution provider: spec `"ep"` or env `OGAR_VISION_EP`:
//!   `auto` (default)  `cuda`, or `cpu` when CUDA does not load
//!   `cuda`            CUDA EP with CUDA graphs; the batch is padded to a power of two, one captured graph per size
//!   `cuda_plain`      CUDA EP, a normal session run per call
//!   `cpu`             CPU EP (about 18 ms per frame: only for tests); intra-op threads: spec `"threads"` / env
//!                     `OGAR_VISION_THREADS` (default 1)
//! onnxruntime itself is loaded at run time (`runs/vision/ort/onnxruntime*.dll`, see `init_runtime`).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{IoBinding, Session};
use ort::value::Tensor;

/// Process-wide inference counters: (forward calls, frames, total ns inside `forward`).
pub static STATS: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
/// CUDA-graph path breakdown in ns: (copy in, graph replay + sync, copy out)
pub static PHASES: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

pub fn phases() -> (u64, u64, u64) {
    (PHASES[0].load(Ordering::Relaxed), PHASES[1].load(Ordering::Relaxed), PHASES[2].load(Ordering::Relaxed))
}

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

enum Backend {
    /// plain session run (CPU, or CUDA without graphs)
    Plain(Mutex<Session>),
    /// CUDA graphs: one session + pre-bound device buffers per padded batch size
    Graph(Mutex<Graphs>),
}

pub struct Net {
    backend: Backend,
    pub path: String,
    pub ep: String,
    /// split threshold stored next to the model by export.py (`<model>.json`), if any
    pub split_thr: Option<f32>,
    /// hidden size of the GRU (0 = not recurrent)
    pub rec: usize,
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
/// `runs/vision/ort/` (relative to the cwd, e.g. the repo root). The CUDA provider needs cuDNN 9 and CUDA 12 runtime
/// DLLs: directories from `OGAR_DLL_DIRS` (and the repo venv's torch wheel, `.venv/Lib/site-packages/torch/lib`) are
/// put on PATH first.
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
        let mut dirs: Vec<std::path::PathBuf> = std::env::var("OGAR_DLL_DIRS").map(|v| std::env::split_paths(&v).collect()).unwrap_or_default();
        dirs.push(std::path::PathBuf::from(".venv/Lib/site-packages/torch/lib"));
        let mut paths: Vec<std::path::PathBuf> = dirs.into_iter().filter(|d| d.is_dir()).filter_map(|d| d.canonicalize().ok()).collect();
        paths.extend(std::env::var_os("PATH").map(|p| std::env::split_paths(&p).collect::<Vec<_>>()).unwrap_or_default());
        if let Ok(joined) = std::env::join_paths(paths) {
            std::env::set_var("PATH", joined);
        }
        ort::init_from(&path).map_err(|e| format!("{}: {e}", path.display()))?.commit();
        Ok(())
    })
    .clone()
}

type CVoid = std::ffi::c_void;

/// The CUDA runtime calls the graph path needs (copies into / out of the buffers the graph is bound to).
struct Cudart {
    _lib: libloading::Library,
    memcpy: unsafe extern "C" fn(*mut CVoid, *const CVoid, usize, i32) -> i32,
    sync: unsafe extern "C" fn() -> i32,
}

impl Cudart {
    fn load() -> Result<Cudart, String> {
        unsafe {
            let name = if cfg!(windows) { "cudart64_12.dll" } else { "libcudart.so.12" };
            let lib = libloading::Library::new(name).map_err(|e| format!("{name}: {e} (set OGAR_DLL_DIRS)"))?;
            let memcpy = *lib.get::<unsafe extern "C" fn(*mut CVoid, *const CVoid, usize, i32) -> i32>(b"cudaMemcpy\0").map_err(|e| e.to_string())?;
            let sync = *lib.get::<unsafe extern "C" fn() -> i32>(b"cudaDeviceSynchronize\0").map_err(|e| e.to_string())?;
            Ok(Cudart { _lib: lib, memcpy, sync })
        }
    }

    fn h2d(&self, dst: *mut CVoid, src: &[u8]) -> Result<(), String> {
        match unsafe { (self.memcpy)(dst, src.as_ptr().cast(), src.len(), 1) } {
            0 => Ok(()),
            c => Err(format!("cudaMemcpy H2D failed: {c}")),
        }
    }

    fn d2h(&self, src: *mut CVoid, n: usize) -> Result<Vec<f32>, String> {
        let mut out = vec![0f32; n];
        match unsafe { (self.memcpy)(out.as_mut_ptr().cast(), src, n * 4, 2) } {
            0 => Ok(out),
            c => Err(format!("cudaMemcpy D2H failed: {c}")),
        }
    }

    fn sync(&self) -> Result<(), String> {
        match unsafe { (self.sync)() } {
            0 => Ok(()),
            c => Err(format!("cudaDeviceSynchronize failed: {c}")),
        }
    }
}

/// One CUDA-graph session for a fixed batch size: inputs and outputs are device tensors bound once; ORT captures the
/// graph on the second run and replays it afterwards (call overhead for 2 frames: 2.7 ms -> 0.9 ms).
struct GraphRun {
    session: Session,
    binding: IoBinding,
    p_view: *mut CVoid,
    p_wide: *mut CVoid,
    p_heat: *mut CVoid,
    p_split: *mut CVoid,
    p_eject: *mut CVoid,
    p_h: *mut CVoid,
    p_hout: *mut CVoid,
    // the tensors behind the raw pointers are owned by `binding`; the allocator must outlive them
    _alloc: Allocator,
}

struct Graphs {
    path: String,
    threads: usize,
    rec: usize,
    runs: HashMap<usize, GraphRun>,
    cudart: Cudart,
}

// The raw device pointers are only used under the Mutex that owns the Graphs.
unsafe impl Send for Graphs {}

fn cuda_session(path: &str, threads: usize, graph: bool) -> Result<Session, String> {
    let e = |e: &dyn std::fmt::Display| format!("{path}: {e}");
    Session::builder()
        .map_err(|x| e(&x))?
        .with_intra_threads(threads)
        .map_err(|x| e(&x))?
        .with_execution_providers([ort::ep::CUDA::default()
            .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
            .with_cuda_graph(graph)
            .build()
            .error_on_failure()])
        .map_err(|x| e(&x))?
        .commit_from_file(path)
        .map_err(|x| e(&x))
}

impl GraphRun {
    fn new(path: &str, threads: usize, b: usize, rec: usize, cudart: &Cudart) -> Result<GraphRun, String> {
        let e = |e: &dyn std::fmt::Display| format!("{path}: {e}");
        let session = cuda_session(path, threads, true)?;
        let alloc = Allocator::new(
            &session,
            MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default).map_err(|x| e(&x))?,
        )
        .map_err(|x| e(&x))?;
        let mut binding = session.create_binding().map_err(|x| e(&x))?;
        let mut d_view = Tensor::<u8>::new(&alloc, [b, 2, PX, PX]).map_err(|x| e(&x))?;
        let mut d_wide = Tensor::<u8>::new(&alloc, [b, 2, WPX, WPX]).map_err(|x| e(&x))?;
        let mut d_heat = Tensor::<f32>::new(&alloc, [b, HEAT, HEAT]).map_err(|x| e(&x))?;
        let mut d_split = Tensor::<f32>::new(&alloc, [b]).map_err(|x| e(&x))?;
        let mut d_eject = Tensor::<f32>::new(&alloc, [b]).map_err(|x| e(&x))?;
        let d_value = Tensor::<f32>::new(&alloc, [b]).map_err(|x| e(&x))?;
        let mut d_h = Tensor::<f32>::new(&alloc, [b, rec.max(1)]).map_err(|x| e(&x))?;
        let mut d_hout = Tensor::<f32>::new(&alloc, [b, rec.max(1)]).map_err(|x| e(&x))?;
        let (p_h, p_hout) = (d_h.data_ptr_mut(), d_hout.data_ptr_mut());
        let (p_view, p_wide) = (d_view.data_ptr_mut(), d_wide.data_ptr_mut());
        let (p_heat, p_split, p_eject) = (d_heat.data_ptr_mut(), d_split.data_ptr_mut(), d_eject.data_ptr_mut());
        // zero inputs for the capture runs
        cudart.h2d(p_view, &vec![0u8; b * 2 * PX * PX])?;
        cudart.h2d(p_wide, &vec![0u8; b * 2 * WPX * WPX])?;
        if rec > 0 {
            cudart.h2d(p_h, &vec![0u8; b * rec * 4])?;
        }
        cudart.sync()?;
        binding.bind_input("view", &d_view).map_err(|x| e(&x))?;
        binding.bind_input("wide", &d_wide).map_err(|x| e(&x))?;
        binding.bind_output("heat", d_heat).map_err(|x| e(&x))?;
        binding.bind_output("split", d_split).map_err(|x| e(&x))?;
        binding.bind_output("eject", d_eject).map_err(|x| e(&x))?;
        binding.bind_output("value", d_value).map_err(|x| e(&x))?;
        if rec > 0 {
            binding.bind_input("h", &d_h).map_err(|x| e(&x))?;
            binding.bind_output("h_out", d_hout).map_err(|x| e(&x))?;
        }
        let mut g = GraphRun { session, binding, p_view, p_wide, p_heat, p_split, p_eject, p_h, p_hout, _alloc: alloc };
        for _ in 0..3 {
            g.session.run_binding(&g.binding).map_err(|x| e(&x))?; // warm-up, capture, first replay
        }
        Ok(g)
    }

    fn run(&mut self, n: usize, view: &[u8], wide: &[u8], h: &[f32], rec: usize, c: &Cudart) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let t0 = Instant::now();
        c.h2d(self.p_view, view)?;
        c.h2d(self.p_wide, wide)?;
        if rec > 0 {
            let bytes = unsafe { std::slice::from_raw_parts(h.as_ptr().cast::<u8>(), h.len() * 4) };
            c.h2d(self.p_h, bytes)?;
        }
        c.sync()?;
        let t1 = Instant::now();
        self.session.run_binding(&self.binding).map_err(|x| x.to_string())?;
        c.sync()?;
        let t2 = Instant::now();
        let r = (c.d2h(self.p_heat, n * HEAT * HEAT)?, c.d2h(self.p_split, n)?, c.d2h(self.p_eject, n)?, if rec > 0 { c.d2h(self.p_hout, n * rec)? } else { Vec::new() });
        let t3 = Instant::now();
        PHASES[0].fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
        PHASES[1].fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
        PHASES[2].fetch_add((t3 - t2).as_nanos() as u64, Ordering::Relaxed);
        Ok(r)
    }
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
        let meta = std::fs::read_to_string(std::path::Path::new(path).with_extension("json")).ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok());
        let rec = meta.as_ref().and_then(|v| v["cfg"]["gru"].as_u64()).unwrap_or(0) as usize;
        let e = |e: &dyn std::fmt::Display| format!("{path}: {e}");
        let cpu = || -> Result<Backend, String> {
            Ok(Backend::Plain(Mutex::new(
                Session::builder().map_err(|x| e(&x))?.with_intra_threads(threads).map_err(|x| e(&x))?.commit_from_file(path).map_err(|x| e(&x))?,
            )))
        };
        let cuda = || -> Result<Backend, String> {
            let cudart = Cudart::load()?;
            let mut runs = HashMap::new();
            runs.insert(2, GraphRun::new(path, threads, 2, rec, &cudart)?); // fail early if CUDA does not work
            Ok(Backend::Graph(Mutex::new(Graphs { path: path.to_string(), threads, rec, runs, cudart })))
        };
        let backend = match ep {
            "cpu" => cpu()?,
            "cuda_plain" => Backend::Plain(Mutex::new(cuda_session(path, threads, false)?)),
            "cuda" => cuda()?,
            "auto" => cuda().or_else(|err| {
                eprintln!("[vision] CUDA unavailable ({err}); running {path} on the CPU");
                cpu()
            })?,
            _ => return Err(format!("unknown execution provider {ep:?} (auto | cpu | cuda | cuda_plain)")),
        };
        let split_thr = std::fs::read_to_string(std::path::Path::new(path).with_extension("json"))
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .and_then(|v| v["split_thr"].as_f64())
            .map(|v| v as f32);
        let n = Arc::new(Net { backend, path: path.to_string(), ep: ep.to_string(), split_thr, rec });
        c.insert(key, n.clone());
        Ok(n)
    }

    /// The batch size this net actually runs for `b` frames (CUDA graphs: the next power of two).
    pub fn padded_batch(&self, b: usize) -> usize {
        match self.backend {
            Backend::Graph(_) => b.next_power_of_two(),
            Backend::Plain(_) => b,
        }
    }

    /// Raw forward pass for `b` frame stacks: view [b,2,PX,PX], wide [b,2,WPX,WPX] -> heat [b,HEAT*HEAT], split, eject.
    /// `h`: hidden states [b, rec] (ignored when `rec` = 0); returns heat, split, eject and the new hidden states.
    pub fn forward(&self, b: usize, view: &[u8], wide: &[u8], h: &[f32]) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), String> {
        assert_eq!(h.len(), b * self.rec);
        assert_eq!(view.len(), b * 2 * PX * PX);
        assert_eq!(wide.len(), b * 2 * WPX * WPX);
        let e = |e: &dyn std::fmt::Display| format!("{}: {e}", self.path);
        let t0 = Instant::now();
        let r = match &self.backend {
            Backend::Plain(session) => {
                let tv = Tensor::from_array(([b, 2, PX, PX], view.to_vec())).map_err(|x| e(&x))?;
                let tw = Tensor::from_array(([b, 2, WPX, WPX], wide.to_vec())).map_err(|x| e(&x))?;
                let mut s = session.lock().unwrap();
                let out = if self.rec > 0 {
                    let th = Tensor::from_array(([b, self.rec], h.to_vec())).map_err(|x| e(&x))?;
                    s.run(ort::inputs!["view" => tv, "wide" => tw, "h" => th]).map_err(|x| e(&x))?
                } else {
                    s.run(ort::inputs!["view" => tv, "wide" => tw]).map_err(|x| e(&x))?
                };
                let get = |name: &str| -> Result<Vec<f32>, String> {
                    let (_, d) = out[name].try_extract_tensor::<f32>().map_err(|x| e(&x))?;
                    Ok(d.to_vec())
                };
                (get("heat")?, get("split")?, get("eject")?, if self.rec > 0 { get("h_out")? } else { Vec::new() })
            }
            Backend::Graph(g) => {
                let mut g = g.lock().unwrap();
                let g = &mut *g;
                let pb = b.next_power_of_two();
                if !g.runs.contains_key(&pb) {
                    let run = GraphRun::new(&g.path, g.threads, pb, g.rec, &g.cudart)?;
                    g.runs.insert(pb, run);
                }
                let (mut vp, mut wp);
                let (view, wide) = if pb == b {
                    (view, wide)
                } else {
                    vp = view.to_vec();
                    vp.resize(pb * 2 * PX * PX, 0);
                    wp = wide.to_vec();
                    wp.resize(pb * 2 * WPX * WPX, 0);
                    (&vp[..], &wp[..])
                };
                let run = g.runs.get_mut(&pb).unwrap();
                let mut hp = h.to_vec();
                hp.resize(pb * g.rec, 0.0);
                let (heat, split, eject, hout) = run.run(pb, view, wide, &hp, g.rec, &g.cudart)?;
                (heat[..b * HEAT * HEAT].to_vec(), split[..b].to_vec(), eject[..b].to_vec(), if g.rec > 0 { hout[..b * g.rec].to_vec() } else { Vec::new() })
            }
        };
        STATS[0].fetch_add(1, Ordering::Relaxed);
        STATS[1].fetch_add(b as u64, Ordering::Relaxed);
        STATS[2].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(r)
    }

    /// Decoded outputs and the new hidden states [b, rec] (empty when not recurrent).
    pub fn infer(&self, b: usize, view: &[u8], wide: &[u8], h: &[f32]) -> Result<(Vec<NetOut>, Vec<f32>), String> {
        let (heat, split, eject, hout) = self.forward(b, view, wide, h)?;
        let outs = (0..b)
            .map(|i| NetOut {
                target: decode_target(&heat[i * HEAT * HEAT..(i + 1) * HEAT * HEAT]),
                split_logit: split[i],
                eject_logit: eject[i],
            })
            .collect();
        Ok((outs, hout))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_peak_cell_centre_and_border_pad() {
        // single peak at cell (iy=10, ix=20): neighbours far lower -> the cell centre, (ix + 0.5) * 4
        let mut heat = vec![-10.0f32; HEAT * HEAT];
        heat[10 * HEAT + 20] = 10.0;
        let (u, v) = decode_target(&heat);
        assert!((u - 82.0).abs() < 1e-2 && (v - 42.0).abs() < 1e-2, "{u} {v}");
        // two equal cells side by side: argmax = the first one, the soft-argmax moves half a cell towards the second
        heat[10 * HEAT + 21] = 10.0;
        let (u, _) = decode_target(&heat);
        assert!((u - 84.0).abs() < 0.1, "{u}");
        // corner peak: the padding (-1e4) must not leak outside the map
        let mut h2 = vec![-10.0f32; HEAT * HEAT];
        h2[0] = 5.0;
        let (u, v) = decode_target(&h2);
        assert!(u >= 0.0 && v >= 0.0 && u < 4.0 && v < 4.0, "{u} {v}");
    }
}
