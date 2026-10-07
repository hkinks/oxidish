//! Python bindings for `ogar::vision::VecEnv` (see docs/vision-bot.md, M1).
//!
//! ```python
//! env = ogar_py.VecEnv(32, 4, opponents=[("c34", "bot/c3_4.json", 4)], teacher="bot/plan_c34.json")
//! obs = env.reset()                       # {"view": u8[N,2,px,px], "wide": u8[N,2,px/2,px/2]}
//! obs, reward, done, info = env.step(actions)   # actions f32[N,4] = (u, v, split, eject), u/v in view pixels
//! info["teacher"]                         # f32[N,4]: the teacher's decision for the *current* obs (BC label)
//! ```
//! Arrays are zero-copy views of the env's buffers: valid until the next `step`/`reset` (pass `copy=True` to get
//! private copies).

use numpy::ndarray::{ArrayView1, ArrayView2, ArrayView4};
use numpy::{PyArray1, PyArray2, PyArray4, PyReadonlyArray2, PyUntypedArrayMethods};
use ogar::vision::{VecEnv as Core, VisionOpts};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn load_spec(path: &str) -> PyResult<serde_json::Value> {
    let text = std::fs::read_to_string(path).map_err(|e| PyIOError::new_err(format!("{path}: {e}")))?;
    serde_json::from_str(&text).map_err(|e| PyValueError::new_err(format!("{path}: {e}")))
}

#[pyclass]
struct VecEnv {
    core: std::sync::Mutex<Core>,
}

#[pymethods]
impl VecEnv {
    #[new]
    #[pyo3(signature = (n_worlds, learners_per_world, opponents=None, teacher="bot/plan_c34.json".to_string(), px=256, sync=2,
                        bots=8, food_mult=2.0, map=7000.0, seed=1, threads=1, warmup=0, label=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_worlds: usize,
        learners_per_world: usize,
        opponents: Option<Vec<(String, String, usize)>>,
        teacher: String,
        px: usize,
        sync: u32,
        bots: usize,
        food_mult: f64,
        map: f64,
        seed: u64,
        threads: usize,
        warmup: u32,
        label: bool,
    ) -> PyResult<Self> {
        if n_worlds == 0 || learners_per_world == 0 || px < 8 || px % 2 != 0 || sync == 0 {
            return Err(PyValueError::new_err("need n_worlds, learners_per_world >= 1, even px >= 8, sync >= 1"));
        }
        let mut o = VisionOpts::new(load_spec(&teacher)?);
        o.n_worlds = n_worlds;
        o.learners_per_world = learners_per_world;
        for (name, path, count) in opponents.unwrap_or_default() {
            o.opponents.push((name, load_spec(&path)?, count));
        }
        o.px = px;
        o.sync = sync;
        o.bots = bots;
        o.food_mult = food_mult;
        o.map = map;
        o.seed = seed;
        o.threads = threads.max(1);
        o.warmup = warmup;
        o.label = label;
        Ok(VecEnv { core: std::sync::Mutex::new(Core::new(o)) })
    }

    /// number of learners N = n_worlds * learners_per_world
    #[getter]
    fn n(&self) -> usize {
        self.core.lock().unwrap().n
    }

    #[getter]
    fn px(&self) -> usize {
        self.core.lock().unwrap().opts.px
    }

    /// New worlds; returns the observation dict.
    #[pyo3(signature = (copy=false))]
    fn reset<'py>(slf: &Bound<'py, Self>, py: Python<'py>, copy: bool) -> PyResult<Bound<'py, PyDict>> {
        {
            let me = slf.borrow();
            let mut core = me.core.lock().unwrap();
            let c: &mut Core = &mut core;
            py.allow_threads(|| c.reset());
        }
        obs_dict(slf, py, copy)
    }

    /// actions: float32 [N, 4] = (u, v, split, eject). Returns (obs, reward f32[N], done u8[N], info).
    #[pyo3(signature = (actions, copy=false))]
    fn step<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        actions: PyReadonlyArray2<'py, f32>,
        copy: bool,
    ) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyDict>)> {
        let n = slf.borrow().core.lock().unwrap().n;
        if actions.shape() != [n, 4] {
            return Err(PyValueError::new_err(format!("actions must be float32 [{n}, 4], got {:?}", actions.shape())));
        }
        let acts: Vec<f32> = actions.as_array().iter().copied().collect();
        {
            let me = slf.borrow();
            let mut core = me.core.lock().unwrap();
            let c: &mut Core = &mut core;
            py.allow_threads(|| c.step(&acts));
        }
        let obs = obs_dict(slf, py, copy)?;
        let (reward, done) = {
            let me = slf.borrow();
            let guard = me.core.lock().unwrap();
            let c = &*guard;
            (
                vec1(slf, py, ArrayView1::from(&c.reward[..]), copy),
                vec1(slf, py, ArrayView1::from(&c.done[..]), copy),
            )
        };
        let info = info_dict(slf, py, copy)?;
        Ok((obs, reward, done, info))
    }

    /// Per-learner switch: True / 1 = the teacher drives that learner (its action is ignored). A bool sets all.
    fn set_teacher(&mut self, mask: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(b) = mask.extract::<bool>() {
            self.core.lock().unwrap().teacher_mask.iter_mut().for_each(|m| *m = b as u8);
            return Ok(());
        }
        let v: Vec<bool> = mask.extract()?;
        if v.len() != self.core.lock().unwrap().n {
            return Err(PyValueError::new_err(format!("mask must have {} entries", self.core.lock().unwrap().n)));
        }
        for (m, b) in self.core.lock().unwrap().teacher_mask.iter_mut().zip(v) {
            *m = b as u8;
        }
        Ok(())
    }

    /// Wall-time breakdown in seconds summed over all worlds (divide by threads for wall time).
    fn timing<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let t = self.core.lock().unwrap().timing();
        let d = PyDict::new(py);
        for (k, v) in [
            ("tick", t.tick),
            ("opponents", t.opponents),
            ("snapshot", t.snapshot),
            ("render", t.render),
            ("teacher", t.teacher),
        ] {
            d.set_item(k, v as f64 * 1e-9)?;
        }
        d.set_item("decisions", t.decisions)?;
        Ok(d)
    }
}

fn finish<'py, T: numpy::Element + Copy>(arr: Bound<'py, PyAny>, copy: bool) -> Bound<'py, PyAny> {
    if copy {
        arr.call_method0("copy").unwrap()
    } else {
        arr
    }
}

fn vec1<'py, T: numpy::Element + Copy>(slf: &Bound<'py, VecEnv>, _py: Python<'py>, a: ArrayView1<'_, T>, copy: bool) -> Bound<'py, PyAny> {
    // SAFETY: the buffer lives in `slf` (the base object) and is never reallocated.
    let arr = unsafe { PyArray1::borrow_from_array(&a, slf.clone().into_any()) };
    finish::<T>(arr.into_any(), copy)
}

fn obs_dict<'py>(slf: &Bound<'py, VecEnv>, py: Python<'py>, copy: bool) -> PyResult<Bound<'py, PyDict>> {
    let me = slf.borrow();
    let guard = me.core.lock().unwrap();
    let c = &*guard;
    let (n, px, wpx) = (c.n, c.opts.px, c.wpx);
    let view = ArrayView4::from_shape((n, 2, px, px), &c.view).unwrap();
    let wide = ArrayView4::from_shape((n, 2, wpx, wpx), &c.wide).unwrap();
    // SAFETY: as in vec1.
    let (v, w) = unsafe {
        (PyArray4::borrow_from_array(&view, slf.clone().into_any()), PyArray4::borrow_from_array(&wide, slf.clone().into_any()))
    };
    let d = PyDict::new(py);
    d.set_item("view", finish::<u8>(v.into_any(), copy))?;
    d.set_item("wide", finish::<u8>(w.into_any(), copy))?;
    Ok(d)
}

fn info_dict<'py>(slf: &Bound<'py, VecEnv>, py: Python<'py>, copy: bool) -> PyResult<Bound<'py, PyDict>> {
    let me = slf.borrow();
    let guard = me.core.lock().unwrap();
    let c = &*guard;
    let d = PyDict::new(py);
    let ta: ArrayView2<f32> = ArrayView2::from_shape((c.n, 4), &c.t_act).unwrap();
    // SAFETY: as in vec1.
    let ta = unsafe { PyArray2::borrow_from_array(&ta, slf.clone().into_any()) };
    d.set_item("teacher", finish::<f32>(ta.into_any(), copy))?;
    d.set_item("mass", vec1(slf, py, ArrayView1::from(&c.mass[..]), copy))?;
    d.set_item("rank", vec1(slf, py, ArrayView1::from(&c.rank[..]), copy))?;
    d.set_item("leader", vec1(slf, py, ArrayView1::from(&c.leader[..]), copy))?;
    d.set_item("ncells", vec1(slf, py, ArrayView1::from(&c.ncells[..]), copy))?;
    Ok(d)
}

#[pymodule]
fn ogar_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VecEnv>()?;
    Ok(())
}
