//! The "vision" agent kind (docs/vision-bot.md, M3): decides from rendered grayscale frames with an ONNX net.
//! Spec: `{"kind":"vision","model":"runs/vision/export/vision_bc3.onnx","split":false}` (+ optional `split_thr`,
//! `ep` = cpu|cuda, `threads`, `eject`). Frames are made exactly as the VecEnv makes them (vision.rs `observe`):
//! `view` = the camera square at 256 px, `wide` = 2x that extent at 128 px, channel 0 = this decision,
//! channel 1 = the previous decision (a new life starts with both equal). The net's target pixel is mapped to
//! the world with the VecEnv's camera (`Camera::pixel_to_world`).

use std::sync::Arc;

use arena_client::game::Snapshot;
use arena_client::policy::Out;
use serde_json::Value;

use crate::nn::{Net, NetOut, PX, WPX};
use crate::vision::FrameStack;
use crate::world::World;

pub struct VisionBot {
    pub net: Arc<Net>,
    pub split: bool,
    pub split_thr: f32,
    pub eject: bool,
    pub frames: FrameStack,
    prepared: bool,
    /// the net's answer for the prepared frames (set by the batch pass)
    pending: Option<NetOut>,
    /// a commander's `split on`: allow the net's split decisions although the spec has `split: false`
    pub split_override: Option<bool>,
    pub decisions: u64,
    /// the net output of the latest decision (frames of that decision are still in `frames`)
    pub last: Option<NetOut>,
}

impl VisionBot {
    pub fn from_spec(spec: &Value) -> Result<VisionBot, String> {
        let model = spec["model"].as_str().ok_or("vision spec needs \"model\"")?;
        let ep = spec["ep"].as_str().map(String::from).or_else(|| std::env::var("OGAR_VISION_EP").ok()).unwrap_or_else(|| "cuda".into());
        let threads = spec["threads"]
            .as_u64()
            .map(|t| t as usize)
            .or_else(|| std::env::var("OGAR_VISION_THREADS").ok().and_then(|t| t.parse().ok()))
            .unwrap_or(1);
        let net = Net::load(model, &ep, threads)?;
        let split_thr = spec["split_thr"].as_f64().map(|v| v as f32).or(net.split_thr).unwrap_or(0.9);
        Ok(VisionBot::new(net, spec["split"].as_bool().unwrap_or(false), split_thr, spec["eject"].as_bool().unwrap_or(false)))
    }

    pub fn new(net: Arc<Net>, split: bool, split_thr: f32, eject: bool) -> VisionBot {
        VisionBot { net, split, split_thr, eject, frames: FrameStack::new(PX), prepared: false, pending: None, split_override: None, decisions: 0, last: None }
    }

    /// New life: the next frame stack starts with both channels equal.
    pub fn reset(&mut self) {
        self.frames.reset();
        self.prepared = false;
        self.pending = None;
    }

    /// Render this decision's frames from `s` (the agent's own snapshot and view box), shifting the previous one.
    pub fn prepare(&mut self, w: &World, pi: usize, s: &Snapshot) {
        self.frames.prepare(w, pi, s);
        self.prepared = true;
        self.pending = None;
    }

    pub fn is_prepared(&self) -> bool {
        self.prepared && self.pending.is_none()
    }

    pub fn set_output(&mut self, o: NetOut) {
        self.pending = Some(o);
    }

    /// The action for the prepared frames (runs the net itself when no batch pass has answered).
    pub fn decide(&mut self, w: &World, pi: usize, s: &Snapshot) -> Out {
        if !self.prepared {
            self.prepare(w, pi, s);
        }
        let o = match self.pending.take() {
            Some(o) => o,
            None => self.net.infer(1, &self.frames.view, &self.frames.wide).expect("vision inference failed")[0],
        };
        self.prepared = false;
        self.decisions += 1;
        self.last = Some(o);
        let sig = |l: f32| 1.0 / (1.0 + (-l).exp());
        let (x, y) = self.frames.cam.pixel_to_world(o.target.0 as f64, o.target.1 as f64);
        Out { x, y, split: self.split_override.unwrap_or(self.split) && sig(o.split_logit) > self.split_thr, eject: self.eject && sig(o.eject_logit) > 0.5 }
    }
}

/// One batched forward pass for every prepared bot (all bots must share one net).
pub fn infer_batch(bots: &mut [&mut VisionBot]) {
    if bots.is_empty() {
        return;
    }
    let (vs, ws) = (2 * PX * PX, 2 * WPX * WPX);
    let mut view = Vec::with_capacity(bots.len() * vs);
    let mut wide = Vec::with_capacity(bots.len() * ws);
    for b in bots.iter() {
        view.extend_from_slice(&b.frames.view);
        wide.extend_from_slice(&b.frames.wide);
    }
    let outs = bots[0].net.infer(bots.len(), &view, &wide).expect("vision inference failed");
    for (b, o) in bots.iter_mut().zip(outs) {
        b.set_output(o);
    }
}
