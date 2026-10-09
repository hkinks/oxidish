//! The "vision" agent kind (docs/vision-bot.md, M3): decides from rendered grayscale frames with an ONNX net.
//! Spec: `{"kind":"vision","model":"runs/vision/export/vision_bc3.onnx","split":false}` (+ optional `split_thr`,
//! `ep` = cpu|cuda, `threads`, `eject`). Frames are made exactly as the VecEnv makes them (vision.rs `observe`):
//! `view` = the camera square at 256 px, `wide` = 2x that extent at 128 px, channel 0 = this decision,
//! channel 1 = the previous decision (a new life starts with both equal). The net's target pixel is mapped to
//! the world with the VecEnv's camera (`Camera::pixel_to_world`).

use std::sync::Arc;

use arena_client::game::Snapshot;
use arena_client::policy::Out;
use arena_client::reflex::Reflex;
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
    pending: Option<(NetOut, Vec<f32>)>,
    /// the GRU hidden state of this agent (empty = not recurrent); zeros at the start of a life
    pub hidden: Vec<f32>,
    /// a commander's `split on`: allow the net's split decisions although the spec has `split: false`
    pub split_override: Option<bool>,
    pub decisions: u64,
    /// the net output of the latest decision (frames of that decision are still in `frames`)
    pub last: Option<NetOut>,
    /// the hybrid reflex layer (split-kill / harvest / virus-split; spec keys `reflex_*`), off by default
    pub reflex: Option<Reflex>,
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
        let mut vb = VisionBot::new(net, spec["split"].as_bool().unwrap_or(false), split_thr, spec["eject"].as_bool().unwrap_or(false));
        vb.reflex = Reflex::from_spec(spec);
        Ok(vb)
    }

    pub fn new(net: Arc<Net>, split: bool, split_thr: f32, eject: bool) -> VisionBot {
        let rec = net.rec;
        VisionBot { net, split, split_thr, eject, frames: FrameStack::new(PX), prepared: false, pending: None, hidden: vec![0.0; rec], split_override: None, decisions: 0, last: None, reflex: None }
    }

    /// New life: the next frame stack starts with both channels equal.
    pub fn reset(&mut self) {
        self.frames.reset();
        self.hidden.iter_mut().for_each(|x| *x = 0.0);
        if let Some(r) = self.reflex.as_mut() {
            r.reset();
        }
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

    pub fn set_output(&mut self, o: NetOut, h: Vec<f32>) {
        self.pending = Some((o, h));
    }

    /// The action for the prepared frames (runs the net itself when no batch pass has answered).
    pub fn decide(&mut self, w: &World, pi: usize, s: &Snapshot) -> Out {
        if !self.prepared {
            self.prepare(w, pi, s);
        }
        let (o, h) = match self.pending.take() {
            Some(x) => x,
            None => {
                let (mut outs, h) = self.net.infer(1, &self.frames.view, &self.frames.wide, &self.hidden).expect("vision inference failed");
                (outs.remove(0), h)
            }
        };
        if !h.is_empty() {
            self.hidden = h;
        }
        self.prepared = false;
        self.decisions += 1;
        self.last = Some(o);
        let sig = |l: f32| 1.0 / (1.0 + (-l).exp());
        let (x, y) = self.frames.cam.pixel_to_world(o.target.0 as f64, o.target.1 as f64);
        let out = Out { x, y, split: self.split_override.unwrap_or(self.split) && sig(o.split_logit) > self.split_thr, eject: self.eject && sig(o.eject_logit) > 0.5 };
        match self.reflex.as_mut() {
            Some(r) => r.apply(s, out),
            None => out,
        }
    }
}

/// One batched forward pass for every prepared bot, whatever world it lives in (`ogar endless` passes the bots of all
/// worlds). Bots are grouped by net (one pass per model). Returns the number of frames answered.
pub fn infer_batch(bots: &mut [&mut VisionBot]) -> usize {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for i in 0..bots.len() {
        match groups.iter_mut().find(|g| Arc::ptr_eq(&bots[g[0]].net, &bots[i].net)) {
            Some(g) => g.push(i),
            None => groups.push(vec![i]),
        }
    }
    let (vs, ws) = (2 * PX * PX, 2 * WPX * WPX);
    for g in &groups {
        let net = bots[g[0]].net.clone();
        let pb = net.padded_batch(g.len()); // graph backends run power-of-two sizes: pad here, once
        let mut view = Vec::with_capacity(pb * vs);
        let mut wide = Vec::with_capacity(pb * ws);
        for &i in g {
            view.extend_from_slice(&bots[i].frames.view);
            wide.extend_from_slice(&bots[i].frames.wide);
        }
        view.resize(pb * vs, 0);
        wide.resize(pb * ws, 0);
        let rec = net.rec;
        let mut hs = Vec::with_capacity(pb * rec);
        for &i in g {
            hs.extend_from_slice(&bots[i].hidden);
        }
        hs.resize(pb * rec, 0.0);
        let (outs, hout) = net.infer(pb, &view, &wide, &hs).expect("vision inference failed");
        for (k, (&i, o)) in g.iter().zip(outs).enumerate() {
            bots[i].set_output(o, if rec > 0 { hout[k * rec..(k + 1) * rec].to_vec() } else { Vec::new() });
        }
    }
    bots.len()
}
