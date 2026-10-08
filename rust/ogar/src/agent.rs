//! In-process rule-bot agents: what arena_client does over the network, without the network.
//! The Snapshot is built from the player's server-side view box exactly as the protocol-6 client
//! would reconstruct it (integer positions / sizes, cells in the view box).

use arena_client::client::AgentStats;
use arena_client::game::{Border, Cell, Kind, Snapshot};
use arena_client::policy::Policy;
use serde_json::{json, Value};

use crate::world::{PKind, World, T_EJECT, T_FOOD, T_VIRUS};

/// One decision step of an agent as a recording keeps it (ring buffer, `--record`): what it saw and did.
pub struct StepRec {
    pub tick: u32,
    pub mine: Vec<[i32; 4]>,
    /// non-food nodes in view: id, x, y, size, kind (0 player, 2 virus, 3 ejected), owner player index
    pub others: Vec<[i32; 6]>,
    /// food in view (x, y, size); only kept for the newest few steps
    pub food: Vec<[i32; 3]>,
    pub aim: (f64, f64),
    pub split: bool,
    pub eject: bool,
    pub mode: &'static str,
    pub stage: &'static str,
    pub target: Option<u32>,
    pub split_ok: bool,
}

/// Loss noticed in the last step: (mass before, mass after, died)
pub type Loss = (f64, f64, bool);

const LOSS_DROP: f64 = 0.3;
const LOSS_LOOKBACK: usize = 4;
const LOSS_COOLDOWN: u32 = 20;
const FOOD_KEEP: usize = 3;

pub struct AgentDriver {
    pub pi: usize,
    pub name: String,
    pub policy: Policy,
    /// the genome spec the policy was built from (recorded with loss scenes so they can be replayed)
    pub spec: Value,
    pub stats: AgentStats,
    pub sync: u64,
    alive: bool,
    n_cells: Option<usize>,
    /// ring of the last steps; None = recording off (nothing is collected, no cost)
    pub trace: Option<std::collections::VecDeque<StepRec>>,
    pub trace_len: usize,
    totals: std::collections::VecDeque<f64>,
    cooldown: u32,
    /// set by `step` when the agent lost >= 30% of its mass (or died); the recorder takes it
    pub loss: Option<Loss>,
    /// the commander's standing order (control port `order`), None = none
    pub order: Option<crate::commander::DriverOrder>,
    /// the "vision" agent kind: decides from rendered frames with an ONNX net (see visionbot.rs); `policy` is unused
    #[cfg(feature = "vision-net")]
    pub vision: Option<Box<crate::visionbot::VisionBot>>,
}

fn color_of(c: [u8; 3]) -> u32 {
    ((c[0] as u32) << 16) | ((c[1] as u32) << 8) | c[2] as u32
}

pub fn snapshot(w: &World, pi: usize) -> Snapshot {
    let p = &w.players[pi];
    let mine: Vec<Cell> = p
        .cells
        .iter()
        .map(|&h| {
            let e = &w.ents[h as usize];
            let mut c = Cell::new(e.id, (e.x as i32) as f64, (e.y as i32) as f64, (e.size as u32) as f64, Kind::Mine);
            c.color = color_of(e.color);
            c
        })
        .collect();
    let mut others = Vec::with_capacity(p.view_nodes.len());
    for &h in &p.view_nodes {
        let e = &w.ents[h as usize];
        if e.owner == pi as i32 {
            continue;
        }
        let kind = match e.kind {
            T_FOOD => Kind::Food,
            T_VIRUS => Kind::Virus,
            T_EJECT => Kind::Ejected,
            _ => Kind::Player,
        };
        let mut c = Cell::new(e.id, (e.x as i32) as f64, (e.y as i32) as f64, (e.size as u32) as f64, kind);
        c.color = color_of(e.color);
        others.push(c);
    }
    let b = w.border;
    let dead = mine.is_empty();
    Snapshot { mine, others, border: Border { left: b.0, right: b.2, top: b.1, bottom: b.3 }, dead }
}

impl AgentDriver {
    /// Add a player to the world and wrap it with a rule-bot policy; it spawns on its first step.
    pub fn new(w: &mut World, name: &str, spec: &Value, sync: u64) -> AgentDriver {
        let pi = w.add_player(PKind::Agent);
        w.players[pi].name = name.to_string();
        let is_vision = spec.get("kind").and_then(|k| k.as_str()) == Some("vision");
        #[cfg(feature = "vision-net")]
        let vision = if is_vision {
            Some(Box::new(crate::visionbot::VisionBot::from_spec(spec).unwrap_or_else(|e| panic!("vision agent {name}: {e}"))))
        } else {
            None
        };
        #[cfg(not(feature = "vision-net"))]
        assert!(!is_vision, "this ogar build has no vision-net feature");
        AgentDriver {
            pi,
            name: name.to_string(),
            policy: Policy::from_spec(&if is_vision { json!({}) } else { spec.clone() }),
            spec: spec.clone(),
            stats: AgentStats { name: name.to_string(), ..Default::default() },
            sync,
            alive: false,
            n_cells: None,
            trace: None,
            trace_len: 0,
            totals: Default::default(),
            cooldown: 0,
            loss: None,
            order: None,
            #[cfg(feature = "vision-net")]
            vision,
        }
    }

    /// Start keeping the last `len` steps (see `StepRec`).
    pub fn enable_trace(&mut self, len: usize) {
        self.trace = Some(std::collections::VecDeque::with_capacity(len + 1));
        self.trace_len = len;
    }

    fn step_rec(&self, w: &World) -> StepRec {
        let p = &w.players[self.pi];
        let mine = p.cells.iter().map(|&h| {
            let e = &w.ents[h as usize];
            [e.id as i32, e.x as i32, e.y as i32, e.size as u32 as i32]
        });
        let (mut others, mut food) = (Vec::new(), Vec::new());
        for &h in &p.view_nodes {
            let e = &w.ents[h as usize];
            if e.owner == self.pi as i32 {
                continue;
            }
            let (x, y, sz) = (e.x as i32, e.y as i32, e.size as u32 as i32);
            match e.kind {
                T_FOOD => food.push([x, y, sz]),
                T_VIRUS => others.push([e.id as i32, x, y, sz, 2, -1]),
                T_EJECT => others.push([e.id as i32, x, y, sz, 3, -1]),
                _ => others.push([e.id as i32, x, y, sz, 0, e.owner]),
            }
        }
        StepRec { tick: w.tick, mine: mine.collect(), others, food, aim: (0.0, 0.0), split: false, eject: false,
                  mode: "-", stage: "-", target: None, split_ok: false }
    }

    fn note_loss(&mut self, total: f64, died: bool) {
        self.cooldown = self.cooldown.saturating_sub(1);
        let peak = self.totals.iter().cloned().fold(0.0, f64::max);
        if died {
            self.loss = Some((peak.max(total), 0.0, true));
            self.totals.clear();
            self.cooldown = 0;
            return;
        }
        if self.cooldown == 0 && peak > 0.0 && total < peak * (1.0 - LOSS_DROP) {
            self.loss = Some((peak, total, false));
            self.cooldown = LOSS_COOLDOWN;
        }
        self.totals.push_back(total);
        while self.totals.len() > LOSS_LOOKBACK {
            self.totals.pop_front();
        }
    }

    fn account_events(&mut self, w: &World) {
        let pi = self.pi as i32;
        for k in &w.kill_log {
            if k.killer_owner == pi {
                self.stats.kills += 1;
            } else if k.victim_owner == pi {
                let mine: f64 = w.players[self.pi].cells.iter().map(|&h| w.ents[h as usize].mass()).sum();
                let loss = json!({
                    "by": w.players[k.killer_owner as usize].name, "kind": "player", "mass": k.victim_mass,
                    "killer_mass": k.killer_mass,
                    "killer_cells": k.killer_cells, "killer_total": k.killer_total,
                    "my_cells": w.players[self.pi].cells.len() + 1, "my_total": mine + k.victim_mass,
                    "tick": self.stats.ticks,
                });
                self.stats.losses.push(loss);
            }
        }
    }

    /// One decision step (every `sync` ticks): stats bookkeeping, then act or respawn.
    pub fn step(&mut self, w: &mut World) {
        self.account_events(w);
        let mut s = snapshot(w, self.pi);
        if s.dead {
            if self.stats.steps_alive > 0 && self.alive {
                self.stats.deaths += 1;
            }
            if self.trace.is_some() && self.alive {
                self.note_loss(0.0, true);
            }
            self.alive = false;
            self.policy.reset();
            #[cfg(feature = "vision-net")]
            if let Some(v) = self.vision.as_mut() {
                v.reset();
            }
            let name = self.name.clone();
            w.join_game(self.pi, &name, None);
        } else {
            if !self.alive {
                self.stats.lives += 1;
            }
            self.alive = true;
            let mass = s.total_mass();
            let n = s.mine.len();
            if n >= self.n_cells.unwrap_or(n) + 3 {
                self.stats.popped += 1;
            }
            self.n_cells = Some(n);
            self.stats.steps_alive += 1;
            self.stats.mass_ticks += mass * self.sync as f64;
            self.stats.peak_mass = self.stats.peak_mass.max(mass);
            let mut rec = if self.trace.is_some() { Some(self.step_rec(w)) } else { None };
            let ord = self.current_order(w);
            #[cfg(feature = "vision-net")]
            let a = match self.vision.as_mut() {
                Some(v) => {
                    v.split_override = ord.as_ref().and_then(|o| o.split).filter(|&on| on);
                    let a = v.decide(w, self.pi, &s);
                    match &ord {
                        Some(o) => crate::commander::vision_override(o, &s, a),
                        None => a,
                    }
                }
                None => {
                    self.policy.set_order(ord);
                    self.policy.act(&mut s)
                }
            };
            #[cfg(not(feature = "vision-net"))]
            let a = {
                self.policy.set_order(ord);
                self.policy.act(&mut s)
            };
            if let Some(mut r) = rec.take() {
                r.aim = (a.x, a.y);
                r.split = a.split;
                r.eject = a.eject;
                if let Some((plan, stage)) = self.policy.last_plan() {
                    r.mode = plan.mode.as_str();
                    r.stage = stage;
                    r.target = plan.target_id;
                    r.split_ok = plan.split_ok;
                }
                self.note_loss(mass, false);
                let len = self.trace_len;
                let tr = self.trace.as_mut().unwrap();
                tr.push_back(r);
                if tr.len() > len {
                    tr.pop_front();
                }
                if tr.len() > FOOD_KEEP {
                    let k = tr.len() - 1 - FOOD_KEEP;
                    tr[k].food = Vec::new();
                }
            }
            let p = &mut w.players[self.pi];
            p.mouse_in = Some(((a.x as i32) as f64, (a.y as i32) as f64));
            if a.split {
                p.p_split = true;
            }
            if a.eject {
                p.p_eject = true;
            }
        }
        self.stats.ticks += self.sync;
    }

    pub fn finish(mut self, w: &World) -> AgentStats {
        let p = &w.players[self.pi];
        self.stats.virus_shots = p.shots as u64;
        self.stats.virus_pops = p.pops as u64;
        self.stats.virus_pop_mass = p.pop_mass as f64;
        self.stats.counters = self.policy.counters().clone();
        self.stats
    }
}

/// One decision step for all drivers of a world, in order. The vision agents' forward passes are batched: their
/// frames are rendered first and answered by one net call, then every driver steps as usual (a driver without a
/// vision net behaves exactly as `AgentDriver::step`).
pub fn step_all(drivers: &mut [AgentDriver], w: &mut World) {
    #[cfg(feature = "vision-net")]
    batch_vision(drivers, w);
    for d in drivers.iter_mut() {
        d.step(w);
    }
}

#[cfg(feature = "vision-net")]
fn batch_vision(drivers: &mut [AgentDriver], w: &World) {
    let mut bots: Vec<&mut crate::visionbot::VisionBot> = Vec::new();
    for d in drivers.iter_mut() {
        if let Some(v) = d.vision.as_mut() {
            let s = snapshot(w, d.pi);
            if !s.dead {
                v.prepare(w, d.pi, &s);
                bots.push(v);
            }
        }
    }
    if bots.len() > 1 {
        crate::visionbot::infer_batch(&mut bots);
    }
}
