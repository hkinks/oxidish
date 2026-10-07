//! Rule-based mode selection (bot/modes.py), virus skills (bot/tactics.py) and the
//! HeuristicPolicy that ties them to the steering (arena/policies.py).

use std::collections::HashMap;

use crate::farm::FoodRouter;
use crate::game::{Cell, Kind, Snapshot};
use crate::params::Params;
use crate::steer::{classify, danger_radius, danger_radius_at, eaters_of, guarded_cells, half_of, split_threats, steer, Mode, Plan, EAT_RATIO};

const VIRUS_POP_RATIO: f64 = 1.33;

pub fn heuristic_plan(s: &Snapshot, p: &Params) -> Plan {
    let me = *s.main().unwrap();
    let cl = classify(s, Some(p));
    let gap = |c: &Cell| me.dist(c) - c.size - me.size;
    if cl.threats.iter().any(|t| gap(t) < danger_radius_at(t, &me, p, &s.border) * p.flee_margin) {
        return Plan::new(Mode::Flee);
    }
    if p.flee_pieces > 0.5 {
        for q in guarded_cells(s, p) {
            if q.id != me.id
                && eaters_of(s, &q).iter().any(|t| q.dist(t) - t.size - q.size < danger_radius(t, &q, p) * p.flee_margin)
            {
                return Plan::new(Mode::Flee);
            }
        }
    }
    let (mut rng, mut floor) = (p.hunt_range, me.mass() * p.min_prey_frac);
    if p.big_hunt > 0.5 && me.mass() > p.big_mass {
        rng *= (me.mass() / p.big_mass).sqrt();
        floor = p.big_mass * p.min_prey_frac;
    }
    // target = max over worth by mass / (max(gap, 0) + 100); first max wins
    let mut target: Option<(&Cell, f64)> = None;
    for c in cl.prey.iter().filter(|c| gap(c) < rng && c.mass() >= floor) {
        let k = c.mass() / (gap(c).max(0.0) + 100.0);
        if target.map_or(true, |(_, bk)| k > bk) {
            target = Some((c, k));
        }
    }
    if let Some((t, _)) = target {
        let side = if p.half_split_safety > 0.5 { half_of(&me) } else { me };
        let safe = !split_threats(s, &me, p).iter().any(|t| gap(t) < danger_radius(t, &side, p) * p.split_safety);
        if p.split_ev > 0.0 {
            // split by expected value: steer() weighs gain against risk
            return Plan { mode: Mode::Hunt, target_id: Some(t.id), split_ok: me.mass() / 2.0 > t.mass() * EAT_RATIO };
        }
        return Plan { mode: Mode::Hunt, target_id: Some(t.id), split_ok: safe && me.mass() / 2.0 > t.mass() * EAT_RATIO };
    }
    Plan::new(Mode::Farm)
}

/// Insertion-ordered counters (JSON key order matches the Python dict).
#[derive(Default, Clone, Debug)]
pub struct Counters(pub Vec<(String, i64)>);

impl Counters {
    pub fn bump(&mut self, key: &str, by: i64) {
        if let Some(e) = self.0.iter_mut().find(|e| e.0 == key) {
            e.1 += by;
        } else {
            self.0.push((key.to_string(), by));
        }
    }
}

fn unit(ax: f64, ay: f64, bx: f64, by: f64) -> (f64, f64) {
    let (dx, dy) = (bx - ax, by - ay);
    let d = dx.hypot(dy);
    let d = if d == 0.0 { 1.0 } else { d };
    (dx / d, dy / d)
}

pub struct Out {
    pub x: f64,
    pub y: f64,
    pub split: bool,
    pub eject: bool,
}

pub struct Tactics {
    feeding: Option<u32>,
    second_split: Option<(f64, f64)>,
}

impl Tactics {
    fn new() -> Tactics {
        Tactics { feeding: None, second_split: None }
    }

    fn reset(&mut self) {
        self.feeding = None;
        self.second_split = None;
    }

    fn act(&mut self, s: &Snapshot, plan: &Plan, p: &Params, counters: &mut Counters) -> Option<Out> {
        let me = *s.main().unwrap();
        let cl = classify(s, None);
        let (threats, viruses) = (&cl.threats, &cl.viruses);
        let gap = |c: &Cell| me.dist(c) - c.size - me.size;
        if let Some(f) = self.feeding {
            if !viruses.iter().any(|v| v.id == f) {
                counters.bump("shots_fired", 1);
                self.feeding = None;
            }
        }
        if let Some((x, y)) = self.second_split.take() {
            return Some(Out { x, y, split: true, eject: false });
        }
        let safe = !threats.iter().any(|t| gap(t) < danger_radius_at(t, &me, p, &s.border));
        let total = s.total_mass();

        if p.selffeed_enable > 0.5 && s.mine.len() >= 2 && safe && total >= p.selffeed_min_mass
            && me.mass() / total < p.selffeed_frac
        {
            counters.bump("selffeeds", 1);
            counters.bump("ejects", 1);
            return Some(Out { x: me.x, y: me.y, split: false, eject: true });
        }

        if p.double_split_enable > 0.5 && plan.mode == Mode::Hunt && s.mine.len() == 1 && safe {
            if let Some(tid) = plan.target_id {
                if let Some(t) = s.others.iter().find(|c| c.id == tid) {
                    if me.mass() / 4.0 > t.mass() * EAT_RATIO {
                        let k = p.lead * (me.dist(t) / 600.0).min(1.0);
                        let (tx, ty) = (t.x + t.vx * k, t.y + t.vy * k);
                        let reach = p.split_range * p.split_reach;
                        let d = (tx - me.x).hypot(ty - me.y) - me.size;
                        if reach < d && d < reach * p.double_split_reach {
                            counters.bump("double_splits", 1);
                            self.second_split = Some((tx, ty));
                            return Some(Out { x: tx, y: ty, split: true, eject: false });
                        }
                    }
                }
            }
        }

        if p.hide_enable > 0.5 && !threats.is_empty() {
            let near: Vec<&Cell> =
                threats.iter().filter(|t| gap(t) < danger_radius_at(t, &me, p, &s.border) * p.flee_margin).collect();
            let covers: Vec<&Cell> =
                viruses.iter().filter(|v| v.mass() > me.mass() * EAT_RATIO && gap(v) < p.hide_range).collect();
            if !near.is_empty() && !covers.is_empty() {
                let mut chaser = near[0];
                for t in &near[1..] {
                    if gap(t) < gap(chaser) {
                        chaser = t;
                    }
                }
                let mut v = covers[0];
                for c in &covers[1..] {
                    if me.dist(c) < me.dist(v) {
                        v = c;
                    }
                }
                let (ux, uy) = unit(chaser.x, chaser.y, v.x, v.y);
                counters.bump("hide_ticks", 1);
                return Some(Out { x: v.x + ux * v.size * 0.6, y: v.y + uy * v.size * 0.6, split: false, eject: false });
            }
        }

        if p.shoot_enable > 0.5 && s.mine.len() == 1 && me.mass() >= p.shoot_min_mass {
            if threats.iter().any(|t| gap(t) < danger_radius(t, &me, p) * 0.7) {
                self.feeding = None;
                return None;
            }
            let mut best: Option<Cell> = None;
            let mut best_score = 0.0f64;
            for v in viruses {
                if gap(v) > p.shoot_range || (me.mass() > v.mass() * VIRUS_POP_RATIO && gap(v) < 30.0) {
                    continue;
                }
                let (ax, ay) = unit(me.x, me.y, v.x, v.y);
                for t in s.others.iter().filter(|c| c.kind == Kind::Player) {
                    if t.mass() < v.mass() * VIRUS_POP_RATIO * p.shoot_target_mult {
                        continue;
                    }
                    let d = v.dist(t);
                    if d > p.shoot_reach {
                        continue;
                    }
                    let (bx, by) = unit(v.x, v.y, t.x, t.y);
                    if ax * bx + ay * by < p.shoot_align {
                        continue;
                    }
                    let score = t.mass() / (d + 100.0) * if Some(v.id) == self.feeding { 2.0 } else { 1.0 };
                    if score > best_score {
                        best = Some(*v);
                        best_score = score;
                    }
                }
            }
            if let Some(b) = best {
                if Some(b.id) != self.feeding {
                    counters.bump("shots_started", 1);
                }
                self.feeding = Some(b.id);
                counters.bump("ejects", 1);
                return Some(Out { x: b.x, y: b.y, split: false, eject: true });
            }
            self.feeding = None;
        }
        None
    }
}

/// The browser bot's steering, with rule-based modes instead of Jev.
pub struct HeuristicPolicy {
    router: Option<FoodRouter>,
    p_small: Params,
    p_big: Params,
    use_big: bool,
    tactics: Tactics,
    pub counters: Counters,
    prev: HashMap<u32, (f64, f64)>,
}

impl HeuristicPolicy {
    /// `spec`: {"farm": "route" | "field", "params": {...}}
    pub fn from_spec(spec: &serde_json::Value) -> HeuristicPolicy {
        let farm = spec.get("farm").and_then(|f| f.as_str()).unwrap_or("route");
        let params = spec.get("params").map(Params::from_json).unwrap_or_default();
        let p_big = if params.caution_mass > 0.0 { params.cautious() } else { params.clone() };
        let mut counters = Counters::default();
        for k in ["ejects", "shots_started", "shots_fired", "hide_ticks", "selffeeds", "double_splits"] {
            counters.0.push((k.to_string(), 0));
        }
        HeuristicPolicy {
            router: if farm == "route" { Some(FoodRouter::new()) } else { None },
            p_small: params,
            p_big,
            use_big: false,
            tactics: Tactics::new(),
            counters,
            prev: HashMap::new(),
        }
    }

    /// Decide one step; `s` must have at least one cell of ours.
    pub fn act(&mut self, s: &mut Snapshot) -> Out {
        let mut prev = std::mem::take(&mut self.prev);
        for c in s.others.iter_mut() {
            if c.kind == Kind::Food {
                continue;
            }
            if let Some(&(px, py)) = prev.get(&c.id) {
                c.vx = c.x - px;
                c.vy = c.y - py;
            }
            self.prev.insert(c.id, (c.x, c.y));
        }
        prev.clear();
        let ramp = self.p_small.caution_mass > 0.0;
        if ramp {
            self.use_big = s.total_mass() > self.p_small.caution_mass;
        }
        let p = if ramp && self.use_big { &self.p_big } else { &self.p_small };
        let plan = heuristic_plan(s, p);
        let out = match self.tactics.act(s, &plan, p, &mut self.counters) {
            Some(o) => o,
            None => {
                let (x, y, split) = steer(s, &plan, self.router.as_mut(), p);
                Out { x, y, split, eject: false }
            }
        };
        self.counters.bump("steps", 1);
        self.counters.bump(&format!("mode_{}", plan.mode.as_str()), 1);
        self.counters.bump("splits", out.split as i64);
        out
    }

    pub fn reset(&mut self) {
        self.prev.clear();
        self.tactics.reset();
        if let Some(r) = self.router.as_mut() {
            r.reset();
        }
    }
}
