//! Rule-based mode selection (bot/modes.py), virus skills (bot/tactics.py) and the
//! HeuristicPolicy that ties them to the steering (arena/policies.py).

use std::collections::HashMap;

use crate::farm::FoodRouter;
use crate::game::{Cell, Kind, Snapshot};
use crate::hunter::{Hunter, Phase};
use crate::order::{apply_plan, Order, Verb};
use crate::params::Params;
use crate::steer::{classify, danger_radius, danger_radius_at, eaters_of, guarded_cells, half_of, split_threats, steer_to, Mode, Plan, EAT_RATIO};

const VIRUS_POP_RATIO: f64 = 1.33;
const EJECT_REACH: f64 = 660.0;
const STAGE_ARC: f64 = 0.6;

const HARVEST_R: f64 = 250.0; // harvest: edible cells this close to a target count toward its value

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
    // harvest (Python: modes.heuristic_plan): clouds of pieces, and a 4x lower floor for players in >= 3 cells
    let harvest = p.harvest > 0.5;
    // owner = the protocol colour one player's pieces share (0 = unknown: counts as its own owner)
    let owner = |c: &Cell| -> u64 { if c.color != 0 { c.color as u64 } else { (1u64 << 32) | c.id as u64 } };
    let mut pieces: HashMap<u64, usize> = HashMap::new();
    let mut owner_mass: HashMap<u64, f64> = HashMap::new();
    if harvest {
        for c in s.others.iter().filter(|c| c.kind == Kind::Player && !s.mine.iter().any(|m| m.id == c.id)) {
            *pieces.entry(owner(c)).or_insert(0) += 1;
            *owner_mass.entry(owner(c)).or_insert(0.0) += c.mass();
        }
    }
    // harvest_range: a broken big player (>= 3 pieces, together at least our mass) is hunted from farther away,
    // while its pieces are apart (~30 s until they merge back)
    let my_total = s.total_mass();
    let rng_of = |c: &Cell| -> f64 {
        let o = owner(c);
        if harvest && pieces.get(&o).copied().unwrap_or(0) >= 3 && owner_mass.get(&o).copied().unwrap_or(0.0) >= my_total {
            rng * p.harvest_range
        } else {
            rng
        }
    };
    let floor_of = |c: &Cell| -> f64 {
        if harvest && pieces.get(&owner(c)).copied().unwrap_or(0) >= 3 { floor * 0.25 } else { floor }
    };
    for c in cl.prey.iter().filter(|c| gap(c) < rng_of(c) && c.mass() >= floor_of(c)) {
        let k = if harvest {
            let mut sum = 0.0;
            for o in cl.prey.iter() {
                if o.dist(c) < HARVEST_R {
                    sum += o.mass();
                }
            }
            sum / (gap(c).max(0.0) + 100.0)
        } else {
            c.mass() / (gap(c).max(0.0) + 100.0)
        };
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

#[derive(Debug, Clone, Copy)]
pub struct Out {
    pub x: f64,
    pub y: f64,
    pub split: bool,
    pub eject: bool,
}

pub struct Tactics {
    feeding: Option<u32>,
    second_split: Option<(f64, f64)>,
    stage: Option<(u32, u32)>,
}

impl Tactics {
    fn new() -> Tactics {
        Tactics { feeding: None, second_split: None, stage: None }
    }

    fn reset(&mut self) {
        self.feeding = None;
        self.second_split = None;
        self.stage = None;
    }

    fn act(&mut self, s: &Snapshot, plan: &Plan, p: &Params, counters: &mut Counters, only: Option<&[u32]>) -> Option<Out> {
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

        if p.shoot_enable > 0.5 && (s.mine.len() as f64) <= p.shoot_max_cells && me.mass() >= p.shoot_min_mass {
            // shoot_threats (Python: covered): a threat that is a shot target behind a shielding virus doesn't veto
            let covered = |t: &Cell| -> bool {
                if p.shoot_threats <= 0.5 {
                    return false;
                }
                for v in viruses.iter() {
                    if gap(v) > p.shoot_range || (me.mass() > v.mass() * VIRUS_POP_RATIO && gap(v) < 30.0) {
                        continue;
                    }
                    if me.dist(v) >= me.dist(t) || v.dist(t) > p.shoot_reach {
                        continue;
                    }
                    if t.mass() < v.mass() * VIRUS_POP_RATIO * p.shoot_target_mult {
                        continue;
                    }
                    let (ax, ay) = unit(me.x, me.y, v.x, v.y);
                    let (bx, by) = unit(v.x, v.y, t.x, t.y);
                    if ax * bx + ay * by >= p.shoot_align {
                        return true;
                    }
                }
                false
            };
            if threats.iter().any(|t| gap(t) < danger_radius(t, &me, p) * p.shoot_safety && !covered(t)) {
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
                for t in s.others.iter().filter(|c| c.kind == Kind::Player && only.map_or(true, |o| o.contains(&c.id))) {
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
                if p.shoot_stage > 0.5 {
                    let (hx, hy) = unit(me.x, me.y, b.x, b.y);
                    return Some(Out { x: me.x + hx * me.size * 0.9, y: me.y + hy * me.size * 0.9, split: false, eject: true });
                }
                return Some(Out { x: b.x, y: b.y, split: false, eject: true });
            }
            self.feeding = None;
            if p.shoot_stage > 0.5 {
                return self.stage_move(&me, viruses, s, p, counters, only);
            }
        }
        None
    }

    /// Walk round a virus to the side away from a big player, so that a shot through it lines up.
    fn stage_move(&mut self, me: &Cell, viruses: &[Cell], s: &Snapshot, p: &Params, counters: &mut Counters, only: Option<&[u32]>) -> Option<Out> {
        let mut best: Option<(Cell, Cell)> = None;
        let mut best_score = 0.0f64;
        for v in viruses {
            let walk = me.dist(v);
            if walk > p.shoot_walk || me.size + v.size + p.shoot_standoff > EJECT_REACH {
                continue;
            }
            for t in s.others.iter().filter(|c| c.kind == Kind::Player && only.map_or(true, |o| o.contains(&c.id))) {
                if t.mass() < v.mass() * VIRUS_POP_RATIO * p.shoot_target_mult {
                    continue;
                }
                let d = v.dist(t);
                if d > p.shoot_reach || d < t.size {
                    continue;
                }
                let score = t.mass() / (walk + d + 400.0) * if self.stage == Some((v.id, t.id)) { 2.0 } else { 1.0 };
                if score > best_score {
                    best = Some((*v, *t));
                    best_score = score;
                }
            }
        }
        let (v, t) = match best {
            Some(b) => b,
            None => {
                self.stage = None;
                return None;
            }
        };
        self.stage = Some((v.id, t.id));
        let stand = me.size + v.size + p.shoot_standoff;
        let here = (me.y - v.y).atan2(me.x - v.x);
        let want = (v.y - t.y).atan2(v.x - t.x);
        let mut turn = want - here;
        while turn > std::f64::consts::PI {
            turn -= std::f64::consts::PI * 2.0;
        }
        while turn < -std::f64::consts::PI {
            turn += std::f64::consts::PI * 2.0;
        }
        turn = turn.max(-STAGE_ARC).min(STAGE_ARC);
        counters.bump("stage_ticks", 1);
        Some(Out { x: v.x + (here + turn).cos() * stand, y: v.y + (here + turn).sin() * stand, split: false, eject: false })
    }
}

/// The browser bot's steering, with rule-based modes instead of Jev.
pub struct HeuristicPolicy {
    router: Option<FoodRouter>,
    p_small: Params,
    p_big: Params,
    p_giant: Option<Params>,
    use_big: bool,
    tactics: Tactics,
    pub counters: Counters,
    prev: HashMap<u32, (f64, f64)>,
    flee_dir: Option<(f64, f64)>,
    /// plan and genome stage ("small" / "big" / "giant") of the last `act` (read-only, for recordings)
    pub last_plan: Option<(Plan, &'static str)>,
    /// the commander's standing order (set by the server each step; None = none)
    pub order: Option<Order>,
    /// lurk-and-harvest hunter (genes lurk / commit / split_harvest); `hunter.phase != Idle` when it decided the last step
    pub hunter: Hunter,
}

impl HeuristicPolicy {
    /// `spec`: {"farm": "route" | "field", "params": {...}}
    pub fn from_spec(spec: &serde_json::Value) -> HeuristicPolicy {
        let farm = spec.get("farm").and_then(|f| f.as_str()).unwrap_or("route");
        let params = spec.get("params").map(Params::from_json).unwrap_or_default();
        let p_big = if params.caution_mass > 0.0 { params.cautious() } else { params.clone() };
        let p_giant = if params.giant_mass > 0.0 { Some(params.giant()) } else { None };
        let mut counters = Counters::default();
        for k in ["ejects", "shots_started", "shots_fired", "hide_ticks", "selffeeds", "double_splits"] {
            counters.0.push((k.to_string(), 0));
        }
        HeuristicPolicy {
            router: if farm == "route" { Some(FoodRouter::new()) } else { None },
            p_small: params,
            p_giant,
            p_big,
            use_big: false,
            tactics: Tactics::new(),
            counters,
            prev: HashMap::new(),
            flee_dir: None,
            last_plan: None,
            order: None,
            hunter: Hunter::new(),
        }
    }

    /// The genes the next decision uses at this total mass (ramp / giant stages), without changing any state.
    pub fn params_for(&self, total: f64) -> &Params {
        let mut p = if self.p_small.caution_mass > 0.0 && total > self.p_small.caution_mass { &self.p_big } else { &self.p_small };
        if let Some(g) = &self.p_giant {
            if total > self.p_small.giant_mass {
                p = g;
            }
        }
        p
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
        let mut p = if ramp && self.use_big { &self.p_big } else { &self.p_small };
        let mut stage = if ramp && self.use_big { "big" } else { "small" };
        if let Some(g) = &self.p_giant {
            if s.total_mass() > self.p_small.giant_mass {
                p = g; // giant stage: its own genes above giant_mass
                stage = "giant";
            }
        }
        if self.order.is_none() && Hunter::enabled(p) {
            if let Some(mut o) = self.hunter.step(s, p) {
                self.last_plan = Some((Plan::new(Mode::Hunt), stage));
                if self.hunter.phase == Phase::Lurk && p.lurk_shoot > 0.5 {
                    // feed the virus into the giant: the shoot tactic, on the giant only
                    if let Some(g) = self.hunter.lurk_target {
                        let forced = Params { shoot_enable: 1.0, shoot_target_mult: 1.0, shoot_min_mass: 40.0, shoot_max_cells: 16.0,
                                              shoot_stage: 0.0, shoot_threats: 1.0, shoot_range: p.shoot_range.max(900.0), shoot_reach: p.shoot_reach.max(1300.0), ..p.clone() };
                        if let Some(sh) = self.tactics.act(s, &Plan::new(Mode::Hunt), &forced, &mut self.counters, Some(&[g])) {
                            if sh.eject {
                                o = sh;
                            }
                        }
                    }
                }
                self.flee_dir = None;
                self.counters.bump(if self.hunter.phase == Phase::Lurk { "hunter_lurk" } else { "hunter_commit" }, 1);
                self.counters.bump("steps", 1);
                self.counters.bump("splits", o.split as i64);
                return o;
            }
        }
        let mut plan = heuristic_plan(s, p);
        let mut goto = None;
        let mut forced: Option<Params> = None;
        let mut only: Option<Vec<u32>> = None;
        if let Some(o) = &self.order {
            let a = apply_plan(plan, s, p, o);
            plan = a.plan;
            goto = a.goto;
            self.counters.bump(if a.honoured { "order_steps" } else { "order_ignored" }, 1);
            if o.verb == Verb::VirusSplit && plan.mode != Mode::Flee {
                // shoot viruses at the ordered target only: every shooting gene forced on
                forced = Some(Params { shoot_enable: 1.0, shoot_target_mult: 1.0, shoot_min_mass: 40.0, shoot_max_cells: 16.0,
                                       shoot_stage: 1.0, shoot_range: p.shoot_range.max(700.0), ..p.clone() });
                only = Some(o.ids.clone());
            }
        }
        let p = forced.as_ref().unwrap_or(p);
        self.last_plan = Some((plan, stage));
        let acted = self.tactics.act(s, &plan, p, &mut self.counters, only.as_deref());
        let split_off = self.order.as_ref().map_or(false, |o| o.split == Some(false));
        let out = match acted {
            Some(o) => {
                if plan.mode != Mode::Flee {
                    self.flee_dir = None;
                }
                o
            }
            None => {
                let (x, y, split) = steer_to(s, &plan, self.router.as_mut(), p, goto);
                let mut o = Out { x, y, split, eject: false };
                // flee_smooth (Python: HeuristicPolicy._smooth_flee): commit to one escape heading
                if plan.mode != Mode::Flee || p.flee_smooth <= 0.0 || o.split {
                    if plan.mode != Mode::Flee {
                        self.flee_dir = None;
                    }
                } else {
                    let me = *s.main().unwrap();
                    let (dx, dy) = (o.x - me.x, o.y - me.y);
                    let d = {
                        let h = dx.hypot(dy);
                        if h == 0.0 { 1.0 } else { h }
                    };
                    let (mut ux, mut uy) = (dx / d, dy / d);
                    if let Some((px, py)) = self.flee_dir {
                        let dot = ux * px + uy * py;
                        if (-0.2..0.5).contains(&dot) {
                            let (bx, by) = (p.flee_smooth * px + (1.0 - p.flee_smooth) * ux,
                                            p.flee_smooth * py + (1.0 - p.flee_smooth) * uy);
                            let n = bx.hypot(by);
                            if n > 1e-9 {
                                ux = bx / n;
                                uy = by / n;
                                o.x = me.x + ux * d;
                                o.y = me.y + uy * d;
                            }
                        }
                    }
                    self.flee_dir = Some((ux, uy));
                }
                o
            }
        };
        let mut out = out;
        if split_off {
            out.split = false;
        }
        self.counters.bump("steps", 1);
        self.counters.bump(&format!("mode_{}", plan.mode.as_str()), 1);
        self.counters.bump("splits", out.split as i64);
        out
    }

    pub fn reset(&mut self) {
        self.prev.clear();
        self.flee_dir = None;
        self.tactics.reset();
        self.hunter.reset();
        if let Some(r) = self.router.as_mut() {
            r.reset();
        }
    }
}

/// Any policy kind a spec can select: the rule bot (default) or the AposBot port ({"kind": "apos"}).
pub enum Policy {
    Heuristic(HeuristicPolicy),
    Apos(crate::apos::AposPolicy),
    Plan(crate::plan::PlanPolicy),
}

impl Policy {
    pub fn from_spec(spec: &serde_json::Value) -> Policy {
        match spec.get("kind").and_then(|k| k.as_str()) {
            Some("apos") => Policy::Apos(crate::apos::AposPolicy::from_spec(spec)),
            Some("plan") => Policy::Plan(crate::plan::PlanPolicy::from_spec(spec)),
            _ => Policy::Heuristic(HeuristicPolicy::from_spec(spec)),
        }
    }

    pub fn act(&mut self, s: &mut Snapshot) -> Out {
        match self {
            Policy::Heuristic(p) => p.act(s),
            Policy::Apos(p) => p.act(s),
            Policy::Plan(p) => p.act(s),
        }
    }

    /// Set or clear the commander's order (rule bots with modes only; AposBot ignores it).
    pub fn set_order(&mut self, o: Option<Order>) {
        match self {
            Policy::Heuristic(p) => p.order = o,
            Policy::Plan(p) => p.base.order = o,
            Policy::Apos(_) => {}
        }
    }

    pub fn reset(&mut self) {
        match self {
            Policy::Heuristic(p) => p.reset(),
            Policy::Apos(p) => p.reset(),
            Policy::Plan(p) => p.reset(),
        }
    }

    /// The last decision's plan (mode, target) and stage; None for policies without modes.
    pub fn last_plan(&self) -> Option<(Plan, &'static str)> {
        match self {
            Policy::Heuristic(p) => p.last_plan,
            Policy::Apos(_) => None,
            Policy::Plan(p) => p.base.last_plan, // the planner's base rule bot decides first (its plan and stage)
        }
    }

    pub fn counters(&self) -> &Counters {
        match self {
            Policy::Heuristic(p) => &p.counters,
            Policy::Apos(p) => &p.counters,
            Policy::Plan(p) => &p.base.counters,
        }
    }
}

#[cfg(test)]
mod harvest_tests {
    use super::*;
    use crate::game::{Border, Cell, Kind, Snapshot};

    fn snap(others: Vec<Cell>) -> Snapshot {
        let me = Cell::new(1, 0.0, 0.0, (3000.0f64 * 100.0).sqrt(), Kind::Mine);
        Snapshot { mine: vec![me], others, border: Border { left: -3500.0, right: 3500.0, top: -3500.0, bottom: 3500.0 }, dead: false }
    }

    fn piece(id: u32, x: f64, y: f64, mass: f64, color: u32) -> Cell {
        let mut c = Cell::new(id, x, y, (mass * 100.0).sqrt(), Kind::Player);
        c.color = color;
        c
    }

    /// A lone 700 prey vs five 300 pieces of one (popped) player, both ~900 away: harvest goes for the cloud.
    #[test]
    fn harvest_prefers_a_cloud_of_pieces() {
        let mut others = vec![piece(10, 900.0, 0.0, 700.0, 0xff0000)];
        for i in 0..5 {
            others.push(piece(20 + i, -900.0 - 40.0 * i as f64, 30.0 * i as f64, 300.0, 0x00ff00));
        }
        let s = snap(others);
        let base = Params::default();
        let on = Params { harvest: 1.0, ..Params::default() };
        assert_eq!(heuristic_plan(&s, &base).target_id, Some(10));
        let t = heuristic_plan(&s, &on).target_id.unwrap();
        assert!((20..25).contains(&t), "harvest target {t} is not in the cloud of pieces");
    }
}

#[cfg(test)]
mod scavenge_tests {
    use super::*;
    use crate::game::{Border, Cell, Kind, Snapshot};

    /// A 9000 giant just popped into 12 pieces of 750, ~2200 away (beyond hunt_range 900) from our 2500 cell: with harvest_range the bot
    /// rushes in (the user: challenge the big ones by eating them efficiently when they break).
    #[test]
    fn rush_a_broken_giant() {
        let me = Cell::new(1, 0.0, 0.0, (2500.0f64 * 100.0).sqrt(), Kind::Mine);
        let mut others = Vec::new();
        for i in 0..12 {
            let a = i as f64 * 0.5;
            let mut c = Cell::new(30 + i, 2200.0 + 150.0 * a.cos(), 150.0 * a.sin(), (750.0f64 * 100.0).sqrt(), Kind::Player);
            c.color = 0x0000ff;
            others.push(c);
        }
        let s = Snapshot { mine: vec![me], others, border: Border { left: -3500.0, right: 3500.0, top: -3500.0, bottom: 3500.0 }, dead: false };
        let harvest = Params { harvest: 1.0, ..Params::default() };
        let scavenge = Params { harvest: 1.0, harvest_range: 2.5, ..Params::default() };
        assert_eq!(heuristic_plan(&s, &harvest).mode, Mode::Farm, "pieces beyond hunt_range should be ignored without it");
        assert_eq!(heuristic_plan(&s, &scavenge).mode, Mode::Hunt, "no rush onto the broken giant's pieces");
    }
}

#[cfg(test)]
mod hunter_policy_tests {
    use super::*;
    use crate::game::{Border, Cell, Kind, Snapshot};

    fn snap(giant_x: f64) -> Snapshot {
        let me = Cell::new(1, 0.0, 0.0, (500.0f64 * 100.0).sqrt(), Kind::Mine);
        let virus = Cell::new(2, 700.0, 0.0, (120.0f64 * 100.0).sqrt(), Kind::Virus);
        let mut giant = Cell::new(3, giant_x, 0.0, (5000.0f64 * 100.0).sqrt(), Kind::Player);
        giant.color = 0x00ff00;
        Snapshot { mine: vec![me], others: vec![virus, giant], border: Border { left: -4000.0, right: 4000.0, top: -4000.0, bottom: 4000.0 }, dead: false }
    }

    fn policy(genes: serde_json::Value) -> HeuristicPolicy {
        HeuristicPolicy::from_spec(&serde_json::json!({"farm": "field", "params": genes}))
    }

    #[test]
    fn lurk_shoots_the_virus_into_the_giant_only_with_lurk_shoot() {
        // we stand in line: us - virus - giant, giant outside its split reach
        let mut on = policy(serde_json::json!({"lurk": 1.0, "lurk_shoot": 1.0}));
        let mut s = snap(1700.0);
        let out = on.act(&mut s);
        assert_eq!(on.hunter.phase, Phase::Lurk);
        assert!(out.eject, "no shot at the virus: {out:?}");
        let mut off = policy(serde_json::json!({"lurk": 1.0}));
        let mut s = snap(1700.0);
        let out = off.act(&mut s);
        assert_eq!(off.hunter.phase, Phase::Lurk);
        assert!(!out.eject && !out.split);
        // genes off: the ordinary bot, no hunter
        let mut base = policy(serde_json::json!({}));
        let mut s = snap(1700.0);
        base.act(&mut s);
        assert_eq!(base.hunter.phase, Phase::Idle);
    }
}
