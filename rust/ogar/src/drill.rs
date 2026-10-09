//! `ogar drill`: the giant-pop harvest drill (docs/harvest-drill.md).
//!
//! A repeatable measurement of how well an agent eats a giant that has just burst on a virus. Each rep builds a small
//! world straight from a seed (no server bots, no stray viruses): our agent (default 500 mass), a giant (default 5k
//! mass) driven by a scripted walker, optionally a third-party rule bot nearby. The giant walks onto a virus (scenario
//! `giant_pop`, our agent is live meanwhile) or sits on it (`giant_popped`, the pop happens on the first tick); the pop is
//! the real MultiOgar `virus_eaten` code, so the piece distribution is the game's. The clock starts at the pop tick.
//!
//! Distance: `dist` (D) is the distance from our cell to the virus. The giant bursts on it from the far side, its
//! centre `giant size - 34` beyond the virus, so the centre-to-centre distance at the pop is D + giant size - 34 and the
//! empty space between the two edges is D - 34 - our size.
//!
//! Giant after the pop (`hunt`): all its pieces follow one cursor, the pieces' centroid (they regroup), but when a piece
//! that can eat one of ours is within HUNT_REACH of it the cursor goes to that cell of ours instead. `passive`: the
//! cursor stays on the centroid. The walker never splits and never ejects.
//!
//! Agents: rule bots / planners / vision nets (`name=spec.json`, as everywhere) and four scripted harvesters,
//! `oracle` (sees every piece on the map, picks mass / distance like the rule bots), `oracle_near` (the nearest piece instead), `oracle_view` (only what its view box shows) and `oracle_brave` (oracle without any fear).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use arena_client::game::{Kind, Snapshot};
use arena_client::policy::Out;
use serde_json::{json, Value};

use crate::agent::{snapshot, AgentDriver};
use crate::config::Config;
use crate::rng::Rng;
use crate::world::{PKind, World};

/// size at which a cell can eat another: 1.15 x its size (config player_eat_mult)
const EAT: f64 = 1.15;
/// the giant's pieces chase one of our cells when a capable piece is this close (centre distance)
const HUNT_REACH: f64 = 1100.0;
/// ticks per second of game time
const TPS: f64 = 25.0;
/// decision interval (ticks), as everywhere else
const SYNC: u64 = 2;
/// give up when the giant has not popped after this many ticks
const MAX_PRE_TICKS: u32 = 2500;
/// the giant starts this far (centre) from the pop point on its walk
const WALK: f64 = 500.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scenario {
    GiantPop,
    GiantPopped,
}

impl Scenario {
    pub fn name(self) -> &'static str {
        match self {
            Scenario::GiantPop => "giant_pop",
            Scenario::GiantPopped => "giant_popped",
        }
    }
    pub fn parse(s: &str) -> Result<Scenario, String> {
        match s {
            "giant_pop" => Ok(Scenario::GiantPop),
            "giant_popped" => Ok(Scenario::GiantPopped),
            _ => Err(format!("unknown scenario {s:?} (giant_pop | giant_popped)")),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GiantMode {
    Hunt,
    Passive,
    /// after the pop a rule bot (`Cond::giant_bot`) drives the giant's pieces
    Bot,
}

impl GiantMode {
    pub fn name(self) -> &'static str {
        match self {
            GiantMode::Hunt => "hunt",
            GiantMode::Passive => "passive",
            GiantMode::Bot => "bot",
        }
    }
    pub fn parse(s: &str) -> Result<GiantMode, String> {
        match s {
            "hunt" => Ok(GiantMode::Hunt),
            "passive" => Ok(GiantMode::Passive),
            "bot" => Ok(GiantMode::Bot),
            _ => Err(format!("unknown giant mode {s:?} (hunt | passive)")),
        }
    }
}

/// One drill condition (everything but the agent and the seed).
#[derive(Clone, Debug)]
pub struct Cond {
    pub scenario: Scenario,
    pub our_mass: f64,
    pub giant_mass: f64,
    /// edge-to-edge distance between us and the giant at the pop
    pub dist: f64,
    pub giant_mode: GiantMode,
    /// the rule bot spec of `GiantMode::Bot`
    pub giant_bot: Option<Value>,
    /// a third-party rule bot (spec) standing as far from the pop as we do, on the side
    pub competitor: Option<Value>,
    pub comp_mass: f64,
    pub horizon_s: f64,
    pub map: f64,
    pub food_mult: f64,
}

impl Cond {
    pub fn new(scenario: Scenario, our_mass: f64, giant_mass: f64, dist: f64) -> Cond {
        Cond { scenario, our_mass, giant_mass, dist, giant_mode: GiantMode::Hunt, giant_bot: None, competitor: None, comp_mass: 800.0, horizon_s: 30.0, map: 7000.0, food_mult: 2.0 }
    }

    pub fn key(&self) -> String {
        format!("{}/our{}/giant{}/d{}/{}{}", self.scenario.name(), self.our_mass, self.giant_mass, self.dist, self.giant_mode.name(),
                if self.competitor.is_some() { format!("/comp{}", self.comp_mass) } else { String::new() })
    }
}

#[derive(Clone, Debug)]
pub enum AgentKind {
    /// scripted harvester; `full` = sees every piece on the map; `brave` = ignores every danger (no fleeing, no safety test)
    Oracle { full: bool, brave: bool, nearest: bool },
    /// any `Policy` / vision spec
    Policy(Value),
}

#[derive(Clone, Debug)]
pub struct AgentDef {
    pub name: String,
    pub kind: AgentKind,
}

impl AgentDef {
    /// "name=oracle" | "name=oracle_view" | "name=path/to/spec.json"
    pub fn parse(arg: &str) -> Result<AgentDef, String> {
        let (name, val) = arg.split_once('=').ok_or(format!("{arg}: want name=spec.json | name=oracle | name=oracle_view"))?;
        let kind = match val {
            "oracle" => AgentKind::Oracle { full: true, brave: false, nearest: false },
            "oracle_near" => AgentKind::Oracle { full: true, brave: false, nearest: true },
            "oracle_view" => AgentKind::Oracle { full: false, brave: false, nearest: false },
            "oracle_brave" => AgentKind::Oracle { full: true, brave: true, nearest: false },
            path => {
                let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
                AgentKind::Policy(serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?)
            }
        };
        Ok(AgentDef { name: name.to_string(), kind })
    }
}

// ---------------------------------------------------------------------------------------------- scene

pub struct Scene {
    pub us: usize,
    pub giant: usize,
    pub comp: Option<usize>,
    pub virus_pos: (f64, f64),
    pub pop_point: (f64, f64),
}

fn size_of(mass: f64) -> f64 {
    (100.0 * mass).sqrt()
}

fn spawn_at(w: &mut World, pi: usize, mass: f64, pos: (f64, f64)) {
    w.players[pi].spawn_size = size_of(mass);
    let c = w.random_color();
    w.players[pi].color = c;
    w.spawn_player(pi, pos);
    let h = w.players[pi].cells[0];
    w.place_cell(h, pos.0, pos.1);
}

fn clamp_in(w: &World, p: (f64, f64), margin: f64) -> (f64, f64) {
    let b = w.border;
    (p.0.clamp(b.0 + margin, b.2 - margin), p.1.clamp(b.1 + margin, b.3 - margin))
}

/// Build the world of one rep. `mk(world, role)` creates the player of role "us" and "comp" (so the caller decides what
/// drives them); the giant is created here. Everything is a function of (cond, seed).
pub fn build_scene(cond: &Cond, seed: u64, mk: &mut dyn FnMut(&mut World, &str) -> usize) -> (World, Scene) {
    let mut cfg = Config::new(cond.map, 0, cond.food_mult);
    cfg.virus_min_amount = 0; // no stray viruses: the one under the giant is the only one
    let mut w = World::new(cfg, seed);
    let mut lr = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0x1234_5677) | 1);
    let (gs, os) = (size_of(cond.giant_mass), size_of(cond.our_mass));
    let _ = os;
    // the virus, near the middle of the map; e = the direction from us to it; the giant bursts on it from the far side
    // (its centre `gs - 34` beyond the virus: just inside the eating range), walking towards us
    let vpos = clamp_in(&w, ((lr.f64() - 0.5) * 1200.0, (lr.f64() - 0.5) * 1200.0), 0.0);
    let ang = lr.f64() * std::f64::consts::TAU;
    let e = (ang.cos(), ang.sin());
    let side = if lr.f64() < 0.5 { 1.0 } else { -1.0 };
    let pop = (vpos.0 + e.0 * (gs - 34.0), vpos.1 + e.1 * (gs - 34.0));
    let us_pos = clamp_in(&w, (vpos.0 - e.0 * cond.dist, vpos.1 - e.1 * cond.dist), 300.0);

    let us = mk(&mut w, "us");
    let giant = if cond.giant_mode == GiantMode::Bot {
        mk(&mut w, "giant")
    } else {
        w.add_player(PKind::Agent)
    };
    w.players[giant].name = "giant".into();
    let comp = cond.competitor.as_ref().map(|_| mk(&mut w, "comp"));

    spawn_at(&mut w, us, cond.our_mass, us_pos);
    let gpos = match cond.scenario {
        Scenario::GiantPop => (pop.0 + e.0 * WALK, pop.1 + e.1 * WALK),
        Scenario::GiantPopped => pop,
    };
    spawn_at(&mut w, giant, cond.giant_mass, gpos);
    if let Some(ci) = comp {
        // as far from the virus as we are, 55 - 120 degrees round from our side
        let a = ang + std::f64::consts::PI + side * (0.95 + 1.15 * lr.f64());
        let cp = clamp_in(&w, (vpos.0 + a.cos() * cond.dist, vpos.1 + a.sin() * cond.dist), 300.0);
        spawn_at(&mut w, ci, cond.comp_mass, cp);
    }
    w.add_virus_at(vpos.0, vpos.1);
    (w, Scene { us, giant, comp, virus_pos: vpos, pop_point: pop })
}

fn cells_of(w: &World, pi: usize) -> Vec<(f64, f64, f64)> {
    w.players[pi].cells.iter().map(|&h| {
        let e = &w.ents[h as usize];
        (e.x, e.y, e.size)
    }).collect()
}

fn total_mass(w: &World, pi: usize) -> f64 {
    w.players[pi].total_mass(w)
}

/// The giant walker's cursor for this tick.
fn giant_aim(w: &World, sc: &Scene, mode: GiantMode, popped: bool) -> (f64, f64) {
    let g = cells_of(w, sc.giant);
    if g.is_empty() {
        return sc.pop_point;
    }
    if !popped {
        return sc.virus_pos;
    }
    let tot: f64 = g.iter().map(|c| c.2 * c.2).sum();
    let (cx, cy) = (g.iter().map(|c| c.0 * c.2 * c.2).sum::<f64>() / tot, g.iter().map(|c| c.1 * c.2 * c.2).sum::<f64>() / tot);
    if mode == GiantMode::Hunt {
        // the pair (piece that can eat ours, our cell) closest to each other, if close enough
        let mut best: Option<(f64, (f64, f64))> = None;
        for o in cells_of(w, sc.us) {
            for p in &g {
                if p.2 >= EAT * o.2 {
                    let d = (p.0 - o.0).hypot(p.1 - o.1);
                    if d < HUNT_REACH && best.map_or(true, |b| d < b.0) {
                        best = Some((d, (o.0, o.1)));
                    }
                }
            }
        }
        if let Some((_, t)) = best {
            return t;
        }
    }
    (cx, cy)
}

/// What a perfect eater could take of the giant in the shape the pop left it, if the pieces just sat there: eat the
/// pieces smallest first while our mass is at least 1.15^2 times the next one. Returns (share of the giant's mass, whether
/// that leaves us heavier than what remains of it). A bound on mass arithmetic alone: no chasing, no danger, no food.
fn chain_bound(w: &World, us: usize, giant: usize) -> (f64, bool) {
    let mut ps: Vec<f64> = w.players[giant].cells.iter().map(|&h| w.ents[h as usize].mass()).collect();
    ps.sort_by(|a, b| a.total_cmp(b));
    let total: f64 = ps.iter().sum();
    let (mut m, mut got) = (total_mass(w, us), 0.0);
    for p in ps {
        if m < EAT * EAT * p {
            break;
        }
        m += p;
        got += p;
    }
    (got / total, m > total - got)
}

// ---------------------------------------------------------------------------------------------- oracle

/// The oracle flees when the edge gap to a piece that can eat us is below this (+LUNGE if the piece's half could eat us
/// too: it can split onto us); a target is safe when no such piece is within its size + SAFE_MARGIN (centre distance).
const ORACLE_FLEE_GAP: f64 = 200.0;
const ORACLE_LUNGE: f64 = 300.0;
const ORACLE_SAFE_MARGIN: f64 = 100.0;

/// The scripted harvester: the best safe edible piece first (mass / distance, or simply the nearest), split onto it when that is safe, keep away from pieces that
/// can eat us, eat food otherwise.
pub struct Oracle {
    full: bool,
    brave: bool,
    nearest: bool,
    split_cd: u32,
}

#[derive(Clone, Copy)]
struct Piece {
    x: f64,
    y: f64,
    size: f64,
}

fn seg_dist(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let l2 = dx * dx + dy * dy;
    let t = if l2 < 1e-9 { 0.0 } else { (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / l2).clamp(0.0, 1.0) };
    (p.0 - (a.0 + t * dx)).hypot(p.1 - (a.1 + t * dy))
}

impl Oracle {
    pub fn new(full: bool, brave: bool, nearest: bool) -> Oracle {
        Oracle { full, brave, nearest, split_cd: 0 }
    }

    pub fn decide(&mut self, w: &World, pi: usize, s: &Snapshot) -> Out {
        self.split_cd = self.split_cd.saturating_sub(1);
        let me = *s.main().unwrap();
        let total = s.total_mass();
        // the cells that matter for safety: those with a quarter of our mass at least
        let ref_size = s.mine.iter().filter(|c| c.mass() >= 0.25 * total).map(|c| c.size).fold(f64::MAX, f64::min);
        let rivals: Vec<Piece> = if self.full {
            w.nodes_player.iter().map(|&h| &w.ents[h as usize]).filter(|e| !e.removed && e.owner >= 0 && e.owner != pi as i32)
                .map(|e| Piece { x: e.x, y: e.y, size: e.size }).collect()
        } else {
            s.others.iter().filter(|c| c.kind == Kind::Player).map(|c| Piece { x: c.x, y: c.y, size: c.size }).collect()
        };
        let edible = |c: &Piece| c.size * EAT * 1.03 <= me.size && c.size * c.size / 100.0 >= 15.0;
        let threat = |c: &Piece| c.size >= 1.1 * ref_size;
        let threats: Vec<Piece> = if self.brave { Vec::new() } else { rivals.iter().copied().filter(|c| threat(c)).collect() };
        let at = |x: f64, y: f64| Out { x, y, split: false, eject: false };
        let gap_to = |c: &Piece| s.mine.iter().map(|m| (m.x - c.x).hypot(m.y - c.y) - m.size - c.size).fold(f64::MAX, f64::min);

        // 1. flee from what can eat us
        let (mut fx, mut fy, mut near) = (0.0, 0.0, false);
        let (cx, cy) = (s.mine.iter().map(|c| c.x * c.mass()).sum::<f64>() / total, s.mine.iter().map(|c| c.y * c.mass()).sum::<f64>() / total);
        for t in &threats {
            let g = gap_to(t);
            // a piece whose half could eat us can lunge by splitting
            let reach = if t.size * t.size / 200.0 >= EAT * EAT * me.mass() { ORACLE_LUNGE } else { 0.0 };
            if g < 600.0 + reach {
                let (dx, dy) = (cx - t.x, cy - t.y);
                let d = dx.hypot(dy).max(1.0);
                let wgt = 1.0 / g.max(20.0);
                fx += dx / d * wgt;
                fy += dy / d * wgt;
                if g < ORACLE_FLEE_GAP + reach {
                    near = true;
                }
            }
        }
        if near {
            let b = s.border;
            // away from the wall as well
            let m = 350.0;
            if cx - b.left < m { fx += 0.01; }
            if b.right - cx < m { fx -= 0.01; }
            if cy - b.top < m { fy += 0.01; }
            if b.bottom - cy < m { fy -= 0.01; }
            let n = fx.hypot(fy).max(1e-9);
            return at(cx + fx / n * 500.0, cy + fy / n * 500.0);
        }
        // 2. the nearest edible piece that is safe to approach
        let mut best: Option<(f64, Piece, f64)> = None;
        for c in rivals.iter().filter(|c| edible(c)) {
            let safe = threats.iter().all(|t| {
                (t.x - c.x).hypot(t.y - c.y) > t.size + ORACLE_SAFE_MARGIN && seg_dist((t.x, t.y), (me.x, me.y), (c.x, c.y)) > t.size + 60.0
            });
            let g = gap_to(c);
            // the user's rule is the nearest piece; the default weighs a piece's mass against the way to it (as the rule bots do)
            let score = if self.nearest { -g } else { c.size * c.size / 100.0 / (g.max(0.0) + 100.0) };
            if safe && best.map_or(true, |b| score > b.2) {
                best = Some((g, *c, score));
            }
        }
        if let Some((g, c, _)) = best {
            let mut out = at(c.x, c.y);
            // split onto it: the half cell must still eat it, the lunge (~800) must reach it, no threat near a half cell
            let half_ok = s.mine.len() <= 4 && me.mass() / 2.0 > c.size * c.size / 100.0 * EAT * EAT * 1.05;
            let lunge_ok = g > 60.0 && g < 650.0;
            let half = me.size / std::f64::consts::SQRT_2;
            let half_safe = threats.iter().all(|t| t.size < EAT * half * 1.05 || (t.x - me.x).hypot(t.y - me.y) - t.size - me.size > 1100.0);
            if self.split_cd == 0 && half_ok && lunge_ok && half_safe {
                out.split = true;
                self.split_cd = 10;
            }
            return out;
        }
        // 3. food (not next to a threat), else stay
        let food = s.others.iter().filter(|c| c.kind == Kind::Food && threats.iter().all(|t| (t.x - c.x).hypot(t.y - c.y) - t.size > 300.0))
            .min_by(|a, b| ((a.x - cx).hypot(a.y - cy)).total_cmp(&(b.x - cx).hypot(b.y - cy)));
        match food {
            Some(f) => at(f.x, f.y),
            None => at(cx, cy),
        }
    }
}

// ---------------------------------------------------------------------------------------------- one rep

enum Ctl {
    Drv(Box<AgentDriver>),
    Orc(Oracle),
}

fn apply(w: &mut World, pi: usize, o: &Out) {
    let p = &mut w.players[pi];
    p.mouse_in = Some(((o.x as i32) as f64, (o.y as i32) as f64));
    if o.split {
        p.p_split = true;
    }
    if o.eject {
        p.p_eject = true;
    }
}

#[derive(Default)]
struct Diag {
    samples: u32,
    edible_vis: u32,
    closing: u32,
    still: u32,
    edible_mass_sum: f64,
    threat_near: u32,
    fleeing: u32,
    splits: u32,
    max_cells: usize,
}

fn centroid(c: &[(f64, f64, f64)]) -> (f64, f64) {
    let t: f64 = c.iter().map(|c| c.2 * c.2).sum();
    (c.iter().map(|c| c.0 * c.2 * c.2).sum::<f64>() / t, c.iter().map(|c| c.1 * c.2 * c.2).sum::<f64>() / t)
}

/// Run one rep; returns its metrics (see docs/harvest-drill.md).
pub fn run_rep(agent: &AgentDef, cond: &Cond, seed: u64) -> Value {
    let (mut us_drv, mut comp_drv): (Option<Box<AgentDriver>>, Option<Box<AgentDriver>>) = (None, None);
    let comp_spec = cond.competitor.clone();
    let giant_spec = cond.giant_bot.clone();
    let mut giant_drv: Option<Box<AgentDriver>> = None;
    let (mut w, sc) = {
        let mut mk = |w: &mut World, role: &str| -> usize {
            match (role, &agent.kind) {
                ("us", AgentKind::Policy(spec)) => {
                    let d = AgentDriver::new(w, "us", spec, SYNC);
                    let pi = d.pi;
                    us_drv = Some(Box::new(d));
                    pi
                }
                ("giant", _) => {
                    let d = AgentDriver::new(w, "giant", giant_spec.as_ref().expect("giant mode bot needs --giant-bot"), SYNC);
                    let pi = d.pi;
                    giant_drv = Some(Box::new(d));
                    pi
                }
                ("comp", _) => {
                    let d = AgentDriver::new(w, "comp", comp_spec.as_ref().unwrap(), SYNC);
                    let pi = d.pi;
                    comp_drv = Some(Box::new(d));
                    pi
                }
                _ => {
                    let pi = w.add_player(PKind::Agent);
                    w.players[pi].name = "us".into();
                    pi
                }
            }
        };
        build_scene(cond, seed, &mut mk)
    };
    let mut us_ctl = match (&agent.kind, us_drv) {
        (AgentKind::Oracle { full, brave, nearest }, _) => Ctl::Orc(Oracle::new(*full, *brave, *nearest)),
        (_, Some(d)) => Ctl::Drv(d),
        _ => unreachable!(),
    };
    let (us, giant, comp) = (sc.us, sc.giant, sc.comp);
    let horizon = (cond.horizon_s * TPS).round() as u32;
    let mut windows: Vec<u32> = vec![10, 20, 30];
    if cond.horizon_s > 30.0 {
        windows.push(cond.horizon_s.round() as u32);
    }

    let mut popped_at: Option<u32> = None;
    let (mut g0, mut cap, mut pieces, mut stolen, mut lost_giant, mut lost_third, mut ate_comp) = (0.0f64, 0.0f64, 0u32, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut t_first, mut t_over, mut died_t): (Option<f64>, Option<f64>, Option<f64>) = (None, None, None);
    let mut died_by = "";
    let mut died_pre = false;
    let mut peak_ours = 0.0f64;
    let mut checkpoints: Vec<Value> = Vec::new();
    let mut next_cp = 0usize;
    let mut diag = Diag::default();
    let mut seen_kills = 0usize;
    let mut prev_c: Option<(f64, f64)> = None;
    let mut prev_n = 0usize;
    let mut dist_at_pop = f64::NAN;
    let (mut chain_cap, mut chain_over) = (0.0f64, false);
    let mut alive = true;
    let start_tick = w.tick;
    let trace = std::env::var("OGAR_DRILL_TRACE").is_ok();

    loop {
        if !(cond.giant_mode == GiantMode::Bot && popped_at.is_some()) {
            w.players[giant].mouse_in = Some(giant_aim(&w, &sc, cond.giant_mode, popped_at.is_some()));
        }
        w.tick();
        // pop detection: the giant has more than one cell for the first time
        if popped_at.is_none() && w.players[giant].cells.len() > 1 {
            popped_at = Some(w.tick);
            g0 = total_mass(&w, giant);
            (chain_cap, chain_over) = chain_bound(&w, us, giant);
            if alive {
                let (a, b) = (centroid(&cells_of(&w, us)), centroid(&cells_of(&w, giant)));
                dist_at_pop = (a.0 - b.0).hypot(a.1 - b.1);
            }
        }
        let t = popped_at.map(|p| w.tick - p);
        // kill events since the last look
        for k in &w.kill_log[seen_kills..] {
            let (v, kl) = (k.victim_owner as usize, k.killer_owner as usize);
            if kl == us && v == giant {
                cap += k.victim_mass;
                pieces += 1;
                if t_first.is_none() {
                    t_first = t.map(|t| t as f64 / TPS);
                }
            } else if Some(kl) == comp && v == giant {
                stolen += k.victim_mass;
            } else if kl == us && Some(v) == comp {
                ate_comp += k.victim_mass;
            } else if v == us {
                if kl == giant {
                    lost_giant += k.victim_mass;
                } else {
                    lost_third += k.victim_mass;
                }
            }
        }
        seen_kills = w.kill_log.len();
        if alive && w.players[us].cells.is_empty() {
            alive = false;
            died_pre = popped_at.is_none();
            died_t = t.map(|t| t as f64 / TPS);
            // who ate the last cell: the last kill event against us
            died_by = w.kill_log.iter().rev().find(|k| k.victim_owner as usize == us).map_or("other", |k| if k.killer_owner as usize == giant { "giant" } else { "third" });
        }
        let ours = total_mass(&w, us);
        let gm = total_mass(&w, giant);
        peak_ours = peak_ours.max(ours);
        if trace && t.map_or(false, |t| t % 25 == 0) {
            let o = cells_of(&w, us);
            let mut gp: Vec<(i64, i64)> = cells_of(&w, giant).iter().map(|g| ((g.2 * g.2 / 100.0) as i64, o.first().map_or(0, |c| (c.0 - g.0).hypot(c.1 - g.1) as i64))).collect();
            gp.sort();
            eprintln!("seed {seed} t={:>3} ours {:>5.0} cells {} alive {} giant {:>5.0} pieces(mass,dist) {:?}", t.unwrap() / 25, ours, o.len(), alive, gm, gp);
        }
        if let Some(t) = t {
            if t_over.is_none() && alive && ours > gm {
                t_over = Some(t as f64 / TPS);
            }
            while next_cp < windows.len() && t >= windows[next_cp] * TPS as u32 {
                checkpoints.push(json!({"s": windows[next_cp], "capture": cap / g0, "ours": ours, "giant": gm, "alive": alive, "overtaken": t_over.is_some()}));
                next_cp += 1;
            }
            if t >= horizon || (!alive && next_cp >= windows.len()) {
                break;
            }
            if !alive && t > 0 {
                // dead: the rest of the checkpoints are the same as now
                while next_cp < windows.len() {
                    checkpoints.push(json!({"s": windows[next_cp], "capture": cap / g0, "ours": ours, "giant": gm, "alive": false, "overtaken": t_over.is_some()}));
                    next_cp += 1;
                }
                break;
            }
        } else if w.tick - start_tick > MAX_PRE_TICKS || !alive {
            break;
        }
        if w.tick as u64 % SYNC == 0 {
            if alive {
                // diagnostics (post-pop only), then the agent's decision
                if popped_at.is_some() {
                    let mine = cells_of(&w, us);
                    let n = mine.len();
                    if n > prev_n && prev_n > 0 {
                        diag.splits += 1;
                    }
                    prev_n = n;
                    diag.max_cells = diag.max_cells.max(n);
                    let big = mine.iter().map(|c| c.2).fold(0.0, f64::max);
                    let small = mine.iter().filter(|c| c.2 * c.2 >= 0.25 * mine.iter().map(|c| c.2 * c.2).sum::<f64>()).map(|c| c.2).fold(f64::MAX, f64::min);
                    let (cx, cy) = centroid(&mine);
                    let mv = prev_c.map(|p| (cx - p.0, cy - p.1));
                    prev_c = Some((cx, cy));
                    let (vx0, vy0, vx1, vy1) = w.view_box(us);
                    diag.samples += 1;
                    let mut nearest_e: Option<(f64, (f64, f64))> = None;
                    let mut emass = 0.0;
                    let mut nearest_t: Option<(f64, (f64, f64))> = None;
                    for p in cells_of(&w, giant) {
                        let d = (p.0 - cx).hypot(p.1 - cy) - p.2;
                        let invis = p.0 < vx0 || p.0 > vx1 || p.1 < vy0 || p.1 > vy1;
                        if p.2 * EAT <= big && !invis {
                            emass += p.2 * p.2 / 100.0;
                            if nearest_e.map_or(true, |b| d < b.0) {
                                nearest_e = Some((d, (p.0 - cx, p.1 - cy)));
                            }
                        }
                        if p.2 >= EAT * small && d < 400.0 && nearest_t.map_or(true, |b| d < b.0) {
                            nearest_t = Some((d, (cx - p.0, cy - p.1)));
                        }
                    }
                    let cosine = |a: (f64, f64), b: (f64, f64)| (a.0 * b.0 + a.1 * b.1) / (a.0.hypot(a.1) * b.0.hypot(b.1)).max(1e-9);
                    if let Some((_, dirv)) = nearest_e {
                        diag.edible_vis += 1;
                        diag.edible_mass_sum += emass;
                        match mv {
                            Some(m) if m.0.hypot(m.1) > 2.0 => {
                                if cosine(m, dirv) > 0.5 {
                                    diag.closing += 1;
                                }
                            }
                            _ => diag.still += 1,
                        }
                    }
                    if let Some((_, away)) = nearest_t {
                        diag.threat_near += 1;
                        if let Some(m) = mv {
                            if m.0.hypot(m.1) > 2.0 && cosine(m, away) > 0.3 {
                                diag.fleeing += 1;
                            }
                        }
                    }
                }
                match &mut us_ctl {
                    Ctl::Drv(d) => d.step(&mut w),
                    Ctl::Orc(o) => {
                        let s = snapshot(&w, us);
                        if !s.dead {
                            let out = o.decide(&w, us, &s);
                            apply(&mut w, us, &out);
                        }
                    }
                }
            }
            if let (true, Some(d)) = (popped_at.is_some() && !w.players[giant].cells.is_empty(), giant_drv.as_mut()) {
                d.step(&mut w);
            }
            if let (Some(ci), Some(d)) = (comp, comp_drv.as_mut()) {
                if !w.players[ci].cells.is_empty() {
                    d.step(&mut w);
                }
            }
            w.kill_log.clear();
            seen_kills = 0;
        }
    }
    let popped = popped_at.is_some();
    let n = diag.samples.max(1) as f64;
    json!({
        "agent": agent.name, "cond": cond.key(), "seed": seed,
        "popped": popped, "died_pre": died_pre, "g0": g0, "dist_at_pop": dist_at_pop, "chain_cap": chain_cap, "chain_overtake": chain_over,
        "capture": if g0 > 0.0 { cap / g0 } else { 0.0 }, "captured_mass": cap, "pieces": pieces, "t_first": t_first, "t_overtake": t_over,
        "died": popped && !alive, "died_t": died_t, "died_by": if !alive && popped { died_by } else { "" },
        "lost_giant": lost_giant, "lost_third": lost_third, "stolen_by_comp": stolen, "ate_comp": ate_comp,
        "peak_ours": peak_ours, "checkpoints": checkpoints,
        "diag": {"samples": diag.samples, "edible_vis": diag.edible_vis as f64 / n, "closing_of_vis": diag.closing as f64 / diag.edible_vis.max(1) as f64,
                 "still_of_vis": diag.still as f64 / diag.edible_vis.max(1) as f64, "edible_mass_vis": diag.edible_mass_sum / diag.edible_vis.max(1) as f64,
                 "threat_near": diag.threat_near as f64 / n, "fleeing_of_threat": diag.fleeing as f64 / diag.threat_near.max(1) as f64,
                 "splits": diag.splits, "max_cells": diag.max_cells},
    })
}

// ---------------------------------------------------------------------------------------------- batch + stats

pub struct DrillOpts {
    pub agents: Vec<AgentDef>,
    pub conds: Vec<Cond>,
    pub reps: usize,
    pub seed: u64,
    pub threads: usize,
}

/// All (cond, agent, rep) jobs over `threads` workers; every agent sees the same seeds (paired comparison).
pub fn run_all(o: &DrillOpts) -> Vec<Value> {
    let jobs: Vec<(usize, usize, usize)> = (0..o.conds.len()).flat_map(|c| (0..o.agents.len()).flat_map(move |a| (0..o.reps).map(move |r| (c, a, r)))).collect();
    let next = AtomicUsize::new(0);
    let out: Mutex<Vec<(usize, Value)>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..o.threads.max(1) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= jobs.len() {
                    break;
                }
                let (c, a, r) = jobs[i];
                let row = run_rep(&o.agents[a], &o.conds[c], o.seed + r as u64);
                out.lock().unwrap().push((i, row));
                if i % 50 == 49 {
                    eprintln!("  {}/{} reps", i + 1, jobs.len());
                }
            });
        }
    });
    let mut v = out.into_inner().unwrap();
    v.sort_by_key(|(i, _)| *i);
    v.into_iter().map(|(_, r)| r).collect()
}

/// Wilson 95% interval of k successes in n.
pub fn wilson(k: usize, n: usize) -> (f64, f64, f64) {
    if n == 0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let (p, n_, z) = (k as f64 / n as f64, n as f64, 1.96f64);
    let den = 1.0 + z * z / n_;
    let c = (p + z * z / (2.0 * n_)) / den;
    let h = z * ((p * (1.0 - p) / n_ + z * z / (4.0 * n_ * n_)).sqrt()) / den;
    (p, (c - h).max(0.0), (c + h).min(1.0))
}

/// mean and 95% half-width of the mean
pub fn mean_ci(x: &[f64]) -> (f64, f64) {
    if x.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    if x.len() < 2 {
        return (m, f64::NAN);
    }
    let var = x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n - 1.0);
    (m, 1.96 * (var / n).sqrt())
}

fn median(x: &mut Vec<f64>) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    x.sort_by(|a, b| a.total_cmp(b));
    x[x.len() / 2]
}

fn jf(v: &Value, k: &str) -> f64 {
    v[k].as_f64().unwrap_or(f64::NAN)
}

/// Summary of the reps of one (cond, agent) cell.
pub fn summarize(rows: &[&Value]) -> Value {
    let valid: Vec<&&Value> = rows.iter().filter(|r| r["popped"] == true).collect();
    let n = valid.len();
    let col = |f: &dyn Fn(&Value) -> f64| -> Vec<f64> { valid.iter().map(|r| f(r)).collect() };
    let cp = |r: &Value, i: usize, k: &str| r["checkpoints"].get(i).map_or(f64::NAN, |c| c[k].as_f64().unwrap_or(f64::NAN));
    let mut win = serde_json::Map::new();
    let nwin = valid.first().map_or(0, |r| r["checkpoints"].as_array().map_or(0, |a| a.len()));
    for i in 0..nwin {
        let s = valid[0]["checkpoints"][i]["s"].as_i64().unwrap_or(0);
        let capv = col(&|r| cp(r, i, "capture"));
        let ratio = col(&|r| cp(r, i, "ours") / cp(r, i, "giant").max(1.0));
        let over = valid.iter().filter(|r| r["checkpoints"][i]["overtaken"] == true).count();
        let (cm, cc) = mean_ci(&capv);
        let (rm, rc) = mean_ci(&ratio);
        let (op, ol, oh) = wilson(over, n);
        win.insert(format!("{s}s"), json!({"capture": cm, "capture_ci": cc, "ours_over_giant": rm, "ours_over_giant_ci": rc, "overtake": op, "overtake_lo": ol, "overtake_hi": oh}));
    }
    let died = valid.iter().filter(|r| r["died"] == true).count();
    let by_giant = valid.iter().filter(|r| r["died_by"] == "giant").count();
    let by_third = valid.iter().filter(|r| r["died_by"] == "third").count();
    let (dp, dl, dh) = wilson(died, n);
    let over_all = valid.iter().filter(|r| r["t_overtake"].is_f64()).count();
    let (op, ol, oh) = wilson(over_all, n);
    let mut tover: Vec<f64> = valid.iter().filter_map(|r| r["t_overtake"].as_f64()).collect();
    let tover_mean = mean_ci(&tover).0;
    let got_first = valid.iter().filter(|r| r["t_first"].is_f64()).count();
    let tfirst: Vec<f64> = valid.iter().filter_map(|r| r["t_first"].as_f64()).collect();
    let (fp, fl, fh) = wilson(got_first, n);
    let m = |f: &dyn Fn(&Value) -> f64| mean_ci(&col(f)).0;
    json!({
        "n": n, "n_total": rows.len(), "no_pop": rows.len() - n,
        "windows": win,
        "overtake_any": op, "overtake_any_lo": ol, "overtake_any_hi": oh,
        "t_overtake_median": median(&mut tover), "t_overtake_mean": tover_mean,
        "died": dp, "died_lo": dl, "died_hi": dh, "died_by_giant": by_giant as f64 / n.max(1) as f64, "died_by_third": by_third as f64 / n.max(1) as f64,
        "first_piece": fp, "first_piece_lo": fl, "first_piece_hi": fh, "t_first_mean": mean_ci(&tfirst).0,
        "pieces": m(&|r| jf(r, "pieces")), "lost_third": m(&|r| jf(r, "lost_third")), "stolen_by_comp": m(&|r| jf(r, "stolen_by_comp")), "ate_comp": m(&|r| jf(r, "ate_comp")),
        "dist_at_pop": m(&|r| jf(r, "dist_at_pop")), "chain_cap": m(&|r| jf(r, "chain_cap")), "chain_overtake": m(&|r| if r["chain_overtake"] == true { 1.0 } else { 0.0 }), "g0": m(&|r| jf(r, "g0")), "peak_ours": m(&|r| jf(r, "peak_ours")),
        "diag": {
            "edible_vis": m(&|r| r["diag"]["edible_vis"].as_f64().unwrap_or(0.0)),
            "closing_of_vis": m(&|r| r["diag"]["closing_of_vis"].as_f64().unwrap_or(0.0)),
            "still_of_vis": m(&|r| r["diag"]["still_of_vis"].as_f64().unwrap_or(0.0)),
            "edible_mass_vis": m(&|r| r["diag"]["edible_mass_vis"].as_f64().unwrap_or(0.0)),
            "threat_near": m(&|r| r["diag"]["threat_near"].as_f64().unwrap_or(0.0)),
            "fleeing_of_threat": m(&|r| r["diag"]["fleeing_of_threat"].as_f64().unwrap_or(0.0)),
            "splits": m(&|r| r["diag"]["splits"].as_f64().unwrap_or(0.0)),
            "max_cells": m(&|r| r["diag"]["max_cells"].as_f64().unwrap_or(0.0)),
        },
    })
}

/// Group rows per (cond, agent) in first-seen order and summarise.
pub fn summarize_all(rows: &[Value]) -> Vec<Value> {
    let mut keys: Vec<(String, String)> = Vec::new();
    for r in rows {
        let k = (r["cond"].as_str().unwrap_or("").to_string(), r["agent"].as_str().unwrap_or("").to_string());
        if !keys.contains(&k) {
            keys.push(k);
        }
    }
    keys.into_iter().map(|(c, a)| {
        let g: Vec<&Value> = rows.iter().filter(|r| r["cond"] == c.as_str() && r["agent"] == a.as_str()).collect();
        let mut s = summarize(&g);
        s["cond"] = json!(c);
        s["agent"] = json!(a);
        s
    }).collect()
}

fn pct(x: f64) -> String {
    format!("{:.0}%", 100.0 * x)
}

/// A readable table of the summaries, one block per condition.
pub fn print_table(sums: &[Value]) {
    let mut last = String::new();
    for s in sums {
        let c = s["cond"].as_str().unwrap_or("");
        if c != last {
            println!("\n== {c}  (giant ~{:.0} mass, pop distance ~{:.0} centre, mass-chain bound: capture {:.0}%, overtake {:.0}%)", s["g0"].as_f64().unwrap_or(0.0),
                     s["dist_at_pop"].as_f64().unwrap_or(0.0), 100.0 * s["chain_cap"].as_f64().unwrap_or(0.0), 100.0 * s["chain_overtake"].as_f64().unwrap_or(0.0));
            println!("{:<18}{:>5}  {:<17}{:<17}{:<17}{:<17}{:>7}{:>7}{:>13}{:>8}", "agent", "n", "capture@10/20/30", "ours/giant@30", "overtake (any)", "died (giant/3rd)", "t_first", "pieces", "t_overtake", "lost3");
            last = c.to_string();
        }
        let w = |k: &str, f: &str| s["windows"][k][f].as_f64().unwrap_or(f64::NAN);
        let ci = |m: f64, h: f64| format!("{:.0}+-{:.0}%", 100.0 * m, 100.0 * h);
        let cap = format!("{}/{}/{}", pct(w("10s", "capture")), pct(w("20s", "capture")), ci(w("30s", "capture"), w("30s", "capture_ci")));
        let ratio = format!("{:.2}+-{:.2}", w("30s", "ours_over_giant"), w("30s", "ours_over_giant_ci"));
        let g = |k: &str| s[k].as_f64().unwrap_or(f64::NAN);
        let over = format!("{} [{}-{}]", pct(g("overtake_any")), pct(g("overtake_any_lo")), pct(g("overtake_any_hi")));
        let died = format!("{} ({}/{})", pct(g("died")), pct(g("died_by_giant")), pct(g("died_by_third")));
        println!("{:<18}{:>5}  {:<17}{:<17}{:<17}{:<17}{:>6.1}s{:>7.1}{:>12.1}s{:>8.0}", s["agent"].as_str().unwrap_or(""), s["n"], cap, ratio, over, died,
                 g("t_first_mean"), g("pieces"), g("t_overtake_median"), g("lost_third"));
    }
}

// ---------------------------------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot_of(w: &World) -> Vec<(u32, i64, i64, i64, i32)> {
        let mut v: Vec<_> = w.ents.iter().filter(|e| !e.removed && e.kind != crate::world::T_FOOD).map(|e| (e.id, (e.x * 1000.0) as i64, (e.y * 1000.0) as i64, (e.size * 1000.0) as i64, e.owner)).collect();
        v.sort();
        v
    }

    fn plain(w: &mut World, _: &str) -> usize {
        w.add_player(PKind::Agent)
    }

    fn cond(sc: Scenario) -> Cond {
        Cond::new(sc, 500.0, 5000.0, 600.0)
    }

    #[test]
    fn scene_is_deterministic_per_seed() {
        for sc in [Scenario::GiantPop, Scenario::GiantPopped] {
            let c = cond(sc);
            let (mut a, sa) = build_scene(&c, 7, &mut plain);
            let (mut b, sb) = build_scene(&c, 7, &mut plain);
            let (d, _) = build_scene(&c, 8, &mut plain);
            assert_eq!(snapshot_of(&a), snapshot_of(&b), "same seed, same scene");
            assert_ne!(snapshot_of(&a), snapshot_of(&d), "another seed, another scene");
            assert_eq!(sa.pop_point, sb.pop_point);
            // and the same evolution, pop included
            let mut popped = false;
            for _ in 0..400 {
                for (w, s) in [(&mut a, &sa), (&mut b, &sb)] {
                    let aim = giant_aim(w, s, GiantMode::Hunt, w.players[s.giant].cells.len() > 1);
                    w.players[s.giant].mouse_in = Some(aim);
                    w.tick();
                }
                popped |= a.players[sa.giant].cells.len() > 1;
            }
            assert!(popped, "{sc:?}: the giant never popped");
            assert_eq!(snapshot_of(&a), snapshot_of(&b), "{sc:?}: same seed, same evolution");
        }
    }

    #[test]
    fn scene_geometry_matches_the_condition() {
        let c = cond(Scenario::GiantPopped);
        let (w, s) = build_scene(&c, 3, &mut plain);
        let at = |pi: usize| {
            let e = &w.ents[w.players[pi].cells[0] as usize];
            (e.x, e.y, e.size)
        };
        let (u, g) = (at(s.us), at(s.giant));
        assert!((g.2 - size_of(5000.0)).abs() < 1e-6 && (u.2 - size_of(500.0)).abs() < 1e-6);
        let centre = (u.0 - g.0).hypot(u.1 - g.1);
        assert!((centre - (600.0 + g.2 - 34.0)).abs() < 1.0, "centre distance {centre}");
        let v = s.virus_pos;
        assert!(((u.0 - v.0).hypot(u.1 - v.1) - 600.0).abs() < 1.0, "us to virus");
    }

    #[test]
    fn oracle_rep_is_deterministic_and_pops() {
        let agent = AgentDef { name: "oracle".into(), kind: AgentKind::Oracle { full: true, brave: false, nearest: false } };
        let mut c = cond(Scenario::GiantPop);
        c.horizon_s = 10.0;
        let (r1, r2) = (run_rep(&agent, &c, 11), run_rep(&agent, &c, 11));
        assert_eq!(r1.to_string(), r2.to_string());
        assert_eq!(r1["popped"], true);
        assert!(r1["g0"].as_f64().unwrap() > 4000.0);
    }

    /// `cargo test dump_pop -- --ignored --nocapture`: the piece masses and spread right after the pop, then 10 s later.
    #[test]
    #[ignore]
    fn dump_pop_pieces() {
        let c = cond(Scenario::GiantPopped);
        let (mut w, s) = build_scene(&c, 3, &mut plain);
        for t in 0..=250 {
            let aim = giant_aim(&w, &s, GiantMode::Passive, true);
            w.players[s.giant].mouse_in = Some(aim);
            w.tick();
            if t == 2 || t == 25 || t == 250 {
                let mut m: Vec<(i64, i64)> = cells_of(&w, s.giant).iter().map(|c| ((c.2 * c.2 / 100.0) as i64, (c.0 - w.ents[w.players[s.us].cells[0] as usize].x).hypot(c.1 - w.ents[w.players[s.us].cells[0] as usize].y) as i64)).collect();
                m.sort();
                eprintln!("t={t} {} pieces (mass, dist to us): {m:?}", m.len());
            }
        }
    }

    #[test]
    fn wilson_and_mean_ci_are_sane() {
        let (p, lo, hi) = wilson(50, 100);
        assert!((p - 0.5).abs() < 1e-9 && lo < 0.45 && hi > 0.55);
        let (m, h) = mean_ci(&[1.0, 2.0, 3.0]);
        assert!((m - 2.0).abs() < 1e-9 && h > 0.0);
    }
}
