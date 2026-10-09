//! `ogar escape`: the escape drill (docs/escape-drill.md).
//!
//! How long does an agent of mass M survive a bigger, human-like chaser? The chaser is scripted to play like the user does
//! in the recorded games (docs/human/2026-10-09-50bots.md): it pursues with a lead (intercept point), herds the target
//! towards the nearest wall (approaches from the inner side), split-kills when the target is inside its split reach and a
//! half cell can eat it (double split when only the second one reaches), keeps its pieces together and does not pop on
//! viruses. Everything is a function of (condition, seed): paired comparisons across agents work as in drill.rs.
//!
//! Geometry (all in world units): `gap` is the edge-to-edge distance, centre distance - both radii, at the start.
//! Placement `open` (map centre), `wall` (300 from a wall, the chaser on the inner side), `corner` (300 from two walls, the
//! chaser towards the middle). `virus`: one virus about 700 from us at a right angle to the line chaser-us. `pieces`:
//! our mass starts split into 1 / 2 / 4 cells (two split commands, a second to settle) before the chaser appears.
//!
//! Split reach (centre distance): a split child (mass / 2, size / sqrt 2) is thrown 780 * size^0.0122 (the real boost,
//! it travels exactly that far) and eats a cell whose centre is closer than child size - target size / 3.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use arena_client::steer::tick_speed;
use serde_json::{json, Value};

use crate::agent::{snapshot, AgentDriver};
use crate::config::Config;
use crate::drill::{mean_ci, wilson, AgentDef, AgentKind, Oracle};
use crate::rng::Rng;
use crate::world::{PKind, World, T_VIRUS};

const TPS: f64 = 25.0;
const SYNC: u64 = 2;
/// size ratio at which a cell eats another (config player_eat_mult)
const EAT: f64 = 1.15;
/// the wall/corner starts this far from the wall (edge of our cell to the wall)
const WALL_START: f64 = 300.0;
/// "touched a wall": the edge of one of our cells came within this of it (the rule bots hover at ~30)
const NEAR_WALL: f64 = 100.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Place {
    Open,
    Wall,
    Corner,
}

impl Place {
    pub fn name(self) -> &'static str {
        match self {
            Place::Open => "open",
            Place::Wall => "wall",
            Place::Corner => "corner",
        }
    }
    pub fn parse(s: &str) -> Result<Place, String> {
        match s {
            "open" => Ok(Place::Open),
            "wall" => Ok(Place::Wall),
            "corner" => Ok(Place::Corner),
            _ => Err(format!("unknown place {s:?} (open | wall | corner)")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EscCond {
    pub our_mass: f64,
    /// chaser mass / our mass (capped at the maximum cell mass)
    pub ratio: f64,
    pub gap: f64,
    pub place: Place,
    pub virus: bool,
    /// our mass starts in this many cells (1, 2 or 4)
    pub pieces: usize,
    pub horizon_s: f64,
    pub map: f64,
    pub food_mult: f64,
}

impl EscCond {
    pub fn new(our_mass: f64, ratio: f64, gap: f64, place: Place) -> EscCond {
        EscCond { our_mass, ratio, gap, place, virus: false, pieces: 1, horizon_s: 60.0, map: 7000.0, food_mult: 2.0 }
    }

    pub fn key(&self) -> String {
        format!("esc/our{}/x{}/gap{}/{}{}{}", self.our_mass, self.ratio, self.gap, self.place.name(),
                if self.virus { "/virus" } else { "" }, if self.pieces > 1 { format!("/p{}", self.pieces) } else { String::new() })
    }

    pub fn chaser_mass(&self) -> f64 {
        (self.our_mass * self.ratio).min(22500.0)
    }
}

fn size_of(mass: f64) -> f64 {
    (100.0 * mass).sqrt()
}

/// Distance a split child is thrown (the world's boost: 780 * size^0.0122, travelled in full).
pub fn split_throw(child_size: f64) -> f64 {
    780.0 * child_size.powf(0.0122)
}

/// Centre distance up to which a cell of size `cs` splitting reaches (and its child, if big enough, eats) a cell of size `ts`.
pub fn split_reach(cs: f64, ts: f64) -> f64 {
    let child = cs / std::f64::consts::SQRT_2;
    split_throw(child) + child - ts / 3.0
}

/// Plain eating distance: centre closer than eater size - prey size / 3.
pub fn eat_range(cs: f64, ts: f64) -> f64 {
    cs - ts / 3.0
}

fn cells_of(w: &World, pi: usize) -> Vec<(u32, f64, f64, f64)> {
    w.players[pi].cells.iter().map(|&h| {
        let e = &w.ents[h as usize];
        (e.id, e.x, e.y, e.size)
    }).collect()
}

fn biggest(c: &[(u32, f64, f64, f64)]) -> Option<(u32, f64, f64, f64)> {
    c.iter().copied().max_by(|a, b| a.3.total_cmp(&b.3))
}

fn centroid(c: &[(u32, f64, f64, f64)]) -> (f64, f64) {
    let t: f64 = c.iter().map(|c| c.3 * c.3).sum::<f64>().max(1e-9);
    (c.iter().map(|c| c.1 * c.3 * c.3).sum::<f64>() / t, c.iter().map(|c| c.2 * c.3 * c.3).sum::<f64>() / t)
}

fn viruses(w: &World) -> Vec<(f64, f64, f64)> {
    w.ents.iter().filter(|e| !e.removed && e.kind == T_VIRUS).map(|e| (e.x, e.y, e.size)).collect()
}

// ---------------------------------------------------------------------------------------------- scene

pub struct EscScene {
    pub us: usize,
    pub chaser: usize,
    pub virus_pos: Option<(f64, f64)>,
    pub start_gap: f64,
    pub start_centre: f64,
}

fn rot(v: (f64, f64), a: f64) -> (f64, f64) {
    (v.0 * a.cos() - v.1 * a.sin(), v.0 * a.sin() + v.1 * a.cos())
}

/// Build the world of one rep. `mk(world)` creates our player (so the caller decides what drives it).
pub fn build_escape_scene(cond: &EscCond, seed: u64, mk: &mut dyn FnMut(&mut World) -> usize) -> (World, EscScene) {
    let mut cfg = Config::new(cond.map, 0, cond.food_mult);
    cfg.virus_min_amount = 0;
    let mut w = World::new(cfg, seed ^ 0x5eed_e5c);
    let mut lr = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0xE5C4_9E) | 1);
    let us = mk(&mut w);
    let chaser = w.add_player(PKind::Agent);
    w.players[chaser].name = "chaser".into();
    let (us_size, ch_mass) = (size_of(cond.our_mass), cond.chaser_mass());
    let ch_size = size_of(ch_mass);
    let h = cond.map / 2.0;
    // our start point and the direction towards the chaser
    let (p0, dir0) = match cond.place {
        Place::Open => {
            let p = ((lr.f64() - 0.5) * 1400.0, (lr.f64() - 0.5) * 1400.0);
            let a = lr.f64() * std::f64::consts::TAU;
            (p, (a.cos(), a.sin()))
        }
        Place::Wall => {
            let side = (lr.f64() * 4.0) as usize % 4;
            let along = (lr.f64() - 0.5) * 3000.0;
            let off = h - us_size / 2.0 - WALL_START - us_size / 2.0;
            let (p, n) = match side {
                0 => ((-off, along), (1.0, 0.0)),
                1 => ((off, along), (-1.0, 0.0)),
                2 => ((along, -off), (0.0, 1.0)),
                _ => ((along, off), (0.0, -1.0)),
            };
            (p, rot(n, (lr.f64() - 0.5) * 100f64.to_radians()))
        }
        Place::Corner => {
            let (sx, sy) = (if lr.f64() < 0.5 { -1.0 } else { 1.0 }, if lr.f64() < 0.5 { -1.0 } else { 1.0 });
            let off = h - us_size - WALL_START;
            let n = (-sx / 2f64.sqrt(), -sy / 2f64.sqrt());
            ((sx * off, sy * off), rot(n, (lr.f64() - 0.5) * 40f64.to_radians()))
        }
    };
    spawn_at(&mut w, us, cond.our_mass, p0);
    if cond.pieces > 1 {
        // two split commands thrown sideways (perpendicular to the chaser line), then a second to settle
        let perp = (-dir0.1, dir0.0);
        let rounds = if cond.pieces >= 4 { 2 } else { 1 };
        for _ in 0..rounds {
            w.players[us].mouse_in = Some((p0.0 + perp.0 * 600.0, p0.1 + perp.1 * 600.0));
            w.players[us].p_split = true;
            for _ in 0..4 {
                w.tick();
            }
        }
        for _ in 0..25 {
            w.tick();
        }
    }
    let ours = cells_of(&w, us);
    let c0 = centroid(&ours);
    let eq_size = size_of(cond.our_mass); // the whole mass as one cell: the geometry of the start
    let lim = h - ch_size / 2.0 - 60.0;
    let mut chosen: Option<((f64, f64), f64)> = None;
    'outer: for shrink in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4] {
        let centre = (cond.gap + eq_size + ch_size) * shrink;
        for k in 0..25 {
            // 0, +15, -15, +30, ...: the intended direction first
            let a = ((k + 1) / 2) as f64 * 15f64.to_radians() * if k % 2 == 1 { 1.0 } else { -1.0 };
            let d = rot(dir0, a);
            let p = (c0.0 + d.0 * centre, c0.1 + d.1 * centre);
            if p.0.abs() < lim && p.1.abs() < lim {
                chosen = Some((p, centre));
                break 'outer;
            }
        }
    }
    let (cp, centre) = chosen.expect("no room for the chaser");
    spawn_at(&mut w, chaser, ch_mass, cp);
    let mut virus_pos = None;
    if cond.virus {
        let away = ((c0.0 - cp.0) / centre, (c0.1 - cp.1) / centre);
        let side = if lr.f64() < 0.5 { 1.0 } else { -1.0 };
        for k in 0..12 {
            let a = side * (70f64.to_radians() + 0.2 * (k / 2) as f64) * if k % 2 == 0 { 1.0 } else { -1.0 };
            let d = rot(away, a);
            let v = (c0.0 + d.0 * 700.0, c0.1 + d.1 * 700.0);
            if v.0.abs() < h - 200.0 && v.1.abs() < h - 200.0 {
                w.add_virus_at(v.0, v.1);
                virus_pos = Some(v);
                break;
            }
        }
    }
    let gap = centre - eq_size - ch_size;
    (w, EscScene { us, chaser, virus_pos, start_gap: gap, start_centre: centre })
}

fn spawn_at(w: &mut World, pi: usize, mass: f64, pos: (f64, f64)) {
    w.players[pi].spawn_size = size_of(mass);
    let c = w.random_color();
    w.players[pi].color = c;
    w.spawn_player(pi, pos);
    let h = w.players[pi].cells[0];
    w.place_cell(h, pos.0, pos.1);
}

// ---------------------------------------------------------------------------------------------- the chaser

pub struct Chaser {
    pi: usize,
    split_cd: u32,
    /// a second split queued (ticks until it fires)
    second: Option<u32>,
    prev: std::collections::HashMap<u32, (f64, f64)>,
    pub splits: u32,
    pub double_splits: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct ChaserOut {
    pub aim: (f64, f64),
    pub split: bool,
}

impl Chaser {
    pub fn new(pi: usize) -> Chaser {
        Chaser { pi, split_cd: 0, second: None, prev: Default::default(), splits: 0, double_splits: 0 }
    }

    /// One decision (every SYNC ticks). Pure function of the world plus the chaser's memory of the last decision.
    pub fn decide(&mut self, w: &World, us: usize) -> ChaserOut {
        self.split_cd = self.split_cd.saturating_sub(SYNC as u32);
        let mine = cells_of(w, self.pi);
        let ours = cells_of(w, us);
        let main = match biggest(&mine) {
            Some(m) => m,
            None => return ChaserOut { aim: (0.0, 0.0), split: false },
        };
        let own_c = centroid(&mine);
        if ours.is_empty() {
            return ChaserOut { aim: own_c, split: false };
        }
        let b = w.border;
        // target: the biggest of our cells it can eat, else the nearest
        let eatable: Vec<_> = ours.iter().copied().filter(|t| main.3 >= EAT * t.3).collect();
        let pool = if eatable.is_empty() { &ours } else { &eatable };
        let t = pool.iter().copied().max_by(|a, b| (a.3 * a.3).total_cmp(&(b.3 * b.3))).unwrap();
        // our velocity per tick, from the last decision
        let v = match self.prev.get(&t.0) {
            Some(&(px, py)) => ((t.1 - px) / SYNC as f64, (t.2 - py) / SYNC as f64),
            None => (0.0, 0.0),
        };
        self.prev.clear();
        for o in &ours {
            self.prev.insert(o.0, (o.1, o.2));
        }
        let s = tick_speed(main.3);
        let (dx, dy) = (t.1 - main.1, t.2 - main.2);
        let dist = dx.hypot(dy);
        // intercept: smallest positive t with |T + v t - C| = s t
        let (a, bq, c) = (v.0 * v.0 + v.1 * v.1 - s * s, 2.0 * (dx * v.0 + dy * v.1), dx * dx + dy * dy);
        let mut tt = dist / s;
        if a.abs() > 1e-9 {
            let disc = bq * bq - 4.0 * a * c;
            if disc >= 0.0 {
                let (r1, r2) = ((-bq - disc.sqrt()) / (2.0 * a), (-bq + disc.sqrt()) / (2.0 * a));
                let best = [r1, r2].into_iter().filter(|r| *r > 0.0).fold(f64::INFINITY, f64::min);
                if best.is_finite() {
                    tt = best;
                }
            }
        }
        let tt = tt.min(60.0);
        let mut aim = (t.1 + v.0 * tt, t.2 + v.1 * tt);
        // herding: the nearest wall to the target; if the chaser is not yet on the inner side, come in from there
        let edges = [t.1 - b.0, b.2 - t.1, t.2 - b.1, b.3 - t.2];
        let (wi, wd) = edges.iter().copied().enumerate().fold((0, f64::INFINITY), |m, (i, e)| if e < m.1 { (i, e) } else { m });
        let inward = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)][wi];
        let reach = split_reach(main.3, t.3);
        if wd < 2000.0 && dist > 0.5 * reach {
            let inner = ((main.1 - t.1) * inward.0 + (main.2 - t.2) * inward.1) / dist.max(1.0);
            if inner < 0.5 {
                let k = (0.3 * dist).min(600.0);
                aim = (aim.0 + inward.0 * k, aim.1 + inward.1 * k);
            }
        }
        // do not pop on a virus (a human keeps clear of it): go round, on the side the aim line passes
        for (vx, vy, vs) in viruses(w) {
            let cm = main.3 * main.3 / 100.0;
            if cm < 1.33 * vs * vs / 100.0 {
                continue;
            }
            let clear = vs + main.3 * 0.9;
            let (ax, ay) = (aim.0 - main.1, aim.1 - main.2);
            let l = ax.hypot(ay).max(1.0);
            let (ux, uy) = (ax / l, ay / l);
            let (rx, ry) = (vx - main.1, vy - main.2);
            let along = rx * ux + ry * uy;
            let across = rx * uy - ry * ux;
            if along > 0.0 && along < l + clear && across.abs() < clear {
                let side = if across > 0.0 { 1.0 } else { -1.0 };
                aim = (vx - uy * side * clear * 1.15, vy + ux * side * clear * 1.15);
                break;
            }
        }
        aim = (aim.0.clamp(b.0, b.2), aim.1.clamp(b.1, b.3));
        let mut split = false;
        if let Some(n) = self.second {
            if n <= SYNC as u32 {
                self.second = None;
                split = true;
            } else {
                self.second = Some(n - SYNC as u32);
            }
        }
        // split-kill: a half cell can eat the target and reaches it (with the target's flight, 10 ticks, taken off)
        if !split && self.split_cd == 0 && mine.len() < 16 && main.3 * main.3 / 200.0 > 0.0 {
            let dd = (t.1 - main.1).hypot(t.2 - main.2);
            let child = main.3 / std::f64::consts::SQRT_2;
            let reach1 = split_reach(main.3, t.3) - 10.0 * tick_speed(t.3) * 0.5;
            let eats1 = child >= EAT * t.3;
            let eats2 = child / std::f64::consts::SQRT_2 >= EAT * t.3;
            if eats1 && dd <= reach1 {
                split = true;
                self.splits += 1;
                self.split_cd = 50;
            } else if eats2 && dd <= 1.8 * reach1 && dd > reach1 && mine.len() == 1 {
                // double split: the second press 4 ticks later throws the pieces on
                split = true;
                self.splits += 1;
                self.double_splits += 1;
                self.split_cd = 50;
                self.second = Some(4);
            }
        }
        // recombine: pieces and the target far away -> gather on the centroid of the pieces
        if mine.len() > 1 && !split && dist > 2.0 * reach {
            aim = own_c;
        }
        ChaserOut { aim, split }
    }
}

// ---------------------------------------------------------------------------------------------- one rep

enum Ctl {
    Drv(Box<AgentDriver>),
    Orc(Oracle),
}

fn apply(w: &mut World, pi: usize, aim: (f64, f64), split: bool) {
    let p = &mut w.players[pi];
    p.mouse_in = Some(((aim.0 as i32) as f64, (aim.1 as i32) as f64));
    if split {
        p.p_split = true;
    }
}

pub fn run_escape_rep(agent: &AgentDef, cond: &EscCond, seed: u64) -> Value {
    let mut us_drv: Option<Box<AgentDriver>> = None;
    let (mut w, sc) = {
        let mut mk = |w: &mut World| -> usize {
            match &agent.kind {
                AgentKind::Policy(spec) => {
                    let d = AgentDriver::new(w, "us", spec, SYNC);
                    let pi = d.pi;
                    us_drv = Some(Box::new(d));
                    pi
                }
                _ => {
                    let pi = w.add_player(PKind::Agent);
                    w.players[pi].name = "us".into();
                    pi
                }
            }
        };
        build_escape_scene(cond, seed, &mut mk)
    };
    let mut ctl = match (&agent.kind, us_drv) {
        (AgentKind::Oracle { full, brave, nearest, no_split }, _) => Ctl::Orc(Oracle::new(*full, *brave, *nearest, *no_split)),
        (_, Some(d)) => Ctl::Drv(d),
        _ => unreachable!(),
    };
    let mut chaser = Chaser::new(sc.chaser);
    let us = sc.us;
    let m0: f64 = w.players[us].total_mass(&w);
    let horizon = (cond.horizon_s * TPS).round() as u32;
    let (t30, t60) = ((30.0 * TPS) as u32, (60.0 * TPS) as u32);
    let (mut mass30, mut mass60): (Option<f64>, Option<f64>) = (None, None);
    let (mut died_t, mut died_where): (Option<f64>, &str) = (None, "");
    let (mut n_in_reach, mut n_in_lethal, mut n_in_eat, mut samples) = (0u32, 0u32, 0u32, 0u32);
    let (mut wall_first, mut corner_first): (Option<f64>, Option<f64>) = (None, None);
    let (mut ever_split, mut virus_popped, mut max_cells) = (false, false, w.players[us].cells.len());
    let mut last_wall = -1000i64;
    let mut last_corner = -1000i64;
    let mut last_pieces = -1000i64;
    let mut prev_n = w.players[us].cells.len();
    let mut prev_ch_n = 1usize;
    let mut last_ch_split_cmd = -1000i64;
    let mut min_edge_gap = f64::INFINITY;
    let mut peak = m0;
    let mut alive = true;
    let mut path_len = 0.0f64;
    let mut prev_c: Option<(f64, f64)> = None;
    let trace = std::env::var("OGAR_ESC_TRACE").is_ok();
    let start = w.tick;
    for _ in 0..horizon {
        let t = (w.tick - start) as i64;
        if w.tick as u64 % SYNC == 0 {
            if !w.players[sc.chaser].cells.is_empty() {
                let o = chaser.decide(&w, us);
                if o.split {
                    last_ch_split_cmd = t;
                }
                apply(&mut w, sc.chaser, o.aim, o.split);
            }
            if alive {
                match &mut ctl {
                    Ctl::Drv(d) => d.step(&mut w),
                    Ctl::Orc(o) => {
                        let s = snapshot(&w, us);
                        if !s.dead {
                            let out = o.decide(&w, us, &s);
                            apply(&mut w, us, (out.x, out.y), out.split);
                        }
                    }
                }
            }
        }
        w.tick();
        let t = (w.tick - start) as i64;
        let ts = t as f64 / TPS;
        let ours = cells_of(&w, us);
        let ch = cells_of(&w, sc.chaser);
        if alive && ours.is_empty() {
            alive = false;
            died_t = Some(ts);
            let recent = |x: i64| t - x <= (3.0 * TPS) as i64;
            died_where = if recent(last_pieces) { "pieces" } else if recent(last_corner) { "corner" } else if recent(last_wall) { "wall" } else { "open" };
        }
        if t as u32 == t30 {
            mass30 = Some(if alive { w.players[us].total_mass(&w) } else { 0.0 });
        }
        if t as u32 == t60 {
            mass60 = Some(if alive { w.players[us].total_mass(&w) } else { 0.0 });
        }
        if !alive {
            break;
        }
        if w.tick as u64 % SYNC == 0 {
            samples += 1;
            let mass = w.players[us].total_mass(&w);
            peak = peak.max(mass);
            let n = ours.len();
            max_cells = max_cells.max(n);
            if n > prev_n && t - last_ch_split_cmd > 0 {
                ever_split = true;
            }
            prev_n = n;
            if n > 1 {
                last_pieces = t;
            }
            if ch.len() > prev_ch_n && t - last_ch_split_cmd > 6 {
                virus_popped = true;
            }
            prev_ch_n = ch.len();
            // wall / corner: a cell pressed on the border
            let b = w.border;
            let mut wall = false;
            let mut corner = false;
            for c in &ours {
                let (ex, ey) = ((c.1 - b.0).min(b.2 - c.1), (c.2 - b.1).min(b.3 - c.2));
                let (px, py) = (ex <= c.3 + NEAR_WALL, ey <= c.3 + NEAR_WALL);
                wall |= px || py;
                corner |= px && py;
            }
            if wall {
                last_wall = t;
                wall_first.get_or_insert(ts);
            }
            if corner {
                last_corner = t;
                corner_first.get_or_insert(ts);
            }
            if let (Some(m), Some(u)) = (biggest(&ch), biggest(&ours)) {
                let d = (m.1 - u.1).hypot(m.2 - u.2);
                min_edge_gap = min_edge_gap.min(d - m.3 - u.3);
                if d <= split_reach(m.3, u.3) {
                    n_in_reach += 1;
                    if m.3 / std::f64::consts::SQRT_2 >= EAT * u.3 {
                        n_in_lethal += 1;
                    }
                }
                if d <= eat_range(m.3, u.3) + 0.0 {
                    n_in_eat += 1;
                }
                if trace && t % 25 == 0 {
                    eprintln!("seed {seed} t={:>3} ours {:>6.0} cells {} chaser {:>6.0}x{} dist {:>6.0} reach {:>5.0} pos ({:>6.0},{:>6.0})", t / 25, mass, n, w.players[sc.chaser].total_mass(&w), ch.len(), d, split_reach(m.3, u.3), u.1, u.2);
                }
            }
            let cc = centroid(&ours);
            if let Some(p) = prev_c {
                path_len += (cc.0 - p.0).hypot(cc.1 - p.1);
            }
            prev_c = Some(cc);
        }
    }
    let end_alive = alive;
    let mass_end = if alive { w.players[us].total_mass(&w) } else { 0.0 };
    // the escape method of a survivor (first that applies)
    let method = if !alive {
        "dead"
    } else if virus_popped {
        "virus"
    } else if ever_split {
        "split"
    } else if wall_first.is_some() {
        "wall_slide"
    } else if (n_in_reach as f64) < 0.2 * samples.max(1) as f64 {
        "outrun"
    } else {
        "other"
    };
    let n = samples.max(1) as f64;
    let dt = SYNC as f64 / TPS;
    json!({
        "agent": agent.name, "cond": cond.key(), "seed": seed,
        "start_gap": sc.start_gap, "start_centre": sc.start_centre, "m0": m0, "chaser_mass": cond.chaser_mass(),
        "alive30": died_t.map_or(true, |d| d > 30.0), "alive60": died_t.map_or(true, |d| d > 60.0),
        "alive_end": end_alive,
        "mass30": mass30.unwrap_or(mass_end), "mass60": mass60.unwrap_or(mass_end), "mass_end": mass_end, "peak": peak,
        "died_t": died_t, "died_where": died_where,
        "t_in_reach": n_in_reach as f64 * dt, "t_in_lethal": n_in_lethal as f64 * dt, "t_in_eat": n_in_eat as f64 * dt, "frac_in_reach": n_in_reach as f64 / n,
        "t_wall": wall_first, "t_corner": corner_first, "touched_wall": wall_first.is_some(), "touched_corner": corner_first.is_some(),
        "ever_split": ever_split, "virus_popped": virus_popped, "max_cells": max_cells, "min_edge_gap": min_edge_gap, "path_len": path_len,
        "chaser_splits": chaser.splits, "chaser_double": chaser.double_splits, "method": method,
    })
}

// ---------------------------------------------------------------------------------------------- batch + stats

pub struct EscOpts {
    pub agents: Vec<AgentDef>,
    pub conds: Vec<EscCond>,
    pub reps: usize,
    pub seed: u64,
    pub threads: usize,
}

pub fn run_all(o: &EscOpts) -> Vec<Value> {
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
                let row = run_escape_rep(&o.agents[a], &o.conds[c], o.seed + r as u64);
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

fn jf(v: &Value, k: &str) -> f64 {
    v[k].as_f64().unwrap_or(f64::NAN)
}

fn jb(v: &Value, k: &str) -> bool {
    v[k] == true
}

/// Summary of the reps of one (cond, agent) cell.
pub fn summarize(rows: &[&Value]) -> Value {
    let n = rows.len();
    let col = |f: &dyn Fn(&Value) -> f64| -> Vec<f64> { rows.iter().map(|r| f(r)).collect() };
    let rate = |f: &dyn Fn(&Value) -> bool| -> Value {
        let k = rows.iter().filter(|r| f(r)).count();
        let (p, lo, hi) = wilson(k, n);
        json!({"p": p, "lo": lo, "hi": hi, "k": k})
    };
    let mean = |f: &dyn Fn(&Value) -> f64| -> Value {
        let (m, h) = mean_ci(&col(f));
        json!({"m": m, "ci": h})
    };
    let mut methods = serde_json::Map::new();
    for r in rows {
        let m = r["method"].as_str().unwrap_or("?").to_string();
        let c = methods.get(&m).and_then(|v| v.as_u64()).unwrap_or(0);
        methods.insert(m, json!(c + 1));
    }
    let mut where_ = serde_json::Map::new();
    for r in rows.iter().filter(|r| !jb(r, "alive_end")) {
        let m = r["died_where"].as_str().unwrap_or("?").to_string();
        let c = where_.get(&m).and_then(|v| v.as_u64()).unwrap_or(0);
        where_.insert(m, json!(c + 1));
    }
    json!({
        "n": n,
        "survive30": rate(&|r| jb(r, "alive30")), "survive60": rate(&|r| jb(r, "alive60")),
        "mass30": mean(&|r| jf(r, "mass30") / jf(r, "m0")), "mass60": mean(&|r| jf(r, "mass60") / jf(r, "m0")),
        "t_in_reach": mean(&|r| jf(r, "t_in_reach")), "t_in_lethal": mean(&|r| jf(r, "t_in_lethal")), "t_in_eat": mean(&|r| jf(r, "t_in_eat")),
        "touched_wall": rate(&|r| jb(r, "touched_wall")), "touched_corner": rate(&|r| jb(r, "touched_corner")),
        "ever_split": rate(&|r| jb(r, "ever_split")), "virus_popped": rate(&|r| jb(r, "virus_popped")),
        "start_gap": mean(&|r| jf(r, "start_gap")), "min_edge_gap": mean(&|r| jf(r, "min_edge_gap")),
        "methods": methods, "died_where": where_,
    })
}

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

fn pc(v: &Value) -> String {
    format!("{:.0}% [{:.0}-{:.0}]", 100.0 * v["p"].as_f64().unwrap_or(f64::NAN), 100.0 * v["lo"].as_f64().unwrap_or(f64::NAN), 100.0 * v["hi"].as_f64().unwrap_or(f64::NAN))
}

fn mv(v: &Value, scale: f64, dec: usize) -> String {
    format!("{:.dec$}+-{:.dec$}", scale * v["m"].as_f64().unwrap_or(f64::NAN), scale * v["ci"].as_f64().unwrap_or(f64::NAN), dec = dec)
}

pub fn print_table(sums: &[Value]) {
    let mut last = String::new();
    for s in sums {
        let c = s["cond"].as_str().unwrap_or("");
        if c != last {
            println!("\n== {c}  (start gap ~{:.0})", s["start_gap"]["m"].as_f64().unwrap_or(f64::NAN));
            println!("{:<16}{:>4}  {:<16}{:<16}{:<12}{:<12}{:<10}{:<10}{:<14}{:<12}", "agent", "n", "survive30", "survive60", "mass30", "mass60", "in_reach s", "in_lethal", "wall/corner", "methods");
            last = c.to_string();
        }
        let mut ms: Vec<String> = s["methods"].as_object().map(|m| m.iter().map(|(k, v)| format!("{k}:{v}")).collect()).unwrap_or_default();
        if let Some(d) = s["died_where"].as_object().filter(|d| !d.is_empty()) {
            ms.push(format!("(died at {})", d.iter().map(|(k, v)| format!("{k}:{v}")).collect::<Vec<_>>().join(" ")));
        }
        println!("{:<16}{:>4}  {:<16}{:<16}{:<12}{:<12}{:<10}{:<10}{:<14}{}", s["agent"].as_str().unwrap_or(""), s["n"], pc(&s["survive30"]), pc(&s["survive60"]),
                 mv(&s["mass30"], 1.0, 2), mv(&s["mass60"], 1.0, 2), mv(&s["t_in_reach"], 1.0, 1), mv(&s["t_in_lethal"], 1.0, 1),
                 format!("{:.0}%/{:.0}%", 100.0 * s["touched_wall"]["p"].as_f64().unwrap_or(f64::NAN), 100.0 * s["touched_corner"]["p"].as_f64().unwrap_or(f64::NAN)), ms.join(" "));
    }
}

// ---------------------------------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(w: &mut World) -> usize {
        w.add_player(PKind::Agent)
    }

    fn snap(w: &World) -> Vec<(u32, i64, i64, i64, i32)> {
        let mut v: Vec<_> = w.ents.iter().filter(|e| !e.removed && e.kind != crate::world::T_FOOD)
            .map(|e| (e.id, (e.x * 1000.0) as i64, (e.y * 1000.0) as i64, (e.size * 1000.0) as i64, e.owner)).collect();
        v.sort();
        v
    }

    #[test]
    fn scene_is_deterministic_per_seed() {
        for place in [Place::Open, Place::Wall, Place::Corner] {
            for pieces in [1, 4] {
                let mut c = EscCond::new(3000.0, 2.0, 1200.0, place);
                c.virus = true;
                c.pieces = pieces;
                let (mut a, sa) = build_escape_scene(&c, 5, &mut plain);
                let (mut b, sb) = build_escape_scene(&c, 5, &mut plain);
                let (d, _) = build_escape_scene(&c, 6, &mut plain);
                assert_eq!(snap(&a), snap(&b), "{place:?} p{pieces}");
                assert_ne!(snap(&a), snap(&d));
                assert_eq!(sa.start_gap, sb.start_gap);
                let (mut ca, mut cb) = (Chaser::new(sa.chaser), Chaser::new(sb.chaser));
                for i in 0..200 {
                    if i % 2 == 0 {
                        let (oa, ob) = (ca.decide(&a, sa.us), cb.decide(&b, sb.us));
                        apply(&mut a, sa.chaser, oa.aim, oa.split);
                        apply(&mut b, sb.chaser, ob.aim, ob.split);
                    }
                    a.tick();
                    b.tick();
                }
                assert_eq!(snap(&a), snap(&b), "{place:?} p{pieces}: evolution");
            }
        }
    }

    #[test]
    fn scene_geometry_matches_the_condition() {
        let c = EscCond::new(1000.0, 2.0, 1200.0, Place::Wall);
        let (w, s) = build_escape_scene(&c, 3, &mut plain);
        let (u, g) = (cells_of(&w, s.us)[0], cells_of(&w, s.chaser)[0]);
        assert!((u.3 - size_of(1000.0)).abs() < 1e-6 && (g.3 - size_of(2000.0)).abs() < 1e-6);
        let centre = (u.1 - g.1).hypot(u.2 - g.2);
        assert!((centre - (1200.0 + u.3 + g.3)).abs() < 1.0, "centre {centre}");
        let b = w.border;
        let wall = (u.1 - b.0).min(b.2 - u.1).min(u.2 - b.1).min(b.3 - u.2);
        assert!((wall - (WALL_START + u.3)).abs() < 1.0, "wall distance {wall}");
        let mut c2 = c.clone();
        c2.pieces = 4;
        let (w2, s2) = build_escape_scene(&c2, 3, &mut plain);
        assert_eq!(cells_of(&w2, s2.us).len(), 4);
        let m: f64 = w2.players[s2.us].total_mass(&w2);
        assert!((m - 1000.0).abs() < 60.0, "mass {m}");
    }

    #[test]
    fn split_reach_formula() {
        // a 10000-mass cell (size 1000): child 707, thrown 780 * 707^0.0122
        let r = split_reach(1000.0, 800.0);
        assert!((r - (split_throw(1000.0 / 2f64.sqrt()) + 1000.0 / 2f64.sqrt() - 800.0 / 3.0)).abs() < 1e-6);
        assert!(r > 1200.0 && r < 1400.0, "{r}");
    }

    #[test]
    fn chaser_splits_only_when_the_half_can_eat() {
        // 4x: the half (2x) eats a target at 600 centre distance; 1.4x: it never splits
        for (ratio, expect) in [(4.0, true), (1.4, false)] {
            let c = EscCond::new(1000.0, ratio, 150.0, Place::Open);
            let (w, s) = build_escape_scene(&c, 9, &mut plain);
            let mut ch = Chaser::new(s.chaser);
            let mut wm = w;
            let o = ch.decide(&wm, s.us);
            let o2 = ch.decide(&wm, s.us);
            assert_eq!(o.split || o2.split, expect, "ratio {ratio}");
            wm.tick();
        }
    }

    #[test]
    fn chaser_leads_and_herds() {
        let c = EscCond::new(1000.0, 2.0, 1500.0, Place::Open);
        let (mut w, s) = build_escape_scene(&c, 4, &mut plain);
        let mut ch = Chaser::new(s.chaser);
        let u0 = cells_of(&w, s.us)[0];
        // our cell runs along +x for a while; the chaser's aim must be ahead of us on that side after it has seen the velocity
        let mut last = ch.decide(&w, s.us);
        for _ in 0..10 {
            w.players[s.us].mouse_in = Some((u0.1 + 3000.0, u0.2));
            w.tick();
            w.tick();
            last = ch.decide(&w, s.us);
        }
        let u = cells_of(&w, s.us)[0];
        assert!(u.1 > u0.1 + 20.0);
        assert!(last.aim.0 > u.1, "aim {:?} should lead our cell {:?}", last.aim, (u.1, u.2));
    }

    #[test]
    fn rep_is_deterministic() {
        let agent = AgentDef { name: "oracle".into(), kind: AgentKind::Oracle { full: true, brave: false, nearest: false, no_split: false } };
        let mut c = EscCond::new(1000.0, 1.4, 1200.0, Place::Wall);
        c.horizon_s = 10.0;
        let (r1, r2) = (run_escape_rep(&agent, &c, 3), run_escape_rep(&agent, &c, 3));
        assert_eq!(r1.to_string(), r2.to_string());
    }
}
