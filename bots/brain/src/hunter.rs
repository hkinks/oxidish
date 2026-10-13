//! "Lurk and harvest": shadow a giant that stands near a virus, and when it bursts go for the pieces.
//!
//! Four behaviours, each behind a gene (all off by default; see `Params`):
//!
//! 1. **lurk** (`lurk`): a player of at least `lurk_ratio` x our mass that is within `lurk_virus_range` of a virus is shadowed from a standoff
//!    just outside its real split reach, on the far side of the virus (the pieces then fly towards us). The lurk gives way to the ordinary
//!    bot logic (returns `None`) when the giant turns on us or another threat appears. `lurk_shoot` feeds the virus into the giant with the
//!    existing virus-shoot tactic (done in `policy.rs`, which owns it).
//! 2. **commit** (`commit`): once a group of >= 3 cells of one owner with `hv_group_ratio` x our mass has edible pieces (a burst giant), go for
//!    the pieces: the best safe edible one first (`commit_order` 0: mass / time-to-reach, 1: smallest first), then grow into the next.
//! 3. **split-harvest** (`split_harvest`): split onto a piece our half still eats when the lunge lands it and nothing that could eat a half
//!    (the regrouping core included, at its real split reach) can get to either half.
//! 4. **race** (`race`): skip pieces a rival that can eat them reaches before us.
//!
//! `Hunter::step` is a pure policy step over a `Snapshot` (it keeps only velocities, cooldowns and counters), so a vision agent can call it as a
//! teacher or a reflex: `if let Some(out) = hunter.step(&snap, &params) { ... out.x, out.y, out.split ... }`.
//! Times are in decision steps (2 ticks).

use std::collections::HashMap;

use crate::game::{Cell, Kind, Snapshot};
use crate::params::Params;
use crate::policy::Out;
use crate::steer::{tick_speed, EAT_RATIO};

/// MultiOgar `playerSplitSpeed`: a split cell is thrown `SPLIT_SPEED * size^0.0122` (the boost decays by a tenth per tick, so that is the total travel).
pub const SPLIT_SPEED: f64 = 780.0;
/// Server ticks per decision step.
pub const STEP_TICKS: f64 = 2.0;
/// mass ratio at which we count a piece as edible (the server needs size x1.15, i.e. 1.3225 in mass)
pub const EDIBLE: f64 = 1.34;
/// Server ticks before a split cell can merge back: `max(30 s, 0.2 * size s) * 25`.
pub fn merge_ticks(size: f64) -> f64 {
    (30.0f64).max(size * 0.2) * 25.0
}

/// How far the cell a split throws travels (the new cell has size `parent / sqrt 2`).
pub fn split_travel(parent_size: f64) -> f64 {
    SPLIT_SPEED * (parent_size / std::f64::consts::SQRT_2).powf(0.0122)
}

/// Centre-to-centre distance within which `t` can eat something by splitting at it (the thrown half covers the victim's centre).
pub fn lunge_reach(t: &Cell) -> f64 {
    split_travel(t.size) + t.size / std::f64::consts::SQRT_2
}

/// Can a half of `t` still eat `victim` (so that `t` may lunge)?
pub fn can_lunge(t: &Cell, victim: &Cell) -> bool {
    t.mass() / 2.0 > victim.mass() * EAT_RATIO
}

/// Centre-to-centre distance within which `t` threatens `victim` over the next ~15 steps: walking up to it, or (when a half can eat) a split lunge.
pub fn threat_reach(t: &Cell, victim: &Cell, p: &Params) -> f64 {
    // a cell that is not closing in on the victim needs a smaller margin than one that walks at it
    let d = t.dist(victim).max(1.0);
    let closing = (t.vx * (victim.x - t.x) + t.vy * (victim.y - t.y)) / d;
    let margin = if closing > 0.25 * STEP_TICKS * tick_speed(t.size) { p.hv_margin } else { p.hv_still };
    let plain = t.size + margin;
    if can_lunge(t, victim) {
        plain.max(lunge_reach(t) * p.hv_lunge + margin * 0.5)
    } else {
        plain
    }
}

/// Where a thrown cell (a burst piece, a split half) ends up: it moves `v` per decision step now and the boost decays a tenth per tick, so it still
/// travels about `v * 0.81 / (1 - 0.81)` more. `None` for a cell that is just walking.
pub fn landing_of(t: &Cell) -> Option<Cell> {
    let v = t.vx.hypot(t.vy);
    if v < 2.2 * STEP_TICKS * tick_speed(t.size) {
        return None;
    }
    let k = 0.81 / (1.0 - 0.81);
    Some(Cell { x: t.x + t.vx * k, y: t.y + t.vy * k, vx: 0.0, vy: 0.0, ..*t })
}

/// Where a lurker stands: on the line giant -> virus, beyond the virus, `standoff` away from the giant (but clear of the virus).
pub fn lurk_spot(giant: (f64, f64), virus: &Cell, me_size: f64, standoff: f64) -> (f64, f64) {
    let (dx, dy) = (virus.x - giant.0, virus.y - giant.1);
    let d = dx.hypot(dy).max(1.0);
    let (ux, uy) = (dx / d, dy / d);
    let off = (standoff - d).max(virus.size + me_size + 60.0);
    (virus.x + ux * off, virus.y + uy * off)
}

/// Distance from `p` to the segment `a`-`b`.
pub fn seg_dist(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let l2 = dx * dx + dy * dy;
    let t = if l2 < 1e-9 { 0.0 } else { (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / l2).clamp(0.0, 1.0) };
    (p.0 - (a.0 + t * dx)).hypot(p.1 - (a.1 + t * dy))
}

/// The one player a set of pieces belongs to (protocol colour; 0 = unknown counts as its own owner).
fn owner(c: &Cell) -> u64 {
    if c.color != 0 {
        c.color as u64
    } else {
        (1u64 << 32) | c.id as u64
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    Idle,
    Lurk,
    Commit,
}

#[derive(Default, Clone, Debug)]
pub struct HunterStats {
    pub lurk_steps: u64,
    pub commit_steps: u64,
    pub splits: u64,
    pub aborts: u64,
    pub flees: u64,
}

pub struct Hunter {
    prev: HashMap<u32, (f64, f64)>,
    split_cd: u32,
    cooldown: u32,
    turned: u32,
    lurk_giant: Option<u32>,
    heading: Option<(f64, f64)>,
    memory: HashMap<u32, (Cell, u32)>,
    pub phase: Phase,
    /// the giant id and the virus the last lurk step used (for the shoot tactic)
    pub lurk_target: Option<u32>,
    pub stats: HunterStats,
}

impl Default for Hunter {
    fn default() -> Hunter {
        Hunter::new()
    }
}

impl Hunter {
    pub fn new() -> Hunter {
        Hunter { prev: HashMap::new(), split_cd: 0, cooldown: 0, turned: 0, lurk_giant: None, heading: None, memory: HashMap::new(), phase: Phase::Idle, lurk_target: None, stats: HunterStats::default() }
    }

    pub fn reset(&mut self) {
        *self = Hunter::new();
    }

    /// Is any hunter gene on?
    pub fn enabled(p: &Params) -> bool {
        p.lurk > 0.5 || p.commit > 0.5 || p.split_harvest > 0.5
    }

    /// One decision. `None` = nothing to do here, let the ordinary bot decide.
    pub fn step(&mut self, s: &Snapshot, p: &Params) -> Option<Out> {
        // velocities per decision step from the positions of the previous call
        let mut players: Vec<Cell> = Vec::new();
        let mut viruses: Vec<Cell> = Vec::new();
        let mut next: HashMap<u32, (f64, f64)> = HashMap::new();
        for c in &s.others {
            if c.kind != Kind::Player && c.kind != Kind::Virus {
                continue;
            }
            let mut c = *c;
            match self.prev.get(&c.id) {
                Some(&(px, py)) => {
                    c.vx = c.x - px;
                    c.vy = c.y - py;
                }
                None => {
                    c.vx = 0.0;
                    c.vy = 0.0;
                }
            }
            next.insert(c.id, (c.x, c.y));
            if c.kind == Kind::Player {
                players.push(c);
            } else {
                viruses.push(c);
            }
        }
        self.prev = next;
        self.split_cd = self.split_cd.saturating_sub(1);
        self.cooldown = self.cooldown.saturating_sub(1);
        self.phase = Phase::Idle;
        self.lurk_target = None;
        let me = *s.main()?;
        if p.hv_memory > 0.5 {
            self.remember(s, p, &me, &mut players);
        }
        if p.commit > 0.5 || p.split_harvest > 0.5 {
            if let Some(o) = self.commit(s, p, &me, &players, &viruses) {
                self.phase = Phase::Commit;
                self.stats.commit_steps += 1;
                return Some(o);
            }
        }
        if p.lurk > 0.5 && self.cooldown == 0 {
            if let Some(o) = self.lurk(s, p, &me, &players, &viruses) {
                self.phase = Phase::Lurk;
                self.stats.lurk_steps += 1;
                return Some(o);
            }
        }
        None
    }

    /// Memory of a burst giant's pieces: cells of a group of >= 3 (with `hv_group_ratio` x our mass) that drop out of view are kept for up to
    /// `hv_memory` x 10 steps at their extrapolated place (thrown ones on their boost path, the rest where they were) and added to `players`,
    /// unless they would be inside our view box (then they are gone: eaten or merged).
    fn remember(&mut self, s: &Snapshot, p: &Params, me: &Cell, players: &mut Vec<Cell>) {
        let total = s.total_mass();
        let mut groups: HashMap<u64, (usize, f64)> = HashMap::new();
        for c in players.iter() {
            let e = groups.entry(owner(c)).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += c.mass();
        }
        let seen: std::collections::HashSet<u32> = players.iter().map(|c| c.id).collect();
        for c in players.iter() {
            let (n, m) = groups[&owner(c)];
            if n as f64 >= p.hv_min_pieces && m >= p.hv_group_ratio * total {
                self.memory.insert(c.id, (*c, 0));
            }
        }
        let (hw, hh) = (7.0 * me.size * 0.9, 4.0 * me.size * 0.9);
        let max_age = (p.hv_memory * 10.0) as u32;
        let mut drop = Vec::new();
        for (id, (c, age)) in self.memory.iter_mut() {
            if seen.contains(id) {
                continue;
            }
            *age += 1;
            c.x += c.vx;
            c.y += c.vy;
            c.vx *= 0.81;
            c.vy *= 0.81;
            let inside = (c.x - me.x).abs() < hw && (c.y - me.y).abs() < hh;
            if *age > max_age || inside {
                drop.push(*id);
            } else {
                players.push(*c);
            }
        }
        for id in drop {
            self.memory.remove(&id);
        }
    }

    // ------------------------------------------------------------------------------------------ lurk

    fn lurk(&mut self, s: &Snapshot, p: &Params, me: &Cell, players: &[Cell], viruses: &[Cell]) -> Option<Out> {
        let total = s.total_mass();
        // the giant: the biggest single (unbroken) player at least lurk_ratio x our mass, near a virus, within range
        let mut best: Option<(&Cell, &Cell, f64)> = None;
        for g in players.iter().filter(|g| g.mass() >= p.lurk_ratio * total && me.dist(g) < p.lurk_range) {
            if players.iter().filter(|c| c.id != g.id && c.color != 0 && c.color == g.color).count() >= 2 {
                continue; // already in pieces: the commit logic's business
            }
            for v in viruses.iter().filter(|v| g.mass() > v.mass() * 1.33) {
                let d = g.dist(v);
                if d < p.lurk_virus_range && best.map_or(true, |b| d < b.2) {
                    best = Some((g, v, d));
                }
            }
        }
        let (g, v, _) = match best {
            Some(b) => b,
            None => {
                self.lurk_giant = None;
                self.turned = 0;
                return None;
            }
        };
        let gap_of = |c: &Cell| me.dist(c) - c.size - me.size;
        // abort: another threat (not the giant) is near
        for t in players.iter().filter(|t| t.id != g.id && t.mass() > me.mass() * EAT_RATIO) {
            if me.dist(t) < threat_reach(t, me, p) + 150.0 {
                return self.abort();
            }
        }
        // abort: the giant turns on us (moving at us, not at the virus, close)
        let standoff = threat_reach(g, me, p) + p.lurk_margin;
        let (ux, uy) = ((me.x - g.x) / me.dist(g).max(1.0), (me.y - g.y) / me.dist(g).max(1.0));
        let closing = g.vx * ux + g.vy * uy;
        let gspeed = STEP_TICKS * tick_speed(g.size);
        let to_v = ((v.x - g.x) * ux + (v.y - g.y) * uy) / g.dist(v).max(1.0); // cos(angle giant -> virus vs giant -> us)
        if self.lurk_giant == Some(g.id) && closing > 0.5 * gspeed && to_v < 0.7 && me.dist(g) < standoff + 500.0 {
            self.turned += 1;
        } else {
            self.turned = 0;
        }
        self.lurk_giant = Some(g.id);
        if self.turned >= 2 || gap_of(g) < 0.0 {
            return self.abort();
        }
        self.lurk_target = Some(g.id);
        let spot = lurk_spot((g.x, g.y), v, me.size, standoff);
        // never nearer to the giant than the standoff: if it is, back straight away from it
        let aim = if me.dist(g) < standoff - 40.0 {
            (me.x + ux * 500.0, me.y + uy * 500.0)
        } else if (spot.0 - me.x).hypot(spot.1 - me.y) < 60.0 {
            (me.x, me.y)
        } else {
            spot
        };
        let aim = avoid_viruses(me, aim, viruses);
        Some(Out { x: aim.0, y: aim.1, split: false, eject: false })
    }

    fn abort(&mut self) -> Option<Out> {
        self.stats.aborts += 1;
        self.cooldown = 50;
        self.turned = 0;
        self.lurk_giant = None;
        None
    }

    // ------------------------------------------------------------------------------------------ commit

    fn commit(&mut self, s: &Snapshot, p: &Params, me: &Cell, players: &[Cell], viruses: &[Cell]) -> Option<Out> {
        let total = s.total_mass();
        // the burst giant: the biggest group of >= 3 pieces with hv_group_ratio x our mass and something we can eat
        let mut groups: HashMap<u64, (usize, f64)> = HashMap::new();
        for c in players {
            let e = groups.entry(owner(c)).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += c.mass();
        }
        let key = groups
            .iter()
            .filter(|(k, (n, m))| *n as f64 >= p.hv_min_pieces && *m >= p.hv_group_ratio * total && players.iter().any(|c| owner(c) == **k && me.mass() > c.mass() * EDIBLE))
            .max_by(|a, b| (a.1 .1).total_cmp(&b.1 .1))
            .map(|(k, _)| *k)?;
        if p.commit_kite > 0.5 {
            return self.kite(s, p, me, players, viruses, key);
        }
        let me_mass = me.mass();
        // everything that can eat us
        let threats: Vec<&Cell> = players.iter().filter(|t| t.mass() > me_mass * EAT_RATIO).collect();
        let guarded: Vec<&Cell> = s.mine.iter().filter(|q| q.id == me.id || q.mass() >= 0.15 * total).collect();
        let landings: Vec<Cell> = threats.iter().filter_map(|t| landing_of(t)).collect();
        let lands_on = |q: &Cell| landings.iter().any(|l| l.mass() > q.mass() * EAT_RATIO && q.dist(l) < l.size + p.hv_margin);
        let danger = |q: &Cell| -> bool { lands_on(q) || players.iter().any(|t| t.mass() > q.mass() * EAT_RATIO && q.dist(t) < threat_reach(t, q, p)) };
        if guarded.iter().any(|q| danger(q)) {
            if trace_on() {
                let who: Vec<String> = players.iter().filter(|t| t.mass() > me_mass * EAT_RATIO && me.dist(t) < threat_reach(t, me, p) + 300.0).map(|t| format!("{:.0}@{:.0}/r{:.0}", t.mass(), me.dist(t), threat_reach(t, me, p))).collect();
                eprintln!("  hunter flee me {:.0} ({:.0},{:.0}) from {who:?}", me_mass, me.x, me.y);
            }
            self.stats.flees += 1;
            let a = flee_point(s, &guarded, players, p);
            return Some(Out { x: a.0, y: a.1, split: false, eject: false });
        }
        // candidates: the group's edible pieces; safe if no threat can get at the walk, or at us once we have eaten it
        let my_speed = STEP_TICKS * tick_speed(me.size);
        let mut best: Option<(&Cell, f64)> = None;
        for c in players.iter().filter(|c| owner(c) == key && me_mass > c.mass() * EDIBLE) {
            let d = me.dist(c);
            let after = Cell { size: ((me_mass + c.mass()) * 100.0).sqrt(), x: c.x, y: c.y, ..*me };
            let step = d.min(200.0);
            let (ax, ay) = (me.x + (c.x - me.x) * step / d.max(1.0), me.y + (c.y - me.y) * step / d.max(1.0));
            let mid = Cell { x: ax, y: ay, ..after };
            let safe = !lands_on(&after) && !lands_on(&mid) && threats.iter().all(|t| {
                t.id == c.id
                    || (t.dist(&after) > threat_reach(t, &after, p)
                        && t.dist(&mid) > threat_reach(t, &mid, p)
                        && seg_dist((t.x, t.y), (me.x, me.y), (c.x, c.y)) > t.size + 60.0)
            });
            if !safe {
                continue;
            }
            let t_us = ((d - me.size).max(0.0) / my_speed).max(1.0);
            if p.race > 0.5 {
                // a rival (another owner) that eats it reaches it first
                let beaten = players.iter().any(|r| {
                    owner(r) != key && r.mass() > c.mass() * EDIBLE && ((r.dist(c) - r.size).max(0.0) / (STEP_TICKS * tick_speed(r.size))) * 1.1 + 3.0 < t_us
                });
                if beaten {
                    continue;
                }
            }
            let score = if p.commit_order > 0.5 { -c.mass() } else { c.mass() / (t_us + 15.0) };
            if best.map_or(true, |b| score > b.1) {
                best = Some((c, score));
            }
        }
        if trace_on() {
            eprintln!("  hunter me {:.0} ({:.0},{:.0}) target {:?}", me_mass, me.x, me.y, best.map(|(c, _)| (c.mass() as i64, me.dist(c) as i64)));
        }
        let c = match best {
            Some((c, _)) => *c,
            None => {
                // nothing safe to eat yet: hold (the pieces come to us), or regroup our halves
                if s.mine.len() > 1 {
                    let (cx, cy) = centroid(&s.mine);
                    return Some(Out { x: cx, y: cy, split: false, eject: false });
                }
                return Some(Out { x: me.x, y: me.y, split: false, eject: false });
            }
        };
        if p.commit <= 0.5 {
            // split-harvest alone: only act when a split is on
            return self.maybe_split(s, p, me, &c, players, &threats, key);
        }
        let k = (me.dist(&c) / 600.0).min(1.0) * 6.0;
        let aim = avoid_viruses(me, (c.x + c.vx * k, c.y + c.vy * k), viruses);
        let mut out = Out { x: aim.0, y: aim.1, split: false, eject: false };
        if p.split_harvest > 0.5 {
            if let Some(sp) = self.maybe_split(s, p, me, &c, players, &threats, key) {
                out = sp;
            }
        }
        Some(out)
    }


    /// The burst giant's pieces as a short race: try 32 headings, play out the next `hv_horizon` steps for each (our cell at full speed; every
    /// cell that can eat us pursuing the spot we are heading for at its own speed, thrown pieces on their boost path, lunge circles kept out of),
    /// and take the safe heading that eats the most piece mass soonest, or at least ends nearest to it. Running from a slow core past the faster
    /// pieces that chase us is what this finds.
    fn kite(&mut self, s: &Snapshot, p: &Params, me: &Cell, players: &[Cell], viruses: &[Cell], key: u64) -> Option<Out> {
        const K: usize = 32;
        const R: f64 = 0.81; // per-step decay of a thrown cell's travel (0.9 per tick)
        let me_mass = me.mass();
        let h = (p.hv_horizon as usize).clamp(4, 80);
        let v_us = STEP_TICKS * tick_speed(me.size);
        let threats: Vec<&Cell> = players.iter().filter(|t| t.mass() > me_mass * EAT_RATIO).collect();
        let edibles: Vec<&Cell> = players.iter().filter(|c| owner(c) == key && me_mass > c.mass() * EDIBLE).collect();
        if edibles.is_empty() {
            return None;
        }
        let ballistic = |c: &Cell, k: usize| -> (f64, f64) {
            let f = R * (1.0 - R.powi(k as i32)) / (1.0 - R);
            (c.x + c.vx * f, c.y + c.vy * f)
        };
        let thrown: Vec<bool> = threats.iter().map(|t| landing_of(t).is_some()).collect();
        let e_thrown: Vec<bool> = edibles.iter().map(|c| landing_of(c).is_some()).collect();
        let e_pos = |i: usize, k: usize| -> (f64, f64) {
            let c = edibles[i];
            if e_thrown[i] {
                ballistic(c, k)
            } else {
                let f = k.min(3) as f64;
                (c.x + c.vx * f, c.y + c.vy * f)
            }
        };
        let b = s.border;
        let (mut best_score, mut best_dir, mut best_first) = (f64::NEG_INFINITY, (0.0, 0.0), None::<usize>);
        let prev = self.heading;
        let nearest_e = *edibles.iter().min_by(|a, b| me.dist(a).total_cmp(&me.dist(b))).unwrap();
        for kk in 0..=K {
            // the extra last pass (trace only) heads straight for the nearest piece and says what stops it
            if kk == K && !trace_on() {
                break;
            }
            let dbg = kk == K;
            let dir = if dbg {
                let d = me.dist(nearest_e).max(1.0);
                ((nearest_e.x - me.x) / d, (nearest_e.y - me.y) / d)
            } else {
                let a = kk as f64 * std::f64::consts::TAU / K as f64;
                (a.cos(), a.sin())
            };
            // threat tracks, pursuing our path
            let mut tpos: Vec<(f64, f64)> = threats.iter().map(|t| (t.x, t.y)).collect();
            let mut violation = 0.0f64;
            let mut value = 0.0;
            let mut eaten = vec![false; edibles.len()];
            let mut first: Option<(usize, usize)> = None;
            let mut pk = (me.x, me.y);
            for k in 1..=h {
                pk = (me.x + dir.0 * v_us * k as f64, me.y + dir.1 * v_us * k as f64);
                // walls and viruses
                let out = (b.left + me.size * 0.5 - pk.0).max(pk.0 - (b.right - me.size * 0.5)).max(b.top + me.size * 0.5 - pk.1).max(pk.1 - (b.bottom - me.size * 0.5)).max(0.0);
                violation = violation.max(out / 100.0);
                for v in viruses.iter().filter(|v| me_mass > v.mass() * 1.33) {
                    let d = (v.x - pk.0).hypot(v.y - pk.1);
                    let need = v.size + me.size * 0.5;
                    if d < need {
                        violation = violation.max((need - d) / 100.0 + 0.5);
                    }
                }
                for (i, t) in threats.iter().enumerate() {
                    let sp = STEP_TICKS * tick_speed(t.size);
                    let tp = &mut tpos[i];
                    if thrown[i] && k <= 10 {
                        *tp = ballistic(t, k);
                    } else {
                        let (dx, dy) = (pk.0 - tp.0, pk.1 - tp.1);
                        let d = dx.hypot(dy).max(1.0);
                        let st = sp.min(d);
                        *tp = (tp.0 + dx / d * st, tp.1 + dy / d * st);
                    }
                    let d = (pk.0 - tp.0).hypot(pk.1 - tp.1);
                    let reach = t.size + p.hv_still;
                    if d < reach {
                        violation = violation.max((reach - d) / 100.0 + 0.5);
                        if dbg {
                            eprintln!("    greedy blocked k={k} by pursuit of {:.0}-mass at {:.0} (now {:.0})", t.mass(), d, me.dist(t));
                        }
                    }
                    if k <= 8 && can_lunge(t, me) {
                        let lr = lunge_reach(t) * p.hv_lunge;
                        let d0 = (pk.0 - t.x).hypot(pk.1 - t.y);
                        let d00 = me.dist(t);
                        let need = lr.min(d00) - 5.0;
                        if d0 < need {
                            violation = violation.max((need - d0) / 100.0 + 0.5);
                            if dbg {
                                eprintln!("    greedy blocked k={k} by lunge circle of {:.0}-mass (need {:.0} have {:.0})", t.mass(), need, d0);
                            }
                        }
                    }
                }
                for i in 0..edibles.len() {
                    if eaten[i] {
                        continue;
                    }
                    let ep = e_pos(i, k);
                    if (ep.0 - pk.0).hypot(ep.1 - pk.1) < me.size * 0.9 {
                        eaten[i] = true;
                        value += edibles[i].mass() / (k as f64 + 8.0);
                        if first.is_none() {
                            first = Some((i, k));
                        }
                    }
                }
            }
            // pieces not reached yet: the nearer the end of the path, the better
            for i in 0..edibles.len() {
                if !eaten[i] {
                    let ep = e_pos(i, h);
                    let rem = ((ep.0 - pk.0).hypot(ep.1 - pk.1) - me.size).max(0.0) / v_us;
                    value += 0.5 * edibles[i].mass() / (h as f64 + rem + 8.0);
                }
            }
            if p.race > 0.5 {
                // a piece a rival (another owner) reaches first is not ours
                for i in 0..edibles.len() {
                    let c = edibles[i];
                    let rival = players.iter().any(|r| owner(r) != key && r.mass() > c.mass() * EDIBLE && r.dist(c) - r.size < 0.9 * (me.dist(c) - me.size) * (tick_speed(r.size) / tick_speed(me.size)));
                    if rival && eaten[i] {
                        value -= c.mass() / (h as f64 + 8.0);
                    }
                }
            }
            let stick = if prev.map_or(false, |pv| (pv.0 * dir.0 + pv.1 * dir.1) > 0.95) { 1.05 } else { 1.0 };
            let score = if violation > 0.0 { -1.0e6 * (1.0 + violation) } else { value * stick };
            if dbg {
                eprintln!("    greedy toward {:.0}-mass at {:.0}: violation {violation:.2} value {value:.2}", nearest_e.mass(), me.dist(&nearest_e));
                continue;
            }
            if score > best_score {
                best_score = score;
                best_dir = dir;
                best_first = first.map(|f| f.0);
            }
        }
        if trace_on() {
            eprintln!("  hunter kite me {:.0} ({:.0},{:.0}) dir ({:.2},{:.2}) score {:.2} first {:?}", me_mass, me.x, me.y, best_dir.0, best_dir.1, best_score, best_first.map(|i| (edibles[i].mass() as i64, me.dist(edibles[i]) as i64)));
        }
        self.heading = Some(best_dir);
        if best_score < 0.0 {
            self.stats.flees += 1;
        }
        let aim = avoid_viruses(me, (me.x + best_dir.0 * 700.0, me.y + best_dir.1 * 700.0), viruses);
        let mut out = Out { x: aim.0, y: aim.1, split: false, eject: false };
        if let (true, Some(i)) = (p.split_harvest > 0.5 && best_score > 0.0, best_first) {
            let thr: Vec<&Cell> = Vec::new();
            if let Some(sp) = self.maybe_split(s, p, me, edibles[i], players, &thr, key) {
                out = sp;
            }
        }
        Some(out)
    }

    /// Split onto `c` when our half still eats it, the lunge lands it, and no cell that could eat a half gets to either half.
    fn maybe_split(&mut self, s: &Snapshot, p: &Params, me: &Cell, c: &Cell, players: &[Cell], _threats: &[&Cell], _key: u64) -> Option<Out> {
        if self.split_cd > 0 || s.mine.len() != 1 || p.split_harvest <= 0.5 {
            return None;
        }
        if c.mass() < p.hv_split_frac * s.total_mass() {
            return None; // not worth two vulnerable halves
        }
        let half_mass = me.mass() / 2.0;
        if half_mass <= c.mass() * EDIBLE * 1.05 {
            return None;
        }
        let travel = split_travel(me.size);
        let d = me.dist(c);
        // lands when the piece is within the throw, and is worth a throw (not at our feet)
        let k = (d / 600.0).min(1.0) * 6.0;
        let (px, py) = (c.x + c.vx * k, c.y + c.vy * k);
        let dl = (px - me.x).hypot(py - me.y);
        if dl < me.size + 40.0 || dl > travel * 0.9 {
            return None;
        }
        let half = Cell { size: me.size / std::f64::consts::SQRT_2, ..*me };
        let landing = Cell { x: px, y: py, size: (half_mass + c.mass()).max(half_mass) * 100.0, ..half };
        let landing = Cell { size: landing.size.sqrt(), ..landing };
        for t in players.iter().filter(|t| t.id != c.id) {
            for q in [&half, &landing] {
                if t.mass() > q.mass() * EAT_RATIO && (t.dist(q) < threat_reach(t, q, p) || seg_dist((t.x, t.y), (me.x, me.y), (px, py)) < t.size + 60.0) {
                    return None;
                }
            }
        }
        self.split_cd = 12;
        self.stats.splits += 1;
        Some(Out { x: px, y: py, split: true, eject: false })
    }
}

fn trace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("OXIDISH_HUNTER_TRACE").is_ok())
}

fn centroid(cells: &[Cell]) -> (f64, f64) {
    let t: f64 = cells.iter().map(|c| c.mass()).sum();
    (cells.iter().map(|c| c.x * c.mass()).sum::<f64>() / t, cells.iter().map(|c| c.y * c.mass()).sum::<f64>() / t)
}

/// Bend an aim point round viruses we would pop on (a virus on the walk, within its radius + our radius + a margin).
pub fn avoid_viruses(me: &Cell, aim: (f64, f64), viruses: &[Cell]) -> (f64, f64) {
    let mut a = aim;
    for v in viruses.iter().filter(|v| me.mass() > v.mass() * 1.33) {
        let r = v.size + me.size * 0.9 + 40.0;
        let sd = seg_dist((v.x, v.y), (me.x, me.y), a);
        if sd < r {
            // push the aim sideways, away from the virus
            let (dx, dy) = (a.0 - me.x, a.1 - me.y);
            let dl = dx.hypot(dy).max(1.0);
            let (nx, ny) = (-dy / dl, dx / dl);
            let side = if (v.x - me.x) * nx + (v.y - me.y) * ny > 0.0 { -1.0 } else { 1.0 };
            a = (a.0 + nx * side * (r - sd + 40.0) * 2.0, a.1 + ny * side * (r - sd + 40.0) * 2.0);
        }
    }
    a
}

/// The clearest of 16 headings: a point 450 away that keeps most clearance from every cell that could eat one of ours; walls count as threats.
pub fn flee_point(s: &Snapshot, guarded: &[&Cell], players: &[Cell], p: &Params) -> (f64, f64) {
    let total: f64 = guarded.iter().map(|c| c.mass()).sum();
    let (cx, cy) = (guarded.iter().map(|c| c.x * c.mass()).sum::<f64>() / total, guarded.iter().map(|c| c.y * c.mass()).sum::<f64>() / total);
    let q = *guarded.iter().max_by(|a, b| a.size.total_cmp(&b.size)).unwrap();
    let b = s.border;
    let mut best = (f64::MIN, (cx, cy));
    for k in 0..16 {
        let a = k as f64 * std::f64::consts::TAU / 16.0;
        let pt = (cx + a.cos() * 450.0, cy + a.sin() * 450.0);
        let wall = (pt.0 - b.left).min(b.right - pt.0).min(pt.1 - b.top).min(b.bottom - pt.1);
        let mut sc = players
            .iter()
            .filter(|t| t.mass() > q.mass() * EAT_RATIO)
            .map(|t| (t.x - pt.0).hypot(t.y - pt.1) - threat_reach(t, &q, p))
            .fold(f64::MAX, f64::min);
        if wall < 250.0 {
            sc -= (250.0 - wall) * 4.0;
        }
        if sc > best.0 {
            best = (sc, pt);
        }
    }
    best.1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Border;

    fn mass_size(m: f64) -> f64 {
        (m * 100.0).sqrt()
    }

    fn player(id: u32, x: f64, y: f64, mass: f64, color: u32) -> Cell {
        let mut c = Cell::new(id, x, y, mass_size(mass), Kind::Player);
        c.color = color;
        c
    }

    fn virus(id: u32, x: f64, y: f64) -> Cell {
        Cell::new(id, x, y, mass_size(120.0), Kind::Virus)
    }

    fn snap(me_mass: f64, others: Vec<Cell>) -> Snapshot {
        let mut me = Cell::new(1, 0.0, 0.0, mass_size(me_mass), Kind::Mine);
        me.color = 0x111111;
        Snapshot { mine: vec![me], others, border: Border { left: -4000.0, right: 4000.0, top: -4000.0, bottom: 4000.0 }, dead: false }
    }

    fn params() -> Params {
        Params { lurk: 1.0, commit: 1.0, split_harvest: 1.0, race: 1.0, hv_split_frac: 0.15, ..Params::default() }
    }

    #[test]
    fn split_reach_formula_matches_the_server() {
        // a 500-mass cell throws its half 780 * (223/sqrt2)^0.0122 ~ 830
        let c = player(2, 0.0, 0.0, 500.0, 5);
        assert!((split_travel(c.size) - 833.0).abs() < 8.0, "{}", split_travel(c.size));
        // 5k giant: half is 500 mass, size 500/... lunge reach = travel + half radius
        let g = player(3, 0.0, 0.0, 5000.0, 6);
        assert!((lunge_reach(&g) - (split_travel(g.size) + g.size / 2f64.sqrt())).abs() < 1e-9);
        assert!(lunge_reach(&g) > 1300.0);
    }

    #[test]
    fn threat_reach_has_a_lunge_only_when_a_half_eats() {
        let p = Params::default();
        let me = player(1, 0.0, 0.0, 500.0, 1);
        let big = player(2, 0.0, 0.0, 5000.0, 2);
        let mid = player(3, 0.0, 0.0, 800.0, 3); // half 400 < 1.3 x 500: no lunge
        assert!(threat_reach(&big, &me, &p) > 1300.0);
        assert!((threat_reach(&mid, &me, &p) - (mid.size + p.hv_still)).abs() < 1e-9);
        // walking at us it needs the full margin
        let mut walker = mid;
        walker.x = 1000.0;
        walker.vx = -8.0;
        assert!((threat_reach(&walker, &me, &p) - (walker.size + p.hv_margin)).abs() < 1e-9);
    }

    #[test]
    fn lurk_spot_is_beyond_the_virus_outside_the_split_reach() {
        let p = params();
        let g = player(10, 1000.0, 0.0, 5000.0, 7);
        let v = virus(11, 300.0, 0.0);
        let me = player(1, -600.0, 0.0, 500.0, 1);
        let standoff = threat_reach(&g, &me, &p) + p.lurk_margin;
        let spot = lurk_spot((g.x, g.y), &v, me.size, standoff);
        let d = (spot.0 - g.x).hypot(spot.1 - g.y);
        assert!(d >= standoff - 1e-6, "spot at {d} inside the standoff {standoff}");
        assert!(spot.0 < v.x - 1.0 && spot.1.abs() < 1e-6, "not behind the virus: {spot:?}");
    }

    #[test]
    fn lurk_moves_to_the_far_side_of_the_virus() {
        let p = params();
        // we stand at the origin, a 5k giant 1000 to the right of a virus at (500, 300); the giant is closer than the standoff
        let g = player(10, 1100.0, 300.0, 5000.0, 7);
        let v = virus(11, 500.0, 300.0);
        let s = snap(500.0, vec![g, v]);
        let mut h = Hunter::new();
        let o = h.step(&s, &p).expect("lurk");
        assert_eq!(h.phase, Phase::Lurk);
        // too close to the giant: back away, i.e. aim to the left of us
        assert!(o.x < -100.0 && !o.split, "{o:?}");
        assert_eq!(h.lurk_target, Some(10));
    }

    #[test]
    fn lurk_ignores_giants_far_from_any_virus_and_small_ones() {
        let p = params();
        let g = player(10, 1500.0, 300.0, 5000.0, 7);
        let v = virus(11, -3000.0, 300.0);
        assert!(Hunter::new().step(&snap(500.0, vec![g, v]), &p).is_none());
        let small = player(10, 1500.0, 300.0, 1500.0, 7); // only 3x
        let v = virus(11, 800.0, 300.0);
        assert!(Hunter::new().step(&snap(500.0, vec![small, v]), &p).is_none());
    }

    #[test]
    fn lurk_aborts_when_the_giant_turns_on_us() {
        let p = params();
        let v = virus(11, -3500.0, 0.0);
        let mut h = Hunter::new();
        // the giant is near a virus, at 2000 from us and then runs straight at us
        let near_v = |gx: f64| {
            let g = player(10, gx, 0.0, 5000.0, 7);
            snap(500.0, vec![g, Cell { x: gx, y: 900.0, ..v }])
        };
        let mut aborted = false;
        for i in 0..6 {
            let gx = 1900.0 - 25.0 * i as f64; // 25 per step ~ its top speed, straight at us
            let out = h.step(&near_v(gx), &p);
            if out.is_none() && i >= 2 {
                aborted = true;
            }
        }
        assert!(aborted, "never gave way");
        assert!(h.stats.aborts >= 1);
    }

    #[test]
    fn lurk_aborts_when_another_threat_appears() {
        let p = params();
        let g = player(10, 1500.0, 300.0, 5000.0, 7);
        let v = virus(11, 500.0, 300.0);
        let other = player(12, -400.0, 0.0, 1500.0, 8); // a 1500 right beside us
        assert!(Hunter::new().step(&snap(500.0, vec![g, v, other]), &p).is_none());
    }

    /// 12 x 28, 318, 637, 1275, 2521 as the burst of a 5k giant, centred at (1100, 0)
    fn burst(extra_x: f64) -> Vec<Cell> {
        let mut o = Vec::new();
        for i in 0..12 {
            let a = i as f64 * 0.52;
            o.push(player(20 + i, 1100.0 + extra_x + 300.0 * a.cos(), 300.0 * a.sin(), 28.0, 9));
        }
        o.push(player(40, 1100.0 + extra_x + 600.0, 500.0, 318.0, 9));
        o.push(player(41, 1100.0 + extra_x + 700.0, -500.0, 637.0, 9));
        o.push(player(42, 1100.0 + extra_x + 1200.0, 0.0, 1275.0, 9));
        o.push(player(43, 1100.0 + extra_x + 2500.0, 0.0, 2521.0, 9));
        o
    }

    #[test]
    fn commit_goes_for_a_safe_small_piece() {
        let p = params();
        let s = snap(500.0, burst(0.0));
        let mut h = Hunter::new();
        let o = h.step(&s, &p).expect("commit");
        assert_eq!(h.phase, Phase::Commit);
        assert!(o.x > 0.0 && !o.split, "{o:?}");
    }

    #[test]
    fn commit_order_smallest_first_vs_value() {
        // a 28 piece far away vs a 318 piece close: value order takes the 318, smallest-first the 28
        let others = vec![
            player(20, 0.0, 700.0, 28.0, 9),
            player(21, 0.0, 701.0, 28.0, 9),
            player(40, 600.0, 0.0, 318.0, 9),
            player(43, 6000.0, 6000.0, 4000.0, 9),
        ];
        let s = snap(500.0, others);
        let mut p = params();
        p.split_harvest = 0.0;
        let o = Hunter::new().step(&s, &p).unwrap();
        assert!(o.x > 100.0 && o.y.abs() < o.x, "value order should head for the 318: {o:?}");
        p.commit_order = 1.0;
        let o = Hunter::new().step(&s, &p).unwrap();
        assert!(o.y > 100.0 && o.x.abs() < o.y, "smallest first should head for the 28s: {o:?}");
    }

    #[test]
    fn commit_flees_a_core_that_can_lunge_at_us() {
        let p = params();
        // the 2521 core 900 away: within its lunge (~1200) and a half (1260) eats a 500: leave along the clear side
        let mut o = burst(0.0);
        o.retain(|c| c.id < 43 || c.id == 43);
        let core = o.iter_mut().find(|c| c.id == 43).unwrap();
        core.x = 900.0;
        core.y = 0.0;
        let s = snap(500.0, o);
        let out = Hunter::new().step(&s, &p).expect("flee");
        assert!(out.x < -100.0 && !out.split, "{out:?}");
    }

    #[test]
    fn commit_skips_a_piece_a_rival_reaches_first() {
        let mut p = params();
        p.split_harvest = 0.0;
        // two 28s at 400: one with a 900 rival right on it (rival hunts it), one clear
        let others = vec![
            player(20, 400.0, 0.0, 28.0, 9),
            player(21, 0.0, 400.0, 28.0, 9),
            player(22, 0.0, 410.0, 28.0, 9),
            player(43, 6000.0, 6000.0, 4000.0, 9),
            player(50, 520.0, 60.0, 900.0, 4),
        ];
        // the rival is a threat (900 > 1.3 x 500) and 130 from piece 20: piece 20 is unsafe/lost either way
        let s = snap(500.0, others);
        let o = Hunter::new().step(&s, &p).unwrap();
        assert!(o.y > 100.0, "went for the contested piece: {o:?}");
    }

    #[test]
    fn split_harvest_splits_onto_a_medium_piece_when_clear() {
        let p = params();
        // 1000 mass: half 500 > 1.3 x 318 x 1.05. The 318 piece 600 away, no threats anywhere near.
        let others = vec![
            player(40, 600.0, 0.0, 318.0, 9),
            player(41, 650.0, 200.0, 40.0, 9),
            player(42, 650.0, -200.0, 40.0, 9),
            player(43, 7000.0, 7000.0, 3000.0, 9),
        ];
        let s = snap(1000.0, others);
        let mut h = Hunter::new();
        let o = h.step(&s, &p).expect("commit");
        assert!(o.split, "{o:?}");
        assert_eq!(h.stats.splits, 1);
        // the cooldown stops an immediate second one
        let o2 = h.step(&s, &p).unwrap();
        assert!(!o2.split);
    }

    #[test]
    fn split_harvest_refuses_when_the_core_could_eat_a_half() {
        let p = params();
        // the 2521 core 1500 from us, 1000 from the piece: it can lunge at the landing half (half of 2521 eats a 500)
        let others = vec![
            player(40, 600.0, 0.0, 318.0, 9),
            player(41, 650.0, 200.0, 40.0, 9),
            player(42, 650.0, -200.0, 40.0, 9),
            player(43, 1500.0, 0.0, 2521.0, 9),
        ];
        let s = snap(1000.0, others);
        let o = Hunter::new().step(&s, &p);
        assert!(o.map_or(true, |o| !o.split), "split into the core's reach");
    }

    #[test]
    fn split_harvest_refuses_when_the_half_cannot_eat_the_piece() {
        let p = params();
        // 700 mass: half 350 < 1.3 x 318
        let others = vec![player(40, 600.0, 0.0, 318.0, 9), player(41, 650.0, 200.0, 40.0, 9), player(42, 650.0, -200.0, 40.0, 9), player(43, 7000.0, 7000.0, 3000.0, 9)];
        let o = Hunter::new().step(&snap(700.0, others), &p).unwrap();
        assert!(!o.split);
    }

    #[test]
    fn a_thrown_piece_is_dodged_by_its_landing_spot() {
        let p = params();
        // a 1275 piece thrown at us: 400 away, moving 60 per step straight at us (landing ~ 250 beyond us)
        let mut thrown = player(42, 400.0, 0.0, 1275.0, 9);
        thrown.vx = -60.0;
        let mut o = burst(0.0);
        o.retain(|c| c.id != 42 && c.id != 43);
        o.push(thrown);
        let landing = landing_of(&thrown).unwrap();
        assert!(landing.x < 200.0 && landing.dist(&Cell::new(1, 0.0, 0.0, 223.0, Kind::Mine)) < landing.size, "{landing:?}");
        // a walking cell has none
        let mut walk = player(5, 0.0, 0.0, 1275.0, 9);
        walk.vx = 8.0;
        assert!(landing_of(&walk).is_none());
        // we leave: the aim is behind us (the piece itself is already inside its 477 reach)
        let out = Hunter::new().step(&snap(500.0, o), &p).unwrap();
        assert!(!out.split && out.x < 0.0, "{out:?}");
    }

    fn kite_params() -> Params {
        Params { commit_kite: 1.0, hv_memory: 3.0, hv_still: 100.0, ..params() }
    }

    #[test]
    fn kite_heads_for_a_small_piece_away_from_a_slow_core() {
        let p = kite_params();
        let others = vec![
            player(20, 700.0, 0.0, 28.0, 9),
            player(21, 720.0, 60.0, 28.0, 9),
            player(22, 730.0, -60.0, 28.0, 9),
            player(43, -1500.0, 0.0, 2521.0, 9),
        ];
        let out = Hunter::new().step(&snap(500.0, others), &p).expect("kite");
        assert!(out.x > 100.0 && !out.split, "{out:?}");
    }

    #[test]
    fn kite_keeps_out_of_the_cores_lunge_circle() {
        let p = kite_params();
        // pieces lie beyond the 2521 core, which is 1000 away (inside its ~1190 lunge circle? no: we are inside it): never walk closer
        let others = vec![
            player(20, 1400.0, 0.0, 28.0, 9),
            player(21, 1420.0, 60.0, 28.0, 9),
            player(22, 1430.0, -60.0, 28.0, 9),
            player(43, 1000.0, 0.0, 2521.0, 9),
        ];
        let out = Hunter::new().step(&snap(500.0, others), &p).expect("kite");
        assert!(out.x <= 1.0, "walked into the lunge circle: {out:?}");
    }

    #[test]
    fn memory_keeps_pieces_that_left_the_view() {
        let p = kite_params();
        let mut h = Hunter::new();
        let seen = snap(500.0, vec![player(20, 1500.0, 0.0, 28.0, 9), player(21, 1520.0, 40.0, 28.0, 9), player(22, 1540.0, -40.0, 28.0, 9), player(43, 3500.0, 0.0, 2521.0, 9)]);
        assert!(h.step(&seen, &p).is_some());
        // next step the group is out of view (beyond the 7 x size half-width): the hunter still heads for it
        let gone = snap(500.0, vec![]);
        let o = h.step(&gone, &p).expect("remembered");
        assert!(o.x > 0.0);
        // and forgets it after hv_memory x 10 steps
        for _ in 0..40 {
            h.step(&gone, &p);
        }
        assert!(h.step(&gone, &p).is_none());
        // without the memory gene it is dropped at once
        let mut h2 = Hunter::new();
        h2.step(&seen, &Params { hv_memory: 0.0, ..p.clone() });
        assert!(h2.step(&gone, &Params { hv_memory: 0.0, ..p }).is_none());
    }

    #[test]
    fn disabled_genes_do_nothing() {
        let s = snap(500.0, burst(0.0));
        assert!(Hunter::new().step(&s, &Params::default()).is_none());
        assert!(!Hunter::enabled(&Params::default()));
    }

    #[test]
    fn merge_timer_is_at_least_thirty_seconds() {
        assert_eq!(merge_ticks(100.0), 750.0);
        assert!(merge_ticks(300.0) > 1400.0);
    }

    #[test]
    fn avoid_virus_bends_the_aim() {
        let me = player(1, 0.0, 0.0, 500.0, 1);
        let v = virus(5, 500.0, 0.0);
        let a = avoid_viruses(&me, (1000.0, 0.0), &[v]);
        assert!(a.1.abs() > 50.0, "{a:?}");
    }
}
