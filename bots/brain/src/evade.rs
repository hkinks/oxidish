//! Escape heading by look-ahead (genes `evade*`, all off by default; docs/escape-drill.md).
//!
//! The rule bots flee with a potential field: straight away from the chaser until the wall, where the field and the
//! wall push cancel and a chaser that cuts the corner catches them. This module replaces the aim of a flee step with
//! the first move of the best of a small search tree: 4 moves (1 + 1 + 2 + 3 s) over `HEADINGS` headings (4096 paths, 7 s).
//! While we hold a heading the chasers follow a lead pursuit (they aim at where we will be) at their own speed;
//! walls clamp our cells (so a heading into a wall becomes a slide along it, and a corner a dead end). A path is scored by its
//! minimum margin to every chaser's lethal reach over the whole horizon, so staying out of the split reach (and not
//! being pinned) is the primary objective, not just "away".
//!
//! Lethal reach (centre distance): the eating distance `size - our size / 3`; for a chaser that can split-kill (mass above
//! `SPLIT_KILL_RATIO` times ours, so the half still eats) also the real split throw, 780 * child size^0.0122 plus the child's
//! radius.
//!
//! Genes: `evade` (1 = on in Flee mode), `evade_trigger` (> 0: also flee when a chaser is within its lethal reach plus this),
//! `evade_buffer` (added to the lethal reach), `evade_min_mass` (the search only runs at or above this total mass), `evade_cap` (margins above this count the same, default 800), `evade_wall` (weight of how central the path ends: 1 = on), `evade_food` (0.3 = a first move towards a lot of pellets is worth up to 300 margin units when we are safe enough), `evade_virus` (1 = chasers
//! that could pop on a virus steer round it, so a virus on the line between us works as a shield; our own cells never aim at a
//! virus they would pop), `evade_pieces` (0 = in several pieces the old flee runs; 1 = search with every big piece guarded; 2 = also aim so that the pieces close up).

use crate::game::{Cell, Kind, Snapshot};
use crate::params::Params;
use crate::steer::{classify, tick_speed, EAT_RATIO, SPLIT_KILL_RATIO};

const HEADINGS: usize = 8;
const DEPTH: usize = 4;
/// ticks of the moves of the search tree (1 + 1 + 2 + 3 s: later moves are coarser), and per sub-step inside a move
const SEGS: [usize; DEPTH] = [25, 25, 50, 75];
const SUB: usize = 5;
const AIM_DISTANCE: f64 = 600.0;
const MAX_GUARDS: usize = 4;
const MAX_CHASERS: usize = 4;
const POP_RATIO: f64 = 1.33;
/// margins above this are all "safe enough": past it the free room decides (a margin below it is paid for 1:1)
const MARGIN_CAP: f64 = 800.0;
/// score per unit of centrality (see `room`)
const ROOM_W: f64 = 0.5;
/// evade_food: pellets within this of us count for the first move
const FOOD_RANGE: f64 = 900.0;

/// Distance a split child is thrown (the server's boost, travelled in full).
pub fn split_throw(child_size: f64) -> f64 {
    780.0 * child_size.powf(0.0122)
}

/// Centre distance up to which chaser `t` eats `q`: plain eating, or a split-kill if the half still eats.
pub fn lethal_reach(t: &Cell, q: &Cell, p: &Params) -> f64 {
    let eat = t.size - q.size / 3.0;
    let mut r = eat;
    if t.mass() > q.mass() * SPLIT_KILL_RATIO {
        let child = t.size / std::f64::consts::SQRT_2;
        r = r.max(split_throw(child) + child - q.size / 3.0);
    }
    r + p.evade_buffer
}

#[derive(Clone, Copy)]
struct St {
    ours: [(f64, f64); MAX_GUARDS],
    ch: [(f64, f64); MAX_CHASERS],
}

struct Ctx {
    ng: usize,
    nc: usize,
    /// per guard: distance per sub-step, half size for the wall clamp, size
    g_step: [f64; MAX_GUARDS],
    g_half: [f64; MAX_GUARDS],
    g_size: [f64; MAX_GUARDS],
    /// guard 0's distance per tick (the chasers lead on it)
    g0_tick: f64,
    /// per chaser: distance per tick, size, mass
    c_tick: [f64; MAX_CHASERS],
    c_size: [f64; MAX_CHASERS],
    c_mass: [f64; MAX_CHASERS],
    /// lethal reach of chaser j against guard i (NaN: it cannot eat that guard)
    reach: [[f64; MAX_GUARDS]; MAX_CHASERS],
    viruses: Vec<Cell>,
    poppers: Vec<Cell>,
    virus_model: bool,
    bounds: (f64, f64, f64, f64),
    cap: f64,
    wall_w: f64,
    headings: [(f64, f64); HEADINGS],
    best: (f64, usize),
}

impl Ctx {
    /// Advance one sub-step of `SUB` ticks holding heading `h`; returns the margin after it (neg inf = a pop).
    fn step(&self, st: &mut St, h: (f64, f64)) -> f64 {
        let b = self.bounds;
        let mut margin = f64::INFINITY;
        for i in 0..self.ng {
            let (x, y) = st.ours[i];
            let r = self.g_half[i];
            let v = self.g_step[i];
            let nx = (x + h.0 * v).clamp(b.0 + r, b.2 - r);
            let ny = (y + h.1 * v).clamp(b.1 + r, b.3 - r);
            st.ours[i] = (nx, ny);
            for pv in &self.poppers {
                if (nx - pv.x).hypot(ny - pv.y) < pv.size + self.g_size[i] * 0.9 {
                    margin = f64::NEG_INFINITY;
                }
            }
        }
        let (gx, gy) = st.ours[0];
        for j in 0..self.nc {
            let (cx, cy) = st.ch[j];
            let sp = self.c_tick[j];
            let d = (gx - cx).hypot(gy - cy);
            let lead = (d / sp).min(40.0) * self.g0_tick;
            let mut aim = ((gx + h.0 * lead).clamp(b.0, b.2), (gy + h.1 * lead).clamp(b.1, b.3));
            if self.virus_model {
                for v in &self.viruses {
                    if self.c_mass[j] < POP_RATIO * v.mass() {
                        continue;
                    }
                    let clear = v.size + self.c_size[j] * 0.9;
                    let (ax, ay) = (aim.0 - cx, aim.1 - cy);
                    let l = ax.hypot(ay).max(1.0);
                    let (ux, uy) = (ax / l, ay / l);
                    let (rx, ry) = (v.x - cx, v.y - cy);
                    let along = rx * ux + ry * uy;
                    let across = rx * uy - ry * ux;
                    if along > 0.0 && along < l + clear && across.abs() < clear {
                        let side = if across > 0.0 { 1.0 } else { -1.0 };
                        aim = (v.x - uy * side * clear * 1.15, v.y + ux * side * clear * 1.15);
                        break;
                    }
                }
            }
            let (dx, dy) = (aim.0 - cx, aim.1 - cy);
            let dn = dx.hypot(dy).max(1e-9);
            let mv = (sp * SUB as f64).min(dn);
            let (nx, ny) = (cx + dx / dn * mv, cy + dy / dn * mv);
            st.ch[j] = (nx, ny);
            for i in 0..self.ng {
                let rc = self.reach[j][i];
                if rc.is_nan() {
                    continue;
                }
                let d = (nx - st.ours[i].0).hypot(ny - st.ours[i].1);
                margin = margin.min(d - rc);
            }
        }
        margin
    }

    /// How central the end of a path is (minus the distance to the middle of the map): the walls and above all the corners are the dead ends.
    fn room(&self, st: &St) -> f64 {
        let b = self.bounds;
        let (x, y) = st.ours[0];
        -((x - (b.0 + b.2) / 2.0).hypot(y - (b.1 + b.3) / 2.0))
    }

    /// DFS over the tree below the first move `first`; updates `best` with the best leaf.
    fn search(&mut self, st: &St, depth: usize, first: usize, min_margin: f64, pref: f64) {
        if depth == DEPTH {
            let mut end_margin = f64::INFINITY;
            for j in 0..self.nc {
                let rc = self.reach[j][0];
                if !rc.is_nan() {
                    end_margin = end_margin.min((st.ch[j].0 - st.ours[0].0).hypot(st.ch[j].1 - st.ours[0].1) - rc);
                }
            }
            let score = min_margin.min(self.cap) + 0.3 * end_margin.min(self.cap) + self.wall_w * ROOM_W * self.room(st) + 1e-3 * pref;
            if score > self.best.0 {
                self.best = (score, first);
            }
            return;
        }
        for k in 0..HEADINGS {
            let h = self.headings[k];
            let mut s2 = *st;
            let mut m = min_margin;
            for _ in 0..SEGS[depth] / SUB {
                m = m.min(self.step(&mut s2, h));
            }
            if m == f64::NEG_INFINITY {
                continue; // a pop: not worth following
            }
            self.search(&s2, depth + 1, first, m, pref);
        }
    }
}

/// The aim point of a flee step, or None when there is nothing to flee from. `pref` breaks ties (the potential field's heading).
pub fn evade_aim(s: &Snapshot, p: &Params, pref: (f64, f64)) -> Option<(f64, f64)> {
    let me = *s.main()?;
    if s.total_mass() < p.evade_min_mass {
        return None; // small bots keep the old flee (the search made them die more in endless, docs/escape-drill.md)
    }
    if s.mine.len() > 1 && p.evade_pieces < 0.5 {
        return None; // in pieces the rule bots' own flee runs (the search measured worse there, docs/escape-drill.md)
    }
    let cl = classify(s, Some(p));
    if cl.threats.is_empty() {
        return None;
    }
    let mut thr = cl.threats.clone();
    thr.sort_by(|a, b| me.dist(a).total_cmp(&me.dist(b)));
    thr.truncate(MAX_CHASERS);
    let total = s.total_mass().max(1.0);
    let mut guards: Vec<Cell> = vec![me];
    if p.evade_pieces > 0.5 {
        let mut rest: Vec<Cell> = s.mine.iter().filter(|c| c.id != me.id && c.mass() >= 0.15 * total).copied().collect();
        rest.sort_by(|a, b| b.mass().total_cmp(&a.mass()));
        guards.extend(rest.into_iter().take(MAX_GUARDS - 1));
    }
    let mut headings = [(0.0, 0.0); HEADINGS];
    for (k, h) in headings.iter_mut().enumerate() {
        let a = 2.0 * std::f64::consts::PI * k as f64 / HEADINGS as f64;
        *h = (a.cos(), a.sin());
    }
    let mut ctx = Ctx {
        ng: guards.len(),
        nc: thr.len(),
        g_step: [0.0; MAX_GUARDS],
        g_half: [0.0; MAX_GUARDS],
        g_size: [0.0; MAX_GUARDS],
        g0_tick: tick_speed(me.size),
        c_tick: [0.0; MAX_CHASERS],
        c_size: [0.0; MAX_CHASERS],
        c_mass: [0.0; MAX_CHASERS],
        reach: [[f64::NAN; MAX_GUARDS]; MAX_CHASERS],
        viruses: cl.viruses.clone(),
        poppers: cl.viruses.iter().filter(|v| me.mass() > v.mass() * EAT_RATIO).copied().collect(),
        virus_model: p.evade_virus > 0.5,
        bounds: (s.border.left, s.border.top, s.border.right, s.border.bottom),
        cap: if p.evade_cap > 0.0 { p.evade_cap } else { MARGIN_CAP },
        wall_w: p.evade_wall,
        headings,
        best: (f64::NEG_INFINITY, 0),
    };
    let mut st = St { ours: [(0.0, 0.0); MAX_GUARDS], ch: [(0.0, 0.0); MAX_CHASERS] };
    for (i, g) in guards.iter().enumerate() {
        st.ours[i] = (g.x, g.y);
        ctx.g_step[i] = tick_speed(g.size) * SUB as f64;
        ctx.g_half[i] = g.size / 2.0;
        ctx.g_size[i] = g.size;
    }
    for (j, t) in thr.iter().enumerate() {
        st.ch[j] = (t.x, t.y);
        ctx.c_tick[j] = tick_speed(t.size);
        ctx.c_size[j] = t.size;
        ctx.c_mass[j] = t.mass();
        for (i, g) in guards.iter().enumerate() {
            if t.mass() > g.mass() * EAT_RATIO {
                ctx.reach[j][i] = lethal_reach(t, g, p);
            }
        }
    }
    let mut best = (f64::NEG_INFINITY, 0usize);
    for k in 0..HEADINGS {
        let h = headings[k];
        let mut s2 = st;
        let mut m = f64::INFINITY;
        for _ in 0..SEGS[0] / SUB {
            m = m.min(ctx.step(&mut s2, h));
        }
        ctx.best = (f64::NEG_INFINITY, k);
        if m != f64::NEG_INFINITY {
            // tie-break: the potential field's heading, plus (evade_food) the pellets lying along this first move
            let mut fb = 0.0;
            if p.evade_food > 0.0 {
                for f in cl.food.iter().filter(|f| f.kind == Kind::Food) {
                    let (dx, dy) = (f.x - me.x, f.y - me.y);
                    let d = dx.hypot(dy).max(1.0);
                    if d < FOOD_RANGE {
                        let cos = (dx * h.0 + dy * h.1) / d;
                        fb += (cos - 0.8).max(0.0) / 0.2 * (1.0 - d / FOOD_RANGE);
                    }
                }
            }
            let bonus = h.0 * pref.0 + h.1 * pref.1 + p.evade_food * 1000.0 * (fb / 10.0).min(1.0);
            ctx.search(&s2, 1, k, m, bonus);
        }
        if ctx.best.0 > best.0 {
            best = (ctx.best.0, k);
        }
    }
    let h = headings[best.1];
    let mut aim = (me.x + h.0 * AIM_DISTANCE, me.y + h.1 * AIM_DISTANCE);
    if p.evade_pieces > 1.5 && s.mine.len() > 1 {
        // close up: aim a little ahead of the pieces' centroid, so that the ones behind catch up
        let (cx, cy) = (s.mine.iter().map(|c| c.x * c.mass()).sum::<f64>() / total, s.mine.iter().map(|c| c.y * c.mass()).sum::<f64>() / total);
        let spread = s.mine.iter().map(|c| (c.x - cx).hypot(c.y - cy)).fold(0.0, f64::max);
        let l = 150.0 + 0.5 * spread;
        aim = (cx + h.0 * l, cy + h.1 * l);
    }
    Some(aim)
}

/// True when some chaser is within its lethal reach plus `p.evade_trigger` (the gene that makes us flee earlier than the
/// rule bots' danger radius would).
pub fn triggered(s: &Snapshot, p: &Params) -> bool {
    if p.evade <= 0.5 || p.evade_trigger <= 0.0 {
        return false;
    }
    let me = match s.main() {
        Some(m) => *m,
        None => return false,
    };
    classify(s, Some(p)).threats.iter().any(|t| me.dist(t) < lethal_reach(t, &me, p) + p.evade_trigger)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Border;

    fn snap(me: Cell, others: Vec<Cell>) -> Snapshot {
        Snapshot { mine: vec![me], others, border: Border { left: -3500.0, right: 3500.0, top: -3500.0, bottom: 3500.0 }, dead: false }
    }

    fn cell(id: u32, x: f64, y: f64, mass: f64, kind: Kind) -> Cell {
        Cell::new(id, x, y, (mass * 100.0).sqrt(), kind)
    }

    fn on() -> Params {
        Params { evade: 1.0, evade_wall: 1.0, ..Params::default() }
    }

    /// In the open the escape runs straight away from the chaser.
    #[test]
    fn open_field_runs_away() {
        let s = snap(cell(1, 0.0, 0.0, 3000.0, Kind::Mine), vec![cell(2, -1500.0, 0.0, 6000.0, Kind::Player)]);
        let (ax, ay) = evade_aim(&s, &on(), (1.0, 0.0)).unwrap();
        assert!(ax > 300.0 && ay.abs() < 300.0, "aim ({ax}, {ay})");
    }

    /// Pressed on the right wall with the chaser to the west and below: the open way is along the wall (north), not into it.
    #[test]
    fn slides_along_the_wall_instead_of_into_the_corner() {
        let me = cell(1, 3500.0 - 548.0 - 100.0, 1200.0, 3000.0, Kind::Mine);
        let s = snap(me, vec![cell(2, me.x - 1500.0, 1200.0 + 700.0, 6000.0, Kind::Player)]);
        let (ax, ay) = evade_aim(&s, &on(), (1.0, 1.0)).unwrap();
        // moves north (negative y: away from the chaser at +700) and does not push into the wall
        assert!(ay < me.y - 150.0, "aim ({ax}, {ay}) from ({}, {})", me.x, me.y);
    }

    /// Near a corner with the chaser on the open side: running "away" (into the corner) is a dead end, the search must not take it.
    #[test]
    fn avoids_the_dead_end_of_a_corner() {
        let me = cell(1, 3500.0 - 700.0, 3500.0 - 700.0, 3000.0, Kind::Mine);
        let s = snap(me, vec![cell(2, me.x - 1100.0, me.y - 700.0, 6000.0, Kind::Player)]);
        let (ax, ay) = evade_aim(&s, &on(), (1.0, 0.64)).unwrap();
        let into_corner = ax > me.x + 250.0 && ay > me.y + 250.0;
        assert!(!into_corner, "aim ({ax}, {ay}) runs into the corner at ({}, {})", me.x, me.y);
    }

    #[test]
    fn nothing_to_flee_from_gives_none() {
        let s = snap(cell(1, 0.0, 0.0, 3000.0, Kind::Mine), vec![cell(2, 500.0, 0.0, 300.0, Kind::Player)]);
        assert!(evade_aim(&s, &on(), (1.0, 0.0)).is_none());
    }

    #[test]
    fn lethal_reach_adds_the_split_throw_only_when_the_half_still_eats() {
        let p = Params::default();
        let q = cell(1, 0.0, 0.0, 1000.0, Kind::Mine);
        let t14 = cell(2, 0.0, 0.0, 1400.0, Kind::Player);
        let t4 = cell(3, 0.0, 0.0, 4000.0, Kind::Player);
        assert!((lethal_reach(&t14, &q, &p) - (t14.size - q.size / 3.0)).abs() < 1e-9);
        assert!(lethal_reach(&t4, &q, &p) > t4.size + 400.0);
        assert!(lethal_reach(&t4, &q, &Params { evade_buffer: 100.0, ..p.clone() }) > lethal_reach(&t4, &q, &p) + 99.0);
    }

    #[test]
    fn pieces_guard_aims_at_closing_up() {
        let a = cell(1, 0.0, 0.0, 1500.0, Kind::Mine);
        let b = cell(4, 0.0, 700.0, 1500.0, Kind::Mine);
        let s = Snapshot { mine: vec![a, b], others: vec![cell(2, -1700.0, 350.0, 6000.0, Kind::Player)],
                           border: Border { left: -3500.0, right: 3500.0, top: -3500.0, bottom: 3500.0 }, dead: false };
        let p = Params { evade_pieces: 2.0, ..on() };
        let (ax, ay) = evade_aim(&s, &p, (1.0, 0.0)).unwrap();
        // aim near the centroid (0, 350) pushed east by under 600, not 600 from the main cell
        assert!(ax > 0.0 && ax < 500.0 && (ay - 350.0).abs() < 400.0, "aim ({ax}, {ay})");
    }
}
