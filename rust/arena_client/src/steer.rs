//! Per-tick steering by potential field (port of bot/steer.py).

use crate::farm::FoodRouter;
use crate::game::{Border, Cell, Kind, Snapshot};
use crate::params::Params;

pub const EAT_RATIO: f64 = 1.3;
pub const SPLIT_KILL_RATIO: f64 = 2.6;
const AIM_DISTANCE: f64 = 600.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Farm,
    Hunt,
    Flee,
    Hide,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Farm => "farm",
            Mode::Hunt => "hunt",
            Mode::Flee => "flee",
            Mode::Hide => "hide",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub mode: Mode,
    pub target_id: Option<u32>,
    pub split_ok: bool,
}

impl Plan {
    pub fn new(mode: Mode) -> Plan {
        Plan { mode, target_id: None, split_ok: false }
    }
}

pub struct Classified {
    pub food: Vec<Cell>,
    pub prey: Vec<Cell>,
    pub threats: Vec<Cell>,
    pub viruses: Vec<Cell>,
}

/// Split other cells into food / prey / threats / viruses relative to our main cell. With `p.contender`,
/// a near-equal player within `p.contender_gap` of us is a threat too.
pub fn classify(s: &Snapshot, p: Option<&Params>) -> Classified {
    let me = s.main().unwrap();
    let mm = me.mass();
    let mut r = Classified { food: vec![], prey: vec![], threats: vec![], viruses: vec![] };
    for c in &s.others {
        match c.kind {
            Kind::Food | Kind::Ejected => {
                if c.kind == Kind::Food || c.mass() * EAT_RATIO < mm {
                    r.food.push(*c);
                }
            }
            Kind::Virus => r.viruses.push(*c),
            _ => {
                if mm > c.mass() * EAT_RATIO {
                    r.prey.push(*c);
                } else if c.mass() > mm * EAT_RATIO {
                    r.threats.push(*c);
                } else if let Some(p) = p {
                    if p.contender > 0.0 && c.mass() > mm * p.contender
                        && me.dist(c) - c.size - me.size < p.contender_gap
                    {
                        r.threats.push(*c);
                    }
                }
            }
        }
    }
    r
}

/// One of the two cells `me` would split into (half the mass, same place).
pub fn half_of(me: &Cell) -> Cell {
    Cell { size: me.size / 2f64.sqrt(), ..*me }
}

/// Threats that count for a split decision: to `me`, or (half_split_safety) to the halves after it.
pub fn split_threats(s: &Snapshot, me: &Cell, p: &Params) -> Vec<Cell> {
    if p.half_split_safety <= 0.5 {
        return classify(s, None).threats;
    }
    let half = half_of(me);
    s.others
        .iter()
        .filter(|c| c.kind == Kind::Player && !s.mine.iter().any(|m| m.id == c.id) && c.mass() > half.mass() * EAT_RATIO)
        .copied()
        .collect()
}

pub fn danger_radius(threat: &Cell, me: &Cell, p: &Params) -> f64 {
    let mut reach = threat.size + me.size;
    if threat.mass() > me.mass() * SPLIT_KILL_RATIO {
        reach += p.split_range;
    }
    reach + p.danger_margin
}

pub const WARY_RANGE: f64 = 1500.0; // wall_wary: a bigger player within this distance widens the wall margin
pub const WALL_ZONE: f64 = 700.0; // wall_danger fades out this far from a wall

/// Extra danger gap when pressed on a wall (wall_danger): full at the wall, 0 at WALL_ZONE from it.
pub fn wall_extra(me: &Cell, b: &Border, p: &Params) -> f64 {
    if p.wall_danger <= 0.0 {
        return 0.0;
    }
    let wd = (me.x - b.left).min(b.right - me.x).min(me.y - b.top).min(b.bottom - me.y) - me.size;
    p.wall_danger * (1.0 - wd / WALL_ZONE).max(0.0).min(1.0)
}

/// danger_radius with the wall term (Python: danger_radius(..., border)); same float order as Python.
pub fn danger_radius_at(threat: &Cell, me: &Cell, p: &Params, b: &Border) -> f64 {
    let mut reach = threat.size + me.size;
    if threat.mass() > me.mass() * SPLIT_KILL_RATIO {
        reach += p.split_range;
    }
    reach += wall_extra(me, b, p);
    reach + p.danger_margin
}

const FLEE_HEADINGS: usize = 32;
const FLEE_CHECKS: [f64; 3] = [6.0, 12.0, 18.0];
const PIECE_SHARE: f64 = 0.15;

/// Distance a cell moves per server tick.
pub fn tick_speed(size: f64) -> f64 {
    88.0 * size.max(1.0).powf(-0.439)
}

/// Our cells the flee protects: the main cell, plus (flee_pieces) every piece >= PIECE_SHARE of our mass.
pub fn guarded_cells(s: &Snapshot, p: &Params) -> Vec<Cell> {
    let main = *s.main().unwrap();
    if p.flee_pieces <= 0.5 {
        return vec![main];
    }
    let total = {
        let t = s.total_mass();
        if t == 0.0 { 1.0 } else { t }
    };
    s.mine.iter().filter(|c| c.id == main.id || c.mass() >= PIECE_SHARE * total).copied().collect()
}

pub fn eaters_of(s: &Snapshot, q: &Cell) -> Vec<Cell> {
    s.others
        .iter()
        .filter(|c| c.kind == Kind::Player && !s.mine.iter().any(|m| m.id == c.id) && c.mass() > q.mass() * EAT_RATIO)
        .copied()
        .collect()
}

struct Guard {
    q: Cell,
    lo_x: f64,
    hi_x: f64,
    lo_y: f64,
    hi_y: f64,
    v_q: f64,
    th: Vec<(f64, f64, f64, f64)>,
}

/// Heading that keeps us out of every threat's reach longest over the next ~1.5 s.
pub fn flee_heading(s: &Snapshot, me: &Cell, threats: &[Cell], poppers: &[Cell], p: &Params,
                    pref: (f64, f64)) -> (f64, f64) {
    let b = &s.border;
    let mut guards: Vec<Guard> = Vec::new();
    for q in guarded_cells(s, p) {
        let qt: Vec<Cell> = if q.id == me.id { threats.to_vec() } else { eaters_of(s, &q) };
        if qt.is_empty() {
            continue;
        }
        let th = qt
            .iter()
            .map(|t| {
                (t.x, t.y, 2.0 * tick_speed(t.size),
                 t.size - q.size / 3.0 + if t.mass() > q.mass() * SPLIT_KILL_RATIO { p.split_range } else { 0.0 })
            })
            .collect();
        guards.push(Guard {
            q,
            lo_x: b.left + q.size / 2.0,
            hi_x: b.right - q.size / 2.0,
            lo_y: b.top + q.size / 2.0,
            hi_y: b.bottom - q.size / 2.0,
            v_q: 2.0 * tick_speed(q.size),
            th,
        });
    }
    let (mut best, mut best_score) = (pref, f64::NEG_INFINITY);
    for k in 0..FLEE_HEADINGS {
        let a = 2.0 * std::f64::consts::PI * k as f64 / FLEE_HEADINGS as f64;
        let (ux, uy) = (a.cos(), a.sin());
        let mut score = f64::INFINITY;
        for g in &guards {
            for &n in &FLEE_CHECKS {
                let x = (g.q.x + ux * g.v_q * n).max(g.lo_x).min(g.hi_x);
                let y = (g.q.y + uy * g.v_q * n).max(g.lo_y).min(g.hi_y);
                for &(tx, ty, v, reach) in &g.th {
                    score = score.min((x - tx).hypot(y - ty) - v * n - reach);
                }
                if g.q.id == me.id && poppers.iter().any(|v| (x - v.x).hypot(y - v.y) < v.size + me.size) {
                    score = f64::NEG_INFINITY;
                }
            }
        }
        score += 1e-3 * (ux * pref.0 + uy * pref.1);
        if score > best_score {
            best = (ux, uy);
            best_score = score;
        }
    }
    best
}

/// World aim point and whether to split this tick.
pub fn steer(s: &Snapshot, plan: &Plan, mut router: Option<&mut FoodRouter>, p: &Params) -> (f64, f64, bool) {
    let me = *s.main().unwrap();
    let Classified { food, prey, threats, viruses } = classify(s, Some(p));
    let (w_food, w_prey, w_threat) = match plan.mode {
        Mode::Farm => (1.0, p.farm_prey, p.farm_threat),
        Mode::Hunt => (p.hunt_food, 1.0, p.hunt_threat),
        Mode::Flee => (0.2, 0.0, p.flee_threat),
        Mode::Hide => (0.2, 0.0, p.flee_threat * 0.75),
    };
    let (mut fx, mut fy) = (0.0f64, 0.0f64);
    macro_rules! pull {
        ($cx:expr, $cy:expr, $weight:expr) => {{
            let (cx, cy, weight): (f64, f64, f64) = ($cx, $cy, $weight);
            let (dx, dy) = (cx - me.x, cy - me.y);
            let d = dx.hypot(dy).max(1.0);
            fx += weight * dx / d;
            fy += weight * dy / d;
        }};
    }

    match router.as_deref_mut() {
        None => {
            for c in &food {
                pull!(c.x, c.y, w_food * 40.0 / (me.dist(c) + 40.0));
            }
        }
        Some(r) => {
            let poppable: Vec<Cell> = viruses
                .iter()
                .filter(|v| me.mass() > v.mass() * EAT_RATIO && p.virus_detour > 0.5)
                .copied()
                .collect();
            let unsafe_cell = |c: &Cell| {
                threats.iter().any(|t| t.dist(c) < danger_radius_at(t, &me, p, &s.border))
                    || poppable.iter().any(|v| v.dist(c) < v.size + me.size)
            };
            let (wx, wy) = r.next_waypoint(s, &food, &unsafe_cell, p.switch_margin);
            pull!(wx, wy, w_food * p.food_pull);
        }
    }
    let ahead = |c: &Cell| -> Cell {
        if p.lead == 0.0 {
            return *c;
        }
        let k = p.lead * (me.dist(c) / 600.0).min(1.0);
        Cell { x: c.x + c.vx * k, y: c.y + c.vy * k, ..*c }
    };

    for c in &prey {
        let bonus = if Some(c.id) == plan.target_id { p.target_bonus } else { 1.0 };
        let a = ahead(c);
        pull!(a.x, a.y, w_prey * bonus * c.mass() / (me.dist(c) + 50.0));
    }
    for c in &threats {
        let d = me.dist(c) - c.size - me.size;
        let r = danger_radius_at(c, &me, p, &s.border);
        if d < r {
            pull!(c.x, c.y, -w_threat * p.threat_force * (r - d) / r);
        }
    }
    if p.piece_threats > 0.5 && s.mine.len() > 1 {
        let total = {
            let t: f64 = s.mine.iter().map(|q| q.mass()).sum();
            if t == 0.0 { 1.0 } else { t }
        };
        for q in &s.mine {
            if q.id == me.id {
                continue;
            }
            for c in &s.others {
                if c.kind != Kind::Player || s.mine.iter().any(|m| m.id == c.id) || c.mass() <= q.mass() * EAT_RATIO {
                    continue;
                }
                let d = q.dist(c) - c.size - q.size;
                let r = danger_radius_at(c, q, p, &s.border);
                if d < r {
                    let (dx, dy) = (q.x - c.x, q.y - c.y);
                    let n = dx.hypot(dy).max(1.0);
                    let w = w_threat * p.threat_force * (r - d) / r * q.mass() / total;
                    fx += w * dx / n;
                    fy += w * dy / n;
                }
            }
        }
    }
    let mut poppers: Vec<Cell> = Vec::new();
    for v in &viruses {
        if me.mass() > v.mass() * EAT_RATIO {
            if p.virus_detour > 0.5 {
                poppers.push(*v);
            } else {
                let d = me.dist(v) - v.size - me.size;
                if d < p.virus_avoid {
                    pull!(v.x, v.y, -20.0 * (p.virus_avoid - d) / p.virus_avoid);
                }
            }
        } else if plan.mode == Mode::Hide {
            pull!(v.x, v.y, 8.0);
        }
    }

    let b = &s.border;
    let mut margin = me.size + p.border_margin;
    if p.wall_wary > 0.0
        && s.others.iter().any(|c| c.kind == Kind::Player && c.mass() > me.mass() * p.wall_wary_ratio && me.dist(c) < WARY_RANGE)
    {
        margin += p.wall_wary; // a big player is about: keep clear of walls we could be pinned on
    }
    let target = if plan.mode == Mode::Hunt { prey.iter().find(|c| Some(c.id) == plan.target_id) } else { None };
    let t_edges = match target {
        Some(t) => [t.x - b.left, b.right - t.x, t.y - b.top, b.bottom - t.y],
        None => [f64::INFINITY; 4],
    };
    let edges = [me.x - b.left, b.right - me.x, me.y - b.top, b.bottom - me.y];
    let pushes = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)];
    for i in 0..4 {
        if edges[i] < margin && t_edges[i] >= edges[i] {
            fx += pushes[i].0 * 10.0 * (margin - edges[i]) / margin;
            fy += pushes[i].1 * 10.0 * (margin - edges[i]) / margin;
        }
    }

    let mut norm = fx.hypot(fy);
    if norm < 1e-6 {
        fx = 1.0;
        fy = 0.0;
        norm = 1.0;
    }
    let (mut ux, mut uy) = (fx / norm, fy / norm);
    let planned = plan.mode == Mode::Flee && p.flee_plan > 0.5 && (!threats.is_empty() || p.flee_pieces > 0.5);
    if planned {
        let h = flee_heading(s, &me, &threats, &poppers, p, (ux, uy));
        ux = h.0;
        uy = h.1;
    }
    let mut blocking: Vec<(f64, Cell, f64, f64)> = Vec::new();
    if !planned {
        for v in &poppers {
            let (vx, vy) = (v.x - me.x, v.y - me.y);
            let t = vx * ux + vy * uy;
            let clear = v.size + me.size + p.virus_avoid * 0.2;
            let h = vx * uy - vy * ux;
            if 0.0 < t && t < clear + p.virus_avoid + 200.0 && h.abs() < clear {
                blocking.push((t, *v, h, clear));
            }
        }
    }
    if !blocking.is_empty() {
        let mut bi = 0;
        for i in 1..blocking.len() {
            if blocking[i].0 < blocking[bi].0 {
                bi = i;
            }
        }
        let (_, v, h, clear) = blocking[bi];
        let side = if h > 0.0 { 1.0 } else { -1.0 };
        let (px, py) = (-uy * side, ux * side);
        let (tx, ty) = (v.x + px * clear * 1.15, v.y + py * clear * 1.15);
        let (dx, dy) = (tx - me.x, ty - me.y);
        let dn = {
            let d = dx.hypot(dy);
            if d == 0.0 { 1.0 } else { d }
        };
        ux = dx / dn;
        uy = dy / dn;
    }
    let ahead_of_main = if p.aim_all_cells > 0.5 {
        s.mine.iter().map(|q| ((q.x - me.x) * ux + (q.y - me.y) * uy).max(0.0)).fold(0.0, f64::max)
    } else {
        0.0
    };
    let reach = AIM_DISTANCE + ahead_of_main;
    let mut aim = (me.x + ux * reach, me.y + uy * reach);

    let mut split = false;
    if plan.split_ok && plan.mode == Mode::Hunt && (s.mine.len() as f64) <= p.split_max_cells.round_ties_even() {
        let target = prey.iter().find(|c| Some(c.id) == plan.target_id).map(|c| ahead(c));
        let side = if p.half_split_safety > 0.5 { half_of(&me) } else { me };
        if let Some(t) = target {
            if me.mass() / 2.0 > t.mass() * EAT_RATIO
                && me.dist(&t) - me.size < p.split_range * p.split_reach
                && !split_threats(s, &me, p).iter().any(|th| me.dist(th) < danger_radius(th, &side, p) + p.split_range)
            {
                aim = (t.x, t.y);
                split = true;
            }
        }
    }
    (aim.0, aim.1, split)
}
