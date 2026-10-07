//! Native port of Apostolique's AposBot (MIT, github.com/Apostolique/Agar.io-bot), decision-identical to
//! bot/apos.py (read its docstring for the adaptation notes). Keep the two in step: every expression
//! keeps the Python / JS operation order so the floats match, and `arena_client parity` checks it.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::game::{Border, Cell, Kind, Snapshot};
use crate::policy::{Counters, Out};

// ----------------------------------------------------------------- the rules
const EAT_RATIO: f64 = 1.25;
const PREY_RATIO: f64 = 1.4;
const THREAT_RATIO: f64 = 1.05;
const SPLIT_THREAT_RATIO: f64 = 2.6;
const SPLIT_KILL_RATIO: f64 = 1.6;
const SPLIT_REACH: f64 = 710.0;
const SPLIT_MIN_MASS: f64 = 35.0;
const MAX_CELLS: usize = 16;
const VIRUS_MASS: f64 = 100.0;
const EJECT_COST: f64 = 18.0;
const EJECT_REACH: f64 = 660.0;
const VIRUS_FEED: i32 = 8;
const PELLET_MASS: f64 = 25.0;
const CASCADE_RANGE: f64 = 900.0;
const FORESEE_RANGE: f64 = 200.0;
const FEED_RANGE: f64 = 300.0;
const FEED_MIN: f64 = 26.0;
const FEED_SHARE: f64 = 0.7;
const FEEDER_RANGE: f64 = 700.0;

// ------------------------------------------------------------- the weighting
const FOOD_GAIN: f64 = 1000.0;
const FOOD_SOFT: f64 = 400.0;
const PREY_GAIN: f64 = 2.2;
const VIRUS_FOOD_GAIN: f64 = 0.5;
const SHELTER_GAIN: f64 = 0.8;
const THREAT_WEIGHT: f64 = 500.0;
const VIRUS_WEIGHT: f64 = 220.0;
const WALL_WEIGHT: f64 = 260.0;
const THREAT_MARGIN: f64 = 120.0;
const THREAT_BUFFER: f64 = 250.0;
const THREAT_LOOKAHEAD: f64 = 400.0;
const CLOSING_LEAD: f64 = 2000.0;
const SPLIT_FEAR: f64 = 0.45;
const SPLIT_BITE_FULL: f64 = 0.12;
const SPLIT_BITE_NONE: f64 = 0.03;
const SPLIT_FEAR_FLOOR: f64 = 0.2;
const SPLIT_SHUN: f64 = 0.4;
const ESCAPE_LEAN: f64 = 0.15;
const FLIGHT_SPEED: f64 = 0.5;
const FLIGHT_CROSS: f64 = 0.5;
const COVER_MARGIN: f64 = 60.0;
const COVER_DISCOUNT: f64 = 0.5;
const VIRUS_MARGIN: f64 = 60.0;
const VIRUS_BUFFER: f64 = 450.0;
const SHOT_WEIGHT: f64 = 2.0;
const SHOT_REACH: f64 = 900.0;
const WALL_LOOKAHEAD: f64 = 500.0;
const MAP_MARGIN: f64 = 20.0;
const WALL_TOUCH: f64 = 40.0;
const SLIDE_LEAN: f64 = 0.15;
const WALL_NEAR: f64 = 250.0;
const WALL_SHOVE: f64 = 90.0;
const WALL_PRESSURE: f64 = 2.0;
const ROAM_GAIN: f64 = 60.0;
const ROAM_RANGE: f64 = 4000.0;
const MENACE_RANGE: f64 = 1600.0;
const ESCAPE_GAIN: f64 = 90.0;
const ESCAPE_REACH: f64 = 3000.0;
const ESCAPE_BOX: f64 = 6000.0;
const FLEE_PRESSURE: f64 = 0.35;
const GATHER_PRESSURE: f64 = 0.1;
const MERGE_PULL: f64 = 0.35;
const CLEAR_MARGIN: f64 = 50.0;
const TURN_MARGIN: f64 = 6.0;
const TURN_MARGIN_SHARE: f64 = 0.08;
const TURN_REVERSAL: f64 = 0.15;
const COMMIT_CONE: f64 = 0.82;
const COMMIT_KEEP: f64 = 0.3;
const COMMIT_SWITCH: f64 = 1.35;
const COMMIT_LEAN: f64 = 3.0;

// ------------------------------------------------------------------ tactics
const SPLIT_MAX_CELLS: usize = 2;
const SPLIT_COOLDOWN: f64 = 900.0;
const SPLIT_ALIGN: f64 = 0.97;
const SPLIT_TRUST: f64 = 0.9;
const SPLIT_CHASE: f64 = 0.5;
const SPLIT_WALK: f64 = 1500.0;
const PREY_FLEEING: f64 = 0.03;
const GRAZE_LEAD: f64 = 1000.0;
const GRAZE_DISCOUNT: f64 = 0.3;
const PREY_REACH: f64 = 1200.0;
const STRIKE_MARGIN: f64 = 200.0;
const TRAP_RANGE: f64 = 350.0;
const TRAP_SPAN: i32 = 4;
const TRAP_STEP: f64 = std::f64::consts::PI / 12.0;
const TRAP_BLOCKED: i32 = 6;
const SPLIT_WORTH: f64 = 0.06;
const SPLIT_CAUTION: f64 = 300.0;
const AGGRO_PRESSURE: f64 = 0.35;
const ESCAPE_CONTACT: f64 = 0.75;
const ESCAPE_CLEAR: f64 = 60.0;
const ESCAPE_SWALLOW: f64 = 0.5;
const ESCAPE_LOSING: f64 = 1.2;
const ESCAPE_AWAY: f64 = 0.3;
const ESCAPE_INTO: f64 = 0.5;
const CHASE_STRIDE: f64 = 25.0;
const POP_MIN_MASS: f64 = 280.0;
const POP_MIN_ENEMY: f64 = 150.0;
const POP_RANGE: f64 = 900.0;
const POP_WALK: f64 = 1600.0;
const POP_STANDOFF: f64 = 90.0;
const POP_ARC: f64 = 0.6;
const POP_STAGE_TIMEOUT: f64 = 8000.0;
const POP_STAGE_REST: f64 = 8000.0;
const HIDE_RANGE: f64 = 1600.0;
const LURE_WALK: f64 = 900.0;
const LURE_RANGE: f64 = 1400.0;
const LURE_BAIL: f64 = 90.0;
const LURE_CLEAR: f64 = 40.0;
const LURE_TIMEOUT: f64 = 5000.0;
const LURE_REST: f64 = 15000.0;
const LURE_CLOSING: f64 = 0.05;
const HUNT_RANGE: f64 = 700.0;
const HUNT_CLOSING: f64 = 0.45;
const HUNT_SUSTAIN: f64 = 600.0;
const HIDE_HOLD: f64 = 1500.0;
const POP_ALIGN: f64 = 0.978;
const POP_CONTACT: f64 = 0.2;
const POP_CONTACT_HOLD: f64 = 0.4;
const FEED_MEMORY: f64 = 30000.0;
const EJECT_INTERVAL: f64 = 120.0;
const POP_TIMEOUT: f64 = 6000.0;
const POP_SCAN: f64 = 400.0;
const BAIT_WINDOW: f64 = 4000.0;
const BAIT_RANGE: f64 = 1400.0;
const BAIT_MARGIN: f64 = 1.35;
const BAIT_DISCOUNT: f64 = 0.15;
const POUR_MERGE_FAR: f64 = 10000.0;
const POUR_REACH: f64 = 600.0;
const POUR_KEEP: f64 = SPLIT_MIN_MASS + EJECT_COST;
const POUR_PRESSURE: f64 = 0.2;
const POUR_TIMEOUT: f64 = 4000.0;
const POUR_REST: f64 = 6000.0;
const POUR_WORTH: f64 = 0.1;

const DIRECTIONS: usize = 64;
const DIR_STEP: f64 = (std::f64::consts::PI * 2.0) / DIRECTIONS as f64;
const FOOD_CONE: i64 = 11;

const MEMORY_MS: f64 = 3000.0;
const VIEW_BASE_X: f64 = 1920.0;
const VIEW_BASE_Y: f64 = 1080.0;
const MIN_SCALE: f64 = 0.15;
const INF: f64 = f64::INFINITY;

type Field = [f64; DIRECTIONS];

struct Tables {
    dx: Field,
    dy: Field,
    kernel: [f64; 12],
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = Tables { dx: [0.0; DIRECTIONS], dy: [0.0; DIRECTIONS], kernel: [0.0; 12] };
        for i in 0..DIRECTIONS {
            t.dx[i] = (i as f64 * DIR_STEP).cos();
            t.dy[i] = (i as f64 * DIR_STEP).sin();
        }
        for j in 0..12 {
            let c = (j as f64 * DIR_STEP).cos();
            t.kernel[j] = c * c * c;
        }
        t
    })
}

fn mass_of(size: f64) -> f64 {
    (size * size) / 100.0
}

fn size_of(mass: f64) -> f64 {
    (mass * 100.0).sqrt()
}

fn walk_speed(size: f64) -> f64 {
    2.2 * size.powf(-0.45)
}

fn clamp01(v: f64) -> f64 {
    if v < 0.0 {
        0.0
    } else if v > 1.0 {
        1.0
    } else {
        v
    }
}

fn split_appetite(meal: f64, mass: f64) -> f64 {
    let bite = clamp01((meal / mass - SPLIT_BITE_NONE) / (SPLIT_BITE_FULL - SPLIT_BITE_NONE));
    SPLIT_FEAR_FLOOR + (1.0 - SPLIT_FEAR_FLOOR) * bite
}

fn span(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = ax - bx;
    let dy = ay - by;
    (dx * dx + dy * dy).sqrt()
}

fn merge_delay(mass: f64) -> f64 {
    let agario = 30000.0 + 23.3 * mass;
    let ogar = f64::max(30000.0, size_of(mass) * 200.0);
    f64::max(agario, ogar)
}

fn distance_to_segment(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = bx - ax;
    let dy = by - ay;
    let length = dx * dx + dy * dy;
    let mut t = if length > 0.0 { ((px - ax) * dx + (py - ay) * dy) / length } else { 0.0 };
    if t < 0.0 {
        t = 0.0;
    } else if t > 1.0 {
        t = 1.0;
    }
    span(px, py, ax + dx * t, ay + dy * t)
}

fn js_round(v: f64) -> i64 {
    (v + 0.5).floor() as i64
}

fn or1(v: f64) -> f64 {
    if v == 0.0 {
        1.0
    } else {
        v
    }
}

#[derive(Clone, Copy, Debug)]
struct C {
    id: u32,
    x: f64,
    y: f64,
    size: f64,
    color: u32,
    kind: Kind,
}

impl C {
    fn from(c: &Cell, kind: Kind) -> C {
        C { id: c.id, x: c.x, y: c.y, size: c.size, color: c.color, kind }
    }
    fn virus(&self) -> bool {
        self.kind == Kind::Virus
    }
    fn ejected(&self) -> bool {
        self.kind == Kind::Ejected
    }
    fn mass(&self) -> f64 {
        mass_of(self.size)
    }
}

#[derive(Clone, Debug)]
struct E {
    x: f64,
    y: f64,
    size: f64,
    mass: f64,
    id: Option<u32>,
    cell: Option<C>,
    remembered: bool,
    potential: f64,
    grown: f64,
    absorbed: Option<Vec<u32>>,
    eats: f64,
    vx: f64,
    vy: f64,
    trapped: bool,
    catch: Option<f64>,
}

impl E {
    fn new(x: f64, y: f64, size: f64, mass: f64, id: Option<u32>, cell: Option<C>, remembered: bool) -> E {
        E { x, y, size, mass, id, cell, remembered, potential: 0.0, grown: 0.0, absorbed: None, eats: 0.0, vx: 0.0, vy: 0.0,
            trapped: false, catch: None }
    }
}

struct Piece {
    id: u32,
    mass: f64,
    x: f64,
    y: f64,
    size: f64,
}

struct Ahead {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
}

struct Near {
    cell: Option<usize>,
    distance: f64,
    ux: f64,
    uy: f64,
}

#[derive(Clone, Copy)]
struct Motion {
    x: f64,
    y: f64,
    at: f64,
    vx: f64,
    vy: f64,
    moved: bool,
}

struct Pick {
    id: u32,
    distance: f64,
    ux: f64,
    uy: f64,
}

struct View {
    mine: Vec<C>,
    centre: (f64, f64),
    spread: f64,
    big: C,
    small: C,
    big_mass: f64,
    small_mass: f64,
    total: f64,
    food: Vec<C>,
    viruses: Vec<C>,
    ents: Vec<E>,         // enemies first (n_enemies), then merged-pair threat entries
    n_enemies: usize,
    prey: Vec<usize>,
    threats: Vec<usize>,
    zones: Vec<(f64, f64, f64, f64, f64)>,
}

#[derive(Clone, Copy)]
struct Feeding {
    virus: u32,
    enemy: u32,
    fed: i32,
    last: f64,
    since: f64,
}

#[derive(Clone, Copy)]
struct Staging {
    virus: u32,
    enemy: u32,
    since: f64,
}

#[derive(Clone, Copy)]
struct Pouring {
    since: f64,
    last: f64,
}

#[derive(Clone, Copy)]
struct Luring {
    virus: u32,
    since: f64,
}

type Pt = (f64, f64);

pub struct AposPolicy {
    aggressive: bool,
    step_ms: f64,
    memory_ms: f64,
    pub counters: Counters,
    now: f64,
    memory: HashMap<u32, (C, f64)>,
    state: &'static str,
    detour: Option<usize>,
    hunted_at: f64,
    being_chased: bool,
    target_id: Option<u32>,
    feeding: Option<Feeding>,
    staging: Option<Staging>,
    pouring: Option<Pouring>,
    luring: Option<Luring>,
    point_x: f64,
    point_y: f64,
    heading: Option<usize>,
    slide_x: f64,
    slide_y: f64,
    pursuit: HashMap<u32, f64>,
    stage_rest: HashMap<u32, f64>,
    fed_viruses: HashMap<u32, (i32, f64)>,
    pour_rest: f64,
    lure_rest: f64,
    pop_scan: f64,
    broke_at: f64,
    motion: HashMap<u32, Motion>,
    split_at: f64,
    risk: Field,
    gain: Field,
    prey_gain: Field,
    value: Field,
    repositioning: bool,
    cell_count: usize,
    contact: f64,
    bait_id: Option<u32>,
    merge_left: f64,
    do_split: bool,
    do_eject: bool,
    vd: f64,
    scale: f64,
    border: Border,
}

impl AposPolicy {
    /// `spec`: {"kind": "apos", "aggressive": true, "step_ms": 80, "memory_ms": 3000}
    pub fn from_spec(spec: &serde_json::Value) -> AposPolicy {
        let aggressive = spec.get("aggressive").and_then(|v| v.as_bool()).unwrap_or(true);
        let step_ms = spec.get("step_ms").and_then(|v| v.as_f64()).unwrap_or(80.0);
        let memory_ms = spec.get("memory_ms").and_then(|v| v.as_f64()).unwrap_or(MEMORY_MS);
        let mut counters = Counters::default();
        for k in ["steps", "splits", "ejects", "shots_started", "shots_fired", "virus_hides", "baits", "lures", "pours",
                  "escapes", "double_splits"] {
            counters.0.push((k.to_string(), 0));
        }
        AposPolicy {
            aggressive,
            step_ms,
            memory_ms,
            counters,
            now: 1_000_000.0,
            memory: HashMap::new(),
            state: "roam",
            detour: None,
            hunted_at: 0.0,
            being_chased: false,
            target_id: None,
            feeding: None,
            staging: None,
            pouring: None,
            luring: None,
            point_x: 0.0,
            point_y: 0.0,
            heading: None,
            slide_x: 1.0,
            slide_y: 1.0,
            pursuit: HashMap::new(),
            stage_rest: HashMap::new(),
            fed_viruses: HashMap::new(),
            pour_rest: 0.0,
            lure_rest: 0.0,
            pop_scan: 0.0,
            broke_at: 0.0,
            motion: HashMap::new(),
            split_at: 0.0,
            risk: [0.0; DIRECTIONS],
            gain: [0.0; DIRECTIONS],
            prey_gain: [0.0; DIRECTIONS],
            value: [0.0; DIRECTIONS],
            repositioning: false,
            cell_count: 0,
            contact: 0.0,
            bait_id: None,
            merge_left: 0.0,
            do_split: false,
            do_eject: false,
            vd: 0.0,
            scale: 1.0,
            border: Border { left: -1.0, right: 1.0, top: -1.0, bottom: 1.0 },
        }
    }

    pub fn reset(&mut self) {
        self.now += self.step_ms;
        self.state = "roam";
        self.cell_count = 0;
        self.feeding = None;
        self.staging = None;
        self.luring = None;
        self.pouring = None;
        self.target_id = None;
        self.detour = None;
        self.hunted_at = 0.0;
        self.pursuit.clear();
    }

    pub fn act(&mut self, s: &Snapshot) -> Out {
        self.do_split = false;
        self.do_eject = false;
        self.now += self.step_ms;
        let mut mine: Vec<C> = s.mine.iter().map(|c| C::from(c, Kind::Mine)).collect();
        mine.sort_by(|a, b| b.size.partial_cmp(&a.size).unwrap().then(a.id.cmp(&b.id)));
        self.border = s.border;
        let mut total_size = 0.0;
        for c in &mine {
            total_size += c.size;
        }
        let scale = f64::max((f64::min(64.0 / total_size, 1.0)).powf(0.4), MIN_SCALE);
        self.vd = (VIEW_BASE_X * VIEW_BASE_X + VIEW_BASE_Y * VIEW_BASE_Y).sqrt() / scale;
        self.scale = scale;
        let target = self.main_loop(mine, s);
        if let Some((x, y)) = target {
            self.point_x = x;
            self.point_y = y;
        }
        self.counters.bump("steps", 1);
        self.counters.bump("splits", self.do_split as i64);
        self.counters.bump("ejects", self.do_eject as i64);
        let k = format!("mode_{}", self.state);
        self.counters.bump(&k, 1);
        let (x, y) = target.unwrap_or((self.point_x, self.point_y));
        Out { x, y, split: self.do_split, eject: self.do_eject }
    }

    // --------------------------------------------------------- api helpers

    fn known(&self) -> bool {
        let b = &self.border;
        b.left.is_finite() && b.right.is_finite() && b.top.is_finite() && b.bottom.is_finite()
    }

    // ----------------------------------------------------------- the world

    fn cells(&mut self, mine: &[C], s: &Snapshot) -> Vec<(C, bool)> {
        let mut visible: HashMap<u32, C> = HashMap::new();
        let mut vis_order: Vec<u32> = Vec::new();
        for o in &s.others {
            if mine.iter().any(|m| m.id == o.id) {
                continue;
            }
            visible.insert(o.id, C::from(o, o.kind));
            vis_order.push(o.id);
        }
        let mut cx = 0.0;
        let mut cy = 0.0;
        for c in mine {
            cx += c.x;
            cy += c.y;
        }
        cx /= mine.len() as f64;
        cy /= mine.len() as f64;
        let hw = VIEW_BASE_X / self.scale / 2.0 * 0.9;
        let hh = VIEW_BASE_Y / self.scale / 2.0 * 0.9;
        for &cid in &vis_order {
            let c = visible[&cid];
            if c.kind == Kind::Player || c.kind == Kind::Virus {
                self.memory.insert(cid, (c, self.now));
            }
        }
        let now = self.now;
        let memory_ms = self.memory_ms;
        self.memory.retain(|cid, (c, seen)| {
            if visible.contains_key(cid) {
                return true;
            }
            !(mine.iter().any(|m| m.id == *cid) || now - *seen > memory_ms || ((c.x - cx).abs() < hw && (c.y - cy).abs() < hh))
        });
        let mut out: Vec<(C, bool)> = vis_order.iter().map(|id| (visible[id], false)).collect();
        for (cid, (c, _)) in self.memory.iter() {
            if !visible.contains_key(cid) {
                out.push((*c, true));
            }
        }
        out.sort_by_key(|t| t.0.id);
        out
    }

    fn survey(&mut self, mine: Vec<C>, s: &Snapshot) -> View {
        let mut total = 0.0;
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut big = mine[0];
        let mut small = mine[0];
        for cell in &mine {
            total += cell.mass();
            cx += cell.x;
            cy += cell.y;
            if cell.size > big.size {
                big = *cell;
            }
            if cell.size < small.size {
                small = *cell;
            }
        }
        cx /= mine.len() as f64;
        cy /= mine.len() as f64;
        let mut spread = 0.0;
        for cell in &mine {
            spread = f64::max(spread, span(cell.x, cell.y, cx, cy) + cell.size);
        }
        let big_mass = big.mass();
        let small_mass = small.mass();
        let mut food: Vec<C> = Vec::new();
        let mut viruses: Vec<C> = Vec::new();
        let mut ents: Vec<E> = Vec::new();
        for (cell, remembered) in self.cells(&mine, s) {
            if cell.virus() {
                viruses.push(cell);
                continue;
            }
            let mass = cell.mass();
            let eatable = mass * PREY_RATIO < big_mass;
            if cell.ejected() {
                if eatable && !remembered {
                    food.push(cell);
                }
                continue;
            }
            if eatable && cell.kind == Kind::Food {
                if !remembered {
                    food.push(cell);
                }
                continue;
            }
            ents.push(E::new(cell.x, cell.y, cell.size, mass, Some(cell.id), Some(cell), remembered));
        }
        foresee(&mut ents, &food);
        let mut ladder: Vec<Piece> =
            mine.iter().map(|c| Piece { id: c.id, mass: c.mass(), x: c.x, y: c.y, size: c.size }).collect();
        ladder.sort_by(|a, b| a.mass.partial_cmp(&b.mass).unwrap());
        for e in ents.iter_mut() {
            cascade(e, &ladder);
        }
        let n_enemies = ents.len();
        let mut prey = Vec::new();
        let mut threats = Vec::new();
        for (i, e) in ents.iter().enumerate() {
            if !e.remembered && e.grown * PREY_RATIO < big_mass {
                prey.push(i);
            }
            if e.grown > small_mass * THREAT_RATIO {
                threats.push(i);
            }
        }
        merge_threats(&mut ents, &mut threats, small_mass, &ladder);
        let mut zones = Vec::new();
        for &ti in &threats {
            let threat = &ents[ti];
            let grown = grown_for(threat, &small);
            let splits = grown > small_mass * SPLIT_THREAT_RATIO;
            let mouth = threat.size + big.size + THREAT_MARGIN;
            let jump = mouth + SPLIT_REACH;
            zones.push((
                threat.x,
                threat.y,
                mouth * mouth,
                if splits { jump * jump } else { 0.0 },
                if splits { split_appetite(f64::max(small_mass, threat.eats), threat.mass) } else { 0.0 },
            ));
        }
        View { mine, centre: (cx, cy), spread, big, small, big_mass, small_mass, total, food, viruses, ents, n_enemies, prey,
               threats, zones }
    }

    fn track_motion(&mut self, view: &mut View) {
        let now = self.now;
        let mut seen: Vec<u32> = Vec::new();
        let motion = &mut self.motion;
        let mut remember = |cid: u32, x: f64, y: f64, entry: Option<&mut E>| {
            seen.push(cid);
            match motion.get_mut(&cid) {
                None => {
                    motion.insert(cid, Motion { x, y, at: now, vx: 0.0, vy: 0.0, moved: false });
                    return;
                }
                Some(last) => {
                    if now > last.at {
                        let elapsed = now - last.at;
                        let vx = (x - last.x) / elapsed;
                        let vy = (y - last.y) / elapsed;
                        last.vx = if last.moved { (last.vx + vx) / 2.0 } else { vx };
                        last.vy = if last.moved { (last.vy + vy) / 2.0 } else { vy };
                        last.moved = true;
                        last.x = x;
                        last.y = y;
                        last.at = now;
                    }
                    if let Some(e) = entry {
                        e.vx = last.vx;
                        e.vy = last.vy;
                    }
                }
            }
        };
        for &i in &view.prey {
            let e = &mut view.ents[i];
            if let Some(c) = e.cell {
                let (x, y) = (e.x, e.y);
                remember(c.id, x, y, Some(e));
            }
        }
        for &i in &view.threats {
            let e = &mut view.ents[i];
            if let Some(c) = e.cell {
                let (x, y) = (e.x, e.y);
                remember(c.id, x, y, Some(e));
            }
        }
        for c in &view.mine {
            remember(c.id, c.x, c.y, None);
        }
        self.motion.retain(|id, _| seen.contains(id));
    }

    fn speed_of(&self, id: u32) -> f64 {
        match self.motion.get(&id) {
            None => 0.0,
            Some(l) => (l.vx * l.vx + l.vy * l.vy).sqrt(),
        }
    }

    // ------------------------------------------------------------- scoring

    fn score(&mut self, view: &mut View) -> f64 {
        let mut risk = [0.0; DIRECTIONS];
        let mut gain = [0.0; DIRECTIONS];
        let mut prey_gain = [0.0; DIRECTIONS];
        let pressure = self.score_threats(view, &mut risk);
        self.score_food(view, &mut gain);
        self.score_prey(view, &mut prey_gain);
        self.score_viruses(view, &mut risk, &mut gain, pressure);
        self.score_walls(view, &mut risk, pressure);
        self.score_open(view, &mut gain);
        self.score_escape(view, &mut gain);
        self.risk = risk;
        self.gain = gain;
        self.prey_gain = prey_gain;
        pressure
    }

    fn horizon(&self) -> f64 {
        f64::max(600.0, self.vd * 0.5)
    }

    fn score_threats(&mut self, view: &View, risk: &mut Field) -> f64 {
        let mut pressure = 0.0;
        self.contact = 0.0;
        let horizon = self.horizon();
        for &ti in &view.threats {
            let threat = &view.ents[ti];
            let baited = threat.id.is_some() && threat.id == self.bait_id;
            let discount = if baited { BAIT_DISCOUNT } else { 1.0 };
            let ahead = ahead_of(threat);
            for cell in &view.mine {
                let mass = cell.mass();
                let grown = grown_for(threat, cell);
                if grown < mass * THREAT_RATIO {
                    continue;
                }
                let weight = if self.covered(view, cell, threat) { discount * COVER_DISCOUNT } else { discount };
                let reach = threat.size + cell.size + THREAT_MARGIN;
                let splits = grown > mass * SPLIT_THREAT_RATIO && grown >= SPLIT_MIN_MASS;
                let fear = if splits { SPLIT_FEAR * split_appetite(f64::max(mass, threat.eats), threat.mass) } else { 0.0 };
                let gap = f64::min(span(threat.x, threat.y, cell.x, cell.y), span(ahead.x, ahead.y, cell.x, cell.y)) - reach;
                let near = clamp01(1.0 - gap / THREAT_BUFFER);
                let jump = if splits { clamp01(1.0 - (gap - SPLIT_REACH) / THREAT_BUFFER) } else { 0.0 };
                if near > 0.0 || jump > 0.0 {
                    pressure = f64::max(pressure, weight * f64::max(near, jump * fear));
                    self.contact = f64::max(self.contact, weight * near);
                }
                let stake = mass / view.big_mass;
                let nearer = f64::max(0.0, self.closing_on(threat, cell)) * CLOSING_LEAD;
                sweep(risk, cell, ahead.x, ahead.y, ahead.vx, ahead.vy, reach, THREAT_BUFFER, weight * stake * THREAT_WEIGHT,
                      horizon, nearer);
                if splits {
                    sweep(risk, cell, ahead.x, ahead.y, ahead.vx, ahead.vy, reach + SPLIT_REACH, THREAT_BUFFER,
                          weight * stake * THREAT_WEIGHT * fear, horizon, nearer);
                }
            }
        }
        pressure
    }

    fn covered(&self, view: &View, cell: &C, threat: &E) -> bool {
        if view.mine.len() < 2 {
            return false;
        }
        if view.big_mass < threat.grown * PREY_RATIO {
            return false;
        }
        let dx = threat.x - cell.x;
        let dy = threat.y - cell.y;
        let away = or1((dx * dx + dy * dy).sqrt());
        let bite = threat.size + cell.size;
        let mx = cell.x + (dx / away) * bite;
        let my = cell.y + (dy / away) * bite;
        for own in &view.mine {
            if own.id == cell.id {
                continue;
            }
            if own.mass() < threat.grown * PREY_RATIO {
                continue;
            }
            if span(own.x, own.y, mx, my) <= own.size + COVER_MARGIN {
                return true;
            }
        }
        false
    }

    fn closing_on(&self, threat: &E, cell: &C) -> f64 {
        let last = match threat.id {
            Some(id) => self.motion.get(&id),
            None => None,
        };
        match last {
            None => 0.0,
            Some(l) => {
                let dx = cell.x - threat.x;
                let dy = cell.y - threat.y;
                let away = or1((dx * dx + dy * dy).sqrt());
                (l.vx * dx + l.vy * dy) / away
            }
        }
    }

    fn score_food(&self, view: &View, gain: &mut Field) {
        let grazers = grazers_of(view);
        for cell in &view.food {
            let near = nearest_cell(&view.mine, cell.x, cell.y, None);
            if near.distance < 1.0 {
                continue;
            }
            let mut value = (FOOD_GAIN * cell.mass()) / (near.distance + FOOD_SOFT);
            if !grazers.is_empty() && grazed(&grazers, cell) {
                value *= GRAZE_DISCOUNT;
            }
            spread(gain, near.ux, near.uy, value);
        }
    }

    fn score_prey(&self, view: &mut View, prey_gain: &mut Field) {
        let ours = self.our_speed(view);
        for k in 0..view.prey.len() {
            let i = view.prey[k];
            let (ex, ey, emass, egrown) = {
                let e = &view.ents[i];
                (e.x, e.y, e.mass, e.grown)
            };
            let near = nearest_cell(&view.mine, ex, ey, Some(emass));
            let nc = match near.cell {
                Some(c) => c,
                None => continue,
            };
            if near.distance < 1.0 {
                continue;
            }
            let trapped = self.cornered(&view.ents[i]) || (near.distance < PREY_REACH && self.trapped(view, i, &near));
            view.ents[i].trapped = trapped;
            let catch = self.catchable(view, i, &near, ours);
            view.ents[i].catch = Some(catch);
            let near_cell = view.mine[nc];
            let mut value = (FOOD_GAIN * emass * catch) / (near.distance + FOOD_SOFT);
            if near.distance < near_cell.size + SPLIT_REACH && near_cell.mass() / 2.0 >= egrown * SPLIT_KILL_RATIO {
                value *= 2.0;
            }
            if trapped {
                value *= 1.5;
            }
            if near.distance > PREY_REACH {
                value *= PREY_REACH / near.distance;
            }
            spread(prey_gain, near.ux, near.uy, value);
        }
    }

    fn trapped(&self, view: &View, ei: usize, near: &Near) -> bool {
        let entry = &view.ents[ei];
        let pops = entry.mass > VIRUS_MASS * EAT_RATIO;
        let heading = near.uy.atan2(near.ux);
        let mut blocked = 0;
        for k in -TRAP_SPAN..=TRAP_SPAN {
            let angle = heading + k as f64 * TRAP_STEP;
            let dx = angle.cos();
            let dy = angle.sin();
            if self.room_along(entry.x, entry.y, entry.size, dx, dy) < TRAP_RANGE {
                blocked += 1;
                continue;
            }
            if !pops {
                continue;
            }
            let end_x = entry.x + dx * TRAP_RANGE;
            let end_y = entry.y + dy * TRAP_RANGE;
            for virus in &view.viruses {
                let gap = distance_to_segment(virus.x, virus.y, entry.x, entry.y, end_x, end_y);
                if gap < virus.size {
                    blocked += 1;
                    break;
                }
            }
        }
        blocked >= TRAP_BLOCKED
    }

    fn room_along(&self, cx: f64, cy: f64, size: f64, dx: f64, dy: f64) -> f64 {
        if !self.known() {
            return INF;
        }
        let b = &self.border;
        let (min_x, max_x, min_y, max_y) = (b.left, b.right, b.top, b.bottom);
        let mut exit = INF;
        if dx > 0.0 {
            exit = (max_x - size - cx) / dx;
        } else if dx < 0.0 {
            exit = (min_x + size - cx) / dx;
        }
        if dy > 0.0 {
            exit = f64::min(exit, (max_y - size - cy) / dy);
        } else if dy < 0.0 {
            exit = f64::min(exit, (min_y + size - cy) / dy);
        }
        if exit < 0.0 {
            0.0
        } else {
            exit
        }
    }

    fn catchable(&self, view: &View, ei: usize, near: &Near, ours: f64) -> f64 {
        let entry = &view.ents[ei];
        let away = entry.vx * near.ux + entry.vy * near.uy;
        if away <= 0.0 {
            return 1.0;
        }
        if entry.trapped {
            return 1.0;
        }
        if view.big_mass / 2.0 >= entry.grown * SPLIT_KILL_RATIO {
            let half = size_of(view.big_mass / 2.0);
            let reach = half + SPLIT_REACH * SPLIT_TRUST;
            if span(entry.x, entry.y, view.big.x, view.big.y) < reach {
                return 1.0;
            }
        }
        let sp = self.speed_of(view.mine[near.cell.unwrap()].id);
        let chaser = if sp != 0.0 { sp } else { ours };
        if chaser <= 0.0 {
            return 1.0;
        }
        f64::max(PREY_FLEEING, 1.0 - away / chaser)
    }

    fn cornered(&self, entry: &E) -> bool {
        let b = &self.border;
        if !(b.left.is_finite() && b.right.is_finite()) {
            return false;
        }
        let margin = entry.size + 300.0;
        entry.x - b.left < margin || b.right - entry.x < margin || entry.y - b.top < margin || b.bottom - entry.y < margin
    }

    fn score_viruses(&self, view: &View, risk: &mut Field, gain: &mut Field, pressure: f64) {
        let room = MAX_CELLS.saturating_sub(view.mine.len());
        let shelter = self.being_chased && shelters(view);
        let horizon = self.horizon();
        let breaks = view.big_mass > VIRUS_MASS * EAT_RATIO;
        for virus in &view.viruses {
            let near = nearest_cell(&view.mine, virus.x, virus.y, None);
            if near.distance < 1.0 {
                continue;
            }
            if breaks {
                if room == 0 {
                    let value = (VIRUS_FOOD_GAIN * FOOD_GAIN * VIRUS_MASS) / (near.distance + FOOD_SOFT);
                    spread(gain, near.ux, near.uy, value);
                    continue;
                }
                let weight = VIRUS_WEIGHT * (room as f64 / MAX_CELLS as f64);
                let shot = loaded(view, virus);
                for cell in &view.mine {
                    let mass = cell.mass();
                    if mass <= VIRUS_MASS * EAT_RATIO {
                        continue;
                    }
                    let radius = virus.size + cell.size + VIRUS_MARGIN;
                    let share = (weight * mass) / view.big_mass;
                    sweep(risk, cell, virus.x, virus.y, 0.0, 0.0, radius, VIRUS_BUFFER, share, horizon, 0.0);
                    let shot = match shot {
                        Some(s) => s,
                        None => continue,
                    };
                    let rx = virus.x - cell.x;
                    let ry = virus.y - cell.y;
                    if rx * shot.0 + ry * shot.1 >= 0.0 {
                        continue;
                    }
                    if rx * rx + ry * ry > SHOT_REACH * SHOT_REACH {
                        continue;
                    }
                    sweep_flight(risk, shot.0, shot.1, rx, ry, radius, VIRUS_BUFFER, share * SHOT_WEIGHT);
                }
            } else if shelter {
                let value = (SHELTER_GAIN * pressure * FOOD_GAIN * VIRUS_MASS) / (near.distance + FOOD_SOFT);
                spread(gain, near.ux, near.uy, value);
            }
        }
    }

    fn score_walls(&self, view: &View, risk: &mut Field, pressure: f64) {
        if !self.known() {
            return;
        }
        let t = tables();
        let b = &self.border;
        let (min_x, min_y, max_x, max_y) = (b.left, b.top, b.right, b.bottom);
        let widen = 1.0 + pressure * WALL_PRESSURE;
        let lookahead = WALL_LOOKAHEAD * widen;
        for cell in &view.mine {
            let inset = cell.size;
            let left = min_x + inset - cell.x;
            let right = max_x - inset - cell.x;
            let top = min_y + inset - cell.y;
            let bottom = max_y - inset - cell.y;
            for i in 0..DIRECTIONS {
                let dx = t.dx[i];
                let dy = t.dy[i];
                let mut exit = INF;
                if dx > 0.0 {
                    exit = right / dx;
                } else if dx < 0.0 {
                    exit = left / dx;
                }
                if dy > 0.0 {
                    exit = f64::min(exit, bottom / dy);
                } else if dy < 0.0 {
                    exit = f64::min(exit, top / dy);
                }
                if exit >= lookahead {
                    continue;
                }
                if exit < 0.0 {
                    exit = 0.0;
                }
                let crowding = 1.0 - exit / lookahead;
                risk[i] += WALL_WEIGHT * crowding * crowding;
            }
            let near = (WALL_NEAR + cell.size / 2.0) * widen;
            push_wall(risk, cell.x - min_x, -1.0, 0.0, near);
            push_wall(risk, max_x - cell.x, 1.0, 0.0, near);
            push_wall(risk, cell.y - min_y, 0.0, -1.0, near);
            push_wall(risk, max_y - cell.y, 0.0, 1.0, near);
        }
    }

    fn score_escape(&self, view: &View, gain: &mut Field) {
        if !self.known() {
            return;
        }
        let t = tables();
        let b = &self.border;
        let (min_x, min_y, max_x, max_y) = (b.left, b.top, b.right, b.bottom);
        let mut menace: f64 = 0.0;
        for &ti in &view.threats {
            let threat = &view.ents[ti];
            for cell in &view.mine {
                let mass = cell.mass();
                if grown_for(threat, cell) < mass * THREAT_RATIO {
                    continue;
                }
                if self.covered(view, cell, threat) {
                    continue;
                }
                let gap = span(threat.x, threat.y, cell.x, cell.y) - threat.size - cell.size;
                let near = clamp01(1.0 - gap / MENACE_RANGE);
                if near > menace {
                    menace = near;
                }
            }
        }
        if menace <= 0.0 {
            return;
        }
        let (ax, ay) = view.centre;
        let mut here = ax - min_x;
        if max_x - ax < here {
            here = max_x - ax;
        }
        if ay - min_y < here {
            here = ay - min_y;
        }
        if max_y - ay < here {
            here = max_y - ay;
        }
        let boxed = clamp01(1.0 - here / ESCAPE_BOX);
        if boxed <= 0.0 {
            return;
        }
        let weight = ESCAPE_GAIN * menace * boxed;
        for i in 0..DIRECTIONS {
            let x = ax + t.dx[i] * ESCAPE_REACH;
            let y = ay + t.dy[i] * ESCAPE_REACH;
            let mut room = x - min_x;
            if max_x - x < room {
                room = max_x - x;
            }
            if y - min_y < room {
                room = y - min_y;
            }
            if max_y - y < room {
                room = max_y - y;
            }
            if room < 0.0 {
                room = 0.0;
            } else if room > ESCAPE_REACH {
                room = ESCAPE_REACH;
            }
            gain[i] += (weight * room) / ESCAPE_REACH;
        }
    }

    fn score_open(&mut self, view: &View, gain: &mut Field) {
        self.repositioning = false;
        if !view.food.is_empty() || !view.prey.is_empty() {
            return;
        }
        if !self.known() {
            return;
        }
        let t = tables();
        let b = &self.border;
        let (min_x, min_y, max_x, max_y) = (b.left, b.top, b.right, b.bottom);
        self.repositioning = true;
        let cell = view.big;
        for i in 0..DIRECTIONS {
            let x = cell.x + t.dx[i] * ROAM_RANGE;
            let y = cell.y + t.dy[i] * ROAM_RANGE;
            let mut room = x - min_x;
            if max_x - x < room {
                room = max_x - x;
            }
            if y - min_y < room {
                room = y - min_y;
            }
            if max_y - y < room {
                room = max_y - y;
            }
            if room < 0.0 {
                room = 0.0;
            } else if room > ROAM_RANGE {
                room = ROAM_RANGE;
            }
            gain[i] += (ROAM_GAIN * room) / ROAM_RANGE;
        }
    }

    // ------------------------------------------------------------ main loop

    fn main_loop(&mut self, mine: Vec<C>, s: &Snapshot) -> Option<Pt> {
        let now = self.now;
        if mine.len() > self.cell_count {
            self.broke_at = now;
        }
        self.cell_count = mine.len();
        let mut view = self.survey(mine, s);
        self.track_motion(&mut view);
        self.merge_left = self.merge_in(&view, now);
        self.bait_id = self.bait_target(&view, now);
        self.being_chased = self.chased(&view, now);
        let pressure = self.score(&mut view);
        self.rank();
        if let Some(o) = self.plan_pop(&view, now) {
            return Some(o);
        }
        if let Some(o) = self.plan_escape(&view, now) {
            return Some(o);
        }
        if let Some(o) = self.plan_lure(&view, now) {
            return Some(o);
        }
        if let Some(o) = self.plan_hide(&view) {
            return Some(o);
        }
        if let Some(o) = self.plan_pour(&view, now, pressure) {
            return Some(o);
        }
        if let Some(o) = self.plan_split(&view, now, pressure) {
            return Some(o);
        }
        Some(self.steer(&view, pressure))
    }

    fn our_speed(&self, view: &View) -> f64 {
        let mut fastest = 0.0;
        for cell in &view.mine {
            let speed = self.speed_of(cell.id);
            if speed > fastest {
                fastest = speed;
            }
        }
        fastest
    }

    fn merge_in(&self, view: &View, now: f64) -> f64 {
        if view.mine.len() < 2 {
            return 0.0;
        }
        f64::max(0.0, self.broke_at + merge_delay(view.big_mass) - now)
    }

    fn bait_target(&self, view: &View, now: f64) -> Option<u32> {
        if view.mine.len() < 2 || !self.aggressive {
            return None;
        }
        if self.merge_in(view, now) > BAIT_WINDOW {
            return None;
        }
        let mut best: Option<&E> = None;
        let mut best_distance = INF;
        for &ti in &view.threats {
            let threat = &view.ents[ti];
            if threat.id.is_none() {
                continue;
            }
            if threat.grown * BAIT_MARGIN > view.total {
                continue;
            }
            if threat.grown < view.small_mass * THREAT_RATIO {
                continue;
            }
            let distance = span(threat.x, threat.y, view.centre.0, view.centre.1);
            if distance > BAIT_RANGE || distance >= best_distance {
                continue;
            }
            best = Some(threat);
            best_distance = distance;
        }
        best.and_then(|b| b.id)
    }

    fn rank(&mut self) {
        let t = tables();
        let _ = t;
        let mut best = 0usize;
        let mut best_value = -INF;
        let mut low = INF;
        for i in 0..DIRECTIONS {
            self.value[i] = self.gain[i] + PREY_GAIN * self.prey_gain[i] - self.risk[i];
            if self.value[i] > best_value {
                best_value = self.value[i];
                best = i;
            }
            if self.value[i] < low {
                low = self.value[i];
            }
        }
        if let Some(h) = self.heading {
            let mut steps = (best as i64 - h as i64).abs() as f64;
            if steps > DIRECTIONS as f64 / 2.0 {
                steps = DIRECTIONS as f64 - steps;
            }
            let turning = (1.0 - (steps * DIR_STEP).cos()) / 2.0;
            let margin = TURN_MARGIN + (TURN_MARGIN_SHARE + TURN_REVERSAL * turning) * (best_value - low);
            if self.value[h] >= best_value - margin {
                best = h;
            }
        }
        self.heading = Some(best);
    }

    fn steer(&mut self, view: &View, pressure: f64) -> Pt {
        let t = tables();
        let best = self.heading.unwrap();
        let chasing = PREY_GAIN * self.prey_gain[best] > self.gain[best];
        if self.bait_id.is_some() {
            self.state = "bait";
            self.counters.bump("baits", 1);
        } else if pressure > FLEE_PRESSURE {
            self.state = "flee";
        } else if chasing {
            self.state = "hunt";
        } else if self.repositioning {
            self.state = "roam";
        } else if self.gain[best] > 0.0 {
            self.state = "feed";
        } else {
            self.state = "roam";
        }
        let stride = self.stride(view, pressure);
        let target = self.commit(view, t.dx[best], t.dy[best]);
        match target {
            None => {
                let (idx, reach) = self.clear_heading(view, best, stride);
                self.point(view, t.dx[idx], t.dy[idx], reach)
            }
            Some(target) => {
                let mut reach = f64::min(target.distance, stride);
                if !self.merging(view) {
                    reach = f64::max(reach, floor_along(view, target.ux, target.uy));
                }
                self.point(view, target.ux, target.uy, reach)
            }
        }
    }

    fn commit(&mut self, view: &View, ux: f64, uy: f64) -> Option<Pick> {
        let mut best: Option<Pick> = None;
        let mut best_value = 0.0;
        let mut held: Option<Pick> = None;
        let mut held_value = 0.0;
        let (cx, cy) = view.centre;
        let target_id = self.target_id;

        // consider(id, x, y, worth) -> updates best / held
        let mut cands: Vec<(u32, f64, f64, f64, bool)> = Vec::new(); // id, x, y, worth (lazy marker), is_prey
        let _ = &mut cands;
        let mouthful = mouthfuls(view);

        macro_rules! consider {
            ($id:expr, $x:expr, $y:expr, $worth:expr) => {{
                let (id, x, y): (u32, f64, f64) = ($id, $x, $y);
                let dx = x - cx;
                let dy = y - cy;
                let distance = (dx * dx + dy * dy).sqrt();
                if distance >= 1.0 {
                    let along = (dx * ux + dy * uy) / distance;
                    let keeping = Some(id) == target_id;
                    if !(along < (if keeping { COMMIT_KEEP } else { COMMIT_CONE })) {
                        let raw = $worth / (distance + FOOD_SOFT);
                        if !(!keeping && raw <= best_value) {
                            let shun = shunned(view, x, y);
                            if shun > 0.0 {
                                let value = raw * shun * along.powf(COMMIT_LEAN);
                                if !(!keeping && value <= best_value) {
                                    if !self.crosses(view, x, y) {
                                        let pick = Pick { id, distance, ux: dx / distance, uy: dy / distance };
                                        if keeping {
                                            held = Some(Pick { id, distance, ux: dx / distance, uy: dy / distance });
                                            held_value = value;
                                        }
                                        if value > best_value {
                                            best_value = value;
                                            best = Some(pick);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }};
        }

        for cell in &view.food {
            consider!(cell.id, cell.x, cell.y, mouthful(cell));
        }
        let strike = view.big.size + SPLIT_REACH + STRIKE_MARGIN;
        for &i in &view.prey {
            let entry = &view.ents[i];
            let caught = entry.catch.unwrap_or(1.0);
            if caught < 0.5 {
                continue;
            }
            if span(entry.x, entry.y, view.big.x, view.big.y) > strike {
                continue;
            }
            consider!(entry.id.unwrap(), entry.x, entry.y, entry.mass * PREY_GAIN * caught);
        }
        let pick = if held.is_some() && held_value * COMMIT_SWITCH >= best_value { held } else { best };
        self.target_id = pick.as_ref().map(|p| p.id);
        pick
    }

    fn crosses(&self, view: &View, x: f64, y: f64) -> bool {
        let room = view.mine.len() < MAX_CELLS;
        for cell in &view.mine {
            let mass = cell.mass();
            for &ti in &view.threats {
                let threat = &view.ents[ti];
                if grown_for(threat, cell) < mass * THREAT_RATIO {
                    continue;
                }
                let reach = threat.size + cell.size + THREAT_MARGIN;
                if through(cell, threat.x, threat.y, reach, x, y) {
                    return true;
                }
            }
            if !room || mass <= VIRUS_MASS * EAT_RATIO {
                continue;
            }
            for virus in &view.viruses {
                let reach = virus.size + cell.size;
                if through(cell, virus.x, virus.y, reach, x, y) {
                    return true;
                }
            }
        }
        false
    }

    fn stride(&self, view: &View, pressure: f64) -> f64 {
        let floor = view.big.size + CLEAR_MARGIN;
        if self.merging(view) {
            return view.spread * MERGE_PULL;
        }
        let gather = pressure > GATHER_PRESSURE || self.state == "bait";
        if view.mine.len() > 1 && gather {
            return f64::max(view.spread + view.big.size, floor);
        }
        f64::max(self.vd * 0.5, floor)
    }

    fn merging(&self, view: &View) -> bool {
        view.mine.len() > 1 && self.merge_left <= 0.0
    }

    fn clear_heading(&mut self, view: &View, best: usize, stride: f64) -> (usize, f64) {
        let reach = self.clear_reach(view, best, stride);
        if reach > 0.0 {
            self.detour = None;
            return (best, reach);
        }
        if let Some(d) = self.detour {
            let held = self.clear_reach(view, d, stride);
            if held > 0.0 {
                return (d, held);
            }
        }
        let side = self.side_of(best);
        for step in 1..=(DIRECTIONS / 2) {
            let up = (best + step) % DIRECTIONS;
            let down = (best + DIRECTIONS - step) % DIRECTIONS;
            let mut first = if self.value[down] > self.value[up] { down } else { up };
            if side < 0 {
                first = down;
            } else if side > 0 {
                first = up;
            }
            let second = if first == up { down } else { up };
            for i in [first, second] {
                let got = self.clear_reach(view, i, stride);
                if got <= 0.0 {
                    continue;
                }
                self.detour = Some(i);
                return (i, got);
            }
        }
        (best, stride)
    }

    fn side_of(&self, best: usize) -> i32 {
        match self.detour {
            None => 0,
            Some(d) => {
                let mut step = d as i64 - best as i64;
                if step > DIRECTIONS as i64 / 2 {
                    step -= DIRECTIONS as i64;
                } else if step < -(DIRECTIONS as i64) / 2 {
                    step += DIRECTIONS as i64;
                }
                if step > 0 {
                    1
                } else if step < 0 {
                    -1
                } else {
                    0
                }
            }
        }
    }

    fn clear_reach(&mut self, view: &View, index: usize, stride: f64) -> f64 {
        let t = tables();
        let far = self.point(view, t.dx[index], t.dy[index], stride);
        if !self.crosses(view, far.0, far.1) {
            return stride;
        }
        let floor = floor_along(view, t.dx[index], t.dy[index]);
        if floor >= stride {
            return 0.0;
        }
        let near = self.point(view, t.dx[index], t.dy[index], floor);
        if !self.crosses(view, near.0, near.1) {
            return floor;
        }
        0.0
    }

    fn point(&mut self, view: &View, ux: f64, uy: f64, distance: f64) -> Pt {
        let mut x = view.centre.0 + ux * distance;
        let mut y = view.centre.1 + uy * distance;
        if self.known() {
            let b = self.border;
            let (min_x, max_x, min_y, max_y) = (b.left, b.right, b.top, b.bottom);
            let low = MAP_MARGIN;
            let cell = view.big;
            let touch = cell.size + WALL_TOUCH;
            let block_x = (x < min_x + low && cell.x - min_x < touch) || (x > max_x - low && max_x - cell.x < touch);
            let block_y = (y < min_y + low && cell.y - min_y < touch) || (y > max_y - low && max_y - cell.y < touch);
            if block_x && !block_y {
                if uy.abs() >= SLIDE_LEAN {
                    self.slide_y = if uy < 0.0 { -1.0 } else { 1.0 };
                }
                x = view.centre.0;
                y = view.centre.1 + self.slide_y * distance;
            } else if block_y && !block_x {
                if ux.abs() >= SLIDE_LEAN {
                    self.slide_x = if ux < 0.0 { -1.0 } else { 1.0 };
                }
                x = view.centre.0 + self.slide_x * distance;
                y = view.centre.1;
            }
            x = f64::min(f64::max(x, min_x + low), max_x - low);
            y = f64::min(f64::max(y, min_y + low), max_y - low);
        }
        (x, y)
    }

    // ------------------------------------------------------------- tactics

    fn plan_escape(&mut self, view: &View, now: f64) -> Option<Pt> {
        if self.contact < ESCAPE_CONTACT {
            return None;
        }
        if now - self.split_at < SPLIT_COOLDOWN {
            return None;
        }
        if view.mine.len() >= MAX_CELLS {
            return None;
        }
        if view.big_mass < SPLIT_MIN_MASS * 2.0 {
            return None;
        }
        let t = tables();
        let cell = view.big;
        let mass = cell.mass();
        let known = self.known();
        let b = self.border;
        let (min_x, min_y, max_x, max_y) = (b.left, b.top, b.right, b.bottom);
        let mut pressing: Option<&E> = None;
        let mut closest = INF;
        for &ti in &view.threats {
            let threat = &view.ents[ti];
            if grown_for(threat, &cell) < mass * THREAT_RATIO {
                continue;
            }
            let gap = span(threat.x, threat.y, cell.x, cell.y) - threat.size;
            if gap < closest {
                closest = gap;
                pressing = Some(threat);
            }
        }
        let pressing = pressing?;
        let away_x = cell.x - pressing.x;
        let away_y = cell.y - pressing.y;
        let away = or1((away_x * away_x + away_y * away_y).sqrt());
        let ux = away_x / away;
        let uy = away_y / away;
        let ours = f64::max(self.our_speed(view), walk_speed(cell.size));
        let flying = self.closing_on(pressing, &cell) > ours * ESCAPE_LOSING;
        let grown = grown_for(pressing, &cell);
        let splits = grown > mass * SPLIT_THREAT_RATIO && grown >= SPLIT_MIN_MASS;
        if !splits && !flying {
            if self.run_open(view, &cell) {
                return None;
            }
        } else {
            let swallowed = flying && closest < cell.size * ESCAPE_SWALLOW;
            let mut boxed = true;
            for i in 0..DIRECTIONS {
                if t.dx[i] * ux + t.dy[i] * uy <= ESCAPE_AWAY {
                    continue;
                }
                if self.room_along(cell.x, cell.y, cell.size, t.dx[i], t.dy[i]) > SPLIT_REACH {
                    boxed = false;
                    break;
                }
            }
            if !boxed && !swallowed {
                return None;
            }
        }
        let half = size_of(view.big_mass / 2.0);
        let mut best: Option<usize> = None;
        let mut best_room = 0.0;
        for i in 0..DIRECTIONS {
            if t.dx[i] * -ux + t.dy[i] * -uy > ESCAPE_INTO {
                continue;
            }
            let x = cell.x + t.dx[i] * SPLIT_REACH;
            let y = cell.y + t.dy[i] * SPLIT_REACH;
            if known && (x < min_x + half || x > max_x - half || y < min_y + half || y > max_y - half) {
                continue;
            }
            let mut worst = INF;
            for &ti in &view.threats {
                let threat = &view.ents[ti];
                let gap = span(threat.x, threat.y, x, y) - threat.size - half - ESCAPE_CLEAR;
                if gap < worst {
                    worst = gap;
                }
            }
            if worst <= 0.0 {
                continue;
            }
            if worst > best_room {
                best_room = worst;
                best = Some(i);
            }
        }
        let best = best?;
        let at = self.point(view, t.dx[best], t.dy[best], SPLIT_REACH);
        let dx = at.0 - view.centre.0;
        let dy = at.1 - view.centre.1;
        let length = or1((dx * dx + dy * dy).sqrt());
        let aim_x = self.point_x - view.centre.0;
        let aim_y = self.point_y - view.centre.1;
        let aim_length = or1((aim_x * aim_x + aim_y * aim_y).sqrt());
        let aligned = (dx * aim_x + dy * aim_y) / (length * aim_length) > SPLIT_ALIGN;
        if aligned {
            self.do_split = true;
            self.split_at = now;
            self.state = "escape";
            self.counters.bump("escapes", 1);
        } else {
            self.state = "flee";
        }
        Some(self.point(view, t.dx[best], t.dy[best], SPLIT_REACH))
    }

    fn run_open(&self, view: &View, cell: &C) -> bool {
        let t = tables();
        let mass = cell.mass();
        for i in 0..DIRECTIONS {
            let room = self.room_along(cell.x, cell.y, cell.size, t.dx[i], t.dy[i]);
            if room <= SPLIT_REACH {
                continue;
            }
            let mut cut = false;
            for &ti in &view.threats {
                let threat = &view.ents[ti];
                if grown_for(threat, cell) < mass * THREAT_RATIO {
                    continue;
                }
                if cuts_off(threat, cell, t.dx[i], t.dy[i], room) {
                    cut = true;
                    break;
                }
            }
            if !cut {
                return true;
            }
        }
        false
    }

    fn plan_split(&mut self, view: &View, now: f64, pressure: f64) -> Option<Pt> {
        if !self.aggressive {
            return None;
        }
        if pressure > AGGRO_PRESSURE {
            return None;
        }
        if now - self.split_at < SPLIT_COOLDOWN {
            return None;
        }
        if view.mine.len() > SPLIT_MAX_CELLS {
            return None;
        }
        let cell = view.big;
        let mass = view.big_mass;
        if mass < SPLIT_MIN_MASS * 2.0 {
            return None;
        }
        let half = size_of(mass / 2.0);
        let rng = half + SPLIT_REACH * SPLIT_TRUST;
        let ours = self.our_speed(view);
        let mut best: Option<Pt> = None;
        let mut best_score = 0.0;
        for &i in &view.prey {
            let entry = &view.ents[i];
            if mass / 2.0 < entry.grown * SPLIT_KILL_RATIO {
                continue;
            }
            let (tx, ty) = lead(entry, &cell);
            let distance = span(tx, ty, cell.x, cell.y);
            if distance > rng {
                continue;
            }
            if ours > 0.0
                && self.speed_of(entry.id.unwrap()) < ours * SPLIT_CHASE
                && distance - cell.size < ours * SPLIT_WALK
            {
                continue;
            }
            if self.crosses(view, tx, ty) {
                continue;
            }
            if !split_is_safe(view, &cell, tx, ty) {
                continue;
            }
            let haul = haul(view, i, tx, ty, half, mass);
            if haul < mass * SPLIT_WORTH {
                continue;
            }
            let score = haul / (distance + 200.0);
            if score > best_score {
                best_score = score;
                best = Some((tx, ty));
            }
        }
        let best = best?;
        let dx = best.0 - cell.x;
        let dy = best.1 - cell.y;
        let length = or1((dx * dx + dy * dy).sqrt());
        let aim_x = self.point_x - cell.x;
        let aim_y = self.point_y - cell.y;
        let aim_length = or1((aim_x * aim_x + aim_y * aim_y).sqrt());
        let aligned = (dx * aim_x + dy * aim_y) / (length * aim_length) > SPLIT_ALIGN;
        if aligned {
            self.do_split = true;
            self.split_at = now;
            self.state = "split";
        } else {
            self.state = "hunt";
        }
        Some((cell.x + (dx / length) * rng, cell.y + (dy / length) * rng))
    }

    // ---------------------------------------------------------- virus feeds

    fn plan_pop(&mut self, view: &View, now: f64) -> Option<Pt> {
        if !self.aggressive || view.mine.len() > 1 {
            self.feeding = None;
            self.staging = None;
            return None;
        }
        if self.feeding.is_some() {
            if let Some(h) = self.hold_pop(view, now) {
                return Some(h);
            }
            self.feeding = None;
        }
        let limit = if self.staging.is_some() { POP_CONTACT_HOLD } else { POP_CONTACT };
        if self.contact > limit
            || view.total < POP_MIN_MASS + VIRUS_FEED as f64 * EJECT_COST
            || view.big_mass < SPLIT_MIN_MASS * 2.0
        {
            self.rest_stage(now);
            return None;
        }
        if self.staging.is_some() {
            if let Some(w) = self.hold_station(view, now) {
                return Some(w);
            }
            self.rest_stage(now);
        }
        if now - self.pop_scan < POP_SCAN {
            return None;
        }
        self.pop_scan = now;
        if let Some((vi, ei)) = self.find_pop(view, now) {
            let (vid, eid) = (view.viruses[vi].id, view.ents[ei].id.unwrap());
            self.feeding = Some(Feeding { virus: vid, enemy: eid, fed: self.fed_into(vid, now), last: 0.0, since: now });
            self.counters.bump("shots_started", 1);
            return self.hold_pop(view, now);
        }
        let (vi, ei) = self.find_station(view, now)?;
        self.staging = Some(Staging { virus: view.viruses[vi].id, enemy: view.ents[ei].id.unwrap(), since: now });
        if let Some(w) = self.hold_station(view, now) {
            return Some(w);
        }
        self.rest_stage(now);
        None
    }

    fn rest_stage(&mut self, now: f64) {
        if let Some(st) = self.staging {
            self.stage_rest.insert(st.enemy, now + POP_STAGE_REST);
        }
        self.staging = None;
        self.feeding = None;
    }

    fn pop_worthy(&mut self, view: &View, enemy: &E, now: f64) -> bool {
        let id = match enemy.id {
            Some(id) => id,
            None => return false,
        };
        if enemy.mass < POP_MIN_ENEMY {
            return false;
        }
        if view.big_mass / 2.0 >= enemy.grown * SPLIT_KILL_RATIO {
            return false;
        }
        match self.stage_rest.get(&id) {
            None => true,
            Some(&rest) => {
                if now < rest {
                    return false;
                }
                self.stage_rest.remove(&id);
                true
            }
        }
    }

    fn hold_station(&mut self, view: &View, now: f64) -> Option<Pt> {
        let staging = self.staging.unwrap();
        if now - staging.since > POP_STAGE_TIMEOUT {
            return None;
        }
        let virus = *view.viruses.iter().find(|c| c.id == staging.virus)?;
        let enemy = view.ents[..view.n_enemies].iter().find(|e| e.id == Some(staging.enemy))?;
        let cell = view.big;
        if pop_line(&cell, &virus, enemy).is_some() {
            self.feeding = Some(Feeding { virus: virus.id, enemy: staging.enemy, fed: self.fed_into(virus.id, now), last: 0.0,
                                          since: now });
            self.staging = None;
            return self.hold_pop(view, now);
        }
        if span(virus.x, virus.y, enemy.x, enemy.y) > POP_RANGE {
            return None;
        }
        let stand = cell.size + virus.size + POP_STANDOFF;
        if stand > EJECT_REACH {
            return None;
        }
        let here = (cell.y - virus.y).atan2(cell.x - virus.x);
        let want = (virus.y - enemy.y).atan2(virus.x - enemy.x);
        let mut turn = want - here;
        while turn > std::f64::consts::PI {
            turn -= std::f64::consts::PI * 2.0;
        }
        while turn < -std::f64::consts::PI {
            turn += std::f64::consts::PI * 2.0;
        }
        if turn > POP_ARC {
            turn = POP_ARC;
        } else if turn < -POP_ARC {
            turn = -POP_ARC;
        }
        for share in [1.0, 0.5, 0.25] {
            let angle = here + turn * share;
            let x = virus.x + angle.cos() * stand;
            let y = virus.y + angle.sin() * stand;
            if shunned(view, x, y) <= 0.0 {
                continue;
            }
            if self.crosses(view, x, y) {
                continue;
            }
            self.state = "stage";
            return Some((x, y));
        }
        None
    }

    fn find_station(&mut self, view: &View, now: f64) -> Option<(usize, usize)> {
        let cell = view.big;
        let mut best: Option<(usize, usize)> = None;
        let mut best_score = 0.0;
        for (vi, virus) in view.viruses.iter().enumerate() {
            let walk = span(virus.x, virus.y, cell.x, cell.y);
            if walk > POP_WALK {
                continue;
            }
            if cell.size + virus.size + POP_STANDOFF > EJECT_REACH {
                continue;
            }
            if self.dud(virus, now) {
                continue;
            }
            for ei in 0..view.n_enemies {
                let enemy = &view.ents[ei];
                if !self.pop_worthy(view, enemy, now) {
                    continue;
                }
                let to_enemy = span(virus.x, virus.y, enemy.x, enemy.y);
                if to_enemy > POP_RANGE || to_enemy < enemy.size {
                    continue;
                }
                let score = enemy.mass / (walk + to_enemy + 400.0);
                if score > best_score {
                    best_score = score;
                    best = Some((vi, ei));
                }
            }
        }
        best
    }

    fn find_pop(&mut self, view: &View, now: f64) -> Option<(usize, usize)> {
        let cell = view.big;
        let mut best: Option<(usize, usize)> = None;
        let mut best_align = POP_ALIGN;
        for (vi, virus) in view.viruses.iter().enumerate() {
            if self.dud(virus, now) {
                continue;
            }
            for ei in 0..view.n_enemies {
                let enemy = &view.ents[ei];
                if !self.pop_worthy(view, enemy, now) {
                    continue;
                }
                let line = match pop_line(&cell, virus, enemy) {
                    Some(l) => l,
                    None => continue,
                };
                if line.2 <= best_align {
                    continue;
                }
                best_align = line.2;
                best = Some((vi, ei));
            }
        }
        best
    }

    fn hold_pop(&mut self, view: &View, now: f64) -> Option<Pt> {
        let mut feeding = self.feeding.unwrap();
        if now - feeding.since > POP_TIMEOUT {
            return None;
        }
        if feeding.fed >= VIRUS_FEED {
            return None;
        }
        if self.contact > POP_CONTACT_HOLD {
            return None;
        }
        if view.total < POP_MIN_MASS {
            return None;
        }
        let virus = *view.viruses.iter().find(|c| c.id == feeding.virus)?;
        let enemy = view.ents[..view.n_enemies].iter().find(|e| e.id == Some(feeding.enemy))?;
        let cell = view.big;
        let line = pop_line(&cell, &virus, enemy)?;
        let aim_x = self.point_x - cell.x;
        let aim_y = self.point_y - cell.y;
        let aim_length = or1((aim_x * aim_x + aim_y * aim_y).sqrt());
        let aimed = (aim_x * line.0 + aim_y * line.1) / aim_length >= POP_ALIGN;
        if aimed && now - feeding.last >= EJECT_INTERVAL {
            self.do_eject = true;
            feeding.fed += 1;
            feeding.last = now;
            self.fed_viruses.insert(virus.id, (feeding.fed, now));
            if feeding.fed >= VIRUS_FEED {
                self.counters.bump("shots_fired", 1);
            }
        }
        self.feeding = Some(feeding);
        self.state = "pop";
        let hold = cell.size * 0.9;
        Some((cell.x + line.0 * hold, cell.y + line.1 * hold))
    }

    fn fed_into(&mut self, virus_id: u32, now: f64) -> i32 {
        match self.fed_viruses.get(&virus_id) {
            None => 0,
            Some(&(fed, at)) => {
                if now - at > FEED_MEMORY {
                    self.fed_viruses.remove(&virus_id);
                    0
                } else {
                    fed
                }
            }
        }
    }

    fn dud(&mut self, virus: &C, now: f64) -> bool {
        self.fed_into(virus.id, now) >= VIRUS_FEED
    }

    fn nearest_virus(&self, view: &View, rng: f64, ok: Option<&dyn Fn(&C) -> bool>) -> Option<C> {
        let cell = view.big;
        let mut best: Option<C> = None;
        let mut best_distance = rng;
        for virus in &view.viruses {
            let distance = span(virus.x, virus.y, cell.x, cell.y);
            if distance >= best_distance {
                continue;
            }
            if let Some(f) = ok {
                if !f(virus) {
                    continue;
                }
            }
            if self.crosses(view, virus.x, virus.y) {
                continue;
            }
            best = Some(*virus);
            best_distance = distance;
        }
        best
    }

    fn plan_lure(&mut self, view: &View, now: f64) -> Option<Pt> {
        if !self.aggressive || view.mine.len() != 1 || now < self.lure_rest {
            self.luring = None;
            return None;
        }
        let cell = view.big;
        let big_mass = view.big_mass;
        let fits = move |virus: &C| cell.size >= virus.size && big_mass <= virus.mass() * EAT_RATIO;
        let mut virus: Option<C> = None;
        if let Some(l) = self.luring {
            if now - l.since > LURE_TIMEOUT {
                self.lure_rest = now + LURE_REST;
                self.luring = None;
            } else {
                virus = view.viruses.iter().find(|v| v.id == l.virus).copied();
                if virus.is_none() {
                    self.luring = None;
                }
            }
        }
        if virus.is_none() {
            virus = self.nearest_virus(view, LURE_WALK, Some(&fits));
        }
        let virus = match virus {
            Some(v) if fits(&v) => v,
            _ => {
                self.luring = None;
                return None;
            }
        };
        let bite = virus.mass() * EAT_RATIO;
        let mut mark: Option<&E> = None;
        let mut closest = LURE_RANGE;
        for &ti in &view.threats {
            let threat = &view.ents[ti];
            if threat.mass <= bite {
                continue;
            }
            if self.closing_on(threat, &cell) < LURE_CLOSING {
                continue;
            }
            let gap = span(threat.x, threat.y, cell.x, cell.y) - threat.size - cell.size;
            if gap >= closest {
                continue;
            }
            mark = Some(threat);
            closest = gap;
        }
        let mark = match mark {
            Some(m) => m,
            None => {
                self.luring = None;
                return None;
            }
        };
        if self.luring.is_none() {
            self.luring = Some(Luring { virus: virus.id, since: now });
            self.counters.bump("lures", 1);
        }
        self.state = "lure";
        if closest < LURE_BAIL {
            let dx = virus.x - mark.x;
            let dy = virus.y - mark.y;
            let away = or1((dx * dx + dy * dy).sqrt());
            let step = virus.size + cell.size + LURE_CLEAR;
            return Some((virus.x + (dx / away) * step, virus.y + (dy / away) * step));
        }
        Some((virus.x, virus.y))
    }

    fn plan_hide(&mut self, view: &View) -> Option<Pt> {
        if view.big_mass > VIRUS_MASS * EAT_RATIO {
            return None;
        }
        if !self.being_chased {
            return None;
        }
        for &ti in &view.threats {
            let threat = &view.ents[ti];
            for cell in &view.mine {
                if grown_for(threat, cell) < cell.mass() * THREAT_RATIO {
                    continue;
                }
                if threat.mass <= VIRUS_MASS * EAT_RATIO {
                    return None;
                }
            }
        }
        let best = self.nearest_virus(view, HIDE_RANGE, None)?;
        self.state = "hide";
        self.counters.bump("virus_hides", 1);
        Some((best.x, best.y))
    }

    fn plan_pour(&mut self, view: &View, now: f64, pressure: f64) -> Option<Pt> {
        if !self.aggressive
            || view.mine.len() < 3
            || self.merge_left < POUR_MERGE_FAR
            || now < self.pour_rest
            || pressure > POUR_PRESSURE
            || self.contact > 0.0
        {
            self.pouring = None;
            return None;
        }
        let big = view.big;
        let mut pour = 0.0;
        for cell in &view.mine {
            if cell.id == big.id {
                continue;
            }
            let mass = cell.mass();
            if mass <= POUR_KEEP {
                continue;
            }
            let gap = span(cell.x, cell.y, big.x, big.y) - cell.size - big.size;
            if gap > POUR_REACH {
                continue;
            }
            pour += (mass - POUR_KEEP) * FEED_SHARE;
        }
        if pour < view.big_mass * POUR_WORTH {
            self.pouring = None;
            return None;
        }
        if pour_for(view, view.big_mass + pour).is_none() {
            self.pouring = None;
            return None;
        }
        if self.pouring.is_none() {
            self.pouring = Some(Pouring { since: now, last: 0.0 });
            self.counters.bump("pours", 1);
        }
        let mut pouring = self.pouring.unwrap();
        if now - pouring.since > POUR_TIMEOUT {
            self.pouring = None;
            self.pour_rest = now + POUR_REST;
            return None;
        }
        let on_big = span(self.point_x, self.point_y, big.x, big.y) < big.size;
        if on_big && now - pouring.last >= EJECT_INTERVAL {
            self.do_eject = true;
            pouring.last = now;
        }
        self.pouring = Some(pouring);
        self.state = "pour";
        Some((big.x, big.y))
    }

    fn chased(&mut self, view: &View, now: f64) -> bool {
        let ours = self.our_speed(view);
        let mut seen: Vec<u32> = Vec::new();
        let mut chased = false;
        for &ti in &view.threats {
            let threat = &view.ents[ti];
            let tid = match threat.id {
                Some(id) => id,
                None => continue,
            };
            let mut closing = false;
            for cell in &view.mine {
                let mass = cell.mass();
                if grown_for(threat, cell) < mass * THREAT_RATIO {
                    continue;
                }
                if self.covered(view, cell, threat) {
                    continue;
                }
                let gap = span(threat.x, threat.y, cell.x, cell.y) - threat.size - cell.size;
                if gap > HUNT_RANGE {
                    continue;
                }
                if ours > 0.0 && self.closing_on(threat, cell) > ours * HUNT_CLOSING {
                    closing = true;
                    break;
                }
            }
            if !closing {
                continue;
            }
            seen.push(tid);
            let since = *self.pursuit.entry(tid).or_insert(now);
            if now - since >= HUNT_SUSTAIN {
                chased = true;
            }
        }
        self.pursuit.retain(|id, _| seen.contains(id));
        if chased {
            self.hunted_at = now;
        }
        chased || (self.hunted_at > 0.0 && now - self.hunted_at < HIDE_HOLD && near_threat(view, HUNT_RANGE))
    }
}

// ------------------------------------------------------------ free helpers

fn foresee(ents: &mut [E], food: &[C]) {
    let ejected: Vec<&C> = food.iter().filter(|c| c.ejected()).collect();
    let mut potentials = vec![0.0; ents.len()];
    for (ei, entry) in ents.iter().enumerate() {
        let mut extra = 0.0;
        let ecolor = entry.cell.unwrap().color;
        for (oi, other) in ents.iter().enumerate() {
            if oi == ei {
                continue;
            }
            if other.cell.unwrap().color == ecolor {
                continue;
            }
            if entry.mass < other.mass * EAT_RATIO {
                continue;
            }
            let gap = span(entry.x, entry.y, other.x, other.y) - entry.size - other.size;
            let reach = if entry.mass >= other.mass * SPLIT_THREAT_RATIO { FORESEE_RANGE + SPLIT_REACH } else { FORESEE_RANGE };
            if gap <= reach {
                extra += other.mass;
            }
        }
        let mut fed = 0.0;
        for cell in &ejected {
            if cell.color == ecolor {
                continue;
            }
            let gap = span(entry.x, entry.y, cell.x, cell.y) - entry.size;
            if gap <= FEED_RANGE {
                fed += cell.mass();
            }
        }
        if fed >= FEED_MIN {
            extra += fed;
            let mut feeder: Option<usize> = None;
            let mut closest = FEEDER_RANGE;
            for (oi, other) in ents.iter().enumerate() {
                if oi == ei {
                    continue;
                }
                if other.cell.unwrap().color == ecolor {
                    continue;
                }
                let gap = span(entry.x, entry.y, other.x, other.y) - entry.size - other.size;
                if gap < closest {
                    closest = gap;
                    feeder = Some(oi);
                }
            }
            if let Some(f) = feeder {
                extra += ents[f].mass * FEED_SHARE;
            }
        }
        potentials[ei] = extra;
    }
    for (e, p) in ents.iter_mut().zip(potentials) {
        e.potential = p;
    }
}

fn cascade(entry: &mut E, ladder: &[Piece]) {
    let mut grown = entry.mass + entry.potential;
    let mut absorbed: Option<Vec<u32>> = None;
    let mut eats = 0.0;
    for piece in ladder {
        if grown < piece.mass * EAT_RATIO {
            break;
        }
        let gap = span(entry.x, entry.y, piece.x, piece.y) - entry.size - piece.size;
        if gap > CASCADE_RANGE {
            continue;
        }
        grown += piece.mass;
        eats += piece.mass;
        absorbed.get_or_insert_with(Vec::new).push(piece.id);
    }
    entry.grown = grown;
    entry.absorbed = absorbed;
    entry.eats = eats;
}

fn grown_for(entry: &E, cell: &C) -> f64 {
    match &entry.absorbed {
        None => entry.grown,
        Some(a) => {
            if !a.contains(&cell.id) {
                entry.grown
            } else {
                entry.grown - cell.mass()
            }
        }
    }
}

fn merge_threats(ents: &mut Vec<E>, threats: &mut Vec<usize>, small_mass: f64, ladder: &[Piece]) {
    let mut order: Vec<u32> = Vec::new();
    let mut groups: HashMap<u32, Vec<C>> = HashMap::new();
    for e in ents.iter() {
        let c = match e.cell {
            Some(c) => c,
            None => continue,
        };
        groups.entry(c.color).or_insert_with(|| {
            order.push(c.color);
            Vec::new()
        }).push(c);
    }
    for color in order {
        let group = &groups[&color];
        if group.len() < 2 {
            continue;
        }
        for i in 0..group.len() {
            for j in (i + 1)..group.len() {
                let a = group[i];
                let b = group[j];
                let reach = a.size + b.size;
                if (a.x - b.x).abs() > reach || (a.y - b.y).abs() > reach {
                    continue;
                }
                if span(a.x, a.y, b.x, b.y) > reach * 0.9 {
                    continue;
                }
                let mass = a.mass() + b.mass();
                let mut entry = E::new((a.x + b.x) / 2.0, (a.y + b.y) / 2.0, size_of(mass), mass, None, None, false);
                cascade(&mut entry, ladder);
                if entry.grown <= small_mass * THREAT_RATIO {
                    continue;
                }
                ents.push(entry);
                threats.push(ents.len() - 1);
            }
        }
    }
}

fn ahead_of(threat: &E) -> Ahead {
    if threat.vx == 0.0 && threat.vy == 0.0 {
        return Ahead { x: threat.x, y: threat.y, vx: threat.vx, vy: threat.vy };
    }
    Ahead { x: threat.x + threat.vx * THREAT_LOOKAHEAD, y: threat.y + threat.vy * THREAT_LOOKAHEAD, vx: threat.vx, vy: threat.vy }
}

#[allow(clippy::too_many_arguments)]
fn sweep(risk: &mut Field, cell: &C, tx: f64, ty: f64, tvx: f64, tvy: f64, radius: f64, buffer: f64, weight: f64, horizon: f64,
         nearer: f64) {
    let t = tables();
    let rx = tx - cell.x;
    let ry = ty - cell.y;
    let distance = (rx * rx + ry * ry).sqrt();
    if distance > horizon + radius + buffer {
        return;
    }
    if tvx * tvx + tvy * tvy >= FLIGHT_SPEED * FLIGHT_SPEED && rx * tvx + ry * tvy < 0.0 {
        sweep_flight(risk, tvx, tvy, rx, ry, radius, buffer, weight);
        return;
    }
    let band = radius + buffer;
    for i in 0..DIRECTIONS {
        let along = rx * t.dx[i] + ry * t.dy[i];
        if along <= 0.0 {
            let clear = distance - radius;
            if clear >= buffer {
                continue;
            }
            let close = if clear > 0.0 { 1.0 - clear / buffer } else { 1.0 };
            let leaving = -along / distance;
            risk[i] -= weight * close * close * leaving * ESCAPE_LEAN;
            continue;
        }
        let due = if along > nearer { along - nearer } else { 0.0 };
        if due >= horizon {
            continue;
        }
        let cross = rx * t.dy[i] - ry * t.dx[i];
        let clearance = if cross < 0.0 { -cross } else { cross };
        let gap = clearance - radius;
        if gap >= buffer {
            continue;
        }
        let near = if gap > 0.0 { 1.0 - gap / buffer } else { 1.0 };
        let closing = clamp01((distance - clearance) / band);
        let soon = 1.0 - due / horizon;
        risk[i] += weight * near * near * closing * soon;
    }
}

#[allow(clippy::too_many_arguments)]
fn sweep_flight(risk: &mut Field, tvx: f64, tvy: f64, rx: f64, ry: f64, radius: f64, buffer: f64, weight: f64) {
    let t = tables();
    let speed = (tvx * tvx + tvy * tvy).sqrt();
    let vx = tvx / speed;
    let vy = tvy / speed;
    let clear = (rx * vy - ry * vx).abs() - radius;
    if clear >= buffer {
        return;
    }
    let close = if clear > 0.0 { 1.0 - clear / buffer } else { 1.0 };
    let toward = if ry * vx - rx * vy > 0.0 { 1.0 } else { -1.0 };
    for i in 0..DIRECTIONS {
        let along = t.dx[i] * vx + t.dy[i] * vy;
        let across = (t.dy[i] * vx - t.dx[i] * vy) * toward;
        let mut cost = along * along;
        if across > 0.0 {
            cost += FLIGHT_CROSS * across * across;
        }
        risk[i] += weight * close * cost;
    }
}

fn push_wall(risk: &mut Field, gap: f64, ux: f64, uy: f64, near: f64) {
    if gap >= near {
        return;
    }
    let t = tables();
    let crowding = if gap > 0.0 { 1.0 - gap / near } else { 1.0 };
    let weight = WALL_SHOVE * crowding * crowding;
    for i in 0..DIRECTIONS {
        let dot = t.dx[i] * ux + t.dy[i] * uy;
        if dot > 0.0 {
            risk[i] += weight * dot * dot;
        }
    }
}

fn spread(field: &mut Field, ux: f64, uy: f64, value: f64) {
    let t = tables();
    let b = js_round(uy.atan2(ux) / DIR_STEP);
    for j in -FOOD_CONE..=FOOD_CONE {
        let index = (b + j + DIRECTIONS as i64).rem_euclid(DIRECTIONS as i64) as usize;
        field[index] += value * t.kernel[j.unsigned_abs() as usize];
    }
}

fn nearest_cell(mine: &[C], x: f64, y: f64, prey_mass: Option<f64>) -> Near {
    let mut cell: Option<usize> = None;
    let mut best = INF;
    for (i, own) in mine.iter().enumerate() {
        if let Some(pm) = prey_mass {
            if own.mass() < pm * PREY_RATIO {
                continue;
            }
        }
        let dx = x - own.x;
        let dy = y - own.y;
        let distance = dx * dx + dy * dy;
        if distance < best {
            best = distance;
            cell = Some(i);
        }
    }
    match cell {
        None => Near { cell: None, distance: INF, ux: 0.0, uy: 0.0 },
        Some(i) => {
            let distance = or1(best.sqrt());
            Near { cell, distance, ux: (x - mine[i].x) / distance, uy: (y - mine[i].y) / distance }
        }
    }
}

fn grazers_of(view: &View) -> Vec<(f64, f64, f64, f64, f64)> {
    let mut out = Vec::new();
    for &i in &view.prey {
        let e = &view.ents[i];
        let speed = (e.vx * e.vx + e.vy * e.vy).sqrt();
        if speed <= 0.0 {
            continue;
        }
        out.push((e.x, e.y, e.vx / speed, e.vy / speed, e.size + speed * GRAZE_LEAD));
    }
    out
}

fn grazed(grazers: &[(f64, f64, f64, f64, f64)], cell: &C) -> bool {
    for &(gx, gy, ux, uy, reach) in grazers {
        let dx = cell.x - gx;
        let dy = cell.y - gy;
        if dx * ux + dy * uy <= 0.0 {
            continue;
        }
        if dx * dx + dy * dy < reach * reach {
            return true;
        }
    }
    false
}

fn loaded(view: &View, virus: &C) -> Option<(f64, f64)> {
    let mut fx = 0.0;
    let mut fy = 0.0;
    let mut feeder: Option<&E> = None;
    for cell in &view.food {
        if !cell.ejected() || cell.color == view.big.color {
            continue;
        }
        let gap = span(cell.x, cell.y, virus.x, virus.y) - virus.size;
        if gap > FEED_RANGE {
            continue;
        }
        fx += virus.x - cell.x;
        fy += virus.y - cell.y;
        for e in &view.ents[..view.n_enemies] {
            if e.cell.unwrap().color != cell.color {
                continue;
            }
            if span(e.x, e.y, virus.x, virus.y) > EJECT_REACH {
                continue;
            }
            feeder = Some(e);
        }
    }
    if fx == 0.0 && fy == 0.0 {
        return None;
    }
    if let Some(f) = feeder {
        fx = virus.x - f.x;
        fy = virus.y - f.y;
    }
    let length = or1((fx * fx + fy * fy).sqrt());
    Some((fx / length, fy / length))
}

fn shelters(view: &View) -> bool {
    view.threats.iter().any(|&ti| view.ents[ti].mass > VIRUS_MASS * EAT_RATIO)
}

fn mouthfuls(view: &View) -> impl Fn(&C) -> f64 + '_ {
    let radius = f64::max(view.big.size, 1.0);
    let mut buckets: HashMap<(i64, i64), Vec<&C>> = HashMap::new();
    for cell in &view.food {
        buckets.entry(((cell.x / radius).floor() as i64, (cell.y / radius).floor() as i64)).or_default().push(cell);
    }
    let reach = radius * radius;
    move |cell: &C| {
        let bx = (cell.x / radius).floor() as i64;
        let by = (cell.y / radius).floor() as i64;
        let mut mass = 0.0;
        for i in [-1i64, 0, 1] {
            for j in [-1i64, 0, 1] {
                if let Some(bucket) = buckets.get(&(bx + i, by + j)) {
                    for other in bucket {
                        let dx = other.x - cell.x;
                        let dy = other.y - cell.y;
                        if dx * dx + dy * dy <= reach {
                            mass += other.mass();
                        }
                    }
                }
            }
        }
        mass
    }
}

fn shunned(view: &View, x: f64, y: f64) -> f64 {
    let mut shun: f64 = 1.0;
    for &(zx, zy, r2, jump2, appetite) in &view.zones {
        let dx = x - zx;
        let dy = y - zy;
        let reach = dx * dx + dy * dy;
        if reach < r2 {
            return 0.0;
        }
        if jump2 != 0.0 && reach < jump2 {
            shun = f64::min(shun, 1.0 - (1.0 - SPLIT_SHUN) * appetite);
        }
    }
    shun
}

fn through(cell: &C, tx: f64, ty: f64, reach: f64, x: f64, y: f64) -> bool {
    let gap = distance_to_segment(tx, ty, cell.x, cell.y, x, y);
    if gap >= reach {
        return false;
    }
    let now = span(tx, ty, cell.x, cell.y);
    if now >= reach {
        return true;
    }
    gap < now - 1.0
}

fn floor_along(view: &View, ux: f64, uy: f64) -> f64 {
    let mut floor = 0.0;
    for cell in &view.mine {
        let along = (cell.x - view.centre.0) * ux + (cell.y - view.centre.1) * uy;
        let edge = along + cell.size + CLEAR_MARGIN;
        if edge > floor {
            floor = edge;
        }
    }
    floor
}

fn cuts_off(threat: &E, cell: &C, hx: f64, hy: f64, room: f64) -> bool {
    let mouth = f64::max(0.0, threat.size - cell.size / 3.0);
    let pace = walk_speed(threat.size) / walk_speed(cell.size);
    let run = f64::min(room, ESCAPE_REACH);
    let mut x = cell.x;
    let mut y = cell.y;
    let mut tx = threat.x;
    let mut ty = threat.y;
    let mut s = 0.0;
    while s <= run {
        let dx = x - tx;
        let dy = y - ty;
        let gap = (dx * dx + dy * dy).sqrt();
        if gap <= mouth {
            return true;
        }
        if pace < 1.0 && (dx * hx + dy * hy) / gap > pace {
            return false;
        }
        x += hx * CHASE_STRIDE;
        y += hy * CHASE_STRIDE;
        tx += (dx / gap) * pace * CHASE_STRIDE;
        ty += (dy / gap) * pace * CHASE_STRIDE;
        s += CHASE_STRIDE;
    }
    false
}

fn lead(entry: &E, cell: &C) -> (f64, f64) {
    if entry.vx == 0.0 && entry.vy == 0.0 {
        return (entry.x, entry.y);
    }
    let distance = span(entry.x, entry.y, cell.x, cell.y);
    let ahead = f64::min(1000.0, (distance / SPLIT_REACH) * 1000.0);
    (entry.x + entry.vx * ahead, entry.y + entry.vy * ahead)
}

fn haul(view: &View, ei: usize, tx: f64, ty: f64, half: f64, mass: f64) -> f64 {
    let mut total = view.ents[ei].mass;
    for &oi in &view.prey {
        if oi == ei {
            continue;
        }
        let other = &view.ents[oi];
        if mass / 2.0 < other.grown * SPLIT_KILL_RATIO {
            continue;
        }
        if span(other.x, other.y, tx, ty) < half {
            total += other.mass;
        }
    }
    total
}

fn split_is_safe(view: &View, cell: &C, tx: f64, ty: f64) -> bool {
    let half = cell.mass() / 2.0;
    for enemy in &view.ents[..view.n_enemies] {
        if enemy.grown < half * THREAT_RATIO {
            continue;
        }
        let splits = enemy.grown > half * SPLIT_THREAT_RATIO;
        let reach = enemy.size + cell.size + THREAT_MARGIN + SPLIT_CAUTION + (if splits { SPLIT_REACH } else { 0.0 });
        let landing = span(enemy.x, enemy.y, tx, ty);
        let behind = span(enemy.x, enemy.y, cell.x, cell.y);
        if landing < reach || behind < reach {
            return false;
        }
    }
    if half > VIRUS_MASS * EAT_RATIO {
        let dx = tx - cell.x;
        let dy = ty - cell.y;
        let length = or1((dx * dx + dy * dy).sqrt());
        let reach = cell.size + SPLIT_REACH;
        let end_x = cell.x + (dx / length) * reach;
        let end_y = cell.y + (dy / length) * reach;
        for virus in &view.viruses {
            let gap = distance_to_segment(virus.x, virus.y, cell.x, cell.y, end_x, end_y);
            if gap < virus.size + size_of(half) * 0.5 {
                return false;
            }
        }
    }
    true
}

fn pop_line(cell: &C, virus: &C, enemy: &E) -> Option<(f64, f64, f64)> {
    let dvx = virus.x - cell.x;
    let dvy = virus.y - cell.y;
    let to_virus = or1((dvx * dvx + dvy * dvy).sqrt());
    if to_virus > EJECT_REACH {
        return None;
    }
    if to_virus < cell.size + virus.size + 40.0 {
        return None;
    }
    let dex = enemy.x - virus.x;
    let dey = enemy.y - virus.y;
    let to_enemy = or1((dex * dex + dey * dey).sqrt());
    if to_enemy > POP_RANGE {
        return None;
    }
    let gap = span(enemy.x, enemy.y, cell.x, cell.y);
    if gap < enemy.size + cell.size + 300.0 {
        return None;
    }
    let ux = dvx / to_virus;
    let uy = dvy / to_virus;
    let align = ux * (dex / to_enemy) + uy * (dey / to_enemy);
    if align < POP_ALIGN {
        return None;
    }
    Some((ux, uy, align))
}

fn pour_for<'a>(view: &'a View, poured: f64) -> Option<&'a E> {
    let mut best: Option<&E> = None;
    let mut closest = PREY_REACH;
    for entry in &view.ents[..view.n_enemies] {
        let distance = span(entry.x, entry.y, view.big.x, view.big.y);
        if distance >= closest {
            continue;
        }
        let split_now = view.big_mass / 2.0 >= entry.grown * SPLIT_KILL_RATIO;
        let split_then = poured / 2.0 >= entry.grown * SPLIT_KILL_RATIO;
        let eat_now = view.big_mass >= entry.grown * PREY_RATIO;
        let eat_then = poured >= entry.grown * PREY_RATIO;
        if (split_then && !split_now) || (eat_then && !eat_now) {
            best = Some(entry);
            closest = distance;
        }
    }
    best
}

fn near_threat(view: &View, rng: f64) -> bool {
    for &ti in &view.threats {
        let threat = &view.ents[ti];
        for cell in &view.mine {
            let mass = cell.mass();
            if grown_for(threat, cell) < mass * THREAT_RATIO {
                continue;
            }
            let gap = span(threat.x, threat.y, cell.x, cell.y) - threat.size - cell.size;
            if gap <= rng {
                return true;
            }
        }
    }
    false
}

#[allow(dead_code)]
const _PELLET: f64 = PELLET_MASS;
