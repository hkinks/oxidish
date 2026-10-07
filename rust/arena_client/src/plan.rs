//! Lookahead planner ({"kind": "plan"}): a light forward simulation built from the Snapshot only.
//!
//! Every decision step (when a rival player is within reach) it simulates a short horizon for a set of
//! candidate openings (headings x split / no split, plus the base rule bot's own action), each held for
//! `hold` ticks and then continued with the base rule bot's steering (rollout policy), while the rivals
//! move with a cheap greedy rule (flee what can eat them, chase what they can eat, split-kill us when in
//! range). The opening with the best score replaces the base action when it beats the base's own by a margin.

use std::time::Instant;

use crate::game::{Border, Cell, Kind, Snapshot};
use crate::params::Params;
use crate::policy::{heuristic_plan, HeuristicPolicy, Out};
use crate::steer::{steer, tick_speed, EAT_RATIO};

const SIZE_EAT: f64 = 1.15; // eater.size >= 1.15 * victim.size (server rule, on radius)
const VIRUS_EAT: f64 = 1.1576;
const MIN_SPLIT_SIZE: f64 = 60.0;
const MIN_DECAY: f64 = 32.0;
const MAX_CELLS: usize = 16;
const FAR: f64 = 700.0;

#[derive(Clone, Debug)]
pub struct PlanCfg {
    pub horizon: usize,
    pub hold: usize,
    pub headings: usize,
    pub replan: usize,
    pub rival_every: usize,
    pub gate: f64,
    pub margin: f64,
    pub loss_w: f64,
    pub eat_w: f64,
    pub death_w: f64,
    pub near_w: f64,
    pub near_range: f64,
    pub lead_w: f64,
    pub rival_split: bool,
    pub split_cands: usize,
    pub max_rivals: usize,
    pub max_food: usize,
    pub sync: f64,
    pub min_rival_mass: f64,
}

impl Default for PlanCfg {
    fn default() -> PlanCfg {
        PlanCfg {
            horizon: 50,
            hold: 10,
            headings: 16,
            replan: 6,
            rival_every: 4,
            gate: 900.0,
            margin: 0.02,
            loss_w: 0.5,
            eat_w: 0.3,
            death_w: 2.0,
            near_w: 0.5,
            near_range: 400.0,
            lead_w: 0.0,
            rival_split: true,
            split_cands: 3,
            max_rivals: 10,
            max_food: 48,
            sync: 2.0,
            min_rival_mass: 0.0,
        }
    }
}

impl PlanCfg {
    pub fn from_json(v: &serde_json::Value) -> PlanCfg {
        let mut c = PlanCfg::default();
        let f = |k: &str| v.get(k).and_then(|x| x.as_f64());
        macro_rules! set {
            ($($n:ident),*) => { $( if let Some(x) = f(stringify!($n)) { c.$n = x; } )* };
        }
        set!(gate, margin, loss_w, eat_w, death_w, near_w, near_range, lead_w, sync, min_rival_mass);
        macro_rules! seti {
            ($($n:ident),*) => { $( if let Some(x) = f(stringify!($n)) { c.$n = x as usize; } )* };
        }
        seti!(horizon, hold, headings, replan, rival_every, split_cands, max_rivals, max_food);
        if let Some(x) = f("rival_split") {
            c.rival_split = x > 0.5;
        }
        c
    }
}

#[derive(Clone, Copy, Debug)]
struct PC {
    id: u32,
    x: f64,
    y: f64,
    size: f64,
    owner: u8, // 0 = us
    bdx: f64,
    bdy: f64,
    boost: f64,
    vx: f64,
    vy: f64,
    color: u32,
    alive: bool,
}

impl PC {
    fn mass(&self) -> f64 {
        self.size * self.size / 100.0
    }
}

#[derive(Clone)]
struct Sim {
    cells: Vec<PC>,
    viruses: Vec<(u32, f64, f64, f64, bool)>, // id, x, y, size, alive
    food: Vec<(u32, f64, f64, bool)>,
    border: Border,
    mouse: Vec<(f64, f64)>, // per owner
    dir: Vec<(f64, f64)>,   // rival heading
    split_used: Vec<bool>,
    n_owners: usize,
    next_id: u32,
    lost: f64,
    gained: f64,
}

fn unit(dx: f64, dy: f64) -> (f64, f64) {
    let d = dx.hypot(dy);
    if d < 1e-9 {
        (0.0, 0.0)
    } else {
        (dx / d, dy / d)
    }
}

impl Sim {
    fn from_snapshot(s: &Snapshot, cfg: &PlanCfg) -> Sim {
        let mut cells: Vec<PC> = Vec::with_capacity(48);
        for c in &s.mine {
            cells.push(PC { id: c.id, x: c.x, y: c.y, size: c.size, owner: 0, bdx: 0.0, bdy: 0.0, boost: 0.0, vx: 0.0, vy: 0.0, color: c.color, alive: true });
        }
        let (mut cx, mut cy) = (0.0, 0.0);
        for c in &s.mine {
            cx += c.x / s.mine.len() as f64;
            cy += c.y / s.mine.len() as f64;
        }
        // rival players: group by colour; keep the nearest `max_rivals` groups
        let mut groups: Vec<(u32, f64, Vec<&Cell>)> = Vec::new(); // colour, nearest distance, cells
        let mut anon = 0u32;
        for c in s.others.iter().filter(|c| c.kind == Kind::Player) {
            let key = if c.color == 0 {
                anon += 1;
                0x0100_0000 + anon
            } else {
                c.color
            };
            let d = (c.x - cx).hypot(c.y - cy);
            match groups.iter_mut().find(|g| g.0 == key) {
                Some(g) => {
                    g.1 = g.1.min(d);
                    g.2.push(c);
                }
                None => groups.push((key, d, vec![c])),
            }
        }
        groups.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        groups.truncate(cfg.max_rivals);
        let n_owners = groups.len() + 1;
        let mut dir = vec![(0.0, 0.0); n_owners];
        for (gi, g) in groups.iter().enumerate() {
            let owner = (gi + 1) as u8;
            let mut vsum = (0.0, 0.0);
            let mut big = 0.0;
            for c in &g.2 {
                let (vx, vy) = (c.vx / cfg.sync, c.vy / cfg.sync);
                let ts = tick_speed(c.size);
                let sp = vx.hypot(vy);
                let (mut bdx, mut bdy, mut boost) = (0.0, 0.0, 0.0);
                if sp > 1.6 * ts + 2.0 {
                    let (ux, uy) = unit(vx, vy);
                    bdx = ux;
                    bdy = uy;
                    boost = (sp - ts) * 10.0;
                }
                if c.size > big {
                    big = c.size;
                    vsum = (vx, vy);
                }
                cells.push(PC { id: c.id, x: c.x, y: c.y, size: c.size, owner, bdx, bdy, boost, vx, vy, color: c.color, alive: true });
            }
            dir[gi + 1] = unit(vsum.0, vsum.1);
        }
        let mut viruses: Vec<(u32, f64, f64, f64, bool)> = s
            .others
            .iter()
            .filter(|c| c.kind == Kind::Virus && (c.x - cx).hypot(c.y - cy) < 2500.0)
            .map(|c| (c.id, c.x, c.y, c.size, true))
            .collect();
        viruses.truncate(24);
        let mut foods: Vec<(f64, u32, f64, f64)> = s
            .others
            .iter()
            .filter(|c| c.kind == Kind::Food || (c.kind == Kind::Ejected && c.size < 30.0))
            .map(|c| ((c.x - cx).hypot(c.y - cy), c.id, c.x, c.y))
            .filter(|f| f.0 < 900.0)
            .collect();
        foods.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        foods.truncate(cfg.max_food);
        let food = foods.into_iter().map(|f| (f.1, f.2, f.3, true)).collect();
        Sim {
            cells,
            viruses,
            food,
            border: s.border,
            mouse: vec![(cx, cy); n_owners],
            dir,
            split_used: vec![false; n_owners],
            n_owners,
            next_id: 1 << 30,
            lost: 0.0,
            gained: 0.0,
        }
    }

    fn count(&self, owner: u8) -> usize {
        self.cells.iter().filter(|c| c.alive && c.owner == owner).count()
    }

    fn total(&self, owner: u8) -> f64 {
        self.cells.iter().filter(|c| c.alive && c.owner == owner).map(|c| c.mass()).sum()
    }

    fn main_of(&self, owner: u8) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (i, c) in self.cells.iter().enumerate() {
            if c.alive && c.owner == owner && best.map_or(true, |b| c.size > self.cells[b].size) {
                best = Some(i);
            }
        }
        best
    }

    /// Split every cell of `owner` big enough towards (mx, my).
    fn split(&mut self, owner: u8, mx: f64, my: f64) {
        let n0 = self.cells.len();
        let mut count = self.count(owner);
        for i in 0..n0 {
            if !self.cells[i].alive || self.cells[i].owner != owner || self.cells[i].size <= MIN_SPLIT_SIZE {
                continue;
            }
            if count >= MAX_CELLS {
                break;
            }
            let c = self.cells[i];
            let ns = c.size / 2f64.sqrt();
            if ns < MIN_DECAY {
                continue;
            }
            let (ux, uy) = unit(mx - c.x, my - c.y);
            let (ux, uy) = if ux == 0.0 && uy == 0.0 { (0.0, 1.0) } else { (ux, uy) };
            self.cells[i].size = ns;
            let id = self.next_id;
            self.next_id += 1;
            self.cells.push(PC { id, x: c.x, y: c.y, size: ns, owner, bdx: ux, bdy: uy, boost: 780.0 * ns.powf(0.0122), vx: 0.0, vy: 0.0, color: c.color, alive: true });
            count += 1;
        }
    }

    fn pop(&mut self, i: usize) {
        let owner = self.cells[i].owner;
        let mut left = MAX_CELLS as i64 - self.count(owner) as i64;
        if left <= 0 {
            return;
        }
        let cell_mass = self.cells[i].mass();
        let split_min = 36.0;
        let mut splits: Vec<f64> = Vec::new();
        if cell_mass / (left as f64) < split_min {
            let mut n: i64 = 2;
            let mut sm = cell_mass / n as f64;
            while sm > split_min && 2 * n < left {
                n *= 2;
                sm = cell_mass / n as f64;
            }
            sm = cell_mass / (n + 1) as f64;
            for _ in 0..n {
                splits.push(sm);
            }
        } else {
            let mut mass_left = cell_mass / 2.0;
            let mut sm = cell_mass / 2.0;
            'outer: loop {
                let go = left > 0;
                left -= 1;
                if !go {
                    break;
                }
                if mass_left / (left as f64) < split_min {
                    sm = mass_left / left as f64;
                    loop {
                        let g2 = left > 0;
                        left -= 1;
                        if !g2 {
                            break 'outer;
                        }
                        splits.push(sm);
                    }
                }
                while sm >= mass_left && left > 0 {
                    sm /= 2.0;
                }
                splits.push(sm);
                mass_left -= sm;
            }
        }
        let k = splits.len();
        let c = self.cells[i];
        for (j, m) in splits.into_iter().enumerate() {
            let a = std::f64::consts::PI * 2.0 * (j as f64 + 0.37) / k as f64;
            let s1 = (100.0 * m).sqrt();
            let r2 = self.cells[i].size * self.cells[i].size - s1 * s1;
            if r2 <= 0.0 {
                break;
            }
            self.cells[i].size = r2.sqrt();
            let id = self.next_id;
            self.next_id += 1;
            self.cells.push(PC { id, x: c.x, y: c.y, size: s1, owner, bdx: a.cos(), bdy: a.sin(), boost: 780.0 * s1.powf(0.0122), vx: 0.0, vy: 0.0, color: c.color, alive: true });
        }
    }

    /// One server tick.
    fn tick(&mut self) {
        let b = self.border;
        for c in self.cells.iter_mut().filter(|c| c.alive) {
            let (ox, oy) = (c.x, c.y);
            if c.boost > 1.0 {
                let sp = c.boost / 10.0;
                c.boost -= sp;
                c.x += c.bdx * sp;
                c.y += c.bdy * sp;
            }
            let (mx, my) = self.mouse[c.owner as usize];
            let (dx, dy) = (mx - c.x, my - c.y);
            let d = dx.hypot(dy);
            if d >= 1.0 {
                let sp = tick_speed(c.size) * d.min(32.0) / 32.0;
                c.x += dx / d * sp;
                c.y += dy / d * sp;
            }
            let r = c.size / 2.0;
            c.x = c.x.max(b.left + r).min(b.right - r);
            c.y = c.y.max(b.top + r).min(b.bottom - r);
            c.vx = c.x - ox;
            c.vy = c.y - oy;
        }
        // player vs player
        let n = self.cells.len();
        for i in 0..n {
            if !self.cells[i].alive {
                continue;
            }
            for j in (i + 1)..n {
                if !self.cells[j].alive || !self.cells[i].alive || self.cells[i].owner == self.cells[j].owner {
                    continue;
                }
                let (a, bb) = (self.cells[i], self.cells[j]);
                let (big, small, bi, si) = if a.size >= bb.size { (a, bb, i, j) } else { (bb, a, j, i) };
                if big.size < SIZE_EAT * small.size {
                    continue;
                }
                let r = big.size - small.size / 3.0;
                let (dx, dy) = (big.x - small.x, big.y - small.y);
                if dx * dx + dy * dy < r * r {
                    self.cells[si].alive = false;
                    self.cells[bi].size = (big.size * big.size + small.size * small.size).sqrt();
                    if small.owner == 0 {
                        self.lost += small.mass();
                    } else if big.owner == 0 {
                        self.gained += small.mass();
                    }
                }
            }
        }
        // our cells vs food and viruses
        for i in 0..n {
            if !self.cells[i].alive || self.cells[i].owner != 0 {
                continue;
            }
            for f in self.food.iter_mut().filter(|f| f.3) {
                let c = &mut self.cells[i];
                let r = c.size - 3.3;
                let (dx, dy) = (c.x - f.1, c.y - f.2);
                if dx * dx + dy * dy < r * r {
                    f.3 = false;
                    c.size = (c.size * c.size + 100.0).sqrt();
                }
            }
            let mut pop = false;
            for v in self.viruses.iter_mut().filter(|v| v.4) {
                let c = &mut self.cells[i];
                if c.size < VIRUS_EAT * v.3 {
                    continue;
                }
                let r = c.size - v.3 / 3.0;
                let (dx, dy) = (c.x - v.1, c.y - v.2);
                if dx * dx + dy * dy < r * r {
                    v.4 = false;
                    c.size = (c.size * c.size + v.3 * v.3).sqrt();
                    pop = true;
                    break;
                }
            }
            if pop {
                self.pop(i);
            }
        }
        if self.cells.len() > 160 {
            self.cells.retain(|c| c.alive);
        }
    }

    /// Greedy rival steering (server-bot-like): chase what they can eat, flee what can eat them; keep heading otherwise.
    fn rivals_think(&mut self, cfg: &PlanCfg) {
        for owner in 1..self.n_owners as u8 {
            let Some(mi) = self.main_of(owner) else { continue };
            let m = self.cells[mi];
            let (mut fx, mut fy) = (0.0f64, 0.0f64);
            let mut split_at: Option<(f64, f64)> = None;
            let mut best_split = 0.0;
            for o in self.cells.iter().filter(|o| o.alive && o.owner != owner) {
                let (dx, dy) = (o.x - m.x, o.y - m.y);
                let d = dx.hypot(dy).max(1.0);
                let gap = d - o.size - m.size;
                if m.size > o.size * 1.14 {
                    if d < 1200.0 {
                        let w = o.mass() / (d + 50.0);
                        fx += w * dx / d;
                        fy += w * dy / d;
                    }
                    if cfg.rival_split && o.owner == 0 && !self.split_used[owner as usize] && m.size / 2f64.sqrt() > SIZE_EAT * o.size
                        && gap < 620.0 && m.size > 90.0 && o.mass() > best_split
                    {
                        best_split = o.mass();
                        split_at = Some((o.x, o.y));
                    }
                } else if o.size > m.size * 1.14 {
                    let reach = if o.mass() > m.mass() * 2.6 { 900.0 } else { 300.0 };
                    if gap < reach {
                        let w = 10.0 * (1.0 - gap.max(0.0) / reach) * 10.0 + 4.0;
                        fx -= w * dx / d;
                        fy -= w * dy / d;
                    }
                }
            }
            if let Some((sx, sy)) = split_at {
                self.split_used[owner as usize] = true;
                self.mouse[owner as usize] = (sx, sy);
                self.split(owner, sx, sy);
                continue;
            }
            let n = fx.hypot(fy);
            if n > 1e-9 {
                self.dir[owner as usize] = (fx / n, fy / n);
            }
            let (ux, uy) = self.dir[owner as usize];
            self.mouse[owner as usize] = (m.x + ux * FAR, m.y + uy * FAR);
            if ux == 0.0 && uy == 0.0 {
                self.mouse[owner as usize] = (m.x, m.y);
            }
        }
    }

    /// The snapshot the rule bot would see from our main cell.
    fn snapshot(&self, border: Border, buf_ids: &mut u32) -> Option<Snapshot> {
        let mine: Vec<Cell> = self
            .cells
            .iter()
            .filter(|c| c.alive && c.owner == 0)
            .map(|c| {
                let mut k = Cell::new(c.id, c.x, c.y, c.size, Kind::Mine);
                k.color = c.color;
                k
            })
            .collect();
        if mine.is_empty() {
            return None;
        }
        let mut others: Vec<Cell> = Vec::with_capacity(self.cells.len() + self.food.len() + self.viruses.len());
        for c in self.cells.iter().filter(|c| c.alive && c.owner != 0) {
            let mut k = Cell::new(c.id, c.x, c.y, c.size, Kind::Player);
            k.vx = c.vx * 2.0;
            k.vy = c.vy * 2.0;
            k.color = c.color;
            others.push(k);
        }
        for v in self.viruses.iter().filter(|v| v.4) {
            others.push(Cell::new(v.0, v.1, v.2, v.3, Kind::Virus));
        }
        for f in self.food.iter().filter(|f| f.3) {
            others.push(Cell::new(f.0, f.1, f.2, 10.0, Kind::Food));
        }
        *buf_ids += 1;
        Some(Snapshot { mine, others, border, dead: false })
    }
}

fn clamp01(x: f64) -> f64 {
    x.max(0.0).min(1.0)
}

pub struct PlanPolicy {
    pub base: HeuristicPolicy,
    cfg: PlanCfg,
}

struct Opening {
    aim: (f64, f64),
    split: bool,
}

impl PlanPolicy {
    /// spec: {"kind": "plan", "base": "bot/c3_4.json" | {rule-bot spec}, ...planner settings}
    pub fn from_spec(spec: &serde_json::Value) -> PlanPolicy {
        let base_spec = match spec.get("base") {
            Some(serde_json::Value::String(path)) => {
                let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("plan base {path}: {e}"));
                serde_json::from_str(&text).unwrap_or_else(|e| panic!("plan base {path}: {e}"))
            }
            Some(v) => v.clone(),
            None => serde_json::json!({ "farm": "route" }),
        };
        let mut base = HeuristicPolicy::from_spec(&base_spec);
        for k in ["plan_steps", "plan_gated", "plan_override", "plan_split", "plan_us"] {
            base.counters.0.push((k.to_string(), 0));
        }
        PlanPolicy { base, cfg: PlanCfg::from_json(spec) }
    }

    pub fn reset(&mut self) {
        self.base.reset();
    }

    fn gated(&self, s: &Snapshot) -> bool {
        let me = s.main().unwrap();
        let tm = s.total_mass();
        s.others.iter().any(|c| {
            c.kind == Kind::Player
                && c.mass() >= self.cfg.min_rival_mass
                && s.mine.iter().any(|q| q.dist(c) - q.size - c.size < self.cfg.gate + if c.mass() > tm * 2.6 { 500.0 } else { 0.0 })
                && (c.mass() > me.mass() * 0.2 || c.mass() * EAT_RATIO > me.mass())
        })
    }

    fn evaluate(&self, base: &Sim, s: &Snapshot, p: &Params, o: &Opening) -> f64 {
        let cfg = &self.cfg;
        let mut sim = base.clone();
        let start_mass = base.total(0);
        let border = s.border;
        let mut ids = 0u32;
        let (mx, my) = o.aim;
        sim.mouse[0] = (mx, my);
        if o.split {
            sim.split(0, mx, my);
        }
        for t in 0..cfg.horizon {
            if t % cfg.rival_every == 0 {
                sim.rivals_think(cfg);
            }
            if t >= cfg.hold && (t - cfg.hold) % cfg.replan == 0 {
                match sim.snapshot(border, &mut ids) {
                    Some(sn) => {
                        let plan = heuristic_plan(&sn, p);
                        let (ax, ay, sp) = steer(&sn, &plan, None, p);
                        sim.mouse[0] = (ax, ay);
                        if sp {
                            sim.split(0, ax, ay);
                        }
                    }
                    None => break,
                }
            }
            sim.tick();
            if sim.count(0) == 0 {
                break;
            }
        }
        let end = sim.total(0);
        let mut score = end - cfg.loss_w * sim.lost + cfg.eat_w * sim.gained;
        if end <= 0.0 {
            return -cfg.death_w * start_mass - sim.lost;
        }
        // threatened at the end: a cell that can eat one of ours is close
        let mut pen = 0.0;
        for q in sim.cells.iter().filter(|c| c.alive && c.owner == 0) {
            let mut worst: f64 = 0.0;
            for t in sim.cells.iter().filter(|c| c.alive && c.owner != 0 && c.size >= 1.14 * q.size) {
                let extra = if t.mass() > q.mass() * 2.6 { 300.0 } else { 0.0 };
                let gap = (t.x - q.x).hypot(t.y - q.y) - t.size - q.size - extra;
                worst = worst.max(clamp01(1.0 - gap / cfg.near_range));
            }
            pen += q.mass() * worst;
        }
        score -= cfg.near_w * pen;
        if cfg.lead_w > 0.0 {
            let mut rmax: f64 = 0.0;
            for ow in 1..sim.n_owners as u8 {
                rmax = rmax.max(sim.total(ow));
            }
            score += cfg.lead_w * (end - rmax).max(-end).min(end);
        }
        score
    }

    pub fn act(&mut self, s: &mut Snapshot) -> Out {
        let out = self.base.act(s);
        self.base.counters.bump("plan_steps", 1);
        if out.eject || !self.gated(s) {
            return out;
        }
        let t0 = Instant::now();
        self.base.counters.bump("plan_gated", 1);
        let tm = s.total_mass();
        let p = self.base.params_for(tm).clone();
        let sim = Sim::from_snapshot(s, &self.cfg);
        let me = *s.main().unwrap();
        let mut openings: Vec<Opening> = vec![Opening { aim: (out.x, out.y), split: out.split }];
        for k in 0..self.cfg.headings {
            let a = std::f64::consts::PI * 2.0 * k as f64 / self.cfg.headings as f64;
            openings.push(Opening { aim: (me.x + a.cos() * FAR, me.y + a.sin() * FAR), split: false });
        }
        if me.size > MIN_SPLIT_SIZE * 1.1 && s.mine.len() < MAX_CELLS {
            // split towards the nearest cells we could eat with half our mass
            let mut tg: Vec<&Cell> = s
                .others
                .iter()
                .filter(|c| c.kind == Kind::Player && me.size / 2f64.sqrt() > SIZE_EAT * c.size && me.dist(c) - me.size < 800.0)
                .collect();
            tg.sort_by(|a, b| me.dist(a).partial_cmp(&me.dist(b)).unwrap());
            for c in tg.into_iter().take(self.cfg.split_cands) {
                openings.push(Opening { aim: (c.x, c.y), split: true });
            }
        }
        let mut best = (f64::NEG_INFINITY, 0usize);
        let mut base_score = 0.0;
        for (i, o) in openings.iter().enumerate() {
            let sc = self.evaluate(&sim, s, &p, o);
            if i == 0 {
                base_score = sc;
            }
            if sc > best.0 {
                best = (sc, i);
            }
        }
        self.base.counters.bump("plan_us", t0.elapsed().as_micros() as i64);
        if best.1 != 0 && best.0 > base_score + self.cfg.margin * tm {
            self.base.counters.bump("plan_override", 1);
            let o = &openings[best.1];
            if o.split {
                self.base.counters.bump("plan_split", 1);
            }
            return Out { x: o.aim.0, y: o.aim.1, split: o.split, eject: false };
        }
        out
    }
}
