//! Lookahead planner ({"kind": "plan"}): a light forward simulation built from the Snapshot only.
//!
//! Every decision step (when a rival player is within reach) it simulates a short horizon for a set of
//! candidate openings (headings x split / no split, plus the base rule bot's own action), each held for
//! `hold` ticks and then continued with the base rule bot's steering (rollout policy), while the rivals
//! move with a cheap greedy rule (flee what can eat them, chase what they can eat, split-kill us when in
//! range). The opening with the best score replaces the base action when it beats the base's own by a margin.

use std::collections::HashMap;
use std::time::Instant;

use crate::game::{Border, Cell, Kind, Snapshot};
use crate::params::Params;
use crate::policy::{heuristic_plan, HeuristicPolicy, Out};
use crate::steer::{steer, tick_speed, Mode, EAT_RATIO};

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
    pub frag_w: f64,
    pub r_food: f64,
    pub r_prey: f64,
    pub r_flee: f64,
    pub r_inert: f64,
    pub pop_w: f64,
    pub rival_split: bool,
    pub adv: bool,
    pub need_loss: f64,
    pub virus_gain: f64,
    pub scen: Vec<u8>,
    pub min_w: f64,
    pub split_cands: usize,
    pub max_rivals: usize,
    pub max_food: usize,
    pub sync: f64,
    pub min_rival_mass: f64,
    /// rule-bot rival model: how many of the nearest rivals are simulated with a rule bot (scenario >= 3)
    pub rule_rivals: usize,
    /// view radius around a rival for its rule-bot snapshot
    pub rule_view: f64,
    /// blend of the old heading into the rule bot's heading
    pub rule_inert: f64,
    /// farming rival with fewer than this many known pellets in view keeps its heading (the sim knows few pellets)
    pub rule_food: f64,
    /// scenario 7: fraction of the observed turn rate that is carried on
    pub turn_damp: f64,
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
            frag_w: 0.3,
            r_food: 1.0,
            r_prey: 1.0,
            r_flee: 1.0,
            r_inert: 0.5,
            pop_w: 0.5,
            rival_split: true,
            adv: true,
            need_loss: 0.1,
            virus_gain: 0.0,
            scen: vec![0, 1, 2],
            min_w: 0.5,
            split_cands: 3,
            max_rivals: 10,
            max_food: 48,
            sync: 2.0,
            min_rival_mass: 0.0,
            rule_rivals: 4,
            rule_view: 1100.0,
            rule_inert: 0.3,
            rule_food: 3.0,
            turn_damp: 0.8,
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
        set!(need_loss, virus_gain, min_w, gate, margin, loss_w, eat_w, death_w, near_w, near_range, lead_w, frag_w, r_food, r_prey, r_flee, r_inert, pop_w, sync, min_rival_mass, rule_view, rule_inert, rule_food, turn_damp);
        macro_rules! seti {
            ($($n:ident),*) => { $( if let Some(x) = f(stringify!($n)) { c.$n = x as usize; } )* };
        }
        seti!(rule_rivals, horizon, hold, headings, replan, rival_every, split_cands, max_rivals, max_food);
        if let Some(a) = v.get("scenarios").and_then(|x| x.as_array()) {
            c.scen = a.iter().filter_map(|x| x.as_u64()).map(|x| x as u8).collect();
        }
        if let Some(x) = f("adv") {
            c.adv = x > 0.5;
        }
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
    spd: Vec<f64>,          // rival observed speed as a fraction of its full speed
    turn: Vec<f64>,         // rival heading turn rate (rad per tick)
    split_used: Vec<bool>,
    n_owners: usize,
    next_id: u32,
    lost: f64,
    gained: f64,
    popped: f64,
    trace: bool,
    t: usize,
    snaps: Vec<(usize, Vec<PC>)>,
    record: bool,
    scen: u8,
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
    fn from_snapshot(s: &Snapshot, cfg: &PlanCfg, turns: &HashMap<u32, f64>) -> Sim {
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
        let mut spd = vec![1.0; n_owners];
        let mut turn = vec![0.0; n_owners];
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
                    spd[gi + 1] = (sp / ts).min(1.0);
                    turn[gi + 1] = turns.get(&c.id).copied().unwrap_or(0.0) / cfg.sync;
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
            spd,
            turn,
            split_used: vec![false; n_owners],
            n_owners,
            next_id: 1 << 30,
            lost: 0.0,
            gained: 0.0,
            popped: 0.0,
            trace: false,
            t: 0,
            snaps: Vec::new(),
            record: false,
            scen: 0,
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
        if owner == 0 {
            self.popped += cell_mass;
        }
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
    fn tick(&mut self, virus_gain: f64) {
        self.t += 1;
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
                    if self.trace {
                        eprintln!("  t={} owner{} (size {:.0} at {:.0},{:.0}) eats owner{} (size {:.0} at {:.0},{:.0})", self.t, big.owner, big.size, big.x, big.y, small.owner, small.size, small.x, small.y);
                    }
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
                    c.size = (c.size * c.size + virus_gain * v.3 * v.3).sqrt();
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
    fn rivals_think(&mut self, cfg: &PlanCfg, models: &[HeuristicPolicy]) {
        if self.scen == 5 {
            for owner in 1..self.n_owners as u8 {
                if let Some(mi) = self.main_of(owner) {
                    self.mouse[owner as usize] = (self.cells[mi].x, self.cells[mi].y);
                }
            }
            return;
        }
        if self.scen == 7 {
            // inertial with the observed turn rate carried on (damped): rivals orbit / curve towards a fixed target
            for owner in 1..self.n_owners as u8 {
                if let Some(mi) = self.main_of(owner) {
                    let m = self.cells[mi];
                    let (ux, uy) = self.dir[owner as usize];
                    let a = cfg.turn_damp * self.turn[owner as usize] * cfg.rival_every as f64;
                    let (c, sn) = (a.cos(), a.sin());
                    let nd = (ux * c - uy * sn, ux * sn + uy * c);
                    self.dir[owner as usize] = nd;
                    self.mouse[owner as usize] = if nd.0 == 0.0 && nd.1 == 0.0 { (m.x, m.y) } else { (m.x + nd.0 * FAR, m.y + nd.1 * FAR) };
                }
            }
            return;
        }
        if self.scen == 6 {
            for owner in 1..self.n_owners as u8 {
                if let Some(mi) = self.main_of(owner) {
                    let m = self.cells[mi];
                    let (ux, uy) = self.dir[owner as usize];
                    let reach = if self.spd[owner as usize] > 0.95 { FAR } else { 32.0 * self.spd[owner as usize] };
                    self.mouse[owner as usize] = if ux == 0.0 && uy == 0.0 { (m.x, m.y) } else { (m.x + ux * reach, m.y + uy * reach) };
                }
            }
            return;
        }
        if self.scen == 1 {
            // inertial: every rival keeps its observed heading
            for owner in 1..self.n_owners as u8 {
                if let Some(mi) = self.main_of(owner) {
                    let m = self.cells[mi];
                    let (ux, uy) = self.dir[owner as usize];
                    self.mouse[owner as usize] = if ux == 0.0 && uy == 0.0 { (m.x, m.y) } else { (m.x + ux * FAR, m.y + uy * FAR) };
                }
            }
            return;
        }
        let adv = cfg.adv && self.scen == 0;
        for owner in 1..self.n_owners as u8 {
            let Some(mi) = self.main_of(owner) else { continue };
            let m = self.cells[mi];
            if self.scen >= 3 && self.scen != 5 && (owner as usize) <= cfg.rule_rivals && !models.is_empty() {
                // rule-bot model: this rival decides like the rule bot with genome `scen - 3`, from its own view
                let model = &models[(self.scen as usize - 3) % models.len()];
                if let Some(sn) = self.rival_snapshot(owner, cfg.rule_view) {
                    let p = model.params_for(sn.total_mass());
                    let plan = heuristic_plan(&sn, p);
                    let (ax, ay, sp) = steer(&sn, &plan, None, p);
                    let od = self.dir[owner as usize];
                    let blind = plan.mode == Mode::Farm && ((sn.others.iter().filter(|c| c.kind == Kind::Food).count() as f64) < cfg.rule_food);
                    let (rx, ry) = if blind { od } else { unit(ax - m.x, ay - m.y) };
                    let (nx, ny) = (cfg.rule_inert * od.0 + (1.0 - cfg.rule_inert) * rx, cfg.rule_inert * od.1 + (1.0 - cfg.rule_inert) * ry);
                    let nn = nx.hypot(ny);
                    let nd = if nn > 1e-9 { (nx / nn, ny / nn) } else { od };
                    self.dir[owner as usize] = nd;
                    self.mouse[owner as usize] = if nd.0 == 0.0 && nd.1 == 0.0 { (m.x, m.y) } else { (m.x + nd.0 * FAR, m.y + nd.1 * FAR) };
                    if sp && !self.split_used[owner as usize] && cfg.rival_split {
                        self.split_used[owner as usize] = true;
                        self.split(owner, ax, ay);
                    }
                }
                continue;
            }
            let (mut fx, mut fy) = (0.0f64, 0.0f64);
            let mut split_at: Option<(f64, f64)> = None;
            let mut best_split = 0.0;
            for o in self.cells.iter().filter(|o| o.alive && o.owner != owner) {
                let (dx, dy) = (o.x - m.x, o.y - m.y);
                let d = dx.hypot(dy).max(1.0);
                let gap = d - o.size - m.size;
                if m.size > o.size * 1.14 {
                    if d < 1200.0 {
                        let w = cfg.r_prey * o.mass() / (d + 50.0);
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
                        let w = cfg.r_flee * (10.0 * (1.0 - gap.max(0.0) / reach) * 10.0 + 4.0);
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
            if cfg.r_food > 0.0 {
                for f in self.food.iter().filter(|f| f.3) {
                    let (dx, dy) = (f.1 - m.x, f.2 - m.y);
                    let d = dx.hypot(dy).max(1.0);
                    if d < 800.0 {
                        let w = cfg.r_food * 40.0 / (d + 40.0);
                        fx += w * dx / d;
                        fy += w * dy / d;
                    }
                }
            }
            if adv {
                // pessimistic: what can eat one of ours chases the nearest such cell; what we can eat runs from us
                let mut chase: Option<(f64, f64, f64)> = None;
                let mut run: Option<(f64, f64, f64)> = None;
                for q in self.cells.iter().filter(|q| q.alive && q.owner == 0) {
                    let (dx, dy) = (q.x - m.x, q.y - m.y);
                    let d = dx.hypot(dy).max(1.0);
                    let gap = d - m.size - q.size;
                    if m.size >= 1.14 * q.size && d < 1500.0 && chase.map_or(true, |c| gap < c.2) {
                        chase = Some((dx / d, dy / d, gap));
                    }
                    if q.size >= 1.14 * m.size && gap < 500.0 && run.map_or(true, |c| gap < c.2) {
                        run = Some((-dx / d, -dy / d, gap));
                    }
                }
                if let Some(c) = chase {
                    fx = c.0;
                    fy = c.1;
                } else if let Some(c) = run {
                    fx = c.0;
                    fy = c.1;
                }
            }
            let n = fx.hypot(fy);
            if n > 1e-9 {
                let od = self.dir[owner as usize];
                let (nx, ny) = (cfg.r_inert * od.0 + (1.0 - cfg.r_inert) * fx / n, cfg.r_inert * od.1 + (1.0 - cfg.r_inert) * fy / n);
                let nn = nx.hypot(ny);
                self.dir[owner as usize] = if nn > 1e-9 { (nx / nn, ny / nn) } else { od };
            }
            let (ux, uy) = self.dir[owner as usize];
            self.mouse[owner as usize] = (m.x + ux * FAR, m.y + uy * FAR);
            if ux == 0.0 && uy == 0.0 {
                self.mouse[owner as usize] = (m.x, m.y);
            }
        }
    }

    /// The snapshot a rival's rule bot would see: its cells as `mine`, everything within `view` of its main cell as others.
    fn rival_snapshot(&self, owner: u8, view: f64) -> Option<Snapshot> {
        let mi = self.main_of(owner)?;
        let m = self.cells[mi];
        let mine: Vec<Cell> = self
            .cells
            .iter()
            .filter(|c| c.alive && c.owner == owner)
            .map(|c| {
                let mut k = Cell::new(c.id, c.x, c.y, c.size, Kind::Mine);
                k.color = c.color;
                k
            })
            .collect();
        let near = |x: f64, y: f64, r: f64| (x - m.x).hypot(y - m.y) < view + r;
        let mut others: Vec<Cell> = Vec::with_capacity(self.cells.len() + self.food.len() + self.viruses.len());
        for c in self.cells.iter().filter(|c| c.alive && c.owner != owner && near(c.x, c.y, c.size)) {
            let mut k = Cell::new(c.id, c.x, c.y, c.size, Kind::Player);
            k.vx = c.vx * 2.0;
            k.vy = c.vy * 2.0;
            k.color = c.color;
            others.push(k);
        }
        for v in self.viruses.iter().filter(|v| v.4 && near(v.1, v.2, v.3)) {
            others.push(Cell::new(v.0, v.1, v.2, v.3, Kind::Virus));
        }
        for f in self.food.iter().filter(|f| f.3 && near(f.1, f.2, 0.0)) {
            others.push(Cell::new(f.0, f.1, f.2, 10.0, Kind::Food));
        }
        Some(Snapshot { mine, others, border: self.border, dead: false })
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
    step: i64,
    pending: Vec<(i64, usize, Vec<PC>, Vec<u32>)>,
    val: bool,
    val_scen: u8,
    models: Vec<HeuristicPolicy>,
    /// per rival cell id: (last velocity angle, smoothed turn rate per step, step of the last sample)
    heading: HashMap<u32, (f64, f64, i64)>,
}

struct Opening {
    aim: (f64, f64),
    split: bool,
    trace: bool,
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
        for k in ["plan_steps", "plan_gated", "plan_deep", "plan_override", "plan_split", "plan_us"] {
            base.counters.0.push((k.to_string(), 0));
        }
        // rival_bots: rule-bot specs (path or inline) that scenarios 3, 4, ... simulate rivals with
        let models: Vec<HeuristicPolicy> = spec
            .get("rival_bots")
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .map(|v| match v {
                        serde_json::Value::String(path) => {
                            let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("rival bot {path}: {e}"));
                            HeuristicPolicy::from_spec(&serde_json::from_str(&text).unwrap_or_else(|e| panic!("rival bot {path}: {e}")))
                        }
                        v => HeuristicPolicy::from_spec(v),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let val_scen = std::env::var("AGAR_PLAN_VAL_SCEN").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        PlanPolicy { base, cfg: PlanCfg::from_json(spec), step: 0, pending: Vec::new(), val: std::env::var("AGAR_PLAN_VAL").is_ok(), val_scen, models, heading: HashMap::new() }
    }

    pub fn reset(&mut self) {
        self.base.reset();
    }

    /// Track every visible rival cell's heading and its smoothed turn rate (rad per decision step).
    fn track_headings(&mut self, s: &Snapshot) {
        let step = self.step;
        for c in s.others.iter().filter(|c| c.kind == Kind::Player) {
            if c.vx.hypot(c.vy) < 4.0 {
                self.heading.remove(&c.id);
                continue;
            }
            let a = c.vy.atan2(c.vx);
            let e = self.heading.entry(c.id).or_insert((a, 0.0, step));
            if e.2 == step - 1 {
                let mut d = a - e.0;
                while d > std::f64::consts::PI { d -= 2.0 * std::f64::consts::PI; }
                while d < -std::f64::consts::PI { d += 2.0 * std::f64::consts::PI; }
                e.1 = 0.5 * e.1 + 0.5 * d;
            } else {
                e.1 = 0.0;
            }
            e.0 = a;
            e.2 = step;
        }
        if step % 50 == 0 {
            self.heading.retain(|_, v| v.2 > step - 10);
        }
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

    /// Score of an opening: blend of the worst and the mean over the rival-behaviour scenarios.
    fn evaluate(&self, base: &Sim, s: &Snapshot, p: &Params, o: &Opening) -> (f64, f64) {
        let (mut mn, mut sum, mut lost) = (f64::INFINITY, 0.0, 0.0f64);
        for &sc in &self.cfg.scen {
            let (v, d, _) = self.evaluate_rec(base, s, p, o, false, sc);
            mn = mn.min(v);
            sum += v;
            lost = lost.max(d[1] + if d[0] <= 0.0 { base.total(0) } else { 0.0 });
        }
        let mean = sum / self.cfg.scen.len().max(1) as f64;
        (self.cfg.min_w * mn + (1.0 - self.cfg.min_w) * mean, lost)
    }

    fn evaluate_full(&self, base: &Sim, s: &Snapshot, p: &Params, o: &Opening) -> (f64, [f64; 4]) {
        let (a, b, _) = self.evaluate_rec(base, s, p, o, false, self.cfg.scen.first().copied().unwrap_or(0));
        (a, b)
    }

    fn evaluate_rec(&self, base: &Sim, s: &Snapshot, p: &Params, o: &Opening, record: bool, scen: u8) -> (f64, [f64; 4], Vec<(usize, Vec<PC>)>) {
        let cfg = &self.cfg;
        let mut sim = base.clone();
        sim.trace = o.trace;
        sim.record = record;
        sim.scen = scen;
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
                sim.rivals_think(cfg, &self.models);
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
            sim.tick(cfg.virus_gain);
            if sim.record && (t + 1 == 10 || t + 1 == 24 || t + 1 == 50) {
                let v: Vec<PC> = sim.cells.iter().filter(|c| c.alive).copied().collect();
                sim.snaps.push((t + 1, v));
            }
            if sim.count(0) == 0 {
                break;
            }
        }
        let end = sim.total(0);
        let mut score = end - cfg.loss_w * sim.lost + cfg.eat_w * sim.gained;
        if end <= 0.0 {
            return (-cfg.death_w * start_mass - sim.lost, [end, sim.lost, sim.gained, sim.popped], sim.snaps);
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
        let largest = sim.cells.iter().filter(|c| c.alive && c.owner == 0).map(|c| c.mass()).fold(0.0, f64::max);
        score -= cfg.frag_w * (end - largest) + cfg.pop_w * sim.popped;
        if cfg.lead_w > 0.0 {
            let mut rmax: f64 = 0.0;
            for ow in 1..sim.n_owners as u8 {
                rmax = rmax.max(sim.total(ow));
            }
            score += cfg.lead_w * (end - rmax).max(-end).min(end);
        }
        (score, [end, sim.lost, sim.gained, sim.popped], sim.snaps)
    }

    pub fn act(&mut self, s: &mut Snapshot) -> Out {
        let out = self.base.act(s);
        self.base.counters.bump("plan_steps", 1);
        self.step += 1;
        self.track_headings(s);
        if self.val {
            let step = self.step;
            let mut keep = Vec::new();
            for (due, h, cells, near) in std::mem::take(&mut self.pending) {
                if due != step {
                    keep.push((due, h, cells, near));
                    continue;
                }
                let (mut n, mut perr, mut serr) = (0, 0.0, 0.0);
                let (mut nn, mut nerr) = (0, 0.0);
                let (mut pred_mine, mut pred_n) = (0.0, 0);
                for c in &cells {
                    if c.owner == 0 {
                        pred_mine += c.mass();
                        pred_n += 1;
                        continue;
                    }
                    if let Some(a) = s.others.iter().find(|a| a.id == c.id) {
                        n += 1;
                        perr += (a.x - c.x).hypot(a.y - c.y);
                        if std::env::var("AGAR_PLAN_VALV").is_ok() && near.contains(&c.id) { eprintln!("VALV id={} h={} pred=({:.0},{:.0}) act=({:.0},{:.0}) size={:.0} actv=({:.1},{:.1})", c.id, h, c.x, c.y, a.x, a.y, a.size, a.vx, a.vy); }
                        if near.contains(&c.id) {
                            nn += 1;
                            nerr += (a.x - c.x).hypot(a.y - c.y);
                        }
                        serr += a.size - c.size;
                    }
                }
                let npred = cells.iter().filter(|c| c.owner != 0).count();
                eprintln!("VAL dec={} h={} matched={} of {} perr={:.0} serr={:.1} near={} nerr={:.0} mine_pred={:.0}({}) mine_act={:.0}({})", step - (h as i64 % 1000) / 2, h, n, npred, perr / n.max(1) as f64, serr / n.max(1) as f64, nn, nerr / nn.max(1) as f64, pred_mine, pred_n, s.total_mass(), s.mine.len());
            }
            self.pending = keep;
        }
        if self.val && std::env::var("AGAR_PLAN_VALV").is_ok() { for c in s.others.iter().filter(|c| c.kind == Kind::Player) { eprintln!("VALV0 step={} id={} pos=({:.0},{:.0}) v=({:.1},{:.1}) size={:.0}", self.step, c.id, c.x, c.y, c.vx, c.vy, c.size); } }
        if out.eject || !self.gated(s) {
            return out;
        }
        let t0 = Instant::now();
        self.base.counters.bump("plan_gated", 1);
        let tm = s.total_mass();
        let p = self.base.params_for(tm).clone();
        let turns: HashMap<u32, f64> = self.heading.iter().map(|(k, v)| (*k, v.1)).collect();
        let sim = Sim::from_snapshot(s, &self.cfg, &turns);
        let me = *s.main().unwrap();
        let mut openings: Vec<Opening> = vec![Opening { aim: (out.x, out.y), split: out.split, trace: false }];
        for k in 0..self.cfg.headings {
            let a = std::f64::consts::PI * 2.0 * k as f64 / self.cfg.headings as f64;
            openings.push(Opening { aim: (me.x + a.cos() * FAR, me.y + a.sin() * FAR), split: false, trace: false });
        }
        if me.size > MIN_SPLIT_SIZE * 1.1 && (s.mine.len() as f64) <= p.split_max_cells.round_ties_even() {
            // split towards the nearest cells we could eat with half our mass
            let mut tg: Vec<&Cell> = s
                .others
                .iter()
                .filter(|c| c.kind == Kind::Player && me.size / 2f64.sqrt() > SIZE_EAT * c.size && me.dist(c) - me.size < 800.0)
                .collect();
            tg.sort_by(|a, b| me.dist(a).partial_cmp(&me.dist(b)).unwrap());
            for c in tg.into_iter().take(self.cfg.split_cands) {
                openings.push(Opening { aim: (c.x, c.y), split: true, trace: false });
            }
        }
        let mut best = (f64::NEG_INFINITY, 0usize);
        let mut base_score = 0.0;
        let mut base_lost = 0.0;
        for (i, o) in openings.iter().enumerate() {
            let (sc, lost) = self.evaluate(&sim, s, &p, o);
            if i == 0 {
                base_score = sc;
                base_lost = lost;
                if base_lost < self.cfg.need_loss * tm {
                    best = (sc, 0);
                    break; // nothing at stake: the base action stands, no need to price the alternatives
                }
                self.base.counters.bump("plan_deep", 1);
            }
            if sc > best.0 {
                best = (sc, i);
            }
        }
        self.base.counters.bump("plan_us", t0.elapsed().as_micros() as i64);
        if self.val {
            let ch = if best.1 != 0 && best.0 > base_score + self.cfg.margin * tm && base_lost >= self.cfg.need_loss * tm { best.1 } else { 0 };
            let (_, _, snaps) = self.evaluate_rec(&sim, s, &p, &openings[ch], true, self.val_scen);
            for (t, cells) in snaps {
                let tag = if ch != 0 { t + 2000 + if openings[ch].split { 1000 } else { 0 } } else { t };
                let near: Vec<u32> = s.others.iter().filter(|c| c.kind == Kind::Player && s.mine.iter().any(|q| q.dist(c) - q.size - c.size < 600.0)).map(|c| c.id).collect();
                self.pending.push((self.step + (t as i64) / 2, tag, cells, near));
            }
        }
        if std::env::var("AGAR_PLAN_TRACE").ok().and_then(|v| v.parse::<i64>().ok()) == self.base.counters.0.iter().find(|e| e.0 == "plan_steps").map(|e| e.1) {
            openings[best.1].trace = true;
            eprintln!("TRACE best opening {} aim=({:.0},{:.0}) split={} me=({:.0},{:.0})", best.1, openings[best.1].aim.0, openings[best.1].aim.1, openings[best.1].split, me.x, me.y);
            for c in &s.mine { eprintln!("  mine id{} ({:.0},{:.0}) size {:.0}", c.id, c.x, c.y, c.size); }
            for c in s.others.iter().filter(|c| c.kind == Kind::Player) { eprintln!("  other id{} col{:x} ({:.0},{:.0}) size {:.0} v=({:.0},{:.0})", c.id, c.color, c.x, c.y, c.size, c.vx, c.vy); }
            self.evaluate_full(&sim, s, &p, &openings[best.1]);
        }
        if std::env::var("AGAR_PLAN_LOG").is_ok() {
            let sc: Vec<String> = openings.iter().map(|o| {
                let (s, d) = self.evaluate_full(&sim, s, &p, o);
                format!("{:.0}[{:.0},{:.0},{:.0},{:.0}]", s, d[0], d[1], d[2], d[3])
            }).collect();
            let ang = |o: &Opening| ((o.aim.1 - me.y).atan2(o.aim.0 - me.x).to_degrees()).round();
            eprintln!("PLAN step={} m={:.0} cells={} me=({:.0},{:.0}) riv={} base_ang={} {} best={} ang={} split={} scores={:?}",
                self.base.counters.0.iter().find(|e| e.0 == "plan_steps").map(|e| e.1).unwrap_or(0), tm, s.mine.len(), me.x, me.y, sim.n_owners - 1,
                ang(&openings[0]), openings[0].split, best.1, ang(&openings[best.1]), openings[best.1].split, sc);
        }
        if best.1 != 0 && best.0 > base_score + self.cfg.margin * tm && base_lost >= self.cfg.need_loss * tm {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Border, Cell, Kind, Snapshot};

    fn snap(mine: Vec<Cell>, others: Vec<Cell>) -> Snapshot {
        Snapshot { mine, others, border: Border { left: -3500.0, right: 3500.0, top: -3500.0, bottom: 3500.0 }, dead: false }
    }

    #[test]
    fn bigger_cell_eats_smaller_owner() {
        let s = snap(vec![Cell::new(1, 0.0, 0.0, 100.0, Kind::Mine)], vec![Cell::new(2, 20.0, 0.0, 50.0, Kind::Player)]);
        let mut sim = Sim::from_snapshot(&s, &PlanCfg::default(), &HashMap::new());
        sim.tick(0.0);
        assert_eq!(sim.count(1), 0);
        assert!(sim.total(0) > 100.0);
        assert!(sim.gained > 0.0);
    }

    #[test]
    fn standing_next_to_an_eater_is_priced() {
        let s = snap(vec![Cell::new(1, 0.0, 0.0, 60.0, Kind::Mine)], vec![Cell::new(2, 90.0, 0.0, 120.0, Kind::Player)]);
        let pol = PlanPolicy::from_spec(&serde_json::json!({"kind": "plan"}));
        let sim = Sim::from_snapshot(&s, &pol.cfg, &HashMap::new());
        let p = Params::default();
        let stay = Opening { aim: (0.0, 0.0), split: false, trace: false };
        let (score, lost) = pol.evaluate(&sim, &s, &p, &stay);
        assert!(lost > 0.0 && score < 0.0);
    }

    #[test]
    fn no_rival_in_reach_keeps_the_base_action() {
        let mut s = snap(vec![Cell::new(1, 0.0, 0.0, 100.0, Kind::Mine)], vec![Cell::new(2, 3000.0, 3000.0, 300.0, Kind::Player)]);
        let mut a = PlanPolicy::from_spec(&serde_json::json!({"kind": "plan"}));
        let mut b = HeuristicPolicy::from_spec(&serde_json::json!({"farm": "route"}));
        let (oa, ob) = (a.act(&mut s.clone()), b.act(&mut s));
        assert_eq!((oa.x, oa.y, oa.split), (ob.x, ob.y, ob.split));
    }

    #[test]
    fn planner_runs_away_from_a_big_rival() {
        let mut s = snap(vec![Cell::new(1, 0.0, 0.0, 70.0, Kind::Mine)], vec![Cell::new(2, 200.0, 0.0, 160.0, Kind::Player)]);
        let mut a = PlanPolicy::from_spec(&serde_json::json!({"kind": "plan"}));
        let o = a.act(&mut s);
        assert!(o.x < 1.0, "should head away from the eater, aimed at {}", o.x);
    }
}
