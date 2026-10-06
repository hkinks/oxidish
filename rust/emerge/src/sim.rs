//! EMERGE simulation core: a closed "substrate" (conserved mass ledger) plus stage-gated rules.
//! No rendering here; the same code runs headless for --sim.

use crate::econ;
use macroquad::prelude::{vec2, Vec2};
use std::collections::VecDeque;

pub const MAP: f32 = 3200.0;
pub const NT: usize = 20; // heat lattice is NT x NT tiles
pub const TILE: f32 = MAP / NT as f32;
pub const DT: f32 = 1.0 / 30.0;
pub const FOOD_CAP: usize = 1800;
pub const FOOD_MASS: f32 = 2.0;
pub const HAVEN_R: f32 = 150.0;
pub const ORE_BURN_EFF: f32 = 0.6;

pub const STAGE_NAMES: [&str; 6] = ["SOUP", "HEAT", "SCARCITY", "PACTS", "STATIONS", "MARKET"];

pub struct Rng(pub u64);
impl Rng {
    pub fn u(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }
    pub fn f(&mut self) -> f32 {
        (self.u() >> 8) as f32 / 16777216.0
    }
    pub fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f()
    }
    pub fn chance(&mut self, p: f32) -> bool {
        self.f() < p
    }
    pub fn int(&mut self, n: usize) -> usize {
        (self.u() as usize) % n.max(1)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Player,
    Bot,
    Drifter,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    Forager = 0,
    Hunter = 1,
    Miner = 2,
    Trader = 3,
}
pub const ROLE_NAMES: [&str; 4] = ["forager", "hunter", "miner", "trader"];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Act {
    Dash,
    Pact,
    Leave,
    Build,
    SellOre,
    BuyPlate,
    SellPlate,
    Equip,
    Burn,
}

#[derive(Clone)]
pub struct Agent {
    pub id: usize,
    pub kind: Kind,
    pub name: String,
    pub hue: f32,
    pub pos: Vec2,
    pub vel: Vec2,
    pub want: Vec2,
    pub mass: f32,
    pub ore: f32,
    pub plates: u32,
    pub credits: f32,
    pub armor: u32,
    pub clan: Option<usize>,
    pub alive: bool,
    pub respawn_t: f32,
    pub dash_t: f32,
    pub dash_cd: f32,
    pub dash_dir: Vec2,
    pub role: Role,
    pub role_t: f32,
    pub skill: [f32; 4],
    pub bias: [f32; 4],
    pub goal: Vec2,
    pub goal_t: f32,
    pub wander: Vec2,
    pub act_cd: f32,
    pub social_t: f32,
    pub dock_t: f32,
    pub acts: Vec<Act>,
    pub mining: bool,
    pub kills: u32,
    pub deaths: u32,
    pub mined: f32,
    pub earned: f32,
}

impl Agent {
    pub fn radius(&self) -> f32 {
        radius(self.mass)
    }
    pub fn eff_mass(&self) -> f32 {
        self.mass * (1.0 + 0.07 * self.armor as f32)
    }
    pub fn cargo_cap(&self) -> f32 {
        0.6 * self.mass + 10.0
    }
}

pub fn radius(m: f32) -> f32 {
    4.0 * m.max(1.0).sqrt()
}

#[derive(Clone, Copy)]
pub struct Food {
    pub pos: Vec2,
    pub m: f32,
}

pub struct Vein {
    pub pos: Vec2,
    pub stock: f32,
    pub cap: f32,
}
pub const VEIN_R: f32 = 85.0;

pub struct Clan {
    pub name: String,
    pub hue: f32,
    pub members: Vec<usize>,
    pub treasury: f32,
    pub station: Option<usize>,
    pub alive: bool,
}

pub struct Station {
    pub pos: Vec2,
    pub clan: usize,
    pub ore: f32,
    pub plates: f32,
    pub struct_mass: f32,
    pub ore_bid: f32,
    pub plate_ask: f32,
    pub trades: u32,
    pub alive: bool,
    pub refine_acc: f32,
}

pub struct Rules {
    pub sun: f32,
    pub eat_eff: f32,
    pub decay: f32,
    pub dash: bool,
    pub heat_food: bool,
    pub veins: bool,
    pub drifters: usize,
    pub pacts: bool,
    pub stations: bool,
    pub market: bool,
}

pub fn rules(stage: usize) -> Rules {
    Rules {
        sun: [40.0, 28.0, 20.0, 20.0, 20.0, 20.0][stage],
        eat_eff: if stage >= 1 { 0.8 } else { 1.0 },
        decay: if stage >= 1 { 0.003 } else { 0.0 },
        dash: stage >= 1,
        heat_food: stage >= 1,
        veins: stage >= 2,
        drifters: if stage >= 2 { 5 } else { 0 },
        pacts: stage >= 3,
        stations: stage >= 4,
        market: stage >= 5,
    }
}

#[derive(Default, Clone)]
pub struct Stats {
    pub sun_in: f32,
    pub respawn_in: f32,
    pub eats: u32,
    pub kills_agents: u32,
    pub ore_mined: f32,
    pub ore_delivered: f32,
    pub trades: u32,
    pub money_faucet_bounty: f32,
    pub money_faucet_grant: f32,
    pub money_sink_fee: f32,
    pub money_sink_loot: f32,
    pub money_sink_other: f32,
    pub pacts_formed: u32,
    pub betrayals: u32,
    pub stations_built: u32,
    pub plates_made: u32,
}

pub struct World {
    pub t: f32,
    pub tick: u64,
    pub stage: usize,
    pub stage_t: f32,
    pub rng: Rng,
    pub agents: Vec<Agent>,
    pub food: Vec<Food>,
    pub heat: Vec<f32>,
    pub veins: Vec<Vein>,
    pub clans: Vec<Clan>,
    pub stations: Vec<Station>,
    pub log: VecDeque<(f32, String)>,
    pub stats: Stats,
    pub ledger_base: f32,
    pub sun_budget: f32,
    pub vein_budget: f32,
    pub scarce_t: f32,
    pub first_station_t: Option<f32>,
    pub nerf_until: f32,
    pub patch_done: bool,
    pub bounty_mult: f32,
    pub price_hist: VecDeque<f32>,
    pub money_hist: VecDeque<f32>,
    pub gini_hist: VecDeque<f32>,
    pub stage_times: Vec<f32>,
    pub new_stage_flash: f32,
    pub forced_stage: bool,
    pub autoplay: bool,
}

const NAMES: [&str; 28] = [
    "Aria", "Bolt", "Cinder", "Dune", "Ember", "Flux", "Gale", "Haze", "Iris", "Jolt", "Kite", "Lumen", "Mote", "Nova",
    "Orbit", "Pixel", "Quark", "Rune", "Sable", "Tide", "Umbra", "Vex", "Wisp", "Xeno", "Yarrow", "Zephyr", "Atlas",
    "Bryn",
];
const CLAN_NAMES: [&str; 10] = ["Vanguard", "Lattice", "Tessera", "Meridian", "Cadre", "Halcyon", "Rift", "Quorum", "Anvil", "Ledger"];

pub fn hsv(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h = (h.rem_euclid(1.0)) * 6.0;
    let i = h.floor();
    let f = h - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i as i32 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, v, p),
        _ => (v, p, q),
    }
}

pub fn tile_of(p: Vec2) -> usize {
    let tx = ((p.x / TILE) as isize).clamp(0, NT as isize - 1) as usize;
    let ty = ((p.y / TILE) as isize).clamp(0, NT as isize - 1) as usize;
    ty * NT + tx
}

impl World {
    pub fn new(seed: u64, with_player: bool, n_bots: usize) -> World {
        let mut w = World {
            t: 0.0,
            tick: 0,
            stage: 0,
            stage_t: 0.0,
            rng: Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1),
            agents: vec![],
            food: vec![],
            heat: vec![0.0; NT * NT],
            veins: vec![],
            clans: vec![],
            stations: vec![],
            log: VecDeque::new(),
            stats: Stats::default(),
            ledger_base: 0.0,
            sun_budget: 0.0,
            vein_budget: 0.0,
            scarce_t: 0.0,
            first_station_t: None,
            nerf_until: 0.0,
            patch_done: false,
            bounty_mult: 1.0,
            price_hist: VecDeque::new(),
            money_hist: VecDeque::new(),
            gini_hist: VecDeque::new(),
            stage_times: vec![0.0],
            new_stage_flash: 0.0,
            forced_stage: false,
            autoplay: false,
        };
        let mut id = 0;
        if with_player {
            let a = w.make_agent(id, Kind::Player, "You".into());
            w.agents.push(a);
            id += 1;
        }
        for k in 0..n_bots {
            let nm = NAMES[k % NAMES.len()].to_string();
            let a = w.make_agent(id, Kind::Bot, nm);
            w.agents.push(a);
            id += 1;
        }
        for _ in 0..1500 {
            let p = vec2(w.rng.range(20.0, MAP - 20.0), w.rng.range(20.0, MAP - 20.0));
            w.food.push(Food { pos: p, m: FOOD_MASS });
        }
        w.ledger_base = w.total_mass();
        w.say("Stage 0: SOUP. Pellets are bits. Eat to grow. Bigger eats smaller.");
        w
    }

    fn make_agent(&mut self, id: usize, kind: Kind, name: String) -> Agent {
        let pos = vec2(self.rng.range(100.0, MAP - 100.0), self.rng.range(100.0, MAP - 100.0));
        Agent {
            id,
            kind,
            name,
            hue: self.rng.f(),
            pos,
            vel: Vec2::ZERO,
            want: Vec2::ZERO,
            mass: 10.0,
            ore: 0.0,
            plates: 0,
            credits: 0.0,
            armor: 0,
            clan: None,
            alive: true,
            respawn_t: 0.0,
            dash_t: 0.0,
            dash_cd: 0.0,
            dash_dir: Vec2::X,
            role: Role::Forager,
            role_t: self.rng.range(2.0, 8.0),
            skill: [0.0; 4],
            bias: [self.rng.range(-0.25, 0.25), self.rng.range(-0.25, 0.25), self.rng.range(-0.25, 0.25), self.rng.range(-0.25, 0.25)],
            goal: pos,
            goal_t: 0.0,
            wander: Vec2::from_angle(self.rng.range(0.0, 6.28)),
            act_cd: 0.0,
            social_t: 0.0,
            dock_t: 0.0,
            acts: vec![],
            mining: false,
            kills: 0,
            deaths: 0,
            mined: 0.0,
            earned: 0.0,
        }
    }

    pub fn say<S: Into<String>>(&mut self, s: S) {
        let s = s.into();
        self.log.push_back((self.t, s));
        while self.log.len() > 40 {
            self.log.pop_front();
        }
    }

    pub fn rules(&self) -> Rules {
        rules(self.stage)
    }

    // ---------- conservation ledger ----------
    pub fn total_mass(&self) -> f32 {
        let mut s = 0.0;
        for f in &self.food {
            s += f.m;
        }
        for a in &self.agents {
            if a.alive {
                s += a.mass + a.ore + 3.0 * a.plates as f32 + 3.0 * a.armor as f32;
            }
        }
        for v in &self.veins {
            s += v.stock;
        }
        for h in &self.heat {
            s += *h;
        }
        for st in &self.stations {
            if st.alive {
                s += st.struct_mass + st.ore + 3.0 * st.plates;
            }
        }
        s
    }
    pub fn heat_total(&self) -> f32 {
        self.heat.iter().sum()
    }
    /// Expected total = initial + all sunlight + newcomer endowments.
    pub fn ledger_expected(&self) -> f32 {
        self.ledger_base + self.stats.sun_in + self.stats.respawn_in
    }
    pub fn money_supply(&self) -> f32 {
        let mut m = 0.0;
        for a in &self.agents {
            if a.alive {
                m += a.credits;
            }
        }
        for c in &self.clans {
            if c.alive {
                m += c.treasury;
            }
        }
        m
    }

    pub fn add_heat(&mut self, p: Vec2, h: f32) {
        let i = tile_of(p);
        self.heat[i] += h;
    }

    pub fn in_haven(&self, p: Vec2) -> bool {
        if !self.rules().stations {
            return false;
        }
        self.stations.iter().any(|s| s.alive && s.pos.distance(p) < HAVEN_R)
    }
    pub fn station_at(&self, p: Vec2) -> Option<usize> {
        self.stations
            .iter()
            .enumerate()
            .filter(|(_, s)| s.alive && s.pos.distance(p) < HAVEN_R + 20.0)
            .map(|(i, _)| i)
            .next()
    }

    fn rand_pos_free(&mut self) -> Vec2 {
        for _ in 0..6 {
            let p = vec2(self.rng.range(10.0, MAP - 10.0), self.rng.range(10.0, MAP - 10.0));
            if !self.in_haven(p) {
                return p;
            }
        }
        vec2(self.rng.range(10.0, MAP - 10.0), self.rng.range(10.0, MAP - 10.0))
    }

    // ---------- main step ----------
    pub fn step(&mut self) {
        let dt = DT;
        self.t += dt;
        self.tick += 1;
        self.stage_t += dt;
        if self.new_stage_flash > 0.0 {
            self.new_stage_flash -= dt;
        }
        let rl = self.rules();
        self.sun_spawn(&rl, dt);
        if self.tick % 30 == 0 {
            self.slow_step();
        }
        crate::ai::think(self, dt);
        self.apply_acts();
        self.move_agents(dt);
        self.mine(dt);
        self.eat_food();
        self.eat_agents();
        self.decay(dt);
        self.respawn(dt);
    }

    fn sun_spawn(&mut self, rl: &Rules, dt: f32) {
        self.sun_budget += rl.sun * dt;
        while self.sun_budget >= FOOD_MASS {
            self.sun_budget -= FOOD_MASS;
            // in stage 2+, a quarter of sunlight feeds veins
            if rl.veins && !self.veins.is_empty() && self.rng.chance(0.30) {
                let k = self.rng.int(self.veins.len());
                if self.veins[k].stock + FOOD_MASS <= self.veins[k].cap {
                    self.veins[k].stock += FOOD_MASS;
                    self.stats.sun_in += FOOD_MASS;
                    continue;
                }
            }
            self.stats.sun_in += FOOD_MASS;
            if self.food.len() < FOOD_CAP {
                let p = self.rand_pos_free();
                self.food.push(Food { pos: p, m: FOOD_MASS });
            } else {
                let p = vec2(self.rng.range(0.0, MAP), self.rng.range(0.0, MAP));
                self.add_heat(p, FOOD_MASS);
            }
        }
    }

    /// Once per second: heat diffusion + blooms, clans, stations, drifters, metrics, stage gates.
    fn slow_step(&mut self) {
        let rl = self.rules();
        // heat diffusion (conserving)
        let old = self.heat.clone();
        for ty in 0..NT {
            for tx in 0..NT {
                let i = ty * NT + tx;
                let share = old[i] * 0.04;
                if share <= 0.0 {
                    continue;
                }
                let nb = [(tx as isize - 1, ty as isize), (tx as isize + 1, ty as isize), (tx as isize, ty as isize - 1), (tx as isize, ty as isize + 1)];
                for (x, y) in nb {
                    if x >= 0 && y >= 0 && (x as usize) < NT && (y as usize) < NT {
                        self.heat[i] -= share;
                        self.heat[y as usize * NT + x as usize] += share;
                    }
                }
            }
        }
        // heat blooms into pellets / ore
        if rl.heat_food {
            for i in 0..NT * NT {
                let h = self.heat[i];
                if h < 2.0 {
                    continue;
                }
                let conv = h * 0.10;
                let mut n = (conv / FOOD_MASS).floor() as i32;
                if self.rng.chance(conv / FOOD_MASS - n as f32) {
                    n += 1;
                }
                let (tx, ty) = (i % NT, i / NT);
                for _ in 0..n {
                    if self.heat[i] < FOOD_MASS {
                        break;
                    }
                    let p = vec2((tx as f32 + self.rng.f()) * TILE, (ty as f32 + self.rng.f()) * TILE);
                    // some heat becomes ore in tiles that hold a vein
                    let vi = self.veins.iter().position(|v| v.pos.distance(p) < TILE * 0.9 && v.stock + FOOD_MASS <= v.cap);
                    if rl.veins && vi.is_some() && self.rng.chance(0.4) {
                        self.veins[vi.unwrap()].stock += FOOD_MASS;
                        self.heat[i] -= FOOD_MASS;
                    } else if self.food.len() < FOOD_CAP && !self.in_haven(p) {
                        self.food.push(Food { pos: p, m: FOOD_MASS });
                        self.heat[i] -= FOOD_MASS;
                    } else {
                        break;
                    }
                }
            }
        }
        // armor wears off (a sink that keeps plate demand alive)
        for k in 0..self.agents.len() {
            if self.agents[k].alive && self.agents[k].armor > 0 && self.rng.chance(1.0 / 70.0) {
                self.agents[k].armor -= 1;
                let p = self.agents[k].pos;
                self.add_heat(p, 3.0);
            }
        }
        self.spawn_drifters(&rl);
        econ::slow(self);
        self.update_metrics();
        self.check_stage();
    }

    fn spawn_drifters(&mut self, rl: &Rules) {
        if rl.drifters == 0 || self.tick % 180 != 0 {
            return;
        }
        let alive = self.agents.iter().filter(|a| a.kind == Kind::Drifter && a.alive).count();
        if alive >= rl.drifters {
            return;
        }
        let (mut bi, mut bh) = (0, 0.0);
        for (i, h) in self.heat.iter().enumerate() {
            if *h > bh {
                bh = *h;
                bi = i;
            }
        }
        if bh < 30.0 {
            return;
        }
        let m = (30.0 + 0.5 * (bh - 30.0)).min(110.0);
        self.heat[bi] -= m;
        let (tx, ty) = (bi % NT, bi / NT);
        let pos = vec2((tx as f32 + 0.5) * TILE, (ty as f32 + 0.5) * TILE);
        if let Some(k) = self.agents.iter().position(|a| a.kind == Kind::Drifter && !a.alive) {
            let id = self.agents[k].id;
            let mut a = self.make_agent(id, Kind::Drifter, "drifter".into());
            a.pos = pos;
            a.mass = m;
            a.hue = 0.0;
            self.agents[k] = a;
        } else {
            let id = self.agents.len();
            let mut a = self.make_agent(id, Kind::Drifter, "drifter".into());
            a.pos = pos;
            a.mass = m;
            a.hue = 0.0;
            self.agents.push(a);
        }
    }

    fn update_metrics(&mut self) {
        if self.tick % 150 != 0 {
            return;
        }
        let g = self.gini();
        push_hist(&mut self.gini_hist, g);
        let ms = self.money_supply();
        push_hist(&mut self.money_hist, ms);
        let live: Vec<&Station> = self.stations.iter().filter(|s| s.alive).collect();
        if !live.is_empty() {
            let p = live.iter().map(|s| s.plate_ask).sum::<f32>() / live.len() as f32;
            push_hist(&mut self.price_hist, p);
        }
    }

    pub fn avg_plate_ask(&self) -> Option<f32> {
        let v: Vec<f32> = self.stations.iter().filter(|s| s.alive).map(|s| s.plate_ask).collect();
        if v.is_empty() {
            None
        } else {
            Some(v.iter().sum::<f32>() / v.len() as f32)
        }
    }

    pub fn net_worth(&self, a: &Agent) -> f32 {
        let mut w = a.mass + a.ore + 3.0 * (a.plates + a.armor) as f32;
        if let Some(p) = self.avg_plate_ask() {
            if self.stage >= 5 {
                w += a.credits / (p / 3.0).max(1.0);
            }
        }
        w
    }

    pub fn gini(&self) -> f32 {
        let mut v: Vec<f32> = self.agents.iter().filter(|a| a.alive && a.kind != Kind::Drifter).map(|a| self.net_worth(a)).collect();
        gini_of(&mut v)
    }
    pub fn top_share(&self, frac: f32) -> f32 {
        let mut v: Vec<f32> = self.agents.iter().filter(|a| a.alive && a.kind != Kind::Drifter).map(|a| self.net_worth(a)).collect();
        v.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let tot: f32 = v.iter().sum();
        let k = ((v.len() as f32 * frac).ceil() as usize).max(1);
        if tot <= 0.0 {
            0.0
        } else {
            v.iter().take(k).sum::<f32>() / tot
        }
    }

    pub fn max_mass(&self) -> f32 {
        self.agents.iter().filter(|a| a.alive && a.kind != Kind::Drifter).map(|a| a.mass).fold(0.0, f32::max)
    }
    pub fn best_clan_mass(&self) -> (usize, f32) {
        let mut best = (0, 0.0);
        for c in self.clans.iter().filter(|c| c.alive) {
            let al: Vec<&usize> = c.members.iter().filter(|m| self.agents[**m].alive).collect();
            let m: f32 = al.iter().map(|i| self.agents[**i].mass).sum();
            if al.len() >= 2 && m > best.1 {
                best = (al.len(), m);
            }
        }
        best
    }

    /// Text describing the condition for the next stage and progress 0..1.
    pub fn next_gate(&self) -> Option<(String, f32)> {
        match self.stage {
            0 => {
                let p = (self.max_mass() / 140.0).max(self.stage_t / 50.0).min(1.0);
                Some((format!("a persistent structure forms (largest cell {:.0}/140 mass)", self.max_mass()), p))
            }
            1 => {
                let dens = self.food.len() as f32 / FOOD_CAP as f32;
                let p = (0.45 / dens.max(0.01)).min(1.0).max(self.stage_t / 80.0).min(1.0);
                Some((format!("wild food runs short (pellets {:.0}% of capacity, need < 45%)", dens * 100.0), p))
            }
            2 => {
                let p = (self.stats.ore_mined / 250.0).max(self.stage_t / 70.0).min(1.0);
                Some((format!("ore is worked at scale ({:.0}/250 mined)", self.stats.ore_mined), p))
            }
            3 => {
                let (n, m) = self.best_clan_mass();
                let p = (m / 400.0).min(1.0) * if n >= 2 { 1.0 } else { 0.5 };
                Some((format!("a pact gets heavy enough to hold ground (best pact {} cells, {:.0}/400 mass)", n, m), p))
            }
            4 => {
                let live = self.stations.iter().filter(|s| s.alive).count();
                if live == 0 {
                    Some(("a pact founds a station (B, needs 80 ore)".into(), 0.0))
                } else {
                    let since = self.t - self.first_station_t.unwrap_or(self.t);
                    let p = (self.stats.ore_delivered / 150.0).max(since / 50.0).min(1.0);
                    Some((format!("barter proves itself ({:.0}/150 ore delivered): coins will emerge", self.stats.ore_delivered), p))
                }
            }
            _ => None,
        }
    }

    fn check_stage(&mut self) {
        if self.forced_stage {
            return;
        }
        let go = match self.stage {
            0 => self.max_mass() >= 140.0 || self.stage_t >= 50.0,
            1 => {
                if (self.food.len() as f32) < 0.45 * FOOD_CAP as f32 {
                    self.scarce_t += 1.0;
                } else {
                    self.scarce_t = 0.0;
                }
                (self.scarce_t >= 5.0 && self.stage_t >= 35.0) || self.stage_t >= 80.0
            }
            2 => (self.stats.ore_mined >= 250.0 && self.stage_t >= 40.0) || self.stage_t >= 70.0,
            3 => {
                let (n, m) = self.best_clan_mass();
                n >= 2 && m >= 400.0 && self.stage_t >= 30.0
            }
            4 => {
                let live = self.stations.iter().any(|s| s.alive);
                live && self.stage_t >= 40.0 && (self.stats.ore_delivered >= 150.0 || self.t - self.first_station_t.unwrap_or(self.t) >= 50.0)
            }
            _ => false,
        };
        if go {
            self.advance_stage();
        }
    }

    pub fn advance_stage(&mut self) {
        if self.stage >= 5 {
            return;
        }
        self.stage += 1;
        self.stage_t = 0.0;
        self.new_stage_flash = 6.0;
        self.stage_times.push(self.t);
        let msg = match self.stage {
            1 => "Stage 1: HEAT. Eating is lossy (80%): the rest becomes heat, which blooms into food. Big cells shrink. SPACE dashes.",
            2 => "Stage 2: SCARCITY. Sunlight dims. Ore veins appear (hold still inside one to mine). Drifters roam. R burns ore into mass.",
            3 => "Stage 3: PACTS. F near a cell proposes a pact: no friendly eating, speed when close. G leaves (betrayal is possible).",
            4 => "Stage 4: STATIONS. A pact can found a station (B, 80 ore). Havens: nobody can be eaten inside. Barter ore for plates (1).",
            _ => "Stage 5: MARKET. Credits exist. Stations quote prices from stock. Bounties print money; fees and losses burn it.",
        };
        self.say(msg);
        match self.stage {
            2 => self.spawn_veins(),
            5 => {
                for c in self.clans.iter_mut().filter(|c| c.alive) {
                    c.treasury += 200.0;
                    self.stats.money_faucet_grant += 200.0;
                }
                self.bounty_mult = 1.0;
            }
            _ => {}
        }
    }

    fn spawn_veins(&mut self) {
        let mut tries = 0;
        while self.veins.len() < 11 && tries < 400 {
            tries += 1;
            let p = vec2(self.rng.range(250.0, MAP - 250.0), self.rng.range(250.0, MAP - 250.0));
            if self.veins.iter().any(|v| v.pos.distance(p) < 450.0) {
                continue;
            }
            let stock = 260.0;
            self.stats.sun_in += stock;
            self.veins.push(Vein { pos: p, stock, cap: 520.0 });
        }
    }

    // ---------- actions ----------
    fn apply_acts(&mut self) {
        let rl = self.rules();
        for i in 0..self.agents.len() {
            if self.agents[i].acts.is_empty() {
                continue;
            }
            let acts: Vec<Act> = std::mem::take(&mut self.agents[i].acts);
            if !self.agents[i].alive {
                continue;
            }
            for act in acts {
                match act {
                    Act::Dash => {
                        let a = &mut self.agents[i];
                        if rl.dash && a.dash_cd <= 0.0 && a.mass > 25.0 {
                            let dir = if a.want.length() > 0.05 { a.want.normalize() } else if a.vel.length() > 1.0 { a.vel.normalize() } else { Vec2::X };
                            let cost = a.mass * 0.05;
                            a.mass -= cost;
                            a.dash_dir = dir;
                            a.dash_t = 0.3;
                            a.dash_cd = 2.5;
                            let p = a.pos;
                            self.add_heat(p, cost);
                        }
                    }
                    Act::Burn => {
                        let a = &mut self.agents[i];
                        if rl.veins && a.ore > 0.0 {
                            let o = a.ore;
                            a.ore = 0.0;
                            a.mass += o * ORE_BURN_EFF;
                            let p = a.pos;
                            self.add_heat(p, o * (1.0 - ORE_BURN_EFF));
                        }
                    }
                    Act::Equip => {
                        let a = &mut self.agents[i];
                        if a.plates > 0 && a.armor < 8 {
                            a.plates -= 1;
                            a.armor += 1;
                        }
                    }
                    Act::Pact => {
                        if rl.pacts {
                            econ::propose_pact(self, i);
                        }
                    }
                    Act::Leave => econ::leave_clan(self, i, true),
                    Act::Build => {
                        if rl.stations {
                            econ::build_station(self, i);
                        }
                    }
                    Act::SellOre | Act::BuyPlate | Act::SellPlate => {
                        if rl.stations {
                            econ::trade(self, i, act);
                        }
                    }
                }
            }
        }
    }

    fn move_agents(&mut self, dt: f32) {
        let nearby: Vec<(Option<usize>, Vec2, bool)> = self.agents.iter().map(|a| (a.clan, a.pos, a.alive)).collect();
        let havens: Vec<Vec2> = if self.rules().stations { self.stations.iter().filter(|s| s.alive).map(|s| s.pos).collect() } else { vec![] };
        for a in self.agents.iter_mut() {
            if !a.alive {
                continue;
            }
            a.dash_cd = (a.dash_cd - dt).max(0.0);
            let mut vmax = (340.0 * a.mass.powf(-0.25)).max(45.0);
            if a.kind == Kind::Drifter {
                vmax *= 0.85;
            }
            if let Some(c) = a.clan {
                let buddy = nearby.iter().enumerate().any(|(j, (cc, p, al))| *al && j != a.id && *cc == Some(c) && p.distance(a.pos) < 500.0);
                if buddy {
                    vmax *= 1.10;
                }
            }
            vmax *= 1.0 + 0.04 * a.skill[Role::Hunter as usize].min(3.0) * if a.role == Role::Hunter { 1.0 } else { 0.0 };
            vmax /= 1.0 + a.ore / (1.5 * a.mass);
            let want = if a.want.length() > 1.0 { a.want.normalize() } else { a.want };
            if a.dash_t > 0.0 {
                a.dash_t -= dt;
                a.vel = a.dash_dir * vmax * 2.6;
            } else {
                let desired = want * vmax;
                let k = (6.0 * dt).min(1.0);
                a.vel += (desired - a.vel) * k;
            }
            a.pos += a.vel * dt;
            let r = radius(a.mass);
            a.pos.x = a.pos.x.clamp(r.min(MAP / 2.0), (MAP - r).max(MAP / 2.0));
            a.pos.y = a.pos.y.clamp(r.min(MAP / 2.0), (MAP - r).max(MAP / 2.0));
            if a.kind == Kind::Drifter {
                for h in &havens {
                    let d = a.pos.distance(*h);
                    if d < HAVEN_R + r {
                        a.pos = *h + (a.pos - *h).normalize_or_zero() * (HAVEN_R + r + 1.0);
                    }
                }
            }
        }
    }

    fn mine(&mut self, dt: f32) {
        let rl = self.rules();
        for a in self.agents.iter_mut() {
            a.mining = false;
        }
        if !rl.veins {
            return;
        }
        let yield_mult = if self.t < self.nerf_until { 0.5 } else { 1.0 };
        for i in 0..self.agents.len() {
            let a = &self.agents[i];
            if !a.alive || a.kind == Kind::Drifter || a.want.length() > 0.35 || a.dash_t > 0.0 {
                continue;
            }
            let room = a.cargo_cap() - a.ore;
            if room <= 0.0 {
                continue;
            }
            let Some(vi) = self.veins.iter().position(|v| v.stock > 0.5 && v.pos.distance(a.pos) < VEIN_R) else { continue };
            let sk = if a.role == Role::Miner { a.skill[Role::Miner as usize].min(3.0) } else { 0.0 };
            let rate = 0.9 * a.mass.sqrt() * (1.0 + 0.5 * sk) * yield_mult * dt;
            let take = rate.min(self.veins[vi].stock).min(room);
            self.veins[vi].stock -= take;
            let a = &mut self.agents[i];
            a.ore += take;
            a.mined += take;
            a.mining = true;
            self.stats.ore_mined += take;
        }
    }

    fn eat_food(&mut self) {
        let mut k = 0;
        while k < self.food.len() {
            let f = self.food[k];
            let mut eaten = None;
            for a in self.agents.iter() {
                if a.alive && a.kind != Kind::Drifter || (a.alive && a.kind == Kind::Drifter) {
                    let r = radius(a.mass);
                    if (a.pos - f.pos).length_squared() < r * r {
                        eaten = Some(a.id);
                        break;
                    }
                }
            }
            if let Some(id) = eaten {
                let a = &mut self.agents[id];
                let sk = if a.role == Role::Forager { a.skill[0].min(3.0) } else { 0.0 };
                let gain = f.m * (1.0 + 0.08 * sk);
                // the skill bonus is a "free" gain only if heat pays for it: take it out of the tile
                let extra = gain - f.m;
                a.mass += f.m;
                if extra > 0.0 {
                    let pos = f.pos;
                    let i = tile_of(pos);
                    let pay = extra.min(self.heat[i]);
                    self.heat[i] -= pay;
                    self.agents[id].mass += pay;
                }
                self.stats.eats += 1;
                self.food.swap_remove(k);
            } else {
                k += 1;
            }
        }
    }

    fn eat_agents(&mut self) {
        let rl = self.rules();
        let n = self.agents.len();
        let mut pairs: Vec<(usize, usize)> = vec![];
        for v in 0..n {
            let vic = &self.agents[v];
            if !vic.alive || self.in_haven(vic.pos) {
                continue;
            }
            let mut best: Option<usize> = None;
            for a in 0..n {
                if a == v {
                    continue;
                }
                let att = &self.agents[a];
                if !att.alive {
                    continue;
                }
                if att.kind == Kind::Drifter && vic.kind == Kind::Drifter {
                    continue;
                }
                if att.clan.is_some() && att.clan == vic.clan {
                    continue;
                }
                if att.mass <= vic.eff_mass() * 1.25 {
                    continue;
                }
                if att.pos.distance(vic.pos) < att.radius() - 0.3 * vic.radius() {
                    if best.map_or(true, |b| self.agents[b].mass < att.mass) {
                        best = Some(a);
                    }
                }
            }
            if let Some(b) = best {
                pairs.push((b, v));
            }
        }
        for (a, v) in pairs {
            if !self.agents[a].alive || !self.agents[v].alive {
                continue;
            }
            self.consume(a, v, &rl);
        }
    }

    fn consume(&mut self, a: usize, v: usize, rl: &Rules) {
        let vic = self.agents[v].clone();
        let gain = vic.mass * rl.eat_eff;
        let mut heat = vic.mass - gain;
        let mut ore = 0.0;
        let mut plates = 0;
        let mut credits = 0.0;
        if rl.veins {
            ore = vic.ore * 0.7;
            heat += vic.ore - ore;
            plates = (vic.plates as f32 * 0.7).floor() as u32;
            heat += 3.0 * (vic.plates - plates) as f32;
            heat += 3.0 * vic.armor as f32;
            credits = vic.credits * 0.6;
            self.stats.money_sink_loot += vic.credits - credits;
        }
        let att = &mut self.agents[a];
        att.mass += gain;
        att.ore += ore;
        att.plates += plates;
        att.credits += credits;
        att.kills += 1;
        att.earned += credits;
        let att_kind = att.kind;
        let att_name = att.name.clone();
        let att_clan = att.clan;
        // bounty: money printing
        if rl.market && vic.kind == Kind::Drifter && att_kind != Kind::Drifter {
            let b = vic.mass * 0.25 * self.bounty_mult;
            self.agents[a].credits += b;
            self.agents[a].earned += b;
            self.stats.money_faucet_bounty += b;
        }
        let _ = att_clan;
        self.add_heat(vic.pos, heat);
        let vv = &mut self.agents[v];
        vv.alive = false;
        vv.deaths += 1;
        vv.respawn_t = match vv.kind {
            Kind::Player => 1.5,
            _ => 4.0 + (v % 5) as f32 * 2.0,
        };
        vv.mass = 0.0;
        vv.ore = 0.0;
        vv.plates = 0;
        vv.armor = 0;
        vv.credits = 0.0;
        vv.mining = false;
        self.stats.kills_agents += 1;
        if vic.kind == Kind::Player {
            self.say(format!("{} ate you ({:.0} mass)", att_name, vic.mass));
        } else if vic.mass > 120.0 && vic.kind != Kind::Drifter {
            self.say(format!("{} ate {} ({:.0} mass)", att_name, vic.name, vic.mass));
        }
    }

    fn decay(&mut self, dt: f32) {
        let rl = self.rules();
        if rl.decay <= 0.0 {
            return;
        }
        for i in 0..self.agents.len() {
            let a = &mut self.agents[i];
            if !a.alive || a.kind == Kind::Drifter || a.mass <= 60.0 {
                continue;
            }
            let loss = (a.mass - 60.0) * rl.decay * dt;
            a.mass -= loss;
            let p = a.pos;
            self.add_heat(p, loss);
        }
    }

    fn respawn(&mut self, dt: f32) {
        for i in 0..self.agents.len() {
            if self.agents[i].alive || self.agents[i].kind == Kind::Drifter {
                continue;
            }
            self.agents[i].respawn_t -= dt;
            if self.agents[i].respawn_t <= 0.0 {
                let p = self.rand_pos_free();
                let a = &mut self.agents[i];
                a.alive = true;
                a.mass = 10.0;
                a.pos = p;
                a.vel = Vec2::ZERO;
                a.want = Vec2::ZERO;
                self.stats.respawn_in += 10.0;
            }
        }
    }

    pub fn role_counts(&self) -> [usize; 4] {
        let mut c = [0; 4];
        for a in &self.agents {
            if a.kind == Kind::Bot && a.alive {
                c[a.role as usize] += 1;
            }
        }
        c
    }

    pub fn clan_name_for(&mut self) -> (String, f32) {
        let k = self.clans.len();
        (CLAN_NAMES[k % CLAN_NAMES.len()].to_string(), (0.13 + 0.17 * k as f32) % 1.0)
    }
}

fn push_hist(h: &mut VecDeque<f32>, v: f32) {
    h.push_back(v);
    while h.len() > 90 {
        h.pop_front();
    }
}

pub fn gini_of(v: &mut Vec<f32>) -> f32 {
    if v.len() < 2 {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len() as f32;
    let sum: f32 = v.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }
    let mut acc = 0.0;
    for (i, x) in v.iter().enumerate() {
        acc += (2.0 * (i as f32 + 1.0) - n - 1.0) * x;
    }
    acc / (n * sum)
}
