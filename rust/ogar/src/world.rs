//! The MultiOgar (FFA) game world: cells, physics, eating/splitting/merging, spawning, decay.
//! A port of vendor/MultiOgar GameServer.js + entity/* (statistically, not bit-for-bit, equivalent).

use std::collections::HashMap;

use crate::config::Config;
use crate::rng::Rng;

pub const T_PLAYER: u8 = 0;
pub const T_FOOD: u8 = 1;
pub const T_VIRUS: u8 = 2;
pub const T_EJECT: u8 = 3;

#[derive(Clone, Debug)]
pub struct Entity {
    pub id: u32,
    pub kind: u8,
    pub owner: i32,
    pub x: f64,
    pub y: f64,
    pub size: f64,
    pub radius: f64, // size^2
    pub color: [u8; 3],
    pub birth: u32,
    pub moving: bool,
    pub boost: f64,
    pub bdx: f64,
    pub bdy: f64,
    pub bangle: f64,
    pub removed: bool,
    pub remerge: bool,
    pub shooter: i32,
    pub feeder: i32,
    gr: (i32, i32, i32, i32),
}

impl Entity {
    #[inline]
    pub fn mass(&self) -> f64 {
        self.radius / 100.0
    }
}

struct Grid {
    g: f64,
    minx: f64,
    miny: f64,
    nx: i32,
    ny: i32,
    buckets: Vec<Vec<u32>>,
    stamp: Vec<u32>,
    cur: u32,
}

impl Grid {
    fn new(map: f64) -> Grid {
        let g = 400.0;
        let n = (map / g).ceil() as i32 + 1;
        Grid { g, minx: -map / 2.0, miny: -map / 2.0, nx: n, ny: n, buckets: vec![Vec::new(); (n * n) as usize], stamp: Vec::new(), cur: 0 }
    }

    fn range(&self, minx: f64, miny: f64, maxx: f64, maxy: f64) -> (i32, i32, i32, i32) {
        let c = |v: f64, o: f64, n: i32| (((v - o) / self.g).floor() as i32).clamp(0, n - 1);
        (c(minx, self.minx, self.nx), c(miny, self.miny, self.ny), c(maxx, self.minx, self.nx), c(maxy, self.miny, self.ny))
    }

    fn insert(&mut self, h: u32, r: (i32, i32, i32, i32)) {
        for gy in r.1..=r.3 {
            for gx in r.0..=r.2 {
                self.buckets[(gy * self.nx + gx) as usize].push(h);
            }
        }
    }

    fn remove(&mut self, h: u32, r: (i32, i32, i32, i32)) {
        for gy in r.1..=r.3 {
            for gx in r.0..=r.2 {
                let b = &mut self.buckets[(gy * self.nx + gx) as usize];
                if let Some(p) = b.iter().position(|&x| x == h) {
                    b.swap_remove(p);
                }
            }
        }
    }

    /// Candidate handles whose buckets overlap the box (deduplicated; caller does the exact test).
    fn query(&mut self, minx: f64, miny: f64, maxx: f64, maxy: f64, out: &mut Vec<u32>, nents: usize) {
        if self.stamp.len() < nents {
            self.stamp.resize(nents + 1024, 0);
        }
        self.cur = self.cur.wrapping_add(1);
        if self.cur == 0 {
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.cur = 1;
        }
        let r = self.range(minx, miny, maxx, maxy);
        for gy in r.1..=r.3 {
            for gx in r.0..=r.2 {
                for &h in &self.buckets[(gy * self.nx + gx) as usize] {
                    if self.stamp[h as usize] != self.cur {
                        self.stamp[h as usize] = self.cur;
                        out.push(h);
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PKind {
    Human,
    Bot,
    Agent,
}

#[derive(Clone, Debug)]
pub struct Player {
    pub pid: u32,
    pub kind: PKind,
    pub name: String,
    pub skin: Option<String>,
    pub color: [u8; 3],
    pub cells: Vec<u32>,
    pub mouse: (f64, f64),
    pub mouse_in: Option<(f64, f64)>,
    pub center: (f64, f64),
    pub vminx: f64,
    pub vminy: f64,
    pub vmaxx: f64,
    pub vmaxy: f64,
    pub half_w: f64,
    pub scale: f64,
    pub score: f64,
    pub mass_changed: bool,
    pub spectating: bool,
    pub free_roam: bool,
    pub last_q_tick: u32,
    pub p_split: bool,
    pub p_eject: bool,
    pub p_q: bool,
    pub last_eject: Option<u32>,
    pub spawn_size: f64,
    /// 0xCD (+500 mass) used this life; shows only in the [players] log ("b": 1)
    pub boosted: bool,
    pub view_nodes: Vec<u32>,
    pub client_nodes: Vec<u32>,
    pub connected: bool,
    pub closed_tick: u32,
    pub removed: bool,
    pub rejoined: bool,
    pub new_cells: Vec<u32>,
    pub tick_lb: u32,
    pub follow_prefix: Option<String>,
    // bot state
    pub split_cooldown: i32,
    pub target_pursuit: i32,
    pub split_target: Option<(u32, u32)>,
    pub merge_override: bool,
    // arena.js virus attribution stats
    pub shots: u32,
    pub pops: u32,
    pub pop_mass: f32,
    /// tick of the last virus this player ate and burst on (recordings only)
    pub last_pop_tick: Option<u32>,
}

impl Player {
    fn new(pid: u32, kind: PKind) -> Player {
        Player {
            pid,
            kind,
            name: String::new(),
            skin: None,
            color: [0, 0, 0],
            cells: Vec::new(),
            mouse: (0.0, 0.0),
            mouse_in: None,
            center: (0.0, 0.0),
            vminx: 0.0,
            vminy: 0.0,
            vmaxx: 0.0,
            vmaxy: 0.0,
            half_w: 0.0,
            scale: 1.0,
            score: 0.0,
            mass_changed: true,
            spectating: false,
            free_roam: false,
            last_q_tick: 0,
            p_split: false,
            p_eject: false,
            p_q: false,
            last_eject: None,
            spawn_size: 0.0,
            boosted: false,
            view_nodes: Vec::new(),
            client_nodes: Vec::new(),
            connected: true,
            closed_tick: 0,
            removed: false,
            rejoined: false,
            new_cells: Vec::new(),
            tick_lb: 0,
            follow_prefix: None,
            split_cooldown: 0,
            target_pursuit: 0,
            split_target: None,
            merge_override: false,
            shots: 0,
            pops: 0,
            pop_mass: 0.0,
            last_pop_tick: None,
        }
    }

    pub fn total_mass(&self, w: &World) -> f64 {
        self.cells.iter().map(|&h| w.ents[h as usize].mass()).sum()
    }
}

/// A player cell eaten by another player's cell (for the in-process agents' kills / losses stats).
#[derive(Clone, Debug)]
pub struct KillEvent {
    pub victim_owner: i32,
    pub killer_owner: i32,
    pub victim_mass: f64,
    pub killer_mass: f64,
    pub killer_cells: usize,
    pub killer_total: f64,
}

/// One player cell eaten by another player's cell, with the context a recording needs (`record_eats`).
#[derive(Clone, Debug)]
pub struct EatEvent {
    pub tick: u32,
    pub eater_owner: i32,
    pub victim_owner: i32,
    pub eater_cell_mass: f64,
    pub eater_total: f64,
    pub eater_cells: usize,
    pub eater_pos: (f64, f64),
    pub victim_cell_mass: f64,
    pub victim_total: f64,
    pub victim_cells: usize,
    pub victim_pos: (f64, f64),
    /// ticks since the victim's owner ate a virus (u32::MAX = never)
    pub victim_pop_ago: u32,
}

struct Col {
    cell: u32,
    check: u32,
    r: f64,
    dx: f64,
    dy: f64,
    d: f64,
    push: f64,
    squared: f64,
}

pub struct World {
    /// serve --full-view: human players receive every node of the map (fog of war off)
    pub full_view: bool,
    pub cfg: Config,
    pub rng: Rng,
    pub ents: Vec<Entity>,
    free: Vec<u32>,
    pending_free: Vec<u32>,
    grid: Grid,
    pub players: Vec<Player>,
    pub nodes_player: Vec<u32>,
    nodes_moving: Vec<u32>,
    nodes_virus: Vec<u32>,
    nodes_eject: Vec<u32>,
    pub food_count: i64,
    pub tick: u32,
    last_node_id: u32,
    last_player_id: u32,
    pub border: (f64, f64, f64, f64), // minx, miny, maxx, maxy
    pub leaderboard: Vec<usize>,
    pub largest: Option<usize>,
    /// victim id -> killer id, for kills across owners this tick (the "eat" records of UpdateNodes)
    pub eaten: HashMap<u32, u32>,
    pub track_eats: bool,
    pub kill_log: Vec<KillEvent>,
    /// opt-in (`ogar serve --record`): rich eat events for recordings; drained by the recorder
    pub record_eats: bool,
    pub eat_events: Vec<EatEvent>,
    shooter: i32,
    cand: Vec<u32>,
    bot_names: Vec<&'static str>,
    skins: Vec<&'static str>,
    /// extra callbacks collected for the server layer
    pub chat_out: Vec<(usize, String)>,
}

fn irnd(x: f64) -> f64 {
    // JS ~~x
    (x as i32) as f64
}

impl World {
    pub fn new(cfg: Config, seed: u64) -> World {
        let map = cfg.map;
        let mut w = World {
            full_view: false,
            grid: Grid::new(map),
            border: (-map / 2.0, -map / 2.0, map / 2.0, map / 2.0),
            rng: Rng::new(seed),
            ents: Vec::new(),
            free: Vec::new(),
            pending_free: Vec::new(),
            players: Vec::new(),
            nodes_player: Vec::new(),
            nodes_moving: Vec::new(),
            nodes_virus: Vec::new(),
            nodes_eject: Vec::new(),
            food_count: 0,
            tick: 0,
            last_node_id: 1,
            last_player_id: 1,
            leaderboard: Vec::new(),
            largest: None,
            eaten: HashMap::new(),
            track_eats: false,
            kill_log: Vec::new(),
            record_eats: false,
            eat_events: Vec::new(),
            shooter: -1,
            cand: Vec::new(),
            bot_names: include_str!("../data/botnames.txt").lines().filter(|l| !l.is_empty()).collect(),
            skins: include_str!("../data/skins.txt").lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect(),
            chat_out: Vec::new(),
            cfg,
        };
        for _ in 0..w.cfg.bots {
            w.add_bot();
        }
        w
    }

    // ------------------------------------------------------------------ entities

    pub fn age(&self, h: u32) -> u32 {
        self.tick.saturating_sub(self.ents[h as usize].birth)
    }

    fn new_entity(&mut self, kind: u8, owner: i32, x: f64, y: f64, size: f64) -> Entity {
        let id = self.last_node_id;
        self.last_node_id += 1;
        Entity {
            id,
            kind,
            owner,
            x,
            y,
            size,
            radius: size * size,
            color: [0, 0, 0],
            birth: self.tick,
            moving: false,
            boost: 0.0,
            bdx: 1.0,
            bdy: 0.0,
            bangle: 0.0,
            removed: false,
            remerge: false,
            shooter: -1,
            feeder: -1,
            gr: (0, 0, 0, 0),
        }
    }

    fn bound_of(e: &Entity) -> (f64, f64, f64, f64) {
        (e.x - e.size, e.y - e.size, e.x + e.size, e.y + e.size)
    }

    pub fn add_node(&mut self, mut e: Entity) -> u32 {
        if self.shooter >= 0 && (e.kind == T_VIRUS || e.kind == T_EJECT) {
            e.shooter = self.shooter;
        }
        let owner = e.owner;
        if owner >= 0 {
            e.color = self.players[owner as usize].color;
        }
        let b = Self::bound_of(&e);
        e.gr = self.grid.range(b.0, b.1, b.2, b.3);
        let kind = e.kind;
        let id = e.id;
        let moving_unowned = e.moving && owner < 0;
        let h = if let Some(h) = self.free.pop() {
            self.ents[h as usize] = e;
            h
        } else {
            self.ents.push(e);
            (self.ents.len() - 1) as u32
        };
        let gr = self.ents[h as usize].gr;
        self.grid.insert(h, gr);
        if owner >= 0 {
            let p = &mut self.players[owner as usize];
            p.cells.push(h);
            if p.kind == PKind::Human {
                p.new_cells.push(id);
            }
        }
        match kind {
            T_PLAYER => self.nodes_player.push(h),
            T_FOOD => self.food_count += 1,
            T_VIRUS => self.nodes_virus.push(h),
            _ => self.nodes_eject.push(h),
        }
        if moving_unowned {
            self.nodes_moving.push(h);
        }
        h
    }

    pub fn remove_node(&mut self, h: u32) {
        let (kind, owner, gr) = {
            let e = &mut self.ents[h as usize];
            if e.removed && e.gr.0 < 0 {
                return;
            }
            e.removed = true;
            let gr = e.gr;
            e.gr.0 = -1;
            (e.kind, e.owner, gr)
        };
        if gr.0 < 0 {
            return;
        }
        self.grid.remove(h, gr);
        if let Some(p) = self.nodes_moving.iter().position(|&x| x == h) {
            self.nodes_moving.remove(p);
        }
        match kind {
            T_PLAYER => {
                if owner >= 0 {
                    let cells = &mut self.players[owner as usize].cells;
                    if let Some(p) = cells.iter().position(|&x| x == h) {
                        cells.remove(p);
                    }
                }
                if let Some(p) = self.nodes_player.iter().position(|&x| x == h) {
                    self.nodes_player.remove(p);
                }
            }
            T_FOOD => self.food_count -= 1,
            T_VIRUS => {
                if let Some(p) = self.nodes_virus.iter().position(|&x| x == h) {
                    self.nodes_virus.swap_remove(p);
                }
            }
            _ => {
                if let Some(p) = self.nodes_eject.iter().position(|&x| x == h) {
                    self.nodes_eject.swap_remove(p);
                }
            }
        }
        self.pending_free.push(h);
    }

    #[inline]
    pub fn set_size(&mut self, h: u32, size: f64) {
        let e = &mut self.ents[h as usize];
        e.size = size;
        e.radius = size * size;
        if e.owner >= 0 {
            self.players[e.owner as usize].mass_changed = true;
        }
    }

    fn set_boost(&mut self, h: u32, dist: f64, angle: f64) {
        let e = &mut self.ents[h as usize];
        e.boost = dist;
        e.bdx = angle.sin();
        e.bdy = angle.cos();
        e.bangle = angle;
        e.moving = true;
        if e.owner < 0 && !self.nodes_moving.contains(&h) {
            self.nodes_moving.push(h);
        }
    }

    fn update_node_quad(&mut self, h: u32) {
        let (b, old) = {
            let e = &self.ents[h as usize];
            (Self::bound_of(e), e.gr)
        };
        let nr = self.grid.range(b.0, b.1, b.2, b.3);
        if nr != old {
            self.grid.remove(h, old);
            self.grid.insert(h, nr);
            self.ents[h as usize].gr = nr;
        }
    }

    pub fn random_position(&mut self) -> (f64, f64) {
        let (minx, miny, maxx, maxy) = self.border;
        (minx + (maxx - minx) * self.rng.f64(), miny + (maxy - miny) * self.rng.f64())
    }

    pub fn random_color(&mut self) -> [u8; 3] {
        let h = 360.0 * self.rng.f64() / 60.0;
        let s = 248.0 / 255.0;
        let i = h as i32;
        let f = h - i as f64;
        let p = 1.0 * (1.0 - s);
        let q = 1.0 * (1.0 - s * f);
        let t = 1.0 * (1.0 - s * (1.0 - f));
        let (r, g, b) = match i {
            0 => (1.0, t, p),
            1 => (q, 1.0, p),
            2 => (p, 1.0, t),
            3 => (p, q, 1.0),
            4 => (t, p, 1.0),
            _ => (1.0, p, q),
        };
        [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
    }

    // ------------------------------------------------------------------ players

    pub fn add_player(&mut self, kind: PKind) -> usize {
        let mut p = Player::new(self.last_player_id, kind);
        self.last_player_id += 1;
        p.mass_changed = true;
        self.players.push(p);
        self.players.len() - 1
    }

    pub fn add_bot(&mut self) -> usize {
        let i = self.add_player(PKind::Bot);
        let mut skin = String::new();
        if !self.skins.is_empty() && self.rng.f64() < 0.75 {
            skin = format!("{{{}}}", self.skins[self.rng.index(self.skins.len())]);
        }
        let name = if self.bot_names.is_empty() { "bot".to_string() } else { self.bot_names[self.rng.index(self.bot_names.len())].to_string() };
        self.join_game_text(i, &format!("{skin}{name}"));
        i
    }

    /// PacketHandler.nickName: "{skin}name" parsing, then joinGame.
    pub fn join_game_text(&mut self, pi: usize, text: &str) {
        let mut name = text.to_string();
        let mut skin: Option<String> = None;
        if text.starts_with('{') {
            if let Some(n) = text[1..].find('}') {
                let inner = &text[1..1 + n];
                if !inner.is_empty() {
                    skin = Some(if inner == "r" && !self.skins.is_empty() { self.skins[self.rng.index(self.skins.len())].to_string() } else { inner.to_string() });
                } else {
                    skin = Some(String::new());
                }
                name = text[1 + n + 1..].to_string();
            }
        }
        if name.chars().count() > self.cfg.player_max_nick {
            name = name.chars().take(self.cfg.player_max_nick).collect();
        }
        self.join_game(pi, &name, skin);
    }

    pub fn join_game(&mut self, pi: usize, name: &str, skin: Option<String>) {
        if !self.players[pi].cells.is_empty() {
            return;
        }
        {
            let p = &mut self.players[pi];
            if let Some(s) = skin {
                if !s.is_empty() {
                    p.skin = Some(s);
                }
            }
            p.name = name.to_string();
            p.spectating = false;
            p.free_roam = false;
            if matches!(p.kind, PKind::Human | PKind::Agent) {
                p.client_nodes.clear();
                p.rejoined = true;
            }
        }
        self.on_player_spawn(pi);
    }

    fn on_player_spawn(&mut self, pi: usize) {
        self.players[pi].boosted = false;
        let c = self.random_color();
        self.players[pi].color = c;
        let pos = self.random_position();
        self.spawn_player(pi, pos);
    }

    pub fn spawn_player(&mut self, pi: usize, mut pos: (f64, f64)) {
        let p = &self.players[pi];
        let mut start = self.cfg.player_start_size;
        if p.spawn_size > 0.0 {
            start = p.spawn_size;
        } else if p.kind == PKind::Bot {
            start = self.cfg.bot_start_size;
        }
        if self.cfg.eject_spawn_chance > 0.0 && !self.nodes_eject.is_empty() {
            let ei = self.rng.index(self.nodes_eject.len());
            let eh = self.nodes_eject[ei];
            let roll = (self.rng.f64() * 100.0).floor();
            let e = &self.ents[eh as usize];
            if e.boost < 1.0 && roll <= self.cfg.eject_spawn_chance {
                let (c, ex, ey, es) = (e.color, e.x, e.y, e.size);
                self.players[pi].color = c;
                pos = (ex, ey);
                start = start.max(es);
                self.remove_node(eh);
            }
        }
        for _ in 0..10 {
            if !self.will_collide(pos, start) {
                break;
            }
            pos = self.random_position();
        }
        let e = self.new_entity(T_PLAYER, pi as i32, pos.0, pos.1, start);
        self.add_node(e);
        self.players[pi].mouse = pos;
    }

    fn will_collide(&mut self, pos: (f64, f64), size: f64) -> bool {
        let (minx, miny, maxx, maxy) = (pos.0 - size, pos.1 - size, pos.0 + size, pos.1 + size);
        if minx * minx + miny * miny + size * size <= size * 2.0 {
            return false;
        }
        let mut cand = std::mem::take(&mut self.cand);
        cand.clear();
        self.grid.query(minx, miny, maxx, maxy, &mut cand, self.ents.len());
        let hit = cand.iter().any(|&h| {
            let e = &self.ents[h as usize];
            !e.removed && e.kind != T_EJECT && e.kind != T_FOOD && !(e.x - e.size >= maxx || e.x + e.size <= minx || e.y - e.size >= maxy || e.y + e.size <= miny)
        });
        self.cand = cand;
        hit
    }

    pub fn remove_player(&mut self, pi: usize) {
        let cells = self.players[pi].cells.clone();
        for h in cells {
            self.remove_node(h);
        }
        self.players[pi].removed = true;
        self.players[pi].cells.clear();
    }

    // ------------------------------------------------------------------ actions

    pub fn press_space(&mut self, pi: usize) {
        let p = &self.players[pi];
        if !p.spectating && !p.merge_override {
            self.split_cells(pi);
        }
    }

    pub fn press_eject(&mut self, pi: usize) {
        if !self.players[pi].spectating {
            self.eject_mass(pi);
        }
    }

    fn press_q(&mut self, pi: usize) {
        let tick = self.tick;
        let p = &mut self.players[pi];
        if !p.spectating || tick.wrapping_sub(p.last_q_tick) < 20 {
            return;
        }
        p.last_q_tick = tick;
        p.free_roam = !p.free_roam;
    }

    fn split_cells(&mut self, pi: usize) {
        let max = self.cfg.player_max_cells;
        let mut known: Vec<u32> = Vec::new();
        let cells = self.players[pi].cells.clone();
        for &h in &cells {
            if self.ents[h as usize].size > self.cfg.player_min_split {
                if cells.len() as i64 >= max {
                    break;
                }
                known.push(h);
            }
        }
        let mouse = self.players[pi].mouse;
        for h in known {
            if self.ents[h as usize].removed {
                continue;
            }
            let e = &self.ents[h as usize];
            let (mut x, mut y) = (irnd(mouse.0 - e.x), irnd(mouse.1 - e.y));
            if x * x + y * y < 1.0 {
                x = 0.0;
                y = 0.0;
            }
            let angle = x.atan2(y);
            self.split_player_cell(pi, h, angle, None, Some(max));
        }
    }

    fn split_player_cell(&mut self, pi: usize, parent: u32, angle: f64, mass: Option<f64>, max: Option<i64>) {
        if let Some(m) = max {
            if self.players[pi].cells.len() as i64 >= m {
                return;
            }
        }
        let pe = &self.ents[parent as usize];
        let mut size2 = pe.size / self.cfg.player_split_div.sqrt();
        let mut size1 = 0.0;
        if let Some(m) = mass {
            if m != 0.0 {
                size1 = (100.0 * m).sqrt();
                size2 = (pe.radius - size1 * size1).sqrt();
            }
        }
        if size2.is_nan() || size2 < self.cfg.player_min_decay {
            return;
        }
        let (px, py) = (pe.x, pe.y);
        self.set_size(parent, size2);
        let size = if size1 != 0.0 { size1 } else { size2 };
        let e = self.new_entity(T_PLAYER, pi as i32, px, py, size);
        let h = self.add_node(e);
        self.set_boost(h, self.cfg.player_split_speed * size.powf(0.0122), angle);
    }

    fn can_eject(&mut self, pi: usize) -> bool {
        let tick = self.tick;
        let cd = self.cfg.eject_cooldown;
        let p = &mut self.players[pi];
        match p.last_eject {
            None => {
                p.last_eject = Some(tick);
                true
            }
            Some(l) => {
                if tick - l < cd {
                    return false;
                }
                p.last_eject = Some(tick);
                true
            }
        }
    }

    fn eject_mass(&mut self, pi: usize) {
        if !self.can_eject(pi) {
            return;
        }
        let cells = self.players[pi].cells.clone();
        let mouse = self.players[pi].mouse;
        self.shooter = pi as i32;
        for h in cells {
            let e = &self.ents[h as usize];
            if e.removed || e.size < self.cfg.player_min_eject {
                continue;
            }
            let (mut dx, mut dy) = (mouse.0 - e.x, mouse.1 - e.y);
            let sq = dx * dx + dy * dy;
            if sq > 1.0 {
                dx /= sq.sqrt();
                dy /= sq.sqrt();
            } else {
                dx = 0.0;
                dy = 0.0;
            }
            let loss = self.cfg.eject_size_loss;
            let ns = (e.radius - loss * loss).sqrt();
            let (cx, cy, color) = (e.x, e.y, e.color);
            self.set_size(h, ns);
            let cs = self.ents[h as usize].size;
            let pos = (cx + dx * cs, cy + dy * cs);
            let mut angle = dx.atan2(dy);
            if angle.is_nan() {
                angle = std::f64::consts::PI / 2.0;
            } else {
                angle += 0.6 * self.rng.f64() - 0.3;
            }
            let mut eject = self.new_entity(T_EJECT, -1, pos.0, pos.1, self.cfg.eject_size);
            eject.color = color;
            let speed = self.cfg.eject_speed;
            let eh = self.add_node(eject);
            self.set_boost(eh, speed, angle);
        }
        self.shooter = -1;
    }

    fn shoot_virus(&mut self, parent: u32, angle: f64) {
        let by = self.ents[parent as usize].feeder;
        if by >= 0 {
            self.players[by as usize].shots += 1;
        }
        self.shooter = by;
        let (x, y) = (self.ents[parent as usize].x, self.ents[parent as usize].y);
        let v = self.new_entity(T_VIRUS, -1, x, y, self.cfg.virus_min_size);
        let h = self.add_node(v);
        self.set_boost(h, self.cfg.virus_eject_speed, angle);
        self.shooter = -1;
    }

    // ------------------------------------------------------------------ physics

    fn update_merge(&mut self, h: u32) {
        let age = self.age(h);
        let size = self.ents[h as usize].size;
        let time = self.cfg.player_merge_time.max(size * 0.2) * 25.0;
        self.ents[h as usize].remerge = age as f64 >= time;
    }

    fn get_speed(&self, size: f64, dist: f64) -> f64 {
        let speed = 2.2 * size.powf(-0.439) * 40.0 * (self.cfg.player_speed / 30.0);
        speed * dist.min(32.0) / 32.0
    }

    fn move_player(&mut self, h: u32, pi: usize) {
        let p = &self.players[pi];
        if !p.connected {
            return;
        }
        let e = &self.ents[h as usize];
        let dx = irnd(p.mouse.0 - e.x);
        let dy = irnd(p.mouse.1 - e.y);
        let sq = dx * dx + dy * dy;
        if sq < 1.0 {
            return;
        }
        let sqrt = sq.sqrt();
        let speed = self.get_speed(e.size, sqrt);
        if speed <= 0.0 {
            return;
        }
        let e = &mut self.ents[h as usize];
        e.x += dx / sqrt * speed;
        e.y += dy / sqrt * speed;
    }

    fn move_cell(&mut self, h: u32) {
        let (minx, miny, maxx, maxy) = self.border;
        let e = &mut self.ents[h as usize];
        if e.moving && (e.boost == 0.0 || e.removed) {
            e.boost = 0.0;
            e.moving = false;
            return;
        }
        let speed = e.boost / 10.0;
        e.boost -= speed;
        e.x += e.bdx * speed;
        e.y += e.bdy * speed;
        let r = e.size / 2.0;
        if e.x < minx + r || e.x > maxx - r {
            e.bdx = -e.bdx;
        }
        if e.y < miny + r || e.y > maxy - r {
            e.bdy = -e.bdy;
        }
        e.x = e.x.max(minx + r).min(maxx - r);
        e.y = e.y.max(miny + r).min(maxy - r);
    }

    fn auto_split(&mut self, h: u32, pi: usize) {
        let max = self.cfg.player_max_size;
        let p = &self.players[pi];
        let e = &self.ents[h as usize];
        if p.merge_override || e.size < max {
            return;
        }
        if p.cells.len() as i64 >= self.cfg.player_max_cells {
            self.set_size(h, max);
        } else {
            let angle = 2.0 * std::f64::consts::PI * self.rng.f64();
            let mass = e.mass() / self.cfg.player_split_div;
            self.split_player_cell(pi, h, angle, Some(mass), None);
        }
    }

    fn check_collision(&self, cell: u32, check: u32) -> Col {
        let (a, b) = (&self.ents[cell as usize], &self.ents[check as usize]);
        let r = a.size + b.size;
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let squared = dx * dx + dy * dy;
        let d = squared.sqrt();
        Col { cell, check, r, dx, dy, d, push: ((r - d) / d).min(r - d), squared }
    }

    fn check_rigid(&self, m: &Col) -> bool {
        let (a, b) = (&self.ents[m.cell as usize], &self.ents[m.check as usize]);
        if a.owner < 0 || b.owner < 0 {
            return false;
        }
        if a.owner != b.owner {
            return false;
        }
        if self.players[a.owner as usize].merge_override {
            return false;
        }
        let r = self.cfg.split_restore_ticks;
        if self.age(m.cell) < r || self.age(m.check) < r {
            return false;
        }
        !a.remerge || !b.remerge
    }

    fn resolve_rigid(&mut self, m: &Col) {
        if m.d > m.r {
            return;
        }
        let (ma, mb) = (self.ents[m.cell as usize].mass(), self.ents[m.check as usize].mass());
        let rt = ma + mb;
        let (r1, r2) = (ma / rt, mb / rt);
        let (fx, fy) = (irnd(m.dx), irnd(m.dy));
        {
            let a = &mut self.ents[m.cell as usize];
            a.x -= m.push * fx * r2;
            a.y -= m.push * fy * r2;
        }
        let b = &mut self.ents[m.check as usize];
        b.x += m.push * fx * r1;
        b.y += m.push * fy * r1;
    }

    fn can_eat(&self, eater: u32, victim: u32) -> bool {
        match self.ents[eater as usize].kind {
            T_PLAYER => true,
            T_VIRUS => (self.nodes_virus.len() as i64) < self.cfg.virus_max_amount && self.ents[victim as usize].kind == T_EJECT,
            _ => false,
        }
    }

    fn resolve_collision(&mut self, m: &Col) {
        let (mut cell, mut check) = (m.cell, m.check);
        if self.ents[cell as usize].size > self.ents[check as usize].size {
            std::mem::swap(&mut cell, &mut check);
        }
        let (ce, ke) = (&self.ents[cell as usize], &self.ents[check as usize]);
        if ce.removed || ke.removed {
            return;
        }
        let size = ke.size - ce.size / 3.0;
        if m.squared >= size * size {
            return;
        }
        if ce.kind == T_EJECT && self.age(cell) < 1 {
            return;
        }
        if ce.owner >= 0 && ce.owner == ke.owner {
            if self.age(cell) < self.cfg.split_restore_ticks || self.age(check) < self.cfg.split_restore_ticks {
                return;
            }
            if self.players[ce.owner as usize].cells.len() <= 2 {
                self.players[ce.owner as usize].merge_override = false;
            }
        } else {
            let mult = match ce.kind {
                T_VIRUS => self.cfg.virus_eat_mult,
                T_FOOD | T_EJECT => 1.0,
                _ => self.cfg.player_eat_mult,
            };
            if !self.can_eat(check, cell) || ke.size < mult * ce.size {
                return;
            }
        }
        self.eat(check, cell);
    }

    fn eat(&mut self, eater: u32, victim: u32) {
        let (ek, eo, eid, vo, vid, vk) = {
            let (e, v) = (&self.ents[eater as usize], &self.ents[victim as usize]);
            (e.kind, e.owner, e.id, v.owner, v.id, v.kind)
        };
        if vk == T_PLAYER && eo >= 0 && vo != eo {
            let (vm, em) = (self.ents[victim as usize].mass(), self.ents[eater as usize].mass());
            let (mut n, mut tot) = (0usize, 0.0);
            for &h in &self.players[eo as usize].cells {
                n += 1;
                tot += self.ents[h as usize].mass();
            }
            self.kill_log.push(KillEvent { victim_owner: vo, killer_owner: eo, victim_mass: vm, killer_mass: em, killer_cells: n, killer_total: tot });
            if self.record_eats {
                let vp = &self.players[vo as usize];
                let vtot: f64 = vp.cells.iter().map(|&h| self.ents[h as usize].mass()).sum();
                let (ee, ve) = (&self.ents[eater as usize], &self.ents[victim as usize]);
                self.eat_events.push(EatEvent {
                    tick: self.tick,
                    eater_owner: eo,
                    victim_owner: vo,
                    eater_cell_mass: em,
                    eater_total: tot,
                    eater_cells: n,
                    eater_pos: (ee.x, ee.y),
                    victim_cell_mass: vm,
                    victim_total: vtot,
                    victim_cells: vp.cells.len(),
                    victim_pos: (ve.x, ve.y),
                    victim_pop_ago: vp.last_pop_tick.map_or(u32::MAX, |t| self.tick.saturating_sub(t)),
                });
            }
        }
        self.ents[victim as usize].removed = true;
        // check.onEat(cell)
        match ek {
            T_VIRUS => {
                let sh = self.ents[victim as usize].shooter;
                if sh >= 0 {
                    self.ents[eater as usize].feeder = sh;
                }
                let (er, vr, ang) = (self.ents[eater as usize].radius, self.ents[victim as usize].radius, self.ents[victim as usize].bangle);
                self.set_size(eater, (er + vr).sqrt());
                if self.ents[eater as usize].size >= self.cfg.virus_max_size {
                    self.set_size(eater, self.cfg.virus_min_size);
                    self.shoot_virus(eater, ang);
                }
            }
            _ => {
                let (es, vs) = (self.ents[eater as usize].size, self.ents[victim as usize].size);
                if es >= 250.0 && vs <= 41.23 && vk == T_PLAYER {
                    self.ents[victim as usize].radius = 0.0;
                }
                let (er, vr) = (self.ents[eater as usize].radius, self.ents[victim as usize].radius);
                self.set_size(eater, (er + vr).sqrt());
            }
        }
        // cell.onEaten(check): a virus explodes the player that ate it
        if vk == T_VIRUS && eo >= 0 {
            self.virus_eaten(victim, eater);
        }
        if self.track_eats && vo != eo {
            self.eaten.insert(vid, eid);
        }
        self.update_node_quad(eater);
        self.remove_node(victim);
    }

    fn virus_eaten(&mut self, virus: u32, eater: u32) {
        let owner = self.ents[eater as usize].owner as usize;
        let cells_now = self.players[owner].cells.len() as i64;
        let mut cells_left = self.cfg.virus_max_cells - cells_now;
        let vshooter = self.ents[virus as usize].shooter;
        if vshooter >= 0 && vshooter as usize != owner && cells_left > 0 {
            let m = self.ents[eater as usize].mass();
            let s = &mut self.players[vshooter as usize];
            s.pops += 1;
            s.pop_mass += m as f32;
        }
        if cells_left <= 0 {
            return;
        }
        if self.record_eats {
            self.players[owner].last_pop_tick = Some(self.tick);
        }
        let split_min = self.cfg.virus_split_div;
        let cell_mass = self.ents[eater as usize].mass();
        let mut splits: Vec<f64> = Vec::new();
        if cell_mass / (cells_left as f64) < split_min {
            let mut split_count: i64 = 2;
            let mut split_mass = cell_mass / split_count as f64;
            while split_mass > split_min && 2 * split_count < cells_left {
                split_count *= 2;
                split_mass = cell_mass / split_count as f64;
            }
            split_mass = cell_mass / (split_count + 1) as f64;
            while split_count > 0 {
                split_count -= 1;
                splits.push(split_mass);
            }
        } else {
            let mut mass_left = cell_mass / 2.0;
            let mut split_mass = cell_mass / 2.0;
            loop {
                let go = cells_left > 0;
                cells_left -= 1;
                if !go {
                    break;
                }
                if mass_left / (cells_left as f64) < split_min {
                    split_mass = mass_left / cells_left as f64;
                    loop {
                        let g2 = cells_left > 0;
                        cells_left -= 1;
                        if !g2 {
                            break;
                        }
                        splits.push(split_mass);
                    }
                }
                while split_mass >= mass_left && cells_left > 0 {
                    split_mass /= 2.0;
                }
                splits.push(split_mass);
                mass_left -= split_mass;
            }
        }
        for m in splits {
            let angle = std::f64::consts::PI * 2.0 * self.rng.f64();
            self.split_player_cell(owner, eater, angle, Some(m), None);
        }
    }

    fn collide_all(&mut self, h: u32, moving_pass: bool) {
        let mut cand = std::mem::take(&mut self.cand);
        cand.clear();
        let (minx, miny, maxx, maxy) = Self::bound_of(&self.ents[h as usize]);
        self.grid.query(minx, miny, maxx, maxy, &mut cand, self.ents.len());
        for &o in &cand {
            if o == h {
                continue;
            }
            let (e, f) = (&self.ents[h as usize], &self.ents[o as usize]);
            if e.removed || f.removed {
                continue;
            }
            // exact bound test (QuadNode.intersects)
            if f.x - f.size >= maxx || f.x + f.size <= minx || f.y - f.size >= maxy || f.y + f.size <= miny {
                continue;
            }
            let m = self.check_collision(h, o);
            if !moving_pass {
                if self.check_rigid(&m) {
                    self.resolve_rigid(&m);
                } else {
                    self.resolve_collision(&m);
                }
            } else if self.ents[h as usize].kind == T_EJECT && self.ents[o as usize].kind == T_EJECT {
                self.resolve_rigid(&m);
            } else {
                self.resolve_collision(&m);
            }
        }
        self.cand = cand;
    }

    fn spawn_cells(&mut self) {
        let want = self.cfg.food_min_amount - self.food_count;
        let n = want.min(self.cfg.food_spawn_amount);
        for _ in 0..n.max(0) {
            let (x, y) = self.random_position();
            let mut f = self.new_entity(T_FOOD, -1, x, y, self.cfg.food_size);
            f.color = self.random_color();
            self.add_node(f);
        }
        let vwant = self.cfg.virus_min_amount - self.nodes_virus.len() as i64;
        let vn = vwant.min(2);
        let pos = self.random_position();
        for _ in 0..vn.max(0) {
            if !self.will_collide(pos, self.cfg.virus_min_size) {
                let mut v = self.new_entity(T_VIRUS, -1, pos.0, pos.1, self.cfg.virus_min_size);
                v.color = [51, 255, 51];
                self.add_node(v);
            }
        }
    }

    fn update_decay(&mut self) {
        for pi in 0..self.players.len() {
            if self.players[pi].removed {
                continue;
            }
            let n = self.players[pi].cells.len();
            for j in 0..n {
                let Some(&h) = self.players[pi].cells.get(j) else { break };
                let e = &self.ents[h as usize];
                if e.removed || e.size <= self.cfg.player_min_decay {
                    break;
                }
                let decay = 1.0 - self.cfg.player_decay_rate;
                let size = (e.size * e.size * decay).sqrt().max(self.cfg.player_min_decay);
                self.set_size(h, size);
            }
        }
    }

    /// One server tick (GameServer.mainLoop): physics, spawning, decay, then per-client updates.
    pub fn tick(&mut self) {
        self.eaten.clear();
        let mut i = 0;
        while i < self.nodes_player.len() {
            let h = self.nodes_player[i];
            i += 1;
            let owner = self.ents[h as usize].owner;
            if self.ents[h as usize].removed || owner < 0 {
                continue;
            }
            let pi = owner as usize;
            self.update_merge(h);
            self.move_cell(h);
            self.move_player(h, pi);
            self.auto_split(h, pi);
            self.update_node_quad(h);
            self.collide_all(h, false);
        }
        let mut i = 0;
        while i < self.nodes_moving.len() {
            let h = self.nodes_moving[i];
            if self.ents[h as usize].removed {
                i += 1;
                continue;
            }
            self.move_cell(h);
            self.update_node_quad(h);
            if !self.ents[h as usize].moving {
                self.nodes_moving.remove(i);
            }
            self.collide_all(h, true);
            i += 1;
        }
        if self.tick % self.cfg.spawn_interval == 0 {
            self.spawn_cells();
        }
        if (self.tick + 3) % 25 == 0 {
            self.update_decay();
        }
        self.tick += 1;
        self.update_clients();
        for h in std::mem::take(&mut self.pending_free) {
            self.free.push(h);
        }
        if (self.tick + 7) % self.cfg.lb_update == 0 {
            self.update_leaderboard();
        }
    }

    // ------------------------------------------------------------------ per-client update

    fn update_clients(&mut self) {
        let n = self.players.len();
        for i in 0..n {
            self.check_connection(i);
        }
        for i in 0..n {
            self.update_tick(i);
            if self.players[i].kind == PKind::Bot && !self.players[i].removed {
                self.bot_decide(i);
            }
        }
    }

    fn check_connection(&mut self, pi: usize) {
        if self.players[pi].removed {
            return;
        }
        match self.players[pi].kind {
            PKind::Bot => {
                if self.players[pi].cells.is_empty() {
                    self.on_player_spawn(pi);
                }
            }
            PKind::Human | PKind::Agent => {
                let p = &mut self.players[pi];
                if !p.connected {
                    if self.tick.wrapping_sub(p.closed_tick) >= self.cfg.disconnect_ticks {
                        self.remove_player(pi);
                        return;
                    }
                    p.mouse = p.center;
                    p.p_split = false;
                    p.p_eject = false;
                    p.p_q = false;
                }
            }
        }
    }

    fn update_scale(&mut self, pi: usize) {
        let mut total = 0.0;
        let mut score = 0.0;
        for &h in &self.players[pi].cells {
            let e = &self.ents[h as usize];
            total += e.size;
            score += e.radius;
        }
        let p = &mut self.players[pi];
        if total == 0.0 {
            p.score = 0.0;
        } else {
            p.score = score;
            p.scale = (64.0 / total).min(1.0).powf(0.4);
        }
        p.mass_changed = false;
    }

    pub fn get_scale(&mut self, pi: usize) -> f64 {
        if self.players[pi].mass_changed {
            self.update_scale(pi);
        }
        self.players[pi].scale
    }

    pub fn spec_target(&self, pi: usize) -> Option<usize> {
        if let Some(prefix) = &self.players[pi].follow_prefix {
            if !prefix.is_empty() {
                let mut best: Option<usize> = None;
                let mut best_mass = -1.0;
                for (i, p) in self.players.iter().enumerate() {
                    if p.removed || p.cells.is_empty() || !p.name.starts_with(prefix.as_str()) {
                        continue;
                    }
                    let m = p.total_mass(self);
                    if m > best_mass {
                        best = Some(i);
                        best_mass = m;
                    }
                }
                if best.is_some() {
                    return best;
                }
            }
        }
        match self.largest {
            Some(t) if !self.players[t].removed && !self.players[t].cells.is_empty() => Some(t),
            _ => None,
        }
    }

    fn update_tick(&mut self, pi: usize) {
        if self.players[pi].removed {
            return;
        }
        // PacketHandler.process
        {
            let (sp, ej, q) = {
                let p = &mut self.players[pi];
                let r = (p.p_split, p.p_eject, p.p_q);
                p.p_split = false;
                p.p_eject = false;
                p.p_q = false;
                r
            };
            if sp {
                self.press_space(pi);
            }
            if ej {
                self.press_eject(pi);
            }
            if q {
                self.press_q(pi);
            }
            let p = &mut self.players[pi];
            if let Some(m) = p.mouse_in.take() {
                p.mouse = m;
            }
        }
        self.update_view_nodes(pi);
        let scale = self.get_scale(pi).max(self.cfg.min_scale);
        let mut w = self.cfg.view_base_x / scale / 2.0;
        let mut h = self.cfg.view_base_y / scale / 2.0;
        if self.full_view && self.players[pi].kind == PKind::Human {
            w = w.max(self.cfg.map * 1.5);
            h = h.max(self.cfg.map * 1.5);
        }
        let p = &mut self.players[pi];
        p.vminx = p.center.0 - w;
        p.vminy = p.center.1 - h;
        p.vmaxx = p.center.0 + w;
        p.vmaxy = p.center.1 + h;
        p.half_w = w;
        let (minx, miny, maxx, maxy) = (p.vminx, p.vminy, p.vmaxx, p.vmaxy);
        let mut nodes = std::mem::take(&mut self.players[pi].view_nodes);
        nodes.clear();
        let mut cand = std::mem::take(&mut self.cand);
        cand.clear();
        self.grid.query(minx, miny, maxx, maxy, &mut cand, self.ents.len());
        for &o in &cand {
            let e = &self.ents[o as usize];
            if e.removed || e.owner == pi as i32 || e.x - e.size >= maxx || e.x + e.size <= minx || e.y - e.size >= maxy || e.y + e.size <= miny {
                continue;
            }
            nodes.push(o);
        }
        self.cand = cand;
        nodes.extend(self.players[pi].cells.iter().copied());
        let ents = &self.ents;
        nodes.sort_unstable_by_key(|&o| ents[o as usize].id);
        self.players[pi].view_nodes = nodes;
    }

    fn update_view_nodes(&mut self, pi: usize) {
        let n = self.players[pi].cells.len();
        if !self.players[pi].spectating || n > 0 {
            if n > 0 {
                let (mut cx, mut cy) = (0.0, 0.0);
                for &h in &self.players[pi].cells {
                    let e = &self.ents[h as usize];
                    cx += e.x / n as f64;
                    cy += e.y / n as f64;
                }
                self.players[pi].center = (cx, cy);
            }
        } else {
            let target = self.spec_target(pi);
            if self.players[pi].free_roam || target.is_none() {
                // free roam = the whole map: centred, zoomed out to fit (arena.js)
                let (minx, miny, maxx, maxy) = self.border;
                let p = &mut self.players[pi];
                p.scale = self.cfg.spec_scale;
                p.center = ((minx + maxx) / 2.0, (miny + maxy) / 2.0);
            } else if let Some(t) = target {
                let c = self.players[t].center;
                let s = self.get_scale(t);
                self.set_center(pi, c.0, c.1);
                self.players[pi].scale = s;
            }
        }
    }

    fn set_center(&mut self, pi: usize, x: f64, y: f64) {
        let (minx, miny, maxx, maxy) = self.border;
        self.players[pi].center = (x.max(minx).min(maxx), y.max(miny).min(maxy));
    }

    fn update_leaderboard(&mut self) {
        let mut lb: Vec<usize> = Vec::new();
        for (i, p) in self.players.iter().enumerate() {
            if p.removed || p.cells.is_empty() || !p.connected {
                continue;
            }
            let mut j = 0;
            while j < lb.len() {
                if self.players[lb[j]].score < p.score {
                    break;
                }
                j += 1;
            }
            lb.insert(j, i);
        }
        self.largest = lb.first().copied();
        self.leaderboard = lb;
    }

    // ------------------------------------------------------------------ helpers for drivers

    pub fn view_box(&self, pi: usize) -> (f64, f64, f64, f64) {
        let p = &self.players[pi];
        (p.vminx, p.vminy, p.vmaxx, p.vmaxy)
    }
}

impl World {
    /// Recompute a player view (centre, view box, visible nodes) now instead of at the next tick; the vision env
    /// uses it to have a valid frame right after a respawn. Applies (and clears) pending input like a tick would.
    pub fn refresh_view(&mut self, pi: usize) {
        self.update_tick(pi);
    }
}
