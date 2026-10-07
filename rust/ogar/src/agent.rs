//! In-process rule-bot agents: what arena_client does over the network, without the network.
//! The Snapshot is built from the player's server-side view box exactly as the protocol-6 client
//! would reconstruct it (integer positions / sizes, cells in the view box).

use arena_client::client::AgentStats;
use arena_client::game::{Border, Cell, Kind, Snapshot};
use arena_client::policy::Policy;
use serde_json::{json, Value};

use crate::world::{PKind, World, T_EJECT, T_FOOD, T_VIRUS};

pub struct AgentDriver {
    pub pi: usize,
    pub name: String,
    pub policy: Policy,
    pub stats: AgentStats,
    pub sync: u64,
    alive: bool,
    n_cells: Option<usize>,
}

fn color_of(c: [u8; 3]) -> u32 {
    ((c[0] as u32) << 16) | ((c[1] as u32) << 8) | c[2] as u32
}

pub fn snapshot(w: &World, pi: usize) -> Snapshot {
    let p = &w.players[pi];
    let mine: Vec<Cell> = p
        .cells
        .iter()
        .map(|&h| {
            let e = &w.ents[h as usize];
            let mut c = Cell::new(e.id, (e.x as i32) as f64, (e.y as i32) as f64, (e.size as u32) as f64, Kind::Mine);
            c.color = color_of(e.color);
            c
        })
        .collect();
    let mut others = Vec::with_capacity(p.view_nodes.len());
    for &h in &p.view_nodes {
        let e = &w.ents[h as usize];
        if e.owner == pi as i32 {
            continue;
        }
        let kind = match e.kind {
            T_FOOD => Kind::Food,
            T_VIRUS => Kind::Virus,
            T_EJECT => Kind::Ejected,
            _ => Kind::Player,
        };
        let mut c = Cell::new(e.id, (e.x as i32) as f64, (e.y as i32) as f64, (e.size as u32) as f64, kind);
        c.color = color_of(e.color);
        others.push(c);
    }
    let b = w.border;
    let dead = mine.is_empty();
    Snapshot { mine, others, border: Border { left: b.0, right: b.2, top: b.1, bottom: b.3 }, dead }
}

impl AgentDriver {
    /// Add a player to the world and wrap it with a rule-bot policy; it spawns on its first step.
    pub fn new(w: &mut World, name: &str, spec: &Value, sync: u64) -> AgentDriver {
        let pi = w.add_player(PKind::Agent);
        w.players[pi].name = name.to_string();
        AgentDriver {
            pi,
            name: name.to_string(),
            policy: Policy::from_spec(spec),
            stats: AgentStats { name: name.to_string(), ..Default::default() },
            sync,
            alive: false,
            n_cells: None,
        }
    }

    fn account_events(&mut self, w: &World) {
        let pi = self.pi as i32;
        for k in &w.kill_log {
            if k.killer_owner == pi {
                self.stats.kills += 1;
            } else if k.victim_owner == pi {
                let mine: f64 = w.players[self.pi].cells.iter().map(|&h| w.ents[h as usize].mass()).sum();
                let loss = json!({
                    "by": w.players[k.killer_owner as usize].name, "kind": "player", "mass": k.victim_mass,
                    "killer_mass": k.killer_mass,
                    "killer_cells": k.killer_cells, "killer_total": k.killer_total,
                    "my_cells": w.players[self.pi].cells.len() + 1, "my_total": mine + k.victim_mass,
                    "tick": self.stats.ticks,
                });
                self.stats.losses.push(loss);
            }
        }
    }

    /// One decision step (every `sync` ticks): stats bookkeeping, then act or respawn.
    pub fn step(&mut self, w: &mut World) {
        self.account_events(w);
        let mut s = snapshot(w, self.pi);
        if s.dead {
            if self.stats.steps_alive > 0 && self.alive {
                self.stats.deaths += 1;
            }
            self.alive = false;
            self.policy.reset();
            let name = self.name.clone();
            w.join_game(self.pi, &name, None);
        } else {
            if !self.alive {
                self.stats.lives += 1;
            }
            self.alive = true;
            let mass = s.total_mass();
            let n = s.mine.len();
            if n >= self.n_cells.unwrap_or(n) + 3 {
                self.stats.popped += 1;
            }
            self.n_cells = Some(n);
            self.stats.steps_alive += 1;
            self.stats.mass_ticks += mass * self.sync as f64;
            self.stats.peak_mass = self.stats.peak_mass.max(mass);
            let a = self.policy.act(&mut s);
            let p = &mut w.players[self.pi];
            p.mouse_in = Some(((a.x as i32) as f64, (a.y as i32) as f64));
            if a.split {
                p.p_split = true;
            }
            if a.eject {
                p.p_eject = true;
            }
        }
        self.stats.ticks += self.sync;
    }

    pub fn finish(mut self, w: &World) -> AgentStats {
        let p = &w.players[self.pi];
        self.stats.virus_shots = p.shots as u64;
        self.stats.virus_pops = p.pops as u64;
        self.stats.virus_pop_mass = p.pop_mass as f64;
        self.stats.counters = self.policy.counters().clone();
        self.stats
    }
}
