//! Lockstep MultiOgar client (port of arena/client.py): cell table from UpdateNodes deltas,
//! per-step policy decision, stats.

use std::collections::{HashMap, HashSet};

use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio_tungstenite::tungstenite::Message;

use crate::game::{Border, Cell, Kind, Snapshot};
use crate::policy::{Counters, Policy};

const OP_AGENT: u8 = 0xC8;
const OP_STEP: u8 = 0xC9;

#[derive(Default, Debug)]
pub struct AgentStats {
    pub name: String,
    pub steps_alive: u64,
    pub mass_ticks: f64,
    pub peak_mass: f64,
    pub deaths: u64,
    pub kills: u64,
    pub lives: u64,
    pub ticks: u64,
    pub counters: Counters,
    pub losses: Vec<Value>,
    pub popped: u64,
    /// virus shots fired / enemy cells popped by our viruses / mass of the popped (ogar engine only)
    pub virus_shots: u64,
    pub virus_pops: u64,
    pub virus_pop_mass: f64,
    /// ogar engine: split decisions taken / kills within 3 s of one (and their mass) / eaten cells of players that had
    /// >= 3 cells at that moment (popped or split apart) and their mass
    pub splits: u64,
    pub split_kills: u64,
    pub split_kill_mass: f64,
    pub harvest_kills: u64,
    pub harvest_mass: f64,
}

impl AgentStats {
    pub fn avg_mass(&self) -> f64 {
        self.mass_ticks / (self.ticks.max(1) as f64)
    }

    /// One row in the format of arena.tournament's `--out` files.
    pub fn row(&self, seed: u64) -> Value {
        let avg = self.avg_mass();
        let counters: serde_json::Map<String, Value> =
            self.counters.0.iter().map(|(k, v)| (k.clone(), json!(v))).collect();
        json!({
            "policy": self.name.rsplit_once('-').map_or(self.name.as_str(), |(a, _)| a),
            "seed": seed,
            "avg_mass": avg,
            "log_mass": avg.max(1.0).ln(),
            "peak_mass": self.peak_mass,
            "deaths": self.deaths,
            "kills": self.kills,
            "losses": self.losses,
            "counters": counters,
            "popped": self.popped,
            "virus_shots": self.virus_shots,
            "virus_pops": self.virus_pops,
            "virus_pop_mass": self.virus_pop_mass,
            "ticks": self.ticks,
            "splits": self.splits,
            "split_kills": self.split_kills,
            "split_kill_mass": self.split_kill_mass,
            "harvest_kills": self.harvest_kills,
            "harvest_mass": self.harvest_mass,
        })
    }
}

struct Slot {
    alive: bool,
    c: Cell,
}

/// Cell table that keeps Python dict iteration order (insertion order, removals leave the rest in place).
pub struct World {
    slots: Vec<Slot>,
    index: HashMap<u32, usize>,
    names: HashMap<u32, String>,
    mine: HashSet<u32>,
    border: Border,
}

impl World {
    pub fn new() -> World {
        World {
            slots: Vec::new(),
            index: HashMap::new(),
            names: HashMap::new(),
            mine: HashSet::new(),
            border: Border { left: -1.0, right: 1.0, top: -1.0, bottom: 1.0 },
        }
    }

    fn get(&self, id: u32) -> Option<&Cell> {
        self.index.get(&id).map(|&i| &self.slots[i].c)
    }

    fn remove(&mut self, id: u32) {
        if let Some(i) = self.index.remove(&id) {
            self.slots[i].alive = false;
            self.names.remove(&id);
            if self.slots.len() > 64 && self.index.len() * 2 < self.slots.len() {
                self.compact();
            }
        }
    }

    fn compact(&mut self) {
        self.slots.retain(|s| s.alive);
        for (i, s) in self.slots.iter().enumerate() {
            self.index.insert(s.c.id, i);
        }
    }

    fn clear(&mut self) {
        self.slots.clear();
        self.index.clear();
        self.names.clear();
    }

    fn live(&self) -> impl Iterator<Item = &Cell> {
        self.slots.iter().filter(|s| s.alive).map(|s| &s.c)
    }

    pub fn snapshot(&mut self) -> Snapshot {
        let mut mine = Vec::new();
        let mut others = Vec::with_capacity(self.index.len());
        for s in self.slots.iter_mut().filter(|s| s.alive) {
            if self.mine.contains(&s.c.id) {
                s.c.kind = Kind::Mine; // sticky, like the Python client
                mine.push(s.c);
            } else {
                others.push(s.c);
            }
        }
        let dead = mine.is_empty();
        Snapshot { mine, others, border: self.border, dead }
    }
}

fn rd<const N: usize>(buf: &[u8], i: usize) -> Option<[u8; N]> {
    buf.get(i..i + N)?.try_into().ok()
}

fn read_str(buf: &[u8], i: usize) -> Option<(String, usize)> {
    let end = i + buf.get(i..)?.iter().position(|&b| b == 0)?;
    Some((String::from_utf8_lossy(&buf[i..end]).into_owned(), end + 1))
}

fn kind_of(flags: u8) -> Kind {
    if flags & 0x80 != 0 {
        Kind::Food
    } else if flags & 0x11 != 0 {
        Kind::Virus
    } else if flags & 0x20 != 0 {
        Kind::Ejected
    } else {
        Kind::Player
    }
}

pub struct AgentClient {
    pub name: String,
    policy: Policy,
    end_tick: u64,
    sync: u64,
    pub stats: AgentStats,
    world: World,
    alive: bool,
    n_cells: Option<usize>,
}

impl AgentClient {
    pub fn new(name: &str, policy: Policy, end_tick: u64, sync: u64) -> AgentClient {
        AgentClient {
            name: name.to_string(),
            policy,
            end_tick,
            sync,
            stats: AgentStats { name: name.to_string(), ..Default::default() },
            world: World::new(),
            alive: false,
            n_cells: None,
        }
    }

    fn on_update(&mut self, buf: &[u8]) -> Option<()> {
        let mut i = 1;
        let n = u16::from_le_bytes(rd(buf, i)?) as usize;
        i += 2;
        for _ in 0..n {
            let killer = u32::from_le_bytes(rd(buf, i)?);
            let killed = u32::from_le_bytes(rd(buf, i + 4)?);
            i += 8;
            let victim = self.world.get(killed).copied();
            let w = &self.world;
            if let Some(v) = victim {
                if w.mine.contains(&killer) && v.kind == Kind::Player && !w.mine.contains(&killed) {
                    self.stats.kills += 1;
                }
                if w.mine.contains(&killed) && !w.mine.contains(&killer) {
                    self.record_loss(killer, &v);
                }
            }
            self.world.remove(killed);
            self.world.mine.remove(&killed);
        }
        loop {
            let cid = u32::from_le_bytes(rd(buf, i)?);
            i += 4;
            if cid == 0 {
                break;
            }
            let x = i32::from_le_bytes(rd(buf, i)?);
            let y = i32::from_le_bytes(rd(buf, i + 4)?);
            let size = u16::from_le_bytes(rd(buf, i + 8)?);
            let flags = *buf.get(i + 10)?;
            i += 11;
            let mut color = 0u32;
            if flags & 0x02 != 0 {
                color = ((*buf.get(i)? as u32) << 16) | ((*buf.get(i + 1)? as u32) << 8) | (*buf.get(i + 2)? as u32);
                i += 3;
            }
            if flags & 0x04 != 0 {
                i = read_str(buf, i)?.1;
            }
            let mut name = None;
            if flags & 0x08 != 0 {
                let (nm, ni) = read_str(buf, i)?;
                i = ni;
                name = Some(nm);
            }
            let w = &mut self.world;
            if let Some(&slot) = w.index.get(&cid) {
                let c = &mut w.slots[slot].c;
                c.x = x as f64;
                c.y = y as f64;
                c.size = size as f64;
                if color != 0 {
                    c.color = color;
                }
            } else {
                w.index.insert(cid, w.slots.len());
                let mut cell = Cell::new(cid, x as f64, y as f64, size as f64, kind_of(flags));
                cell.color = color;
                w.slots.push(Slot { alive: true, c: cell });
            }
            if let Some(nm) = name {
                if !nm.is_empty() {
                    w.names.insert(cid, nm);
                }
            }
        }
        let n = u16::from_le_bytes(rd(buf, i)?) as usize;
        i += 2;
        for _ in 0..n {
            let cid = u32::from_le_bytes(rd(buf, i)?);
            i += 4;
            self.world.remove(cid);
            self.world.mine.remove(&cid);
        }
        Some(())
    }

    /// Who ate one of our cells: killer name and size, and whether either side was split.
    fn record_loss(&mut self, killer_id: u32, victim: &Cell) {
        let w = &self.world;
        let killer = w.get(killer_id);
        let name = w.names.get(&killer_id).cloned().unwrap_or_default();
        let kind = killer.map_or("unseen", |k| k.kind.as_str());
        let mine: Vec<&Cell> = w.live().filter(|c| w.mine.contains(&c.id)).collect();
        let theirs: Vec<&Cell> = if name.is_empty() {
            vec![]
        } else {
            w.live()
                .filter(|c| !w.mine.contains(&c.id) && w.names.get(&c.id).map_or(false, |n| *n == name))
                .collect()
        };
        let by = if name.is_empty() { kind.to_string() } else { name };
        let loss = json!({
            "by": by, "kind": kind, "mass": victim.mass(),
            "killer_mass": killer.map_or(0.0, |k| k.mass()),
            "killer_cells": theirs.len(), "killer_total": theirs.iter().map(|c| c.mass()).sum::<f64>(),
            "my_cells": mine.len(), "my_total": mine.iter().map(|c| c.mass()).sum::<f64>(),
            "tick": self.stats.ticks,
        });
        self.stats.losses.push(loss);
    }

    /// Advance stats and return the packets to send for this step.
    fn on_step(&mut self, tick: u32) -> Vec<Vec<u8>> {
        let mut s = self.world.snapshot();
        let mut out: Vec<Vec<u8>> = Vec::new();
        if s.dead {
            if self.stats.steps_alive > 0 && self.alive {
                self.stats.deaths += 1;
            }
            self.alive = false;
            self.policy.reset();
            let mut p = vec![0u8];
            p.extend_from_slice(self.name.as_bytes());
            p.push(0);
            out.push(p);
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
            let mut p = vec![0x10u8];
            p.extend_from_slice(&(a.x as i32).to_le_bytes());
            p.extend_from_slice(&(a.y as i32).to_le_bytes());
            p.extend_from_slice(&0u32.to_le_bytes());
            out.push(p);
            if a.split {
                out.push(vec![0x11]);
            }
            if a.eject {
                out.push(vec![0x15]);
            }
        }
        self.stats.ticks += self.sync;
        let mut p = vec![OP_STEP];
        p.extend_from_slice(&tick.to_le_bytes());
        out.push(p);
        out
    }

    /// Apply one server packet; returns the tick for step packets.
    fn handle(&mut self, msg: &[u8]) -> Option<u32> {
        match *msg.first()? {
            0x10 => {
                self.on_update(msg);
            }
            0x20 => {
                if let Some(b) = rd(msg, 1) {
                    self.world.mine.insert(u32::from_le_bytes(b));
                }
            }
            0x12 => {
                self.world.clear();
                self.world.mine.clear();
            }
            0x14 => self.world.mine.clear(),
            0x40 => {
                let f = |o: usize| rd(msg, o).map(f64::from_le_bytes);
                if let (Some(l), Some(t), Some(r), Some(b)) = (f(1), f(9), f(17), f(25)) {
                    self.world.border = Border { left: l, top: t, right: r, bottom: b };
                }
            }
            OP_STEP => return rd(msg, 1).map(u32::from_le_bytes),
            _ => {}
        }
        None
    }

    pub async fn run(mut self, url: &str) -> Result<AgentStats, Box<dyn std::error::Error>> {
        let (ws, _) = tokio_tungstenite::connect_async_with_config(url, None, true).await?;
        let (mut tx, mut rx) = ws.split();
        for hello in [vec![254u8, 6, 0, 0, 0], vec![255, 1, 0, 0, 0], vec![OP_AGENT]] {
            tx.send(Message::Binary(hello.into())).await?;
        }
        let mut started_at: Option<u32> = None;
        while let Some(msg) = rx.next().await {
            let data = match msg? {
                Message::Binary(b) => b,
                Message::Close(_) => break,
                _ => continue,
            };
            if let Some(tick) = self.handle(&data) {
                let start = *started_at.get_or_insert(tick);
                for packet in self.on_step(tick) {
                    tx.feed(Message::Binary(packet.into())).await?;
                }
                tx.flush().await?;
                if (tick - start) as u64 >= self.end_tick {
                    break;
                }
            }
        }
        let _ = tx.close().await;
        self.stats.counters = self.policy.counters().clone();
        Ok(self.stats)
    }
}
