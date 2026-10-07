//! `ogar serve --record DIR`: recordings of human play for learning from it (off by default).
//!
//! Three gzip JSONL files per session, `<kind>-<stamp>.jsonl.gz`:
//!   events - `eat` (every player cell eaten by another player: tick, eater / victim name, kind
//!            human|agent|bot, cell mass, owner total, cell count, positions, whether the victim burst
//!            on a virus just before), `top` once a second (top-5 by mass + every live human),
//!            and a `meta` line at the start.
//!   scenes - "loss scenes": an in-process agent lost >= 30% of its mass within ~0.3 s or died; its
//!            last ~3 s of decision steps (cells, nearby nodes, aim, split, plan mode) and the eaters.
//!            Only when a human was the eater, or for every eater with `--record-all`.
//!   humans - the humans' inputs and view per decision step (the arena/server/arena.js --record format,
//!            readable by rl/human_data.py): {t, n, m: [mouse x, y], s: splits, w: ejects,
//!            mine: [[id, x, y, size]], o: [[id, x, y, size, type, owner name]], b: [l, r, t, b]}.
//! Files are flushed every few seconds; a killed server leaves a readable (truncated) gzip.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

use flate2::write::GzEncoder;
use flate2::Compression;
use serde_json::{json, Value};

use crate::agent::AgentDriver;
use crate::world::{EatEvent, PKind, World, T_EJECT, T_FOOD, T_VIRUS};

const FLUSH_EVERY: Duration = Duration::from_secs(5);
const TOP_EVERY: u32 = 25;
/// seconds of agent history in a loss scene
const SCENE_SECS: f64 = 3.0;

pub struct Recorder {
    all_losses: bool,
    events: GzEncoder<File>,
    scenes: GzEncoder<File>,
    humans: GzEncoder<File>,
    presses: HashMap<usize, (u32, u32)>,
    /// eat events since the last decision step (loss scenes name their eaters from these)
    window: Vec<Value>,
    last_flush: Instant,
}

fn kind_str(k: PKind) -> &'static str {
    match k {
        PKind::Human => "human",
        PKind::Agent => "agent",
        PKind::Bot => "bot",
    }
}

fn open(dir: &Path, kind: &str, stamp: &str) -> Result<GzEncoder<File>, String> {
    let path = dir.join(format!("{kind}-{stamp}.jsonl.gz"));
    let f = File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(GzEncoder::new(f, Compression::fast()))
}

fn line(enc: &mut GzEncoder<File>, v: &Value) {
    let _ = enc.write_all(v.to_string().as_bytes());
    let _ = enc.write_all(b"\n");
}

impl Recorder {
    pub fn new(dir: &str, all_losses: bool, meta: Value) -> Result<Recorder, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
        let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let stamp = format!("{secs}");
        let d = Path::new(dir);
        let mut r = Recorder {
            all_losses,
            events: open(d, "events", &stamp)?,
            scenes: open(d, "scenes", &stamp)?,
            humans: open(d, "humans", &stamp)?,
            presses: HashMap::new(),
            window: Vec::new(),
            last_flush: Instant::now(),
        };
        let mut m = meta;
        m["e"] = json!("meta");
        m["unix"] = json!(secs);
        line(&mut r.events, &m);
        r.flush();
        Ok(r)
    }

    pub fn flush(&mut self) {
        let _ = self.events.flush();
        let _ = self.scenes.flush();
        let _ = self.humans.flush();
        self.last_flush = Instant::now();
    }

    /// A human pressed split (17) or eject (21).
    pub fn press(&mut self, pi: usize, op: u8) {
        let e = self.presses.entry(pi).or_default();
        if op == 17 {
            e.0 += 1;
        } else {
            e.1 += 1;
        }
    }

    fn who(w: &World, owner: i32, cell_mass: f64, total: f64, cells: usize, pos: (f64, f64)) -> Value {
        let p = &w.players[owner as usize];
        json!({"n": p.name, "k": kind_str(p.kind), "cm": cell_mass.round() as i64, "tot": total.round() as i64,
               "cells": cells, "x": pos.0.round() as i64, "y": pos.1.round() as i64})
    }

    fn eat_json(w: &World, e: &EatEvent) -> Value {
        let mut victim = Self::who(w, e.victim_owner, e.victim_cell_mass, e.victim_total, e.victim_cells, e.victim_pos);
        victim["last"] = json!(e.victim_cells <= 1);
        let popped = e.victim_pop_ago != u32::MAX && e.victim_pop_ago <= 75;
        json!({"e": "eat", "t": e.tick,
               "eater": Self::who(w, e.eater_owner, e.eater_cell_mass, e.eater_total, e.eater_cells, e.eater_pos),
               "victim": victim, "popped": popped,
               "pop_ago": if e.victim_pop_ago == u32::MAX { Value::Null } else { json!(e.victim_pop_ago) }})
    }

    /// After every world tick: log the eats of this tick and, once a second, the top of the board.
    pub fn on_tick(&mut self, w: &mut World) {
        let evs = std::mem::take(&mut w.eat_events);
        for e in &evs {
            let j = Self::eat_json(w, e);
            line(&mut self.events, &j);
            self.window.push(j);
        }
        if w.tick % TOP_EVERY == 0 {
            let mut alive: Vec<(f64, usize)> = Vec::new();
            for (i, p) in w.players.iter().enumerate() {
                if p.removed || p.cells.is_empty() {
                    continue;
                }
                alive.push((p.total_mass(w), i));
            }
            alive.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let row = |&(m, i): &(f64, usize)| {
                let p = &w.players[i];
                json!([p.name, kind_str(p.kind), m.round() as i64, p.cells.len(), p.boosted as u8])
            };
            let top: Vec<Value> = alive.iter().take(5).map(row).collect();
            let humans: Vec<Value> = alive.iter().filter(|&&(_, i)| w.players[i].kind == PKind::Human).map(row).collect();
            line(&mut self.events, &json!({"e": "top", "t": w.tick, "n": alive.len(), "top": top, "h": humans}));
        }
    }

    fn human_frames(&mut self, w: &World) {
        let b = w.border;
        for (pi, p) in w.players.iter().enumerate() {
            if p.kind != PKind::Human || p.removed || !p.connected || p.cells.is_empty() {
                continue;
            }
            let r = |v: f64| v.round() as i64;
            let mine: Vec<Value> = p
                .cells
                .iter()
                .map(|&h| {
                    let e = &w.ents[h as usize];
                    json!([e.id, r(e.x), r(e.y), r(e.size)])
                })
                .collect();
            let mut o: Vec<Value> = Vec::with_capacity(p.view_nodes.len());
            for &h in &p.view_nodes {
                let e = &w.ents[h as usize];
                if e.owner == pi as i32 {
                    continue;
                }
                let (ty, owner) = match e.kind {
                    T_FOOD => (1, ""),
                    T_VIRUS => (2, ""),
                    T_EJECT => (3, ""),
                    _ => (0, w.players[e.owner as usize].name.as_str()),
                };
                o.push(json!([e.id, r(e.x), r(e.y), r(e.size), ty, owner]));
            }
            let (s, wj) = self.presses.remove(&pi).unwrap_or((0, 0));
            line(
                &mut self.humans,
                &json!({"t": w.tick, "n": p.name, "m": [r(p.mouse.0), r(p.mouse.1)], "s": s, "w": wj,
                        "mine": mine, "o": o, "b": [b.0, b.2, b.1, b.3]}),
            );
        }
    }

    fn dump_scene(&mut self, w: &World, d: &AgentDriver, loss: (f64, f64, bool), eaters: &[Value]) {
        let Some(tr) = &d.trace else { return };
        let steps: Vec<Value> = tr
            .iter()
            .map(|r| {
                let others: Vec<Value> = r
                    .others
                    .iter()
                    .map(|o| {
                        let owner = if o[5] >= 0 { w.players[o[5] as usize].name.as_str() } else { "" };
                        json!([o[0], o[1], o[2], o[3], o[4], owner])
                    })
                    .collect();
                json!({"t": r.tick, "mine": r.mine, "o": others, "food": r.food, "aim": [r.aim.0.round(), r.aim.1.round()],
                       "split": r.split, "eject": r.eject, "mode": r.mode, "stage": r.stage, "target": r.target,
                       "split_ok": r.split_ok})
            })
            .collect();
        let b = w.border;
        let human = eaters.iter().any(|e| e["eater"]["k"] == "human");
        line(
            &mut self.scenes,
            &json!({"e": "scene", "t": w.tick, "agent": d.name, "kind": if loss.2 { "death" } else { "drop" },
                    "before": loss.0.round() as i64, "after": loss.1.round() as i64,
                    "eaters": eaters, "human_eater": human, "sync": d.sync,
                    "border": [b.0, b.2, b.1, b.3], "steps": steps}),
        );
    }

    /// After the agents' decision step: human frames, and loss scenes for agents that lost mass.
    pub fn after_step(&mut self, w: &World, drivers: &mut [AgentDriver]) {
        self.human_frames(w);
        for d in drivers.iter_mut() {
            let Some(loss) = d.loss.take() else { continue };
            let name = json!(w.players[d.pi].name);
            let eaters: Vec<Value> = self.window.iter().filter(|e| e["victim"]["n"] == name).cloned().collect();
            let human = eaters.iter().any(|e| e["eater"]["k"] == "human");
            if human || (self.all_losses && !eaters.is_empty()) {
                self.dump_scene(w, d, loss, &eaters);
            }
            if loss.2 {
                if let Some(t) = d.trace.as_mut() {
                    t.clear();
                }
            }
        }
        self.window.clear();
        if self.last_flush.elapsed() >= FLUSH_EVERY {
            self.flush();
        }
    }

    /// Steps of history an agent's ring buffer should keep for the scenes.
    pub fn trace_len(sync: u64) -> usize {
        (SCENE_SECS * 25.0 / sync as f64).ceil() as usize + 1
    }
}
