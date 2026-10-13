//! `ogar endless`: the fast endless evaluator (training/eval/endless.py without one `ogar serve` process per world).
//!
//! N independent endless worlds (seeds seed_base+i) live in ONE process and are stepped in lockstep on worker threads.
//! Every world is built and driven exactly as `ogar serve --seed S --agents ...` does (same World::new, same agent
//! order and names, tick, then a decision step on every `sync`-th tick, then the `[players]` sample), so a rule-bot
//! lineup yields the very same per-world samples. Vision agents of ALL worlds decide in one batched forward pass per
//! decision tick (workers render frames, the coordinator runs the net, workers apply the actions).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
#[cfg(feature = "vision-net")]
use std::sync::{atomic::AtomicBool, Barrier};
use std::time::Instant;

use serde_json::{json, Value};

use crate::agent::AgentDriver;
use crate::config::Config;
use crate::matches::parse_lineup;
use crate::world::World;

pub struct EndlessOpts {
    pub agents: Vec<String>,
    pub worlds: usize,
    pub game_min: f64,
    pub seed_base: u64,
    pub threads: usize,
    pub bots: usize,
    pub map: f64,
    pub food_mult: f64,
    pub sync: u64,
    pub log_s: f64,
}

struct WorldRun {
    w: World,
    drivers: Vec<AgentDriver>,
    samples: Vec<Value>,
    done: bool,
    log_every: u32,
    end_tick: u32,
    sync: u64,
}

impl WorldRun {
    fn new(o: &EndlessOpts, lineup: &[(String, Value)], seed: u64) -> WorldRun {
        let mut w = World::new(Config::new(o.map, o.bots, o.food_mult), seed);
        let mut drivers: Vec<AgentDriver> = Vec::new();
        for (name, spec) in lineup {
            // same naming as `ogar serve` (server.rs add_agents)
            let key = name.rsplit_once('-').map_or(name.as_str(), |(k, _)| k).to_string();
            let mut idx = 0;
            while drivers.iter().any(|d| d.name == format!("{key}-{idx}")) {
                idx += 1;
            }
            drivers.push(AgentDriver::new(&mut w, &format!("{key}-{idx}"), spec, o.sync));
        }
        WorldRun {
            w,
            drivers,
            samples: Vec::new(),
            done: false,
            log_every: (o.log_s * 25.0).round() as u32,
            end_tick: (o.game_min * 1500.0) as u32,
            sync: o.sync,
        }
    }

    /// server.rs `log_players`
    fn log_if_due(&mut self) {
        if self.log_every == 0 || self.w.tick % self.log_every != 0 {
            return;
        }
        let mut parts = Vec::new();
        for p in &self.w.players {
            if p.removed || p.cells.is_empty() {
                continue;
            }
            let mut row = json!({"n": p.name, "m": p.total_mass(&self.w).round() as i64, "c": p.cells.len()});
            if p.boosted {
                row["b"] = json!(1);
            }
            parts.push(row);
        }
        self.samples.push(json!({"t": self.w.tick, "p": parts}));
        if self.w.tick >= self.end_tick {
            self.done = true;
        }
    }

    /// Ticks up to (and including) the next decision tick, sampling on the way; then renders the vision frames.
    fn pre(&mut self) {
        loop {
            self.w.tick();
            if self.w.tick as u64 % self.sync == 0 {
                break;
            }
            self.log_if_due();
        }
        #[cfg(feature = "vision-net")]
        for d in self.drivers.iter_mut() {
            if let Some(v) = d.vision.as_mut() {
                let s = crate::agent::snapshot(&self.w, d.pi);
                if !s.dead {
                    v.prepare(&self.w, d.pi, &s);
                }
            }
        }
    }

    /// The decision step of every driver (vision answers already set), then the sample of that tick.
    fn post(&mut self) {
        for d in self.drivers.iter_mut() {
            d.step(&mut self.w);
        }
        self.w.kill_log.clear();
        self.log_if_due();
    }
}

#[derive(Default)]
pub struct EndlessStats {
    pub wall_s: f64,
    pub rounds: u64,
    pub infer_calls: u64,
    pub infer_frames: u64,
    pub infer_s: f64,
    /// per world: each agent's `AgentStats::row` (deaths, kills, splits, split kills, virus pops, harvest mass, ...)
    pub agents: Vec<Vec<Value>>,
}

/// Run all worlds; returns each world's `[players]` samples (`{t, p:[{n,m,c}]}`), up to the first sample at or past the end.
pub fn run(o: &EndlessOpts) -> Result<(Vec<Vec<Value>>, EndlessStats), String> {
    let lineup = parse_lineup(&o.agents)?;
    if lineup.is_empty() {
        return Err("need --agents".into());
    }
    let t0 = Instant::now();
    let worlds: Vec<Mutex<WorldRun>> = (0..o.worlds).map(|i| Mutex::new(WorldRun::new(o, &lineup, o.seed_base + i as u64))).collect();
    let has_vision = worlds.iter().any(|m| m.lock().unwrap().drivers.iter().any(is_vision));
    let threads = o.threads.clamp(1, o.worlds.max(1));
    let mut st = EndlessStats::default();
    if !has_vision {
        // independent worlds: no synchronisation at all, a queue of worlds over the worker threads
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    if i >= worlds.len() {
                        break;
                    }
                    let mut ws = worlds[i].lock().unwrap();
                    while !ws.done {
                        ws.pre();
                        ws.post();
                    }
                    eprintln!("  world {}/{} done", i + 1, worlds.len());
                });
            }
        });
    } else {
        run_lockstep(&worlds, threads, &mut st);
    }
    st.wall_s = t0.elapsed().as_secs_f64();
    let mut all = Vec::new();
    for (i, m) in worlds.into_iter().enumerate() {
        let ws = m.into_inner().unwrap();
        let seed = o.seed_base + i as u64;
        let w = ws.w;
        st.agents.push(ws.drivers.into_iter().map(|d| { let name = d.name.clone(); let mut r = d.finish(&w).row(seed); r["name"] = json!(name); r }).collect());
        all.push(ws.samples);
    }
    Ok((all, st))
}

fn is_vision(d: &AgentDriver) -> bool {
    #[cfg(feature = "vision-net")]
    return d.vision.is_some();
    #[cfg(not(feature = "vision-net"))]
    {
        let _ = d;
        false
    }
}

#[cfg(feature = "vision-net")]
fn run_lockstep(worlds: &[Mutex<WorldRun>], threads: usize, st: &mut EndlessStats) {
    let barrier = Barrier::new(threads + 1);
    let finished = AtomicBool::new(false);
    std::thread::scope(|s| {
        for t in 0..threads {
            let (barrier, finished) = (&barrier, &finished);
            s.spawn(move || {
                let mine: Vec<&Mutex<WorldRun>> = worlds.iter().skip(t).step_by(threads).collect();
                for m in &mine {
                    m.lock().unwrap().pre();
                }
                loop {
                    barrier.wait(); // all frames rendered
                    barrier.wait(); // the coordinator has answered them
                    if finished.load(Ordering::SeqCst) {
                        break;
                    }
                    for m in &mine {
                        let mut ws = m.lock().unwrap();
                        if !ws.done {
                            ws.post();
                            if !ws.done {
                                ws.pre();
                            }
                        }
                    }
                }
            });
        }
        let mut last = Instant::now();
        loop {
            barrier.wait();
            if worlds.iter().all(|m| m.lock().unwrap().done) {
                finished.store(true, Ordering::SeqCst);
                barrier.wait();
                break;
            }
            let t = Instant::now();
            let mut guards: Vec<_> = worlds.iter().map(|m| m.lock().unwrap()).collect();
            let tick = guards[0].w.tick;
            let mut bots: Vec<&mut crate::visionbot::VisionBot> = Vec::new();
            for g in guards.iter_mut() {
                for d in g.drivers.iter_mut() {
                    if let Some(v) = d.vision.as_mut() {
                        if v.is_prepared() {
                            bots.push(v);
                        }
                    }
                }
            }
            let n = crate::visionbot::infer_batch(&mut bots);
            drop(bots);
            drop(guards);
            st.infer_s += t.elapsed().as_secs_f64();
            st.infer_calls += (n > 0) as u64;
            st.infer_frames += n as u64;
            st.rounds += 1;
            if last.elapsed().as_secs() >= 30 {
                last = Instant::now();
                eprintln!("  tick {tick}: {} rounds, {:.2} ms/infer call, {:.1} frames/call", st.rounds,
                          st.infer_s * 1e3 / st.infer_calls.max(1) as f64, st.infer_frames as f64 / st.infer_calls.max(1) as f64);
            }
            barrier.wait();
        }
    });
}

#[cfg(not(feature = "vision-net"))]
fn run_lockstep(_: &[Mutex<WorldRun>], _: usize, _: &mut EndlessStats) {
    unreachable!("no vision agents without the vision-net feature")
}
