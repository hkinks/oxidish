//! Headless matches: world + server bots + rule-bot agents, no websockets.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use serde_json::Value;

use crate::agent::{step_all, AgentDriver};
use crate::config::Config;
use crate::world::World;
use arena_client::client::AgentStats;

pub use arena_client::lineup::parse_lineup;

#[derive(Clone)]
pub struct MatchOpts {
    pub ticks: u64,
    pub sync: u64,
    pub bots: usize,
    pub map: f64,
    pub food_mult: f64,
}

/// One match; returns each agent's stats (same accounting as the lockstep client).
pub fn run_match(lineup: &[(String, Value)], o: &MatchOpts, seed: u64) -> Vec<AgentStats> {
    let cfg = Config::new(o.map, o.bots, o.food_mult);
    let mut w = World::new(cfg, seed);
    let mut agents: Vec<AgentDriver> = lineup.iter().map(|(n, s)| AgentDriver::new(&mut w, n, s, o.sync)).collect();
    let mut started: Option<u32> = None;
    loop {
        w.tick();
        if w.tick as u64 % o.sync != 0 {
            continue;
        }
        let start = *started.get_or_insert(w.tick);
        step_all(&mut agents, &mut w);
        w.kill_log.clear();
        if (w.tick - start) as u64 >= o.ticks {
            break;
        }
    }
    agents.into_iter().map(|a| a.finish(&w)).collect()
}

/// `matches` seeded matches (seed, seed+1, ...) on `threads` worker threads, one thread per match.
pub fn run_many(lineup: &[(String, Value)], o: &MatchOpts, seed: u64, matches: usize, threads: usize) -> Vec<Value> {
    let next = AtomicUsize::new(0);
    let rows: Mutex<Vec<(usize, Vec<Value>)>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= matches {
                    break;
                }
                let sd = seed + i as u64;
                let stats = run_match(lineup, o, sd);
                let r: Vec<Value> = stats.iter().map(|s| s.row(sd)).collect();
                rows.lock().unwrap().push((i, r));
                eprintln!("  match {}/{} done", i + 1, matches);
            });
        }
    });
    let mut all = rows.into_inner().unwrap();
    all.sort_by_key(|(i, _)| *i);
    all.into_iter().flat_map(|(_, r)| r).collect()
}
