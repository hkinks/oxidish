//! Headless matches: world + server bots + rule-bot agents, no websockets.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use serde_json::{json, Value};

use crate::agent::{step_all, AgentDriver};
use crate::config::Config;
use crate::world::World;
use arena_client::client::AgentStats;

/// "route:4 champ=runs/x.json:4" -> [(agent name, policy spec)] (arena.match.parse_lineup, rule bots only).
pub fn parse_lineup(specs: &[String]) -> Result<Vec<(String, Value)>, String> {
    let mut lineup = Vec::new();
    for spec in specs {
        let (head, tail) = match spec.rsplit_once(':') {
            Some((h, t)) => (h, t),
            None => ("", spec.as_str()),
        };
        let (key, count) = if !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) { (head, tail) } else { (spec.as_str(), "1") };
        let (key, policy) = if let Some((k, path)) = key.split_once('=') {
            if path.ends_with(".pt") || path.contains("jev:") || path.contains("hybrid") || path.contains("guard") {
                return Err(format!("{spec}: NN / Jev policies are not supported (rule bots only)"));
            }
            let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
            (k, serde_json::from_str::<Value>(&text).map_err(|e| format!("{path}: {e}"))?)
        } else if key == "route" || key == "field" {
            (key, json!({ "farm": key }))
        } else if key == "apos" {
            (key, json!({ "kind": "apos" }))
        } else {
            return Err(format!("unknown policy {key:?}; known: route, field or name=file.json"));
        };
        let n: usize = if count.is_empty() { 1 } else { count.parse().map_err(|_| format!("bad count in {spec}"))? };
        for i in 0..n {
            lineup.push((format!("{key}-{i}"), policy.clone()));
        }
    }
    Ok(lineup)
}

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
