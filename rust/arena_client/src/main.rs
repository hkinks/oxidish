//! Rust lockstep client + rule bot for the agar.io arena.
//!
//!   arena_client run --port 9601 --agents route:4 champ=runs/final_champion.json:4 --ticks 3000 --seed 1 --out rows.json
//!   arena_client parity dump.jsonl       # replay Python-recorded snapshots, compare decisions
//!   arena_client bench dump.jsonl        # decisions per second on the recorded snapshots

mod client;
mod farm;
mod game;
mod params;
mod policy;
mod steer;

use std::collections::HashMap;
use std::time::Instant;

use serde_json::{json, Value};

use client::{AgentClient, AgentStats};
use game::{Border, Cell, Kind, Snapshot};
use policy::HeuristicPolicy;

/// "route:4 champ=runs/x.json:4" -> [(agent name, policy spec)] (arena.match.parse_lineup, rule bots only).
fn parse_lineup(specs: &[String]) -> Result<Vec<(String, Value)>, String> {
    let mut lineup = Vec::new();
    for spec in specs {
        let (head, tail) = match spec.rsplit_once(':') {
            Some((h, t)) => (h, t),
            None => ("", spec.as_str()),
        };
        let (key, count) = if !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) {
            (head, tail)
        } else {
            (spec.as_str(), "1")
        };
        let (key, policy) = if let Some((k, path)) = key.split_once('=') {
            if path.ends_with(".pt") || path.contains("jev:") || path.contains("hybrid") || path.contains("guard") {
                return Err(format!("{spec}: NN / Jev policies are not supported by the Rust runner"));
            }
            let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
            (k, serde_json::from_str::<Value>(&text).map_err(|e| format!("{path}: {e}"))?)
        } else if key == "route" || key == "field" {
            (key, json!({ "farm": key }))
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

struct Opts {
    host: String,
    port: u16,
    agents: Vec<String>,
    ticks: u64,
    sync: u64,
    seed: u64,
    out: Option<String>,
}

fn parse_opts(args: &[String]) -> Result<Opts, String> {
    let mut o = Opts { host: "127.0.0.1".into(), port: 0, agents: vec![], ticks: 3000, sync: 2, seed: 1, out: None };
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        let mut val = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            args.get(*i).cloned().ok_or(format!("{key} needs a value"))
        };
        match key {
            "--host" => o.host = val(&mut i)?,
            "--port" => o.port = val(&mut i)?.parse().map_err(|_| "bad --port")?,
            "--ticks" => o.ticks = val(&mut i)?.parse().map_err(|_| "bad --ticks")?,
            "--sync" => o.sync = val(&mut i)?.parse().map_err(|_| "bad --sync")?,
            "--seed" => o.seed = val(&mut i)?.parse().map_err(|_| "bad --seed")?,
            "--out" => o.out = Some(val(&mut i)?),
            "--agents" => {
                while i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    i += 1;
                    o.agents.push(args[i].clone());
                }
            }
            _ => return Err(format!("unknown option {key}")),
        }
        i += 1;
    }
    if o.port == 0 || o.agents.is_empty() {
        return Err("need --port and --agents".into());
    }
    Ok(o)
}

fn print_table(stats: &[AgentStats], wall: f64, ticks: u64) {
    println!("\n{:<10} {:>9} {:>7} {:>7} {:>6}", "agent", "avg mass", "peak", "deaths", "kills");
    let mut sorted: Vec<&AgentStats> = stats.iter().collect();
    sorted.sort_by(|a, b| b.avg_mass().partial_cmp(&a.avg_mass()).unwrap());
    for s in sorted {
        println!("{:<10} {:9.1} {:7.0} {:7} {:6}", s.name, s.avg_mass(), s.peak_mass, s.deaths, s.kills);
    }
    println!("\n{} ticks ({:.1} game-min) in {:.1}s wall = {:.1}x realtime", ticks,
             ticks as f64 * 0.04 / 60.0, wall, ticks as f64 * 0.04 / wall);
}

fn run(args: &[String]) -> Result<(), String> {
    let o = parse_opts(args)?;
    let lineup = parse_lineup(&o.agents)?;
    let url = format!("ws://{}:{}", o.host, o.port);
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().map_err(|e| e.to_string())?;
    let started = Instant::now();
    let results: Vec<Result<AgentStats, String>> = rt.block_on(async {
        let handles: Vec<_> = lineup
            .iter()
            .map(|(name, spec)| {
                let c = AgentClient::new(name, HeuristicPolicy::from_spec(spec), o.ticks, o.sync);
                let url = url.clone();
                tokio::spawn(async move { c.run(&url).await.map_err(|e| e.to_string()) })
            })
            .collect();
        let mut out = Vec::new();
        for h in handles {
            out.push(h.await.map_err(|e| e.to_string()).and_then(|r| r));
        }
        out
    });
    let wall = started.elapsed().as_secs_f64();
    let mut stats = Vec::new();
    for r in results {
        stats.push(r?);
    }
    print_table(&stats, wall, o.ticks);
    if let Some(path) = o.out {
        let rows: Vec<Value> = stats.iter().map(|s| s.row(o.seed)).collect();
        std::fs::write(&path, serde_json::to_string(&rows).unwrap()).map_err(|e| e.to_string())?;
    }
    Ok(())
}

// ---------------------------------------------------------------- parity / bench

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}

fn parse_snapshot(v: &Value) -> Snapshot {
    let mine = v["mine"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| Cell::new(c[0].as_u64().unwrap() as u32, num(&c[1]), num(&c[2]), num(&c[3]), Kind::Mine))
        .collect();
    let others = v["others"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| Cell::new(c[0].as_u64().unwrap() as u32, num(&c[1]), num(&c[2]), num(&c[3]), Kind::parse(c[4].as_str().unwrap())))
        .collect();
    let b = &v["b"];
    Snapshot { mine, others, border: Border { left: num(&b[0]), top: num(&b[1]), right: num(&b[2]), bottom: num(&b[3]) }, dead: false }
}

enum Event {
    Agent(usize, Value),
    Reset(usize),
    Step(usize, Box<Snapshot>, [f64; 2], [bool; 2]),
}

fn load_dump(path: &str) -> Result<Vec<Event>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut ev = Vec::new();
    for line in text.lines().filter(|l| !l.is_empty()) {
        let v: Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
        let a = v["a"].as_u64().unwrap() as usize;
        match v["t"].as_str().unwrap() {
            "agent" => ev.push(Event::Agent(a, v["spec"].clone())),
            "r" => ev.push(Event::Reset(a)),
            _ => {
                let o = &v["out"];
                ev.push(Event::Step(a, Box::new(parse_snapshot(&v)), [num(&o[0]), num(&o[1])],
                                    [o[2].as_bool().unwrap_or(false), o[3].as_bool().unwrap_or(false)]));
            }
        }
    }
    Ok(ev)
}

fn parity(path: &str) -> Result<(), String> {
    let events = load_dump(path)?;
    let mut pol: HashMap<usize, HeuristicPolicy> = HashMap::new();
    let (mut steps, mut bad_aim, mut bad_flag, mut max_d) = (0u64, 0u64, 0u64, 0.0f64);
    let mut first_bad: Vec<String> = Vec::new();
    for e in events {
        match e {
            Event::Agent(a, spec) => {
                pol.insert(a, HeuristicPolicy::from_spec(&spec));
            }
            Event::Reset(a) => pol.get_mut(&a).unwrap().reset(),
            Event::Step(a, mut s, aim, flags) => {
                let o = pol.get_mut(&a).unwrap().act(&mut s);
                steps += 1;
                let d = (o.x - aim[0]).abs().max((o.y - aim[1]).abs());
                max_d = max_d.max(d);
                if d > 1e-3 {
                    bad_aim += 1;
                    if first_bad.len() < 5 {
                        first_bad.push(format!("step {steps} agent {a}: rust ({:.4},{:.4}) python ({:.4},{:.4})", o.x, o.y, aim[0], aim[1]));
                    }
                }
                if o.split != flags[0] || o.eject != flags[1] {
                    bad_flag += 1;
                    if first_bad.len() < 5 {
                        first_bad.push(format!("step {steps} agent {a}: rust split/eject {}/{} python {}/{}", o.split, o.eject, flags[0], flags[1]));
                    }
                }
            }
        }
    }
    println!("parity: {steps} decisions, aim mismatches (>1e-3): {bad_aim}, split/eject mismatches: {bad_flag}, max |aim diff| {max_d:.3e}");
    for l in first_bad {
        println!("  {l}");
    }
    if bad_aim + bad_flag > 0 { Err("parity failed".into()) } else { Ok(()) }
}

fn bench(path: &str, repeat: usize) -> Result<(), String> {
    let events = load_dump(path)?;
    let mut total = 0.0f64;
    let mut n = 0u64;
    for _ in 0..repeat {
        let mut pol: HashMap<usize, HeuristicPolicy> = HashMap::new();
        for e in &events {
            match e {
                Event::Agent(a, spec) => {
                    pol.insert(*a, HeuristicPolicy::from_spec(spec));
                }
                Event::Reset(a) => pol.get_mut(a).unwrap().reset(),
                Event::Step(a, s, _, _) => {
                    let mut s = (**s).clone();
                    let p = pol.get_mut(a).unwrap();
                    let t = Instant::now();
                    std::hint::black_box(p.act(&mut s));
                    total += t.elapsed().as_secs_f64();
                    n += 1;
                }
            }
        }
    }
    println!("rust: {n} decisions in {total:.3}s = {:.0} decisions/s ({:.1} us each)", n as f64 / total, total / n as f64 * 1e6);
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("parity") => parity(args.get(1).map(String::as_str).unwrap_or("dump.jsonl")),
        Some("bench") => bench(args.get(1).map(String::as_str).unwrap_or("dump.jsonl"),
                               args.get(2).and_then(|r| r.parse().ok()).unwrap_or(3)),
        Some("run") => run(&args[1..]),
        _ => Err("usage: arena_client run|parity|bench ...".into()),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
