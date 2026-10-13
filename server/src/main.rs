//! ogar CLI.
//!
//!   ogar match --agents ramp300=bots/specs/r2g9_ramp300.json:1 champ=bots/specs/final_champion.json:1 --ticks 22500 \
//!              --food-mult 2 --seed 1 --matches 8 --threads 2 --out rows.json
//!   ogar endless --agents c3_4=bots/specs/c3_4.json:2 ... --worlds 8 --game-min 240 --seed-base 700 [--threads T] --out samples.json
//!                (the fast endless evaluator: all worlds in one process, vision agents batched across worlds; endless.rs)
//!   ogar drill --agents c3_4=bots/specs/c3_4.json oracle=oracle --scenario giant_pop,giant_popped --our-mass 500 --giant-mass 5000 --dist 600 //!              --reps 50 --seed 1 [--giant-mode hunt|passive] [--competitor bots/specs/c3_4.json --comp-mass 800] [--horizon 30] --out drill.json
//!                (the giant-pop harvest drill: how well does an agent eat a giant that just burst on a virus; drill.rs)
//!   ogar vdump --agents vis=bots/specs/vision_bc3.json:2 ... --count 1000 --out DIR   (vision parity frames, training/rl/parity.py)
//!   ogar serve --port 9600 --bots 8 --food-mult 2 --map 7000 [--agents name=genome.json:N] [--full-view]
//!                [--record DIR [--record-all]]   (eat events, agent loss scenes, human frames: see record.rs)

use ogar::matches::{parse_lineup, run_many, MatchOpts};
use serde_json::Value;

/// Escape non-ASCII like Python's json.dumps.
fn ascii_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch.is_ascii() {
            out.push(ch);
        } else {
            let mut buf = [0u16; 2];
            for u in ch.encode_utf16(&mut buf) {
                out.push_str(&format!("\\u{:04x}", u));
            }
        }
    }
    out
}

pub struct Args {
    pub agents: Vec<String>,
    pub kv: std::collections::HashMap<String, String>,
}

pub fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut a = Args { agents: vec![], kv: Default::default() };
    let mut i = 0;
    while i < args.len() {
        let key = args[i].clone();
        if !key.starts_with("--") {
            return Err(format!("unexpected argument {key}"));
        }
        if key == "--agents" {
            while i + 1 < args.len() && !args[i + 1].starts_with("--") {
                i += 1;
                a.agents.push(args[i].clone());
            }
        } else if matches!(key.as_str(), "--realtime" | "--full-view" | "--record-all") {
            a.kv.insert(key[2..].to_string(), "1".into());
        } else {
            i += 1;
            let v = args.get(i).ok_or(format!("{key} needs a value"))?;
            a.kv.insert(key[2..].to_string(), v.clone());
        }
        i += 1;
    }
    Ok(a)
}

impl Args {
    pub fn num<T: std::str::FromStr>(&self, k: &str, default: T) -> Result<T, String> {
        match self.kv.get(k) {
            Some(v) => v.parse().map_err(|_| format!("bad --{k}")),
            None => Ok(default),
        }
    }
}

fn run_match_cmd(a: &Args) -> Result<(), String> {
    let lineup = parse_lineup(&a.agents)?;
    if lineup.is_empty() {
        return Err("need --agents".into());
    }
    let o = MatchOpts {
        ticks: a.num("ticks", 3000u64)?,
        sync: a.num("sync", 2u64)?,
        bots: a.num("bots", 8usize)?,
        map: a.num("map", 7000.0f64)?,
        food_mult: a.num("food-mult", 1.0f64)?,
    };
    let seed: u64 = a.num("seed", 1u64)?;
    let matches: usize = a.num("matches", 1usize)?;
    let threads: usize = a.num("threads", 1usize)?;
    let t0 = std::time::Instant::now();
    let rows = run_many(&lineup, &o, seed, matches, threads);
    let wall = t0.elapsed().as_secs_f64();
    // per-policy summary (mean over agents of each policy and matches)
    let mut names: Vec<String> = Vec::new();
    for r in &rows {
        let p = r["policy"].as_str().unwrap().to_string();
        if !names.contains(&p) {
            names.push(p);
        }
    }
    println!("\n{:<14}{:>10}{:>10}{:>9}{:>9}{:>9}{:>8}{:>8}", "policy", "log_mass", "avg_mass", "peak", "deaths", "kills", "vshots", "vpops");
    for n in &names {
        let g: Vec<&Value> = rows.iter().filter(|r| r["policy"] == *n).collect();
        let m = |k: &str| g.iter().map(|r| r[k].as_f64().unwrap()).sum::<f64>() / g.len() as f64;
        println!("{:<14}{:>10.3}{:>10.1}{:>9.0}{:>9.2}{:>9.2}{:>8.2}{:>8.2}", n, m("log_mass"), m("avg_mass"), m("peak_mass"), m("deaths"), m("kills"),
                 m("virus_shots"), m("virus_pops"));
    }
    let game_min = matches as f64 * o.ticks as f64 * 0.04 / 60.0;
    println!("\n{matches} matches ({game_min:.0} game-min) in {wall:.1}s wall on {threads} thread(s) = {:.0}x realtime aggregate, {:.2} matches/min/thread",
             game_min * 60.0 / wall, matches as f64 / wall * 60.0 / threads as f64);
    #[cfg(feature = "vision-net")]
    {
        let (calls, frames, ns) = ogar::nn::stats();
        if calls > 0 {
            println!("vision net: {calls} calls, {frames} frames, {:.2} ms/call, {:.2} ms/frame (inside the net call, all threads)",
                     ns as f64 / calls as f64 / 1e6, ns as f64 / frames as f64 / 1e6);
        }
    }
    if let Some(path) = a.kv.get("out") {
        std::fs::write(path, ascii_json(&serde_json::to_string(&rows).unwrap())).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn run_endless_cmd(a: &Args) -> Result<(), String> {
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let worlds: usize = a.num("worlds", 8usize)?;
    let o = ogar::endless::EndlessOpts {
        agents: a.agents.clone(),
        worlds,
        game_min: a.num("game-min", 240.0f64)?,
        seed_base: a.num("seed-base", 700u64)?,
        threads: a.num("threads", worlds.min(cpus))?,
        bots: a.num("bots", 8usize)?,
        map: a.num("map", 7000.0f64)?,
        food_mult: a.num("food-mult", 2.0f64)?,
        sync: a.num("sync", 2u64)?,
        log_s: a.num("log-s", 15.0f64)?,
    };
    let (samples, st) = ogar::endless::run(&o)?;
    let n: usize = samples.iter().map(|s| s.len()).sum();
    println!("{} worlds x {:.0} game-min, {n} samples, {:.0}s wall", o.worlds, o.game_min, st.wall_s);
    if st.infer_calls > 0 {
        println!("vision: {} batched calls, {:.1} frames/call, {:.2} ms/call ({:.0}s of the wall inside inference)", st.infer_calls,
                 st.infer_frames as f64 / st.infer_calls as f64, st.infer_s * 1e3 / st.infer_calls as f64, st.infer_s);
    }
    #[cfg(feature = "vision-net")]
    {
        let (c, r, d) = ogar::nn::phases();
        if st.infer_calls > 0 {
            let k = st.infer_calls as f64 * 1e6;
            println!("per call: copy in {:.2} ms, graph run {:.2} ms, copy out {:.2} ms", c as f64 / k, r as f64 / k, d as f64 / k);
        }
    }
    if let Some(path) = a.kv.get("out") {
        let stats = serde_json::json!({"wall_s": st.wall_s, "infer_calls": st.infer_calls, "infer_frames": st.infer_frames, "infer_s": st.infer_s, "agents": st.agents});
        let doc = serde_json::json!({"worlds": samples, "stats": stats});
        std::fs::write(path, ascii_json(&serde_json::to_string(&doc).unwrap())).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn run_serve_cmd(a: &Args) -> Result<(), String> {
    // a live server runs one world, so the vision nets may use a few inference threads (headless runs default to 1:
    // they run many worlds in parallel)
    if std::env::var_os("OGAR_VISION_THREADS").is_none() {
        let n = std::thread::available_parallelism().map(|n| (n.get() / 4).clamp(1, 4)).unwrap_or(1);
        std::env::set_var("OGAR_VISION_THREADS", n.to_string());
    }
    let o = ogar::server::ServeOpts {
        full_view: a.kv.contains_key("full-view"),
        port: a.num("port", 9600u16)?,
        bind: a.kv.get("bind").cloned().unwrap_or_else(|| "0.0.0.0".into()),
        bots: a.num("bots", 8usize)?,
        map: a.num("map", 7000.0f64)?,
        food_mult: a.num("food-mult", 1.0f64)?,
        seed: a.num("seed", 0u64)?,
        sync: a.num("sync", 2u64)?,
        max_tps: a.num("max-tps", 25.0f64)?,
        agents: a.agents.clone(),
        follow: a.kv.get("follow").cloned().unwrap_or_default(),
        log_players: a.num("log-players", 0.0f64)?,
        control_port: a.kv.get("control-port").map(|v| v.parse().map_err(|_| "bad --control-port")).transpose()?,
        control_token: a.kv.get("control-token").cloned(),
        control_bind: a.kv.get("control-bind").cloned().unwrap_or_else(|| "127.0.0.1".into()),
        record: a.kv.get("record").cloned(),
        record_all: a.kv.contains_key("record-all"),
    };
    ogar::server::serve(o)
}

/// `ogar render --agents ... --ticks T --seed S --out DIR [--px 256]`: play T ticks, then write what each agent's
/// vision would be (its camera view at px, and a 2x wider view at px/2) as grayscale PNGs.
fn run_render_cmd(a: &Args) -> Result<(), String> {
    use ogar::agent::{snapshot, step_all, AgentDriver};
    use ogar::render::{png_gray, render};
    let lineup = parse_lineup(&a.agents)?;
    let ticks: u64 = a.num("ticks", 3000u64)?;
    let px: usize = a.num("px", 256usize)?;
    let out = a.kv.get("out").cloned().unwrap_or_else(|| "logs/vision".into());
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let cfg = ogar::config::Config::new(a.num("map", 7000.0f64)?, a.num("bots", 8usize)?, a.num("food-mult", 1.0f64)?);
    let mut w = ogar::world::World::new(cfg, a.num("seed", 1u64)?);
    let mut agents: Vec<AgentDriver> = lineup.iter().map(|(n, s)| AgentDriver::new(&mut w, n, s, 2)).collect();
    for _ in 0..ticks {
        w.tick();
        if w.tick % 2 == 0 {
            step_all(&mut agents, &mut w);
            w.kill_log.clear();
        }
    }
    for ag in &agents {
        let s = snapshot(&w, ag.pi);
        if s.mine.is_empty() {
            continue;
        }
        let (vx0, vy0, vx1, vy1) = w.view_box(ag.pi);
        let (cx, cy, half) = ((vx0 + vx1) / 2.0, (vy0 + vy1) / 2.0, (vx1 - vx0) / 2.0);
        let mass: f64 = s.mine.iter().map(|c| c.mass()).sum();
        for (tag, h, p) in [("view", half, px), ("wide", half * 2.0, px / 2)] {
            let path = format!("{out}/{}_{tag}_{p}.png", ag.name);
            std::fs::write(&path, png_gray(p, p, &render(&s, cx, cy, h, p))).map_err(|e| e.to_string())?;
        }
        println!("{} mass {:.0} cells {} view width {:.0}", ag.name, mass, s.mine.len(), 2.0 * half);
    }
    Ok(())
}

/// `ogar vdump --agents vis=bots/specs/vision_bc3.json:2 champ=...:2 --ticks T --seed S --every K --count N --out DIR`: play,
/// and every K decisions record each vision agent's input frames and the Rust net output (parity check against torch,
/// see training/rl/parity.py). Files: view.u8 [N,2,256,256], wide.u8 [N,2,128,128], rust.f32 [N,4] = (u, v, split, eject).
#[cfg(feature = "vision-net")]
fn run_vdump_cmd(a: &Args) -> Result<(), String> {
    use ogar::agent::{step_all, AgentDriver};
    let lineup = parse_lineup(&a.agents)?;
    let ticks: u64 = a.num("ticks", 20000u64)?;
    let every: u64 = a.num("every", 10u64)?;
    let count: usize = a.num("count", 1000usize)?;
    let out = a.kv.get("out").cloned().unwrap_or_else(|| "logs/vision/parity".into());
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let cfg = ogar::config::Config::new(a.num("map", 7000.0f64)?, a.num("bots", 8usize)?, a.num("food-mult", 2.0f64)?);
    let mut w = ogar::world::World::new(cfg, a.num("seed", 1u64)?);
    let mut agents: Vec<AgentDriver> = lineup.iter().map(|(n, s)| AgentDriver::new(&mut w, n, s, 2)).collect();
    let (mut view, mut wide, mut rust): (Vec<u8>, Vec<u8>, Vec<f32>) = (vec![], vec![], vec![]);
    let (mut n, mut step) = (0usize, 0u64);
    for _ in 0..ticks {
        w.tick();
        if w.tick % 2 != 0 {
            continue;
        }
        step_all(&mut agents, &mut w);
        w.kill_log.clear();
        step += 1;
        if step % every != 0 {
            continue;
        }
        for ag in &agents {
            if let (Some(v), true) = (ag.vision.as_ref(), n < count) {
                if let Some(o) = v.last {
                    view.extend_from_slice(&v.frames.view);
                    wide.extend_from_slice(&v.frames.wide);
                    rust.extend_from_slice(&[o.target.0, o.target.1, o.split_logit, o.eject_logit]);
                    n += 1;
                }
            }
        }
        if n >= count {
            break;
        }
    }
    std::fs::write(format!("{out}/view.u8"), &view).map_err(|e| e.to_string())?;
    std::fs::write(format!("{out}/wide.u8"), &wide).map_err(|e| e.to_string())?;
    let bytes: Vec<u8> = rust.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(format!("{out}/rust.f32"), bytes).map_err(|e| e.to_string())?;
    println!("recorded {n} frames to {out}");
    Ok(())
}

/// "500,1000" -> [500.0, 1000.0]
fn num_list(a: &Args, k: &str, default: &str) -> Result<Vec<f64>, String> {
    a.kv.get(k).map(String::as_str).unwrap_or(default).split(',').map(|v| v.trim().parse::<f64>().map_err(|_| format!("bad --{k}"))).collect()
}

fn run_drill_cmd(a: &Args) -> Result<(), String> {
    use ogar::drill::{print_table, run_all, summarize_all, AgentDef, Cond, DrillOpts, GiantMode, Scenario};
    let agents: Vec<AgentDef> = a.agents.iter().map(|s| AgentDef::parse(s)).collect::<Result<_, _>>()?;
    if agents.is_empty() {
        return Err("need --agents name=spec.json|oracle|oracle_view ...".into());
    }
    let scenarios: Vec<Scenario> = a.kv.get("scenario").map(String::as_str).unwrap_or("giant_pop").split(',').map(Scenario::parse).collect::<Result<_, _>>()?;
    let competitor = match a.kv.get("competitor") {
        Some(p) => Some(serde_json::from_str::<Value>(&std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?).map_err(|e| format!("{p}: {e}"))?),
        None => None,
    };
    let modes: Vec<GiantMode> = a.kv.get("giant-mode").map(String::as_str).unwrap_or("hunt").split(',').map(GiantMode::parse).collect::<Result<_, _>>()?;
    let giant_bot = match a.kv.get("giant-bot") {
        Some(p) => Some(serde_json::from_str::<Value>(&std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?).map_err(|e| format!("{p}: {e}"))?),
        None if !modes.contains(&GiantMode::Bot) => None,
        None => Some(serde_json::from_str::<Value>(&std::fs::read_to_string("bots/specs/c3_4.json").map_err(|e| format!("bots/specs/c3_4.json: {e} (run from the repo root or pass --giant-bot)"))?).map_err(|e| e.to_string())?),
    };
    let mut conds = Vec::new();
    for (sc, mode) in scenarios.iter().flat_map(|s| modes.iter().map(move |m| (s, m))) {
        for om in num_list(a, "our-mass", "500")? {
            for gm in num_list(a, "giant-mass", "5000")? {
                for dist in num_list(a, "dist", "600")? {
                    let mut c = Cond::new(*sc, om, gm, dist);
                    c.giant_mode = *mode;
                    c.competitor = competitor.clone();
                    c.giant_bot = giant_bot.clone();
                    c.comp_mass = a.num("comp-mass", 800.0f64)?;
                    c.horizon_s = a.num("horizon", 30.0f64)?;
                    c.map = a.num("map", 7000.0f64)?;
                    c.food_mult = a.num("food-mult", 2.0f64)?;
                    conds.push(c);
                }
            }
        }
    }
    let o = DrillOpts { agents, conds, reps: a.num("reps", 50usize)?, seed: a.num("seed", 1u64)?, threads: a.num("threads", 4usize)? };
    let t0 = std::time::Instant::now();
    let rows = run_all(&o);
    let sums = summarize_all(&rows);
    print_table(&sums);
    println!("
{} reps in {:.0}s wall on {} thread(s)", rows.len(), t0.elapsed().as_secs_f64(), o.threads);
    if let Some(path) = a.kv.get("out") {
        let doc = serde_json::json!({"summary": sums, "reps": rows});
        std::fs::write(path, ascii_json(&serde_json::to_string(&doc).unwrap())).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// `ogar escape`: the escape drill (escape.rs, docs/escape-drill.md).
fn run_escape_cmd(a: &Args) -> Result<(), String> {
    use ogar::drill::AgentDef;
    use ogar::escape::{print_table, run_all, summarize_all, EscCond, EscOpts, Place};
    let agents: Vec<AgentDef> = a.agents.iter().map(|s| AgentDef::parse(s)).collect::<Result<_, _>>()?;
    if agents.is_empty() {
        return Err("need --agents name=spec.json|oracle ...".into());
    }
    let places: Vec<Place> = a.kv.get("place").map(String::as_str).unwrap_or("open,wall,corner").split(',').map(Place::parse).collect::<Result<_, _>>()?;
    let virus: Vec<bool> = a.kv.get("virus").map(String::as_str).unwrap_or("0").split(',').map(|v| v == "1").collect();
    let pieces: Vec<f64> = num_list(a, "pieces", "1")?;
    let mut conds = Vec::new();
    for om in num_list(a, "our-mass", "3000")? {
        for ratio in num_list(a, "ratio", "2")? {
            for gap in num_list(a, "gap", "1200")? {
                for &pl in &places {
                    for &vr in &virus {
                        for &pc in &pieces {
                            let mut c = EscCond::new(om, ratio, gap, pl);
                            c.virus = vr;
                            c.pieces = pc as usize;
                            c.horizon_s = a.num("horizon", 60.0f64)?;
                            c.map = a.num("map", 7000.0f64)?;
                            c.food_mult = a.num("food-mult", 2.0f64)?;
                            conds.push(c);
                        }
                    }
                }
            }
        }
    }
    let o = EscOpts { agents, conds, reps: a.num("reps", 50usize)?, seed: a.num("seed", 1u64)?, threads: a.num("threads", 3usize)? };
    let t0 = std::time::Instant::now();
    let rows = run_all(&o);
    let sums = summarize_all(&rows);
    print_table(&sums);
    println!("
{} reps in {:.0}s wall on {} thread(s)", rows.len(), t0.elapsed().as_secs_f64(), o.threads);
    if let Some(path) = a.kv.get("out") {
        let doc = serde_json::json!({"summary": sums, "reps": rows});
        std::fs::write(path, ascii_json(&serde_json::to_string(&doc).unwrap())).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("match") => parse_args(&args[1..]).and_then(|a| run_match_cmd(&a)),
        Some("endless") => parse_args(&args[1..]).and_then(|a| run_endless_cmd(&a)),
        Some("drill") => parse_args(&args[1..]).and_then(|a| run_drill_cmd(&a)),
        Some("escape") => parse_args(&args[1..]).and_then(|a| run_escape_cmd(&a)),
        Some("serve") => parse_args(&args[1..]).and_then(|a| run_serve_cmd(&a)),
        Some("render") => parse_args(&args[1..]).and_then(|a| run_render_cmd(&a)),
        #[cfg(feature = "vision-net")]
        Some("vdump") => parse_args(&args[1..]).and_then(|a| run_vdump_cmd(&a)),
        _ => Err("usage: ogar match|endless|drill|serve|render ...".into()),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
