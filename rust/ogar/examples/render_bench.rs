//! Renderer micro-benchmark: `cargo run --release --example render_bench` (from the repo root, needs bot/c3_4.json).
use ogar::agent::{snapshot, AgentDriver};
use ogar::render::{render_into, Scratch};
use arena_client::game::Snapshot;
use std::time::Instant;

fn main() {
    let spec: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("bot/c3_4.json").unwrap()).unwrap();
    let mut w = ogar::world::World::new(ogar::config::Config::new(7000.0, 8, 2.0), 3);
    let mut agents: Vec<AgentDriver> = (0..6).map(|i| AgentDriver::new(&mut w, &format!("a{i}"), &spec, 2)).collect();
    for _ in 0..3000 {
        w.tick();
        if w.tick % 2 == 0 {
            agents.iter_mut().for_each(|a| a.step(&mut w));
            w.kill_log.clear();
        }
    }
    let snaps: Vec<_> = agents.iter().map(|a| (snapshot(&w, a.pi), w.view_box(a.pi))).filter(|(s, _)| !s.mine.is_empty()).collect();
    let mut sc = Scratch::default();
    let kinds = |s: &Snapshot| {
        use arena_client::game::Kind;
        let c = |k: Kind| s.others.iter().filter(|o| o.kind == k).count();
        (c(Kind::Food), c(Kind::Player), c(Kind::Virus), s.mine.len())
    };
    println!("food/player/virus/mine per snapshot: {:?}", snaps.iter().map(|(s, _)| kinds(s)).collect::<Vec<_>>());
    let empty: Vec<_> = snaps.iter().map(|(s, b)| { let mut s = s.clone(); s.others.clear(); s.mine.clear(); (s, *b) }).collect();
    let nofood: Vec<_> = snaps.iter().map(|(s, b)| { let mut s = s.clone(); s.others.retain(|c| c.kind != arena_client::game::Kind::Food); (s, *b) }).collect();
    for (name, set) in [("full", &snaps), ("no cells", &empty), ("no food", &nofood)] {
        let px = 256;
        let mut out = vec![0u8; px * px];
        let t = Instant::now();
        for _ in 0..300 { for (s, (x0, y0, x1, y1)) in set { render_into(s, (x0 + x1) / 2.0, (y0 + y1) / 2.0, (x1 - x0) / 2.0, px, &mut sc, &mut out); } }
        println!("{name}: {:.0} us", t.elapsed().as_secs_f64() * 1e6 / (300 * set.len()) as f64);
    }
    for (px, mult) in [(256usize, 1.0f64), (128, 2.0)] {
        let mut out = vec![0u8; px * px];
        let t = Instant::now();
        let n = 300;
        for _ in 0..n {
            for (s, (x0, y0, x1, y1)) in &snaps {
                let (cx, cy, half) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (x1 - x0) / 2.0);
                render_into(s, cx, cy, half * mult, px, &mut sc, &mut out);
            }
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / (n * snaps.len()) as f64;
        println!("px {px}: {us:.0} us/frame, avg cells in view {:.0}", snaps.iter().map(|(s, _)| s.others.len()).sum::<usize>() as f64 / snaps.len() as f64);
    }
}
