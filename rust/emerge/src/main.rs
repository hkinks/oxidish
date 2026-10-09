//! EMERGE: an agar.io that grows an economy. Simulation and client in one process.
//! See docs/emerge.md.

mod ai;
mod econ;
mod render;
mod sim;

use macroquad::prelude::*;
use sim::*;

struct Args {
    sim: Option<u64>,
    bots: usize,
    seed: u64,
    stage: usize,
    warm: f32,
    screenshot: Option<String>,
    shot_at: f32,
    speed: f32,
    autoplay: bool,
    help: bool,
    dump: bool,
    zoom: f32,
}

fn parse_args() -> Args {
    let mut a = Args { sim: None, bots: 27, seed: 7, stage: 0, warm: 0.0, screenshot: None, shot_at: 3.0, speed: 1.0, autoplay: false, help: false, dump: false, zoom: 1.0 };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let num = |it: &mut std::iter::Skip<std::env::Args>| it.next().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
        match k.as_str() {
            "--sim" => a.sim = Some(num(&mut it) as u64),
            "--bots" => a.bots = num(&mut it) as usize,
            "--seed" => a.seed = num(&mut it) as u64,
            "--stage" => a.stage = num(&mut it) as usize,
            "--warm" => a.warm = num(&mut it) as f32,
            "--screenshot" => a.screenshot = it.next(),
            "--shot-at" => a.shot_at = num(&mut it) as f32,
            "--speed" => a.speed = num(&mut it) as f32,
            "--zoom" => a.zoom = num(&mut it) as f32,
            "--autoplay" => a.autoplay = true,
            "--dump" => a.dump = true,
            "--help-open" => a.help = true,
            "-h" | "--help" => {
                println!("emerge [--sim TICKS] [--bots N] [--seed S] [--stage N --warm SECS] [--screenshot out.png --shot-at SECS] [--speed X] [--autoplay]");
                std::process::exit(0);
            }
            o => eprintln!("ignoring {o}"),
        }
    }
    a
}

fn roles_str(w: &World) -> String {
    let c = w.role_counts();
    format!("F{} H{} G{} T{}", c[0], c[1], c[2], c[3])
}

fn report(w: &World) {
    let tot = w.total_mass();
    println!(
        "t={:>5.0}s stage={} food={:>4} alive={:>2} gini={:.2} top10%={:>2.0}% patches={:>4.0} nests={} colonies={} roles[{}] gathered={:.0} deliv={:.0} trades={} money={:.0} chitin={} mass_err={:+.2}",
        w.t,
        w.stage,
        w.food.len(),
        w.agents.iter().filter(|a| a.alive && a.kind != Kind::Drifter).count(),
        w.gini(),
        w.top_share(0.1) * 100.0,
        w.veins.iter().map(|v| v.stock).sum::<f32>(),
        w.stations.iter().filter(|s| s.alive).count(),
        w.clans.iter().filter(|c| c.alive).count(),
        roles_str(w),
        w.stats.ore_mined,
        w.stats.ore_delivered,
        w.stats.trades,
        w.money_supply(),
        w.avg_plate_ask().map_or("-".to_string(), |p| format!("{:.0}", p)),
        tot - w.ledger_expected()
    );
}

/// Advance the world; with `target_stage`, force a stage up every 25s so structures actually form.
fn run_warm(w: &mut World, secs: f32, target_stage: usize) {
    if target_stage > 0 {
        w.forced_stage = true;
    }
    let n = (secs / DT) as u64;
    let mut next_force = 25.0;
    for _ in 0..n {
        if target_stage > 0 && w.stage < target_stage && w.t >= next_force {
            w.advance_stage();
            next_force += 25.0;
        }
        w.step();
    }
}

fn run_sim(args: &Args, ticks: u64) {
    let mut w = World::new(args.seed, false, args.bots);
    let t0 = std::time::Instant::now();
    let mut last_stage = 0;
    println!("EMERGE headless sim: seed {} bots {} ticks {} ({:.0}s of game time)", args.seed, args.bots, ticks, ticks as f32 * DT);
    if args.stage > 0 {
        w.forced_stage = true;
    }
    let mut next_force = 25.0;
    for k in 0..ticks {
        if args.stage > 0 && w.stage < args.stage && w.t >= next_force {
            w.advance_stage();
            next_force += 25.0;
        }
        w.step();
        if w.stage != last_stage {
            last_stage = w.stage;
            println!("  >>> STAGE {} {} at t={:.0}s", w.stage, STAGE_NAMES[w.stage], w.t);
            report(&w);
        }
        if k % 1800 == 1799 {
            report(&w);
        }
    }
    if args.dump {
        dump(&w);
    }
    println!("--- final ---");
    report(&w);
    let s = &w.stats;
    println!(
        "eats={} agent_kills={} bonds_formed={} betrayals={} takeovers={} nests_built={} chitin_made={} harvest_faucet={:.0} grant_faucet={:.0} fee_sink={:.0} loot_sink={:.0} other_sink={:.0}",
        s.eats, s.kills_agents, s.pacts_formed, s.betrayals, s.takeovers, s.stations_built, s.plates_made, s.money_faucet_bounty, s.money_faucet_grant, s.money_sink_fee, s.money_sink_loot, s.money_sink_other
    );
    println!("stage times: {:?}", w.stage_times.iter().map(|t| t.round()).collect::<Vec<_>>());
    println!("mass: total {:.2} expected {:.2} (detritus {:.0})", w.total_mass(), w.ledger_expected(), w.heat_total());
    if let Some(p) = w.price_hist.front().zip(w.price_hist.back()) {
        println!("chitin price: first {:.0} last {:.0}", p.0, p.1);
    }
    let mut top: Vec<&Agent> = w.agents.iter().filter(|a| a.kind != Kind::Drifter).collect();
    top.sort_by(|a, b| w.net_worth(b).partial_cmp(&w.net_worth(a)).unwrap());
    for a in top.iter().take(5) {
        println!(
            "  {:<8} nw={:>5.0} mass={:>4.0} role={:<8} skills F{:.1} H{:.1} M{:.1} T{:.1} spores={:.0} kills={} deaths={}",
            a.name, w.net_worth(a), a.mass, ROLE_NAMES[a.role as usize], a.skill[0], a.skill[1], a.skill[2], a.skill[3], a.credits, a.kills, a.deaths
        );
    }
    println!("wall time {:.1}s ({:.0} ticks/s)", t0.elapsed().as_secs_f32(), ticks as f32 / t0.elapsed().as_secs_f32());
}

fn dump(w: &World) {
    for a in w.agents.iter().filter(|a| a.kind != Kind::Drifter) {
        let nv = w.veins.iter().map(|v| (v.pos.distance(a.pos) as i32, v.stock as i32)).min().unwrap_or((0, 0));
        println!(
            "  {:<8} {:<7} m={:>5.0} nutrients={:>5.0}/{:>4.0} want={:.2} goalD={:>5.0} nearestVein={:?} clan={:?} refuge={} gathering={}",
            a.name, ROLE_NAMES[a.role as usize], a.mass, a.ore, a.cargo_cap(), a.want.length(), a.pos.distance(a.goal), nv, a.clan, w.in_haven(a.pos), a.mining
        );
    }
}

fn window_conf() -> Conf {
    Conf { window_title: "EMERGE".to_owned(), window_width: 1280, window_height: 720, window_resizable: true, sample_count: 2, ..Default::default() }
}

fn main() {
    let args = parse_args();
    if let Some(t) = args.sim {
        run_sim(&args, t);
        return;
    }
    macroquad::Window::from_config(window_conf(), game(args));
}

async fn game(args: Args) {
    let mut w = World::new(args.seed, true, args.bots);
    w.autoplay = args.autoplay;
    if args.stage > 0 || args.warm > 0.0 {
        run_warm(&mut w, args.warm.max(if args.stage > 0 { args.stage as f32 * 25.0 + 40.0 } else { 0.0 }), args.stage);
    }
    let start_t = w.t;
    let mut ui = render::Ui { show_help: args.help || args.screenshot.is_none(), show_econ: true };
    let mut zoom = args.zoom;
    let mut acc = 0.0f32;
    let mut speed = args.speed;
    let mut help_t = 0.0f32;
    loop {
        let dt = get_frame_time().min(0.1);
        help_t += dt;
        // ---- input
        if is_key_pressed(KeyCode::Escape) {
            break;
        }
        if is_key_pressed(KeyCode::H) {
            ui.show_help = !ui.show_help;
        }
        if is_key_pressed(KeyCode::M) {
            ui.show_econ = !ui.show_econ;
        }
        if is_key_pressed(KeyCode::RightBracket) {
            w.advance_stage();
        }
        if is_key_pressed(KeyCode::Equal) {
            speed = (speed * 2.0).min(16.0);
        }
        if is_key_pressed(KeyCode::Minus) {
            speed = (speed / 2.0).max(1.0);
        }
        let (_, wy) = mouse_wheel();
        if wy != 0.0 {
            zoom = (zoom * (1.0 + wy.signum() * 0.1)).clamp(0.35, 3.0);
        }
        if !w.autoplay && w.agents[0].alive {
            let m = Vec2::from(mouse_position());
            let c = vec2(screen_width() / 2.0, screen_height() / 2.0);
            let scale = screen_height() * 0.28;
            let mut d = (m - c) / scale;
            if d.length() > 1.0 {
                d = d.normalize();
            }
            w.agents[0].want = d;
            let keys = [
                (KeyCode::Space, Act::Dash),
                (KeyCode::F, Act::Pact),
                (KeyCode::G, Act::Leave),
                (KeyCode::B, Act::Build),
                (KeyCode::R, Act::Burn),
                (KeyCode::Key1, Act::SellOre),
                (KeyCode::Key2, Act::BuyPlate),
                (KeyCode::Key3, Act::Equip),
                (KeyCode::Key4, Act::SellPlate),
            ];
            for (k, act) in keys {
                if is_key_pressed(k) {
                    w.agents[0].acts.push(act);
                }
            }
            // holding 1 keeps selling (convenient for big loads)
            if is_key_down(KeyCode::Key1) && w.tick % 6 == 0 {
                w.agents[0].acts.push(Act::SellOre);
            }
        }
        // ---- simulate
        acc += dt * speed;
        let mut n = 0;
        while acc >= DT && n < 40 {
            w.step();
            acc -= DT;
            n += 1;
        }
        if n == 40 {
            acc = 0.0;
        }
        // ---- draw
        let view = render::make_view(&w, zoom, 0);
        render::draw_world(&w, &view);
        render::draw_hud(&w, &ui, 0, &view, get_fps(), speed, w.autoplay);
        if let Some(path) = &args.screenshot {
            if w.t >= start_t + args.shot_at || help_t > 60.0 {
                get_screen_data().export_png(path);
                println!("screenshot -> {path} (stage {}, t={:.0}s)", w.stage, w.t);
                break;
            }
        }
        next_frame().await;
    }
}
