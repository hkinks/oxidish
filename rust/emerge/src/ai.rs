//! Rule-based agents. They follow the same rules the player does, including the later-stage
//! mechanics (mining, pacts, stations, trade). Specialization emerges from per-role skill
//! (learning by doing) feeding back into role choice.

use crate::sim::*;
use macroquad::prelude::{vec2, Vec2};

struct Snap {
    id: usize,
    pos: Vec2,
    mass: f32,
    eff: f32,
    kind: Kind,
    clan: Option<usize>,
    cargo: f32,
    haven: bool,
}

struct Ctx {
    stage: usize,
    t: f32,
    avg_mass: f32,
    food_frac: f32,
    nerf: bool,
    bounty: f32,
    avg_bid: f32,
    min_ask: f32,
    max_ask: f32,
    n_stations: usize,
    counts: [usize; 4],
}

pub fn think(w: &mut World, dt: f32) {
    let rl = w.rules();
    let snaps: Vec<Snap> = w
        .agents
        .iter()
        .filter(|a| a.alive)
        .map(|a| Snap {
            id: a.id,
            pos: a.pos,
            mass: a.mass,
            eff: a.eff_mass(),
            kind: a.kind,
            clan: a.clan,
            cargo: a.ore + 3.0 * a.plates as f32 + a.credits * 0.05,
            haven: false,
        })
        .map(|mut s| {
            s.haven = w.in_haven(s.pos);
            s
        })
        .collect();
    let live: Vec<&Station> = w.stations.iter().filter(|s| s.alive).collect();
    let n_st = live.len();
    let ctx = Ctx {
        stage: w.stage,
        t: w.t,
        avg_mass: {
            let v: Vec<f32> = snaps.iter().filter(|s| s.kind != Kind::Drifter).map(|s| s.mass).collect();
            (v.iter().sum::<f32>() / v.len().max(1) as f32).max(1.0)
        },
        food_frac: w.food.len() as f32 / FOOD_CAP as f32,
        nerf: w.t < w.nerf_until,
        bounty: w.bounty_mult,
        avg_bid: if n_st > 0 { live.iter().map(|s| s.ore_bid).sum::<f32>() / n_st as f32 } else { 10.0 },
        min_ask: live.iter().map(|s| s.plate_ask).fold(1e9, f32::min),
        max_ask: live.iter().map(|s| s.plate_ask).fold(0.0, f32::max),
        n_stations: n_st,
        counts: w.role_counts(),
    };
    let World { agents, food, veins, stations, clans, rng, autoplay, .. } = w;
    for a in agents.iter_mut() {
        if !a.alive || (a.kind == Kind::Player && !*autoplay) {
            continue;
        }
        if a.kind == Kind::Drifter {
            drifter(a, &snaps, rng, dt);
        } else {
            bot(a, &ctx, &rl, &snaps, food, veins, stations, clans, rng, dt);
        }
    }
}

fn dir_to(from: Vec2, to: Vec2) -> Vec2 {
    (to - from).normalize_or_zero()
}

fn edge_push(p: Vec2) -> Vec2 {
    let m = 150.0;
    let mut v = Vec2::ZERO;
    if p.x < m {
        v.x += 1.0;
    }
    if p.x > MAP - m {
        v.x -= 1.0;
    }
    if p.y < m {
        v.y += 1.0;
    }
    if p.y > MAP - m {
        v.y -= 1.0;
    }
    v
}

fn wander(a: &mut Agent, rng: &mut Rng, dt: f32) -> Vec2 {
    let ang = rng.range(-1.0, 1.0) * dt * 2.5;
    a.wander = Vec2::from_angle(ang).rotate(a.wander);
    let ep = edge_push(a.pos);
    if ep != Vec2::ZERO {
        a.wander = (a.wander + ep * 2.0).normalize_or_zero();
    }
    a.wander * 0.7
}

fn drifter(a: &mut Agent, snaps: &[Snap], rng: &mut Rng, dt: f32) {
    let r = a.radius();
    let mut best: Option<(f32, Vec2)> = None;
    for s in snaps {
        if s.id == a.id || s.kind == Kind::Drifter || s.haven {
            continue;
        }
        if a.mass > s.eff * 1.3 {
            let d = a.pos.distance(s.pos);
            if d < 650.0 && best.map_or(true, |(bd, _)| d < bd) {
                best = Some((d, s.pos));
            }
        }
    }
    a.want = match best {
        Some((d, p)) if d > r * 0.3 => dir_to(a.pos, p),
        _ => wander(a, rng, dt),
    };
}

fn choose_role(a: &mut Agent, c: &Ctx, veins: &[Vein], clans: &[Clan]) {
    let mut sc = [-9.0f32; 4];
    sc[0] = 1.0 + a.bias[0] + 0.15 * a.skill[0] + 0.6 * (c.food_frac - 0.4).min(0.0).abs() * -1.0 + 0.3 * c.food_frac;
    let rel = (a.mass / c.avg_mass - 1.0).clamp(-1.0, 1.5);
    sc[1] = 0.65 + a.bias[1] + 0.3 * rel + 0.15 * a.skill[1] + 0.5 * (1.0 - (c.food_frac * 2.0).min(1.0)) + if c.stage >= 5 { 0.12 * c.bounty } else { 0.0 };
    if c.stage >= 2 && veins.iter().any(|v| v.stock > 20.0) {
        sc[2] = 0.8 + a.bias[2] + 0.15 * a.skill[2];
        if c.stage >= 4 && c.n_stations > 0 {
            sc[2] += 0.25;
        }
        if c.stage >= 4 {
            if let Some(cl) = a.clan {
                if clans[cl].station.is_none() {
                    sc[2] += 0.5;
                }
            }
        }
        if c.stage >= 5 {
            sc[2] += 0.25 * (c.avg_bid / 10.0 - 1.0) * if c.nerf { 0.5 } else { 1.0 };
            if c.nerf {
                sc[2] -= 0.3;
            }
        }
    }
    if c.stage >= 5 && c.n_stations >= 2 && c.max_ask > c.min_ask * 1.5 && (a.credits > c.min_ask * 1.2 || a.plates > 0) {
        sc[3] = 0.4 + a.bias[3] + 0.15 * a.skill[3] + 1.2 * (c.max_ask / c.min_ask - 1.5).min(1.5);
    }
    for k in 0..4 {
        if a.role as usize == k {
            sc[k] += 0.05; // inertia
        }
        sc[k] -= 0.05 * c.counts[k] as f32; // crowding: niches saturate
    }
    let mut best = 0;
    for k in 1..4 {
        if sc[k] > sc[best] {
            best = k;
        }
    }
    a.role = match best {
        0 => Role::Forager,
        1 => Role::Hunter,
        2 => Role::Miner,
        _ => Role::Trader,
    };
}

#[allow(clippy::too_many_arguments)]
fn bot(a: &mut Agent, c: &Ctx, rl: &Rules, snaps: &[Snap], food: &[Food], veins: &[Vein], stations: &[Station], clans: &[Clan], rng: &mut Rng, dt: f32) {
    a.act_cd -= dt;
    a.dock_t -= dt;
    a.goal_t -= dt;
    a.role_t -= dt;
    a.social_t -= dt;
    let r = a.radius();
    let eff = a.eff_mass();
    let ri = a.role as usize;
    a.skill[ri] = (a.skill[ri] + 0.012 * dt).min(3.0);
    if a.role_t <= 0.0 {
        a.role_t = rng.range(8.0, 16.0);
        choose_role(a, c, veins, clans);
        if rl.pacts && a.clan.map_or(true, |cl| clans[cl].members.len() < 3) && rng.chance(0.55) {
            a.social_t = rng.range(8.0, 16.0);
        }
    }
    let my_haven = rl.stations && stations.iter().any(|s| s.alive && s.pos.distance(a.pos) < HAVEN_R);

    // --- perception: threats and prey
    let mut flee = Vec2::ZERO;
    let mut closest_threat = 1e9f32;
    let mut prey: Option<(f32, usize, Vec2)> = None;
    let mut mate: Option<(f32, Vec2)> = None;
    for s in snaps {
        if s.id == a.id {
            continue;
        }
        let d = a.pos.distance(s.pos);
        let same = a.clan.is_some() && a.clan == s.clan;
        if same {
            continue;
        }
        let sr = radius(s.mass);
        if !my_haven && s.mass > 1.2 * eff && d < sr + 280.0 {
            let wgt = 1.0 - d / (sr + 280.0);
            flee += dir_to(s.pos, a.pos) * wgt.max(0.05);
            closest_threat = closest_threat.min(d - sr);
        } else if a.mass > 1.3 * s.eff && d < 700.0 && !s.haven && s.kind != Kind::Drifter || (s.kind == Kind::Drifter && a.mass > 1.3 * s.eff && d < 500.0 && !s.haven) {
            let score = (s.mass + s.cargo * 1.5 + if s.kind == Kind::Drifter && c.stage >= 5 { 40.0 * c.bounty } else { 0.0 }) / (d + 80.0);
            if prey.map_or(true, |(b, _, _)| score > b) {
                prey = Some((score, s.id, s.pos));
            }
        }
        if rl.pacts && s.kind != Kind::Drifter && s.kind != Kind::Player && s.clan.is_none() && d < 1200.0 && s.mass > 0.4 * a.mass && s.mass < 2.5 * a.mass && mate.map_or(true, |(bd, _)| d < bd) {
            mate = Some((d, s.pos));
        }
    }

    // --- betrayal: a hunter in a pact may defect when a rich, smaller pact-mate is close
    if a.role == Role::Hunter && a.clan.is_some() && rng.chance(0.03 * dt) {
        let tempt = snaps.iter().any(|s| s.id != a.id && s.clan == a.clan && a.mass > 1.4 * s.eff && s.cargo > 10.0 && a.pos.distance(s.pos) < 450.0);
        if tempt {
            a.acts.push(Act::Leave);
        }
    }

    // --- dock services
    if rl.stations {
        if a.armor < 6 && a.plates > 0 && a.role != Role::Trader {
            a.acts.push(Act::Equip);
        }
        if my_haven && a.dock_t <= 0.0 {
            a.dock_t = 0.8;
            if a.ore >= 1.0 && a.role != Role::Trader {
                a.acts.push(Act::SellOre);
            }
            if rl.market {
                let ask = stations.iter().filter(|s| s.alive && s.pos.distance(a.pos) < HAVEN_R + 20.0).map(|s| s.plate_ask).next().unwrap_or(1e9);
                if a.role != Role::Trader && a.armor + a.plates < 5 && a.credits > ask * 1.15 {
                    a.acts.push(Act::BuyPlate);
                }
                if a.role != Role::Trader && a.armor >= 6 && a.plates > 1 {
                    a.acts.push(Act::SellPlate);
                }
            }
        }
    }

    // --- flee dominates
    if flee != Vec2::ZERO && !my_haven {
        a.want = flee.normalize_or_zero();
        if rl.dash && closest_threat < 40.0 && a.dash_cd <= 0.0 && a.mass > 40.0 {
            a.acts.push(Act::Dash);
        }
        return;
    }

    // --- build a station
    if rl.stations && a.act_cd <= 0.0 {
        if let Some(cl) = a.clan {
            if clans[cl].station.is_none() && a.ore >= 80.0 && a.mass >= 80.0 {
                a.acts.push(Act::Build);
                a.act_cd = 1.5;
            }
        }
    }

    // --- look for a pact partner
    if rl.pacts && a.social_t > 0.0 {
        if let Some((d, p)) = mate {
            if d < 260.0 {
                if a.act_cd <= 0.0 {
                    a.act_cd = 1.0;
                    a.acts.push(Act::Pact);
                }
                a.want = dir_to(a.pos, p) * 0.3;
                return;
            } else {
                a.want = dir_to(a.pos, p);
                return;
            }
        }
    }

    // --- shopping trip: spend credits on armor
    if rl.market && a.role != Role::Trader && a.armor + a.plates < 4 && a.credits > c.min_ask * 1.4 && c.n_stations > 0 && !my_haven {
        if let Some(st) = stations.iter().filter(|s| s.alive && s.plates >= 1.0).min_by(|x, y| x.pos.distance(a.pos).partial_cmp(&y.pos.distance(a.pos)).unwrap()) {
            a.want = dir_to(a.pos, st.pos);
            return;
        }
    }

    // --- role behaviour
    match a.role {
        Role::Miner if rl.veins => {
            let cap = a.cargo_cap();
            let any_vein = veins.iter().any(|v| v.stock > 15.0);
            let full = a.ore >= 0.75 * cap || (a.ore > 5.0 && !any_vein);
            if full {
                if rl.stations && !stations.iter().all(|s| !s.alive) {
                    let best = stations
                        .iter()
                        .filter(|s| s.alive)
                        .max_by(|x, y| {
                            let vx = if rl.market { x.ore_bid } else { 10.0 } / (1.0 + a.pos.distance(x.pos) / 1200.0);
                            let vy = if rl.market { y.ore_bid } else { 10.0 } / (1.0 + a.pos.distance(y.pos) / 1200.0);
                            vx.partial_cmp(&vy).unwrap()
                        })
                        .unwrap();
                    let d = a.pos.distance(best.pos);
                    if d < HAVEN_R - 30.0 {
                        // broke station and market: fall back to burning ore for mass
                        let broke = rl.market && clans[best.clan].treasury < best.ore_bid * 4.0;
                        let barter_dry = !rl.market && best.plates < 1.0;
                        if (broke || barter_dry) && a.act_cd <= 0.0 {
                            a.acts.push(Act::Burn);
                            a.act_cd = 1.0;
                        }
                        a.want = Vec2::ZERO;
                    } else {
                        a.want = dir_to(a.pos, best.pos);
                    }
                } else if a.act_cd <= 0.0 {
                    a.acts.push(Act::Burn);
                    a.act_cd = 1.0;
                }
                return;
            }
            // go to / work a vein
            if a.goal_t <= 0.0 {
                a.goal_t = rng.range(0.6, 1.2);
                if let Some(v) = veins.iter().filter(|v| v.stock > 15.0).min_by(|x, y| (a.pos.distance(x.pos) / (x.stock + 40.0)).partial_cmp(&(a.pos.distance(y.pos) / (y.stock + 40.0))).unwrap()) {
                    a.goal = v.pos;
                }
            }
            let d = a.pos.distance(a.goal);
            if d < VEIN_R * 0.5 {
                a.want = dir_to(a.pos, a.goal) * 0.15;
            } else {
                a.want = dir_to(a.pos, a.goal);
            }
            return;
        }
        Role::Trader if rl.market && c.n_stations >= 2 => {
            let sell_to = stations.iter().filter(|s| s.alive).max_by(|x, y| x.plate_ask.partial_cmp(&y.plate_ask).unwrap());
            let buy_from = stations.iter().filter(|s| s.alive && s.plates >= 1.0).min_by(|x, y| x.plate_ask.partial_cmp(&y.plate_ask).unwrap());
            if a.plates > 0 {
                if let Some(s) = sell_to {
                    let d = a.pos.distance(s.pos);
                    if d < HAVEN_R - 30.0 {
                        if a.act_cd <= 0.0 {
                            a.act_cd = 0.7;
                            a.acts.push(Act::SellPlate);
                        }
                        a.want = Vec2::ZERO;
                    } else {
                        a.want = dir_to(a.pos, s.pos);
                    }
                    return;
                }
            } else if let Some(s) = buy_from {
                if a.credits > s.plate_ask * 1.15 && c.max_ask > s.plate_ask * 1.4 {
                    let d = a.pos.distance(s.pos);
                    if d < HAVEN_R - 30.0 {
                        if a.act_cd <= 0.0 {
                            a.act_cd = 0.7;
                            a.acts.push(Act::BuyPlate);
                        }
                        a.want = Vec2::ZERO;
                    } else {
                        a.want = dir_to(a.pos, s.pos);
                    }
                    return;
                }
            }
            // nothing to arbitrage: fall through to foraging
        }
        Role::Hunter => {
            if let Some((_, _, p)) = prey {
                let d = a.pos.distance(p);
                a.want = dir_to(a.pos, p);
                if rl.dash && d < r * 3.0 + 80.0 && a.dash_cd <= 0.0 && a.mass > 50.0 && rng.chance(0.05) {
                    a.acts.push(Act::Dash);
                }
                return;
            }
        }
        _ => {}
    }

    // opportunistic prey for everyone, only if very near
    if let Some((_, _, p)) = prey {
        if a.pos.distance(p) < r * 2.5 + 60.0 {
            a.want = dir_to(a.pos, p);
            return;
        }
    }
    // forage
    if a.goal_t <= 0.0 {
        a.goal_t = rng.range(0.3, 0.7);
        let mut best = (1e12f32, None);
        for f in food {
            let d = (f.pos - a.pos).length_squared();
            if d < best.0 {
                best = (d, Some(f.pos));
            }
        }
        if let Some(p) = best.1 {
            a.goal = p;
        } else {
            a.goal = a.pos + a.wander * 300.0;
        }
    }
    let d = a.pos.distance(a.goal);
    if d > 4.0 {
        a.want = dir_to(a.pos, a.goal);
    } else {
        a.want = wander(a, rng, dt);
    }
    let _ = vec2(0.0, 0.0);
}
