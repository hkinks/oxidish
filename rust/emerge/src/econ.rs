//! Pacts (clans), stations, refinery, barter and the credit market.
//! Money is tracked with explicit faucets (bounties, grants) and sinks (fees, destroyed loot).

use crate::sim::*;
use macroquad::prelude::Vec2;

const BUILD_ORE: f32 = 80.0;
const BUILD_MASS: f32 = 40.0;
pub const BARTER_RATIO: f32 = 5.0; // ore per plate in stage 4

pub fn fee_rate(w: &World, who: usize, st: usize) -> f32 {
    let a = &w.agents[who];
    let member = a.clan == Some(w.stations[st].clan);
    let base = if member { 0.02 } else { 0.06 };
    let sk = if a.role == Role::Trader { a.skill[Role::Trader as usize].min(3.0) } else { 0.0 };
    base * (1.0 - 0.2 * sk)
}

pub fn propose_pact(w: &mut World, i: usize) {
    let me = &w.agents[i];
    if me.kind == Kind::Drifter {
        return;
    }
    let (mp, mm, mc, mk) = (me.pos, me.mass, me.clan, me.kind);
    let mut best: Option<(usize, f32)> = None;
    for (j, o) in w.agents.iter().enumerate() {
        if j == i || !o.alive || o.kind == Kind::Drifter || (mk == Kind::Bot && o.kind == Kind::Player) {
            continue;
        }
        if mc.is_some() && o.clan == mc {
            continue;
        }
        let d = o.pos.distance(mp);
        if d < 320.0 && best.map_or(true, |(_, bd)| d < bd) {
            best = Some((j, d));
        }
    }
    let Some((j, _)) = best else {
        if mk == Kind::Player {
            w.say("No one in range for a colony (get within ~300).");
        }
        return;
    };
    let ratio = w.agents[j].mass / mm.max(1.0);
    let ok_ratio = (0.3..=3.5).contains(&ratio);
    let accept = if mk == Kind::Player {
        ok_ratio && (w.agents[j].role != Role::Hunter || w.rng.chance(0.5))
    } else {
        ok_ratio && w.rng.chance(0.6)
    };
    if !accept {
        if mk == Kind::Player {
            let n = w.agents[j].name.clone();
            w.say(format!("{} declined your colony.", n));
        }
        return;
    }
    let (jc, ic) = (w.agents[j].clan, w.agents[i].clan);
    let target_clan = match (ic, jc) {
        (Some(_), Some(_)) => return,
        (Some(c), None) => {
            if w.clans[c].members.len() >= 6 {
                return;
            }
            w.agents[j].clan = Some(c);
            w.clans[c].members.push(j);
            c
        }
        (None, Some(c)) => {
            if w.clans[c].members.len() >= 6 {
                return;
            }
            w.agents[i].clan = Some(c);
            w.clans[c].members.push(i);
            c
        }
        (None, None) => {
            let (name, hue) = w.clan_name_for();
            w.clans.push(Clan { name, hue, members: vec![i, j], treasury: 0.0, station: None, alive: true });
            let c = w.clans.len() - 1;
            w.agents[i].clan = Some(c);
            w.agents[j].clan = Some(c);
            c
        }
    };
    w.stats.pacts_formed += 1;
    let (a, b, c) = (w.agents[i].name.clone(), w.agents[j].name.clone(), w.clans[target_clan].name.clone());
    w.say(format!("{} and {} are bound by colony: {}", a, b, c));
}

pub fn leave_clan(w: &mut World, i: usize, announce: bool) {
    let Some(c) = w.agents[i].clan else { return };
    w.agents[i].clan = None;
    w.clans[c].members.retain(|m| *m != i);
    if announce {
        let (n, cn) = (w.agents[i].name.clone(), w.clans[c].name.clone());
        w.say(format!("{} left {}", n, cn));
        w.stats.betrayals += 1;
    }
    if w.clans[c].members.is_empty() {
        w.clans[c].alive = false;
        // dissolved clan: its station is abandoned (decays in slow())
        w.stats.money_sink_other += w.clans[c].treasury;
        w.clans[c].treasury = 0.0;
    }
}

pub fn build_station(w: &mut World, i: usize) {
    let a = &w.agents[i];
    let Some(c) = a.clan else {
        if a.kind == Kind::Player {
            w.say("You need a colony to found a nest (F near another cell).");
        }
        return;
    };
    let is_player = a.kind == Kind::Player;
    if w.clans[c].station.is_some() {
        if is_player {
            w.say("Your colony already has a nest.");
        }
        return;
    }
    if a.ore < BUILD_ORE || a.mass < BUILD_MASS + 30.0 {
        if is_player {
            w.say(format!("Nest needs {:.0} nutrients in cargo and {:.0}+ mass.", BUILD_ORE, BUILD_MASS + 30.0));
        }
        return;
    }
    let p = a.pos;
    if p.x < 200.0 || p.y < 200.0 || p.x > MAP - 200.0 || p.y > MAP - 200.0 || w.stations.iter().any(|s| s.alive && s.pos.distance(p) < 700.0) {
        if is_player {
            w.say("Too close to the edge or to another nest (need 700 apart).");
        }
        return;
    }
    let a = &mut w.agents[i];
    a.ore -= BUILD_ORE;
    a.mass -= BUILD_MASS;
    let st = Station { pos: p, clan: c, ore: BUILD_ORE, plates: 0.0, struct_mass: BUILD_MASS, ore_bid: 10.0, plate_ask: 45.0, trades: 0, alive: true, refine_acc: 0.0 };
    w.stations.push(st);
    let k = w.stations.len() - 1;
    w.clans[c].station = Some(k);
    w.stats.stations_built += 1;
    if w.first_station_t.is_none() {
        w.first_station_t = Some(w.t);
    }
    if w.stage >= 5 {
        w.clans[c].treasury += 150.0;
        w.stats.money_faucet_grant += 150.0;
    }
    let (n, cn) = (w.agents[i].name.clone(), w.clans[c].name.clone());
    w.say(format!("{} founded a nest for {}. Haven: nobody can be eaten inside.", n, cn));
}

/// Dock services. Stage 4: fixed barter. Stage 5: priced market with fees.
pub fn trade(w: &mut World, i: usize, act: Act) {
    let pos = w.agents[i].pos;
    let Some(s) = w.station_at(pos) else { return };
    let market = w.stage >= 5;
    let f = fee_rate(w, i, s);
    let clan = w.stations[s].clan;
    match act {
        Act::SellOre => {
            let ore = w.agents[i].ore;
            if ore < 1.0 {
                return;
            }
            if !market {
                // barter: 5 ore -> 1 plate out of station stock
                let units = ((ore / BARTER_RATIO).floor()).min(w.stations[s].plates.floor());
                if units < 1.0 {
                    return;
                }
                let give = units * BARTER_RATIO;
                w.agents[i].ore -= give;
                w.agents[i].plates += units as u32;
                w.stations[s].ore += give;
                w.stations[s].plates -= units;
                w.stations[s].trades += 1;
                w.stats.trades += 1;
                w.stats.ore_delivered += give;
            } else {
                let bid = w.stations[s].ore_bid;
                let afford = (w.clans[clan].treasury / (bid * (1.0 - f)).max(0.01)).floor();
                let qty = ore.floor().min(afford);
                if qty < 1.0 {
                    return;
                }
                let v = qty * bid;
                w.clans[clan].treasury -= v * (1.0 - f);
                let burn = v * 0.01;
                let got = v * (1.0 - f - 0.01);
                w.agents[i].credits += got;
                w.agents[i].earned += got;
                w.agents[i].ore -= qty;
                w.stations[s].ore += qty;
                w.stations[s].trades += 1;
                w.stats.trades += 1;
                w.stats.ore_delivered += qty;
                w.stats.money_sink_fee += burn;
            }
        }
        Act::BuyPlate => {
            if !market {
                return;
            }
            let p = w.stations[s].plate_ask;
            let cost = p * (1.0 + f + 0.01);
            if w.agents[i].credits < cost || w.stations[s].plates < 1.0 {
                return;
            }
            w.agents[i].credits -= cost;
            w.agents[i].plates += 1;
            w.stations[s].plates -= 1.0;
            w.clans[clan].treasury += p * (1.0 + f);
            w.stats.money_sink_fee += p * 0.01;
            w.stations[s].trades += 1;
            w.stats.trades += 1;
        }
        Act::SellPlate => {
            if !market || w.agents[i].plates == 0 {
                return;
            }
            let bid = w.stations[s].plate_ask * 0.8;
            let pay = bid * (1.0 - f);
            if w.clans[clan].treasury < pay {
                return;
            }
            w.clans[clan].treasury -= pay;
            let got = bid * (1.0 - f - 0.01);
            w.stats.money_sink_fee += bid * 0.01;
            w.agents[i].credits += got;
            w.agents[i].earned += got;
            w.agents[i].plates -= 1;
            w.stations[s].plates += 1.0;
            w.stations[s].trades += 1;
            w.stats.trades += 1;
        }
        _ => {}
    }
}

/// Once per second.
pub fn slow(w: &mut World) {
    let stage = w.stage;
    // station upkeep: refinery, prices, abandonment
    for s in 0..w.stations.len() {
        if !w.stations[s].alive {
            continue;
        }
        let clan_alive = w.clans[w.stations[s].clan].alive;
        if !clan_alive {
            // abandoned: decays into heat
            let st = &mut w.stations[s];
            st.struct_mass -= 1.0;
            let p = st.pos;
            if st.struct_mass <= 0.0 {
                st.alive = false;
                let back = st.ore + 3.0 * st.plates + 1.0;
                st.ore = 0.0;
                st.plates = 0.0;
                st.struct_mass = 0.0;
                w.add_heat(p, back);
                w.say("An abandoned nest crumbled into detritus.");
            } else {
                w.add_heat(p, 1.0);
            }
            continue;
        }
        // refinery: 4 ore -> 1 plate (3 mass) + 1 heat; up to 2 per second
        for _ in 0..2 {
            if w.stations[s].ore >= 4.0 && w.stations[s].plates < 40.0 {
                let st = &mut w.stations[s];
                st.ore -= 4.0;
                st.plates += 1.0;
                let p = st.pos;
                w.stats.plates_made += 1;
                w.add_heat(p, 1.0);
            }
        }
        let st = &mut w.stations[s];
        st.ore_bid = (10.0 * (50.0 / (st.ore + 10.0)).powf(0.8)).clamp(2.5, 40.0);
        st.plate_ask = (45.0 * (10.0 / (st.plates + 2.0)).powf(0.9)).clamp(14.0, 260.0);
    }
    // dividends: half the treasury is paid to members every 30 s (welfare keeps the labour force)
    if stage >= 5 && w.tick % 900 == 0 {
        for c in 0..w.clans.len() {
            if !w.clans[c].alive || w.clans[c].treasury < 40.0 {
                continue;
            }
            let alive: Vec<usize> = w.clans[c].members.iter().cloned().filter(|m| w.agents[*m].alive).collect();
            if alive.is_empty() {
                continue;
            }
            let pool = w.clans[c].treasury * 0.25;
            w.clans[c].treasury -= pool;
            for m in &alive {
                w.agents[*m].credits += pool / alive.len() as f32;
            }
        }
    }
    // dev patch: the EVE lesson. Nerf mining, boost bounties (money printing) -> stagflation.
    // takeovers: a dominant colony absorbs a much smaller one (members, treasury and nest)
    if stage >= 5 && w.tick % 1200 == 0 {
        absorb_smallest(w);
    }
    if stage >= 5 && !w.patch_done && w.stage_t > 110.0 {
        w.patch_done = true;
        w.nerf_until = w.t + 150.0;
        w.bounty_mult = 2.5;
        w.say("DROUGHT: patches yield -50% for 150s, parasite blooms make harvests x2.5. Watch prices.");
    }
    if stage >= 5 && w.patch_done && w.bounty_mult > 1.0 && w.t > w.nerf_until {
        w.bounty_mult = 1.0;
        w.say("The rains return: patches recover, harvests normal. The extra spores remain.");
    }
}

fn colony_mass(w: &World, c: usize) -> f32 {
    w.clans[c].members.iter().filter(|m| w.agents[**m].alive).map(|m| w.agents[*m].mass).sum()
}

fn absorb_smallest(w: &mut World) {
    let live: Vec<usize> = (0..w.clans.len()).filter(|c| w.clans[*c].alive && w.clans[*c].station.is_some()).collect();
    if live.len() < 2 {
        return;
    }
    let big = *live.iter().max_by(|a, b| colony_mass(w, **a).partial_cmp(&colony_mass(w, **b)).unwrap()).unwrap();
    let small = *live.iter().filter(|c| **c != big).min_by(|a, b| colony_mass(w, **a).partial_cmp(&colony_mass(w, **b)).unwrap()).unwrap();
    if colony_mass(w, big) < 2.5 * colony_mass(w, small) {
        return;
    }
    let members = std::mem::take(&mut w.clans[small].members);
    for m in &members {
        w.agents[*m].clan = Some(big);
        w.clans[big].members.push(*m);
    }
    let tr = w.clans[small].treasury;
    w.clans[small].treasury = 0.0;
    w.clans[big].treasury += tr;
    if let Some(s) = w.clans[small].station.take() {
        w.stations[s].clan = big;
    }
    w.clans[small].alive = false;
    w.stats.takeovers += 1;
    let (b, s) = (w.clans[big].name.clone(), w.clans[small].name.clone());
    w.say(format!("TAKEOVER: colony {} absorbed {} ({} members, its nest and {:.0} spores)", b, s, members.len(), tr));
}

pub fn clan_members_alive(w: &World, c: usize) -> Vec<usize> {
    w.clans[c].members.iter().cloned().filter(|m| w.agents[*m].alive).collect()
}

pub fn nearest_station(w: &World, p: Vec2) -> Option<usize> {
    w.stations
        .iter()
        .enumerate()
        .filter(|(_, s)| s.alive)
        .min_by(|a, b| a.1.pos.distance(p).partial_cmp(&b.1.pos.distance(p)).unwrap())
        .map(|(i, _)| i)
}
