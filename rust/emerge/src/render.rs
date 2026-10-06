//! Drawing + HUD (macroquad). Reads the World; never mutates it.

use crate::sim::*;
use macroquad::prelude::*;

pub const STAGE_DESC: [&str; 6] = [
    "Distinction. Specks of food are bits of mass. Eat to grow. Anything 25% smaller than you is food; anything 25% bigger is death.",
    "Constraint. Eating is lossy: 20% of a meal rots into detritus, and detritus feeds new growth where fights happened. Big cells shrink. SPACE lunges (costs mass).",
    "Scarcity. Sunlight dims, so the world is finite. Nutrient patches appear: hold still inside one to gather (cursor near your cell). R digests nutrients into mass. Spiked parasites hunt.",
    "Symbiosis. Cells can bond: F near a cell offers a bond (no friendly eating, faster when close). G leaves. Defection is allowed.",
    "Colonies. A colony can build a nest (B, 80 nutrients): a refuge where nobody can be eaten, with a digester. At the nest press 1 to trade nutrients for chitin, 3 to harden your shell.",
    "Exchange. Spores are money. Nest prices follow local stock. Parasite harvests make spores; tolls and lost loot destroy them. Watch prices, inequality, the drought and takeovers.",
];

pub struct View {
    pub cam: Vec2,
    pub scale: f32,
    pub center: Vec2,
}

impl View {
    pub fn w2s(&self, p: Vec2) -> Vec2 {
        (p - self.cam) * self.scale + self.center
    }
}

pub fn col(h: f32, s: f32, v: f32, a: f32) -> Color {
    let (r, g, b) = hsv(h, s, v);
    Color::new(r, g, b, a)
}

pub fn make_view(w: &World, zoom: f32, focus: usize) -> View {
    let a = &w.agents[focus];
    let r = a.radius();
    let view_h = (620.0 + r * 5.5) / zoom;
    View { cam: a.pos, scale: screen_height() / view_h, center: vec2(screen_width() / 2.0, screen_height() / 2.0) }
}

fn text_shadow(t: &str, x: f32, y: f32, size: f32, c: Color) {
    draw_text(t, x + 1.0, y + 1.0, size, Color::new(0.0, 0.0, 0.0, 0.7));
    draw_text(t, x, y, size, c);
}

fn wrap(t: &str, max: usize) -> Vec<String> {
    let mut out = vec![];
    let mut cur = String::new();
    for word in t.split_whitespace() {
        if !cur.is_empty() && cur.len() + 1 + word.len() > max {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(word);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn panel(x: f32, y: f32, w: f32, h: f32) {
    draw_rectangle(x, y, w, h, Color::new(0.05, 0.07, 0.05, 0.88));
    draw_rectangle_lines(x, y, w, h, 1.0, Color::new(0.55, 0.7, 0.45, 0.35));
}

pub struct Ui {
    pub show_help: bool,
    pub show_econ: bool,
}

pub fn draw_world(w: &World, v: &View, player: usize) {
    clear_background(Color::new(0.04, 0.07, 0.06, 1.0));
    let sw = screen_width();
    let sh = screen_height();
    // substrate lattice with heat field
    let tl = v.cam - v.center / v.scale;
    let br = v.cam + (vec2(sw, sh) - v.center) / v.scale;
    let tx0 = ((tl.x / TILE).floor() as isize).clamp(0, NT as isize);
    let tx1 = ((br.x / TILE).ceil() as isize).clamp(0, NT as isize);
    let ty0 = ((tl.y / TILE).floor() as isize).clamp(0, NT as isize);
    let ty1 = ((br.y / TILE).ceil() as isize).clamp(0, NT as isize);
    for ty in ty0..ty1 {
        for tx in tx0..tx1 {
            let h = w.heat[ty as usize * NT + tx as usize];
            let p = v.w2s(vec2(tx as f32 * TILE, ty as f32 * TILE));
            if h > 1.0 {
                draw_rectangle(p.x, p.y, TILE * v.scale, TILE * v.scale, Color::new(0.55, 0.36, 0.14, (h / 90.0).min(0.38)));
            }
            draw_rectangle_lines(p.x, p.y, TILE * v.scale, TILE * v.scale, 1.0, Color::new(0.35, 0.6, 0.45, 0.10));
        }
    }
    // map edge
    let e0 = v.w2s(vec2(0.0, 0.0));
    draw_rectangle_lines(e0.x, e0.y, MAP * v.scale, MAP * v.scale, 4.0, Color::new(0.65, 0.5, 0.3, 0.7));

    // veins
    for vn in &w.veins {
        let p = v.w2s(vn.pos);
        let r = VEIN_R * v.scale;
        if p.x < -r || p.y < -r || p.x > sw + r || p.y > sh + r {
            continue;
        }
        let fill = (vn.stock / vn.cap).clamp(0.0, 1.0);
        draw_circle(p.x, p.y, r, Color::new(0.25, 0.5, 0.2, 0.25 + 0.2 * fill));
        draw_circle_lines(p.x, p.y, r, 2.0, Color::new(0.6, 0.85, 0.4, 0.8));
        for k in 0..((fill * 9.0) as i32 + 1) {
            let ang = k as f32 * 2.4 + vn.pos.x;
            let rr = r * 0.55 * ((k as f32 * 0.37 + 0.2) % 1.0);
            draw_circle(p.x + ang.cos() * rr, p.y + ang.sin() * rr, (3.0 + 5.0 * fill) * v.scale.max(0.5), Color::new(0.8, 0.9, 0.35, 0.9));
        }
        if v.scale > 0.4 {
            text_shadow(&format!("nutrients {:.0}", vn.stock), p.x - r * 0.4, p.y + r + 14.0, 16.0, Color::new(0.95, 0.75, 0.45, 0.9));
        }
    }

    // stations
    for (i, s) in w.stations.iter().enumerate() {
        if !s.alive {
            continue;
        }
        let p = v.w2s(s.pos);
        let r = HAVEN_R * v.scale;
        let cl = &w.clans[s.clan];
        draw_circle(p.x, p.y, r, col(cl.hue, 0.5, 0.9, 0.10));
        draw_circle_lines(p.x, p.y, r, 3.0, col(cl.hue, 0.6, 1.0, 0.8));
        let q = 26.0 * v.scale.max(0.5);
        draw_circle(p.x, p.y, q * 1.3, col(cl.hue, 0.45, 0.6, 0.95));
        draw_circle(p.x, p.y, q * 0.8, col(cl.hue, 0.5, 0.85, 1.0));
        draw_circle_lines(p.x, p.y, q * 1.3, 3.0, Color::new(0.95, 0.9, 0.7, 0.9));
        let mut lines = vec![format!("NEST {} [{}]", i + 1, cl.name)];
        if w.stage >= 5 {
            lines.push(format!("nutrient bid {:.1}  chitin ask {:.0}", s.ore_bid, s.plate_ask));
        }
        lines.push(format!("stock nutrients {:.0} chitin {:.0}", s.ore, s.plates));
        for (k, l) in lines.iter().enumerate() {
            let tw = measure_text(l, None, 16, 1.0).width;
            text_shadow(l, p.x - tw / 2.0, p.y - r - 8.0 - 16.0 * (lines.len() - 1 - k) as f32, 16.0, WHITE);
        }
    }

    // food
    for f in &w.food {
        let p = v.w2s(f.pos);
        if p.x < -6.0 || p.y < -6.0 || p.x > sw + 6.0 || p.y > sh + 6.0 {
            continue;
        }
        let h = ((f.pos.x * 0.013 + f.pos.y * 0.017).sin() * 0.5 + 0.5) * 0.35 + 0.25;
        draw_circle(p.x, p.y, (5.5 * v.scale).max(2.5), col(h, 0.55, 0.95, 0.9));
    }

    // pact links (coupling made visible)
    for (ci, c) in w.clans.iter().enumerate() {
        if !c.alive {
            continue;
        }
        let m: Vec<&Agent> = c.members.iter().map(|i| &w.agents[*i]).filter(|a| a.alive).collect();
        for i in 0..m.len() {
            for j in (i + 1)..m.len() {
                let d = m[i].pos.distance(m[j].pos);
                if d < 900.0 {
                    let (a, b) = (v.w2s(m[i].pos), v.w2s(m[j].pos));
                    draw_line(a.x, a.y, b.x, b.y, 2.0, col(c.hue, 0.6, 1.0, 0.55 * (1.0 - d / 900.0) + 0.1));
                }
            }
        }
        let _ = ci;
    }

    // agents
    let mut order: Vec<usize> = (0..w.agents.len()).filter(|i| w.agents[*i].alive).collect();
    order.sort_by(|a, b| w.agents[*a].mass.partial_cmp(&w.agents[*b].mass).unwrap());
    for i in order {
        let a = &w.agents[i];
        let p = v.w2s(a.pos);
        let r = a.radius() * v.scale;
        if p.x < -r - 50.0 || p.y < -r - 50.0 || p.x > sw + r + 50.0 || p.y > sh + r + 50.0 {
            continue;
        }
        if a.mining {
            if let Some(vn) = w.veins.iter().find(|vn| vn.pos.distance(a.pos) < VEIN_R) {
                let q = v.w2s(vn.pos);
                draw_line(p.x, p.y, q.x, q.y, 2.0, Color::new(0.7, 0.9, 0.4, 0.7));
            }
        }
        if a.kind == Kind::Drifter {
            draw_circle(p.x, p.y, r, Color::new(0.42, 0.18, 0.48, 0.92));
            draw_circle_lines(p.x, p.y, r, 3.0, Color::new(1.0, 0.45, 0.35, 1.0));
            let n = 12;
            for k in 0..n {
                let ang = k as f32 / n as f32 * 6.2832 + w.t * 0.5;
                let (c, s) = (ang.cos(), ang.sin());
                draw_line(p.x + c * r, p.y + s * r, p.x + c * (r + 7.0 * v.scale.max(0.6)), p.y + s * (r + 7.0 * v.scale.max(0.6)), 2.0, Color::new(1.0, 0.4, 0.3, 1.0));
            }
        } else {
            if let Some(c) = a.clan {
                draw_circle(p.x, p.y, r + 5.0 * v.scale.max(0.7), col(w.clans[c].hue, 0.6, 1.0, 0.55));
            }
            draw_circle(p.x, p.y, r, col(a.hue, 0.55, 0.9, 1.0));
            let ol = if a.kind == Kind::Player { WHITE } else { col(a.hue, 0.7, 0.55, 1.0) };
            draw_circle_lines(p.x, p.y, r, 2.0 + a.armor as f32 * 1.2, if a.armor > 0 { Color::new(0.88, 0.74, 0.45, 1.0) } else { ol });
            if a.kind == Kind::Player {
                draw_circle_lines(p.x, p.y, r + 3.0, 2.0, WHITE);
            }
        }
        if r > 9.0 {
            let label = if a.kind == Kind::Drifter { "parasite".to_string() } else { a.name.clone() };
            let fs = (r * 0.55).clamp(13.0, 30.0);
            let tw = measure_text(&label, None, fs as u16, 1.0).width;
            text_shadow(&label, p.x - tw / 2.0, p.y + fs * 0.2, fs, WHITE);
            if r > 22.0 {
                let sub = format!("{:.0}", a.mass);
                let tw2 = measure_text(&sub, None, (fs * 0.8) as u16, 1.0).width;
                text_shadow(&sub, p.x - tw2 / 2.0, p.y + fs * 1.1, fs * 0.8, Color::new(1.0, 1.0, 1.0, 0.8));
            }
            if a.kind == Kind::Bot && w.stage >= 2 && r > 16.0 {
                let rl = ROLE_NAMES[a.role as usize];
                let tw3 = measure_text(rl, None, 14, 1.0).width;
                text_shadow(rl, p.x - tw3 / 2.0, p.y - r - 5.0, 14.0, Color::new(1.0, 1.0, 1.0, 0.65));
            }
        }
    }
}

fn bar(x: f32, y: f32, w: f32, h: f32, f: f32, c: Color) {
    draw_rectangle(x, y, w, h, Color::new(1.0, 1.0, 1.0, 0.12));
    draw_rectangle(x, y, w * f.clamp(0.0, 1.0), h, c);
}

fn spark(x: f32, y: f32, w: f32, h: f32, d: &std::collections::VecDeque<f32>, c: Color) {
    draw_rectangle(x, y, w, h, Color::new(1.0, 1.0, 1.0, 0.06));
    if d.len() < 2 {
        return;
    }
    let (mn, mx) = d.iter().fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
    let rng = (mx - mn).max(1e-3);
    let n = d.len();
    for k in 1..n {
        let x0 = x + (k - 1) as f32 / (n - 1) as f32 * w;
        let x1 = x + k as f32 / (n - 1) as f32 * w;
        let y0 = y + h - (d[k - 1] - mn) / rng * (h - 2.0) - 1.0;
        let y1 = y + h - (d[k] - mn) / rng * (h - 2.0) - 1.0;
        draw_line(x0, y0, x1, y1, 1.5, c);
    }
}

pub fn draw_hud(w: &World, ui: &Ui, player: usize, view: &View, fps: i32, speed: f32, autoplay: bool) {
    let sw = screen_width();
    let sh = screen_height();
    let p = &w.agents[player];

    // ---- stage panel (top-left)
    let pw = 470.0;
    let desc = wrap(STAGE_DESC[w.stage], 62);
    let gate = w.next_gate();
    let ph = 62.0 + desc.len() as f32 * 17.0 + if gate.is_some() { 56.0 } else { 0.0 };
    panel(10.0, 10.0, pw, ph);
    // stage pips
    for s in 0..6 {
        let c = if s < w.stage { Color::new(0.4, 0.8, 0.5, 1.0) } else if s == w.stage { Color::new(1.0, 0.85, 0.3, 1.0) } else { Color::new(1.0, 1.0, 1.0, 0.2) };
        draw_rectangle(20.0 + s as f32 * 26.0, 20.0, 22.0, 6.0, c);
    }
    text_shadow(&format!("STAGE {}  {}", w.stage, STAGE_NAMES[w.stage]), 20.0, 48.0, 28.0, Color::new(1.0, 0.9, 0.5, 1.0));
    for (k, l) in desc.iter().enumerate() {
        text_shadow(l, 20.0, 70.0 + k as f32 * 17.0, 17.0, Color::new(0.88, 0.92, 1.0, 1.0));
    }
    if let Some((g, f)) = gate {
        let y = 70.0 + desc.len() as f32 * 17.0 + 6.0;
        bar(20.0, y, pw - 20.0, 7.0, f, Color::new(0.5, 0.8, 1.0, 0.9));
        let gl = wrap(&format!("Next stage when {}", g), 66);
        for (k, l) in gl.iter().take(2).enumerate() {
            text_shadow(l, 20.0, y + 22.0 + k as f32 * 15.0, 15.0, Color::new(0.7, 0.8, 0.95, 0.9));
        }
    }

    // ---- player status
    let sy = 10.0 + ph + 8.0;
    let mut lines: Vec<String> = vec![format!("mass {:.0}", p.mass)];
    if w.stage >= 1 {
        lines[0].push_str(&format!("   dash {}", if p.dash_cd <= 0.0 { "ready".to_string() } else { format!("{:.1}s", p.dash_cd) }));
    }
    if w.stage >= 2 {
        lines.push(format!("nutrients {:.0}/{:.0}   chitin {}   shell {}", p.ore, p.cargo_cap(), p.plates, p.armor));
    }
    if w.stage >= 3 {
        lines.push(match p.clan {
            Some(c) => format!("colony: {} ({} members)", w.clans[c].name, w.clans[c].members.len()),
            None => "colony: none".to_string(),
        });
    }
    if w.stage >= 5 {
        lines.push(format!("spores {:.0}", p.credits));
    }
    panel(10.0, sy, 300.0, 12.0 + lines.len() as f32 * 19.0);
    for (k, l) in lines.iter().enumerate() {
        text_shadow(l, 20.0, sy + 20.0 + k as f32 * 19.0, 19.0, WHITE);
    }
    // context hint
    let mut hint = String::new();
    if !p.alive {
        hint = "You were eaten. Respawning...".into();
    } else if let Some(s) = w.station_at(p.pos) {
        let _ = s;
        hint = if w.stage >= 5 { "AT NEST: 1 sell nutrients  2 buy chitin  3 harden shell  4 sell chitin".into() } else { "AT NEST: 1 trade nutrients for chitin (5:1)  3 harden shell".into() };
    } else if w.stage >= 2 && w.veins.iter().any(|v| v.pos.distance(p.pos) < VEIN_R && v.stock > 1.0) {
        hint = if p.want.length() <= 0.35 { "Gathering... (R digests nutrients for mass)".into() } else { "In a patch: hold the cursor near your cell to gather".into() };
    } else if w.stage >= 4 && p.clan.is_some() && w.clans[p.clan.unwrap()].station.is_none() {
        hint = "B: found a nest here (80 nutrients + 40 mass)".into();
    } else if w.stage >= 3 {
        if let Some(o) = w.agents.iter().filter(|o| o.alive && o.kind == Kind::Bot && o.id != player && (p.clan.is_none() || o.clan != p.clan) && o.pos.distance(p.pos) < 320.0).next() {
            hint = format!("F: bond with {}", o.name);
        }
    }
    if !hint.is_empty() {
        let y = sy + 12.0 + lines.len() as f32 * 19.0 + 8.0;
        let tw = measure_text(&hint, None, 18, 1.0).width;
        panel(10.0, y, tw + 20.0, 26.0);
        text_shadow(&hint, 20.0, y + 19.0, 18.0, Color::new(0.6, 1.0, 0.7, 1.0));
    }

    // ---- leaderboard (top-right)
    let mut ranked: Vec<&Agent> = w.agents.iter().filter(|a| a.alive && a.kind != Kind::Drifter).collect();
    ranked.sort_by(|a, b| w.net_worth(b).partial_cmp(&w.net_worth(a)).unwrap());
    let lw = 250.0;
    let lx = sw - lw - 10.0;
    let rc = w.role_counts();
    let lh = 28.0 + 8.0 * 17.0 + if w.stage >= 2 { 22.0 } else { 0.0 };
    panel(lx, 10.0, lw, lh);
    text_shadow("NET WORTH", lx + 10.0, 28.0, 17.0, Color::new(0.8, 0.9, 1.0, 1.0));
    for (k, a) in ranked.iter().take(8).enumerate() {
        let c = if a.kind == Kind::Player { Color::new(1.0, 1.0, 0.5, 1.0) } else { WHITE };
        let tag = if w.stage >= 2 { &ROLE_NAMES[a.role as usize][..1] } else { "" };
        let l = format!("{}. {:<8} {:>5.0} {}", k + 1, a.name, w.net_worth(a), if a.kind == Kind::Bot { tag } else { "" });
        text_shadow(&l, lx + 10.0, 48.0 + k as f32 * 17.0, 16.0, c);
        if let Some(cl) = a.clan {
            draw_circle(lx + lw - 16.0, 43.0 + k as f32 * 17.0, 5.0, col(w.clans[cl].hue, 0.6, 1.0, 1.0));
        }
    }
    if w.stage >= 2 {
        text_shadow(&format!("roles F{} H{} G{} T{}", rc[0], rc[1], rc[2], rc[3]), lx + 10.0, 10.0 + lh - 8.0, 15.0, Color::new(0.7, 0.85, 1.0, 0.9));
    }

    // ---- economy / substrate panel
    if ui.show_econ {
        let ey = 10.0 + lh + 8.0;
        let eh = if w.stage >= 5 { 250.0 } else { 150.0 };
        panel(lx, ey, lw, eh);
        let mut y = ey + 20.0;
        let line = |s: String, y: &mut f32| {
            text_shadow(&s, lx + 10.0, *y, 15.0, Color::new(0.85, 0.92, 1.0, 1.0));
            *y += 17.0;
        };
        text_shadow("SUBSTRATE LEDGER", lx + 10.0, y, 17.0, Color::new(0.8, 0.9, 1.0, 1.0));
        y += 20.0;
        let tot = w.total_mass();
        let exp = w.ledger_expected();
        line(format!("mass in world {:.0}  (sun in {:.0})", tot, w.stats.sun_in + w.stats.respawn_in), &mut y);
        line(format!("detritus {:.0}  food {} specks", w.heat_total(), w.food.len()), &mut y);
        line(format!("conservation error {:+.2}", tot - exp), &mut y);
        line(format!("Gini {:.2}   top10% own {:.0}%", w.gini(), w.top_share(0.1) * 100.0), &mut y);
        if w.stage >= 5 {
            y += 4.0;
            text_shadow("EXCHANGE", lx + 10.0, y, 17.0, Color::new(1.0, 0.9, 0.5, 1.0));
            y += 19.0;
            line(format!("money supply {:.0} spores", w.money_supply()), &mut y);
            if let Some(pa) = w.avg_plate_ask() {
                line(format!("chitin price {:.0} spores   trades {}", pa, w.stats.trades), &mut y);
            }
            line(format!("faucets: harvest {:.0} grant {:.0}", w.stats.money_faucet_bounty, w.stats.money_faucet_grant), &mut y);
            line(format!("sinks: fees {:.0} loot {:.0}", w.stats.money_sink_fee, w.stats.money_sink_loot + w.stats.money_sink_other), &mut y);
            spark(lx + 10.0, y, lw - 20.0, 30.0, &w.price_hist, Color::new(1.0, 0.8, 0.3, 1.0));
            text_shadow("chitin price", lx + 14.0, y + 12.0, 12.0, Color::new(1.0, 0.8, 0.3, 0.8));
            y += 36.0;
            spark(lx + 10.0, y, lw - 20.0, 30.0, &w.money_hist, Color::new(0.5, 1.0, 0.7, 1.0));
            text_shadow("money supply", lx + 14.0, y + 12.0, 12.0, Color::new(0.5, 1.0, 0.7, 0.8));
        }
    }

    // ---- event log (bottom-left)
    let n = 7.min(w.log.len());
    let lg: Vec<&(f32, String)> = w.log.iter().rev().take(n).collect();
    for (k, (t, s)) in lg.iter().enumerate() {
        let age = w.t - t;
        let a = (1.0 - (age - 14.0).max(0.0) / 6.0).clamp(0.0, 1.0);
        if a <= 0.0 {
            continue;
        }
        let y = sh - 16.0 - k as f32 * 18.0;
        let s2: String = s.chars().take(110).collect();
        text_shadow(&s2, 14.0, y, 17.0, Color::new(0.9, 0.95, 1.0, a));
    }

    // ---- minimap (bottom-right)
    let ms = 170.0;
    let mx = sw - ms - 12.0;
    let my = sh - ms - 28.0;
    panel(mx, my, ms, ms);
    let k = ms / MAP;
    for ty in 0..NT {
        for tx in 0..NT {
            let h = w.heat[ty * NT + tx];
            if h > 2.0 {
                draw_rectangle(mx + tx as f32 * TILE * k, my + ty as f32 * TILE * k, TILE * k, TILE * k, Color::new(0.55, 0.38, 0.15, (h / 100.0).min(0.5)));
            }
        }
    }
    for vn in &w.veins {
        draw_circle(mx + vn.pos.x * k, my + vn.pos.y * k, 3.0, Color::new(0.9, 0.65, 0.3, 1.0));
    }
    for s in w.stations.iter().filter(|s| s.alive) {
        draw_rectangle(mx + s.pos.x * k - 4.0, my + s.pos.y * k - 4.0, 8.0, 8.0, col(w.clans[s.clan].hue, 0.6, 1.0, 1.0));
    }
    for a in w.agents.iter().filter(|a| a.alive) {
        let c = match a.kind {
            Kind::Player => WHITE,
            Kind::Drifter => Color::new(1.0, 0.2, 0.2, 1.0),
            Kind::Bot => col(a.hue, 0.5, 0.9, 0.9),
        };
        draw_circle(mx + a.pos.x * k, my + a.pos.y * k, if a.kind == Kind::Player { 3.5 } else { 2.0 }, c);
    }
    let vt = view.cam - view.center / view.scale;
    draw_rectangle_lines(mx + vt.x * k, my + vt.y * k, sw / view.scale * k, sh / view.scale * k, 1.0, Color::new(1.0, 1.0, 1.0, 0.5));

    // ---- footer
    let foot = format!("t {:.0}s   fps {}   x{:.0}{}   H help  M ledger  wheel zoom{}", w.t, fps, speed, if autoplay { "   AUTOPLAY" } else { "" }, "");
    text_shadow(&foot, 14.0, sh - 16.0 - 7.0 * 18.0 - 6.0, 14.0, Color::new(0.7, 0.75, 0.85, 0.7));

    // ---- stage flash
    if w.new_stage_flash > 0.0 && w.stage > 0 {
        let a = (w.new_stage_flash / 2.0).min(1.0);
        let t = format!("STAGE {}: {}", w.stage, STAGE_NAMES[w.stage]);
        let tw = measure_text(&t, None, 64, 1.0).width;
        text_shadow(&t, sw / 2.0 - tw / 2.0, sh * 0.2, 64.0, Color::new(1.0, 0.9, 0.4, a));
        let t2 = "a new rule has emerged";
        let tw2 = measure_text(t2, None, 26, 1.0).width;
        text_shadow(t2, sw / 2.0 - tw2 / 2.0, sh * 0.2 + 32.0, 26.0, Color::new(0.9, 0.95, 1.0, a));
    }

    // ---- help overlay
    if ui.show_help {
        let hw = 640.0;
        let hh = 430.0;
        let x = sw / 2.0 - hw / 2.0;
        let y = sh / 2.0 - hh / 2.0;
        panel(x, y, hw, hh);
        let mut yy = y + 32.0;
        text_shadow("EMERGE: eat, grow, and watch an economy grow out of it", x + 20.0, yy, 22.0, Color::new(1.0, 0.9, 0.5, 1.0));
        yy += 30.0;
        let rows = [
            "Mouse: steer (far from your cell = fast, near = slow). Wheel: zoom.",
            "SPACE lunge (stage 1+)      R digest nutrients into mass (stage 2+)",
            "Hold still in a nutrient patch to gather (stage 2+)",
            "F bond with the nearest cell, G leave your colony (stage 3+)",
            "B build a nest with your colony (stage 4+)",
            "At a nest: 1 sell nutrients  2 buy chitin  3 harden shell  4 sell chitin",
            "H this help   M ledger panel   Esc quit   ] skip to next stage",
            "",
            "The game starts as plain agar.io. Each new rule appears by itself when the",
            "world reaches a state (measured in the top-left bar), not from a menu.",
            "Everything is conserved: detritus, food, nutrients and cells are one pool of mass.",
            "Spores (money) are made only by parasite harvests and first-bloom grants,",
            "and destroyed by tolls and lost loot. Spiked parasites hunt small cells.",
            "Tan rings = shell (hardened chitin). Glowing circles = colony refuges.",
            "Bots follow the same rules: they specialize (F/H/G/T in the list), form colonies,",
            "build nests, trade between them, betray, and big colonies absorb small ones.",
        ];
        for r in rows.iter() {
            text_shadow(r, x + 20.0, yy, 18.0, Color::new(0.9, 0.94, 1.0, 1.0));
            yy += 22.0;
        }
        text_shadow("Press H to close", x + 20.0, y + hh - 14.0, 16.0, Color::new(0.6, 0.9, 0.7, 1.0));
    }
}
