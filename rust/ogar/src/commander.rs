//! The commander layer (docs/commander.md): standing orders for in-process agents (`order` on the control port),
//! and the `scene` observation a commander (a human, Claude over MCP, Jev) decides from.
//!
//! An order is a verb (+ target) with an expiry in game time, plus an independent split go/no-go flag. Rule bots
//! apply it through `arena_client::order` (their reflexes still veto it); vision agents through `vision_override`.

use arena_client::game::{Cell, Kind, Snapshot};
use arena_client::order::{Order, Verb};
use arena_client::params::Params;
use arena_client::policy::Out;
use arena_client::steer::{classify, danger_radius, EAT_RATIO, SPLIT_KILL_RATIO};
use serde_json::{json, Value};

use crate::agent::{snapshot, AgentDriver};
use crate::render::{png_gray, render};
use crate::world::{PKind, World, T_PLAYER, T_VIRUS};

pub const DEFAULT_TTL_S: f64 = 10.0;
const TICKS_PER_S: f64 = 25.0;
const SPLIT_REACH: f64 = 650.0;

#[derive(Clone, Debug, PartialEq)]
pub enum Target {
    Name(String),
    Cell(u32),
}

impl Target {
    fn parse(words: &[String]) -> Result<Target, String> {
        if words.is_empty() {
            return Err("needs a target: a player name or a cell id (#123)".into());
        }
        let text = words.join(" ");
        if let Some(n) = text.strip_prefix('#').and_then(|n| n.parse::<u32>().ok()) {
            return Ok(Target::Cell(n));
        }
        Ok(Target::Name(text))
    }
    fn show(&self) -> String {
        match self {
            Target::Name(n) => n.clone(),
            Target::Cell(c) => format!("#{c}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum OVerb {
    Hunt(Target),
    Goto(f64, f64),
    Farm,
    Flee,
    Hide,
    VirusSplit(Target),
    Harvest(Target),
}

impl OVerb {
    pub fn name(&self) -> &'static str {
        match self {
            OVerb::Hunt(_) => "hunt",
            OVerb::Goto(..) => "goto",
            OVerb::Farm => "farm",
            OVerb::Flee => "flee",
            OVerb::Hide => "hide",
            OVerb::VirusSplit(_) => "virus-split",
            OVerb::Harvest(_) => "harvest",
        }
    }
    fn target(&self) -> Option<&Target> {
        match self {
            OVerb::Hunt(t) | OVerb::VirusSplit(t) | OVerb::Harvest(t) => Some(t),
            _ => None,
        }
    }
    pub fn show(&self) -> String {
        match (self, self.target()) {
            (OVerb::Goto(x, y), _) => format!("goto {x:.0} {y:.0}"),
            (_, Some(t)) => format!("{} {}", self.name(), t.show()),
            _ => self.name().to_string(),
        }
    }
}

/// What the `order` control command asks for.
#[derive(Clone, Debug, PartialEq)]
pub enum OrderCmd {
    Set { agent: String, verb: OVerb, ttl_s: f64 },
    Split { agent: String, on: bool, ttl_s: f64 },
    Free { agent: String },
}

/// `order <agent> hunt <name|#id> | goto <x> <y> | farm | flee | hide | virus-split <target> | harvest <target>
///  | split on|off | free   [ttl=<seconds>]`
pub fn parse_order(parts: &[String]) -> Result<OrderCmd, String> {
    const USAGE: &str = "usage: order <agent> hunt <player|#cell> | goto <x> <y> | farm | flee | hide | virus-split <player|#cell> | harvest <player|#cell> | split on|off | free [ttl=<s>]";
    let mut words: Vec<String> = parts.iter().skip(1).cloned().collect(); // drop "order"
    if words.len() < 2 {
        return Err(USAGE.into());
    }
    let agent = words.remove(0);
    let mut ttl_s = DEFAULT_TTL_S;
    if let Some(i) = words.iter().position(|w| w.starts_with("ttl=")) {
        let t = words.remove(i);
        ttl_s = t[4..].parse::<f64>().ok().filter(|v| *v > 0.0 && v.is_finite()).ok_or("bad ttl (seconds of game time)")?;
    }
    let verb = words.remove(0).to_ascii_lowercase();
    let rest = words;
    let verb = match verb.as_str() {
        "free" | "clear" => return Ok(OrderCmd::Free { agent }),
        "split" => {
            let on = match rest.first().map(String::as_str) {
                Some("on") | Some("go") | Some("1") => true,
                Some("off") | Some("nogo") | Some("0") => false,
                _ => return Err("usage: order <agent> split on|off [ttl=<s>]".into()),
            };
            return Ok(OrderCmd::Split { agent, on, ttl_s });
        }
        "hunt" => OVerb::Hunt(Target::parse(&rest).map_err(|e| format!("hunt {e}"))?),
        "virus-split" | "virussplit" | "vsplit" => OVerb::VirusSplit(Target::parse(&rest).map_err(|e| format!("virus-split {e}"))?),
        "harvest" => OVerb::Harvest(Target::parse(&rest).map_err(|e| format!("harvest {e}"))?),
        "goto" => {
            let n: Vec<f64> = rest.iter().filter_map(|v| v.parse().ok()).collect();
            if rest.len() != 2 || n.len() != 2 || n.iter().any(|v| !v.is_finite()) {
                return Err("usage: order <agent> goto <x> <y> [ttl=<s>]".into());
            }
            OVerb::Goto(n[0], n[1])
        }
        "farm" => OVerb::Farm,
        "flee" => OVerb::Flee,
        "hide" => OVerb::Hide,
        _ => return Err(USAGE.into()),
    };
    if verb.target().is_none() && !matches!(verb, OVerb::Goto(..)) && !rest.is_empty() {
        return Err(format!("{} takes no argument", verb.name()));
    }
    Ok(OrderCmd::Set { agent, verb, ttl_s })
}

/// An agent's standing order. The verb and the split flag expire independently.
#[derive(Clone, Debug, Default)]
pub struct DriverOrder {
    pub verb: Option<(OVerb, u32)>,
    pub split: Option<(bool, u32)>,
}

impl DriverOrder {
    pub fn is_empty(&self) -> bool {
        self.verb.is_none() && self.split.is_none()
    }
}

pub fn ticks(ttl_s: f64) -> u32 {
    (ttl_s * TICKS_PER_S).ceil() as u32
}

/// The player index a target names (a cell id resolves to its owner). Names match exactly (ignoring case), else
/// by a unique substring. Dead players do not match.
pub fn resolve_player(w: &World, target: &Target, not: usize) -> Option<usize> {
    let alive = |i: usize| i != not && !w.players[i].removed && !w.players[i].cells.is_empty();
    match target {
        Target::Cell(id) => (0..w.players.len())
            .find(|&i| alive(i) && w.players[i].cells.iter().any(|&h| w.ents[h as usize].id == *id)),
        Target::Name(n) => {
            let n = n.to_lowercase();
            let exact: Vec<usize> = (0..w.players.len()).filter(|&i| alive(i) && w.players[i].name.to_lowercase() == n).collect();
            if let Some(&i) = exact.first() {
                return Some(i);
            }
            let part: Vec<usize> = (0..w.players.len()).filter(|&i| alive(i) && w.players[i].name.to_lowercase().contains(&n)).collect();
            if part.len() == 1 { Some(part[0]) } else { None }
        }
    }
}

impl AgentDriver {
    /// Drop expired parts of the order and return the order as `arena_client` applies it (None = no order).
    pub fn current_order(&mut self, w: &World) -> Option<Order> {
        let now = w.tick;
        let Some(mut d) = self.order.take() else { return None };
        if d.verb.as_ref().map_or(false, |(_, until)| now >= *until) {
            d.verb = None;
        }
        if d.split.as_ref().map_or(false, |(_, until)| now >= *until) {
            d.split = None;
        }
        if d.is_empty() {
            return None;
        }
        let split = d.split.map(|(on, _)| on);
        let out = match &d.verb {
            None => Order { verb: Verb::Keep, ids: vec![], split },
            Some((v, _)) => {
                let ids = match v.target().and_then(|t| resolve_player(w, t, self.pi)) {
                    Some(i) => w.players[i].cells.iter().map(|&h| w.ents[h as usize].id).collect(),
                    None => vec![],
                };
                let verb = match v {
                    OVerb::Hunt(_) => Verb::Hunt,
                    OVerb::Goto(x, y) => Verb::Goto(x.to_owned(), y.to_owned()),
                    OVerb::Farm => Verb::Farm,
                    OVerb::Flee => Verb::Flee,
                    OVerb::Hide => Verb::Hide,
                    OVerb::VirusSplit(_) => Verb::VirusSplit,
                    OVerb::Harvest(_) => Verb::Harvest,
                };
                Order { verb, ids, split }
            }
        };
        self.order = Some(d);
        Some(out)
    }
}

/// Vision agents: `hunt` / `harvest` / `goto` replace the net's target point, but only while no bigger cell is in
/// reach (then the net decides); `flee`, `farm`, `hide`, `virus-split` leave the net in charge. `split off`
/// vetoes the net's split; `split on` was already applied through `VisionBot::split_override`.
pub fn vision_override(o: &Order, s: &Snapshot, mut a: Out) -> Out {
    if o.split == Some(false) {
        a.split = false;
    }
    let me = *s.main().unwrap();
    let p = Params::default();
    let cl = classify(s, Some(&p));
    let gap = |c: &Cell| me.dist(c) - c.size - me.size;
    if cl.threats.iter().any(|t| gap(t) < danger_radius(t, &me, &p) * p.flee_margin) {
        return a;
    }
    match o.verb {
        Verb::Goto(x, y) => {
            a.x = x;
            a.y = y;
        }
        Verb::Hunt | Verb::Harvest => {
            let best = cl
                .prey
                .iter()
                .filter(|c| o.ids.contains(&c.id))
                .max_by(|x, y| (x.mass() / (gap(x).max(0.0) + 100.0)).partial_cmp(&(y.mass() / (gap(y).max(0.0) + 100.0))).unwrap());
            if let Some(t) = best {
                a.x = t.x;
                a.y = t.y;
            }
        }
        _ => {}
    }
    a
}

fn r0(v: f64) -> i64 {
    v.round() as i64
}

/// Leaderboard by mass: (player index, total mass), top `n`.
pub fn top_players(w: &World, n: usize) -> Vec<(usize, f64)> {
    let mut v: Vec<(usize, f64)> = (0..w.players.len())
        .filter(|&i| !w.players[i].removed && !w.players[i].cells.is_empty() && (w.players[i].kind != PKind::Human || w.players[i].connected))
        .map(|i| (i, w.players[i].total_mass(w)))
        .collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    v.truncate(n);
    v
}

pub fn leaderboard_json(w: &World, n: usize) -> Value {
    Value::Array(
        top_players(w, n)
            .iter()
            .enumerate()
            .map(|(r, &(i, m))| json!({"rank": r + 1, "name": w.players[i].name, "mass": r0(m), "cells": w.players[i].cells.len()}))
            .collect(),
    )
}

pub fn order_json(d: &AgentDriver, now: u32) -> Value {
    let Some(o) = &d.order else { return Value::Null };
    let left = |until: u32| ((until.saturating_sub(now)) as f64 / TICKS_PER_S * 10.0).round() / 10.0;
    json!({
        "verb": o.verb.as_ref().map(|(v, _)| v.show()),
        "ttl_left_s": o.verb.as_ref().map(|(_, u)| left(*u)),
        "split": o.split.map(|(on, _)| if on { "on" } else { "off" }),
    })
}

/// The commander's observation of one agent, relative to its biggest cell (x right, y down, world units).
pub fn scene(w: &World, d: &AgentDriver) -> Value {
    let pi = d.pi;
    let p = &w.players[pi];
    let t_s = w.tick as f64 / TICKS_PER_S;
    let lb: Vec<Value> = top_players(w, 10)
        .iter()
        .map(|&(i, m)| json!({"name": w.players[i].name, "mass": r0(m), "you": i == pi}))
        .collect();
    if p.cells.is_empty() {
        return json!({"agent": d.name, "t": t_s, "alive": false, "leaderboard": lb, "order": order_json(d, w.tick)});
    }
    let s: Snapshot = snapshot(w, pi);
    let me = *s.main().unwrap();
    let total = s.total_mass();
    let b = s.border;
    let params = Params::default();
    let gap = |c: &Cell| me.dist(c) - c.size - me.size;

    // visible players, grouped by owner
    struct G {
        owner: usize,
        cells: usize,
        mass: f64,
        big: Cell,
        big_mass: f64,
        near: Cell,
    }
    let mut groups: Vec<G> = Vec::new();
    let mut viruses: Vec<Cell> = Vec::new();
    for &h in &p.view_nodes {
        let e = &w.ents[h as usize];
        if e.owner == pi as i32 {
            continue;
        }
        let c = Cell::new(e.id, e.x, e.y, e.size, if e.kind == T_VIRUS { Kind::Virus } else { Kind::Player });
        if e.kind == T_VIRUS {
            viruses.push(c);
        } else if e.kind == T_PLAYER && e.owner >= 0 {
            let o = e.owner as usize;
            match groups.iter_mut().find(|g| g.owner == o) {
                Some(g) => {
                    g.cells += 1;
                    g.mass += c.mass();
                    if c.mass() > g.big_mass {
                        g.big = c;
                        g.big_mass = c.mass();
                    }
                    if gap(&c) < gap(&g.near) {
                        g.near = c;
                    }
                }
                None => groups.push(G { owner: o, cells: 1, mass: c.mass(), big: c, big_mass: c.mass(), near: c }),
            }
        }
    }
    groups.sort_by(|a, b| gap(&a.near).partial_cmp(&gap(&b.near)).unwrap());
    groups.truncate(12);
    let min_virus = viruses.iter().map(|v| v.mass()).fold(f64::INFINITY, f64::min);
    let players: Vec<Value> = groups
        .iter()
        .map(|g| {
            let n = &g.near;
            let gp = gap(n);
            json!({
                "name": w.players[g.owner].name,
                "cell_id": n.id,
                "mass": r0(g.mass),
                "biggest_cell_mass": r0(g.big_mass),
                "cells_visible": g.cells,
                "cells": w.players[g.owner].cells.len(),
                "x": r0(n.x - me.x), "y": r0(n.y - me.y),
                "dist": r0(me.dist(n)), "gap": r0(gp),
                "can_eat_us": g.big_mass > me.mass() * EAT_RATIO,
                "we_can_eat": me.mass() > g.big_mass * EAT_RATIO,
                "split_threat": g.big_mass > me.mass() * SPLIT_KILL_RATIO && gp < SPLIT_REACH + params.danger_margin,
                "split_kill": me.mass() / 2.0 > g.big_mass * EAT_RATIO && gp < SPLIT_REACH,
                "virus_pop_possible": g.big_mass > min_virus * 1.33,
                "popped": g.cells >= 3,
            })
        })
        .collect();
    viruses.sort_by(|a, b| gap(a).partial_cmp(&gap(b)).unwrap());
    let virus_json: Vec<Value> = viruses
        .iter()
        .take(6)
        .map(|v| {
            json!({"x": r0(v.x - me.x), "y": r0(v.y - me.y), "mass": r0(v.mass()), "gap": r0(gap(v)),
                   "pops_us": me.mass() > v.mass() * 1.33, "hides_us": v.mass() > me.mass() * EAT_RATIO})
        })
        .collect();
    let cl = classify(&s, Some(&params));
    let danger = cl.threats.iter().any(|t| gap(t) < danger_radius(t, &me, &params) * params.flee_margin);
    let last_mode = d.policy.last_plan().map(|(pl, _)| pl.mode.as_str());
    json!({
        "agent": d.name,
        "t": t_s,
        "alive": true,
        "me": {"mass": r0(total), "cells": s.mine.len(), "x": r0(me.x), "y": r0(me.y), "biggest_cell_mass": r0(me.mass()),
               "kills": d.stats.kills, "deaths": d.stats.deaths, "mode": last_mode, "in_danger": danger},
        "leaderboard": lb,
        "players": players,
        "viruses": virus_json,
        "walls": {"left": r0(me.x - b.left - me.size), "right": r0(b.right - me.x - me.size),
                  "top": r0(me.y - b.top - me.size), "bottom": r0(b.bottom - me.y - me.size)},
        "order": order_json(d, w.tick),
    })
}

/// The vision-style frame the agent sees (camera = its view box), 256 px grayscale PNG.
pub fn frame_png(w: &World, pi: usize) -> Option<Vec<u8>> {
    if w.players[pi].cells.is_empty() {
        return None;
    }
    let s = snapshot(w, pi);
    let (x0, y0, x1, y1) = w.view_box(pi);
    let px = 256;
    let img = render(&s, (x0 + x1) / 2.0, (y0 + y1) / 2.0, (x1 - x0) / 2.0, px);
    Some(png_gray(px, px, &img))
}

pub fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len() * 4 / 3 + 4);
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn parses_orders() {
        assert_eq!(
            parse_order(&p("order c3-0 hunt Big Bob ttl=20")).unwrap(),
            OrderCmd::Set { agent: "c3-0".into(), verb: OVerb::Hunt(Target::Name("Big Bob".into())), ttl_s: 20.0 }
        );
        assert_eq!(
            parse_order(&p("order a hunt #42")).unwrap(),
            OrderCmd::Set { agent: "a".into(), verb: OVerb::Hunt(Target::Cell(42)), ttl_s: DEFAULT_TTL_S }
        );
        assert_eq!(
            parse_order(&p("order a goto 100 -250.5 ttl=3")).unwrap(),
            OrderCmd::Set { agent: "a".into(), verb: OVerb::Goto(100.0, -250.5), ttl_s: 3.0 }
        );
        assert_eq!(parse_order(&p("order a split off")).unwrap(), OrderCmd::Split { agent: "a".into(), on: false, ttl_s: DEFAULT_TTL_S });
        assert_eq!(parse_order(&p("order a free")).unwrap(), OrderCmd::Free { agent: "a".into() });
        assert!(matches!(parse_order(&p("order a virus-split Foo")).unwrap(), OrderCmd::Set { verb: OVerb::VirusSplit(_), .. }));
        assert!(matches!(parse_order(&p("order a harvest Foo")).unwrap(), OrderCmd::Set { verb: OVerb::Harvest(_), .. }));
        for v in ["farm", "flee", "hide"] {
            assert!(parse_order(&p(&format!("order a {v}"))).is_ok());
        }
    }

    #[test]
    fn rejects_bad_orders() {
        for bad in ["order", "order a", "order a hunt", "order a goto 1", "order a goto x y", "order a dance", "order a farm now",
                    "order a split maybe", "order a flee ttl=-3", "order a harvest"] {
            assert!(parse_order(&p(bad)).is_err(), "{bad} should fail");
        }
    }

    #[test]
    fn base64_matches_reference() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }
}
