//! Server control page (F2): talks to `ogar serve --control-port` over a plain TCP line protocol on a
//! background thread, so the render loop never blocks. Shows status and offers speed / pause / food /
//! add-bot / remove-bot actions.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::Duration;

use macroquad::prelude::*;

use crate::Game;

pub const SPEEDS: [(&str, f64); 6] = [("0.5x", 0.5), ("1x", 1.0), ("2x", 2.0), ("4x", 4.0), ("10x", 10.0), ("max", 0.0)];
pub const PRESETS: [(&str, &str); 6] = [
    ("champ", "runs/final_champion.json"),
    ("r2g9", "runs/r2/champion_009.json"),
    ("safe", "bot/champ_safe.json"),
    ("ramp_r2", "bot/r2g9_ramp800.json"),
    ("ramp300", "runs/r2g9_ramp300.json"),
    ("c3_4", "bot/c3_4.json"),
];
const POLL_SECS: f64 = 2.0;
const PW: f32 = 780.0;
const PH: f32 = 600.0;

/// Default control address: the game server's host with the port + 1 (9600 -> 9601).
pub fn derive_addr(server: &str) -> String {
    let rest = server.split("://").nth(1).unwrap_or(server);
    let hostport = rest.split('/').next().unwrap_or(rest);
    match hostport.rsplit_once(':') {
        Some((h, p)) => match p.parse::<u16>() {
            Ok(p) => format!("{h}:{}", p.saturating_add(1)),
            Err(_) => "127.0.0.1:9601".into(),
        },
        None => format!("{hostport}:9601"),
    }
}

pub enum Req {
    Poll,
    Cmds(Vec<String>),
}

pub enum Msg {
    Status(String),
    Reply(String),
    Err(String),
}

#[derive(Clone, Debug, Default)]
pub struct Status {
    pub tick: u64,
    pub tps: f64,
    pub speed: f64,
    pub paused: bool,
    pub food: String,
    pub foodmult: f64,
    pub bots: String,
    pub players: String,
    pub clients: String,
    pub alive: String,
    pub total: String,
    pub agents: Vec<(String, String)>,
}

pub fn parse_status(line: &str) -> Option<Status> {
    let mut s = Status::default();
    let mut any = false;
    for kv in line.split_whitespace() {
        let Some((k, v)) = kv.split_once('=') else { continue };
        any = true;
        match k {
            "tick" => s.tick = v.parse().unwrap_or(0),
            "tps" => s.tps = v.parse().unwrap_or(0.0),
            "speed" => s.speed = v.parse().unwrap_or(1.0),
            "paused" => s.paused = v == "1",
            "food" => s.food = v.into(),
            "foodmult" => s.foodmult = v.parse().unwrap_or(1.0),
            "bots" => s.bots = v.into(),
            "players" => s.players = v.into(),
            "clients" => s.clients = v.into(),
            "alive" => s.alive = v.into(),
            "total" => s.total = v.into(),
            "agentlist" => {
                s.agents = v
                    .split(',')
                    .filter(|x| !x.is_empty())
                    .filter_map(|x| x.rsplit_once(':').map(|(n, m)| (n.to_string(), m.to_string())))
                    .collect()
            }
            _ => {}
        }
    }
    if any && line.contains("tick=") { Some(s) } else { None }
}

fn exchange(addr: &str, token: &str, lines: &[String]) -> Result<Vec<String>, String> {
    let sa = addr.to_socket_addrs().map_err(|e| format!("bad address {addr}: {e}"))?.next().ok_or("no address")?;
    let mut st = TcpStream::connect_timeout(&sa, Duration::from_millis(1500)).map_err(|_| format!("control port not reachable ({addr})"))?;
    let _ = st.set_read_timeout(Some(Duration::from_secs(6)));
    let _ = st.set_write_timeout(Some(Duration::from_secs(3)));
    let mut rd = BufReader::new(st.try_clone().map_err(|e| e.to_string())?);
    let mut out = Vec::new();
    for l in lines {
        let l = if token.is_empty() { l.clone() } else { format!("{token} {l}") };
        st.write_all(format!("{l}\n").as_bytes()).map_err(|e| format!("send failed: {e}"))?;
        let mut r = String::new();
        rd.read_line(&mut r).map_err(|e| format!("no reply: {e}"))?;
        let r = r.trim().to_string();
        if r.contains("unauthorized") {
            return Err("unauthorized: wrong or missing control token".into());
        }
        out.push(r);
    }
    Ok(out)
}

fn worker(rx: Receiver<(String, String, Req)>, tx: Sender<Msg>) {
    while let Ok((addr, token, req)) = rx.recv() {
        let mut lines = match req {
            Req::Poll => vec![],
            Req::Cmds(c) => c,
        };
        let n = lines.len();
        lines.push("status".into());
        match exchange(&addr, &token, &lines) {
            Ok(replies) => {
                for r in &replies[..n] {
                    let _ = tx.send(Msg::Reply(r.clone()));
                }
                let _ = tx.send(Msg::Status(replies[n].clone()));
            }
            Err(e) => {
                let _ = tx.send(Msg::Err(e));
            }
        }
    }
}

pub struct Control {
    tx: Sender<(String, String, Req)>,
    rx: Receiver<Msg>,
    pub open: bool,
    /// set when F2 opened the panel this frame: the same key press must not close it again in control_frame
    pub just_opened: bool,
    pub status: Option<Status>,
    pub message: String,
    pub error: Option<String>,
    in_flight: u32,
    last_poll: f64,
    pub focus: Option<usize>, // 0 = address, 1 = token
    /// (label, genome path)
    pub genomes: Vec<(String, String)>,
    pub sel_genome: usize,
    pub genome_scroll: usize,
    pub count: u32,
    pub sel_agent: Option<String>,
}

impl Control {
    pub fn new() -> Control {
        let (tx, wrx) = channel();
        let (wtx, rx) = channel();
        std::thread::spawn(move || worker(wrx, wtx));
        Control {
            tx,
            rx,
            open: false,
            just_opened: false,
            status: None,
            message: String::new(),
            error: None,
            in_flight: 0,
            last_poll: -100.0,
            focus: None,
            genomes: Vec::new(),
            sel_genome: 0,
            genome_scroll: 0,
            count: 1,
            sel_agent: None,
        }
    }

    pub fn scan_genomes(&mut self) {
        let mut g: Vec<(String, String)> = PRESETS.iter().map(|(n, p)| (format!("{n}  ({p})"), p.to_string())).collect();
        for dir in ["bot", "runs/ab"] {
            let mut files: Vec<String> = std::fs::read_dir(dir)
                .map(|r| r.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.ends_with(".json")).collect())
                .unwrap_or_default();
            files.sort();
            for f in files {
                let p = format!("{dir}/{f}");
                if !g.iter().any(|(_, q)| *q == p) {
                    g.push((p.clone(), p));
                }
            }
        }
        self.genomes = g;
        self.sel_genome = self.sel_genome.min(self.genomes.len().saturating_sub(1));
    }

    fn send(&mut self, addr: &str, token: &str, req: Req, now: f64) {
        self.in_flight += 1;
        self.last_poll = now;
        let _ = self.tx.send((addr.to_string(), token.to_string(), req));
    }

    fn drain(&mut self) {
        while let Ok(m) = self.rx.try_recv() {
            match m {
                Msg::Status(l) => {
                    self.in_flight = self.in_flight.saturating_sub(1);
                    match parse_status(&l) {
                        Some(s) => {
                            self.status = Some(s);
                            self.error = None;
                        }
                        None => self.error = Some(format!("unexpected reply: {l}")),
                    }
                }
                Msg::Reply(r) => self.message = r,
                Msg::Err(e) => {
                    self.in_flight = self.in_flight.saturating_sub(1);
                    self.error = Some(e);
                    self.status = None;
                }
            }
        }
    }
}

/// Name for an agent derived from the genome path, e.g. bot/c3_4.json -> c3_4.
pub fn agent_key(label: &str, path: &str) -> String {
    let name = if let Some((n, _)) = PRESETS.iter().find(|(_, p)| *p == path) {
        n.to_string()
    } else {
        let stem = path.rsplit('/').next().unwrap_or(path);
        stem.trim_end_matches(".json").to_string()
    };
    let _ = label;
    name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect()
}

fn inside(mx: f32, my: f32, x: f32, y: f32, w: f32, h: f32) -> bool {
    mx >= x && mx <= x + w && my >= y && my <= y + h
}

impl Game {
    pub fn control_addr(&self) -> String {
        if self.set.control_addr.trim().is_empty() { derive_addr(&self.set.server) } else { self.set.control_addr.trim().to_string() }
    }

    pub fn open_control(&mut self) {
        self.ctl.open = true;
        self.ctl.just_opened = true;
        self.ctl.focus = None;
        self.ctl.last_poll = -100.0;
        self.ctl.scan_genomes();
    }

    pub fn close_control(&mut self) {
        self.ctl.open = false;
        self.ctl.focus = None;
        self.apply_settings();
    }

    /// +/- keys: the next speed step up or down (0.25x .. 10x, then max = unthrottled), via the control port.
    pub fn step_speed(&mut self, up: bool, now: f64) {
        const STEPS: [f64; 6] = [0.25, 0.5, 1.0, 2.0, 4.0, 10.0];
        let cur = self.ctl.status.as_ref().map_or(1.0, |s| s.speed);
        let next = if up {
            if cur == 0.0 { 0.0 } else { STEPS.iter().copied().find(|&v| v > cur + 1e-9).unwrap_or(0.0) }
        } else if cur == 0.0 {
            10.0
        } else {
            STEPS.iter().rev().copied().find(|&v| v < cur - 1e-9).unwrap_or(0.25)
        };
        if let Some(st) = self.ctl.status.as_mut() {
            st.speed = next; // show it at once; the next status poll confirms
        }
        self.status = if next == 0.0 { "speed: max".into() } else { format!("speed: {next}x") };
        self.ctl_send(vec![format!("speed {next}")], now);
    }

    fn ctl_send(&mut self, cmds: Vec<String>, now: f64) {
        let (a, t) = (self.control_addr(), self.set.control_token.clone());
        self.ctl.send(&a, &t, Req::Cmds(cmds), now);
    }

    fn ctl_button(&self, label: &str, x: f32, y: f32, w: f32, h: f32, on: bool, enabled: bool) -> bool {
        let (mx, my) = mouse_position();
        let hover = enabled && inside(mx, my, x, y, w, h);
        let bg = if on { Color::new(0.2, 0.5, 0.35, 0.95) } else if hover { Color::new(0.3, 0.38, 0.5, 0.95) } else { Color::new(0.2, 0.24, 0.32, 0.95) };
        draw_rectangle(x, y, w, h, bg);
        let tw = self.fonts.size(label, 20.0).width;
        let c = if enabled { WHITE } else { Color::new(0.5, 0.5, 0.55, 1.0) };
        self.fonts.draw(label, x + (w - tw) / 2.0, y + h / 2.0 + 7.0, 20.0, c);
        enabled && hover && is_mouse_button_pressed(MouseButton::Left)
    }

    /// While the page is closed: keep polling the status (for the HUD player count), at a slower pace.
    pub fn control_background(&mut self, now: f64) {
        self.ctl.drain();
        if self.ctl.in_flight == 0 && now - self.ctl.last_poll >= 2.0 * POLL_SECS {
            let (a, t) = (self.control_addr(), self.set.control_token.clone());
            self.ctl.send(&a, &t, Req::Poll, now);
        }
    }

    /// Handles input and draws the control page (immediate mode). Called once per frame while open.
    pub fn control_frame(&mut self, now: f64) {
        self.ctl.drain();
        if self.ctl.in_flight == 0 && now - self.ctl.last_poll >= POLL_SECS {
            let (a, t) = (self.control_addr(), self.set.control_token.clone());
            self.ctl.send(&a, &t, Req::Poll, now);
        }
        // text input
        while let Some(ch) = get_char_pressed() {
            if ch.is_control() {
                continue;
            }
            match self.ctl.focus {
                Some(0) if self.set.control_addr.len() < 80 => self.set.control_addr.push(ch),
                Some(1) if self.set.control_token.len() < 80 => self.set.control_token.push(ch),
                _ => {}
            }
        }
        if is_key_pressed(KeyCode::Backspace) {
            match self.ctl.focus {
                Some(0) => {
                    self.set.control_addr.pop();
                }
                Some(1) => {
                    self.set.control_token.pop();
                }
                _ => {}
            }
        }
        if is_key_pressed(KeyCode::Enter) && self.ctl.focus.is_some() {
            self.ctl.focus = None;
            self.apply_settings();
            self.ctl.last_poll = -100.0;
        }
        let click = is_mouse_button_pressed(MouseButton::Left);
        let (mx, my) = mouse_position();
        let (w, h) = (screen_width(), screen_height());
        draw_rectangle(0.0, 0.0, w, h, Color::new(0.0, 0.0, 0.0, 0.5));
        let ox = ((w - PW) / 2.0).max(4.0);
        let oy = ((h - PH) / 2.0).max(4.0);
        draw_rectangle(ox, oy, PW, PH, Color::new(0.1, 0.11, 0.15, 0.97));
        let dim = Color::new(0.7, 0.75, 0.8, 1.0);
        let val = Color::new(0.85, 0.9, 1.0, 1.0);
        let mut reapply = false;
        let f = &self.fonts;
        f.draw("Server control", ox + 16.0, oy + 34.0, 30.0, WHITE);
        f.draw("F2 / Esc: close", ox + PW - 160.0, oy + 32.0, 18.0, dim);

        // address + token fields
        let blink = (now * 2.0) as i64 % 2 == 0;
        let fields = [("Control address", self.set.control_addr.clone(), 0usize), ("Token (optional)", "*".repeat(self.set.control_token.chars().count()), 1usize)];
        for (i, (label, v, idx)) in fields.iter().enumerate() {
            let y = oy + 50.0 + i as f32 * 34.0;
            let focused = self.ctl.focus == Some(*idx);
            if focused || inside(mx, my, ox + 6.0, y, PW - 12.0, 32.0) {
                draw_rectangle(ox + 6.0, y, PW - 12.0, 32.0, Color::new(0.25, 0.3, 0.4, 0.8));
            }
            f.draw(label, ox + 18.0, y + 23.0, 22.0, WHITE);
            let mut shown = v.clone();
            if *idx == 0 && shown.is_empty() {
                shown = format!("{}  (default)", self.control_addr());
            }
            let caret = if focused && blink { "|" } else { "" };
            f.draw(&format!("{shown}{caret}"), ox + 250.0, y + 23.0, 22.0, val);
            if click && inside(mx, my, ox + 6.0, y, PW - 12.0, 32.0) {
                self.ctl.focus = Some(*idx);
            }
        }
        if click && !inside(mx, my, ox + 6.0, oy + 50.0, PW - 12.0, 66.0) {
            reapply = self.ctl.focus.is_some();
            self.ctl.focus = None;
        }

        // status
        let st = self.ctl.status.clone();
        let y0 = oy + 134.0;
        match (&st, &self.ctl.error) {
            (Some(s), _) => {
                let line = format!(
                    "tick {}   tps {:.0}   speed {}   {}   food {} (x{})   bots {}  players {}  agents {}",
                    s.tick,
                    s.tps,
                    if s.speed == 0.0 { "max".into() } else { format!("{}x", s.speed) },
                    if s.paused { "PAUSED" } else { "running" },
                    s.food,
                    s.foodmult,
                    s.bots,
                    s.players,
                    s.agents.len()
                );
                f.draw(&line, ox + 16.0, y0, 20.0, Color::new(0.5, 1.0, 0.6, 1.0));
            }
            (None, Some(e)) => f.draw(e, ox + 16.0, y0, 20.0, Color::new(1.0, 0.5, 0.5, 1.0)),
            (None, None) => f.draw("connecting...", ox + 16.0, y0, 20.0, dim),
        }
        let connected = st.is_some();

        // speed presets
        f.draw("Speed", ox + 16.0, y0 + 44.0, 22.0, WHITE);
        let mut cmds: Vec<String> = Vec::new();
        for (i, (label, x)) in SPEEDS.iter().enumerate() {
            let on = st.as_ref().map_or(false, |s| (s.speed - x).abs() < 1e-9);
            if self.ctl_button(label, ox + 110.0 + i as f32 * 78.0, y0 + 22.0, 70.0, 32.0, on, connected) {
                cmds.push(format!("speed {x}"));
            }
        }
        // pause / food
        let paused = st.as_ref().map_or(false, |s| s.paused);
        if self.ctl_button(if paused { "Resume" } else { "Pause" }, ox + 16.0, y0 + 66.0, 110.0, 32.0, paused, connected) {
            cmds.push(if paused { "resume".into() } else { "pause".into() });
        }
        f.draw("Food density", ox + 160.0, y0 + 89.0, 22.0, WHITE);
        let fm = st.as_ref().map_or(1.0, |s| s.foodmult);
        if self.ctl_button("-", ox + 310.0, y0 + 66.0, 40.0, 32.0, false, connected) {
            cmds.push(format!("food {}", (fm - 0.5).max(0.25)));
        }
        f.draw(&format!("x{fm}"), ox + 360.0, y0 + 89.0, 22.0, val);
        if self.ctl_button("+", ox + 430.0, y0 + 66.0, 40.0, 32.0, false, connected) {
            cmds.push(format!("food {}", (fm + 0.5).min(10.0)));
        }

        // add bot (left) / agents (right)
        let ly = y0 + 120.0;
        f.draw("Add bot (genome)", ox + 16.0, ly, 22.0, WHITE);
        f.draw("Agents (click to select)", ox + 400.0, ly, 22.0, WHITE);
        let rows = 9usize;
        let rh = 26.0;
        let list_y = ly + 10.0;
        draw_rectangle(ox + 12.0, list_y, 372.0, rows as f32 * rh, Color::new(0.06, 0.07, 0.1, 0.9));
        if inside(mx, my, ox + 12.0, list_y, 372.0, rows as f32 * rh) {
            let (_, wheel) = mouse_wheel();
            if wheel > 0.0 {
                self.ctl.genome_scroll = self.ctl.genome_scroll.saturating_sub(1);
            } else if wheel < 0.0 {
                self.ctl.genome_scroll = (self.ctl.genome_scroll + 1).min(self.ctl.genomes.len().saturating_sub(rows));
            }
        }
        for r in 0..rows {
            let i = self.ctl.genome_scroll + r;
            let Some((label, _)) = self.ctl.genomes.get(i) else { break };
            let y = list_y + r as f32 * rh;
            let sel = i == self.ctl.sel_genome;
            if sel {
                draw_rectangle(ox + 12.0, y, 372.0, rh, Color::new(0.2, 0.5, 0.35, 0.8));
            } else if inside(mx, my, ox + 12.0, y, 372.0, rh) {
                draw_rectangle(ox + 12.0, y, 372.0, rh, Color::new(0.25, 0.3, 0.4, 0.8));
            }
            let t: String = label.chars().take(38).collect();
            f.draw(&t, ox + 18.0, y + 19.0, 18.0, WHITE);
            if click && inside(mx, my, ox + 12.0, y, 372.0, rh) {
                self.ctl.sel_genome = i;
            }
        }
        let by = list_y + rows as f32 * rh + 10.0;
        f.draw("Count", ox + 16.0, by + 23.0, 22.0, WHITE);
        if self.ctl_button("-", ox + 90.0, by, 34.0, 32.0, false, true) {
            self.ctl.count = self.ctl.count.saturating_sub(1).max(1);
        }
        f.draw(&self.ctl.count.to_string(), ox + 134.0, by + 23.0, 22.0, val);
        if self.ctl_button("+", ox + 160.0, by, 34.0, 32.0, false, true) {
            self.ctl.count = (self.ctl.count + 1).min(4);
        }
        if self.ctl_button("Add bot", ox + 240.0, by, 140.0, 32.0, false, connected && !self.ctl.genomes.is_empty()) {
            if let Some((label, path)) = self.ctl.genomes.get(self.ctl.sel_genome) {
                cmds.push(format!("add {}={}:{}", agent_key(label, path), path, self.ctl.count));
            }
        }

        // agents list
        draw_rectangle(ox + 396.0, list_y, PW - 408.0, rows as f32 * rh, Color::new(0.06, 0.07, 0.1, 0.9));
        let agents = st.as_ref().map(|s| s.agents.clone()).unwrap_or_default();
        for (r, (name, mass)) in agents.iter().take(rows).enumerate() {
            let y = list_y + r as f32 * rh;
            let sel = self.ctl.sel_agent.as_deref() == Some(name.as_str());
            if sel {
                draw_rectangle(ox + 396.0, y, PW - 408.0, rh, Color::new(0.5, 0.25, 0.25, 0.8));
            } else if inside(mx, my, ox + 396.0, y, PW - 408.0, rh) {
                draw_rectangle(ox + 396.0, y, PW - 408.0, rh, Color::new(0.25, 0.3, 0.4, 0.8));
            }
            f.draw(name, ox + 402.0, y + 19.0, 18.0, WHITE);
            f.draw(&format!("{mass} mass"), ox + 600.0, y + 19.0, 18.0, val);
            if click && inside(mx, my, ox + 396.0, y, PW - 408.0, rh) {
                self.ctl.sel_agent = Some(name.clone());
            }
        }
        if agents.len() > rows {
            f.draw(&format!("... {} more", agents.len() - rows), ox + 402.0, list_y + rows as f32 * rh - 4.0, 14.0, dim);
        }
        let sel_ok = self.ctl.sel_agent.as_ref().map_or(false, |s| agents.iter().any(|(n, _)| n == s));
        if self.ctl_button("Remove selected", ox + 396.0, by, 180.0, 32.0, false, connected && sel_ok) {
            if let Some(n) = self.ctl.sel_agent.take() {
                cmds.push(format!("remove {n}"));
            }
        }
        if self.ctl_button("Remove all bots", ox + 586.0, by, 180.0, 32.0, false, connected && !agents.is_empty()) {
            for (n, _) in &agents {
                cmds.push(format!("remove {n}"));
            }
        }
        f.draw(&self.ctl.message, ox + 16.0, oy + PH - 14.0, 16.0, dim);

        if reapply {
            self.apply_settings();
            self.ctl.last_poll = -100.0;
        }
        if !cmds.is_empty() {
            self.ctl_send(cmds, now);
        }
        if is_key_pressed(KeyCode::F2) && !self.ctl.just_opened {
            self.close_control();
        }
        self.ctl.just_opened = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive() {
        assert_eq!(derive_addr("ws://127.0.0.1:9600"), "127.0.0.1:9601");
        assert_eq!(derive_addr("ws://example.com:9600/"), "example.com:9601");
        assert_eq!(derive_addr("ws://example.com"), "example.com:9601");
    }

    #[test]
    fn status_parse() {
        let s = parse_status("tick=5 tps=25.0 speed=2 paused=1 bots=8 players=0 agents=2 clients=0 food=736 foodmult=2 agentlist=a-0:19,b-0:20").unwrap();
        assert_eq!(s.speed, 2.0);
        assert!(s.paused);
        assert_eq!(s.agents, vec![("a-0".to_string(), "19".to_string()), ("b-0".to_string(), "20".to_string())]);
        assert!(parse_status("error: unauthorized").is_none());
    }
}
