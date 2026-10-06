//! Real-time websocket server: MultiOgar's client protocol (protocol 6) plus arena.js' extras
//! (0xC8/0xC9 lockstep agents, 0xCA follow, 0xCB spawn size, 0xCC food density, 0xCD +500 mass),
//! in-process rule-bot agents and a small text control interface (stdin / TCP) to add or remove them.

use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc::unbounded_channel, mpsc::UnboundedSender, oneshot};
use tokio_tungstenite::tungstenite::Message;

use crate::agent::AgentDriver;
use crate::config::Config;
use crate::matches::parse_lineup;
use crate::protocol::*;
use crate::world::{PKind, World};

const SERVER_NAME: &str = "MultiOgar-Edited 2.0.0";
const GROW_MASS: f64 = 500.0;
const ACK_TIMEOUT: Duration = Duration::from_millis(2000);

pub struct ServeOpts {
    pub full_view: bool,
    pub port: u16,
    pub bind: String,
    pub bots: usize,
    pub map: f64,
    pub food_mult: f64,
    pub seed: u64,
    pub sync: u64,
    pub max_tps: f64,
    pub agents: Vec<String>,
    pub follow: String,
    pub log_players: f64,
    pub control_port: Option<u16>,
}

enum Ev {
    Open(usize, UnboundedSender<Vec<u8>>),
    Msg(usize, Vec<u8>),
    Close(usize),
    Cmd(String, Option<oneshot::Sender<String>>),
}

struct Conn {
    tx: UnboundedSender<Vec<u8>>,
    player: usize,
    stage: u8, // 0 = want protocol, 1 = want key, 2 = playing
    protocol: u32,
    is_agent: bool,
    ack_tick: i64,
}

async fn handle_conn(stream: TcpStream, id: usize, ev: Sender<Ev>) {
    let _ = stream.set_nodelay(true);
    let Ok(ws) = tokio_tungstenite::accept_async(stream).await else { return };
    let (mut sink, mut source) = ws.split();
    let (tx, mut rx) = unbounded_channel::<Vec<u8>>();
    if ev.send(Ev::Open(id, tx)).is_err() {
        return;
    }
    let writer = tokio::spawn(async move {
        while let Some(b) = rx.recv().await {
            if sink.send(Message::Binary(b.into())).await.is_err() {
                return;
            }
        }
        let _ = sink.close().await;
    });
    while let Some(m) = source.next().await {
        match m {
            Ok(Message::Binary(b)) => {
                if ev.send(Ev::Msg(id, b.to_vec())).is_err() {
                    break;
                }
            }
            Ok(Message::Close(_)) | Err(_) => break,
            _ => {}
        }
    }
    let _ = ev.send(Ev::Close(id));
    writer.abort();
}

async fn control_conn(stream: TcpStream, ev: Sender<Ev>) {
    let (rd, mut wr) = stream.into_split();
    let mut lines = BufReader::new(rd).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let (tx, rx) = oneshot::channel();
        if ev.send(Ev::Cmd(line, Some(tx))).is_err() {
            return;
        }
        let reply = rx.await.unwrap_or_default();
        if wr.write_all(format!("{reply}\n").as_bytes()).await.is_err() {
            return;
        }
    }
}

pub struct Server {
    o: ServeOpts,
    w: World,
    conns: HashMap<usize, Conn>,
    drivers: Vec<AgentDriver>,
    ev_rx: Receiver<Ev>,
    base_food: f64,
    quit: bool,
}

pub fn serve(o: ServeOpts) -> Result<(), String> {
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().map_err(|e| e.to_string())?;
    let (ev_tx, ev_rx) = channel::<Ev>();
    let addr = format!("{}:{}", o.bind, o.port);
    let listener = rt.block_on(TcpListener::bind(&addr)).map_err(|e| format!("bind {addr}: {e}"))?;
    {
        let ev = ev_tx.clone();
        rt.spawn(async move {
            let mut next = 0usize;
            loop {
                let Ok((stream, _)) = listener.accept().await else { continue };
                tokio::spawn(handle_conn(stream, next, ev.clone()));
                next += 1;
            }
        });
    }
    if let Some(cp) = o.control_port {
        let l = rt.block_on(TcpListener::bind(("127.0.0.1", cp))).map_err(|e| format!("control port {cp}: {e}"))?;
        let ev = ev_tx.clone();
        rt.spawn(async move {
            loop {
                let Ok((stream, _)) = l.accept().await else { continue };
                tokio::spawn(control_conn(stream, ev.clone()));
            }
        });
    }
    {
        let ev = ev_tx.clone();
        std::thread::spawn(move || {
            let stdin = std::io::stdin();
            let mut line = String::new();
            while let Ok(n) = stdin.read_line(&mut line) {
                if n == 0 {
                    break; // stdin closed (e.g. launched detached): keep serving
                }
                if ev.send(Ev::Cmd(line.trim().to_string(), None)).is_err() {
                    break;
                }
                line.clear();
            }
        });
    }
    let cfg = Config::new(o.map, o.bots, o.food_mult);
    let base_food = cfg.base_food();
    let mut w = World::new(cfg, o.seed);
    w.full_view = o.full_view;
    w.track_eats = true;
    let mut s = Server { o, w, conns: HashMap::new(), drivers: Vec::new(), ev_rx, base_food, quit: false };
    let specs = std::mem::take(&mut s.o.agents);
    if !specs.is_empty() {
        println!("{}", s.add_agents(&specs));
    }
    println!("[arena] port={} map={} bots={} sync={} seed={} mode=realtime", s.o.port, s.o.map, s.o.bots, s.o.sync, s.o.seed);
    s.run();
    Ok(())
}

impl Server {
    fn add_agents(&mut self, specs: &[String]) -> String {
        let lineup = match parse_lineup(specs) {
            Ok(l) => l,
            Err(e) => return format!("error: {e}"),
        };
        let mut added = Vec::new();
        for (name, spec) in lineup {
            let key = name.rsplit_once('-').map_or(name.as_str(), |(k, _)| k).to_string();
            let mut idx = 0;
            while self.drivers.iter().any(|d| d.name == format!("{key}-{idx}")) {
                idx += 1;
            }
            let nm = format!("{key}-{idx}");
            let d = AgentDriver::new(&mut self.w, &nm, &spec, self.o.sync);
            self.drivers.push(d);
            added.push(nm);
        }
        format!("added {}", added.join(" "))
    }

    fn remove_agents(&mut self, prefix: &str) -> String {
        let mut gone = Vec::new();
        let mut i = 0;
        while i < self.drivers.len() {
            if self.drivers[i].name.starts_with(prefix) {
                let d = self.drivers.remove(i);
                self.w.remove_player(d.pi);
                gone.push(d.name);
            } else {
                i += 1;
            }
        }
        format!("removed {}", gone.join(" "))
    }

    fn command(&mut self, line: &str) -> String {
        let parts: Vec<String> = line.split_whitespace().map(String::from).collect();
        match parts.first().map(String::as_str) {
            Some("add") => self.add_agents(&parts[1..]),
            Some("remove") | Some("rm") => match parts.get(1) {
                Some(p) => self.remove_agents(p),
                None => "usage: remove <name prefix>".into(),
            },
            Some("list") => {
                let v: Vec<String> = self
                    .drivers
                    .iter()
                    .map(|d| {
                        let m = self.w.players[d.pi].total_mass(&self.w);
                        format!("{}({:.0})", d.name, m)
                    })
                    .collect();
                format!("agents: {}", v.join(" "))
            }
            Some("food") => match parts.get(1).and_then(|v| v.parse::<f64>().ok()) {
                Some(m) => {
                    self.w.cfg.set_food_mult(self.base_food, (m * 100.0) as u16);
                    format!("food target {}", self.w.cfg.food_min_amount)
                }
                None => "usage: food <multiplier>".into(),
            },
            Some("quit") => {
                self.quit = true;
                "bye".into()
            }
            Some("help") | None => "commands: add <name=genome.json:N ...> | remove <prefix> | list | food <mult> | quit".into(),
            Some(c) => format!("unknown command {c}; try help"),
        }
    }

    fn send(&self, id: usize, bytes: Vec<u8>) {
        if let Some(c) = self.conns.get(&id) {
            let _ = c.tx.send(bytes);
        }
    }

    fn on_event(&mut self, ev: Ev) {
        match ev {
            Ev::Open(id, tx) => {
                let pi = self.w.add_player(PKind::Human);
                if !self.o.follow.is_empty() {
                    self.w.players[pi].follow_prefix = Some(self.o.follow.clone());
                }
                self.conns.insert(id, Conn { tx, player: pi, stage: 0, protocol: 0, is_agent: false, ack_tick: -1 });
            }
            Ev::Close(id) => {
                if let Some(c) = self.conns.remove(&id) {
                    let p = &mut self.w.players[c.player];
                    p.connected = false;
                    p.closed_tick = self.w.tick;
                }
            }
            Ev::Msg(id, data) => self.on_message(id, &data),
            Ev::Cmd(line, reply) => {
                let r = self.command(&line);
                match reply {
                    Some(tx) => {
                        let _ = tx.send(r);
                    }
                    None => println!("[ctl] {r}"),
                }
            }
        }
    }

    fn on_message(&mut self, id: usize, m: &[u8]) {
        if m.is_empty() || m.len() > 256 {
            if m.len() > 256 {
                self.conns.remove(&id);
            }
            return;
        }
        let Some(c) = self.conns.get(&id) else { return };
        let (pi, stage, protocol, is_agent) = (c.player, c.stage, c.protocol, c.is_agent);
        let op = m[0];
        // arena.js extras are honoured before the handshake, like its monkeypatch
        match op {
            OP_AGENT if m.len() == 1 => {
                let c = self.conns.get_mut(&id).unwrap();
                c.is_agent = true;
                c.ack_tick = -1;
                return;
            }
            OP_STEP if m.len() == 5 => {
                self.conns.get_mut(&id).unwrap().ack_tick = u32::from_le_bytes([m[1], m[2], m[3], m[4]]) as i64;
                return;
            }
            OP_FOLLOW => {
                self.w.players[pi].follow_prefix = Some(String::from_utf8_lossy(&m[1..]).into_owned());
                return;
            }
            OP_FOOD if m.len() == 3 => {
                let pct = u16::from_le_bytes([m[1], m[2]]);
                self.w.cfg.set_food_mult(self.base_food, pct);
                println!("[arena] food target now {}", self.w.cfg.food_min_amount);
                return;
            }
            OP_SPAWN_SIZE if m.len() == 3 => {
                self.w.players[pi].spawn_size = u16::from_le_bytes([m[1], m[2]]) as f64;
                return;
            }
            OP_GROW if m.len() == 1 && !is_agent => {
                let w = &mut self.w;
                let Some(&big) = w.players[pi].cells.iter().max_by(|&&a, &&b| w.ents[a as usize].size.partial_cmp(&w.ents[b as usize].size).unwrap().then(std::cmp::Ordering::Less)) else { return };
                let s = w.ents[big as usize].size;
                w.set_size(big, (s * s + GROW_MASS * 100.0).sqrt());
                w.players[pi].boosted = true;
                return;
            }
            _ => {}
        }
        match stage {
            0 => {
                if op == 254 && m.len() == 5 {
                    let p = u32::from_le_bytes([m[1], m[2], m[3], m[4]]);
                    if !(1..=17).contains(&p) {
                        self.conns.remove(&id);
                        return;
                    }
                    let c = self.conns.get_mut(&id).unwrap();
                    c.protocol = p;
                    c.stage = 1;
                }
            }
            1 => {
                if op == 255 && m.len() == 5 {
                    let key = u32::from_le_bytes([m[1], m[2], m[3], m[4]]);
                    if protocol > 6 && key != 0 {
                        self.conns.remove(&id);
                        return;
                    }
                    let c = self.conns.get_mut(&id).unwrap();
                    c.stage = 2;
                    let tx = c.tx.clone();
                    let _ = tx.send(clear_all());
                    let _ = tx.send(set_border_hello(self.w.border, SERVER_NAME));
                    let red = [255, 0, 0];
                    for t in ["MultiOgar 2.0.0", "Welcome to my MultiOgar-Edited server!", "The chat is disabled."] {
                        let _ = tx.send(chat("SERVER", red, t, true));
                    }
                }
            }
            _ => self.playing_message(id, op, m),
        }
    }

    fn playing_message(&mut self, id: usize, op: u8, m: &[u8]) {
        let c = self.conns.get(&id).unwrap();
        let (pi, protocol) = (c.player, c.protocol);
        let p = &mut self.w.players[pi];
        match op {
            0 => {
                if !p.cells.is_empty() {
                    return;
                }
                let text = if protocol > 6 {
                    let u: Vec<u16> = m[1..].chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).take_while(|&x| x != 0).collect();
                    String::from_utf16_lossy(&u)
                } else {
                    let b = &m[1..];
                    let end = b.iter().position(|&x| x == 0).unwrap_or(b.len());
                    String::from_utf8_lossy(&b[..end]).into_owned()
                };
                self.w.join_game_text(pi, &text);
            }
            1 => {
                if m.len() == 1 && p.cells.is_empty() {
                    p.spectating = true;
                }
            }
            16 => {
                let rd_i32 = |o: usize| i32::from_le_bytes([m[o], m[o + 1], m[o + 2], m[o + 3]]) as f64;
                match m.len() {
                    13 => p.mouse_in = Some((rd_i32(1), rd_i32(5))),
                    9 => p.mouse_in = Some((i16::from_le_bytes([m[1], m[2]]) as f64, i16::from_le_bytes([m[3], m[4]]) as f64)),
                    21 => {
                        let f = |o: usize| f64::from_le_bytes(m[o..o + 8].try_into().unwrap());
                        let (x, y) = (f(1), f(9));
                        if !x.is_nan() && !y.is_nan() {
                            p.mouse_in = Some((x, y));
                        }
                    }
                    _ => {}
                }
            }
            17 if m.len() == 1 => p.p_split = true,
            18 if m.len() == 1 => p.p_q = true,
            21 if m.len() == 1 => p.p_eject = true,
            254 if m.len() == 1 => {
                let (mut total, mut alive, mut dead, mut spect, mut bots) = (0, 0, 0, 0, 0);
                for q in &self.w.players {
                    if q.removed {
                        continue;
                    }
                    if q.kind == PKind::Bot || q.kind == PKind::Agent {
                        bots += 1;
                        continue;
                    }
                    if !q.connected {
                        continue;
                    }
                    total += 1;
                    if !q.cells.is_empty() {
                        alive += 1;
                    } else {
                        dead += 1;
                        if q.spectating {
                            spect += 1;
                        }
                    }
                }
                let j = format!(
                    "{{\"name\":\"My Server\",\"mode\":\"Free For All\",\"uptime\":{},\"update\":\"0.000\",\"playersTotal\":{total},\"playersAlive\":{alive},\"playersDead\":{dead},\"playersSpect\":{spect},\"botsTotal\":{bots},\"playersLimit\":100}}",
                    self.w.tick / 25
                );
                self.send(id, server_stat(&j));
            }
            _ => {}
        }
    }

    /// Per-tick packets for every connected client (PlayerTracker.sendUpdate).
    fn send_updates(&mut self) {
        let ids: Vec<usize> = self.conns.iter().filter(|(_, c)| c.stage == 2).map(|(&i, _)| i).collect();
        for id in ids {
            let (pi, tx) = {
                let c = &self.conns[&id];
                (c.player, c.tx.clone())
            };
            if self.w.players[pi].removed {
                continue;
            }
            if self.w.players[pi].rejoined {
                let _ = tx.send(clear_all());
                let _ = tx.send(set_border(self.w.border));
                self.w.players[pi].rejoined = false;
            }
            for cid in std::mem::take(&mut self.w.players[pi].new_cells) {
                let _ = tx.send(add_node(cid));
            }
            let p = &self.w.players[pi];
            if p.spectating && p.cells.is_empty() {
                let _ = tx.send(update_pos(p.center.0, p.center.1, p.scale));
            }
            let pkt = diff_update(&self.w, pi);
            let _ = tx.send(pkt);
            let ids: Vec<u32> = self.w.players[pi].view_nodes.iter().map(|&h| self.w.ents[h as usize].id).collect();
            let p = &mut self.w.players[pi];
            p.client_nodes = ids;
            p.tick_lb += 1;
            if p.tick_lb > self.w.cfg.lb_update {
                p.tick_lb = 0;
                let _ = tx.send(leaderboard(&self.w, pi));
            }
        }
    }

    fn drain_events(&mut self) {
        while let Ok(ev) = self.ev_rx.try_recv() {
            self.on_event(ev);
        }
    }

    fn lockstep(&mut self) {
        let agents: Vec<usize> = self.conns.iter().filter(|(_, c)| c.is_agent).map(|(&i, _)| i).collect();
        if agents.is_empty() {
            return;
        }
        let tick = self.w.tick;
        for &id in &agents {
            let pi = self.conns[&id].player;
            let p = &self.w.players[pi];
            let pkt = step(tick, p.shots, p.pops, p.pop_mass);
            self.send(id, pkt);
        }
        let deadline = Instant::now() + ACK_TIMEOUT;
        loop {
            let done = self.conns.values().filter(|c| c.is_agent).all(|c| c.ack_tick >= tick as i64);
            if done {
                break;
            }
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            match self.ev_rx.recv_timeout(deadline - now) {
                Ok(ev) => self.on_event(ev),
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
    }

    fn log_players(&self) {
        let mut parts = Vec::new();
        for p in &self.w.players {
            if p.removed || p.cells.is_empty() {
                continue;
            }
            let mut row = serde_json::json!({"n": p.name, "m": p.total_mass(&self.w).round() as i64, "c": p.cells.len()});
            if p.boosted {
                row["b"] = serde_json::json!(1);
            }
            parts.push(row);
        }
        println!("[players] {}", serde_json::json!({"t": self.w.tick, "p": parts}));
    }

    fn run(&mut self) {
        let period = if self.o.max_tps > 0.0 { Some(Duration::from_secs_f64(1.0 / self.o.max_tps)) } else { None };
        let mut next = Instant::now();
        let (mut stat_at, mut stat_ticks) = (Instant::now(), 0u32);
        let log_every = (self.o.log_players * 25.0).round() as u32;
        while !self.quit {
            self.drain_events();
            self.w.tick();
            stat_ticks += 1;
            self.send_updates();
            if self.w.tick as u64 % self.o.sync == 0 {
                for d in self.drivers.iter_mut() {
                    d.step(&mut self.w);
                }
                self.w.kill_log.clear();
                self.lockstep();
            }
            if log_every > 0 && self.w.tick % log_every == 0 {
                self.log_players();
            }
            if stat_at.elapsed() >= Duration::from_secs(5) {
                let tps = stat_ticks as f64 / stat_at.elapsed().as_secs_f64();
                println!("[arena] tick={} tps={:.0} ({:.1}x realtime) agents={} clients={}", self.w.tick, tps, tps / 25.0, self.drivers.len(), self.conns.len());
                stat_at = Instant::now();
                stat_ticks = 0;
            }
            if let Some(p) = period {
                next += p;
                let now = Instant::now();
                if next > now {
                    std::thread::sleep(next - now);
                } else if now - next > p {
                    next = now - p; // behind: catch up at most one tick
                }
            }
        }
    }
}

#[allow(dead_code)]
fn _unused(_: &Value) {}
