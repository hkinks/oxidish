//! Native agar.io client for the local MultiOgar arena server (protocol 6, same wire format as
//! vendor/agarian-client). See docs/showcase.md for keys.

mod auto;
mod menu;
mod net;
mod settings;
mod world;

use macroquad::prelude::*;
use net::{Ev, Net};
use settings::Settings;
use world::{Cell, Kind, World, FADE_SECS};

const MOUSE_SEND_SECS: f64 = 0.025;
const REPEAT_SECS: f64 = 0.04;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Menu,
    Playing,
    Dead,
    Spectating,
}

struct Args {
    server: Option<String>,
    name: Option<String>,
    spectate: bool,
    whole_map: bool,
    settings_open: bool,
    zoom: Option<f64>,
    selftest: bool,
    screenshot: Option<String>,
    shot_at: f64,
}

fn parse_args() -> Args {
    let mut a = Args { server: None, name: None, spectate: false, whole_map: false, settings_open: false, zoom: None, selftest: false, screenshot: None, shot_at: 9.0 };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        match k.as_str() {
            "--server" => a.server = it.next(),
            "--name" => a.name = it.next(),
            "--selftest" => a.selftest = true,
            "--novsync" => {}
            "--config" => {
                it.next();
            }
            "--spectate" => a.spectate = true,
            "--whole-map" => {
                a.spectate = true;
                a.whole_map = true;
            }
            "--settings-open" => a.settings_open = true,
            "--zoom" => a.zoom = it.next().and_then(|v| v.parse().ok()),
            "--screenshot" => a.screenshot = it.next(),
            "--shot-at" => a.shot_at = it.next().and_then(|v| v.parse().ok()).unwrap_or(9.0),
            "-h" | "--help" => {
                println!("game_client [--server ws://127.0.0.1:9600 (Rust server; Node showcase: 9500)] [--name NICK] [--selftest] [--screenshot out.png [--shot-at SECS]]");
                std::process::exit(0);
            }
            other => eprintln!("ignoring unknown argument {other}"),
        }
    }
    a
}

fn window_conf() -> Conf {
    Conf {
        window_title: "agario-bot native client".to_owned(),
        window_width: 1280,
        window_height: 720,
        window_resizable: true,
        sample_count: 4,
        platform: macroquad::miniquad::conf::Platform {
            // vsync is on by default (caps FPS at the monitor refresh); --novsync uncaps it
            swap_interval: if std::env::args().any(|a| a == "--novsync") || !Settings::load().vsync { Some(0) } else { None },
            ..Default::default()
        },
        ..Default::default()
    }
}

struct Camera {
    x: f64,
    y: f64,
    z: f64,
    target_z: f64,
    /// manual zoom (mouse wheel) overrides auto zoom until re-enabled
    manual: bool,
    manual_z: f64,
    fit: f64,
}

impl Camera {
    fn view_mult() -> f64 {
        (screen_height() as f64 / 1080.0).min(screen_width() as f64 / 1920.0).sqrt()
    }
    fn to_screen(&self, x: f64, y: f64) -> (f32, f32) {
        (((x - self.x) * self.z) as f32 + screen_width() / 2.0, ((y - self.y) * self.z) as f32 + screen_height() / 2.0)
    }
}

struct Fonts {
    font: Option<Font>,
}

impl Fonts {
    fn load() -> Fonts {
        for p in ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] {
            if let Ok(bytes) = std::fs::read(p) {
                if let Ok(f) = load_ttf_font_from_bytes(&bytes) {
                    return Fonts { font: Some(f) };
                }
            }
        }
        Fonts { font: None }
    }

    fn size(&self, text: &str, px: f32) -> TextDimensions {
        measure_text(text, self.font.as_ref(), px as u16, 1.0)
    }

    fn draw(&self, text: &str, x: f32, y: f32, px: f32, color: Color) {
        draw_text_ex(text, x, y, TextParams { font: self.font.as_ref(), font_size: px as u16, color, ..Default::default() });
    }

    /// Centered at (cx, cy) with a dark outline.
    fn outlined(&self, text: &str, cx: f32, cy: f32, px: f32, color: Color, alpha: f32) {
        let px = ((px / 4.0).round() * 4.0).clamp(8.0, 120.0);
        let d = self.size(text, px);
        let x = cx - d.width / 2.0;
        let y = cy + d.height / 2.0 - d.offset_y / 2.0 + 0.0;
        let o = (px / 14.0).max(1.0);
        let black = Color::new(0.0, 0.0, 0.0, 0.85 * alpha);
        for (dx, dy) in [(-o, 0.0), (o, 0.0), (0.0, -o), (0.0, o)] {
            self.draw(text, x + dx, y + dy, px, black);
        }
        self.draw(text, x, y, px, Color { a: color.a * alpha, ..color });
    }
}

fn rgb(c: [u8; 3], a: f32) -> Color {
    Color::new(c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0, a)
}

fn darker(c: [u8; 3], a: f32) -> Color {
    Color::new(c[0] as f32 / 255.0 * 0.85, c[1] as f32 / 255.0 * 0.85, c[2] as f32 / 255.0 * 0.85, a)
}

fn sides_for(r_px: f32) -> u8 {
    (r_px.sqrt() * 4.0).clamp(10.0, 120.0) as u8
}

fn draw_virus(x: f32, y: f32, r: f32, alpha: f32) {
    let spikes = ((r * 0.5) as i32).clamp(14, 64);
    let n = spikes * 2;
    let tip = Color::new(0.15, 0.75, 0.15, alpha);
    let pts: Vec<Vec2> = (0..n)
        .map(|i| {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            let rr = if i % 2 == 0 { r } else { r * 0.9 };
            vec2(x + a.cos() * rr, y + a.sin() * rr)
        })
        .collect();
    for i in 0..n as usize {
        draw_triangle(vec2(x, y), pts[i], pts[(i + 1) % n as usize], tip);
    }
    draw_poly(x, y, sides_for(r), r * 0.88, 0.0, Color::new(0.2, 1.0, 0.2, alpha * 0.9));
}

struct Game {
    world: World,
    net: Option<Net>,
    url: String,
    reconnect_at: f64,
    status: String,
    mode: Mode,
    had_cells: bool,
    nick: String,
    cam: Camera,
    border_seen: bool,
    last_mouse: f64,
    last_repeat: f64,
    last_score: i64,
    fonts: Fonts,
    whole_map: bool,
    following: Option<String>,
    set: Settings,
    saved: Settings,
    settings_open: bool,
    focus: Option<usize>,
    quit: bool,
}

impl Game {
    fn send(&self, p: Vec<u8>) {
        if let Some(n) = &self.net {
            let _ = n.out.send(p);
        }
    }

    fn spawn(&mut self) {
        let mut p = vec![0u8];
        p.extend_from_slice(self.nick.as_bytes());
        p.push(0);
        self.send(p);
        self.mode = Mode::Playing;
        self.had_cells = false;
    }

    fn spectate(&mut self) {
        self.send(vec![1]);
        self.mode = Mode::Spectating;
        self.had_cells = false;
        self.whole_map = false;   // the server's free roam is off after a spawn
        self.following = None;
        let mut auto = vec![0xCA];  // follow the largest player (arena extension: name prefix, "" = auto)
        auto.extend_from_slice(b"");
        self.send(auto);
    }

    /// Tab while spectating: follow the next player on the leaderboard (wraps around; arena extension 0xCA).
    fn follow_next(&mut self) {
        let names: Vec<String> = self.world.leaderboard.iter().map(|l| l.name.clone()).filter(|n| !n.is_empty()).collect();
        if names.is_empty() {
            return;
        }
        if self.whole_map {        // free roam would override following: switch it off first
            self.send(vec![18]);
            self.send(vec![19]);
            self.whole_map = false;
        }
        let next = match &self.following {
            Some(cur) => names.iter().position(|n| n == cur).map_or(0, |i| (i + 1) % names.len()),
            None => 1 % names.len(),   // auto already shows the largest (leaderboard #1)
        };
        let name = names[next].clone();
        let mut p = vec![0xCA];
        p.extend_from_slice(name.as_bytes());
        self.send(p);
        self.following = Some(name);
    }

    fn pump_net(&mut self, now: f64) {
        let mut closed = None;
        if let Some(n) = &self.net {
            for ev in n.rx.try_iter() {
                match ev {
                    Ev::Msg(b) => {
                        self.world.handle(&b, now);
                    }
                    Ev::Closed(why) => {
                        closed = Some(why);
                        break;
                    }
                }
            }
        }
        if let Some(why) = closed {
            self.status = format!("disconnected ({why}), retrying...");
            self.net = None;
            self.world.reset();
            self.border_seen = false;
            self.mode = Mode::Menu;
            self.reconnect_at = now + 1.5;
        }
        if self.net.is_none() && now >= self.reconnect_at {
            self.net = Some(net::connect(&self.url));
            self.status = format!("connecting to {} ...", self.url);
        }
        if self.world.border.is_some() && !self.border_seen {
            self.border_seen = true;
            let b = self.world.border.unwrap();
            self.cam.x = (b.left + b.right) / 2.0;
            self.cam.y = (b.top + b.bottom) / 2.0;
            self.cam.z = 1.0;
            self.cam.target_z = 1.0;
            self.status = format!("connected to {}", self.url);
        }
    }

    fn own_cells(&self) -> Vec<&Cell> {
        self.world.mine.iter().filter_map(|id| self.world.cells.get(id)).filter(|c| c.dead_at.is_none()).collect()
    }

    fn update_camera(&mut self, dt: f64) {
        let a_pos = 1.0 - 0.5f64.powf(dt * 60.0);
        let a_far = 1.0 - (19.0f64 / 20.0).powf(dt * 60.0);
        let a_zoom = 1.0 - (8.0f64 / 9.0).powf(dt * 60.0);
        let own = self.own_cells();
        let (bw, bh) = self.map_size();
        let fit = (screen_width() as f64 / bw).min(screen_height() as f64 / bh) * 0.96;
        let mut score = 0.0;
        if !own.is_empty() {
            let n = own.len() as f64;
            let (mut x, mut y, mut s) = (0.0, 0.0, 0.0);
            for c in &own {
                x += c.x;
                y += c.y;
                s += c.s;
                score += (c.ns * c.ns / 100.0).floor();
            }
            self.cam.target_z = (64.0 / s).min(1.0).powf(0.4);
            self.cam.x += (x / n - self.cam.x) * a_pos;
            self.cam.y += (y / n - self.cam.y) * a_pos;
            self.last_score = score as i64;
        } else if let (Some((tx, ty, tz)), Mode::Spectating) = (self.world.spec_target, self.mode) {
            self.cam.x += (tx - self.cam.x) * a_far;
            self.cam.y += (ty - self.cam.y) * a_far;
            self.cam.target_z = tz;
            if self.whole_map {
                // free roam over the whole map: centre the map and fit it to the window at any aspect
                if let Some(b) = self.world.border {
                    self.cam.x += ((b.left + b.right) / 2.0 - self.cam.x) * a_far;
                    self.cam.y += ((b.top + b.bottom) / 2.0 - self.cam.y) * a_far;
                }
                self.cam.target_z = fit / Camera::view_mult();
            }
        }
        let mut goal = self.cam.target_z * Camera::view_mult();
        if self.cam.manual || !self.set.auto_zoom {
            goal = self.cam.manual_z;
        }
        self.cam.fit = fit;
        self.cam.z += (goal.clamp(fit * 0.9, 5.0) - self.cam.z) * a_zoom;
    }

    fn map_size(&self) -> (f64, f64) {
        self.world.border.map_or((7000.0, 7000.0), |b| ((b.right - b.left).max(1.0), (b.bottom - b.top).max(1.0)))
    }

    fn mouse_world(&self) -> (f64, f64) {
        let (mx, my) = mouse_position();
        (
            (mx - screen_width() / 2.0) as f64 / self.cam.z + self.cam.x,
            (my - screen_height() / 2.0) as f64 / self.cam.z + self.cam.y,
        )
    }

    fn send_mouse(&self, wx: f64, wy: f64) {
        let mut p = vec![0x10u8];
        p.extend_from_slice(&(wx as i32).to_le_bytes());
        p.extend_from_slice(&(wy as i32).to_le_bytes());
        p.extend_from_slice(&[0, 0, 0, 0]);
        self.send(p);
    }

    fn input(&mut self, now: f64, aim: Option<(f64, f64)>) {
        if self.settings_open {
            self.settings_input();
            return;
        }
        let (_, wheel) = mouse_wheel();
        if wheel != 0.0 {
            if !self.cam.manual {
                self.cam.manual = true;
                self.cam.manual_z = self.cam.z;
            }
            self.cam.manual_z = (self.cam.manual_z * if wheel > 0.0 { 1.1 } else { 0.9 }).clamp(self.cam.fit * 0.9, 5.0);
        }
        if is_key_pressed(KeyCode::Z) && self.mode != Mode::Menu {
            self.cam.manual = false;
            self.set.auto_zoom = true;
        }
        if is_key_pressed(KeyCode::F1) {
            self.settings_open = true;
        }
        if is_key_pressed(KeyCode::F11) {
            set_fullscreen(true);
        }
        match self.mode {
            Mode::Menu | Mode::Dead | Mode::Spectating => {
                if self.mode == Mode::Menu {
                    while let Some(ch) = get_char_pressed() {
                        if !ch.is_control() && self.nick.chars().count() < 15 {
                            self.nick.push(ch);
                        }
                    }
                    if is_key_pressed(KeyCode::Backspace) {
                        self.nick.pop();
                    }
                } else {
                    while get_char_pressed().is_some() {}
                }
                if self.net.is_some() && self.world.border.is_some() {
                    // a click starts playing from the menu / death screen, but not while spectating (only Enter does)
                    let click = is_mouse_button_pressed(MouseButton::Left) && self.mode != Mode::Spectating;
                    if is_key_pressed(KeyCode::Enter) || click {
                        self.spawn();
                    } else if is_key_pressed(KeyCode::Tab) && self.mode != Mode::Spectating {
                        self.spectate();
                    } else if is_key_pressed(KeyCode::Tab) {
                        self.follow_next();
                    } else if is_key_pressed(KeyCode::Q) && self.mode == Mode::Spectating {
                        // the server's free roam: the whole map (arena.js), or back to following the largest
                        self.send(vec![18]);
                        self.send(vec![19]);
                        self.whole_map = !self.whole_map;
                    }
                }
            }
            Mode::Playing => {
                while get_char_pressed().is_some() {}
                let (wx, wy) = aim.unwrap_or_else(|| self.mouse_world());
                if now - self.last_mouse >= MOUSE_SEND_SECS {
                    self.last_mouse = now;
                    self.send_mouse(wx, wy);
                }
                if is_key_pressed(KeyCode::Space) {
                    self.send(vec![17]);
                }
                if is_key_pressed(KeyCode::Equal) || is_key_pressed(KeyCode::KpAdd) {
                    self.send(vec![205]);
                }
                let repeat = now - self.last_repeat >= REPEAT_SECS;
                if is_key_pressed(KeyCode::W) || (is_key_down(KeyCode::W) && repeat) {
                    self.send(vec![21]);
                    self.last_repeat = now;
                }
            }
        }
    }

    fn track_mode(&mut self, now: f64) {
        let alive = !self.own_cells().is_empty();
        if alive {
            self.had_cells = true;
        }
        if self.mode == Mode::Playing && self.had_cells && self.world.mine.is_empty() {
            self.mode = Mode::Dead;
            let _ = now;
        }
    }

    fn draw_grid(&self) {
        let (w, h) = (screen_width() as f64, screen_height() as f64);
        let z = self.cam.z;
        let mut step = 50.0;
        while step * z < 14.0 {
            step *= 2.0;
        }
        let left = self.cam.x - w / 2.0 / z;
        let right = self.cam.x + w / 2.0 / z;
        let top = self.cam.y - h / 2.0 / z;
        let bottom = self.cam.y + h / 2.0 / z;
        let col = if self.set.dark { Color::new(1.0, 1.0, 1.0, 0.06) } else { Color::new(0.0, 0.0, 0.0, 0.07) };
        let mut x = (left / step).floor() * step;
        while x <= right {
            let sx = ((x - self.cam.x) * z + w / 2.0) as f32;
            draw_line(sx, 0.0, sx, h as f32, 1.0, col);
            x += step;
        }
        let mut y = (top / step).floor() * step;
        while y <= bottom {
            let sy = ((y - self.cam.y) * z + h / 2.0) as f32;
            draw_line(0.0, sy, w as f32, sy, 1.0, col);
            y += step;
        }
    }

    fn draw_world(&self, now: f64) {
        let dark = self.set.dark;
        clear_background(if dark { Color::new(0.02, 0.02, 0.03, 1.0) } else { Color::new(0.55, 0.57, 0.6, 1.0) });
        if let Some(b) = self.world.border {
            let (x0, y0) = self.cam.to_screen(b.left, b.top);
            let (x1, y1) = self.cam.to_screen(b.right, b.bottom);
            let bg = if dark { Color::new(0.09, 0.10, 0.13, 1.0) } else { Color::new(0.95, 0.98, 1.0, 1.0) };
            draw_rectangle(x0, y0, x1 - x0, y1 - y0, bg);
            if self.set.show_grid {
                self.draw_grid();
            }
            draw_rectangle_lines(x0, y0, x1 - x0, y1 - y0, 6.0, Color::new(0.8, 0.2, 0.2, 0.8));
        }
        let (w, h) = (screen_width(), screen_height());
        let z = self.cam.z as f32;
        let mut sorted: Vec<(&Cell, f32, f32, f32, f32)> = Vec::with_capacity(256);
        // food and ejected mass first (unsorted), then players and viruses by size
        for c in self.world.cells.values() {
            let alpha = c.dead_at.map_or(1.0, |d| (1.0 - ((now - d) / FADE_SECS) as f32).clamp(0.0, 1.0));
            let (sx, sy) = self.cam.to_screen(c.x, c.y);
            let r = c.s as f32 * z;
            if sx + r < 0.0 || sx - r > w || sy + r < 0.0 || sy - r > h {
                continue;
            }
            match c.kind {
                Kind::Food => draw_poly(sx, sy, if r > 6.0 { 14 } else { 8 }, r, 0.0, rgb(c.color, alpha)),
                Kind::Ejected => {
                    draw_poly(sx, sy, sides_for(r), r, 0.0, darker(c.color, alpha));
                    draw_poly(sx, sy, sides_for(r), r * 0.85, 0.0, rgb(c.color, alpha));
                }
                _ => sorted.push((c, sx, sy, r, alpha)),
            }
        }
        sorted.sort_by(|a, b| a.0.s.total_cmp(&b.0.s).then(a.0.id.cmp(&b.0.id)));
        for &(c, sx, sy, r, alpha) in &sorted {
            if c.kind == Kind::Virus {
                draw_virus(sx, sy, r, alpha);
                continue;
            }
            let sides = sides_for(r);
            if self.set.outline {
                draw_poly(sx, sy, sides, r, 0.0, darker(c.color, alpha));
                let ring = (r * 0.04).clamp(1.5, 12.0).min(r * 0.5);
                draw_poly(sx, sy, sides, r - ring, 0.0, rgb(c.color, alpha));
            } else {
                draw_poly(sx, sy, sides, r, 0.0, rgb(c.color, alpha));
            }
        }
        for &(c, sx, sy, r, alpha) in &sorted {
            if c.kind != Kind::Player || r < 16.0 {
                continue;
            }
            let white = Color::new(1.0, 1.0, 1.0, 1.0);
            let px = ((0.3 * c.s).max(24.0) as f32 * z).max(10.0);
            let mine = self.world.mine.contains(&c.id);
            if !c.name.is_empty() && self.set.show_names {
                self.fonts.outlined(&c.name, sx, sy, px, white, alpha);
            }
            if self.set.show_mass && (r >= 26.0 || mine) {
                let off = if c.name.is_empty() || !self.set.show_names { 0.0 } else { px * 0.75 };
                self.fonts.outlined(&format!("{}", c.mass() as i64), sx, sy + off, px * 0.6, white, alpha);
            }
        }
    }

    fn draw_hud(&self, now: f64) {
        let (w, h) = (screen_width(), screen_height());
        let f = &self.fonts;
        let fg = if self.set.dark { Color::new(0.9, 0.9, 0.95, 0.9) } else { Color::new(0.1, 0.1, 0.12, 0.9) };
        if self.set.show_fps {
            f.draw(&format!("FPS {}   cells {}   upd {}   zoom {:.2}{}   {}", get_fps(), self.world.cells.len(), self.world.updates, self.cam.z,
                if self.cam.manual || !self.set.auto_zoom { " (manual, Z: auto)" } else { "" }, self.status), 10.0, 20.0, 18.0, fg);
        }
        f.draw("F1: settings", 10.0, 42.0, 18.0, fg);
        if self.mode == Mode::Playing || self.mode == Mode::Dead {
            let s = format!("Score: {}", self.last_score);
            f.outlined(&s, 16.0 + f.size(&s, 36.0).width / 2.0, h - 34.0, 36.0, Color::new(1.0, 1.0, 1.0, 1.0), 1.0);
        }
        // leaderboard
        let pw = 230.0;
        draw_rectangle(w - pw - 10.0, 10.0, pw, 44.0 + 26.0 * self.world.leaderboard.len().min(10) as f32, Color::new(0.0, 0.0, 0.0, 0.45));
        f.draw("Leaderboard", w - pw / 2.0 - 10.0 - f.size("Leaderboard", 28.0).width / 2.0, 40.0, 28.0, Color::new(1.0, 1.0, 1.0, 1.0));
        for (i, it) in self.world.leaderboard.iter().take(10).enumerate() {
            let col = if it.me { Color::new(1.0, 0.65, 0.65, 1.0) } else { Color::new(1.0, 1.0, 1.0, 1.0) };
            let name: String = it.name.chars().take(20).collect();
            f.draw(&format!("{}. {}", i + 1, name), w - pw - 2.0, 66.0 + 26.0 * i as f32, 22.0, col);
        }
        let _ = now;
    }

    fn draw_overlay(&self, now: f64) {
        let (w, h) = (screen_width(), screen_height());
        if self.settings_open {
            self.draw_settings(now);
            return;
        }
        if self.mode == Mode::Playing || self.mode == Mode::Spectating {
            if self.mode == Mode::Spectating {
                let who = self.following.clone().unwrap_or_else(|| "the largest player".into());
                let what = if self.whole_map { "Spectating the whole map.  Q: follow players, Tab: next player".to_string() } else { format!("Spectating {who}.  Tab: next player, Q: whole map") };
                self.fonts.outlined(&format!("{what}.  Enter: play"), w / 2.0, h - 30.0, 24.0, Color::new(1.0, 1.0, 1.0, 1.0), 1.0);
            }
            return;
        }
        draw_rectangle(0.0, 0.0, w, h, Color::new(0.0, 0.0, 0.0, 0.55));
        let white = Color::new(1.0, 1.0, 1.0, 1.0);
        let dim = Color::new(0.8, 0.85, 0.9, 1.0);
        let f = &self.fonts;
        let ready = self.net.is_some() && self.world.border.is_some();
        if self.mode == Mode::Dead {
            f.outlined("You died", w / 2.0, h / 2.0 - 150.0, 56.0, Color::new(1.0, 0.4, 0.4, 1.0), 1.0);
            f.outlined(&format!("Score: {}", self.last_score), w / 2.0, h / 2.0 - 100.0, 32.0, white, 1.0);
        } else {
            f.outlined("agario-bot", w / 2.0, h / 2.0 - 150.0, 56.0, white, 1.0);
        }
        // nickname box
        let (bw, bh) = (340.0, 44.0);
        let (bx, by) = (w / 2.0 - bw / 2.0, h / 2.0 - 60.0);
        draw_rectangle(bx, by, bw, bh, Color::new(1.0, 1.0, 1.0, 0.95));
        let caret = if self.mode == Mode::Menu && (now * 2.0) as i64 % 2 == 0 { "|" } else { "" };
        let shown = if self.nick.is_empty() && self.mode == Mode::Menu { "Nick".to_string() } else { self.nick.clone() };
        let col = if self.nick.is_empty() { Color::new(0.5, 0.5, 0.5, 1.0) } else { Color::new(0.1, 0.1, 0.1, 1.0) };
        f.draw(&format!("{shown}{caret}"), bx + 12.0, by + 31.0, 28.0, col);
        let go = if ready { "Enter / click: play      Tab: spectate" } else { "waiting for server..." };
        f.outlined(go, w / 2.0, h / 2.0 + 20.0, 28.0, white, 1.0);
        f.outlined(&self.status, w / 2.0, h / 2.0 + 56.0, 20.0, dim, 1.0);
        let help = [
            "Mouse: steer    Space: split    W: eject (hold)    Wheel: zoom",
            "+ : add 500 mass    Tab: spectate (Q: whole map)    Esc / F1: settings",
        ];
        for (i, l) in help.iter().enumerate() {
            f.outlined(l, w / 2.0, h / 2.0 + 110.0 + 28.0 * i as f32, 20.0, dim, 1.0);
        }
    }
}

#[cfg(windows)]
extern "system" {
    fn GetCurrentProcess() -> isize;
    fn SetPriorityClass(process: isize, class: u32) -> i32;
}

/// Above-normal priority: with background jobs keeping the CPU at 80-90%, a normal-priority client got
/// preempted and missed vsync frames (choppy). Rendering itself costs well under a millisecond a frame.
fn raise_priority() {
    #[cfg(windows)]
    unsafe {
        SetPriorityClass(GetCurrentProcess(), 0x0000_8000); // ABOVE_NORMAL_PRIORITY_CLASS
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    raise_priority();
    let args = parse_args();
    let mut set = Settings::load();
    if let Some(s) = &args.server {
        set.server = s.clone();
    }
    if let Some(n) = &args.name {
        set.nick = n.clone();
    }
    let mut game = Game {
        world: {
            let mut w = World::new();
            w.lerp = set.lerp_secs();
            w
        },
        net: None,
        url: set.server.clone(),
        reconnect_at: 0.0,
        status: String::new(),
        mode: Mode::Menu,
        had_cells: false,
        nick: set.nick.clone(),
        cam: Camera { x: 0.0, y: 0.0, z: 1.0, target_z: 1.0, manual: false, manual_z: 1.0, fit: 0.1 },
        border_seen: false,
        last_mouse: 0.0,
        last_repeat: 0.0,
        last_score: 0,
        fonts: Fonts::load(),
        whole_map: false,
        following: None,
        saved: set.clone(),
        set,
        settings_open: args.settings_open,
        focus: None,
        quit: false,
    };
    if let Some(z) = args.zoom {
        game.cam.manual = true;
        game.cam.manual_z = z;
    }
    let mut auto = if args.selftest || args.screenshot.is_some() {
        if game.nick.is_empty() {
            game.nick = "selftest".into();
        }
        Some(auto::Auto::new(args.selftest, args.screenshot.clone(), args.shot_at, args.spectate, args.whole_map))
    } else {
        None
    };
    let vsync_on = game.set.vsync && !std::env::args().any(|a| a == "--novsync");
    let mut shot_pending: Option<String> = None;
    loop {
        let now = get_time();
        let dt = get_frame_time() as f64;
        if is_key_pressed(KeyCode::Escape) {
            game.settings_open = !game.settings_open;
            game.focus = None;
            if !game.settings_open {
                game.apply_settings();
            }
        }
        if game.quit {
            break;
        }
        game.pump_net(now);
        game.world.tick(now);
        game.track_mode(now);
        let mut aim = None;
        if let Some(a) = auto.as_mut() {
            let act = a.step(now, game.mode, &game.world, game.nick.is_empty(), game.net.is_some() && game.world.border.is_some(), get_fps());
            if act.spawn {
                game.spawn();
            }
            if act.spectate {
                game.spectate();
            }
            if act.whole_map {
                game.send(vec![18]);
                game.send(vec![19]);
                game.whole_map = true;
            }
            for p in act.packets {
                game.send(p);
            }
            aim = act.aim;
            if act.screenshot.is_some() {
                shot_pending = act.screenshot;
            }
        }
        game.input(now, aim);
        game.update_camera(dt);
        game.draw_world(now);
        game.draw_hud(now);
        game.draw_overlay(now);
        if let Some(path) = shot_pending.take() {
            get_screen_data().export_png(&path);
            println!("[auto] screenshot saved to {path}");
            if !args.selftest {
                break;
            }
        }
        if auto.as_ref().map_or(false, |a| a.finished) {
            break;
        }
        if !vsync_on && game.set.fps_cap > 0 {
            let frame = 1.0 / game.set.fps_cap as f64;
            while get_time() - now < frame - 0.0015 {
                std::thread::sleep(std::time::Duration::from_micros(300));
            }
            while get_time() - now < frame {
                std::hint::spin_loop();
            }
        }
        next_frame().await;
    }
}
