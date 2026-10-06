//! Settings menu (Esc / F1): clickable rows, text fields for server and nickname.

use macroquad::prelude::*;

use crate::settings::{next_in, FPS_CAPS, SMOOTHING};
use crate::{Game, Mode};

const ROWS: usize = 14;
const ROW_H: f32 = 34.0;

fn onoff(b: bool) -> String {
    if b { "ON".into() } else { "off".into() }
}

impl Game {
    fn row_text(&self, i: usize) -> (&'static str, String) {
        let s = &self.set;
        match i {
            0 => ("Server address", self.set.server.clone()),
            1 => ("Nickname", self.nick.clone()),
            2 => ("Auto zoom (Z)", onoff(s.auto_zoom && !self.cam.manual)),
            3 => ("Show mass", onoff(s.show_mass)),
            4 => ("Show names", onoff(s.show_names)),
            5 => ("Show grid", onoff(s.show_grid)),
            6 => ("Show FPS", onoff(s.show_fps)),
            7 => ("VSync (restart to apply)", onoff(s.vsync)),
            8 => ("FPS cap (vsync off only)", if s.fps_cap == 0 { "unlimited".into() } else { s.fps_cap.to_string() }),
            9 => ("Interpolation smoothing", if s.smoothing == 0.0 { "off".into() } else { format!("{}x", s.smoothing) }),
            10 => ("Cell outline", onoff(s.outline)),
            11 => ("Background", if s.dark { "dark".into() } else { "light".into() }),
            12 => ("Close", String::new()),
            _ => ("Quit", String::new()),
        }
    }

    fn panel_origin(&self) -> (f32, f32) {
        let h = ROWS as f32 * ROW_H + 90.0;
        ((screen_width() - 620.0) / 2.0, ((screen_height() - h) / 2.0).max(10.0))
    }

    /// Apply changed settings to the running client and persist them.
    pub fn apply_settings(&mut self) {
        self.set.nick = self.nick.clone();
        self.world.lerp = self.set.lerp_secs();
        if self.set.server != self.url {
            self.url = self.set.server.clone();
            self.net = None;
            self.world.reset();
            self.border_seen = false;
            self.mode = Mode::Menu;
            self.reconnect_at = 0.0;
            self.status = "reconnecting...".into();
        }
        if self.set != self.saved {
            self.set.save();
            self.saved = self.set.clone();
        }
    }

    fn activate(&mut self, i: usize) {
        self.focus = None;
        let s = &mut self.set;
        match i {
            0 | 1 => self.focus = Some(i),
            2 => {
                let on = s.auto_zoom && !self.cam.manual;
                s.auto_zoom = !on;
                self.cam.manual = false;
            }
            3 => s.show_mass = !s.show_mass,
            4 => s.show_names = !s.show_names,
            5 => s.show_grid = !s.show_grid,
            6 => s.show_fps = !s.show_fps,
            7 => s.vsync = !s.vsync,
            8 => s.fps_cap = next_in(&FPS_CAPS, s.fps_cap),
            9 => s.smoothing = next_in(&SMOOTHING, s.smoothing),
            10 => s.outline = !s.outline,
            11 => s.dark = !s.dark,
            12 => {
                self.settings_open = false;
            }
            _ => {
                self.apply_settings();
                self.quit = true;
            }
        }
        self.apply_settings();
    }

    pub fn settings_input(&mut self) {
        while let Some(ch) = get_char_pressed() {
            if ch.is_control() {
                continue;
            }
            match self.focus {
                Some(0) if self.set.server.len() < 80 => self.set.server.push(ch),
                Some(1) if self.nick.chars().count() < 15 => self.nick.push(ch),
                _ => {}
            }
        }
        if is_key_pressed(KeyCode::Backspace) || is_key_pressed_repeat_backspace() {
            match self.focus {
                Some(0) => {
                    self.set.server.pop();
                }
                Some(1) => {
                    self.nick.pop();
                }
                _ => {}
            }
        }
        if is_key_pressed(KeyCode::Enter) {
            self.focus = None;
            self.apply_settings();
        }
        if is_key_pressed(KeyCode::F1) {
            self.settings_open = false;
            self.apply_settings();
        }
        if is_mouse_button_pressed(MouseButton::Left) {
            let (mx, my) = mouse_position();
            let (ox, oy) = self.panel_origin();
            for i in 0..ROWS {
                let y = oy + 56.0 + i as f32 * ROW_H;
                if mx >= ox && mx <= ox + 620.0 && my >= y && my < y + ROW_H - 2.0 {
                    self.activate(i);
                    return;
                }
            }
            self.focus = None;
        }
    }

    pub fn draw_settings(&self, now: f64) {
        let (w, h) = (screen_width(), screen_height());
        draw_rectangle(0.0, 0.0, w, h, Color::new(0.0, 0.0, 0.0, 0.5));
        let (ox, oy) = self.panel_origin();
        draw_rectangle(ox, oy, 620.0, ROWS as f32 * ROW_H + 90.0, Color::new(0.1, 0.11, 0.15, 0.97));
        let f = &self.fonts;
        let white = Color::new(1.0, 1.0, 1.0, 1.0);
        f.draw("Settings", ox + 16.0, oy + 34.0, 30.0, white);
        f.draw("Esc / F1: close", ox + 440.0, oy + 32.0, 18.0, Color::new(0.7, 0.75, 0.8, 1.0));
        let (mx, my) = mouse_position();
        for i in 0..ROWS {
            let y = oy + 56.0 + i as f32 * ROW_H;
            let hover = mx >= ox && mx <= ox + 620.0 && my >= y && my < y + ROW_H - 2.0;
            if hover || self.focus == Some(i) {
                draw_rectangle(ox + 6.0, y, 608.0, ROW_H - 2.0, Color::new(0.25, 0.3, 0.4, 0.8));
            }
            let (label, val) = self.row_text(i);
            let col = if i == 13 { Color::new(1.0, 0.5, 0.5, 1.0) } else { white };
            f.draw(label, ox + 18.0, y + 24.0, 22.0, col);
            let caret = if self.focus == Some(i) && (now * 2.0) as i64 % 2 == 0 { "|" } else { "" };
            let shown = format!("{val}{caret}");
            let tail: String = if shown.chars().count() > 30 { shown.chars().skip(shown.chars().count() - 30).collect() } else { shown };
            let vc = if val == "ON" { Color::new(0.5, 1.0, 0.6, 1.0) } else { Color::new(0.85, 0.9, 1.0, 1.0) };
            f.draw(&tail, ox + 300.0, y + 24.0, 22.0, vc);
        }
        f.draw(&self.status, ox + 16.0, oy + ROWS as f32 * ROW_H + 82.0, 16.0, Color::new(0.7, 0.75, 0.8, 1.0));
    }
}

fn is_key_pressed_repeat_backspace() -> bool {
    false
}
