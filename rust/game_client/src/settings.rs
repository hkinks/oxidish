//! Persistent client settings (%APPDATA%\agario-client\settings.json, or --config PATH).

use serde_json::{json, Value};
use std::path::PathBuf;

pub const FPS_CAPS: [u32; 6] = [0, 60, 120, 144, 240, 360];
pub const SMOOTHING: [f64; 5] = [0.0, 0.5, 1.0, 2.0, 3.0];

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub server: String,
    pub nick: String,
    pub auto_zoom: bool,
    pub show_mass: bool,
    pub show_names: bool,
    pub show_grid: bool,
    pub show_fps: bool,
    pub vsync: bool,
    /// 0 = unlimited (only used when vsync is off)
    pub fps_cap: u32,
    /// multiplier of the web client's 120 ms interpolation window; 0 = no smoothing
    pub smoothing: f64,
    pub outline: bool,
    pub dark: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            server: "ws://127.0.0.1:9600".into(),
            nick: String::new(),
            auto_zoom: true,
            show_mass: true,
            show_names: true,
            show_grid: true,
            show_fps: true,
            vsync: true,
            fps_cap: 0,
            smoothing: 1.0,
            outline: true,
            dark: false,
        }
    }
}

pub fn config_path() -> PathBuf {
    let mut it = std::env::args();
    while let Some(a) = it.next() {
        if a == "--config" {
            if let Some(p) = it.next() {
                return PathBuf::from(p);
            }
        }
    }
    let base = std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("agario-client").join("settings.json")
}

impl Settings {
    pub fn load() -> Settings {
        Settings::load_from(&config_path())
    }

    pub fn load_from(path: &std::path::Path) -> Settings {
        let mut s = Settings::default();
        let Ok(text) = std::fs::read_to_string(path) else { return s };
        let Ok(v) = serde_json::from_str::<Value>(&text) else { return s };
        let b = |k: &str, d: bool| v.get(k).and_then(Value::as_bool).unwrap_or(d);
        if let Some(x) = v.get("server").and_then(Value::as_str) {
            s.server = x.to_string();
        }
        if let Some(x) = v.get("nick").and_then(Value::as_str) {
            s.nick = x.to_string();
        }
        s.auto_zoom = b("auto_zoom", s.auto_zoom);
        s.show_mass = b("show_mass", s.show_mass);
        s.show_names = b("show_names", s.show_names);
        s.show_grid = b("show_grid", s.show_grid);
        s.show_fps = b("show_fps", s.show_fps);
        s.vsync = b("vsync", s.vsync);
        s.outline = b("outline", s.outline);
        s.dark = b("dark", s.dark);
        if let Some(x) = v.get("fps_cap").and_then(Value::as_u64) {
            s.fps_cap = x as u32;
        }
        if let Some(x) = v.get("smoothing").and_then(Value::as_f64) {
            s.smoothing = x.clamp(0.0, 10.0);
        }
        s
    }

    pub fn save(&self) {
        self.save_to(&config_path());
    }

    pub fn save_to(&self, path: &std::path::Path) {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let v = json!({
            "server": self.server, "nick": self.nick, "auto_zoom": self.auto_zoom,
            "show_mass": self.show_mass, "show_names": self.show_names, "show_grid": self.show_grid,
            "show_fps": self.show_fps, "vsync": self.vsync, "fps_cap": self.fps_cap,
            "smoothing": self.smoothing, "outline": self.outline, "dark": self.dark,
        });
        let _ = std::fs::write(path, serde_json::to_string_pretty(&v).unwrap_or_default());
    }

    pub fn lerp_secs(&self) -> f64 {
        (crate::world::LERP_SECS * self.smoothing).max(0.001)
    }
}

pub fn next_in<T: Copy + PartialEq>(list: &[T], cur: T) -> T {
    let i = list.iter().position(|&x| x == cur).unwrap_or(0);
    list[(i + 1) % list.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let dir = std::env::temp_dir().join("agario-client-test");
        let path = dir.join("settings.json");
        let mut s = Settings::default();
        s.server = "ws://example:1".into();
        s.dark = true;
        s.fps_cap = 144;
        s.smoothing = 2.0;
        s.show_mass = false;
        s.save_to(&path);
        assert_eq!(Settings::load_from(&path), s);
        let _ = std::fs::remove_dir_all(dir);
    }
}
