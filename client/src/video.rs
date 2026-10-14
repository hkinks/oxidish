//! `--video FILE`: pipes the rendered frames at a constant 25 fps into ffmpeg (for README / post clips).

use std::io::Write;
use std::process::{Child, Command, Stdio};

use macroquad::prelude::get_screen_data;

const FPS: f64 = 25.0;

pub struct Video {
    ffmpeg: Child,
    size: (u16, u16),
    pub t0: f64,
    next: f64,
}

impl Video {
    pub fn start(path: &str, now: f64) -> Video {
        let img = get_screen_data();
        let size = (img.width, img.height);
        let ffmpeg = Command::new("ffmpeg")
            .args(["-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s"])
            .arg(format!("{}x{}", size.0, size.1))
            .args(["-r", "25", "-i", "-", "-vf", "vflip", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", "-pix_fmt", "yuv420p", path])
            .stdin(Stdio::piped())
            .spawn()
            .expect("--video needs ffmpeg on PATH");
        Video { ffmpeg, size, t0: now, next: now }
    }

    /// Writes a frame when one is due; returns whether it did.
    pub fn frame(&mut self, now: f64) -> bool {
        if now < self.next {
            return false;
        }
        self.next += 1.0 / FPS;
        let img = get_screen_data();
        if (img.width, img.height) != self.size {
            return false; // the window was resized: skip rather than corrupt the stream
        }
        self.ffmpeg.stdin.as_mut().unwrap().write_all(&img.bytes).expect("ffmpeg stopped");
        true
    }

    pub fn finish(mut self) {
        drop(self.ffmpeg.stdin.take());
        let _ = self.ffmpeg.wait();
    }
}
