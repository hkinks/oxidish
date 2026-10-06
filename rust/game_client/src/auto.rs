//! Scripted player for `--selftest` / `--screenshot`: logs what the client sees so the whole
//! loop (connect, spawn, steer, split, eject, eat, die, respawn) can be verified without eyes.
//!
//! Life 1: aim at a fixed world point, hunt food, then run into a bigger player until dead.
//! Life 2 (after auto-respawn): +500 mass, split, eject, hunt, then finish.

use crate::world::{Kind, World};
use crate::Mode;

#[derive(Default)]
pub struct Action {
    pub spawn: bool,
    pub packets: Vec<Vec<u8>>,
    pub aim: Option<(f64, f64)>,
    pub screenshot: Option<String>,
}

pub struct Auto {
    selftest: bool,
    shot: Option<String>,
    shot_at: f64,
    pub finished: bool,
    start: Option<f64>,
    life: u32,
    t0: Option<f64>,
    target: (f64, f64),
    last_log: f64,
    grown: bool,
    split: bool,
    ejects: u32,
    last_eject: f64,
    died_at: Option<f64>,
    shot_done: bool,
}

impl Auto {
    pub fn new(selftest: bool, shot: Option<String>, shot_at: f64) -> Auto {
        Auto {
            selftest,
            shot,
            shot_at,
            finished: false,
            start: None,
            life: 0,
            t0: None,
            target: (0.0, 0.0),
            last_log: 0.0,
            grown: false,
            split: false,
            ejects: 0,
            last_eject: 0.0,
            died_at: None,
            shot_done: false,
        }
    }

    pub fn step(&mut self, now: f64, mode: Mode, world: &World, _no_nick: bool, ready: bool, fps: i32) -> Action {
        let mut act = Action::default();
        let start = *self.start.get_or_insert(now);
        if self.selftest && now - start > 150.0 {
            println!("[auto] SELFTEST TIMEOUT");
            self.finished = true;
            return act;
        }
        match mode {
            Mode::Menu => {
                if ready && now - start > 1.0 {
                    act.spawn = true;
                    println!("[auto] t={:.1} spawn requested (packets={} border={:?})", now - start, world.packets, world.border);
                }
                return act;
            }
            Mode::Dead => {
                let d = *self.died_at.get_or_insert(now);
                if self.t0.take().is_some() {
                    println!("[auto] t={:.1} DEAD in life {}", now - start, self.life);
                }
                if now - d > 1.5 {
                    act.spawn = true;
                    self.died_at = None;
                    println!("[auto] t={:.1} respawn requested", now - start);
                }
                return act;
            }
            Mode::Spectating => return act,
            Mode::Playing => {}
        }
        let own: Vec<_> = world.mine.iter().filter_map(|id| world.cells.get(id)).filter(|c| c.dead_at.is_none()).collect();
        if own.is_empty() {
            return act;
        }
        let n = own.len() as f64;
        let (cx, cy) = (own.iter().map(|c| c.x).sum::<f64>() / n, own.iter().map(|c| c.y).sum::<f64>() / n);
        let mass: f64 = own.iter().map(|c| c.mass()).sum();
        let max_s = own.iter().map(|c| c.ns).fold(0.0, f64::max);
        if self.t0.is_none() {
            self.life += 1;
            self.t0 = Some(now);
            self.grown = false;
            self.split = false;
            self.ejects = 0;
            let b = world.border.map_or(3000.0, |b| b.right.min(-b.left));
            let (dx, dy) = if cx > 0.0 { (-1500.0, -900.0) } else { (1500.0, 900.0) };
            self.target = ((cx + dx).clamp(-b + 200.0, b - 200.0), (cy + dy).clamp(-b + 200.0, b - 200.0));
            println!("[auto] t={:.1} LIFE {} start at ({:.0},{:.0}) mass {:.0}; steer target ({:.0},{:.0})", now - start, self.life, cx, cy, mass, self.target.0, self.target.1);
        }
        let t = now - self.t0.unwrap();
        // choose aim
        let nearest = |pred: &dyn Fn(&crate::world::Cell) -> bool| {
            world
                .cells
                .values()
                .filter(|c| c.dead_at.is_none() && pred(c))
                .min_by(|a, b| ((a.x - cx).hypot(a.y - cy)).total_cmp(&(b.x - cx).hypot(b.y - cy)))
                .map(|c| (c.x, c.y))
        };
        let hunt = || nearest(&|c| c.kind == Kind::Food || (c.kind == Kind::Player && c.ns < 0.7 * max_s && !world.mine.contains(&c.id)));
        let mut aim = Some(self.target);
        if self.life == 1 {
            if t >= 4.0 {
                aim = hunt();
            }
            if t >= 14.0 {
                if let Some(p) = nearest(&|c| c.kind == Kind::Player && c.ns > 1.25 * max_s && !world.mine.contains(&c.id)) {
                    aim = Some(p);
                }
            }
        } else {
            if t >= 1.0 && !self.grown {
                self.grown = true;
                act.packets.push(vec![205]);
                println!("[auto] sent +500 (mass before {:.0})", mass);
            }
            if t >= 3.0 && !self.split {
                self.split = true;
                act.packets.push(vec![17]);
                println!("[auto] sent split (cells before {})", own.len());
            }
            if t >= 5.0 && self.ejects < 4 && now - self.last_eject > 0.3 {
                self.ejects += 1;
                self.last_eject = now;
                act.packets.push(vec![21]);
                println!("[auto] sent eject #{}", self.ejects);
            }
            aim = if t >= 7.0 { hunt() } else { Some(self.target) };
            if t >= 16.0 && self.selftest {
                println!("[auto] SELFTEST DONE (final mass {:.0}, eats seen {})", mass, world.eats);
                self.finished = true;
            }
        }
        act.aim = aim.or(Some(self.target));
        if let Some(path) = &self.shot {
            if !self.shot_done && t >= self.shot_at && (self.life >= 2 || !self.selftest) {
                self.shot_done = true;
                act.screenshot = Some(path.clone());
            }
        }
        if !self.selftest && self.shot.is_some() && self.life == 1 && t >= 1.0 && !self.grown {
            self.grown = true;
            act.packets.push(vec![205]);
        }
        if now - self.last_log >= 0.5 {
            self.last_log = now;
            let dist = (cx - self.target.0).hypot(cy - self.target.1);
            let ejected = world.cells.values().filter(|c| c.kind == Kind::Ejected).count();
            let desc: Vec<String> = own.iter().take(6).map(|c| format!("{}:({:.0},{:.0},s{:.0})", c.id, c.x, c.y, c.ns)).collect();
            println!(
                "[auto] t={:.1} life {} t_life={:.1} cells_own={} mass={:.0} centroid=({:.0},{:.0}) dist_to_target={:.0} visible={} ejected={} eats={} pkts={} fps={} own=[{}]",
                now - start, self.life, t, own.len(), mass, cx, cy, dist, world.cells.len(), ejected, world.eats, world.packets, fps, desc.join(" ")
            );
        }
        act
    }
}
