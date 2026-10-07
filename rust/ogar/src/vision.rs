//! Vectorised vision environment (docs/vision-bot.md, M1): many worlds, each with rule-bot opponents (in-process
//! `AgentDriver`s), server bots and externally controlled "learner" players. A learner decides every `sync`
//! ticks from rendered grayscale frames. A teacher policy (a rule bot / the planner) labels every decision.
//!
//! Frames per learner: `view` [2, px, px] (the camera square: centre = view centre, half-width = view-box
//! half-width) and `wide` [2, px/2, px/2] (2x that extent); channel 0 = now, channel 1 = the previous decision.
//! Action per learner (f32 x 4): target pixel (u, v) in the view frame's pixel coordinates, split flag, eject flag.
//! The teacher's decision (a world-space target from `Policy::act`) is converted to the same format by
//! `world_to_pixel` (clipped to the frame along the ray from the frame centre, which keeps its heading).

use std::sync::Mutex;
use std::time::Instant;

use arena_client::policy::Policy;
use serde_json::Value;

use crate::agent::{snapshot, AgentDriver};
use crate::config::Config;
use crate::render::{render_into, Scratch};
use crate::world::{PKind, World};

#[derive(Clone)]
pub struct VisionOpts {
    pub n_worlds: usize,
    pub learners_per_world: usize,
    /// (name, policy spec, count) rule bots that play as opponents in every world
    pub opponents: Vec<(String, Value, usize)>,
    /// policy spec of the teacher (labels every decision; drives the learners flagged teacher)
    pub teacher: Value,
    pub px: usize,
    /// ticks per decision (also the opponents' decision interval)
    pub sync: u32,
    pub bots: usize,
    pub food_mult: f64,
    pub map: f64,
    pub seed: u64,
    pub threads: usize,
    /// ticks simulated before the first observation (opponents spawn and start growing)
    pub warmup: u32,
    /// run the teacher on every decision (labels); false = skip it (pure evaluation, faster)
    pub label: bool,
}

impl VisionOpts {
    pub fn new(teacher: Value) -> VisionOpts {
        VisionOpts {
            n_worlds: 1,
            learners_per_world: 1,
            opponents: vec![],
            teacher,
            px: 256,
            sync: 2,
            bots: 8,
            food_mult: 2.0,
            map: 7000.0,
            seed: 1,
            threads: 1,
            warmup: 0,
            label: true,
        }
    }
}

/// The camera one observation was drawn with.
#[derive(Clone, Copy, Debug, Default)]
pub struct Camera {
    pub cx: f64,
    pub cy: f64,
    pub half: f64,
    pub px: usize,
}

impl Camera {
    /// pixels per world unit
    pub fn scale(&self) -> f64 {
        self.px as f64 / (2.0 * self.half)
    }
    /// pixel (continuous coords, pixel i covers [i, i+1)) -> world point
    pub fn pixel_to_world(&self, u: f64, v: f64) -> (f64, f64) {
        let s = self.scale();
        (self.cx - self.half + u / s, self.cy - self.half + v / s)
    }
    /// world point -> pixel; targets outside the frame are pulled in along the ray from the frame centre.
    pub fn world_to_pixel(&self, x: f64, y: f64) -> (f64, f64) {
        let s = self.scale();
        let c = self.px as f64 / 2.0;
        let (mut du, mut dv) = ((x - self.cx) * s, (y - self.cy) * s);
        let lim = c - 0.5;
        let m = du.abs().max(dv.abs());
        if m > lim {
            du *= lim / m;
            dv *= lim / m;
        }
        (c + du, c + dv)
    }
}

/// Per-phase wall time in ns (summed over worlds; divide by threads for wall).
#[derive(Clone, Copy, Default, Debug)]
pub struct Timing {
    pub tick: u64,
    pub opponents: u64,
    pub snapshot: u64,
    pub render: u64,
    pub teacher: u64,
    pub decisions: u64,
}

impl Timing {
    fn add(&mut self, o: &Timing) {
        self.tick += o.tick;
        self.opponents += o.opponents;
        self.snapshot += o.snapshot;
        self.render += o.render;
        self.teacher += o.teacher;
        self.decisions += o.decisions;
    }
}

struct Learner {
    pi: usize,
    name: String,
    teacher: Policy,
    cam: Camera,
    /// the teacher's world-space target for the current observation (kept for tests)
    t_world: (f64, f64),
    prev_mass: f64,
}

struct WorldEnv {
    w: World,
    opps: Vec<AgentDriver>,
    learners: Vec<Learner>,
    scratch: Scratch,
    timing: Timing,
}

/// Output buffers of one world (slices into the env's arrays).
struct WorldOut<'a> {
    view: &'a mut [u8],
    wide: &'a mut [u8],
    t_act: &'a mut [f32],
    reward: &'a mut [f32],
    done: &'a mut [u8],
    mass: &'a mut [f32],
    rank: &'a mut [i32],
    leader: &'a mut [u8],
    ncells: &'a mut [i32],
}

pub struct VecEnv {
    pub opts: VisionOpts,
    worlds: Vec<WorldEnv>,
    episode: u64,
    /// total learners
    pub n: usize,
    pub wpx: usize,
    /// [N, 2, px, px]
    pub view: Vec<u8>,
    /// [N, 2, px/2, px/2]
    pub wide: Vec<u8>,
    /// teacher action per learner: [N, 4] = (u, v, split, eject), same format as the student's actions
    pub t_act: Vec<f32>,
    pub reward: Vec<f32>,
    pub done: Vec<u8>,
    pub mass: Vec<f32>,
    /// 0 = largest player in the world
    pub rank: Vec<i32>,
    pub leader: Vec<u8>,
    pub ncells: Vec<i32>,
    /// 1 = the teacher drives this learner (the action passed to `step` is ignored)
    pub teacher_mask: Vec<u8>,
}

fn par<T: Send>(jobs: Vec<T>, threads: usize, f: impl Fn(T) + Sync) {
    if threads <= 1 || jobs.len() <= 1 {
        jobs.into_iter().for_each(f);
        return;
    }
    let n = jobs.len();
    let q = Mutex::new(jobs.into_iter());
    std::thread::scope(|s| {
        for _ in 0..threads.min(n) {
            s.spawn(|| loop {
                let j = q.lock().unwrap().next();
                match j {
                    Some(j) => f(j),
                    None => break,
                }
            });
        }
    });
}

fn rank_of(w: &World, pi: usize) -> i32 {
    let mine = w.players[pi].total_mass(w);
    w.players
        .iter()
        .enumerate()
        .filter(|(i, p)| *i != pi && !p.removed && !p.cells.is_empty() && p.total_mass(w) > mine)
        .count() as i32
}

impl WorldEnv {
    fn new(o: &VisionOpts, seed: u64) -> WorldEnv {
        let cfg = Config::new(o.map, o.bots, o.food_mult);
        let mut w = World::new(cfg, seed);
        let mut opps = Vec::new();
        for (name, spec, count) in &o.opponents {
            for i in 0..*count {
                opps.push(AgentDriver::new(&mut w, &format!("{name}-{i}"), spec, o.sync as u64));
            }
        }
        let mut learners = Vec::new();
        for k in 0..o.learners_per_world {
            let pi = w.add_player(PKind::Agent);
            let name = format!("learner-{k}");
            w.join_game(pi, &name, None);
            learners.push(Learner {
                pi,
                name,
                teacher: Policy::from_spec(&o.teacher),
                cam: Camera::default(),
                t_world: (0.0, 0.0),
                prev_mass: 0.0,
            });
        }
        let mut we = WorldEnv { w, opps, learners, scratch: Scratch::default(), timing: Timing::default() };
        for o2 in we.opps.iter_mut() {
            o2.step(&mut we.w); // spawn
        }
        we.w.kill_log.clear();
        for _ in 0..o.warmup.max(o.sync) {
            we.advance(o.sync);
        }
        we
    }

    /// `sync` ticks, then the opponents decide.
    fn advance(&mut self, sync: u32) {
        let t = Instant::now();
        for _ in 0..sync {
            self.w.tick();
        }
        self.timing.tick += t.elapsed().as_nanos() as u64;
        let t = Instant::now();
        for a in self.opps.iter_mut() {
            a.step(&mut self.w);
        }
        self.w.kill_log.clear();
        self.timing.opponents += t.elapsed().as_nanos() as u64;
    }

    /// Render every learner's frames, run the teacher, fill reward/done/info. `first` = no previous frame yet.
    fn observe(&mut self, o: &VisionOpts, out: &mut WorldOut, first: bool) {
        let (px, wpx) = (o.px, o.px / 2);
        let (vs, ws) = (px * px, wpx * wpx);
        for (k, l) in self.learners.iter_mut().enumerate() {
            let mut died = false;
            let mut mass = self.w.players[l.pi].total_mass(&self.w);
            if self.w.players[l.pi].cells.is_empty() {
                died = true;
                self.w.join_game(l.pi, &l.name, None);
                self.w.refresh_view(l.pi);
                l.teacher.reset();
                mass = self.w.players[l.pi].total_mass(&self.w);
            }
            let t = Instant::now();
            let mut s = snapshot(&self.w, l.pi);
            self.timing.snapshot += t.elapsed().as_nanos() as u64;
            let (vx0, vy0, vx1, vy1) = self.w.view_box(l.pi);
            let (cx, cy, half) = ((vx0 + vx1) / 2.0, (vy0 + vy1) / 2.0, (vx1 - vx0) / 2.0);
            l.cam = Camera { cx, cy, half, px };
            let t = Instant::now();
            let v = &mut out.view[k * 2 * vs..(k + 1) * 2 * vs];
            let wd = &mut out.wide[k * 2 * ws..(k + 1) * 2 * ws];
            if !first && !died {
                v.copy_within(0..vs, vs);
                wd.copy_within(0..ws, ws);
            }
            render_into(&s, cx, cy, half, px, &mut self.scratch, &mut v[..vs]);
            render_into(&s, cx, cy, half * 2.0, wpx, &mut self.scratch, &mut wd[..ws]);
            if first || died {
                v.copy_within(0..vs, vs);
                wd.copy_within(0..ws, ws);
            }
            self.timing.render += t.elapsed().as_nanos() as u64;
            let ta = &mut out.t_act[k * 4..k * 4 + 4];
            if o.label && !s.mine.is_empty() {
                let t = Instant::now();
                let a = l.teacher.act(&mut s);
                self.timing.teacher += t.elapsed().as_nanos() as u64;
                l.t_world = (a.x, a.y);
                let (u, vv) = l.cam.world_to_pixel(a.x, a.y);
                ta.copy_from_slice(&[u as f32, vv as f32, a.split as u8 as f32, a.eject as u8 as f32]);
            } else {
                let c = px as f32 / 2.0;
                ta.copy_from_slice(&[c, c, 0.0, 0.0]);
            }
            if !first {
                out.reward[k] = if died { -l.prev_mass as f32 } else { (mass - l.prev_mass) as f32 };
                out.done[k] = died as u8;
            } else {
                out.reward[k] = 0.0;
                out.done[k] = 0;
            }
            l.prev_mass = mass;
            let r = rank_of(&self.w, l.pi);
            out.mass[k] = mass as f32;
            out.rank[k] = r;
            out.leader[k] = (r == 0) as u8;
            out.ncells[k] = self.w.players[l.pi].cells.len() as i32;
            self.timing.decisions += 1;
        }
    }

    fn step(&mut self, o: &VisionOpts, actions: &[f32], mask: &[u8], out: &mut WorldOut) {
        for (k, l) in self.learners.iter().enumerate() {
            let p = &mut self.w.players[l.pi];
            if p.cells.is_empty() {
                continue;
            }
            let a = if mask[k] != 0 { &out.t_act[k * 4..k * 4 + 4] } else { &actions[k * 4..k * 4 + 4] };
            let (u, v) = (a[0] as f64, a[1] as f64);
            let (u, v) = (u.clamp(0.0, o.px as f64), v.clamp(0.0, o.px as f64));
            let (x, y) = l.cam.pixel_to_world(u, v);
            p.mouse_in = Some(((x as i32) as f64, (y as i32) as f64));
            if a[2] > 0.5 {
                p.p_split = true;
            }
            if a[3] > 0.5 {
                p.p_eject = true;
            }
        }
        self.advance(o.sync);
        self.observe(o, out, false);
    }
}

impl VecEnv {
    pub fn new(opts: VisionOpts) -> VecEnv {
        let n = opts.n_worlds * opts.learners_per_world;
        let (px, wpx) = (opts.px, opts.px / 2);
        VecEnv {
            n,
            wpx,
            worlds: Vec::new(),
            episode: 0,
            view: vec![0; n * 2 * px * px],
            wide: vec![0; n * 2 * wpx * wpx],
            t_act: vec![0.0; n * 4],
            reward: vec![0.0; n],
            done: vec![0; n],
            mass: vec![0.0; n],
            rank: vec![0; n],
            leader: vec![0; n],
            ncells: vec![0; n],
            teacher_mask: vec![0; n],
            opts,
        }
    }

    fn outs(&mut self) -> Vec<WorldOut<'_>> {
        let l = self.opts.learners_per_world;
        let (px, wpx) = (self.opts.px, self.wpx);
        let mut vs = self.view.chunks_mut(l * 2 * px * px);
        let mut ws = self.wide.chunks_mut(l * 2 * wpx * wpx);
        let mut ta = self.t_act.chunks_mut(l * 4);
        let mut rw = self.reward.chunks_mut(l);
        let mut dn = self.done.chunks_mut(l);
        let mut ms = self.mass.chunks_mut(l);
        let mut rk = self.rank.chunks_mut(l);
        let mut ld = self.leader.chunks_mut(l);
        let mut nc = self.ncells.chunks_mut(l);
        (0..self.opts.n_worlds)
            .map(|_| WorldOut {
                view: vs.next().unwrap(),
                wide: ws.next().unwrap(),
                t_act: ta.next().unwrap(),
                reward: rw.next().unwrap(),
                done: dn.next().unwrap(),
                mass: ms.next().unwrap(),
                rank: rk.next().unwrap(),
                leader: ld.next().unwrap(),
                ncells: nc.next().unwrap(),
            })
            .collect()
    }

    /// New worlds (fresh seeds each call); fills the observation buffers (reward / done = 0).
    pub fn reset(&mut self) {
        let o = self.opts.clone();
        let ep = self.episode;
        self.episode += 1;
        let mut slots: Vec<Option<WorldEnv>> = (0..o.n_worlds).map(|_| None).collect();
        {
            let jobs: Vec<(usize, &mut Option<WorldEnv>)> = slots.iter_mut().enumerate().collect();
            par(jobs, o.threads, |(i, slot)| {
                *slot = Some(WorldEnv::new(&o, o.seed.wrapping_add(i as u64 * 7919).wrapping_add(ep * 1_000_003)));
            });
        }
        self.worlds = slots.into_iter().map(|s| s.unwrap()).collect();
        let mut worlds = std::mem::take(&mut self.worlds);
        {
            let outs = self.outs();
            let jobs: Vec<_> = worlds.iter_mut().zip(outs).collect();
            par(jobs, o.threads, |(we, mut out)| we.observe(&o, &mut out, true));
        }
        self.worlds = worlds;
    }

    /// `actions`: [N, 4] f32 = (u, v, split, eject) in view-frame pixels. Learners with `teacher_mask` = 1 are
    /// driven by their teacher label instead. Dead learners respawn at once (done = 1, the obs is the new life).
    pub fn step(&mut self, actions: &[f32]) {
        assert_eq!(actions.len(), self.n * 4, "actions must be [N, 4]");
        let o = self.opts.clone();
        let l = o.learners_per_world;
        let mask = self.teacher_mask.clone();
        let mut worlds = std::mem::take(&mut self.worlds);
        assert!(!worlds.is_empty(), "call reset() first");
        {
            let outs = self.outs();
            let jobs: Vec<_> = worlds.iter_mut().zip(outs).enumerate().collect();
            par(jobs, o.threads, |(i, (we, mut out))| {
                we.step(&o, &actions[i * l * 4..(i + 1) * l * 4], &mask[i * l..(i + 1) * l], &mut out)
            });
        }
        self.worlds = worlds;
    }

    pub fn timing(&self) -> Timing {
        let mut t = Timing::default();
        for w in &self.worlds {
            t.add(&w.timing);
        }
        t
    }

    /// The teachers' decision counters summed over all learners (for the planner: plan_gated, plan_deep, plan_us, ...).
    pub fn teacher_counters(&self) -> Vec<(String, i64)> {
        let mut out: Vec<(String, i64)> = Vec::new();
        for l in self.worlds.iter().flat_map(|w| w.learners.iter()) {
            for (k, v) in &l.teacher.counters().0 {
                match out.iter_mut().find(|e| &e.0 == k) {
                    Some(e) => e.1 += v,
                    None => out.push((k.clone(), *v)),
                }
            }
        }
        out
    }

    /// For tests: each learner's camera of the current observation and the teacher's raw world target.
    pub fn debug_teacher(&self) -> Vec<(Camera, (f64, f64))> {
        self.worlds.iter().flat_map(|w| w.learners.iter().map(|l| (l.cam, l.t_world))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn spec() -> Value {
        let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../bot/c3_4.json")).unwrap();
        serde_json::from_str(&text).unwrap()
    }

    #[test]
    fn pixel_world_roundtrip_is_exact() {
        let c = Camera { cx: 100.0, cy: -50.0, half: 900.0, px: 256 };
        for (u, v) in [(0.5, 0.5), (128.0, 128.0), (255.5, 3.25)] {
            let (x, y) = c.pixel_to_world(u, v);
            let (u2, v2) = c.world_to_pixel(x, y);
            assert!((u - u2).abs() < 1e-9 && (v - v2).abs() < 1e-9);
        }
        // far target: clipped inside the frame, same heading
        let (u, v) = c.world_to_pixel(100.0 + 9000.0, -50.0 + 4500.0);
        assert!(u <= 255.5 && v <= 255.5);
        assert!(((v - 128.0) / (u - 128.0) - 0.5).abs() < 1e-9);
    }

    /// teacher world target -> pixel (the label) -> world (what the env applies) lands within a pixel.
    #[test]
    fn teacher_action_roundtrips_through_world() {
        let mut o = VisionOpts::new(spec());
        o.n_worlds = 2;
        o.learners_per_world = 2;
        o.px = 128;
        o.opponents = vec![("c34".into(), spec(), 2)];
        o.warmup = 100;
        let mut env = VecEnv::new(o);
        env.reset();
        env.teacher_mask.iter_mut().for_each(|m| *m = 1);
        let (mut checked, mut unclipped) = (0, 0);
        for _ in 0..300 {
            for (i, (cam, (tx, ty))) in env.debug_teacher().iter().enumerate() {
                let (u, v) = (env.t_act[i * 4] as f64, env.t_act[i * 4 + 1] as f64);
                let (wx, wy) = cam.pixel_to_world(u, v);
                let err_px = ((wx - tx).hypot(wy - ty)) * cam.scale();
                checked += 1;
                let inside = (tx - cam.cx).abs().max((ty - cam.cy).abs()) * cam.scale() < cam.px as f64 / 2.0 - 0.5;
                if inside {
                    unclipped += 1;
                    assert!(err_px < 1.0, "round trip error {err_px} px");
                }
            }
            env.step(&vec![0.0; env.n * 4]);
        }
        assert!(unclipped * 2 > checked, "most teacher targets should be inside the frame ({unclipped}/{checked})");
    }

    #[test]
    fn shapes_and_determinism() {
        let mk = || {
            let mut o = VisionOpts::new(spec());
            o.n_worlds = 3;
            o.learners_per_world = 2;
            o.px = 64;
            o.threads = 3;
            o.opponents = vec![("c34".into(), json!({}), 1)];
            let mut e = VecEnv::new(o);
            e.reset();
            e.teacher_mask.iter_mut().for_each(|m| *m = 1);
            for _ in 0..20 {
                e.step(&vec![0.0; 6 * 4]);
            }
            e
        };
        let (a, b) = (mk(), mk());
        assert_eq!(a.view.len(), 6 * 2 * 64 * 64);
        assert_eq!(a.wide.len(), 6 * 2 * 32 * 32);
        assert_eq!(a.view, b.view);
        assert_eq!(a.t_act, b.t_act);
        assert!(a.view.iter().any(|&p| p > 40), "pellets / cells should be drawn");
        assert!(a.mass.iter().all(|&m| m > 0.0));
    }
}
