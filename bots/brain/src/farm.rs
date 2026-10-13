//! Food routing (port of bots/specs/farm.py): beam search over pellet routes + exploration memory.

use crate::game::{Cell, Snapshot};

const CANDIDATES: usize = 14;
const ROUTE_DEPTH: usize = 4;
const BEAM: usize = 24;
const GRID: usize = 8;

/// masks[i][j]: bitmask of pellets k within `reach` of segment start_i -> pellet_j.
/// Starts: index 0 = our cell, then each pellet. Bit j is always set.
fn sweep_masks(sx: &[f64], sy: &[f64], px: &[f64], py: &[f64], reach: f64) -> Vec<Vec<u64>> {
    let n = px.len();
    let mut out = vec![vec![0u64; n]; n + 1];
    for i in 0..=n {
        let (ax, ay) = (sx[i], sy[i]);
        for j in 0..n {
            let (dx, dy) = (px[j] - ax, py[j] - ay);
            let seg2 = (dx * dx + dy * dy).max(1e-9);
            let mut m = 0u64;
            for k in 0..n {
                let t = (((px[k] - ax) * dx + (py[k] - ay) * dy) / seg2).clamp(0.0, 1.0);
                if (ax + t * dx - px[k]).hypot(ay + t * dy - py[k]) < reach {
                    m |= 1 << k;
                }
            }
            out[i][j] = m | (1 << j);
        }
    }
    out
}

#[derive(Clone, Copy)]
struct Partial {
    score: f64,
    length: f64,
    eaten: u64,
    at: usize,
    route: [u8; ROUTE_DEPTH],
    len: usize,
}

pub struct FoodRouter {
    step_seconds: f64,
    clock: f64,
    target_id: Option<u32>,
    visit_grid: Option<[[f64; GRID]; GRID]>,
}

impl FoodRouter {
    pub fn new() -> FoodRouter {
        FoodRouter { step_seconds: 0.08, clock: 0.0, target_id: None, visit_grid: None }
    }

    pub fn reset(&mut self) {
        self.target_id = None;
    }

    /// Best route by pellets eaten per unit of path; optionally forced to start at `first`.
    fn plan(&self, me: &Cell, food: &[Cell], first: Option<&Cell>) -> (Vec<Cell>, f64) {
        let mut order: Vec<(f64, usize)> = food.iter().enumerate().map(|(i, c)| (me.dist(c), i)).collect();
        order.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap()); // stable, like Python's sorted
        let mut near: Vec<Cell> = order.iter().take(CANDIDATES).map(|&(_, i)| food[i]).collect();
        if let Some(f) = first {
            if !near.iter().any(|c| c.id == f.id) {
                near.push(*f);
            }
        }
        let n = near.len();
        if n == 0 {
            return (Vec::new(), 0.0);
        }
        let px: Vec<f64> = near.iter().map(|c| c.x).collect();
        let py: Vec<f64> = near.iter().map(|c| c.y).collect();
        let mut sx = vec![me.x];
        sx.extend_from_slice(&px);
        let mut sy = vec![me.y];
        sy.extend_from_slice(&py);
        let masks = sweep_masks(&sx, &sy, &px, &py, me.size * 0.9);
        let dist: Vec<Vec<f64>> = (0..=n)
            .map(|i| (0..n).map(|j| (sx[i] - px[j]).hypot(sy[i] - py[j])).collect())
            .collect();
        let first_j = first.map(|f| near.iter().position(|c| c.id == f.id).unwrap());
        let mut beam = vec![Partial { score: 0.0, length: 0.0, eaten: 0, at: 0, route: [0; ROUTE_DEPTH], len: 0 }];
        let mut best_score = 0.0f64;
        let mut best_route: Option<([u8; ROUTE_DEPTH], usize)> = None;
        let mut grown: Vec<Partial> = Vec::new();
        for depth in 0..ROUTE_DEPTH {
            grown.clear();
            for b in &beam {
                let (row_m, row_d) = (&masks[b.at], &dist[b.at]);
                let (lo, hi) = match (depth, first_j) {
                    (0, Some(fj)) => (fj, fj + 1),
                    _ => (0, n),
                };
                for j in lo..hi {
                    if b.eaten >> j & 1 == 1 {
                        continue;
                    }
                    let now = b.eaten | row_m[j];
                    let new_len = b.length + row_d[j];
                    let mut route = b.route;
                    route[b.len] = j as u8;
                    grown.push(Partial {
                        score: now.count_ones() as f64 / (new_len + me.size),
                        length: new_len,
                        eaten: now,
                        at: j + 1,
                        route,
                        len: b.len + 1,
                    });
                }
            }
            if grown.is_empty() {
                break;
            }
            grown.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            grown.truncate(BEAM);
            beam = grown.clone();
            if beam[0].score > best_score {
                best_score = beam[0].score;
                best_route = Some((beam[0].route, beam[0].len));
            }
        }
        match best_route {
            Some((r, l)) => (r[..l].iter().map(|&j| near[j as usize]).collect(), best_score),
            None => (Vec::new(), best_score),
        }
    }

    fn explore(&mut self, s: &Snapshot, me: &Cell) -> (f64, f64) {
        let b = &s.border;
        let (w, h) = ((b.right - b.left) / GRID as f64, (b.bottom - b.top) / GRID as f64);
        self.clock += self.step_seconds;
        let now = self.clock;
        let hi = ((me.x - b.left) / w).floor() as i64;
        let hj = ((me.y - b.top) / h).floor() as i64;
        let grid = self.visit_grid.get_or_insert([[-1e9; GRID]; GRID]);
        if (0..GRID as i64).contains(&hi) && (0..GRID as i64).contains(&hj) {
            grid[hi as usize][hj as usize] = now;
        }
        let mut best = (f64::NEG_INFINITY, 0usize, 0usize);
        for i in 0..GRID {
            for j in 0..GRID {
                let cx = b.left + (i as f64 + 0.5) * w;
                let cy = b.top + (j as f64 + 0.5) * h;
                let score = (now - grid[i][j]).min(300.0) - (cx - me.x).hypot(cy - me.y) / 50.0;
                if score > best.0 {
                    best = (score, i, j);
                }
            }
        }
        (b.left + (best.1 as f64 + 0.5) * w, b.top + (best.2 as f64 + 0.5) * h)
    }

    /// World point to head for. `unsafe_cell(cell)` is true for pellets in danger zones.
    pub fn next_waypoint(&mut self, s: &Snapshot, food: &[Cell], unsafe_cell: &dyn Fn(&Cell) -> bool,
                         switch_margin: f64) -> (f64, f64) {
        let me = *s.main().unwrap();
        let safe: Vec<Cell> = food.iter().filter(|c| !unsafe_cell(c)).copied().collect();
        let explore_point = self.explore(s, &me); // also records the visit
        if safe.is_empty() {
            self.target_id = None;
            return explore_point;
        }
        let (mut route, score) = self.plan(&me, &safe, None);
        let committed = self.target_id.and_then(|id| safe.iter().find(|c| c.id == id));
        if let Some(c) = committed {
            if !route.is_empty() && route[0].id != c.id {
                let (kept_route, kept) = self.plan(&me, &safe, Some(c));
                if score < kept * switch_margin {
                    route = kept_route;
                }
            }
        }
        self.target_id = route.first().map(|c| c.id);
        match route.first() {
            Some(c) => (c.x, c.y),
            None => explore_point,
        }
    }
}
