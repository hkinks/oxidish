//! MultiOgar's built-in server bot (ai/BotPlayer.js).

use crate::world::{World, T_FOOD, T_PLAYER, T_VIRUS};

impl World {
    pub(crate) fn bot_decide(&mut self, pi: usize) {
        if self.players[pi].split_cooldown > 0 {
            self.players[pi].split_cooldown -= 1;
        }
        // largest cell (first on ties)
        let mut big: Option<u32> = None;
        for &h in &self.players[pi].cells {
            if big.map_or(true, |b| self.ents[h as usize].size > self.ents[b as usize].size) {
                big = Some(h);
            }
        }
        let Some(ch) = big else { return };
        if let Some((th, tid)) = self.players[pi].split_target {
            let t = &self.ents[th as usize];
            let gone = t.id != tid || t.removed;
            if gone {
                self.players[pi].split_target = None;
                self.players[pi].target_pursuit = 0;
            }
            if self.players[pi].split_target.is_some() && self.players[pi].target_pursuit > 0 {
                self.players[pi].target_pursuit -= 1;
                self.players[pi].mouse = (t.x, t.y);
                return;
            }
            self.players[pi].split_target = None;
        }
        let cell = self.ents[ch as usize].clone();
        let p = &self.players[pi];
        let split_cd = 1.5 * (p.cells.len() as f64) < 9.0 && p.split_cooldown == 0;
        let size = cell.size / 1.3;
        let nview = p.view_nodes.len() as f64;
        let max_cells = self.cfg.player_max_cells as usize;
        let (mut rx, mut ry) = (0.0f64, 0.0f64);
        let mut prey: Option<u32> = None;
        for &h in &p.view_nodes {
            let check = &self.ents[h as usize];
            if check.owner == pi as i32 {
                continue;
            }
            let mut influence = 0.0f64;
            match check.kind {
                T_PLAYER => {
                    if cell.size > check.size * 1.3 {
                        influence = check.size / nview.ln();
                    } else if check.size > cell.size * 1.3 {
                        influence = -(check.size / cell.size).ln();
                    } else {
                        influence = -check.size / cell.size;
                    }
                }
                T_FOOD => influence = 1.0,
                T_VIRUS => {
                    if cell.size > check.size * 1.3 {
                        influence = if p.cells.len() >= max_cells { 2.0 } else { -1.0 };
                    }
                }
                _ => {
                    if cell.size > check.size * 1.3 {
                        influence = 2.0;
                    }
                }
            }
            if influence != 0.0 {
                let (dx, dy) = (check.x - cell.x, check.y - cell.y);
                let mut dist = (dx * dx + dy * dy).sqrt();
                if influence < 0.0 {
                    dist -= cell.size + check.size;
                }
                if dist < 1.0 {
                    dist = 1.0;
                }
                influence /= dist;
                // displacement.normalize().scale(influence)
                let len = (dx * dx + dy * dy).sqrt();
                let (nx, ny) = (dx * (1.0 / len), dy * (1.0 / len));
                let (sx, sy) = (nx * influence, ny * influence);
                if split_cd && check.kind == T_PLAYER && size > 1.3 * check.size && cell.size * 0.4 < check.size {
                    let speed = (1.3 * self.cfg.player_split_speed).max(cell.size / 1.4142 * 4.5);
                    if speed >= dist {
                        match prey {
                            Some(pr) => {
                                if check.size > self.ents[pr as usize].size {
                                    prey = Some(h);
                                }
                            }
                            None => prey = Some(h),
                        }
                    }
                }
                rx += sx;
                ry += sy;
            }
        }
        let len = (rx * rx + ry * ry).sqrt();
        let (rx, ry) = (rx * (1.0 / len), ry * (1.0 / len));
        let half = p.half_w;
        let mut mouse = (cell.x + rx * half, cell.y + ry * half);
        let mut set_prey = None;
        if let Some(pr) = prey {
            let e = &self.ents[pr as usize];
            mouse = (e.x, e.y);
            set_prey = Some((pr, e.id));
        }
        let p = &mut self.players[pi];
        p.mouse = mouse;
        if let Some(t) = set_prey {
            p.split_target = Some(t);
            p.target_pursuit = 20;
            p.split_cooldown = 15;
            p.p_split = true;
        }
    }
}
