//! Free helpers of the apos policy: threat projection, risk-field sweeps and geometry predicates.

use super::*;

pub(super) fn foresee(ents: &mut [E], food: &[C]) {
    let ejected: Vec<&C> = food.iter().filter(|c| c.ejected()).collect();
    let mut potentials = vec![0.0; ents.len()];
    for (ei, entry) in ents.iter().enumerate() {
        let mut extra = 0.0;
        let ecolor = entry.cell.unwrap().color;
        for (oi, other) in ents.iter().enumerate() {
            if oi == ei {
                continue;
            }
            if other.cell.unwrap().color == ecolor {
                continue;
            }
            if entry.mass < other.mass * EAT_RATIO {
                continue;
            }
            let gap = span(entry.x, entry.y, other.x, other.y) - entry.size - other.size;
            let reach = if entry.mass >= other.mass * SPLIT_THREAT_RATIO { FORESEE_RANGE + SPLIT_REACH } else { FORESEE_RANGE };
            if gap <= reach {
                extra += other.mass;
            }
        }
        let mut fed = 0.0;
        for cell in &ejected {
            if cell.color == ecolor {
                continue;
            }
            let gap = span(entry.x, entry.y, cell.x, cell.y) - entry.size;
            if gap <= FEED_RANGE {
                fed += cell.mass();
            }
        }
        if fed >= FEED_MIN {
            extra += fed;
            let mut feeder: Option<usize> = None;
            let mut closest = FEEDER_RANGE;
            for (oi, other) in ents.iter().enumerate() {
                if oi == ei {
                    continue;
                }
                if other.cell.unwrap().color == ecolor {
                    continue;
                }
                let gap = span(entry.x, entry.y, other.x, other.y) - entry.size - other.size;
                if gap < closest {
                    closest = gap;
                    feeder = Some(oi);
                }
            }
            if let Some(f) = feeder {
                extra += ents[f].mass * FEED_SHARE;
            }
        }
        potentials[ei] = extra;
    }
    for (e, p) in ents.iter_mut().zip(potentials) {
        e.potential = p;
    }
}

pub(super) fn cascade(entry: &mut E, ladder: &[Piece]) {
    let mut grown = entry.mass + entry.potential;
    let mut absorbed: Option<Vec<u32>> = None;
    let mut eats = 0.0;
    for piece in ladder {
        if grown < piece.mass * EAT_RATIO {
            break;
        }
        let gap = span(entry.x, entry.y, piece.x, piece.y) - entry.size - piece.size;
        if gap > CASCADE_RANGE {
            continue;
        }
        grown += piece.mass;
        eats += piece.mass;
        absorbed.get_or_insert_with(Vec::new).push(piece.id);
    }
    entry.grown = grown;
    entry.absorbed = absorbed;
    entry.eats = eats;
}

pub(super) fn grown_for(entry: &E, cell: &C) -> f64 {
    match &entry.absorbed {
        None => entry.grown,
        Some(a) => {
            if !a.contains(&cell.id) {
                entry.grown
            } else {
                entry.grown - cell.mass()
            }
        }
    }
}

pub(super) fn merge_threats(ents: &mut Vec<E>, threats: &mut Vec<usize>, small_mass: f64, ladder: &[Piece]) {
    let mut order: Vec<u32> = Vec::new();
    let mut groups: HashMap<u32, Vec<C>> = HashMap::new();
    for e in ents.iter() {
        let c = match e.cell {
            Some(c) => c,
            None => continue,
        };
        groups.entry(c.color).or_insert_with(|| {
            order.push(c.color);
            Vec::new()
        }).push(c);
    }
    for color in order {
        let group = &groups[&color];
        if group.len() < 2 {
            continue;
        }
        for i in 0..group.len() {
            for j in (i + 1)..group.len() {
                let a = group[i];
                let b = group[j];
                let reach = a.size + b.size;
                if (a.x - b.x).abs() > reach || (a.y - b.y).abs() > reach {
                    continue;
                }
                if span(a.x, a.y, b.x, b.y) > reach * 0.9 {
                    continue;
                }
                let mass = a.mass() + b.mass();
                let mut entry = E::new((a.x + b.x) / 2.0, (a.y + b.y) / 2.0, size_of(mass), mass, None, None, false);
                cascade(&mut entry, ladder);
                if entry.grown <= small_mass * THREAT_RATIO {
                    continue;
                }
                ents.push(entry);
                threats.push(ents.len() - 1);
            }
        }
    }
}

pub(super) fn ahead_of(threat: &E) -> Ahead {
    if threat.vx == 0.0 && threat.vy == 0.0 {
        return Ahead { x: threat.x, y: threat.y, vx: threat.vx, vy: threat.vy };
    }
    Ahead { x: threat.x + threat.vx * THREAT_LOOKAHEAD, y: threat.y + threat.vy * THREAT_LOOKAHEAD, vx: threat.vx, vy: threat.vy }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sweep(risk: &mut Field, cell: &C, tx: f64, ty: f64, tvx: f64, tvy: f64, radius: f64, buffer: f64, weight: f64, horizon: f64,
         nearer: f64) {
    let t = tables();
    let rx = tx - cell.x;
    let ry = ty - cell.y;
    let distance = (rx * rx + ry * ry).sqrt();
    if distance > horizon + radius + buffer {
        return;
    }
    if tvx * tvx + tvy * tvy >= FLIGHT_SPEED * FLIGHT_SPEED && rx * tvx + ry * tvy < 0.0 {
        sweep_flight(risk, tvx, tvy, rx, ry, radius, buffer, weight);
        return;
    }
    let band = radius + buffer;
    for i in 0..DIRECTIONS {
        let along = rx * t.dx[i] + ry * t.dy[i];
        if along <= 0.0 {
            let clear = distance - radius;
            if clear >= buffer {
                continue;
            }
            let close = if clear > 0.0 { 1.0 - clear / buffer } else { 1.0 };
            let leaving = -along / distance;
            risk[i] -= weight * close * close * leaving * ESCAPE_LEAN;
            continue;
        }
        let due = if along > nearer { along - nearer } else { 0.0 };
        if due >= horizon {
            continue;
        }
        let cross = rx * t.dy[i] - ry * t.dx[i];
        let clearance = if cross < 0.0 { -cross } else { cross };
        let gap = clearance - radius;
        if gap >= buffer {
            continue;
        }
        let near = if gap > 0.0 { 1.0 - gap / buffer } else { 1.0 };
        let closing = clamp01((distance - clearance) / band);
        let soon = 1.0 - due / horizon;
        risk[i] += weight * near * near * closing * soon;
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sweep_flight(risk: &mut Field, tvx: f64, tvy: f64, rx: f64, ry: f64, radius: f64, buffer: f64, weight: f64) {
    let t = tables();
    let speed = (tvx * tvx + tvy * tvy).sqrt();
    let vx = tvx / speed;
    let vy = tvy / speed;
    let clear = (rx * vy - ry * vx).abs() - radius;
    if clear >= buffer {
        return;
    }
    let close = if clear > 0.0 { 1.0 - clear / buffer } else { 1.0 };
    let toward = if ry * vx - rx * vy > 0.0 { 1.0 } else { -1.0 };
    for i in 0..DIRECTIONS {
        let along = t.dx[i] * vx + t.dy[i] * vy;
        let across = (t.dy[i] * vx - t.dx[i] * vy) * toward;
        let mut cost = along * along;
        if across > 0.0 {
            cost += FLIGHT_CROSS * across * across;
        }
        risk[i] += weight * close * cost;
    }
}

pub(super) fn push_wall(risk: &mut Field, gap: f64, ux: f64, uy: f64, near: f64) {
    if gap >= near {
        return;
    }
    let t = tables();
    let crowding = if gap > 0.0 { 1.0 - gap / near } else { 1.0 };
    let weight = WALL_SHOVE * crowding * crowding;
    for i in 0..DIRECTIONS {
        let dot = t.dx[i] * ux + t.dy[i] * uy;
        if dot > 0.0 {
            risk[i] += weight * dot * dot;
        }
    }
}

pub(super) fn spread(field: &mut Field, ux: f64, uy: f64, value: f64) {
    let t = tables();
    let b = js_round(uy.atan2(ux) / DIR_STEP);
    for j in -FOOD_CONE..=FOOD_CONE {
        let index = (b + j + DIRECTIONS as i64).rem_euclid(DIRECTIONS as i64) as usize;
        field[index] += value * t.kernel[j.unsigned_abs() as usize];
    }
}

pub(super) fn nearest_cell(mine: &[C], x: f64, y: f64, prey_mass: Option<f64>) -> Near {
    let mut cell: Option<usize> = None;
    let mut best = INF;
    for (i, own) in mine.iter().enumerate() {
        if let Some(pm) = prey_mass {
            if own.mass() < pm * PREY_RATIO {
                continue;
            }
        }
        let dx = x - own.x;
        let dy = y - own.y;
        let distance = dx * dx + dy * dy;
        if distance < best {
            best = distance;
            cell = Some(i);
        }
    }
    match cell {
        None => Near { cell: None, distance: INF, ux: 0.0, uy: 0.0 },
        Some(i) => {
            let distance = or1(best.sqrt());
            Near { cell, distance, ux: (x - mine[i].x) / distance, uy: (y - mine[i].y) / distance }
        }
    }
}

pub(super) fn grazers_of(view: &View) -> Vec<(f64, f64, f64, f64, f64)> {
    let mut out = Vec::new();
    for &i in &view.prey {
        let e = &view.ents[i];
        let speed = (e.vx * e.vx + e.vy * e.vy).sqrt();
        if speed <= 0.0 {
            continue;
        }
        out.push((e.x, e.y, e.vx / speed, e.vy / speed, e.size + speed * GRAZE_LEAD));
    }
    out
}

pub(super) fn grazed(grazers: &[(f64, f64, f64, f64, f64)], cell: &C) -> bool {
    for &(gx, gy, ux, uy, reach) in grazers {
        let dx = cell.x - gx;
        let dy = cell.y - gy;
        if dx * ux + dy * uy <= 0.0 {
            continue;
        }
        if dx * dx + dy * dy < reach * reach {
            return true;
        }
    }
    false
}

pub(super) fn loaded(view: &View, virus: &C) -> Option<(f64, f64)> {
    let mut fx = 0.0;
    let mut fy = 0.0;
    let mut feeder: Option<&E> = None;
    for cell in &view.food {
        if !cell.ejected() || cell.color == view.big.color {
            continue;
        }
        let gap = span(cell.x, cell.y, virus.x, virus.y) - virus.size;
        if gap > FEED_RANGE {
            continue;
        }
        fx += virus.x - cell.x;
        fy += virus.y - cell.y;
        for e in &view.ents[..view.n_enemies] {
            if e.cell.unwrap().color != cell.color {
                continue;
            }
            if span(e.x, e.y, virus.x, virus.y) > EJECT_REACH {
                continue;
            }
            feeder = Some(e);
        }
    }
    if fx == 0.0 && fy == 0.0 {
        return None;
    }
    if let Some(f) = feeder {
        fx = virus.x - f.x;
        fy = virus.y - f.y;
    }
    let length = or1((fx * fx + fy * fy).sqrt());
    Some((fx / length, fy / length))
}

pub(super) fn shelters(view: &View) -> bool {
    view.threats.iter().any(|&ti| view.ents[ti].mass > VIRUS_MASS * EAT_RATIO)
}

pub(super) fn mouthfuls(view: &View) -> impl Fn(&C) -> f64 + '_ {
    let radius = f64::max(view.big.size, 1.0);
    let mut buckets: HashMap<(i64, i64), Vec<&C>> = HashMap::new();
    for cell in &view.food {
        buckets.entry(((cell.x / radius).floor() as i64, (cell.y / radius).floor() as i64)).or_default().push(cell);
    }
    let reach = radius * radius;
    move |cell: &C| {
        let bx = (cell.x / radius).floor() as i64;
        let by = (cell.y / radius).floor() as i64;
        let mut mass = 0.0;
        for i in [-1i64, 0, 1] {
            for j in [-1i64, 0, 1] {
                if let Some(bucket) = buckets.get(&(bx + i, by + j)) {
                    for other in bucket {
                        let dx = other.x - cell.x;
                        let dy = other.y - cell.y;
                        if dx * dx + dy * dy <= reach {
                            mass += other.mass();
                        }
                    }
                }
            }
        }
        mass
    }
}

pub(super) fn shunned(view: &View, x: f64, y: f64) -> f64 {
    let mut shun: f64 = 1.0;
    for &(zx, zy, r2, jump2, appetite) in &view.zones {
        let dx = x - zx;
        let dy = y - zy;
        let reach = dx * dx + dy * dy;
        if reach < r2 {
            return 0.0;
        }
        if jump2 != 0.0 && reach < jump2 {
            shun = f64::min(shun, 1.0 - (1.0 - SPLIT_SHUN) * appetite);
        }
    }
    shun
}

pub(super) fn through(cell: &C, tx: f64, ty: f64, reach: f64, x: f64, y: f64) -> bool {
    let gap = distance_to_segment(tx, ty, cell.x, cell.y, x, y);
    if gap >= reach {
        return false;
    }
    let now = span(tx, ty, cell.x, cell.y);
    if now >= reach {
        return true;
    }
    gap < now - 1.0
}

pub(super) fn floor_along(view: &View, ux: f64, uy: f64) -> f64 {
    let mut floor = 0.0;
    for cell in &view.mine {
        let along = (cell.x - view.centre.0) * ux + (cell.y - view.centre.1) * uy;
        let edge = along + cell.size + CLEAR_MARGIN;
        if edge > floor {
            floor = edge;
        }
    }
    floor
}

pub(super) fn cuts_off(threat: &E, cell: &C, hx: f64, hy: f64, room: f64) -> bool {
    let mouth = f64::max(0.0, threat.size - cell.size / 3.0);
    let pace = walk_speed(threat.size) / walk_speed(cell.size);
    let run = f64::min(room, ESCAPE_REACH);
    let mut x = cell.x;
    let mut y = cell.y;
    let mut tx = threat.x;
    let mut ty = threat.y;
    let mut s = 0.0;
    while s <= run {
        let dx = x - tx;
        let dy = y - ty;
        let gap = (dx * dx + dy * dy).sqrt();
        if gap <= mouth {
            return true;
        }
        if pace < 1.0 && (dx * hx + dy * hy) / gap > pace {
            return false;
        }
        x += hx * CHASE_STRIDE;
        y += hy * CHASE_STRIDE;
        tx += (dx / gap) * pace * CHASE_STRIDE;
        ty += (dy / gap) * pace * CHASE_STRIDE;
        s += CHASE_STRIDE;
    }
    false
}

pub(super) fn lead(entry: &E, cell: &C) -> (f64, f64) {
    if entry.vx == 0.0 && entry.vy == 0.0 {
        return (entry.x, entry.y);
    }
    let distance = span(entry.x, entry.y, cell.x, cell.y);
    let ahead = f64::min(1000.0, (distance / SPLIT_REACH) * 1000.0);
    (entry.x + entry.vx * ahead, entry.y + entry.vy * ahead)
}

pub(super) fn haul(view: &View, ei: usize, tx: f64, ty: f64, half: f64, mass: f64) -> f64 {
    let mut total = view.ents[ei].mass;
    for &oi in &view.prey {
        if oi == ei {
            continue;
        }
        let other = &view.ents[oi];
        if mass / 2.0 < other.grown * SPLIT_KILL_RATIO {
            continue;
        }
        if span(other.x, other.y, tx, ty) < half {
            total += other.mass;
        }
    }
    total
}

pub(super) fn split_is_safe(view: &View, cell: &C, tx: f64, ty: f64) -> bool {
    let half = cell.mass() / 2.0;
    for enemy in &view.ents[..view.n_enemies] {
        if enemy.grown < half * THREAT_RATIO {
            continue;
        }
        let splits = enemy.grown > half * SPLIT_THREAT_RATIO;
        let reach = enemy.size + cell.size + THREAT_MARGIN + SPLIT_CAUTION + (if splits { SPLIT_REACH } else { 0.0 });
        let landing = span(enemy.x, enemy.y, tx, ty);
        let behind = span(enemy.x, enemy.y, cell.x, cell.y);
        if landing < reach || behind < reach {
            return false;
        }
    }
    if half > VIRUS_MASS * EAT_RATIO {
        let dx = tx - cell.x;
        let dy = ty - cell.y;
        let length = or1((dx * dx + dy * dy).sqrt());
        let reach = cell.size + SPLIT_REACH;
        let end_x = cell.x + (dx / length) * reach;
        let end_y = cell.y + (dy / length) * reach;
        for virus in &view.viruses {
            let gap = distance_to_segment(virus.x, virus.y, cell.x, cell.y, end_x, end_y);
            if gap < virus.size + size_of(half) * 0.5 {
                return false;
            }
        }
    }
    true
}

pub(super) fn pop_line(cell: &C, virus: &C, enemy: &E) -> Option<(f64, f64, f64)> {
    let dvx = virus.x - cell.x;
    let dvy = virus.y - cell.y;
    let to_virus = or1((dvx * dvx + dvy * dvy).sqrt());
    if to_virus > EJECT_REACH {
        return None;
    }
    if to_virus < cell.size + virus.size + 40.0 {
        return None;
    }
    let dex = enemy.x - virus.x;
    let dey = enemy.y - virus.y;
    let to_enemy = or1((dex * dex + dey * dey).sqrt());
    if to_enemy > POP_RANGE {
        return None;
    }
    let gap = span(enemy.x, enemy.y, cell.x, cell.y);
    if gap < enemy.size + cell.size + 300.0 {
        return None;
    }
    let ux = dvx / to_virus;
    let uy = dvy / to_virus;
    let align = ux * (dex / to_enemy) + uy * (dey / to_enemy);
    if align < POP_ALIGN {
        return None;
    }
    Some((ux, uy, align))
}

pub(super) fn pour_for<'a>(view: &'a View, poured: f64) -> Option<&'a E> {
    let mut best: Option<&E> = None;
    let mut closest = PREY_REACH;
    for entry in &view.ents[..view.n_enemies] {
        let distance = span(entry.x, entry.y, view.big.x, view.big.y);
        if distance >= closest {
            continue;
        }
        let split_now = view.big_mass / 2.0 >= entry.grown * SPLIT_KILL_RATIO;
        let split_then = poured / 2.0 >= entry.grown * SPLIT_KILL_RATIO;
        let eat_now = view.big_mass >= entry.grown * PREY_RATIO;
        let eat_then = poured >= entry.grown * PREY_RATIO;
        if (split_then && !split_now) || (eat_then && !eat_now) {
            best = Some(entry);
            closest = distance;
        }
    }
    best
}

pub(super) fn near_threat(view: &View, rng: f64) -> bool {
    for &ti in &view.threats {
        let threat = &view.ents[ti];
        for cell in &view.mine {
            let mass = cell.mass();
            if grown_for(threat, cell) < mass * THREAT_RATIO {
                continue;
            }
            let gap = span(threat.x, threat.y, cell.x, cell.y) - threat.size - cell.size;
            if gap <= rng {
                return true;
            }
        }
    }
    false
}
