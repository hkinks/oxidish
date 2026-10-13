//! MultiOgar protocol-6 packet encoding (what vendor/agarian-client, training.eval.client and the Rust
//! game client speak), byte-compatible with vendor/MultiOgar/src/packet/*.js.

use crate::world::{World, T_EJECT, T_FOOD, T_PLAYER, T_VIRUS};

pub const OP_AGENT: u8 = 0xC8;
pub const OP_STEP: u8 = 0xC9;
pub const OP_FOLLOW: u8 = 0xCA;
pub const OP_SPAWN_SIZE: u8 = 0xCB;
pub const OP_FOOD: u8 = 0xCC;
pub const OP_GROW: u8 = 0xCD;

pub fn clear_all() -> Vec<u8> {
    vec![0x12]
}

pub fn add_node(id: u32) -> Vec<u8> {
    let mut b = vec![0x20];
    b.extend_from_slice(&id.to_le_bytes());
    b
}

/// 33-byte SetBorder (no game type / name).
pub fn set_border(b: (f64, f64, f64, f64)) -> Vec<u8> {
    let mut p = vec![0x40];
    for v in [b.0, b.1, b.2, b.3] {
        p.extend_from_slice(&v.to_le_bytes());
    }
    p
}

/// SetBorder sent at handshake: border + game type + server name (the client keys its ping loop on "MultiOgar").
pub fn set_border_hello(b: (f64, f64, f64, f64), name: &str) -> Vec<u8> {
    let mut p = set_border(b);
    p.extend_from_slice(&0u32.to_le_bytes());
    p.extend_from_slice(name.as_bytes());
    p.push(0);
    p
}

pub fn update_pos(x: f64, y: f64, scale: f64) -> Vec<u8> {
    let mut p = vec![0x11];
    for v in [x, y, scale] {
        p.extend_from_slice(&(v as f32).to_le_bytes());
    }
    p
}

pub fn chat(name: &str, color: [u8; 3], text: &str, server: bool) -> Vec<u8> {
    let mut p = vec![0x63, if server { 0x80 } else { 0 }, color[0], color[1], color[2]];
    p.extend_from_slice(name.as_bytes());
    p.push(0);
    p.extend_from_slice(text.as_bytes());
    p.push(0);
    p
}

pub fn server_stat(json: &str) -> Vec<u8> {
    let mut p = vec![254];
    p.extend_from_slice(json.as_bytes());
    p.push(0);
    p
}

pub fn step(tick: u32, shots: u32, pops: u32, pop_mass: f32) -> Vec<u8> {
    let mut p = vec![OP_STEP];
    p.extend_from_slice(&tick.to_le_bytes());
    p.extend_from_slice(&shots.to_le_bytes());
    p.extend_from_slice(&pops.to_le_bytes());
    p.extend_from_slice(&pop_mass.to_le_bytes());
    p
}

/// FFA leaderboard (0x31) for client `pi`.
pub fn leaderboard(w: &World, pi: usize) -> Vec<u8> {
    let n = w.leaderboard.len().min(w.cfg.max_lb);
    let mut p = vec![0x31];
    p.extend_from_slice(&(n as u32).to_le_bytes());
    for &i in &w.leaderboard[..n] {
        p.extend_from_slice(&(if i == pi { 1u32 } else { 0u32 }).to_le_bytes());
        p.extend_from_slice(w.players[i].name.as_bytes());
        p.push(0);
    }
    p
}

fn put_cell_head(p: &mut Vec<u8>, w: &World, h: u32) {
    let e = &w.ents[h as usize];
    p.extend_from_slice(&e.id.to_le_bytes());
    p.extend_from_slice(&(e.x as i32).to_le_bytes());
    p.extend_from_slice(&(e.y as i32).to_le_bytes());
    p.extend_from_slice(&(e.size as u32 as u16).to_le_bytes());
}

/// UpdateNodes (0x10). `eat`: (victim id, killer id); `del`: ids that left view / vanished.
pub fn update_nodes(w: &World, add: &[u32], upd: &[u32], eat: &[(u32, u32)], del: &[u32]) -> Vec<u8> {
    let mut p = Vec::with_capacity(64 + 24 * (add.len() + upd.len()));
    p.push(0x10);
    p.extend_from_slice(&(eat.len() as u16).to_le_bytes());
    for &(victim, killer) in eat {
        p.extend_from_slice(&killer.to_le_bytes());
        p.extend_from_slice(&victim.to_le_bytes());
    }
    for &h in upd {
        let e = &w.ents[h as usize];
        put_cell_head(&mut p, w, h);
        let mut flags = 0u8;
        if e.kind == T_VIRUS {
            flags |= 0x01;
        }
        if e.kind == T_PLAYER {
            flags |= 0x02;
        }
        if e.kind == T_EJECT {
            flags |= 0x20;
        }
        if e.kind == T_FOOD {
            flags |= 0x80;
        }
        p.push(flags);
        if flags & 0x02 != 0 {
            p.extend_from_slice(&e.color);
        }
    }
    for &h in add {
        let e = &w.ents[h as usize];
        put_cell_head(&mut p, w, h);
        let owner = if e.owner >= 0 { Some(&w.players[e.owner as usize]) } else { None };
        let skin = owner.and_then(|o| o.skin.as_deref());
        let mut flags = 0x02u8;
        if e.kind == T_VIRUS {
            flags |= 0x01;
        }
        if skin.is_some() {
            flags |= 0x04;
        }
        if owner.is_some() {
            flags |= 0x08;
        }
        if e.kind == T_EJECT {
            flags |= 0x20;
        }
        if e.kind == T_FOOD {
            flags |= 0x80;
        }
        p.push(flags);
        p.extend_from_slice(&e.color);
        if let Some(s) = skin {
            p.extend_from_slice(s.as_bytes());
            p.push(0);
        }
        if let Some(o) = owner {
            p.extend_from_slice(o.name.as_bytes());
            p.push(0);
        }
    }
    p.extend_from_slice(&0u32.to_le_bytes());
    p.extend_from_slice(&((eat.len() + del.len()) as u16).to_le_bytes());
    for &(victim, _) in eat {
        p.extend_from_slice(&victim.to_le_bytes());
    }
    for &id in del {
        p.extend_from_slice(&id.to_le_bytes());
    }
    p
}

/// The client's incremental view: diff the new view nodes against what was sent last tick
/// (PlayerTracker.sendUpdate). Returns the packet and updates `client_nodes`.
pub fn diff_update(w: &World, pi: usize) -> Vec<u8> {
    let p = &w.players[pi];
    let vn = &p.view_nodes;
    let cn = &p.client_nodes;
    let (mut add, mut upd, mut eat, mut del): (Vec<u32>, Vec<u32>, Vec<(u32, u32)>, Vec<u32>) = (vec![], vec![], vec![], vec![]);
    let gone = |id: u32, eat: &mut Vec<(u32, u32)>, del: &mut Vec<u32>| match w.eaten.get(&id) {
        Some(&k) => eat.push((id, k)),
        None => del.push(id),
    };
    let (mut ni, mut oi) = (0, 0);
    while ni < vn.len() && oi < cn.len() {
        let e = &w.ents[vn[ni] as usize];
        if e.id < cn[oi] {
            add.push(vn[ni]);
            ni += 1;
        } else if e.id > cn[oi] {
            gone(cn[oi], &mut eat, &mut del);
            oi += 1;
        } else {
            if e.moving || e.kind == T_PLAYER {
                upd.push(vn[ni]);
            }
            if e.kind == T_VIRUS {
                add.push(vn[ni]);
            }
            ni += 1;
            oi += 1;
        }
    }
    while ni < vn.len() {
        add.push(vn[ni]);
        ni += 1;
    }
    while oi < cn.len() {
        gone(cn[oi], &mut eat, &mut del);
        oi += 1;
    }
    update_nodes(w, &add, &upd, &eat, &del)
}
