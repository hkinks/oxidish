//! Vision frames: what a vision bot sees. Plain grayscale, drawn from the player's Snapshot (the same data the
//! client gets), so the only size information is how big a circle looks.
//!   background 40, outside the border 0, pellets / ejected 150, viruses a bright spiky ring,
//!   our own cells 255 (a human sees their own name), other players one shade per player (the client's colour cue
//!   for "these pieces belong together") with a darker outline.
//! Edges are anti-aliased (4x4 subsamples), so pellets smaller than a pixel still show as faint dots.

use arena_client::game::{Kind, Snapshot};

const BG: f32 = 40.0;
const OUTSIDE: f32 = 0.0;
const PELLET: f32 = 150.0;
const VIRUS: f32 = 235.0;
const MINE: f32 = 255.0;

/// One player's shade from its protocol colour: 100..215 (never as bright as our own cells).
fn shade(color: u32) -> f32 {
    let h = color.wrapping_mul(2654435761) >> 24;
    100.0 + (h as f32 / 255.0) * 115.0
}

/// Grayscale `px` x `px` frame of the square world area of half-width `half` around (`cx`, `cy`).
pub fn render(s: &Snapshot, cx: f64, cy: f64, half: f64, px: usize) -> Vec<u8> {
    let scale = px as f64 / (2.0 * half); // pixels per world unit
    let x0 = cx - half;
    let y0 = cy - half;
    let mut img = vec![BG; px * px];
    let b = &s.border;
    for j in 0..px {
        for i in 0..px {
            let wx = x0 + (i as f64 + 0.5) / scale;
            let wy = y0 + (j as f64 + 0.5) / scale;
            if wx < b.left || wx > b.right || wy < b.top || wy > b.bottom {
                img[j * px + i] = OUTSIDE;
            }
        }
    }
    // small first, as the client draws
    let mut cells: Vec<_> = s.others.iter().map(|c| (c, false)).chain(s.mine.iter().map(|c| (c, true))).collect();
    cells.sort_by(|a, b| a.0.size.partial_cmp(&b.0.size).unwrap());
    for (c, mine) in cells {
        let r = c.size * scale;
        let (fx, fy) = ((c.x - x0) * scale, (c.y - y0) * scale);
        let (fill, edge, virus) = match c.kind {
            Kind::Food | Kind::Ejected => (PELLET, PELLET, false),
            Kind::Virus => (BG + 25.0, VIRUS, true),
            _ if mine => (MINE, 200.0, false),
            _ => {
                let s = shade(c.color);
                (s, s - 45.0, false)
            }
        };
        let edge_w = if virus { (r * 0.18).max(1.0) } else { (r * 0.06).max(0.6) };
        let rmax = r * 1.08 + 1.0;
        let i0 = (fx - rmax).floor().max(0.0) as usize;
        let i1 = (fx + rmax).ceil().clamp(0.0, px as f64) as usize;
        let j0 = (fy - rmax).floor().max(0.0) as usize;
        let j1 = (fy + rmax).ceil().clamp(0.0, px as f64) as usize;
        for j in j0..j1 {
            for i in i0..i1 {
                let (mut cov, mut acc) = (0.0f32, 0.0f32);
                for sj in 0..4 {
                    for si in 0..4 {
                        let dx = i as f64 + (si as f64 + 0.5) / 4.0 - fx;
                        let dy = j as f64 + (sj as f64 + 0.5) / 4.0 - fy;
                        let d = dx.hypot(dy);
                        // viruses: spiky outline (radius wobbles with the angle)
                        let rr = if virus { r * (1.0 + 0.07 * (18.0 * dy.atan2(dx)).cos()) } else { r };
                        if d <= rr {
                            cov += 1.0 / 16.0;
                            acc += if d > rr - edge_w { edge } else { fill } / 16.0;
                        }
                    }
                }
                if cov > 0.0 {
                    let p = &mut img[j * px + i];
                    *p = *p * (1.0 - cov) + acc;
                }
            }
        }
    }
    img.iter().map(|v| v.round().clamp(0.0, 255.0) as u8).collect()
}

/// Minimal grayscale PNG (8-bit, zlib via flate2).
pub fn png_gray(w: usize, h: usize, px: &[u8]) -> Vec<u8> {
    use std::io::Write;
    fn crc(data: &[u8]) -> u32 {
        let mut c = 0xffff_ffffu32;
        for &b in data {
            c ^= b as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
            }
        }
        !c
    }
    fn chunk(out: &mut Vec<u8>, kind: &[u8], data: &[u8]) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut body = kind.to_vec();
        body.extend_from_slice(data);
        out.extend_from_slice(&body);
        out.extend_from_slice(&crc(&body).to_be_bytes());
    }
    let mut raw = Vec::with_capacity((w + 1) * h);
    for row in px.chunks(w) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    z.write_all(&raw).unwrap();
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 0, 0, 0, 0]);
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &z.finish().unwrap());
    chunk(&mut out, b"IEND", &[]);
    out
}
