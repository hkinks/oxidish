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

/// Reusable buffers for `render_into` (one per worker thread).
#[derive(Default)]
pub struct Scratch {
    img: Vec<f32>,
    order: Vec<(f32, bool, usize)>,
    colin: Vec<bool>,
}

/// 4x4 subsample offsets inside a pixel.
const SUB: [f32; 4] = [0.125, 0.375, 0.625, 0.875];

/// Grayscale `px` x `px` frame of the square world area of half-width `half` around (`cx`, `cy`).
pub fn render(s: &Snapshot, cx: f64, cy: f64, half: f64, px: usize) -> Vec<u8> {
    let mut out = vec![0u8; px * px];
    render_into(s, cx, cy, half, px, &mut Scratch::default(), &mut out);
    out
}

/// `render` into a caller buffer (`out.len() == px * px`). Only edge pixels are supersampled (4x4); pixels
/// well inside a disc take its fill directly, cells under ~1.6 px across are one coverage-weighted dot.
pub fn render_into(s: &Snapshot, cx: f64, cy: f64, half: f64, px: usize, sc: &mut Scratch, out: &mut [u8]) {
    let scale = px as f64 / (2.0 * half); // pixels per world unit
    let x0 = cx - half;
    let y0 = cy - half;
    let img = &mut sc.img;
    img.clear();
    img.resize(px * px, BG);
    let b = &s.border;
    sc.colin.clear();
    sc.colin.extend((0..px).map(|i| {
        let wx = x0 + (i as f64 + 0.5) / scale;
        wx >= b.left && wx <= b.right
    }));
    for j in 0..px {
        let wy = y0 + (j as f64 + 0.5) / scale;
        let row = &mut img[j * px..(j + 1) * px];
        if wy < b.top || wy > b.bottom {
            row.fill(OUTSIDE);
        } else {
            for (v, &inside) in row.iter_mut().zip(&sc.colin) {
                if !inside {
                    *v = OUTSIDE;
                }
            }
        }
    }
    // small first, as the client draws
    let order = &mut sc.order;
    order.clear();
    order.extend(s.others.iter().enumerate().map(|(i, c)| (c.size as f32, false, i)));
    order.extend(s.mine.iter().enumerate().map(|(i, c)| (c.size as f32, true, i)));
    order.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let pxf = px as f32;
    for &(_, mine, idx) in order.iter() {
        let c = if mine { &s.mine[idx] } else { &s.others[idx] };
        let r = (c.size * scale) as f32;
        let (fx, fy) = (((c.x - x0) * scale) as f32, ((c.y - y0) * scale) as f32);
        let (fill, edge, virus) = match c.kind {
            Kind::Food | Kind::Ejected => (PELLET, PELLET, false),
            Kind::Virus => (BG + 25.0, VIRUS, true),
            _ if mine => (MINE, 200.0, false),
            _ => {
                let s = shade(c.color);
                (s, s - 45.0, false)
            }
        };
        if !virus && r < 0.8 {
            // sub-pixel disc: its area, split bilinearly over the 4 nearest pixels
            let cov = (std::f32::consts::PI * r * r).min(1.0);
            let (gx, gy) = (fx - 0.5, fy - 0.5);
            let (ix, iy) = (gx.floor(), gy.floor());
            let (tx, ty) = (gx - ix, gy - iy);
            for (di, dj, wt) in [(0, 0, (1.0 - tx) * (1.0 - ty)), (1, 0, tx * (1.0 - ty)), (0, 1, (1.0 - tx) * ty), (1, 1, tx * ty)] {
                let (i, j) = (ix + di as f32, iy + dj as f32);
                if i >= 0.0 && j >= 0.0 && i < pxf && j < pxf {
                    let p = &mut img[j as usize * px + i as usize];
                    let a = cov * wt;
                    *p = *p * (1.0 - a) + fill * a;
                }
            }
            continue;
        }
        let edge_w = if virus { (r * 0.18).max(1.0) } else { (r * 0.06).max(0.6) };
        let rmax = r * if virus { 1.08 } else { 1.0 } + 1.0;
        let i0 = (fx - rmax).floor().max(0.0) as usize;
        let i1 = (fx + rmax).ceil().clamp(0.0, pxf) as usize;
        let j0 = (fy - rmax).floor().max(0.0) as usize;
        let j1 = (fy + rmax).ceil().clamp(0.0, pxf) as usize;
        let r_in = r - edge_w - 0.75; // pixels whose centre is this close (or closer) are entirely fill
        let r_out = if virus { r * 1.07 } else { r } + 0.75; // ... and entirely outside beyond this
        for j in j0..j1 {
            let cy_ = j as f32 + 0.5 - fy;
            for i in i0..i1 {
                let cx_ = i as f32 + 0.5 - fx;
                let d2 = cx_ * cx_ + cy_ * cy_;
                let p = &mut img[j * px + i];
                if r_in > 0.0 && d2 <= r_in * r_in {
                    *p = fill;
                    continue;
                }
                if d2 >= r_out * r_out {
                    continue;
                }
                let (mut n_fill, mut n_edge) = (0u32, 0u32);
                for &sj in &SUB {
                    let dy = j as f32 + sj - fy;
                    for &si in &SUB {
                        let dx = i as f32 + si - fx;
                        let d = dx * dx + dy * dy;
                        // viruses: spiky outline (radius wobbles with the angle)
                        let rr = if virus { r * (1.0 + 0.07 * (18.0 * dy.atan2(dx)).cos()) } else { r };
                        if d <= rr * rr {
                            let ri = rr - edge_w;
                            if ri > 0.0 && d <= ri * ri {
                                n_fill += 1;
                            } else {
                                n_edge += 1;
                            }
                        }
                    }
                }
                let cov = (n_fill + n_edge) as f32 / 16.0;
                if cov > 0.0 {
                    *p = *p * (1.0 - cov) + (n_fill as f32 * fill + n_edge as f32 * edge) / 16.0;
                }
            }
        }
    }
    for (o, v) in out.iter_mut().zip(img.iter()) {
        *o = (v + 0.5).clamp(0.0, 255.0) as u8;
    }
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
