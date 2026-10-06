//! Cell table + MultiOgar (protocol 6) packet parsing, with the web client's 120 ms interpolation.

use std::collections::HashMap;

pub const LERP_SECS: f64 = 0.12;
pub const FADE_SECS: f64 = 0.2;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Food,
    Virus,
    Ejected,
    Player,
}

#[derive(Clone, Debug)]
pub struct Cell {
    pub id: u32,
    pub kind: Kind,
    pub color: [u8; 3],
    pub name: String,
    // interpolation: old (o*), new (n*), current (x, y, s)
    pub ox: f64,
    pub oy: f64,
    pub os: f64,
    pub nx: f64,
    pub ny: f64,
    pub ns: f64,
    pub x: f64,
    pub y: f64,
    pub s: f64,
    pub updated: f64,
    pub dead_at: Option<f64>,
    pub died_by: u32,
}

impl Cell {
    pub fn mass(&self) -> f64 {
        self.ns * self.ns / 100.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Border {
    pub left: f64,
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
}

pub struct LbItem {
    pub me: bool,
    pub name: String,
}

pub struct World {
    /// interpolation window in seconds (settings: smoothing)
    pub lerp: f64,
    pub cells: HashMap<u32, Cell>,
    pub mine: Vec<u32>,
    pub border: Option<Border>,
    pub leaderboard: Vec<LbItem>,
    /// spectator camera target from packet 0x11 (x, y, zoom)
    pub spec_target: Option<(f64, f64, f64)>,
    pub packets: u64,
    pub updates: u64,
    pub eats: u64,
}

struct Rd<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Rd<'a> {
    fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let r = self.b.get(self.i..self.i + N)?.try_into().ok()?;
        self.i += N;
        Some(r)
    }
    fn u8(&mut self) -> Option<u8> {
        self.take::<1>().map(|a| a[0])
    }
    fn u16(&mut self) -> Option<u16> {
        self.take().map(u16::from_le_bytes)
    }
    fn u32(&mut self) -> Option<u32> {
        self.take().map(u32::from_le_bytes)
    }
    fn i32(&mut self) -> Option<i32> {
        self.take().map(i32::from_le_bytes)
    }
    fn f32(&mut self) -> Option<f32> {
        self.take().map(f32::from_le_bytes)
    }
    fn f64(&mut self) -> Option<f64> {
        self.take().map(f64::from_le_bytes)
    }
    fn string(&mut self) -> Option<String> {
        let rest = self.b.get(self.i..)?;
        let n = rest.iter().position(|&c| c == 0)?;
        let s = String::from_utf8_lossy(&rest[..n]).into_owned();
        self.i += n + 1;
        Some(s)
    }
}

impl World {
    pub fn new() -> World {
        World {
            lerp: LERP_SECS,
            cells: HashMap::new(),
            mine: Vec::new(),
            border: None,
            leaderboard: Vec::new(),
            spec_target: None,
            packets: 0,
            updates: 0,
            eats: 0,
        }
    }

    pub fn reset(&mut self) {
        self.cells.clear();
        self.mine.clear();
        self.leaderboard.clear();
        self.border = None;
        self.spec_target = None;
    }

    /// Apply one server packet. `now` is the receive time in seconds.
    pub fn handle(&mut self, buf: &[u8], now: f64) -> Option<()> {
        self.packets += 1;
        let mut r = Rd { b: buf, i: 1 };
        match *buf.first()? {
            0x10 => self.on_update(&mut r, now),
            0x11 => {
                self.spec_target = Some((r.f32()? as f64, r.f32()? as f64, r.f32()? as f64));
                Some(())
            }
            0x12 => {
                self.cells.clear();
                self.mine.clear();
                Some(())
            }
            0x14 => {
                self.mine.clear();
                Some(())
            }
            0x20 => {
                self.mine.push(r.u32()?);
                Some(())
            }
            0x30 => {
                let n = r.u32()?;
                self.leaderboard.clear();
                for _ in 0..n {
                    self.leaderboard.push(LbItem { me: false, name: r.string()? });
                }
                Some(())
            }
            0x31 => {
                let n = r.u32()?;
                self.leaderboard.clear();
                for _ in 0..n {
                    let me = r.u32()? != 0;
                    let name = r.string()?;
                    self.leaderboard.push(LbItem { me, name: if name.is_empty() { "An unnamed cell".into() } else { name } });
                }
                Some(())
            }
            0x40 => {
                self.border = Some(Border { left: r.f64()?, top: r.f64()?, right: r.f64()?, bottom: r.f64()? });
                Some(())
            }
            _ => Some(()),
        }
    }

    fn destroy(&mut self, id: u32, killer: u32, now: f64) {
        if let Some(c) = self.cells.get_mut(&id) {
            if c.dead_at.is_none() {
                c.dead_at = Some(now);
                c.died_by = killer;
            }
        }
        self.mine.retain(|&m| m != id);
    }

    fn on_update(&mut self, r: &mut Rd, now: f64) -> Option<()> {
        self.updates += 1;
        let lerp = self.lerp;
        let n = r.u16()?;
        for _ in 0..n {
            let killer = r.u32()?;
            let killed = r.u32()?;
            if self.cells.contains_key(&killer) && self.cells.contains_key(&killed) {
                self.eats += 1;
                self.destroy(killed, killer, now);
            }
        }
        loop {
            let id = r.u32()?;
            if id == 0 {
                break;
            }
            let x = r.i32()? as f64;
            let y = r.i32()? as f64;
            let s = r.u16()? as f64;
            let flags = r.u8()?;
            let color = if flags & 0x02 != 0 { Some([r.u8()?, r.u8()?, r.u8()?]) } else { None };
            if flags & 0x04 != 0 {
                r.string()?; // skin, unused
            }
            let name = if flags & 0x08 != 0 { Some(r.string()?) } else { None };
            match self.cells.get_mut(&id) {
                Some(c) => {
                    interpolate(c, now, lerp);
                    c.ox = c.x;
                    c.oy = c.y;
                    c.os = c.s;
                    c.nx = x;
                    c.ny = y;
                    c.ns = s;
                    c.updated = now;
                    if let Some(col) = color {
                        c.color = col;
                    }
                    if let Some(nm) = name {
                        c.name = nm;
                    }
                }
                None => {
                    let kind = if flags & 0x80 != 0 {
                        Kind::Food
                    } else if flags & 0x11 != 0 {
                        Kind::Virus
                    } else if flags & 0x20 != 0 {
                        Kind::Ejected
                    } else {
                        Kind::Player
                    };
                    self.cells.insert(
                        id,
                        Cell {
                            id,
                            kind,
                            color: color.unwrap_or([200, 200, 200]),
                            name: name.unwrap_or_default(),
                            ox: x,
                            oy: y,
                            os: s,
                            nx: x,
                            ny: y,
                            ns: s,
                            x,
                            y,
                            s,
                            updated: now,
                            dead_at: None,
                            died_by: 0,
                        },
                    );
                }
            }
        }
        let n = r.u16()?;
        for _ in 0..n {
            let id = r.u32()?;
            self.destroy(id, 0, now);
        }
        Some(())
    }

    /// Advance interpolation to `now` and drop faded-out cells.
    pub fn tick(&mut self, now: f64) {
        let lerp = self.lerp;
        let killers: HashMap<u32, (f64, f64)> = self
            .cells
            .values()
            .filter(|c| c.died_by != 0)
            .filter_map(|c| self.cells.get(&c.died_by).map(|k| (c.id, (k.x, k.y))))
            .collect();
        self.cells.retain(|_, c| c.dead_at.map_or(true, |d| now - d <= FADE_SECS));
        for c in self.cells.values_mut() {
            if let Some(&(kx, ky)) = killers.get(&c.id) {
                c.nx = kx;
                c.ny = ky;
            }
            interpolate(c, now, lerp);
        }
    }
}

fn interpolate(c: &mut Cell, now: f64, lerp: f64) {
    let dt = ((now - c.updated) / lerp).clamp(0.0, 1.0);
    c.x = c.ox + (c.nx - c.ox) * dt;
    c.y = c.oy + (c.ny - c.oy) * dt;
    c.s = c.os + (c.ns - c.os) * dt;
}
