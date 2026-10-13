//! Cells and snapshots (port of bots/specs/game.py).

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Food,
    Virus,
    Ejected,
    Player,
    Mine,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Food => "food",
            Kind::Virus => "virus",
            Kind::Ejected => "ejected",
            Kind::Player => "player",
            Kind::Mine => "mine",
        }
    }

    pub fn parse(s: &str) -> Kind {
        match s {
            "food" => Kind::Food,
            "virus" => Kind::Virus,
            "ejected" => Kind::Ejected,
            "mine" => Kind::Mine,
            _ => Kind::Player,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Cell {
    pub id: u32,
    pub x: f64,
    pub y: f64,
    /// radius in world units
    pub size: f64,
    pub kind: Kind,
    pub vx: f64,
    pub vy: f64,
    /// protocol colour (r<<16|g<<8|b): one player's pieces share it; 0 = unknown
    pub color: u32,
}

impl Cell {
    pub fn new(id: u32, x: f64, y: f64, size: f64, kind: Kind) -> Cell {
        Cell { id, x, y, size, kind, vx: 0.0, vy: 0.0, color: 0 }
    }

    #[inline]
    pub fn mass(&self) -> f64 {
        self.size * self.size / 100.0
    }

    #[inline]
    pub fn dist(&self, o: &Cell) -> f64 {
        (self.x - o.x).hypot(self.y - o.y)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Border {
    pub left: f64,
    pub right: f64,
    pub top: f64,
    pub bottom: f64,
}

#[derive(Clone, Debug)]
pub struct Snapshot {
    pub mine: Vec<Cell>,
    pub others: Vec<Cell>,
    pub border: Border,
    pub dead: bool,
}

impl Snapshot {
    /// Our biggest cell (first one on ties, like Python's max).
    pub fn main(&self) -> Option<&Cell> {
        let mut best: Option<&Cell> = None;
        for c in &self.mine {
            if best.map_or(true, |b| c.size > b.size) {
                best = Some(c);
            }
        }
        best
    }

    pub fn total_mass(&self) -> f64 {
        self.mine.iter().map(|c| c.mass()).sum()
    }
}
