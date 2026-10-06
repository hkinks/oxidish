//! mulberry32, the same generator arena.js uses for Math.random (different consumption order, so
//! seeds do not reproduce Node games; they do make the Rust engine deterministic).

pub struct Rng(u32);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let s = if seed == 0 {
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)
        } else {
            seed
        };
        Rng((s ^ (s >> 32)) as u32)
    }

    #[inline]
    pub fn f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x6D2B79F5);
        let mut t = self.0;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        ((t ^ (t >> 14)) as f64) / 4294967296.0
    }

    /// floor(len * random)
    pub fn index(&mut self, len: usize) -> usize {
        ((len as f64 * self.f64()) as usize).min(len.saturating_sub(1))
    }
}
