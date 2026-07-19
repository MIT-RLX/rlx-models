//! Tiny dependency-free deterministic PRNG (SplitMix64).
//!
//! Data generation must be exactly reproducible from a seed so datasets can be
//! regenerated bit-for-bit in CI without vendoring the output. We deliberately
//! avoid the `rand` crate to keep this crate pure-`std` (fast compile, zero
//! resolution risk inside the large workspace).

/// SplitMix64 state. Cheap, well-distributed, and seedable from any `u64`
/// (including 0). Not cryptographic — this is a data-shuffling RNG.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, n)`. Returns 0 when `n == 0`.
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// Uniform in `[lo, hi)`. Caller must pass `hi > lo`.
    #[inline]
    pub fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo)
    }

    /// A float in `[0, 1)`.
    #[inline]
    pub fn frac(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// True with probability `p`.
    #[inline]
    pub fn chance(&mut self, p: f64) -> bool {
        self.frac() < p
    }

    /// Borrow a uniformly-random element.
    #[inline]
    pub fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }

    /// Pick a `&'static str` from a static slice (the common case here).
    #[inline]
    pub fn pick_str(&mut self, xs: &[&'static str]) -> &'static str {
        xs[self.below(xs.len())]
    }

    /// In-place Fisher–Yates shuffle.
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.below(i + 1);
            xs.swap(i, j);
        }
    }
}
