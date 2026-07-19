// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Seeded Gaussian noise for flow-matching stages.
//!
//! Uses SplitMix64 + Box–Muller. This is **not** bit-identical to
//! `torch.randn` (Philox); it is a deterministic host substitute for the
//! pipeline API and small regression tests.

/// SplitMix64 PRNG (https://prng.di.unimi.it/splitmix64.c).
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9e37_79b9_7f4a_7c15),
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `(0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        let u = self.next_u64() >> 11; // 53 bits → keep top of u64
        (u as f64 / ((1u64 << 53) as f64)).clamp(f64::EPSILON, 1.0 - f64::EPSILON) as f32
    }

    /// Standard normal via Box–Muller.
    pub fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32();
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    pub fn fill_gaussian(&mut self, out: &mut [f32]) {
        for v in out.iter_mut() {
            *v = self.next_gaussian();
        }
    }

    pub fn gaussian_vec(&mut self, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        self.fill_gaussian(&mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_is_deterministic() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        let va = a.gaussian_vec(64);
        let vb = b.gaussian_vec(64);
        assert_eq!(va, vb);
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(2);
        assert_ne!(a.gaussian_vec(32), b.gaussian_vec(32));
    }
}
