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

//! The one Gaussian source in the crate (xorshift64\* + Box–Muller).
//!
//! TADA draws noise in two independent places — the codec encoder's stochastic
//! bottleneck when a voice prompt is built, and the initial state of each
//! token's flow-matching solve. Both want the same thing: a correctly scaled
//! standard normal that is *reproducible*, so the same seed gives the same
//! audio.
//!
//! Matching torch's Philox bit-for-bit is explicitly not a goal. The solve's
//! noise is the start of an ODE that contracts toward the conditional
//! distribution, and the encoder's is a trained-in bottleneck — any correctly
//! scaled Gaussian is a valid draw from either. Reproducibility across runs is
//! the property that actually matters, and it is the one this guarantees.

/// Seeded standard-normal stream.
///
/// Box–Muller yields two independent normals per pair of uniforms; the second
/// is held in `spare` so no draw is wasted.
pub struct Normal {
    state: u64,
    spare: Option<f32>,
}

impl Normal {
    pub fn new(seed: u64) -> Self {
        Self {
            // A zero state is a fixed point of xorshift; force it off.
            state: seed | 1,
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `(0, 1]` — Box–Muller takes a log, so zero must be excluded.
    fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 11) as f32 + 1.0) / (1u64 << 53) as f32
    }

    /// One draw from `N(0, 1)`.
    pub fn sample(&mut self) -> f32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let u1 = self.uniform();
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }

    /// Fill `out` with `N(0, scale²)` draws.
    pub fn fill(&mut self, out: &mut [f32], scale: f32) {
        for slot in out.iter_mut() {
            *slot = self.sample() * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moments(v: &[f32]) -> (f32, f32) {
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
        (mean, var)
    }

    #[test]
    fn is_standard_normal() {
        let mut r = Normal::new(11);
        let v: Vec<f32> = (0..8192).map(|_| r.sample()).collect();
        let (mean, var) = moments(&v);
        assert!(mean.abs() < 0.04, "mean {mean}");
        assert!((var - 1.0).abs() < 0.06, "variance {var}");
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn the_same_seed_gives_the_same_stream() {
        let (mut a, mut b) = (Normal::new(42), Normal::new(42));
        let xa: Vec<f32> = (0..256).map(|_| a.sample()).collect();
        let xb: Vec<f32> = (0..256).map(|_| b.sample()).collect();
        assert_eq!(xa, xb);
    }

    #[test]
    fn different_seeds_diverge() {
        let (mut a, mut b) = (Normal::new(1), Normal::new(2));
        let xa: Vec<f32> = (0..64).map(|_| a.sample()).collect();
        let xb: Vec<f32> = (0..64).map(|_| b.sample()).collect();
        assert_ne!(xa, xb);
    }

    #[test]
    fn a_zero_seed_is_not_a_fixed_point() {
        let mut r = Normal::new(0);
        let v: Vec<f32> = (0..64).map(|_| r.sample()).collect();
        assert!(v.windows(2).any(|w| w[0] != w[1]), "stream is constant");
    }

    #[test]
    fn fill_applies_the_scale() {
        let mut r = Normal::new(7);
        let mut v = vec![0f32; 8192];
        r.fill(&mut v, 0.9);
        let (mean, var) = moments(&v);
        assert!((var.sqrt() - 0.9).abs() < 0.05, "sd {}", var.sqrt());
        assert!(mean.abs() < 0.05, "mean {mean}");
    }

    #[test]
    fn a_zero_scale_is_exactly_zero() {
        // The end-to-end parity suite relies on this: `noise_temperature = 0`
        // must put the ODE at the origin, not merely near it.
        let mut r = Normal::new(3);
        let mut v = vec![1f32; 32];
        r.fill(&mut v, 0.0);
        assert!(v.iter().all(|x| *x == 0.0));
    }
}
