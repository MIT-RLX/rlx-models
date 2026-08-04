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

//! A small, seedable RNG for reproducible diffusion/flow sampling.
//!
//! This is a `SplitMix64` core with Box–Muller Gaussian sampling. It is
//! deterministic given a seed — the property that matters for reproducible
//! generation and for comparing two RLX backends on the *same* noise. It is
//! **not** bit-compatible with PyTorch's Philox/MT19937; matching a reference
//! implementation's exact noise stream (needed for tensor-level parity against a
//! Python checkpoint) is a separate, model-specific concern, and such models
//! should instead be handed pre-generated reference noise.

use core::f32::consts::PI;

/// SplitMix64 — a fast, well-distributed 64-bit generator used as the core.
#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// A seedable RNG producing uniform and standard-normal `f32` samples.
#[derive(Debug, Clone)]
pub struct Rng {
    core: SplitMix64,
    /// Box–Muller produces normals in pairs; the second is cached here.
    spare_normal: Option<f32>,
}

impl Rng {
    /// Construct from a 64-bit seed. Same seed ⇒ same stream.
    pub fn seeded(seed: u64) -> Self {
        Self {
            core: SplitMix64::new(seed),
            spare_normal: None,
        }
    }

    /// Uniform `f32` in `[0, 1)` with 24 bits of mantissa precision.
    pub fn next_uniform(&mut self) -> f32 {
        // Top 24 bits → [0, 2^24) → scaled into [0, 1).
        (self.core.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// A single standard-normal (mean 0, variance 1) sample via Box–Muller.
    pub fn standard_normal(&mut self) -> f32 {
        if let Some(s) = self.spare_normal.take() {
            return s;
        }
        // Guard the log against exactly 0.
        let u1 = self.next_uniform().max(1.0e-7);
        let u2 = self.next_uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        self.spare_normal = Some(r * theta.sin());
        r * theta.cos()
    }

    /// Fill `buf` with standard-normal samples (the usual "initial latent noise").
    pub fn fill_standard_normal(&mut self, buf: &mut [f32]) {
        for x in buf.iter_mut() {
            *x = self.standard_normal();
        }
    }

    /// Allocate `n` standard-normal samples.
    pub fn standard_normal_vec(&mut self, n: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; n];
        self.fill_standard_normal(&mut v);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = Rng::seeded(42);
        let mut b = Rng::seeded(42);
        for _ in 0..64 {
            assert_eq!(a.next_uniform().to_bits(), b.next_uniform().to_bits());
        }
    }

    #[test]
    fn different_seed_diverges() {
        let mut a = Rng::seeded(1);
        let mut b = Rng::seeded(2);
        // Overwhelmingly likely to differ within a few draws.
        let differ = (0..8).any(|_| a.next_uniform() != b.next_uniform());
        assert!(differ);
    }

    #[test]
    fn uniform_in_unit_interval() {
        let mut r = Rng::seeded(7);
        for _ in 0..10_000 {
            let u = r.next_uniform();
            assert!((0.0..1.0).contains(&u), "u={u}");
        }
    }

    #[test]
    fn normal_has_unit_moments() {
        let mut r = Rng::seeded(123);
        let n = 200_000;
        let mut mean = 0.0f64;
        let mut m2 = 0.0f64;
        for _ in 0..n {
            let x = r.standard_normal() as f64;
            mean += x;
            m2 += x * x;
        }
        mean /= n as f64;
        let var = m2 / n as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "mean={mean}");
        assert!((var - 1.0).abs() < 0.05, "var={var}");
    }
}
