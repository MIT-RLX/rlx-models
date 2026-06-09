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

//! ONNX `Random*Like` reference fills (vocoder source noise).
//!
//! Without `KITTEN_RLX_RNG_SEED`, kernels emit zeros (stable vs ORT stochastic vocoder).
//! With the env var set, fills use counter-based Philox (same family as RLX weight init).

use rlx_ir::Philox4x32;

fn seed_from_env() -> u64 {
    std::env::var("KITTEN_RLX_RNG_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42)
}

/// Box–Muller normal samples (mean, scale).
pub fn fill_normal(out: &mut [f32], mean: f32, scale: f32, seed: u64) {
    let mut rng = Philox4x32::new(seed);
    for v in out.iter_mut() {
        *v = mean + scale * rng.normal();
    }
}

pub fn fill_uniform(out: &mut [f32], low: f32, high: f32, seed: u64) {
    let mut rng = Philox4x32::new(seed);
    for v in out.iter_mut() {
        *v = rng.uniform(low, high);
    }
}

pub fn normal_seed(node_tag: u64) -> u64 {
    seed_from_env().wrapping_add(node_tag.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

pub fn uniform_seed(node_tag: u64) -> u64 {
    seed_from_env().wrapping_add(node_tag.wrapping_mul(0x517C_C1B7_2722_0A95))
}
