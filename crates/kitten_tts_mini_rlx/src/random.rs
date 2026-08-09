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

//! Legacy custom-kernel RNG fills — delegates to upstream [`rlx_ir`] helpers.
//!
//! Prefer [`CompileOptions::rng`](rlx_runtime::CompileOptions::rng) with native
//! `Op::RngNormal` / `Op::RngUniform` when compiling; these remain for
//! `Op::Custom` GPU/CPU kernel registration fallback.

use rlx_ir::{RngOptions, fill_normal_like, fill_uniform_like};

pub use crate::bundle_compile::rng_options_from_env;

/// Box–Muller normal samples (mean, scale).
pub fn fill_normal(out: &mut [f32], mean: f32, scale: f32, seed: u64) {
    let opts = RngOptions::philox(seed);
    fill_normal_like(out, mean, scale, opts, 0, None);
}

pub fn fill_uniform(out: &mut [f32], low: f32, high: f32, seed: u64) {
    let opts = RngOptions::philox(seed);
    fill_uniform_like(out, low, high, opts, 0, None);
}

pub fn fill_normal_with_opts(
    out: &mut [f32],
    mean: f32,
    scale: f32,
    opts: RngOptions,
    node_tag: u64,
) {
    fill_normal_like(out, mean, scale, opts, node_tag, None);
}

pub fn fill_uniform_with_opts(
    out: &mut [f32],
    low: f32,
    high: f32,
    opts: RngOptions,
    node_tag: u64,
) {
    fill_uniform_like(out, low, high, opts, node_tag, None);
}

pub fn normal_seed(node_tag: u64) -> u64 {
    let opts = rng_options_from_env();
    rlx_ir::combine_seed(opts.seed, node_tag)
}

pub fn uniform_seed(node_tag: u64) -> u64 {
    normal_seed(node_tag.wrapping_add(0xD1B5_4A32_D192_ED03))
}
