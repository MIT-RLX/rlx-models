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

//! Multi-axis RoPE position embeddings for FLUX.2.

use super::config::Flux2Config;

/// `(cos, sin)` each `[total_seq, sum(axes_dims)]` for concatenated txt+img ids.
pub fn flux2_pos_embed(
    cfg: &Flux2Config,
    ids: &[f32],
    seq: usize,
    n_axes: usize,
) -> (Vec<f32>, Vec<f32>) {
    let dim_total: usize = cfg.axes_dims_rope.iter().sum();
    let mut cos = vec![0.0f32; seq * dim_total];
    let mut sin = vec![0.0f32; seq * dim_total];
    let mut offset = 0usize;
    for (axis_i, &axis_dim) in cfg.axes_dims_rope.iter().enumerate() {
        let pos: Vec<f32> = (0..seq).map(|t| ids[t * n_axes + axis_i]).collect();
        let (c, s) = rotary_1d(axis_dim, &pos, seq, cfg.rope_theta);
        for t in 0..seq {
            cos[t * dim_total + offset..t * dim_total + offset + axis_dim]
                .copy_from_slice(&c[t * axis_dim..(t + 1) * axis_dim]);
            sin[t * dim_total + offset..t * dim_total + offset + axis_dim]
                .copy_from_slice(&s[t * axis_dim..(t + 1) * axis_dim]);
        }
        offset += axis_dim;
    }
    (cos, sin)
}

fn rotary_1d(dim: usize, pos: &[f32], seq: usize, theta: usize) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let mut freq = vec![0.0f32; half];
    for i in 0..half {
        let exponent = (i as f32 * 2.0) / dim as f32;
        freq[i] = 1.0 / (theta as f32).powf(exponent);
    }
    let mut cos = vec![0.0f32; seq * dim];
    let mut sin = vec![0.0f32; seq * dim];
    for t in 0..seq {
        let p = pos[t];
        for i in 0..half {
            let angle = p * freq[i];
            let c = angle.cos();
            let s = angle.sin();
            cos[t * dim + 2 * i] = c;
            cos[t * dim + 2 * i + 1] = c;
            sin[t * dim + 2 * i] = s;
            sin[t * dim + 2 * i + 1] = s;
        }
    }
    (cos, sin)
}

/// Apply interleaved real RoPE to Q/K head tensors `[batch*seq, heads, head_dim]`.
pub fn apply_flux2_qk_rope(
    q: &mut [f32],
    k: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    rope_dim: usize,
) {
    let bs = batch * seq;
    for idx in 0..bs {
        let t = idx % seq;
        let cos_row = &cos[t * rope_dim..(t + 1) * rope_dim];
        let sin_row = &sin[t * rope_dim..(t + 1) * rope_dim];
        for h in 0..heads {
            let base = idx * heads * head_dim + h * head_dim;
            apply_rope_row(&mut q[base..base + head_dim], cos_row, sin_row, rope_dim);
            apply_rope_row(&mut k[base..base + head_dim], cos_row, sin_row, rope_dim);
        }
    }
}

fn apply_rope_row(x: &mut [f32], cos: &[f32], sin: &[f32], rope_dim: usize) {
    let dim = x.len().min(rope_dim).min(cos.len()).min(sin.len());
    let pairs = dim / 2;
    let mut rotated = vec![0.0f32; dim];
    for i in 0..pairs {
        let xr = x[2 * i];
        let xi = x[2 * i + 1];
        rotated[2 * i] = -xi;
        rotated[2 * i + 1] = xr;
    }
    for d in 0..dim {
        x[d] = x[d] * cos[d] + rotated[d] * sin[d];
    }
}
