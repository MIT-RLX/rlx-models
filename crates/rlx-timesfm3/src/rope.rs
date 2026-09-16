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

//! TimesFM RoPE cos/sin tables (matches `host/math.rs::apply_rope4` frequencies).

use ndarray::Array2;

const MIN_TIMESCALE: f32 = 1.0;
const MAX_TIMESCALE: f32 = 10_000.0;

fn timescales(head_dim: usize) -> Vec<f32> {
    let half = head_dim / 2;
    (0..half)
        .map(|i| {
            let fraction = 2.0 * i as f32 / head_dim as f32;
            MIN_TIMESCALE * (MAX_TIMESCALE / MIN_TIMESCALE).powf(fraction)
        })
        .collect()
}

/// Build `[batch, seq, head_dim/2]` cos/sin tables for graph inputs.
pub fn build_rope_tables(positions: &Array2<i32>, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let (batch, seq) = positions.dim();
    let half = head_dim / 2;
    let scales = timescales(head_dim);
    let mut cos = vec![0.0f32; batch * seq * half];
    let mut sin = vec![0.0f32; batch * seq * half];
    for bi in 0..batch {
        for si in 0..seq {
            let pos = positions[[bi, si]] as f32;
            let base = (bi * seq + si) * half;
            for d in 0..half {
                let angle = pos / scales[d];
                cos[base + d] = angle.cos();
                sin[base + d] = angle.sin();
            }
        }
    }
    (cos, sin)
}

/// Positions for sequence attention: `ni - front` where `front` is leading true mask count.
pub fn seq_positions(patch_mask: &Array2<bool>) -> Array2<i32> {
    let (batch, seq) = patch_mask.dim();
    let mut positions = Array2::<i32>::zeros((batch, seq));
    for bi in 0..batch {
        let front = patch_mask.row(bi).iter().take_while(|&&m| m).count() as i32;
        for si in 0..seq {
            positions[[bi, si]] = si as i32 - front;
        }
    }
    positions
}

/// Repeat `[batch, seq, half]` tables across `num_heads` for per-head RoPE.
pub fn repeat_heads(
    cos: &[f32],
    sin: &[f32],
    batch: usize,
    seq: usize,
    half: usize,
    heads: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut cos_out = vec![0.0f32; batch * heads * seq * half];
    let mut sin_out = vec![0.0f32; batch * heads * seq * half];
    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq {
                let src = (b * seq + s) * half;
                let dst = ((b * heads + h) * seq + s) * half;
                cos_out[dst..dst + half].copy_from_slice(&cos[src..src + half]);
                sin_out[dst..dst + half].copy_from_slice(&sin[src..src + half]);
            }
        }
    }
    (cos_out, sin_out)
}

/// Apply NeoX RoPE using precomputed `[rows, seq, head_dim/2]` tables.
pub fn apply_rope_tables(
    x: ndarray::ArrayView3<f32>,
    cos: &[f32],
    sin: &[f32],
    head_dim: usize,
) -> ndarray::Array3<f32> {
    let (rows, seq, hd) = x.dim();
    let half = head_dim / 2;
    assert_eq!(hd, head_dim);
    assert_eq!(cos.len(), rows * seq * half);
    assert_eq!(sin.len(), rows * seq * half);
    let mut out = x.to_owned();
    for r in 0..rows {
        for s in 0..seq {
            let base = (r * seq + s) * half;
            for d in 0..half {
                let a = out[[r, s, d]];
                let b = out[[r, s, d + half]];
                let c = cos[base + d];
                let sn = sin[base + d];
                out[[r, s, d]] = a * c - b * sn;
                out[[r, s, d + half]] = b * c + a * sn;
            }
        }
    }
    out
}

/// Multiplicative attention keep mask: `1.0` allowed, `0.0` masked.
#[allow(dead_code)]
pub fn build_attn_keep(patch_mask: &Array2<bool>, seq: usize, causal: bool) -> Vec<f32> {
    let batch = patch_mask.nrows();
    let mut keep = vec![0.0f32; batch * seq * seq];
    for bi in 0..batch {
        let front = patch_mask.row(bi).iter().take_while(|&&m| m).count();
        for qi in 0..seq {
            for ki in 0..seq {
                let ok = ki >= front && (!causal || qi >= ki);
                let masked = !ok || patch_mask[[bi, ki]];
                let idx = bi * seq * seq + qi * seq + ki;
                keep[idx] = if masked { 0.0 } else { 1.0 };
            }
        }
    }
    keep
}

/// Additive attention bias: `0` allowed, `-1e9` masked (bool mask true = masked out).
pub fn build_attn_bias(patch_mask: &Array2<bool>, seq: usize, causal: bool) -> Vec<f32> {
    let batch = patch_mask.nrows();
    let mut bias = vec![0.0f32; batch * seq * seq];
    for bi in 0..batch {
        let front = patch_mask.row(bi).iter().take_while(|&&m| m).count();
        for qi in 0..seq {
            for ki in 0..seq {
                let ok = ki >= front && (!causal || qi >= ki);
                let masked = !ok || patch_mask[[bi, ki]];
                let idx = bi * seq * seq + qi * seq + ki;
                bias[idx] = if masked { -1e9 } else { 0.0 };
            }
        }
    }
    bias
}
