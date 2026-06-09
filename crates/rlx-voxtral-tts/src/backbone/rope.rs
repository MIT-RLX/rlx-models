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

//! Mistral/Ministral RoPE tables (eager CPU).

pub fn build_inv_freq(rope_theta: f64, head_dim: usize) -> Vec<f64> {
    (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / rope_theta.powf(i as f64 / head_dim as f64))
        .collect()
}

pub fn build_rope_tables(inv_freq: &[f64], max_pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let mut cos = vec![0f32; max_pos * half];
    let mut sin = vec![0f32; max_pos * half];
    for pos in 0..max_pos {
        for (i, &freq) in inv_freq.iter().enumerate() {
            let ang = pos as f64 * freq;
            cos[pos * half + i] = ang.cos() as f32;
            sin[pos * half + i] = ang.sin() as f32;
        }
    }
    (cos, sin)
}

/// Apply RoPE to Q/K rows at absolute positions `[start_pos, start_pos + seq)`.
pub fn apply_rope_qk(
    q: &mut ndarray::Array2<f32>,
    k: &mut ndarray::Array2<f32>,
    cos: &[f32],
    sin: &[f32],
    start_pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    let half = head_dim / 2;
    let seq = q.dim().0;
    for ti in 0..seq {
        let pos = start_pos + ti;
        for hi in 0..n_heads {
            rotate_row(q, ti, hi * head_dim, cos, sin, pos, half);
        }
        for hi in 0..n_kv_heads {
            rotate_row(k, ti, hi * head_dim, cos, sin, pos, half);
        }
    }
}

fn rotate_row(
    x: &mut ndarray::Array2<f32>,
    row: usize,
    col_off: usize,
    cos: &[f32],
    sin: &[f32],
    pos: usize,
    half: usize,
) {
    for i in 0..half {
        let c = cos[pos * half + i];
        let s = sin[pos * half + i];
        let x0 = x[[row, col_off + i]];
        let x1 = x[[row, col_off + half + i]];
        x[[row, col_off + i]] = x0 * c - x1 * s;
        x[[row, col_off + half + i]] = x0 * s + x1 * c;
    }
}
