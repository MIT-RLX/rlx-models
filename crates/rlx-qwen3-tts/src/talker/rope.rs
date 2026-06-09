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

//! RoPE tables for talker (`rope_theta` = 1e6).

pub fn build_inv_freq(head_dim: usize, rope_theta: f64) -> Vec<f64> {
    let half = head_dim / 2;
    (0..half)
        .map(|i| {
            let exp = (2 * i) as f64 / head_dim as f64;
            1.0 / rope_theta.powf(exp)
        })
        .collect()
}

pub fn rope_slice(inv_freq: &[f64], position: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0f32; half];
    let mut sin = vec![0f32; half];
    rope_slice_into(inv_freq, position, head_dim, &mut cos, &mut sin);
    (cos, sin)
}

/// Write single-position cos/sin into caller buffers (`len == head_dim / 2`).
pub fn rope_slice_into(
    inv_freq: &[f64],
    position: usize,
    head_dim: usize,
    cos: &mut [f32],
    sin: &mut [f32],
) {
    let half = head_dim / 2;
    debug_assert_eq!(cos.len(), half);
    debug_assert_eq!(sin.len(), half);
    for i in 0..half {
        let angle = position as f64 * inv_freq[i];
        cos[i] = angle.cos() as f32;
        sin[i] = angle.sin() as f32;
    }
}

/// Flattened `[seq * head_half]` cos/sin for one prefill step (per-token absolute positions).
pub fn rope_prefill_feeds(
    inv_freq: &[f64],
    positions: &[usize],
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0f32; positions.len() * half];
    let mut sin = vec![0f32; positions.len() * half];
    for (t, &pos) in positions.iter().enumerate() {
        let (c, s) = rope_slice(inv_freq, pos, head_dim);
        let off = t * half;
        cos[off..off + half].copy_from_slice(&c);
        sin[off..off + half].copy_from_slice(&s);
    }
    (cos, sin)
}

/// Full `[max_pos * head_half]` tables for `RopeTablesStage::param` gather.
pub fn rope_tables_full(inv_freq: &[f64], max_pos: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let positions: Vec<usize> = (0..max_pos).collect();
    rope_prefill_feeds(inv_freq, &positions, head_dim)
}
