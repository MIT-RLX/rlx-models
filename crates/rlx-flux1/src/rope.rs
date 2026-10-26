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

//! FLUX.1 3-axis RoPE tables.
//!
//! Every token carries an id `(a0, a1, a2)`: text tokens `(0, 0, 0)`, image
//! patch `(r, c)` → `(0, r, c)`. Head dimension `d` is split into the axes'
//! widths (`[16, 56, 56]`), each rotated at `θ^(-2j/width)`. BFL computes the
//! angles in f64 and rounds the cos/sin to f32; so does this.
//!
//! Tables are laid out for the half-split kernel (`x[i]` with `x[i + d/2]`):
//! row `s` holds `d/2` angles, pair `j` in the original interleaved order —
//! which [`crate::weights::deinterleave_rows`] maps to exactly this order.

/// Position ids `[txt_len + h·w, 3]`, text first (the joint-attention order).
pub fn position_ids(txt_len: usize, h: usize, w: usize) -> Vec<[f64; 3]> {
    let mut ids = vec![[0.0; 3]; txt_len];
    for r in 0..h {
        for c in 0..w {
            ids.push([0.0, r as f64, c as f64]);
        }
    }
    ids
}

/// `(cos, sin)`, each `[ids.len(), head_dim / 2]`.
pub fn rope_tables(ids: &[[f64; 3]], axes_dim: [usize; 3], theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half: usize = axes_dim.iter().sum::<usize>() / 2;
    let mut omega = Vec::with_capacity(half);
    let mut axis_of = Vec::with_capacity(half);
    for (a, &width) in axes_dim.iter().enumerate() {
        for j in 0..width / 2 {
            let scale = (2 * j) as f64 / width as f64;
            omega.push(1.0 / theta.powf(scale));
            axis_of.push(a);
        }
    }
    let mut cos = Vec::with_capacity(ids.len() * half);
    let mut sin = Vec::with_capacity(ids.len() * half);
    for id in ids {
        for j in 0..half {
            let angle = id[axis_of[j]] * omega[j];
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_rows_are_identity_and_axes_split() {
        let ids = position_ids(2, 2, 3);
        assert_eq!(ids.len(), 8);
        let (cos, sin) = rope_tables(&ids, [16, 56, 56], 10_000.0);
        assert!(cos[..64].iter().all(|&c| c == 1.0) && sin[..64].iter().all(|&s| s == 0.0));
        // Image (r=1, c=2): axis-0 pairs stay unrotated, axis-1 pair 0 turns by r·1.
        let row = 2 + 3 + 2;
        assert_eq!(sin[row * 64], 0.0);
        assert!((sin[row * 64 + 8] - 1f64.sin() as f32).abs() < 1e-7);
        assert!((sin[row * 64 + 36] - 2f64.sin() as f32).abs() < 1e-7);
    }
}
