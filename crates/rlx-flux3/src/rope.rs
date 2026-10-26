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

//! Token position ids `(t, h, w, l)` and 4-axis RoPE tables.
//!
//! Image latents sit at `(0, row·s, col·s, 0)` with `s` the image RoPE scale
//! (0.5); the `n`-th reference image at `(0, row·s, col·s, −(1+n))`; text at
//! `(0, 0, 0, position)`. Angles follow BFL `rope`: frequencies in f64,
//! positions as stored (f32), cos/sin taken in f64.

/// Position ids of a `h × w` latent grid in row-major order, scaled like the
/// reference's `scaled_ids`, on axis `l = layer`.
pub fn latent_ids(h: usize, w: usize, scale: f32, layer: f32) -> Vec<[f32; 4]> {
    let mut ids = Vec::with_capacity(h * w);
    for r in 0..h {
        for c in 0..w {
            ids.push([0.0, r as f32 * scale, c as f32 * scale, layer]);
        }
    }
    ids
}

/// Position ids of `len` text tokens.
pub fn text_ids(len: usize) -> Vec<[f32; 4]> {
    (0..len).map(|p| [0.0, 0.0, 0.0, p as f32]).collect()
}

/// `(cos, sin)` tables `[ids.len(), head_dim / 2]`, axis 0's frequencies
/// first. The pair order is the de-interleaved one
/// ([`crate::weights::DitSource`]), so row `j` rotates channels `j` and
/// `j + head_dim / 2`.
pub fn rope_tables(ids: &[[f32; 4]], axes_dim: [usize; 4], theta: f64) -> (Vec<f64>, Vec<f64>) {
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
            let angle = id[axis_of[j]] as f64 * omega[j];
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axes_take_consecutive_frequency_bands() {
        let ids = [[0.0, 1.5, 2.0, -1.0]];
        let (cos, sin) = rope_tables(&ids, [32, 32, 32, 32], 10_000.0);
        assert_eq!(cos.len(), 64);
        // Axis 0 (t = 0) is unrotated.
        assert!(sin[..16].iter().all(|&s| s == 0.0));
        // Axis 1 band starts at 16 with omega = 1: angle = h.
        assert!((sin[16] - 1.5f64.sin()).abs() < 1e-15);
        assert!((cos[32] - 2.0f64.cos()).abs() < 1e-15);
        assert!((sin[48] - (-1.0f64).sin()).abs() < 1e-15);
        // Second frequency of an axis: theta^(-2/32).
        let w = 1.0 / 10_000f64.powf(2.0 / 32.0);
        assert!((sin[17] - (1.5 * w).sin()).abs() < 1e-15);
    }

    #[test]
    fn latent_ids_scale_rows_and_columns() {
        let ids = latent_ids(2, 3, 0.5, -2.0);
        assert_eq!(ids[4], [0.0, 0.5, 0.5, -2.0]);
        assert_eq!(text_ids(3)[2], [0.0, 0.0, 0.0, 2.0]);
    }
}
