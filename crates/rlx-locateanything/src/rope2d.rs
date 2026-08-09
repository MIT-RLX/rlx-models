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

//! 2D rotary position embeddings for MoonViT (VisionLLaMA-style).

use crate::config::MoonVitConfig;

const ROPE_THETA: f64 = 10_000.0;

/// Complex cis table for one spatial position: `[head_dim/2]` as (re, im) pairs flattened to `head_dim`.
pub fn freqs_cis_for_grid(
    cfg: &MoonVitConfig,
    grid_h: usize,
    grid_w: usize,
    device_theta: f64,
) -> Vec<f32> {
    let head_dim = cfg.head_dim();
    assert!(head_dim.is_multiple_of(4));
    let dim_quarter = head_dim / 4;
    let max_h = cfg.init_pos_emb_height;
    let max_w = cfg.init_pos_emb_width;

    let mut table = vec![0f32; max_h * max_w * (head_dim / 2) * 2];
    for y in 0..max_h {
        for x in 0..max_w {
            let flat = y * max_w + x;
            for i in 0..dim_quarter {
                let freq = 1.0 / device_theta.powf((4 * i) as f64 / head_dim as f64);
                let x_angle = x as f64 * freq;
                let y_angle = y as f64 * freq;
                let base = flat * head_dim;
                table[base + 2 * i] = x_angle.cos() as f32;
                table[base + 2 * i + 1] = x_angle.sin() as f32;
                table[base + head_dim / 2 + 2 * i] = y_angle.cos() as f32;
                table[base + head_dim / 2 + 2 * i + 1] = y_angle.sin() as f32;
            }
        }
    }

    let mut out = Vec::with_capacity(grid_h * grid_w * head_dim);
    for y in 0..grid_h {
        for x in 0..grid_w {
            let src = (y * max_w + x) * head_dim;
            out.extend_from_slice(&table[src..src + head_dim]);
        }
    }
    out
}

/// Cos/sin tables for `HirGraphExt::rope` on the X and Y halves of head_dim (2D RoPE).
pub fn rope_cos_sin_halves_for_grid(
    cfg: &MoonVitConfig,
    grid_h: usize,
    grid_w: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let freqs = freqs_cis_for_grid(cfg, grid_h, grid_w, ROPE_THETA);
    let seq = grid_h * grid_w;
    let dh = cfg.head_dim();
    let quarter = dh / 4;
    let mut cos_x = vec![0f32; seq * quarter];
    let mut sin_x = vec![0f32; seq * quarter];
    let mut cos_y = vec![0f32; seq * quarter];
    let mut sin_y = vec![0f32; seq * quarter];
    for t in 0..seq {
        let base = t * dh;
        let ox = t * quarter;
        let oy = t * quarter;
        for i in 0..quarter {
            cos_x[ox + i] = freqs[base + 2 * i];
            sin_x[ox + i] = freqs[base + 2 * i + 1];
            cos_y[oy + i] = freqs[base + dh / 2 + 2 * i];
            sin_y[oy + i] = freqs[base + dh / 2 + 2 * i + 1];
        }
    }
    (cos_x, sin_x, cos_y, sin_y)
}

/// Apply 2D RoPE to Q and K. `q`, `k`: `[seq, heads, head_dim]`, `freqs`: `[seq, head_dim]`.
pub fn apply_rope_2d(
    q: &mut [f32],
    k: &mut [f32],
    freqs: &[f32],
    seq: usize,
    heads: usize,
    head_dim: usize,
) {
    let half = head_dim / 2;
    for t in 0..seq {
        let f_base = t * head_dim;
        for h in 0..heads {
            let q_base = (t * heads + h) * head_dim;
            let k_base = q_base;
            for i in 0..half / 2 {
                let q0 = q[q_base + 2 * i];
                let q1 = q[q_base + 2 * i + 1];
                let c = freqs[f_base + 2 * i];
                let s = freqs[f_base + 2 * i + 1];
                q[q_base + 2 * i] = q0 * c - q1 * s;
                q[q_base + 2 * i + 1] = q0 * s + q1 * c;

                let k0 = k[k_base + 2 * i];
                let k1 = k[k_base + 2 * i + 1];
                k[k_base + 2 * i] = k0 * c - k1 * s;
                k[k_base + 2 * i + 1] = k0 * s + k1 * c;
            }
            for i in 0..half / 2 {
                let idx = half + 2 * i;
                let q0 = q[q_base + idx];
                let q1 = q[q_base + idx + 1];
                let c = freqs[f_base + idx];
                let s = freqs[f_base + idx + 1];
                q[q_base + idx] = q0 * c - q1 * s;
                q[q_base + idx + 1] = q0 * s + q1 * c;

                let k0 = k[k_base + idx];
                let k1 = k[k_base + idx + 1];
                k[k_base + idx] = k0 * c - k1 * s;
                k[k_base + idx + 1] = k0 * s + k1 * c;
            }
        }
    }
}
