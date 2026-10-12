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

//! Everything the tower needs computed on the host: the image itself, the
//! spatial-merge permutation, the axial-RoPE tables and the resized position
//! table.
//!
//! The permutation is the load-bearing one — see the [module
//! documentation](super) for why it has to run *before* the transformer.

use anyhow::{Result, bail};

use super::config::VisionConfig;

// ───────────────────────── host-side preprocessing ─────────────────────────

/// Resize preserving aspect ratio within the configured pixel bounds, aligned
/// to one merge block.
///
/// Matches llama.cpp `img_tool::calc_size_preserved_ratio`, which is HF's
/// `smart_resize`.
pub fn smart_resize(cfg: &VisionConfig, w: usize, h: usize) -> (usize, usize) {
    let align = cfg.align_size();
    assert!(align > 0, "align_size must be > 0");
    if w == 0 || h == 0 {
        return (0, 0);
    }
    let round = |x: f32| ((x / align as f32).round() * align as f32) as usize;
    let ceil = |x: f32| (x / align as f32).ceil() as usize * align;
    let floor = |x: f32| (x / align as f32).floor() as usize * align;

    let mut h_bar = align.max(round(h as f32));
    let mut w_bar = align.max(round(w as f32));
    if h_bar * w_bar > cfg.image_max_pixels {
        let beta = ((h * w) as f32 / cfg.image_max_pixels as f32).sqrt();
        h_bar = align.max(floor(h as f32 / beta));
        w_bar = align.max(floor(w as f32 / beta));
    } else if h_bar * w_bar < cfg.image_min_pixels {
        let beta = (cfg.image_min_pixels as f32 / (h * w) as f32).sqrt();
        h_bar = ceil(h as f32 * beta);
        w_bar = ceil(w as f32 * beta);
    }
    (w_bar, h_bar)
}

/// Nearest-neighbour resize of a packed RGB8 buffer.
pub fn resize_rgb_nearest(rgb: &[u8], w: usize, h: usize, out_w: usize, out_h: usize) -> Vec<u8> {
    let mut out = vec![0u8; out_w * out_h * 3];
    for oy in 0..out_h {
        let sy = oy * h / out_h;
        for ox in 0..out_w {
            let sx = ox * w / out_w;
            let src = (sy * w + sx) * 3;
            let dst = (oy * out_w + ox) * 3;
            out[dst..dst + 3].copy_from_slice(&rgb[src..src + 3]);
        }
    }
    out
}

/// `(pixel/255 − mean) / std` into NCHW f32.
pub fn rgb_to_nchw_f32(rgb: &[u8], w: usize, h: usize, cfg: &VisionConfig) -> Vec<f32> {
    let mut out = vec![0f32; 3 * w * h];
    for y in 0..h {
        for x in 0..w {
            let px = (y * w + x) * 3;
            for c in 0..3 {
                let v = rgb[px + c] as f32 / 255.0;
                out[c * w * h + y * w + x] = (v - cfg.image_mean[c]) / cfg.image_std[c];
            }
        }
    }
    out
}

/// smart resize → normalize → NCHW f32, returning the resized `(w, h)`.
pub fn preprocess_rgb(
    rgb: &[u8],
    w: usize,
    h: usize,
    cfg: &VisionConfig,
) -> (Vec<f32>, usize, usize) {
    let (tw, th) = smart_resize(cfg, w, h);
    let resized = resize_rgb_nearest(rgb, w, h, tw, th);
    (rgb_to_nchw_f32(&resized, tw, th, cfg), tw, th)
}

/// Patch indices in spatial-merge-block order: `[y0x0, y0x1, y1x0, y1x1, …]`.
///
/// This permutation is what makes the merger's reshape a reshape. Everything
/// position-dependent — the position embedding and the RoPE coordinates — has
/// to be generated in this same order.
pub fn spatial_merge_gather_idx(hp: usize, wp: usize, merge: usize) -> Vec<f32> {
    let mut idx = Vec::with_capacity(hp * wp);
    for y in (0..hp).step_by(merge) {
        for x in (0..wp).step_by(merge) {
            for dy in 0..merge {
                for dx in 0..merge {
                    let (py, px) = (y + dy, x + dx);
                    if py < hp && px < wp {
                        idx.push((py * wp + px) as f32);
                    }
                }
            }
        }
    }
    idx
}

/// `(row, col)` patch coordinates in merge-block order.
pub fn vision_position_hw(cfg: &VisionConfig, img_w: usize, img_h: usize) -> Vec<(i32, i32)> {
    let (wp, hp) = cfg.grid(img_w, img_h);
    spatial_merge_gather_idx(hp, wp, cfg.n_merge)
        .into_iter()
        .map(|i| {
            let i = i as usize;
            ((i / wp) as i32, (i % wp) as i32)
        })
        .collect()
}

/// Per-token cos/sin tables for the axial 2-D RoPE, `[n_pos · head_dim]` each.
///
/// `inv_freq` has `head_dim / 4` entries. The first half of each head carries
/// `[row freqs | column freqs]` and the second half duplicates it, so
/// `rotate_half` at `head_dim / 2` pairs row with row and column with column.
/// This is HF's `recomposition_frequencies`: `cat([cat([f_h, f_w]); ×2])`.
pub fn vision_rope_tables(cfg: &VisionConfig, position_hw: &[(i32, i32)]) -> (Vec<f32>, Vec<f32>) {
    let head_dim = cfg.head_dim();
    let half = head_dim / 2;
    let axis_dim = half / 2;
    let base = cfg.rope_theta as f64;
    let inv_freq: Vec<f64> = (0..axis_dim)
        .map(|j| 1.0 / base.powf((2 * j) as f64 / half as f64))
        .collect();
    let mut cos = vec![0f32; position_hw.len() * head_dim];
    let mut sin = vec![0f32; position_hw.len() * head_dim];
    for (t, &(y, x)) in position_hw.iter().enumerate() {
        for j in 0..axis_dim {
            for (pos, off) in [(y, j), (x, axis_dim + j)] {
                let (s, c) = (pos as f64 * inv_freq[j]).sin_cos();
                let i = t * head_dim + off;
                cos[i] = c as f32;
                cos[i + half] = c as f32;
                sin[i] = s as f32;
                sin[i + half] = s as f32;
            }
        }
    }
    (cos, sin)
}

/// Bilinearly resize the learned square position table to the patch grid, then
/// permute it into merge-block order.
///
/// `table` is `[num_pos, n_embd]` token-major, `num_pos` a perfect square.
pub(super) fn position_embd_for_grid(
    table: &[f32],
    n: usize,
    hp: usize,
    wp: usize,
    merge_idx: &[f32],
) -> Result<Vec<f32>> {
    if n == 0 || !table.len().is_multiple_of(n) {
        bail!(
            "qwen4exp vision: v.position_embd.weight has {} values, not a \
             multiple of embedding_length {n}",
            table.len()
        );
    }
    let src_rows = table.len() / n;
    let side = src_rows.isqrt();
    if side * side != src_rows {
        bail!(
            "qwen4exp vision: v.position_embd.weight has {src_rows} rows, which \
             is not a square grid — the bilinear resize needs one"
        );
    }
    let n_pos = hp * wp;
    let mut raster = vec![0f32; n_pos * n];
    if side == hp && side == wp {
        raster.copy_from_slice(&table[..n_pos * n]);
    } else {
        // HF `get_vision_interpolation_indices_and_weights`: linspace onto the
        // square table, then bilinear sample.
        let s = side as f32;
        for ty in 0..hp {
            let gy = if hp == 1 {
                0.0
            } else {
                ty as f32 * (s - 1.0) / (hp as f32 - 1.0)
            };
            let (y0, fy) = (gy.floor() as usize, gy - gy.floor());
            let y1 = (y0 + 1).min(side - 1);
            for tx in 0..wp {
                let gx = if wp == 1 {
                    0.0
                } else {
                    tx as f32 * (s - 1.0) / (wp as f32 - 1.0)
                };
                let (x0, fx) = (gx.floor() as usize, gx - gx.floor());
                let x1 = (x0 + 1).min(side - 1);
                let w = [
                    (1.0 - fy) * (1.0 - fx),
                    (1.0 - fy) * fx,
                    fy * (1.0 - fx),
                    fy * fx,
                ];
                let src = [
                    (y0 * side + x0) * n,
                    (y0 * side + x1) * n,
                    (y1 * side + x0) * n,
                    (y1 * side + x1) * n,
                ];
                let dst = (ty * wp + tx) * n;
                for d in 0..n {
                    raster[dst + d] = w[0] * table[src[0] + d]
                        + w[1] * table[src[1] + d]
                        + w[2] * table[src[2] + d]
                        + w[3] * table[src[3] + d];
                }
            }
        }
    }
    // Permute into merge-block order, to match the token reordering.
    let mut out = vec![0f32; n_pos * n];
    for (dst, &src) in merge_idx.iter().enumerate() {
        let src = src as usize;
        out[dst * n..(dst + 1) * n].copy_from_slice(&raster[src * n..(src + 1) * n]);
    }
    Ok(out)
}
