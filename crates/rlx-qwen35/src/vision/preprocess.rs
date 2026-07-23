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

//! RGB preprocessing — smart resize + mean/std normalize (NCHW f32).

use super::config::MmProjConfig;

/// Resize RGB image preserving aspect ratio within min/max pixel bounds.
/// Matches llama.cpp `img_tool::calc_size_preserved_ratio` in `mtmd-image.cpp`.
pub fn smart_resize(cfg: &MmProjConfig, w: usize, h: usize) -> (usize, usize) {
    calc_size_preserved_ratio(
        w,
        h,
        cfg.align_size(),
        cfg.image_min_pixels,
        cfg.image_max_pixels,
    )
}

/// Longest-edge variant (pixtral / idefics path).
#[allow(dead_code)]
pub fn smart_resize_longest_edge(
    w: usize,
    h: usize,
    align: usize,
    longest_edge: usize,
) -> (usize, usize) {
    if w == 0 || h == 0 || longest_edge == 0 {
        return (0, 0);
    }
    let scale = (longest_edge as f32 / w as f32).min(longest_edge as f32 / h as f32);
    let tw = ceil_by_factor(w as f32 * scale, align);
    let th = ceil_by_factor(h as f32 * scale, align);
    (tw, th)
}

fn calc_size_preserved_ratio(
    w: usize,
    h: usize,
    align: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    assert!(align > 0, "align_size must be > 0");
    if w == 0 || h == 0 {
        return (0, 0);
    }

    let mut h_bar = align.max(round_by_factor(h as f32, align));
    let mut w_bar = align.max(round_by_factor(w as f32, align));

    if h_bar * w_bar > max_pixels {
        let beta = ((h * w) as f32 / max_pixels as f32).sqrt();
        h_bar = align.max(floor_by_factor(h as f32 / beta, align));
        w_bar = align.max(floor_by_factor(w as f32 / beta, align));
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f32 / (h * w) as f32).sqrt();
        h_bar = ceil_by_factor(h as f32 * beta, align);
        w_bar = ceil_by_factor(w as f32 * beta, align);
    }
    (w_bar, h_bar)
}

fn round_by_factor(x: f32, f: usize) -> usize {
    ((x / f as f32).round() * f as f32) as usize
}
fn ceil_by_factor(x: f32, f: usize) -> usize {
    (x / f as f32).ceil() as usize * f
}
fn floor_by_factor(x: f32, f: usize) -> usize {
    (x / f as f32).floor() as usize * f
}

/// Bilinear-ish nearest resize of RGB u8 buffer to `(out_w, out_h)`.
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

/// Normalize resized RGB to NCHW f32: `(pixel - mean) / std`.
pub fn rgb_to_nchw_f32(rgb: &[u8], w: usize, h: usize, cfg: &MmProjConfig) -> Vec<f32> {
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

/// Full preprocess pipeline: smart resize → normalize → NCHW f32.
pub fn preprocess_rgb(
    rgb: &[u8],
    w: usize,
    h: usize,
    cfg: &MmProjConfig,
) -> (Vec<f32>, usize, usize) {
    let (tw, th) = smart_resize(cfg, w, h);
    let resized = resize_rgb_nearest(rgb, w, h, tw, th);
    let nchw = rgb_to_nchw_f32(&resized, tw, th, cfg);
    (nchw, tw, th)
}

#[cfg(feature = "qwen35-vlm")]
pub fn load_rgb_image(path: &str) -> anyhow::Result<(Vec<u8>, usize, usize)> {
    use anyhow::Context;
    use image::GenericImageView;
    let img = image::open(path).with_context(|| format!("open image {path}"))?;
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8().into_raw();
    Ok((rgb, w as usize, h as usize))
}

/// Build patch-grid indices in merge-block order:
/// `[y0x0, y0x1, y1x0, y1x1, ...]`.
pub fn build_spatial_merge_gather_idx(ph: usize, pw: usize, merge: usize) -> Vec<f32> {
    let mut idx = Vec::with_capacity(ph * pw);
    for y in (0..ph).step_by(merge) {
        for x in (0..pw).step_by(merge) {
            for dy in 0..merge {
                for dx in 0..merge {
                    let py = y + dy;
                    let px = x + dx;
                    if py < ph && px < pw {
                        idx.push((py * pw + px) as f32);
                    }
                }
            }
        }
    }
    idx
}

/// `(row, col)` patch coordinates in merge-block order.
pub fn build_vision_position_hw(img_w: usize, img_h: usize, cfg: &MmProjConfig) -> Vec<(i32, i32)> {
    let patch = cfg.patch_size;
    let merge = cfg.n_merge;
    let pw = img_w / patch;
    let ph = img_h / patch;
    build_spatial_merge_gather_idx(ph, pw, merge)
        .into_iter()
        .map(|i| {
            let i = i as usize;
            ((i / pw) as i32, (i % pw) as i32)
        })
        .collect()
}

/// Build Qwen3-VL vision-side MRoPE `positions` input (i32, 4 × n_pos).
pub fn build_vision_positions(img_w: usize, img_h: usize, cfg: &MmProjConfig) -> Vec<i32> {
    let position_hw = build_vision_position_hw(img_w, img_h, cfg);
    let n_pos = position_hw.len();
    let mut positions = vec![0i32; n_pos * 4];
    for (i, &(y, x)) in position_hw.iter().enumerate() {
        positions[i] = y;
        positions[n_pos + i] = x;
        positions[2 * n_pos + i] = y;
        positions[3 * n_pos + i] = x;
    }
    positions
}

/// Per-token cosine and sine tables for the Qwen vision axial RoPE.
///
/// The vision encoder uses half of each attention head for row and half for
/// column frequencies, then duplicates the result for rotate-half RoPE.
pub fn vision_rope_feeds(position_hw: &[(i32, i32)], head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let axis_dim = half / 2;
    let inv_freq: Vec<f64> = (0..axis_dim)
        .map(|j| 1.0 / 10_000f64.powf((2 * j) as f64 / half as f64))
        .collect();
    let mut cos = vec![0.0; position_hw.len() * head_dim];
    let mut sin = vec![0.0; position_hw.len() * head_dim];
    for (t, &(y, x)) in position_hw.iter().enumerate() {
        for j in 0..axis_dim {
            for (pos, offset) in [(y, j), (x, axis_dim + j)] {
                let angle = pos as f64 * inv_freq[j];
                let (s, c) = angle.sin_cos();
                let i = t * head_dim + offset;
                cos[i] = c as f32;
                cos[i + half] = c as f32;
                sin[i] = s as f32;
                sin[i + half] = s as f32;
            }
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> MmProjConfig {
        MmProjConfig {
            patch_size: 2,
            n_embd: 16,
            n_head: 2,
            n_layer: 1,
            image_size: 4,
            image_min_pixels: 16,
            image_max_pixels: 256,
            n_merge: 2,
            eps: 1e-6,
            projector_type: "qwen3vl".into(),
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            spatial_merge_size: 2,
            llm_hidden_size: 32,
            n_ff: 32,
            deepstack_layers: vec![],
        }
    }

    #[test]
    fn smart_resize_keeps_aspect_within_bounds() {
        let cfg = tiny_cfg();
        let (tw, th) = smart_resize(&cfg, 100, 50);
        assert_eq!(tw % cfg.align_size(), 0);
        assert_eq!(th % cfg.align_size(), 0);
        assert!(tw * th >= cfg.image_min_pixels);
        assert!(tw * th <= cfg.image_max_pixels);
    }

    #[test]
    fn spatial_merge_order_groups_two_by_two_raster_blocks() {
        assert_eq!(
            build_spatial_merge_gather_idx(4, 4, 2),
            vec![
                0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0, 12.0, 13.0, 10.0, 11.0, 14.0,
                15.0
            ]
        );
    }
}
