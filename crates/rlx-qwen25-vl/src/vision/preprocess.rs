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

//! RGB preprocessing + Qwen2.5-VL vision-side window / mRoPE host inputs.

use super::config::MmProjConfig;

/// Default window size used by llama.cpp Qwen2.5-VL (`attn_window_size = 112`).
pub const DEFAULT_ATTN_WINDOW_SIZE: usize = 112;

pub fn smart_resize(cfg: &MmProjConfig, w: usize, h: usize) -> (usize, usize) {
    calc_size_preserved_ratio(
        w,
        h,
        cfg.align_size(),
        cfg.image_min_pixels,
        cfg.image_max_pixels,
    )
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

/// Bicubic-ish resize (Catmull-Rom) — matches HF `PILImageResampling.BICUBIC` closely enough.
/// Without the `qwen25-vl-vision` feature (no `image` dep) it falls back to nearest-neighbor.
pub fn resize_rgb_bicubic(rgb: &[u8], w: usize, h: usize, out_w: usize, out_h: usize) -> Vec<u8> {
    if w == 0 || h == 0 {
        return vec![0u8; out_w * out_h * 3];
    }
    if w == out_w && h == out_h {
        return rgb.to_vec();
    }
    #[cfg(feature = "qwen25-vl-vision")]
    {
        use image::{ImageBuffer, Rgb};
        let img = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(w as u32, h as u32, rgb.to_vec())
            .expect("rgb buffer");
        let resized = image::imageops::resize(
            &img,
            out_w as u32,
            out_h as u32,
            image::imageops::FilterType::CatmullRom,
        );
        resized.into_raw()
    }
    #[cfg(not(feature = "qwen25-vl-vision"))]
    {
        resize_rgb_nearest_impl(rgb, w, h, out_w, out_h)
    }
}

#[cfg(not(feature = "qwen25-vl-vision"))]
fn resize_rgb_nearest_impl(rgb: &[u8], w: usize, h: usize, out_w: usize, out_h: usize) -> Vec<u8> {
    let mut out = vec![0u8; out_w * out_h * 3];
    for oy in 0..out_h {
        let sy = oy * h / out_h.max(1);
        for ox in 0..out_w {
            let sx = ox * w / out_w.max(1);
            let src = (sy * w + sx) * 3;
            let dst = (oy * out_w + ox) * 3;
            out[dst..dst + 3].copy_from_slice(&rgb[src..src + 3]);
        }
    }
    out
}

pub fn rgb_to_nchw_f32(rgb: &[u8], w: usize, h: usize, mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    let mut out = vec![0f32; 3 * w * h];
    for y in 0..h {
        for x in 0..w {
            let px = (y * w + x) * 3;
            for c in 0..3 {
                let v = rgb[px + c] as f32 / 255.0;
                out[c * w * h + y * w + x] = (v - mean[c]) / std[c];
            }
        }
    }
    out
}

pub fn preprocess_rgb(
    rgb: &[u8],
    w: usize,
    h: usize,
    cfg: &MmProjConfig,
) -> (Vec<f32>, usize, usize) {
    let (tw, th) = smart_resize(cfg, w, h);
    preprocess_rgb_to_size(rgb, w, h, tw, th, cfg)
}

/// Preprocess to an explicit target size (HF parity replay).
pub fn preprocess_rgb_to_size(
    rgb: &[u8],
    w: usize,
    h: usize,
    out_w: usize,
    out_h: usize,
    cfg: &MmProjConfig,
) -> (Vec<f32>, usize, usize) {
    let resized = resize_rgb_bicubic(rgb, w, h, out_w, out_h);
    let nchw = rgb_to_nchw_f32(&resized, out_w, out_h, cfg.image_mean, cfg.image_std);
    (nchw, out_w, out_h)
}

/// Gather indices that reorder raster NCHW patches into llama.cpp / HF merge-block order.
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

#[cfg(feature = "qwen25-vl-vision")]
pub fn load_rgb_image(path: &str) -> anyhow::Result<(Vec<u8>, usize, usize)> {
    let img = image::open(path).map_err(|e| anyhow::anyhow!("open {path}: {e}"))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    Ok((rgb.into_raw(), w, h))
}

/// Vision-side mRoPE position ids (i32), layout `[4 * n_pos]` (llama.cpp `positions`).
/// Retained as a test oracle for [`build_spatial_merge_gather_idx`] token order.
#[cfg(test)]
fn build_vision_positions(img_w: usize, img_h: usize, cfg: &MmProjConfig) -> Vec<i32> {
    let patch = cfg.patch_size;
    let merge = cfg.n_merge;
    let pw = img_w / patch;
    let ph = img_h / patch;
    let n_pos = pw * ph;
    let mut positions = vec![0i32; n_pos * 4];
    let mut ptr = 0usize;
    for y in (0..ph).step_by(merge) {
        for x in (0..pw).step_by(merge) {
            for dy in 0..merge {
                for dx in 0..merge {
                    let py = y + dy;
                    let px = x + dx;
                    if py < ph && px < pw {
                        positions[ptr] = py as i32;
                        positions[n_pos + ptr] = px as i32;
                        positions[2 * n_pos + ptr] = py as i32;
                        positions[3 * n_pos + ptr] = px as i32;
                        ptr += 1;
                    }
                }
            }
        }
    }
    debug_assert_eq!(
        ptr, n_pos,
        "vision positions: filled {ptr} != n_pos {n_pos} (ph={ph} pw={pw})"
    );
    positions
}

/// `(row, col)` patch coordinates in HF merge-block order (`get_vision_position_ids`).
pub fn build_vision_position_hw(img_w: usize, img_h: usize, cfg: &MmProjConfig) -> Vec<(i32, i32)> {
    let patch = cfg.patch_size;
    let merge = cfg.n_merge;
    let pw = img_w / patch;
    let ph = img_h / patch;
    let mut out = Vec::with_capacity(pw * ph);
    for y in (0..ph).step_by(merge) {
        for x in (0..pw).step_by(merge) {
            for dy in 0..merge {
                for dx in 0..merge {
                    let py = y + dy;
                    let px = x + dx;
                    if py < ph && px < pw {
                        out.push((py as i32, px as i32));
                    }
                }
            }
        }
    }
    out
}

/// Per-token cos/sin for HF vision RoPE (`[n_pos, head_dim]` row-major).
///
/// Matches `Qwen2_5_VisionRotaryEmbedding` + `torch.cat((emb, emb), dim=-1)`.
pub fn vision_rope_feeds(position_hw: &[(i32, i32)], head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let rope_dim = half / 2;
    let inv_freq: Vec<f64> = (0..rope_dim)
        .map(|j| 1.0 / 10_000f64.powf((2 * j) as f64 / half as f64))
        .collect();
    let n = position_hw.len();
    let mut angles = vec![0f32; n * half];
    for (t, &(h, w)) in position_hw.iter().enumerate() {
        let row = t * half;
        for j in 0..rope_dim {
            angles[row + j] = (h as f64 * inv_freq[j]) as f32;
            angles[row + rope_dim + j] = (w as f64 * inv_freq[j]) as f32;
        }
    }
    let mut cos = vec![0f32; n * head_dim];
    let mut sin = vec![0f32; n * head_dim];
    for t in 0..n {
        for j in 0..half {
            let angle = angles[t * half + j] as f64;
            let (s, c) = angle.sin_cos();
            let c32 = c as f32;
            let s32 = s as f32;
            cos[t * head_dim + j] = c32;
            cos[t * head_dim + half + j] = c32;
            sin[t * head_dim + j] = s32;
            sin[t * head_dim + half + j] = s32;
        }
    }
    (cos, sin)
}

/// Reorder per-token rows (e.g. RoPE cos/sin) with the same window gather as
/// [`window_token_gather_bsn`](rlx_ir::window_token_gather_bsn).
pub fn reorder_seq_by_window_inv(
    data: &mut [f32],
    inv_window_idx: &[f32],
    elem_per_token: usize,
    merge_sq: usize,
) {
    let n_pos = data.len() / elem_per_token;
    debug_assert!(merge_sq > 0 && n_pos.is_multiple_of(merge_sq));
    let n_win = n_pos / merge_sq;
    debug_assert_eq!(inv_window_idx.len(), n_win);
    let mut tmp = vec![0f32; data.len()];
    for (dst_g, &src_f) in inv_window_idx.iter().enumerate() {
        let src_g = src_f as usize;
        for m in 0..merge_sq {
            let dst_t = dst_g * merge_sq + m;
            let src_t = src_g * merge_sq + m;
            let dst = dst_t * elem_per_token;
            let src = src_t * elem_per_token;
            tmp[dst..dst + elem_per_token].copy_from_slice(&data[src..src + elem_per_token]);
        }
    }
    data.copy_from_slice(&tmp);
}

#[derive(Debug, Clone)]
pub struct WindowAttnInputs {
    /// Maps merged window order → original token index (`f32` for gather).
    pub inv_window_idx: Vec<f32>,
    /// Restores original order after merger (`f32` for gather).
    pub window_idx: Vec<f32>,
    /// `[n_pos, n_pos]` additive mask built from the per-window
    /// `cu_window_seqlens` boundaries.
    pub window_mask: Vec<f32>,
}

/// Expand a 2-D window mask for `MaskKind::Bias` attention
/// (`[batch, n_head, n_pos, n_pos]` row-major).
pub fn expand_window_attn_bias(
    mask: &[f32],
    batch: usize,
    n_head: usize,
    n_pos: usize,
) -> Vec<f32> {
    let plane = n_pos * n_pos;
    debug_assert_eq!(mask.len(), plane);
    let mut out = Vec::with_capacity(batch * n_head * plane);
    for _ in 0..batch {
        for _ in 0..n_head {
            out.extend_from_slice(mask);
        }
    }
    out
}

/// Large negative logit for masked vision window pairs (additive bias).
const VISION_WINDOW_MASK_NEG: f32 = -1.0e4;

/// Build window-attention gather indices + mask aligned with HF
/// `get_vision_window_index` / `cu_window_seqlens`.
pub fn build_window_attn_inputs(
    img_w: usize,
    img_h: usize,
    cfg: &MmProjConfig,
    attn_window_size: usize,
) -> WindowAttnInputs {
    let patch = cfg.patch_size;
    let merge = cfg.n_merge;
    let merge_unit = merge * merge;
    let ipw = img_w / patch;
    let iph = img_h / patch;
    let n_pos = ipw * iph;
    let llm_h = iph / merge;
    let llm_w = ipw / merge;
    let grid_window = attn_window_size / patch / merge;

    let pad_h = grid_window - llm_h % grid_window;
    let pad_w = grid_window - llm_w % grid_window;
    let num_wh = (llm_h + pad_h) / grid_window;
    let num_ww = (llm_w + pad_w) / grid_window;

    let mut window_groups = Vec::new();
    let mut cu_seqlens = vec![0i32];
    for wy in 0..num_wh {
        for wx in 0..num_ww {
            let mut groups_in_win = 0usize;
            for dy in 0..grid_window {
                for dx in 0..grid_window {
                    let gy = wy * grid_window + dy;
                    let gx = wx * grid_window + dx;
                    if gy < llm_h && gx < llm_w {
                        window_groups.push(gy * llm_w + gx);
                        groups_in_win += 1;
                    }
                }
            }
            if groups_in_win > 0 {
                let prev = *cu_seqlens.last().unwrap();
                cu_seqlens.push(prev + (groups_in_win * merge_unit) as i32);
            }
        }
    }
    cu_seqlens.dedup();

    let n_groups = llm_h * llm_w;
    let mut idx = vec![0f32; n_groups];
    let mut inv_idx = vec![0f32; n_groups];
    for (dst, &src) in window_groups.iter().enumerate() {
        idx[src] = dst as f32;
        inv_idx[dst] = src as f32;
    }

    let mut mask = vec![VISION_WINDOW_MASK_NEG; n_pos * n_pos];
    for w in 0..cu_seqlens.len().saturating_sub(1) {
        let start = cu_seqlens[w] as usize;
        let end = cu_seqlens[w + 1] as usize;
        for i in start..end.min(n_pos) {
            for j in start..end.min(n_pos) {
                mask[i * n_pos + j] = 0.0;
            }
        }
    }

    WindowAttnInputs {
        inv_window_idx: inv_idx,
        window_idx: idx,
        window_mask: mask,
    }
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
            projector_type: "qwen2.5vl_merger".into(),
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            spatial_merge_size: 2,
            llm_hidden_size: 32,
            n_ff: 32,
            n_wa_pattern: 8,
            use_silu: true,
            use_rms_norm: true,
        }
    }

    #[test]
    fn vision_positions_length_matches_patch_grid() {
        let cfg = tiny_cfg();
        let pos = build_vision_positions(4, 4, &cfg);
        let n_pos = (4 / cfg.patch_size) * (4 / cfg.patch_size);
        assert_eq!(pos.len(), n_pos * 4);
    }

    #[test]
    fn spatial_merge_gather_matches_mrope_token_order() {
        let cfg = tiny_cfg();
        let img = 8usize;
        let ph = img / cfg.patch_size;
        let pw = ph;
        let merge = cfg.n_merge;
        let idx = build_spatial_merge_gather_idx(ph, pw, merge);
        let n_pos = ph * pw;
        assert_eq!(idx.len(), n_pos);
        let pos = build_vision_positions(img, img, &cfg);
        for (t, &raw) in idx.iter().enumerate() {
            let i = raw as usize;
            let y = i / pw;
            let x = i % pw;
            assert_eq!(pos[t] as usize, y, "token {t} y");
            assert_eq!(pos[n_pos + t] as usize, x, "token {t} x");
        }
    }

    #[test]
    fn window_index_matches_hf_grid_26x46() {
        let cfg = MmProjConfig {
            patch_size: 14,
            n_embd: 1280,
            n_head: 16,
            n_layer: 32,
            image_size: 448,
            image_min_pixels: 784,
            image_max_pixels: 1_048_576,
            n_merge: 2,
            eps: 1e-6,
            projector_type: "qwen2.5vl_merger".into(),
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            spatial_merge_size: 2,
            llm_hidden_size: 3584,
            n_ff: 3420,
            n_wa_pattern: 8,
            use_silu: true,
            use_rms_norm: true,
        };
        let w = build_window_attn_inputs(644, 364, &cfg, 112);
        let inv: Vec<usize> = w.inv_window_idx.iter().map(|x| *x as usize).collect();
        let expected = [0, 1, 2, 3, 23, 24, 25, 26, 46, 47, 48, 49, 69, 70, 71];
        assert_eq!(&inv[..expected.len()], expected);
    }

    #[test]
    fn vision_rope_feeds_token0_reference() {
        let hw = vec![(0i32, 0i32)];
        let (cos, sin) = vision_rope_feeds(&hw, 80);
        assert_eq!(cos.len(), 80);
        assert!((cos[0] - 1.0).abs() < 1e-5, "cos0={}", cos[0]);
        assert!(sin[0].abs() < 1e-5, "sin0={}", sin[0]);
    }
}
