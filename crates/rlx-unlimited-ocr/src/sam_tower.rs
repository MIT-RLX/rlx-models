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

//! SAM-ViT-B vision tower — high-resolution local detail branch of the deep
//! encoder.
//!
//! Eager host-f32 port of Meta's `segment_anything/modeling/image_encoder.py`
//! (patch embed → abs-pos → 12 blocks with windowed/global decomposed
//! rel-pos attention → neck), plus DeepSeek-OCR's `net_2` / `net_3`
//! downsample convs that bring the 256-channel neck output down to `1024`
//! channels at CLIP's patch resolution ([`crate::config::SamTowerConfig::downsample_channels`]).
//!
//! Internally everything (patch embed, blocks, neck, downsample) runs in
//! channel-last `[h*w, c]` (token-major) layout, matching [`crate::nn`]'s
//! `conv2d_hwc` / `layer_norm` helpers — a `LayerNorm2d` over `[C,H,W]` is
//! exactly a plain `LayerNorm` over the last axis of `[H*W, C]`. Only the
//! final result is transposed to channel-first `NCHW`, matching what
//! [`crate::clip_tower::ClipTower::encode`] and
//! [`crate::deep_encoder::DeepEncoder`] expect (`sam_features_nchw`).

use crate::config::SamTowerConfig;
use crate::nn;
use crate::weights::{UnlimitedOcrWeightPrefix, UnlimitedOcrWeightStore};
use anyhow::{Context, Result, ensure};
use rlx_core::weight_map::WeightMap;

const PRETRAINED_GRID: usize = 64; // image_size(1024) / patch_size(16)

struct SamBlockWeights {
    norm1_g: Vec<f32>,
    norm1_b: Vec<f32>,
    qkv_w: Vec<f32>, // [3*hidden, hidden] (PyTorch nn.Linear layout)
    qkv_b: Vec<f32>,
    proj_w: Vec<f32>, // [hidden, hidden]
    proj_b: Vec<f32>,
    rel_pos_h: Vec<f32>, // [2*window-1 (or 2*pretrained_grid-1), head_dim]
    rel_pos_w: Vec<f32>,
    norm2_g: Vec<f32>,
    norm2_b: Vec<f32>,
    mlp1_w: Vec<f32>, // [4*hidden, hidden]
    mlp1_b: Vec<f32>,
    mlp2_w: Vec<f32>, // [hidden, 4*hidden]
    mlp2_b: Vec<f32>,
    is_global: bool,
}

/// Uncompiled SAM-ViT-B encoder weights + config.
pub struct SamTower {
    pub config: SamTowerConfig,
    patch_w: Option<Vec<f32>>, // [hidden, 3, patch, patch] (OIHW, PyTorch layout)
    patch_b: Option<Vec<f32>>,
    pos_embed: Option<Vec<f32>>, // [PRETRAINED_GRID, PRETRAINED_GRID, hidden] (HWC)
    blocks: Vec<SamBlockWeights>,
    neck_conv1_w: Option<Vec<f32>>, // [out_chans, hidden, 1, 1]
    neck_ln1_g: Option<Vec<f32>>,
    neck_ln1_b: Option<Vec<f32>>,
    neck_conv2_w: Option<Vec<f32>>, // [out_chans, out_chans, 3, 3]
    neck_ln2_g: Option<Vec<f32>>,
    neck_ln2_b: Option<Vec<f32>>,
    net2_w: Option<Vec<f32>>, // [downsample[0], out_chans, 3, 3]
    net3_w: Option<Vec<f32>>, // [downsample[1], downsample[0], 3, 3]
}

impl SamTower {
    pub fn from_config(config: &SamTowerConfig) -> Self {
        Self {
            config: config.clone(),
            patch_w: None,
            patch_b: None,
            pos_embed: None,
            blocks: Vec::new(),
            neck_conv1_w: None,
            neck_ln1_g: None,
            neck_ln1_b: None,
            neck_conv2_w: None,
            neck_ln2_g: None,
            neck_ln2_b: None,
            net2_w: None,
            net3_w: None,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.config.hidden_size / self.config.num_attention_heads
    }

    pub fn load(&mut self, store: &UnlimitedOcrWeightStore) -> Result<()> {
        let mut map = store.load_sam_tower()?;
        let hidden = self.config.hidden_size;

        self.patch_w = Some(
            map.take(UnlimitedOcrWeightPrefix::sam_patch_embed_w())
                .context("sam patch_embed.proj.weight")?
                .0,
        );
        self.patch_b = Some(
            map.take(UnlimitedOcrWeightPrefix::sam_patch_embed_b())
                .context("sam patch_embed.proj.bias")?
                .0,
        );

        let (pos, pos_shape) = map
            .take(UnlimitedOcrWeightPrefix::sam_pos_embed())
            .context("sam pos_embed")?;
        ensure!(
            pos.len() == hidden * PRETRAINED_GRID * PRETRAINED_GRID,
            "sam pos_embed shape {pos_shape:?} != [.., {PRETRAINED_GRID}, {PRETRAINED_GRID}, {hidden}]"
        );
        self.pos_embed = Some(pos); // checkpoint layout is [1, H, W, C] already

        self.blocks = Vec::with_capacity(self.config.num_hidden_layers);
        for i in 0..self.config.num_hidden_layers {
            let is_global = self.config.global_attn_indexes.contains(&i);
            let take = |m: &mut WeightMap, suffix: &str| -> Result<Vec<f32>> {
                let key = UnlimitedOcrWeightPrefix::sam_block(i, suffix);
                Ok(m.take(&key).with_context(|| format!("sam {key}"))?.0)
            };
            self.blocks.push(SamBlockWeights {
                norm1_g: take(&mut map, "norm1.weight")?,
                norm1_b: take(&mut map, "norm1.bias")?,
                qkv_w: take(&mut map, "attn.qkv.weight")?,
                qkv_b: take(&mut map, "attn.qkv.bias")?,
                proj_w: take(&mut map, "attn.proj.weight")?,
                proj_b: take(&mut map, "attn.proj.bias")?,
                rel_pos_h: take(&mut map, "attn.rel_pos_h")?,
                rel_pos_w: take(&mut map, "attn.rel_pos_w")?,
                norm2_g: take(&mut map, "norm2.weight")?,
                norm2_b: take(&mut map, "norm2.bias")?,
                mlp1_w: take(&mut map, "mlp.lin1.weight")?,
                mlp1_b: take(&mut map, "mlp.lin1.bias")?,
                mlp2_w: take(&mut map, "mlp.lin2.weight")?,
                mlp2_b: take(&mut map, "mlp.lin2.bias")?,
                is_global,
            });
        }

        self.neck_conv1_w = Some(
            map.take(&UnlimitedOcrWeightPrefix::sam_neck(0, "weight"))
                .context("sam neck.0.weight")?
                .0,
        );
        self.neck_ln1_g = Some(
            map.take(&UnlimitedOcrWeightPrefix::sam_neck(1, "weight"))
                .context("sam neck.1.weight")?
                .0,
        );
        self.neck_ln1_b = Some(
            map.take(&UnlimitedOcrWeightPrefix::sam_neck(1, "bias"))
                .context("sam neck.1.bias")?
                .0,
        );
        self.neck_conv2_w = Some(
            map.take(&UnlimitedOcrWeightPrefix::sam_neck(2, "weight"))
                .context("sam neck.2.weight")?
                .0,
        );
        self.neck_ln2_g = Some(
            map.take(&UnlimitedOcrWeightPrefix::sam_neck(3, "weight"))
                .context("sam neck.3.weight")?
                .0,
        );
        self.neck_ln2_b = Some(
            map.take(&UnlimitedOcrWeightPrefix::sam_neck(3, "bias"))
                .context("sam neck.3.bias")?
                .0,
        );
        self.net2_w = Some(
            map.take(UnlimitedOcrWeightPrefix::sam_net2_w())
                .context("sam net_2.weight")?
                .0,
        );
        self.net3_w = Some(
            map.take(UnlimitedOcrWeightPrefix::sam_net3_w())
                .context("sam net_3.weight")?
                .0,
        );

        Ok(())
    }

    /// Encode one preprocessed view (`pixels` is CHW f32, `3*side*side` long)
    /// into `[downsample_channels[1], q, q]` NCHW features
    /// (`q = side/16/4`, matching [`crate::config::num_queries`]).
    pub fn encode(&self, pixels: &[f32], side: usize) -> Result<Vec<f32>> {
        let patch = self.config.patch_size;
        ensure!(
            side.is_multiple_of(patch),
            "sam encode: side {side} not divisible by patch {patch}"
        );
        ensure!(
            pixels.len() == 3 * side * side,
            "sam encode: pixel buffer len mismatch"
        );
        let hidden = self.config.hidden_size;
        let heads = self.config.num_attention_heads;
        let dh = self.head_dim();
        let grid = side / patch;

        let hwc_pixels = chw_to_hwc(pixels, 3, side, side);

        let patch_w = self.patch_w.as_ref().context("sam not loaded")?;
        let patch_b = self.patch_b.as_ref().context("sam not loaded")?;
        let (mut x, oh, ow) = nn::conv2d_hwc(
            &hwc_pixels,
            side,
            side,
            3,
            patch_w,
            hidden,
            patch,
            patch,
            patch,
            0,
            Some(patch_b),
        )?;
        ensure!(
            oh == grid && ow == grid,
            "sam patch_embed grid mismatch: {oh}x{ow} != {grid}x{grid}"
        );

        let pos_embed = self.pos_embed.as_ref().context("sam not loaded")?;
        let pos = if grid == PRETRAINED_GRID {
            pos_embed.clone()
        } else {
            nn::bicubic_resize_hwc(
                pos_embed,
                PRETRAINED_GRID,
                PRETRAINED_GRID,
                hidden,
                grid,
                grid,
            )
        };
        nn::add_inplace(&mut x, &pos);

        for blk in &self.blocks {
            let normed = nn::layer_norm(&x, grid * grid, hidden, &blk.norm1_g, &blk.norm1_b, 1e-6);
            let attn_out = if blk.is_global {
                block_attention(&normed, grid, grid, heads, dh, blk)?
            } else {
                windowed_attention(&normed, grid, heads, dh, self.config.window_size, blk)?
            };
            nn::add_inplace(&mut x, &attn_out);

            let normed2 = nn::layer_norm(&x, grid * grid, hidden, &blk.norm2_g, &blk.norm2_b, 1e-6);
            let mut inter = nn::linear_wt(
                &normed2,
                grid * grid,
                hidden,
                &blk.mlp1_w,
                hidden * 4,
                Some(&blk.mlp1_b),
            )?;
            nn::gelu_erf(&mut inter);
            let ffn = nn::linear_wt(
                &inter,
                grid * grid,
                hidden * 4,
                &blk.mlp2_w,
                hidden,
                Some(&blk.mlp2_b),
            )?;
            nn::add_inplace(&mut x, &ffn);
        }

        let out_chans = self.config.out_chans;
        let conv1_w = self.neck_conv1_w.as_ref().context("sam not loaded")?;
        let (feat, gh, gw) =
            nn::conv2d_hwc(&x, grid, grid, hidden, conv1_w, out_chans, 1, 1, 1, 0, None)?;
        let mut feat = nn::layer_norm(
            &feat,
            gh * gw,
            out_chans,
            self.neck_ln1_g.as_ref().context("sam not loaded")?,
            self.neck_ln1_b.as_ref().context("sam not loaded")?,
            1e-6,
        );
        let conv2_w = self.neck_conv2_w.as_ref().context("sam not loaded")?;
        let (feat2, gh, gw) = nn::conv2d_hwc(
            &feat, gh, gw, out_chans, conv2_w, out_chans, 3, 3, 1, 1, None,
        )?;
        feat = nn::layer_norm(
            &feat2,
            gh * gw,
            out_chans,
            self.neck_ln2_g.as_ref().context("sam not loaded")?,
            self.neck_ln2_b.as_ref().context("sam not loaded")?,
            1e-6,
        );

        ensure!(
            self.config.downsample_channels.len() == 2,
            "sam: expected 2 downsample stages"
        );
        let net2_w = self.net2_w.as_ref().context("sam not loaded")?;
        let mid_chans = self.config.downsample_channels[0];
        let (feat, gh, gw) = nn::conv2d_hwc(
            &feat, gh, gw, out_chans, net2_w, mid_chans, 3, 3, 2, 1, None,
        )?;
        let net3_w = self.net3_w.as_ref().context("sam not loaded")?;
        let out_chans2 = self.config.downsample_channels[1];
        let (feat, gh, gw) = nn::conv2d_hwc(
            &feat, gh, gw, mid_chans, net3_w, out_chans2, 3, 3, 2, 1, None,
        )?;

        Ok(hwc_to_chw(&feat, gh, gw, out_chans2))
    }

    /// Batched [`Self::encode`]: `pixels` is `[batch, 3, side, side]` NCHW,
    /// returns `[batch, downsample_channels[1], q, q]` NCHW features
    /// concatenated along `batch`.
    pub fn encode_batch(&self, pixels: &[f32], side: usize, batch: usize) -> Result<Vec<f32>> {
        let per_image = 3 * side * side;
        ensure!(
            pixels.len() == batch * per_image,
            "sam encode_batch: pixel buffer len {} != {batch}*{per_image}",
            pixels.len()
        );
        let mut out = Vec::new();
        for b in 0..batch {
            let one = self.encode(&pixels[b * per_image..(b + 1) * per_image], side)?;
            out.extend_from_slice(&one);
        }
        Ok(out)
    }
}

fn chw_to_hwc(x: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0f32; h * w * c];
    for ci in 0..c {
        for p in 0..h * w {
            out[p * c + ci] = x[ci * h * w + p];
        }
    }
    out
}

fn hwc_to_chw(x: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0f32; h * w * c];
    for ci in 0..c {
        for p in 0..h * w {
            out[ci * h * w + p] = x[p * c + ci];
        }
    }
    out
}

/// Gather (+ resize when needed) SAM's decomposed rel-pos table into a
/// dense `[size, size, head_dim]` bias lookup (`q_size == k_size == size`
/// for every attention call in this model — no cross-resolution attention).
fn extract_rel_pos(raw: &[f32], dh: usize, size: usize) -> Vec<f32> {
    let raw_len = raw.len() / dh;
    let need = 2 * size - 1;
    let resized = nn::linear_resize_1d(raw, raw_len, dh, need);
    let mut out = vec![0f32; size * size * dh];
    for q in 0..size {
        for k in 0..size {
            let idx = (q as isize - k as isize + (size as isize - 1)) as usize;
            let src = &resized[idx * dh..(idx + 1) * dh];
            let dst = &mut out[(q * size + k) * dh..(q * size + k + 1) * dh];
            dst.copy_from_slice(src);
        }
    }
    out
}

/// Decomposed multi-head self-attention over a square `[size, size]` grid
/// (`x` is `[size*size, hidden]`, already normalized).
fn block_attention(
    x: &[f32],
    h: usize,
    w: usize,
    num_heads: usize,
    head_dim: usize,
    blk: &SamBlockWeights,
) -> Result<Vec<f32>> {
    let s = h * w;
    let hidden = num_heads * head_dim;
    ensure!(
        h == w,
        "SAM rel-pos attention requires a square grid, got {h}x{w}"
    );
    let qkv = nn::linear_wt(x, s, hidden, &blk.qkv_w, 3 * hidden, Some(&blk.qkv_b))?;
    let rel_h = extract_rel_pos(&blk.rel_pos_h, head_dim, h);
    let rel_w = extract_rel_pos(&blk.rel_pos_w, head_dim, w);
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut merged = vec![0f32; s * hidden];
    let mut scores = vec![0f32; s * s];
    let mut rel_h_vec = vec![0f32; h];
    let mut rel_w_vec = vec![0f32; w];
    for head in 0..num_heads {
        let q_off = head * head_dim;
        let k_off = hidden + head * head_dim;
        let v_off = 2 * hidden + head * head_dim;

        for qi in 0..s {
            let qvec = &qkv[qi * 3 * hidden + q_off..qi * 3 * hidden + q_off + head_dim];
            for ki in 0..s {
                let kvec = &qkv[ki * 3 * hidden + k_off..ki * 3 * hidden + k_off + head_dim];
                let dot: f32 = qvec.iter().zip(kvec.iter()).map(|(a, b)| a * b).sum();
                scores[qi * s + ki] = dot * scale;
            }
            let q_row = qi / w;
            let q_col = qi % w;
            for k_row in 0..h {
                let rvec =
                    &rel_h[(q_row * h + k_row) * head_dim..(q_row * h + k_row + 1) * head_dim];
                rel_h_vec[k_row] = qvec.iter().zip(rvec.iter()).map(|(a, b)| a * b).sum();
            }
            for k_col in 0..w {
                let rvec =
                    &rel_w[(q_col * w + k_col) * head_dim..(q_col * w + k_col + 1) * head_dim];
                rel_w_vec[k_col] = qvec.iter().zip(rvec.iter()).map(|(a, b)| a * b).sum();
            }
            let row = &mut scores[qi * s..(qi + 1) * s];
            for k_row in 0..h {
                for k_col in 0..w {
                    row[k_row * w + k_col] += rel_h_vec[k_row] + rel_w_vec[k_col];
                }
            }
        }

        nn::softmax_rows(&mut scores, s, s);

        for qi in 0..s {
            let dst = &mut merged
                [qi * hidden + head * head_dim..qi * hidden + head * head_dim + head_dim];
            for ki in 0..s {
                let w_ik = scores[qi * s + ki];
                if w_ik == 0.0 {
                    continue;
                }
                let vvec = &qkv[ki * 3 * hidden + v_off..ki * 3 * hidden + v_off + head_dim];
                for d in 0..head_dim {
                    dst[d] += w_ik * vvec[d];
                }
            }
        }
    }

    nn::linear_wt(&merged, s, hidden, &blk.proj_w, hidden, Some(&blk.proj_b))
}

/// Windowed attention: pad `grid x grid` to a multiple of `window`,
/// partition into non-overlapping `window x window` blocks, attend
/// independently within each, then reverse the partition and crop.
fn windowed_attention(
    x: &[f32],
    grid: usize,
    num_heads: usize,
    head_dim: usize,
    window: usize,
    blk: &SamBlockWeights,
) -> Result<Vec<f32>> {
    let hidden = num_heads * head_dim;
    if grid <= window {
        let padded = pad_grid(x, grid, grid, hidden, window, window);
        let out = block_attention(&padded, window, window, num_heads, head_dim, blk)?;
        return Ok(crop_grid(&out, window, window, hidden, grid, grid));
    }
    let pad = (window - grid % window) % window;
    let grid_p = grid + pad;
    let n_per_side = grid_p / window;

    let padded = pad_grid(x, grid, grid, hidden, grid_p, grid_p);
    let mut out = vec![0f32; grid_p * grid_p * hidden];
    for wr in 0..n_per_side {
        for wc in 0..n_per_side {
            let mut win = vec![0f32; window * window * hidden];
            for ir in 0..window {
                let src_row = wr * window + ir;
                let src = &padded[(src_row * grid_p + wc * window) * hidden
                    ..(src_row * grid_p + wc * window + window) * hidden];
                win[ir * window * hidden..(ir + 1) * window * hidden].copy_from_slice(src);
            }
            let attn_out = block_attention(&win, window, window, num_heads, head_dim, blk)?;
            for ir in 0..window {
                let src_row = wr * window + ir;
                let dst = &mut out[(src_row * grid_p + wc * window) * hidden
                    ..(src_row * grid_p + wc * window + window) * hidden];
                dst.copy_from_slice(&attn_out[ir * window * hidden..(ir + 1) * window * hidden]);
            }
        }
    }
    Ok(crop_grid(&out, grid_p, grid_p, hidden, grid, grid))
}

fn pad_grid(x: &[f32], h: usize, w: usize, c: usize, dh: usize, dw: usize) -> Vec<f32> {
    if h == dh && w == dw {
        return x.to_vec();
    }
    let mut out = vec![0f32; dh * dw * c];
    for r in 0..h {
        out[(r * dw) * c..(r * dw + w) * c].copy_from_slice(&x[(r * w) * c..(r * w + w) * c]);
    }
    out
}

fn crop_grid(x: &[f32], h: usize, w: usize, c: usize, dh: usize, dw: usize) -> Vec<f32> {
    if h == dh && w == dw {
        return x.to_vec();
    }
    let mut out = vec![0f32; dh * dw * c];
    for r in 0..dh {
        out[(r * dw) * c..(r * dw + dw) * c].copy_from_slice(&x[(r * w) * c..(r * w + dw) * c]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chw_hwc_roundtrip() {
        let x: Vec<f32> = (0..(2 * 3 * 4)).map(|v| v as f32).collect();
        let hwc = chw_to_hwc(&x, 2, 3, 4);
        let back = hwc_to_chw(&hwc, 3, 4, 2);
        assert_eq!(x, back);
    }

    #[test]
    fn pad_and_crop_grid_roundtrip() {
        let x: Vec<f32> = (0..(4 * 4 * 2)).map(|v| v as f32).collect();
        let padded = pad_grid(&x, 4, 4, 2, 6, 6);
        let cropped = crop_grid(&padded, 6, 6, 2, 4, 4);
        assert_eq!(x, cropped);
    }

    #[test]
    fn extract_rel_pos_matches_size_without_resize() {
        let dh = 2;
        let size = 3;
        let raw: Vec<f32> = (0..(2 * size - 1))
            .flat_map(|i| vec![i as f32, i as f32])
            .collect();
        let out = extract_rel_pos(&raw, dh, size);
        assert_eq!(out.len(), size * size * dh);
        let mid = (size - 1) as f32;
        assert_eq!(
            &out[(1 * size + 1) * dh..(1 * size + 1) * dh + dh],
            &[mid, mid]
        );
    }

    #[test]
    fn from_config_head_dim() {
        let cfg = SamTowerConfig::default();
        let tower = SamTower::from_config(&cfg);
        assert_eq!(tower.head_dim(), 64);
    }
}
