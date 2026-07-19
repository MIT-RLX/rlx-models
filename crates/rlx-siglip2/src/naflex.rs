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

//! NaFlex (`model_type = "siglip2"`) host support.
//!
//! NaFlex keeps a single compiled encoder at `seq = max_num_patches` and
//! handles variable resolution entirely on host, mirroring the fixed-res
//! stem: an `Siglip2ImageProcessor`-equivalent resize + patchify produces
//! `pixel_values [n_patches, C·ps·ps]`, which [`assemble_naflex_hidden`]
//! turns into the `[max_patches, width]` graph input via the Linear
//! `patch_embedding` and a **per-image bilinear-antialias resize** of the
//! `√num_patches × √num_patches` position grid. Padding is masked out by the
//! binary key-padding mask from [`build_key_mask`] (`MaskKind::Custom`).
//!
//! Patch row layout matches HF `convert_image_to_patches`: `image[C,H,W]`
//! → `(nph, npw, ps_h, ps_w, C)` → flatten, i.e. element `d = (ry·ps + rx)·C + c`.

use crate::config::{SIGLIP_MEAN, SIGLIP_STD, Siglip2Config};
use anyhow::{Result, ensure};
use rlx_core::image_preprocess::{Filter, pil_resize_rgb8};
use rlx_core::weight_map::WeightMap;

/// NaFlex vision-embed weights extracted from the checkpoint (host).
pub struct NaflexEmbedWeights {
    /// `patch_embedding.weight` `[hidden, patch_dim]` (nn.Linear, as-is).
    pub patch_w: Vec<f32>,
    /// `patch_embedding.bias` `[hidden]`.
    pub patch_b: Vec<f32>,
    /// `position_embedding.weight` as a `[side · side · hidden]` grid.
    pub pos_grid: Vec<f32>,
    pub side: usize,
    pub hidden: usize,
    pub patch_dim: usize,
    pub patch_size: usize,
}

/// Result of the NaFlex image processor for one image.
pub struct NaflexInput {
    /// `pixel_values` `[max_patches · patch_dim]`, zero-padded past `n_valid`.
    pub pixel_values: Vec<f32>,
    /// Patch-grid height / width for this image.
    pub nph: usize,
    pub npw: usize,
    /// Valid patch count (`nph · npw`).
    pub n_valid: usize,
    pub max_patches: usize,
}

pub(crate) fn extract_naflex_embed_weights(
    weights: &mut WeightMap,
    cfg: &Siglip2Config,
) -> Result<NaflexEmbedWeights> {
    let hidden = cfg.vision.width;
    let patch_size = cfg.vision.patch_size;
    let patch_dim = 3 * patch_size * patch_size;
    let side = cfg.vision.pos_grid_side();

    let (patch_w, w_shape) = weights.take("vision_model.embeddings.patch_embedding.weight")?;
    ensure!(
        w_shape == vec![hidden, patch_dim],
        "naflex patch_embedding.weight {w_shape:?} != [{hidden}, {patch_dim}]"
    );
    let (patch_b, _) = weights.take("vision_model.embeddings.patch_embedding.bias")?;
    ensure!(
        patch_b.len() == hidden,
        "naflex patch_embedding.bias != hidden"
    );

    let (pos_grid, p_shape) = weights.take("vision_model.embeddings.position_embedding.weight")?;
    ensure!(
        p_shape == vec![side * side, hidden],
        "naflex position_embedding.weight {p_shape:?} != [{}, {hidden}]",
        side * side
    );

    Ok(NaflexEmbedWeights {
        patch_w,
        patch_b,
        pos_grid,
        side,
        hidden,
        patch_dim,
        patch_size,
    })
}

/// Port of HF `get_image_size_for_max_num_patches`: the largest
/// aspect-preserving `(target_h, target_w)` (each a multiple of `patch_size`)
/// whose patch count fits `max_num_patches`.
pub fn get_image_size_for_max_num_patches(
    image_height: usize,
    image_width: usize,
    patch_size: usize,
    max_num_patches: usize,
) -> (usize, usize) {
    let eps = 1e-5_f64;
    let scaled = |scale: f64, size: usize| -> usize {
        let s = (size as f64) * scale;
        let s = (s / patch_size as f64).ceil() * patch_size as f64;
        (s as usize).max(patch_size)
    };
    let (mut scale_min, mut scale_max) = (eps / 10.0, 100.0_f64);
    while (scale_max - scale_min) >= eps {
        let scale = (scale_min + scale_max) / 2.0;
        let th = scaled(scale, image_height);
        let tw = scaled(scale, image_width);
        let num = (th / patch_size) * (tw / patch_size);
        if num <= max_num_patches {
            scale_min = scale;
        } else {
            scale_max = scale;
        }
    }
    (
        scaled(scale_min, image_height),
        scaled(scale_min, image_width),
    )
}

/// NaFlex image processor: resize (aspect-preserving, bilinear-antialias) to
/// fit `max_num_patches`, normalize (mean = std = 0.5), patchify, pad.
pub fn preprocess(
    rgb: &[u8],
    h_in: usize,
    w_in: usize,
    patch_size: usize,
    max_num_patches: usize,
) -> Result<NaflexInput> {
    ensure!(rgb.len() == h_in * w_in * 3, "rgb len != h*w*3");
    let (th, tw) = get_image_size_for_max_num_patches(h_in, w_in, patch_size, max_num_patches);
    let resized = pil_resize_rgb8(rgb, w_in, h_in, tw, th, Filter::Bilinear);

    let nph = th / patch_size;
    let npw = tw / patch_size;
    let n_valid = nph * npw;
    let patch_dim = 3 * patch_size * patch_size;
    let mut pixel_values = vec![0f32; max_num_patches * patch_dim];

    for py in 0..nph {
        for px in 0..npw {
            let row = py * npw + px;
            let base = row * patch_dim;
            for ry in 0..patch_size {
                let y = py * patch_size + ry;
                for rx in 0..patch_size {
                    let x = px * patch_size + rx;
                    let src = (y * tw + x) * 3;
                    for c in 0..3 {
                        // d = (ry·ps + rx)·C + c  (HF convert_image_to_patches).
                        let d = (ry * patch_size + rx) * 3 + c;
                        let v = resized[src + c] as f32 / 255.0;
                        pixel_values[base + d] = (v - SIGLIP_MEAN[c]) / SIGLIP_STD[c];
                    }
                }
            }
        }
    }

    Ok(NaflexInput {
        pixel_values,
        nph,
        npw,
        n_valid,
        max_patches: max_num_patches,
    })
}

/// Build the `[max_patches · hidden]` graph input: `patch_embedding(pv) +
/// resize(position_grid → nph×npw)`. Padded rows carry `bias + pos[0]` (HF
/// convention) but are masked out downstream.
pub fn assemble_naflex_hidden(pre: &NaflexEmbedWeights, input: &NaflexInput) -> Vec<f32> {
    let h = pre.hidden;
    let pd = pre.patch_dim;
    let max = input.max_patches;
    let mut hidden = vec![0f32; max * h];

    // Linear patch embedding over all rows (padded rows → bias only).
    for p in 0..max {
        let pv = &input.pixel_values[p * pd..(p + 1) * pd];
        let out = &mut hidden[p * h..(p + 1) * h];
        out.copy_from_slice(&pre.patch_b);
        for (d, &val) in pv.iter().enumerate() {
            if val == 0.0 {
                continue;
            }
            // patch_w row-major [hidden, patch_dim]: W[e, d] at e*pd + d.
            for (e, o) in out.iter_mut().enumerate() {
                *o += val * pre.patch_w[e * pd + d];
            }
        }
    }

    // Per-image position-embedding resize (bilinear-antialias) → add.
    let pos = resize_pos_grid(&pre.pos_grid, pre.side, h, input.nph, input.npw);
    for p in 0..max {
        let src = if p < input.n_valid { p } else { 0 };
        let out = &mut hidden[p * h..(p + 1) * h];
        let pe = &pos[src * h..(src + 1) * h];
        for (o, &pv) in out.iter_mut().zip(pe) {
            *o += pv;
        }
    }
    hidden
}

/// Binary key-padding mask `[seq]` (batch 1): `1.0` for valid patches, `0.0`
/// for padding. Consumed by `MaskKind::Custom` attention (shared by the
/// encoder and the MAP head — both mask the same keys).
pub fn build_key_mask(n_valid: usize, seq: usize) -> Vec<f32> {
    let mut mask = vec![0f32; seq];
    for m in mask.iter_mut().take(n_valid.min(seq)) {
        *m = 1.0;
    }
    mask
}

/// Separable float bilinear-antialias resize of a `[side·side·hidden]` grid to
/// `[h·w·hidden]`. Matches PyTorch `F.interpolate(mode="bilinear",
/// align_corners=False, antialias=True)` via Pillow's reducing-filter
/// coefficients (no inter-pass 8-bit rounding).
fn resize_pos_grid(grid: &[f32], side: usize, hidden: usize, h: usize, w: usize) -> Vec<f32> {
    let coeff_h = precompute_coeffs(side, h);
    let coeff_w = precompute_coeffs(side, w);
    let mut out = vec![0f32; h * w * hidden];
    for (oy, (iy0, wy)) in coeff_h.iter().enumerate() {
        for (ox, (ix0, wx)) in coeff_w.iter().enumerate() {
            let o = (oy * w + ox) * hidden;
            for (i, &why) in wy.iter().enumerate() {
                let iy = iy0 + i;
                for (j, &wxx) in wx.iter().enumerate() {
                    let ix = ix0 + j;
                    let coef = why * wxx;
                    if coef == 0.0 {
                        continue;
                    }
                    let src = (iy * side + ix) * hidden;
                    for e in 0..hidden {
                        out[o + e] += coef * grid[src + e];
                    }
                }
            }
        }
    }
    out
}

/// Pillow `precompute_coeffs` (triangle/bilinear filter), antialiased for
/// downscale (`filterscale = max(1, in/out)`). Returns `(window_start, weights)`.
fn precompute_coeffs(in_size: usize, out_size: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 1.0 * filterscale; // bilinear support = 1.0
    let inv = 1.0 / filterscale;
    let kernel = |x: f64| -> f64 {
        let x = x.abs();
        if x < 1.0 { 1.0 - x } else { 0.0 }
    };
    let mut coeffs = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let mut xmin = (center - support + 0.5).floor() as isize;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5).floor() as isize;
        if xmax > in_size as isize {
            xmax = in_size as isize;
        }
        let xmin = xmin as usize;
        let n = (xmax as usize).saturating_sub(xmin);
        let mut weights = Vec::with_capacity(n);
        let mut total = 0.0f64;
        for i in 0..n {
            let wv = kernel(((xmin + i) as f64 - center + 0.5) * inv);
            weights.push(wv);
            total += wv;
        }
        if total != 0.0 {
            for wv in &mut weights {
                *wv /= total;
            }
        }
        coeffs.push((xmin, weights.into_iter().map(|x| x as f32).collect()));
    }
    coeffs
}
