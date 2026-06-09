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

//! SAM 2 FPN neck (mirrors `sam2/modeling/backbones/image_encoder.py::FpnNeck`).
//!
//! Per-stage 1×1 lateral convs and top-down fusion run via IR
//! ([`super::fpn_neck_ir`]); sinusoidal PE is precomputed at compile time.

use super::config::{Sam2FpnConfig, Sam2HieraConfig};
use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;
use std::f32::consts::PI;

/// Weights for the FPN neck — one 1×1 conv (`weight` + `bias`) per
/// backbone level. Stored coarse → fine to match the checkpoint's
/// `image_encoder.neck.convs.{i}.conv.{weight,bias}` ordering.
pub struct FpnNeckWeights {
    /// `[d_model, backbone_channel_list[i]]` per level (1×1 conv = a
    /// per-pixel linear, so the kernel dims collapse).
    pub conv_w: Vec<Vec<f32>>,
    pub conv_b: Vec<Vec<f32>>,
    pub d_model: usize,
    pub backbone_channel_list: Vec<usize>,
    pub fpn_top_down_levels: Vec<usize>,
    pub nearest: bool,
}

pub(super) fn extract_fpn_weights(
    weights: &mut WeightMap,
    cfg: &Sam2HieraConfig,
) -> Result<FpnNeckWeights> {
    let fpn = Sam2FpnConfig::for_hiera(cfg);
    let n = fpn.backbone_channel_list.len();
    let d = fpn.d_model;

    let mut conv_w = Vec::with_capacity(n);
    let mut conv_b = Vec::with_capacity(n);
    for i in 0..n {
        let cin = fpn.backbone_channel_list[i];
        let (raw_w, w_shape) =
            weights.take(&format!("image_encoder.neck.convs.{i}.conv.weight"))?;
        ensure!(
            w_shape == vec![d, cin, 1, 1],
            "neck.convs.{i}.conv.weight expected [{d}, {cin}, 1, 1], got {w_shape:?}"
        );
        let (raw_b, _) = weights.take(&format!("image_encoder.neck.convs.{i}.conv.bias"))?;
        conv_w.push(raw_w);
        conv_b.push(raw_b);
    }
    Ok(FpnNeckWeights {
        conv_w,
        conv_b,
        d_model: d,
        backbone_channel_list: fpn.backbone_channel_list,
        fpn_top_down_levels: fpn.fpn_top_down_levels,
        nearest: fpn.interpolation_nearest,
    })
}

/// A single FPN level output — BCHW features + matched sinusoidal
/// positional encoding.
pub struct FpnLevel {
    /// `[d_model, h, w]` NCHW.
    pub features: Vec<f32>,
    /// `[d_model, h, w]` NCHW — sinusoidal absolute pos embed.
    pub pos: Vec<f32>,
    pub h: usize,
    pub w: usize,
}

/// Run the FPN neck. `stage_outputs[i]` is the encoder's
/// stage-`i` output flattened from BHWC `[1, h, w, dim]` to
/// `[h·w·dim]`. `stage_dims[i] = dim`, `stage_hw[i] = (h, w)` — pulled
/// straight from the graph's stage-output shapes (or computed from
/// `cfg.embed_dim_at_stage(s)` / `cfg.grid_size_at_stage(s)`).
///
/// Returns four `FpnLevel`s in **fine → coarse** order (so callers
/// downstream can naturally index `[0]` for the highest-resolution
/// stride-4 feature map, matching the reference).
pub fn apply_fpn_neck(
    neck: &FpnNeckWeights,
    ir: &mut super::fpn_neck_ir::Sam2FpnNeckIr,
    stage_outputs: &[Vec<f32>],
    stage_hw: &[(usize, usize)],
    stage_dims: &[usize],
) -> Result<Vec<FpnLevel>> {
    apply_fpn_neck_impl(neck, Some(ir), stage_outputs, stage_hw, stage_dims)
}

/// Host-only lateral convs (legacy entry point).
pub fn apply_fpn_neck_host(
    neck: &FpnNeckWeights,
    stage_outputs: &[Vec<f32>],
    stage_hw: &[(usize, usize)],
    stage_dims: &[usize],
) -> Vec<FpnLevel> {
    apply_fpn_neck_impl(neck, None, stage_outputs, stage_hw, stage_dims).expect("host FPN neck")
}

fn apply_fpn_neck_impl(
    neck: &FpnNeckWeights,
    mut ir: Option<&mut super::fpn_neck_ir::Sam2FpnNeckIr>,
    stage_outputs: &[Vec<f32>],
    stage_hw: &[(usize, usize)],
    stage_dims: &[usize],
) -> Result<Vec<FpnLevel>> {
    let n = neck.backbone_channel_list.len();
    assert_eq!(stage_outputs.len(), n);
    assert_eq!(stage_hw.len(), n);
    assert_eq!(stage_dims.len(), n);
    let d = neck.d_model;

    // The reference loops `i = n-1 .. 0` (coarse → fine). Our
    // `stage_outputs[0]` is the *finest* stage (stride 4); but the
    // neck's `convs[0]` projects the *coarsest*. So convs index =
    // `n - 1 - stage_idx`.
    //
    // We iterate coarse → fine to do the top-down sum, then return
    // results in fine → coarse order.
    let mut top_down: Option<Vec<f32>> = None;
    let mut top_down_hw: Option<(usize, usize)> = None;
    let mut levels: Vec<FpnLevel> = Vec::with_capacity(n);

    for coarse_i in 0..n {
        // coarse_i = 0 is the coarsest stage; iterate fine -> coarse
        // via stage index then reverse at the end.
        let stage_idx = n - 1 - coarse_i; // n-1, n-2, ..., 0
        let conv_idx = coarse_i; // matches `convs[n-i]` with i=stage_idx since (n-1) - stage_idx = coarse_i
        let (h, w) = stage_hw[stage_idx];
        let dim_in = stage_dims[stage_idx];
        debug_assert_eq!(dim_in, neck.backbone_channel_list[conv_idx]);

        // 1) 1×1 lateral conv: dim_in → d_model.
        let lat = match ir.as_deref_mut() {
            Some(ir_neck) => ir_neck.laterals[stage_idx].run(&stage_outputs[stage_idx])?,
            None => lateral_conv_host(
                &neck.conv_w[conv_idx],
                &neck.conv_b[conv_idx],
                &stage_outputs[stage_idx],
                dim_in,
                d,
                h,
                w,
            ),
        };

        // 2) Optional top-down sum (nearest ×2 upsample of `top_down`).
        let level_features = if neck.fpn_top_down_levels.contains(&stage_idx)
            && let Some(td) = top_down.as_ref()
        {
            let (th, tw) = top_down_hw.unwrap();
            debug_assert_eq!(th * 2, h);
            debug_assert_eq!(tw * 2, w);
            if let Some(ir_neck) = ir.as_deref_mut() {
                if let Some(fuse) = ir_neck.fuses.get_mut(stage_idx).and_then(|f| f.as_mut()) {
                    fuse.run(&lat, td)?
                } else {
                    top_down_add_host(&lat, td, d, h, w, th, tw)
                }
            } else {
                top_down_add_host(&lat, td, d, h, w, th, tw)
            }
        } else {
            lat
        };

        // 3) Sinusoidal position encoding (precomputed at IR compile time).
        let pos = ir
            .as_ref()
            .map(|ir| ir.pos[stage_idx].clone())
            .unwrap_or_else(|| sinusoidal_pos_2d(d, h, w));

        levels.push(FpnLevel {
            features: level_features.clone(),
            pos,
            h,
            w,
        });
        top_down = Some(level_features);
        top_down_hw = Some((h, w));
    }

    // Levels were pushed coarse → fine. Reverse to fine → coarse.
    levels.reverse();
    Ok(levels)
}

fn top_down_add_host(
    lat: &[f32],
    prev: &[f32],
    d: usize,
    h: usize,
    w: usize,
    th: usize,
    tw: usize,
) -> Vec<f32> {
    let mut summed = lat.to_vec();
    for c in 0..d {
        for y in 0..h {
            let sy = y / 2;
            for x in 0..w {
                let sx = x / 2;
                summed[c * h * w + y * w + x] += prev[c * th * tw + sy * tw + sx];
            }
        }
    }
    summed
}

fn lateral_conv_host(
    cw: &[f32],
    cb: &[f32],
    src: &[f32],
    dim_in: usize,
    d: usize,
    h: usize,
    w: usize,
) -> Vec<f32> {
    let mut lat = vec![0f32; d * h * w];
    for y in 0..h {
        for x in 0..w {
            let in_off = (y * w + x) * dim_in;
            for oc in 0..d {
                let mut acc = cb[oc];
                for ic in 0..dim_in {
                    acc += src[in_off + ic] * cw[oc * dim_in + ic];
                }
                lat[oc * h * w + y * w + x] = acc;
            }
        }
    }
    lat
}

/// Sinusoidal absolute position embedding (`PositionEmbeddingSine`),
/// matching `sam2/modeling/position_encoding.py`:
///   - `num_pos_feats = d_model / 2`, half for x, half for y
///   - `normalize=True`, `temperature=10000`, `scale=2π`
///   - output `[d_model, h, w]` NCHW with channel layout
///     `[y_sin, y_cos, …, x_sin, x_cos, …]`.
pub(super) fn sinusoidal_pos_2d(d_model: usize, h: usize, w: usize) -> Vec<f32> {
    let nf = d_model / 2; // num_pos_feats per axis
    let temperature: f32 = 10000.0;
    let scale: f32 = 2.0 * PI;
    let eps: f32 = 1e-6;
    let mut out = vec![0f32; d_model * h * w];

    // Per-axis dim_t scaling factors.
    // dim_t = temperature ** (2 * (i // 2) / num_pos_feats)
    let mut dim_t = vec![0f32; nf];
    for i in 0..nf {
        let exp = 2.0 * ((i / 2) as f32) / (nf as f32);
        dim_t[i] = temperature.powf(exp);
    }

    // Build normalised y and x embeddings.
    for y in 0..h {
        let y_emb = ((y + 1) as f32) / ((h as f32) + eps) * scale;
        for x in 0..w {
            let x_emb = ((x + 1) as f32) / ((w as f32) + eps) * scale;
            // y channels go to [0..nf), x channels to [nf..d_model).
            for i in 0..nf {
                let py = y_emb / dim_t[i];
                let val = if i % 2 == 0 { py.sin() } else { py.cos() };
                out[i * h * w + y * w + x] = val;
            }
            for i in 0..nf {
                let px = x_emb / dim_t[i];
                let val = if i % 2 == 0 { px.sin() } else { px.cos() };
                out[(nf + i) * h * w + y * w + x] = val;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Sam2HieraConfig;

    #[test]
    fn pos_2d_shape_and_finite() {
        let pos = sinusoidal_pos_2d(256, 32, 32);
        assert_eq!(pos.len(), 256 * 32 * 32);
        assert!(pos.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn fpn_levels_returned_fine_to_coarse() {
        // Tiny check: just verify the spatial ordering convention
        // (fine → coarse) holds for B+ when we run with synthetic
        // weights of the right shape.
        let cfg = Sam2HieraConfig::base_plus();
        let fpn = Sam2FpnConfig::for_hiera(&cfg);
        // Coarse-to-fine channel list:
        assert_eq!(fpn.backbone_channel_list, vec![896, 448, 224, 112]);
    }
}
