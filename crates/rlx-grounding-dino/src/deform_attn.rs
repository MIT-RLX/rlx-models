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

//! Multi-scale deformable attention (CPU-native), matching HF
//! `GroundingDinoMultiscaleDeformableAttention` + `multi_scale_deformable_attention`.

use crate::nn::{self, softmax_rows};
use crate::weights::get;
use anyhow::Result;
use rlx_core::weight_map::WeightMap;

/// Spatial size of one feature level.
#[derive(Debug, Clone, Copy)]
pub struct LevelShape {
    pub h: usize,
    pub w: usize,
}

/// Reference points for the queries.
pub enum RefPoints<'a> {
    /// `[nq, n_levels, 2]` normalized centers (encoder self-attention).
    Two(&'a [f32]),
    /// `[nq, n_levels, 4]` normalized boxes cxcywh (decoder cross-attention).
    Four(&'a [f32]),
}

/// Multi-scale deformable attention module weights.
pub struct MsDeformAttn {
    value_proj_w: Vec<f32>,
    value_proj_b: Vec<f32>,
    sampling_offsets_w: Vec<f32>,
    sampling_offsets_b: Vec<f32>,
    attention_weights_w: Vec<f32>,
    attention_weights_b: Vec<f32>,
    output_proj_w: Vec<f32>,
    output_proj_b: Vec<f32>,
    d: usize,
    n_heads: usize,
    #[allow(dead_code)] // kept for config symmetry; forward derives nl from shapes
    n_levels: usize,
    n_points: usize,
}

impl MsDeformAttn {
    /// Number of sampling points per head/level.
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// Clone the eight projection tensors in canonical order:
    /// `(value_w, value_b, samp_w, samp_b, attw_w, attw_b, out_w, out_b)`.
    #[allow(clippy::type_complexity)]
    pub fn clone_proj(
        &self,
    ) -> (
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    ) {
        (
            self.value_proj_w.clone(),
            self.value_proj_b.clone(),
            self.sampling_offsets_w.clone(),
            self.sampling_offsets_b.clone(),
            self.attention_weights_w.clone(),
            self.attention_weights_b.clone(),
            self.output_proj_w.clone(),
            self.output_proj_b.clone(),
        )
    }

    pub fn from_weights(
        wm: &WeightMap,
        prefix: &str,
        d: usize,
        n_heads: usize,
        n_levels: usize,
        n_points: usize,
    ) -> Result<Self> {
        Ok(Self {
            value_proj_w: get(wm, &format!("{prefix}value_proj.weight"))?,
            value_proj_b: get(wm, &format!("{prefix}value_proj.bias"))?,
            sampling_offsets_w: get(wm, &format!("{prefix}sampling_offsets.weight"))?,
            sampling_offsets_b: get(wm, &format!("{prefix}sampling_offsets.bias"))?,
            attention_weights_w: get(wm, &format!("{prefix}attention_weights.weight"))?,
            attention_weights_b: get(wm, &format!("{prefix}attention_weights.bias"))?,
            output_proj_w: get(wm, &format!("{prefix}output_proj.weight"))?,
            output_proj_b: get(wm, &format!("{prefix}output_proj.bias"))?,
            d,
            n_heads,
            n_levels,
            n_points,
        })
    }

    /// Build from explicit parts (tests).
    #[allow(clippy::too_many_arguments)]
    #[cfg(test)]
    pub(crate) fn from_parts(
        d: usize,
        n_heads: usize,
        n_levels: usize,
        n_points: usize,
        value_proj_w: Vec<f32>,
        value_proj_b: Vec<f32>,
        sampling_offsets_w: Vec<f32>,
        sampling_offsets_b: Vec<f32>,
        attention_weights_w: Vec<f32>,
        attention_weights_b: Vec<f32>,
        output_proj_w: Vec<f32>,
        output_proj_b: Vec<f32>,
    ) -> Self {
        Self {
            value_proj_w,
            value_proj_b,
            sampling_offsets_w,
            sampling_offsets_b,
            attention_weights_w,
            attention_weights_b,
            output_proj_w,
            output_proj_b,
            d,
            n_heads,
            n_levels,
            n_points,
        }
    }

    /// `query` is `[nq, d]` (position embedding already added by the caller where
    /// required). `value_src` is `[seq, d]` with `seq == sum(h*w)`. `value_mask`
    /// (optional `[seq]`, 1 = valid) zeroes padded value rows. Returns `[nq, d]`.
    pub fn forward(
        &self,
        query: &[f32],
        value_src: &[f32],
        ref_points: &RefPoints<'_>,
        shapes: &[LevelShape],
        level_start: &[usize],
        value_mask: Option<&[u8]>,
    ) -> Vec<f32> {
        let (ref_slice, ref_dim) = match ref_points {
            RefPoints::Two(rp) => (*rp, 2usize),
            RefPoints::Four(rp) => (*rp, 4usize),
        };
        let w = DeformWeights {
            value_proj_w: &self.value_proj_w,
            value_proj_b: &self.value_proj_b,
            sampling_offsets_w: &self.sampling_offsets_w,
            sampling_offsets_b: &self.sampling_offsets_b,
            attention_weights_w: &self.attention_weights_w,
            attention_weights_b: &self.attention_weights_b,
            output_proj_w: &self.output_proj_w,
            output_proj_b: &self.output_proj_b,
        };
        deform_forward(
            query,
            value_src,
            ref_slice,
            ref_dim,
            shapes,
            level_start,
            self.d,
            self.n_heads,
            self.n_points,
            &w,
            value_mask,
        )
    }
}

/// The eight projection tensors of a deformable-attention module.
pub struct DeformWeights<'a> {
    pub value_proj_w: &'a [f32],
    pub value_proj_b: &'a [f32],
    pub sampling_offsets_w: &'a [f32],
    pub sampling_offsets_b: &'a [f32],
    pub attention_weights_w: &'a [f32],
    pub attention_weights_b: &'a [f32],
    pub output_proj_w: &'a [f32],
    pub output_proj_b: &'a [f32],
}

/// Fused multi-scale deformable attention forward (memory-efficient: accumulates
/// per query/head without materializing the sampled corners). This is the shared
/// reference used by the native module and the on-device custom-op kernel.
/// `ref_points` is `[nq, n_levels, ref_dim]` with `ref_dim` 2 (centers) or 4
/// (cxcywh boxes). Returns `[nq, d]`.
#[allow(clippy::too_many_arguments)]
pub fn deform_forward(
    query: &[f32],
    value_src: &[f32],
    ref_points: &[f32],
    ref_dim: usize,
    shapes: &[LevelShape],
    level_start: &[usize],
    d: usize,
    nh: usize,
    np: usize,
    w: &DeformWeights<'_>,
    value_mask: Option<&[u8]>,
) -> Vec<f32> {
    let nl = shapes.len();
    let hd = d / nh;
    let nq = query.len() / d;
    let seq = value_src.len() / d;

    let mut value = nn::linear(value_src, seq, d, w.value_proj_w, d, w.value_proj_b);
    if let Some(mask) = value_mask {
        for s in 0..seq {
            if mask[s] == 0 {
                for c in 0..d {
                    value[s * d + c] = 0.0;
                }
            }
        }
    }

    let offsets = nn::linear(
        query,
        nq,
        d,
        w.sampling_offsets_w,
        nh * nl * np * 2,
        w.sampling_offsets_b,
    );
    let mut attn = nn::linear(
        query,
        nq,
        d,
        w.attention_weights_w,
        nh * nl * np,
        w.attention_weights_b,
    );
    softmax_rows(&mut attn, nq * nh, nl * np);

    let mut out = vec![0f32; nq * d];
    for q in 0..nq {
        for m in 0..nh {
            let mut acc = vec![0f32; hd];
            for l in 0..nl {
                let LevelShape { h, w: lw } = shapes[l];
                let base = level_start[l];
                for p in 0..np {
                    let off_base = (((q * nh + m) * nl + l) * np + p) * 2;
                    let off_x = offsets[off_base];
                    let off_y = offsets[off_base + 1];
                    let rb = (q * nl + l) * ref_dim;
                    let (loc_x, loc_y) = if ref_dim == 2 {
                        (
                            ref_points[rb] + off_x / lw as f32,
                            ref_points[rb + 1] + off_y / h as f32,
                        )
                    } else {
                        (
                            ref_points[rb] + off_x / np as f32 * ref_points[rb + 2] * 0.5,
                            ref_points[rb + 1] + off_y / np as f32 * ref_points[rb + 3] * 0.5,
                        )
                    };
                    let aw = attn[(q * nh + m) * (nl * np) + l * np + p];
                    if aw == 0.0 {
                        continue;
                    }
                    grid_sample_accumulate(
                        &value, seq, d, base, h, lw, m, hd, loc_x, loc_y, aw, &mut acc,
                    );
                }
            }
            for c in 0..hd {
                out[q * d + m * hd + c] = acc[c];
            }
        }
    }
    nn::linear(&out, nq, d, w.output_proj_w, d, w.output_proj_b)
}

/// Bilinear-sample one level's value (head `m`) at normalized location and add
/// `weight * sample` into `acc[hd]`. `value` is `[seq, nh*hd]`, level rows start
/// at `base` and span `h*w` in row-major (y, x) order.
#[allow(clippy::too_many_arguments)]
fn grid_sample_accumulate(
    value: &[f32],
    _seq: usize,
    d: usize,
    base: usize,
    h: usize,
    w: usize,
    m: usize,
    hd: usize,
    loc_x: f32,
    loc_y: f32,
    weight: f32,
    acc: &mut [f32],
) {
    // normalized [0,1] → grid [-1,1] → pixel (align_corners=False).
    let gx = 2.0 * loc_x - 1.0;
    let gy = 2.0 * loc_y - 1.0;
    let ix = ((gx + 1.0) * w as f32 - 1.0) * 0.5;
    let iy = ((gy + 1.0) * h as f32 - 1.0) * 0.5;
    let x0 = ix.floor() as isize;
    let y0 = iy.floor() as isize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let wx1 = ix - x0 as f32;
    let wy1 = iy - y0 as f32;
    let wx0 = 1.0 - wx1;
    let wy0 = 1.0 - wy1;
    let corners = [
        (y0, x0, wy0 * wx0),
        (y0, x1, wy0 * wx1),
        (y1, x0, wy1 * wx0),
        (y1, x1, wy1 * wx1),
    ];
    for (cy, cx, cw) in corners {
        if cy < 0 || cx < 0 || cy >= h as isize || cx >= w as isize {
            continue; // zero padding
        }
        let row = base + (cy as usize) * w + (cx as usize);
        let voff = row * d + m * hd;
        let cw = cw * weight;
        for c in 0..hd {
            acc[c] += cw * value[voff + c];
        }
    }
}

/// `[(h,w)]` → per-level start index into the flattened `[sum(h*w)]` sequence.
pub fn level_start_index(shapes: &[LevelShape]) -> Vec<usize> {
    let mut starts = Vec::with_capacity(shapes.len());
    let mut acc = 0;
    for s in shapes {
        starts.push(acc);
        acc += s.h * s.w;
    }
    starts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(d: usize) -> Vec<f32> {
        let mut m = vec![0f32; d * d];
        for i in 0..d {
            m[i * d + i] = 1.0;
        }
        m
    }

    #[test]
    fn constant_value_field_returns_constant() {
        // value_proj = identity, output_proj = identity, sampling offsets/weights
        // logits all zero (uniform softmax), reference points at center → all
        // samples land in-bounds on a constant field → output equals the constant.
        let (d, nh, nl, np) = (4, 2, 1, 4);
        let m = MsDeformAttn::from_parts(
            d,
            nh,
            nl,
            np,
            identity(d),
            vec![0.0; d],
            vec![0.0; (nh * nl * np * 2) * d],
            vec![0.0; nh * nl * np * 2],
            vec![0.0; (nh * nl * np) * d],
            vec![0.0; nh * nl * np],
            identity(d),
            vec![0.0; d],
        );
        let shapes = [LevelShape { h: 5, w: 5 }];
        let starts = level_start_index(&shapes);
        let seq = 25;
        let value_src = vec![3.0f32; seq * d]; // constant field
        let nq = 2;
        let query = vec![0.0f32; nq * d];
        // reference points at center (0.5, 0.5) for each level.
        let rp = vec![0.5f32; nq * nl * 2];
        let out = m.forward(
            &query,
            &value_src,
            &RefPoints::Two(&rp),
            &shapes,
            &starts,
            None,
        );
        assert_eq!(out.len(), nq * d);
        for v in out {
            assert!((v - 3.0).abs() < 1e-4, "expected 3.0, got {v}");
        }
    }

    #[test]
    fn out_of_bounds_samples_zero() {
        // Push reference points far outside [0,1] → grid_sample zero padding →
        // output is just the (zero) bias.
        let (d, nh, nl, np) = (2, 1, 1, 1);
        let m = MsDeformAttn::from_parts(
            d,
            nh,
            nl,
            np,
            identity(d),
            vec![0.0; d],
            vec![0.0; (nh * nl * np * 2) * d],
            vec![0.0; nh * nl * np * 2],
            vec![0.0; (nh * nl * np) * d],
            vec![0.0; nh * nl * np],
            identity(d),
            vec![0.0; d],
        );
        let shapes = [LevelShape { h: 3, w: 3 }];
        let starts = level_start_index(&shapes);
        let value_src = vec![5.0f32; 9 * d];
        let query = vec![0.0f32; d];
        let rp = vec![10.0f32, 10.0]; // way out of bounds
        let out = m.forward(
            &query,
            &value_src,
            &RefPoints::Two(&rp),
            &shapes,
            &starts,
            None,
        );
        assert!(out.iter().all(|&v| v.abs() < 1e-6));
    }
}
