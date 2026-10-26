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

//! The MiniCPM-V 4.6 / 4.7 vision path, one crop at a time:
//!
//! ```text
//!   patches ─ patch_embedding + nearest-bucket position embedding
//!   SigLIP layers 0..=insert_layer_id            (S tokens)
//!   window merger: 2×2 window attention, concat-MLP + window mean  (S/4)
//!   SigLIP layers insert_layer_id+1..            (S/4 tokens)
//!   post_layernorm
//!   merger: 2×2 concat → LayerNorm → Linear → GELU → Linear        (S/16)
//! ```
//!
//! HF packs every crop of a prompt into one sequence and attends within each
//! crop (`cu_seqlens`); crops never see each other, so running them one at a
//! time is the same computation.
//!
//! **Shape buckets.** A graph is compiled for a padded patch count `S` (a
//! multiple of [`BUCKET`]) and serves any crop with at most `S` patches: the
//! grid geometry arrives as data, not shape. Padding rows are masked out of
//! every attention, both 2×2 regroupings are row gathers whose indices the
//! host computes from the grid, and real rows always come first, so the
//! output's leading rows are the crop's tokens. Typical photos slice into
//! crops of ≤ 1024 patches, so one graph covers them.

use anyhow::{Result, anyhow, bail, ensure};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, Emit, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::config::{MiniCpmVConfig, Projector};
pub use crate::feeds::*;
use crate::feeds::{
    IN_FEATS, IN_MASK, IN_MASK2, IN_MERGE, IN_PATCHES, IN_POS, IN_RMASK, IN_RPOS, IN_WIN,
};

/// Patch-count granularity of the compiled buckets.
pub const TOWER: &str = "model.vision_tower";
pub const MERGER: &str = "model.merger.mlp.0";
pub const RESAMPLER: &str = "resampler";
pub const PATCH_LINEAR: &str = "model.vision_tower.embeddings.patch_embedding.linear.weight";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionPart {
    /// Tower and projector: patches in, LM rows out.
    Full,
    /// The tower alone: patches in, post-LayerNorm features `[1, s, c]` out.
    Tower,
    /// The resampler alone over `s` rows of tower features (several frames'
    /// worth for 4.5's packed video), positions and key mask fed by the host.
    Resampler,
}

/// `x @ Wᵀ + b` with HF `nn.Linear` params under `prefix`.
fn linear(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId, dt: DType) -> Result<HirNodeId> {
    let w = emit.load_param_typed(&format!("{prefix}.weight"), true, dt)?;
    let b = emit.load_param(&format!("{prefix}.bias"), false)?;
    let mut gb = HirMut::new(emit.hir());
    let mm = gb.mm(x, w);
    Ok(gb.add(mm, b))
}

fn layer_norm(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId, eps: f32) -> Result<HirNodeId> {
    let g = emit.load_param(&format!("{prefix}.weight"), false)?;
    let b = emit.load_param(&format!("{prefix}.bias"), false)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.ln(x, g, b, eps))
}

/// Multi-head self-attention with separate q/k/v/out projections (all with
/// bias), no RoPE. `x` is `[batch, seq, width]`; `mask` a key-padding mask
/// `[batch, seq]` or `None` for full attention.
fn self_attention(
    emit: &mut Emit<'_>,
    prefix: &str,
    x: HirNodeId,
    width: usize,
    heads: usize,
    mask: Option<HirNodeId>,
    dt: DType,
) -> Result<HirNodeId> {
    let q = linear(emit, &format!("{prefix}.q_proj"), x, dt)?;
    let k = linear(emit, &format!("{prefix}.k_proj"), x, dt)?;
    let v = linear(emit, &format!("{prefix}.v_proj"), x, dt)?;
    let attn = {
        let mut gb = HirMut::new(emit.hir());
        let shape = rlx_ir::shape::attention_shape(gb.shape(q));
        let hd = width / heads;
        match mask {
            Some(m) => gb.attention(q, k, v, m, heads, hd, shape),
            None => gb.attention_kind(q, k, v, heads, hd, MaskKind::None, shape),
        }
    };
    linear(emit, &format!("{prefix}.out_proj"), attn, dt)
}

/// One pre-LN SigLIP encoder layer (`gelu_pytorch_tanh` MLP).
fn encoder_layer(
    emit: &mut Emit<'_>,
    prefix: &str,
    x: HirNodeId,
    cfg: &MiniCpmVConfig,
    mask: HirNodeId,
    dt: DType,
) -> Result<HirNodeId> {
    let v = &cfg.vision;
    let eps = v.layer_norm_eps;
    let n1 = layer_norm(emit, &format!("{prefix}.layer_norm1"), x, eps)?;
    let a = self_attention(
        emit,
        &format!("{prefix}.self_attn"),
        n1,
        v.hidden_size,
        v.num_attention_heads,
        Some(mask),
        dt,
    )?;
    let r1 = HirMut::new(emit.hir()).add(x, a);
    let n2 = layer_norm(emit, &format!("{prefix}.layer_norm2"), r1, eps)?;
    let h = linear(emit, &format!("{prefix}.mlp.fc1"), n2, dt)?;
    let h = HirMut::new(emit.hir()).gelu_approx(h);
    let h = linear(emit, &format!("{prefix}.mlp.fc2"), h, dt)?;
    Ok(HirMut::new(emit.hir()).add(r1, h))
}

/// Gather rows of `x [1, n, c]` in `idx [1, n]` order → `[1, n, c]`.
fn gather_rows(emit: &mut Emit<'_>, x: HirNodeId, idx: HirNodeId, n: usize, c: usize) -> HirNodeId {
    let mut gb = HirMut::new(emit.hir());
    let flat = gb.reshape_(x, vec![n as i64, c as i64]);
    let g = gb.gather_(flat, idx, 0);
    gb.reshape_(g, vec![1, n as i64, c as i64])
}

/// `MiniCPMV4_6ViTWindowAttentionMerger`: `x [1, s, c]` → `[1, s/4, c]`.
fn window_merger(
    emit: &mut Emit<'_>,
    x: HirNodeId,
    cfg: &MiniCpmVConfig,
    s: usize,
    dt: DType,
) -> Result<HirNodeId> {
    let v = &cfg.vision;
    let c = v.hidden_size;
    let win = v.window * v.window;
    let p = format!("{TOWER}.vit_merger");
    let idx = emit.flow_input(IN_WIN)?.hir_id();
    // Reorder first: LayerNorm is per row, so normalizing the reordered rows
    // is HF's `layer_norm1` followed by `[:, window_index]`.
    let xw = gather_rows(emit, x, idx, s, c);
    let n1 = layer_norm(emit, &format!("{p}.layer_norm1"), xw, v.layer_norm_eps)?;
    let nw = s / win;
    let n1b = HirMut::new(emit.hir()).reshape_(n1, vec![nw as i64, win as i64, c as i64]);
    // Each window is its own attention batch row (`cu_seqlens` of size 4).
    let a = self_attention(
        emit,
        &format!("{p}.self_attn"),
        n1b,
        c,
        v.num_attention_heads,
        None,
        dt,
    )?;
    let (flat, mean) = {
        let mut gb = HirMut::new(emit.hir());
        let a = gb.reshape_(a, vec![1, s as i64, c as i64]);
        let r = gb.add(xw, a);
        let grouped = gb.reshape_(r, vec![nw as i64, win as i64, c as i64]);
        let mean = gb.mean(grouped, vec![1], false);
        let flat = gb.reshape_(r, vec![nw as i64, (win * c) as i64]);
        (flat, mean)
    };
    let h = layer_norm(emit, &format!("{p}.pre_norm"), flat, v.layer_norm_eps)?;
    let h = linear(emit, &format!("{p}.linear_1"), h, dt)?;
    let h = HirMut::new(emit.hir()).gelu_approx(h);
    let h = linear(emit, &format!("{p}.linear_2"), h, dt)?;
    let mut gb = HirMut::new(emit.hir());
    let out = gb.add(h, mean);
    Ok(gb.reshape_(out, vec![1, nw as i64, c as i64]))
}

/// `MiniCPMV4_6Merger` (one round): `x [1, n, c]` → `[1, n/4, text_hidden]`.
fn merger(
    emit: &mut Emit<'_>,
    x: HirNodeId,
    cfg: &MiniCpmVConfig,
    n: usize,
    dt: DType,
) -> Result<HirNodeId> {
    let c = cfg.vision.hidden_size;
    let k = cfg.merge_kernel * cfg.merge_kernel;
    let idx = emit.flow_input(IN_MERGE)?.hir_id();
    let g = gather_rows(emit, x, idx, n, c);
    let flat = HirMut::new(emit.hir()).reshape_(g, vec![(n / k) as i64, (k * c) as i64]);
    // HF hard-codes this LayerNorm's eps.
    let h = layer_norm(emit, &format!("{MERGER}.pre_norm"), flat, 1e-6)?;
    let h = linear(emit, &format!("{MERGER}.linear_1"), h, dt)?;
    // `nn.GELU()`: the exact erf form, unlike the tower's tanh GELU.
    let h = HirMut::new(emit.hir()).gelu(h);
    let h = linear(emit, &format!("{MERGER}.linear_2"), h, dt)?;
    Ok(HirMut::new(emit.hir()).reshape_(h, vec![1, (n / k) as i64, cfg.text_hidden_size as i64]))
}

/// The 2.x–4.5 perceiver resampler: `x [1, s, c]` (tower output) →
/// `[1, queries, d]`. Learned queries (`ln_q`) cross-attend to the projected
/// patches (`kv_proj`, `ln_kv`) with the 2-D sin-cos positions added to the
/// keys only, then `ln_post` and `· proj`. Padding rows are masked out.
fn resampler(
    emit: &mut Emit<'_>,
    x: HirNodeId,
    cfg: &MiniCpmVConfig,
    queries: usize,
    dt: DType,
) -> Result<HirNodeId> {
    let d = cfg.text_hidden_size;
    let mask = emit.flow_input(IN_RMASK)?.hir_id();
    let heads = d / 128;
    let p = RESAMPLER;
    let pos = emit.flow_input(IN_RPOS)?.hir_id();
    let kv_w = emit.load_param_typed(&format!("{p}.kv_proj.weight"), true, dt)?;
    let kv = HirMut::new(emit.hir()).mm(x, kv_w);
    let kv = layer_norm(emit, &format!("{p}.ln_kv"), kv, 1e-6)?;
    let query = emit.load_param(&format!("{p}.query"), false)?;
    let q = layer_norm(emit, &format!("{p}.ln_q"), query, 1e-6)?;
    let q = HirMut::new(emit.hir()).reshape_(q, vec![1, queries as i64, d as i64]);
    let keys = HirMut::new(emit.hir()).add(kv, pos);
    let a = format!("{p}.attn");
    let q = linear(emit, &format!("{a}.q_proj"), q, dt)?;
    let k = linear(emit, &format!("{a}.k_proj"), keys, dt)?;
    let v = linear(emit, &format!("{a}.v_proj"), kv, dt)?;
    let attn = {
        let mut gb = HirMut::new(emit.hir());
        let shape = rlx_ir::shape::attention_shape(gb.shape(q));
        gb.attention(q, k, v, mask, heads, d / heads, shape)
    };
    let o = linear(emit, &format!("{a}.out_proj"), attn, dt)?;
    let o = layer_norm(emit, &format!("{p}.ln_post"), o, 1e-6)?;
    let proj = emit.load_param_typed(&format!("{p}.proj"), false, dt)?;
    Ok(HirMut::new(emit.hir()).mm(o, proj))
}

/// Build the vision graph for a bucket of `s` patches. `weights` must hold
/// the tower and merger tensors, with the patch conv already reshaped to
/// [`PATCH_LINEAR`]. Linear weights are stored as `weight_dtype` (F32, or
/// BF16 to keep the checkpoint's 2 B/elem); the math stays f32.
pub fn build_vision(
    cfg: &MiniCpmVConfig,
    weights: &mut WeightMap,
    s: usize,
    weight_dtype: DType,
) -> Result<BuiltModel> {
    build_vision_part(cfg, weights, s, weight_dtype, VisionPart::Full)
}

/// [`build_vision`] for one [`VisionPart`]. For [`VisionPart::Resampler`], `s`
/// is the number of feature rows the graph takes.
pub fn build_vision_part(
    cfg: &MiniCpmVConfig,
    weights: &mut WeightMap,
    s: usize,
    weight_dtype: DType,
    part: VisionPart,
) -> Result<BuiltModel> {
    ensure!(
        s.is_multiple_of(16),
        "minicpm-v: bucket {s} must be a multiple of 16"
    );
    if cfg.merger_times != 1 {
        bail!(
            "minicpm-v: merger_times={} (only the shipped single round is implemented)",
            cfg.merger_times
        );
    }
    let v = cfg.vision.clone();
    let f = DType::F32;
    let dim = 3 * v.patch_size * v.patch_size;
    let queries = match cfg.projector {
        Projector::Resampler { queries } => Some(queries),
        Projector::Merger => None,
    };
    ensure!(
        part != VisionPart::Resampler || queries.is_some(),
        "minicpm-v: a resampler-only graph needs a resampler checkpoint"
    );
    let x16 = queries.is_none() && cfg.downsample.uses_window_merger();
    let s2 = if x16 { s / 4 } else { s };
    let (out_rows, out_width) = match part {
        VisionPart::Tower => (s2, v.hidden_size),
        _ => (
            queries.unwrap_or(s2 / (cfg.merge_kernel * cfg.merge_kernel)),
            cfg.text_hidden_size,
        ),
    };

    let mut flow = ModelFlow::new("minicpm_v_vision").with_profile(CompileProfile::encoder());
    if part == VisionPart::Resampler {
        flow = flow
            .input(IN_FEATS, Shape::new(&[1, s, v.hidden_size], f))
            .input(IN_RPOS, Shape::new(&[1, s, cfg.text_hidden_size], f))
            .input(IN_RMASK, Shape::new(&[1, s], f));
        let cfg = cfg.clone();
        let n = queries.expect("checked above");
        flow = flow.plugin_named("minicpm_v.resampler", move |emit, _| {
            let x = emit.flow_input(IN_FEATS)?.hir_id();
            let out = resampler(emit, x, &cfg, n, weight_dtype)?;
            Ok(Some(emit.wrap(
                out,
                Shape::new(&[1, out_rows, cfg.text_hidden_size], f),
            )))
        });
        return flow
            .output("features")
            .build_with(&mut WeightMapSource(weights), None)
            .map_err(|e| anyhow!("minicpm-v: build resampler graph ({s} rows): {e}"));
    }
    flow = flow
        .input(IN_PATCHES, Shape::new(&[1, s, dim], f))
        .input(IN_POS, Shape::new(&[1, s], f))
        .input(IN_MASK, Shape::new(&[1, s], f));
    if x16 {
        flow = flow
            .input(IN_WIN, Shape::new(&[1, s], f))
            .input(IN_MASK2, Shape::new(&[1, s2], f));
    }
    flow = match (part, queries) {
        (VisionPart::Tower, _) => flow,
        (_, Some(_)) => flow
            .input(IN_RPOS, Shape::new(&[1, s, cfg.text_hidden_size], f))
            .input(IN_RMASK, Shape::new(&[1, s], f)),
        (_, None) => flow.input(IN_MERGE, Shape::new(&[1, s2], f)),
    };

    let cfg = cfg.clone();
    flow = flow.plugin_named("minicpm_v.vision", move |emit, _| {
        let px = emit.flow_input(IN_PATCHES)?.hir_id();
        let pos = emit.flow_input(IN_POS)?.hir_id();
        let mask = emit.flow_input(IN_MASK)?.hir_id();
        let w = emit.load_param_typed(PATCH_LINEAR, true, weight_dtype)?;
        let b = emit.load_param(&format!("{TOWER}.embeddings.patch_embedding.bias"), false)?;
        let table = emit.load_param(
            &format!("{TOWER}.embeddings.position_embedding.weight"),
            false,
        )?;
        let mut x = {
            let mut gb = HirMut::new(emit.hir());
            let e = gb.mm(px, w);
            let e = gb.add(e, b);
            let pe = gb.gather_(table, pos, 0);
            gb.add(e, pe)
        };

        let split = if x16 {
            v.insert_layer_id + 1
        } else {
            v.num_hidden_layers
        };
        for i in 0..split {
            x = encoder_layer(
                emit,
                &format!("{TOWER}.encoder.layers.{i}"),
                x,
                &cfg,
                mask,
                weight_dtype,
            )?;
        }
        if x16 {
            x = window_merger(emit, x, &cfg, s, weight_dtype)?;
            let mask2 = emit.flow_input(IN_MASK2)?.hir_id();
            for i in split..v.num_hidden_layers {
                x = encoder_layer(
                    emit,
                    &format!("{TOWER}.encoder.layers.{i}"),
                    x,
                    &cfg,
                    mask2,
                    weight_dtype,
                )?;
            }
        }
        x = layer_norm(
            emit,
            &format!("{TOWER}.post_layernorm"),
            x,
            v.layer_norm_eps,
        )?;
        let out = match (part, queries) {
            (VisionPart::Tower, _) => x,
            (_, Some(n)) => resampler(emit, x, &cfg, n, weight_dtype)?,
            (_, None) => merger(emit, x, &cfg, s2, weight_dtype)?,
        };
        Ok(Some(
            emit.wrap(out, Shape::new(&[1, out_rows, out_width], f)),
        ))
    });
    flow.output("features")
        .build_with(&mut WeightMapSource(weights), None)
        .map_err(|e| anyhow!("minicpm-v: build vision graph (bucket {s}): {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_order_groups_2x2() {
        // 4 × 4 grid → windows (0,0),(0,1),(1,0),(1,1).
        let o = window_order(4, 4, 2);
        assert_eq!(&o[..8], &[0, 1, 4, 5, 2, 3, 6, 7]);
        assert_eq!(&o[8..], &[8, 9, 12, 13, 10, 11, 14, 15]);
    }

    #[test]
    fn nearest_position_ids_match_torch_bucketize() {
        // torch: h=24 over a 70-side grid → bucket of i/24 is floor(70·i/24)
        // except where i/24 lands exactly on a boundary.
        let ids = nearest_position_ids(24, 44, 70);
        assert_eq!(ids.len(), 24 * 44);
        assert_eq!(ids[0], 0);
        // Row 1: 1/24 = 0.041666 → bucket 2 (2/70 = 0.02857 ≤ x < 3/70).
        assert_eq!(ids[44] / 70, 2);
        // Last column of row 0: 43/44 = 0.977 → 68.
        assert_eq!(ids[43], 68);
        // A 70-wide grid maps one-to-one.
        let full = nearest_position_ids(1, 70, 70);
        assert_eq!(full, (0..70).collect::<Vec<u32>>());
    }

    #[test]
    fn buckets_round_up() {
        assert_eq!(bucket_for(1), 256);
        assert_eq!(bucket_for(1024), 1024);
        assert_eq!(bucket_for(1056), 1280);
    }
}
