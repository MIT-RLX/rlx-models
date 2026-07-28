// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Pixtral vision HIR — ViT (RMSNorm + 2D RoPE + SwiGLU) + patch merger + MLP.
//!
//! Input: `hidden` `[1, n_patches, hidden]` (host patch-embed + pre-RMS).
//!        `vision_rope_cos` / `vision_rope_sin` `[n_patches, head_dim/2]` (GPT-J).
//! Output: `lm_embeds` `[1, n_merged, projector_output_dim]` (before img_break).

use crate::config::PixtralVisionConfig;
use crate::encoder::{PixtralLayerWeights, PixtralWeights};
use anyhow::{Result, ensure};
use rlx_core::flow_util::built_from_hir_with_profile;
use rlx_flow::{BuiltModel, CompileProfile};
use rlx_ir::hir::{FusionPolicy, HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::op::{MaskKind, RopeStyle};
use rlx_ir::{DType, Shape};
use std::collections::HashMap;

type NodeId = HirNodeId;

pub struct PixtralVisionBuilt {
    pub model: BuiltModel,
    pub grid_x: usize,
    pub grid_y: usize,
    pub n_merged: usize,
}

pub fn build_pixtral_vision(
    cfg: &PixtralVisionConfig,
    weights: &PixtralWeights,
    grid_x: usize,
    grid_y: usize,
) -> Result<PixtralVisionBuilt> {
    let (hir, params, n_merged) = build_pixtral_vision_hir(cfg, weights, grid_x, grid_y)?;
    let model = built_from_hir_with_profile(hir, params, CompileProfile::encoder())?;
    Ok(PixtralVisionBuilt {
        model,
        grid_x,
        grid_y,
        n_merged,
    })
}

fn build_pixtral_vision_hir(
    cfg: &PixtralVisionConfig,
    weights: &PixtralWeights,
    grid_x: usize,
    grid_y: usize,
) -> Result<(HirModule, HashMap<String, Vec<f32>>, usize)> {
    let n_merge = cfg.spatial_merge_size.max(1);
    ensure!(grid_x.is_multiple_of(n_merge) && grid_y.is_multiple_of(n_merge));
    let n_pos = grid_x * grid_y;
    let p_x = grid_x / n_merge;
    let p_y = grid_y / n_merge;
    let n_merged = p_x * p_y;
    let merge_sq = n_merge * n_merge;

    let batch = 1usize;
    let h = cfg.hidden_size;
    let ff = cfg.intermediate_size;
    let nh = cfg.num_attention_heads;
    let dh = cfg.head_dim();
    let half = dh / 2;
    let proj = cfg.projector_output_dim;
    let eps = cfg.layer_norm_eps;
    let f = DType::F32;

    let mut hir = HirModule::new("pixtral_vision").with_fusion_policy(FusionPolicy::Direct);
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();

    let hidden = hir.input("hidden", Shape::new(&[batch, n_pos, h], f));
    let rope_cos = hir.input("vision_rope_cos", Shape::new(&[n_pos, half], f));
    let rope_sin = hir.input("vision_rope_sin", Shape::new(&[n_pos, half], f));

    let mut g = HirMut::new(&mut hir);
    let mut h_id = hidden;

    for (il, blk) in weights.layers.iter().enumerate() {
        h_id = build_vit_block(
            &mut g,
            &mut params,
            il,
            blk,
            h_id,
            batch,
            n_pos,
            h,
            ff,
            nh,
            dh,
            eps,
            cfg.use_silu,
            rope_cos,
            rope_sin,
        )?;
    }

    if let (Some(norm_w), Some(merger_w)) = (&weights.mm_input_norm, &weights.mm_patch_merger) {
        let zero = param_zeros(&mut g, &mut params, "mm_input_norm.zero_beta", h);
        let gamma = param_vec(&mut g, &mut params, "mm.input_norm.weight", norm_w, h);
        h_id = g.rms_norm(h_id, gamma, zero, eps);

        let gather_idx = spatial_merge_gather_idx(grid_y, grid_x, n_merge);
        let idx = param_vec(&mut g, &mut params, "spatial_merge_idx", &gather_idx, n_pos);
        let idx_2d = g.reshape_(idx, vec![batch as i64, n_pos as i64]);
        h_id = g.gather_(h_id, idx_2d, 1);
        let merged = g.reshape_(h_id, vec![(batch * n_merged) as i64, (h * merge_sq) as i64]);
        // GGUF merger: ne0=merge_dim, ne1=hidden → mm wants [merge_dim, hidden].
        let mw = param_mat(
            &mut g,
            &mut params,
            "mm.patch_merger.weight",
            &transpose_2d(merger_w, h, h * merge_sq),
            h * merge_sq,
            h,
        )?;
        let mm = g.mm(merged, mw);
        h_id = g.reshape_(mm, vec![batch as i64, n_merged as i64, h as i64]);
    } else {
        ensure!(n_merge == 1, "spatial merge requires mm.patch_merger");
    }

    let rows = batch * n_merged;
    let x2d = g.reshape_(h_id, vec![rows as i64, h as i64]);
    let w1 = param_mat(
        &mut g,
        &mut params,
        "mm.1.weight",
        &transpose_2d(&weights.mm1_w, proj, h),
        h,
        proj,
    )?;
    let mut y = g.mm(x2d, w1);
    if let Some(b) = &weights.mm1_b {
        let b1 = param_vec(&mut g, &mut params, "mm.1.bias", b, proj);
        y = g.add(y, b1);
    }
    y = g.gelu(y);
    let w2 = param_mat(
        &mut g,
        &mut params,
        "mm.2.weight",
        &transpose_2d(&weights.mm2_w, proj, proj),
        proj,
        proj,
    )?;
    let mut out = g.mm(y, w2);
    if let Some(b) = &weights.mm2_b {
        let b2 = param_vec(&mut g, &mut params, "mm.2.bias", b, proj);
        out = g.add(out, b2);
    }
    out = g.reshape_(out, vec![batch as i64, n_merged as i64, proj as i64]);
    g.set_outputs(vec![out]);

    Ok((hir, params, n_merged))
}

fn build_vit_block(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    il: usize,
    blk: &PixtralLayerWeights,
    h_in: NodeId,
    batch: usize,
    seq: usize,
    n: usize,
    n_ff: usize,
    nh: usize,
    dh: usize,
    eps: f32,
    use_silu: bool,
    rope_cos: NodeId,
    rope_sin: NodeId,
) -> Result<NodeId> {
    let p = format!("v.blk.{il}");
    let zero = param_zeros(g, params, &format!("{p}.zero_beta"), n);

    let ln1 = param_vec(g, params, &format!("{p}.ln1.weight"), &blk.ln1, n);
    let x = g.rms_norm(h_in, ln1, zero, eps);

    let x2d = g.reshape_(x, vec![(batch * seq) as i64, n as i64]);
    let q = linear_sq(g, params, &format!("{p}.attn_q"), &blk.q, x2d, n)?;
    let k = linear_sq(g, params, &format!("{p}.attn_k"), &blk.k, x2d, n)?;
    let v = linear_sq(g, params, &format!("{p}.attn_v"), &blk.v, x2d, n)?;
    let q = g.reshape_(q, vec![batch as i64, seq as i64, n as i64]);
    let k = g.reshape_(k, vec![batch as i64, seq as i64, n as i64]);
    let v = g.reshape_(v, vec![batch as i64, seq as i64, n as i64]);

    // Pixtral / llama.cpp `build_rope_2d` uses GPT-J (interleaved) pairs with
    // height on the first half of the head and width on the second.
    let q = g.rope_styled(q, rope_cos, rope_sin, dh, RopeStyle::GptJ);
    let k = g.rope_styled(k, rope_cos, rope_sin, dh, RopeStyle::GptJ);

    let attn = g.attention_kind(
        q,
        k,
        v,
        nh,
        dh,
        MaskKind::None,
        Shape::new(&[batch, seq, n], DType::F32),
    );
    let attn2d = g.reshape_(attn, vec![(batch * seq) as i64, n as i64]);
    let attn_out = linear_sq(g, params, &format!("{p}.attn_out"), &blk.o, attn2d, n)?;
    let attn_bsn = g.reshape_(attn_out, vec![batch as i64, seq as i64, n as i64]);
    let h = g.add(h_in, attn_bsn);

    let ln2 = param_vec(g, params, &format!("{p}.ln2.weight"), &blk.ln2, n);
    let y = g.rms_norm(h, ln2, zero, eps);
    let y2d = g.reshape_(y, vec![(batch * seq) as i64, n as i64]);

    let gate = linear_rect(g, params, &format!("{p}.ffn_gate"), &blk.gate, y2d, n, n_ff)?;
    let up = linear_rect(g, params, &format!("{p}.ffn_up"), &blk.up, y2d, n, n_ff)?;
    let gate_act = if use_silu { g.silu(gate) } else { g.gelu(gate) };
    let ff = g.mul(gate_act, up);
    let down = linear_rect(g, params, &format!("{p}.ffn_down"), &blk.down, ff, n_ff, n)?;
    let down_bsn = g.reshape_(down, vec![batch as i64, seq as i64, n as i64]);
    Ok(g.add(h, down_bsn))
}

fn linear_sq(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    key: &str,
    w_gguf: &[f32],
    x: NodeId,
    n: usize,
) -> Result<NodeId> {
    let w = param_mat(
        g,
        params,
        &format!("{key}.weight"),
        &transpose_2d(w_gguf, n, n),
        n,
        n,
    )?;
    Ok(g.mm(x, w))
}

fn linear_rect(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    key: &str,
    w_gguf: &[f32],
    x: NodeId,
    in_dim: usize,
    out_dim: usize,
) -> Result<NodeId> {
    let w = param_mat(
        g,
        params,
        &format!("{key}.weight"),
        &transpose_2d(w_gguf, out_dim, in_dim),
        in_dim,
        out_dim,
    )?;
    Ok(g.mm(x, w))
}

fn spatial_merge_gather_idx(grid_y: usize, grid_x: usize, n_merge: usize) -> Vec<f32> {
    let p_y = grid_y / n_merge;
    let p_x = grid_x / n_merge;
    let mut idx = Vec::with_capacity(grid_y * grid_x);
    for by in 0..p_y {
        for bx in 0..p_x {
            for my in 0..n_merge {
                for mx in 0..n_merge {
                    let gy = by * n_merge + my;
                    let gx = bx * n_merge + mx;
                    idx.push((gy * grid_x + gx) as f32);
                }
            }
        }
    }
    idx
}

fn param_vec(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    key: &str,
    data: &[f32],
    len: usize,
) -> NodeId {
    debug_assert_eq!(data.len(), len);
    params.insert(key.to_string(), data.to_vec());
    g.param(key, Shape::new(&[len], DType::F32))
}

fn param_zeros(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    key: &str,
    len: usize,
) -> NodeId {
    params.insert(key.to_string(), vec![0f32; len]);
    g.param(key, Shape::new(&[len], DType::F32))
}

fn param_mat(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    key: &str,
    data: &[f32],
    rows: usize,
    cols: usize,
) -> Result<NodeId> {
    ensure!(
        data.len() == rows * cols,
        "{key} len {} != {rows}×{cols}",
        data.len()
    );
    params.insert(key.to_string(), data.to_vec());
    Ok(g.param(key, Shape::new(&[rows, cols], DType::F32)))
}

/// Row-major `[rows, cols]` → `[cols, rows]`.
fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{spatial_merge_gather_idx, transpose_2d};

    #[test]
    fn merge_gather_is_block_contiguous_permutation() {
        // 4×4 grid, 2×2 merge → each 2×2 spatial block becomes contiguous in the
        // output order the patch-merger weight expects (row-major within a block).
        let idx: Vec<usize> = spatial_merge_gather_idx(4, 4, 2)
            .iter()
            .map(|&f| f as usize)
            .collect();
        assert_eq!(
            idx,
            vec![0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
        );
        // Must be a true permutation — every source patch referenced exactly once.
        let mut seen = idx.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..16).collect::<Vec<_>>());
    }

    #[test]
    fn transpose_2d_swaps_layout() {
        // [[1,2,3],[4,5,6]] (2×3) → [[1,4],[2,5],[3,6]] (3×2).
        let t = transpose_2d(&[1., 2., 3., 4., 5., 6.], 2, 3);
        assert_eq!(t, vec![1., 4., 2., 5., 3., 6.]);
    }
}
