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

//! SAM 2 Hiera image encoder HIR builder.
//!
//! Mirrors `sam2/modeling/backbones/hieradet.py::Hiera` exactly. The
//! body operates in **BHWC** throughout (matching the reference's
//! internal tensor layout) and emits one feature map per stage —
//! `[1, h_s, w_s, dim_s]` for `s ∈ 0..4` — which the FpnNeck consumes
//! to produce stride-{4, 8, 16, 32} multi-scale features.
//!
//! ## Per-block structure ("MultiScaleBlock")
//!
//! ```text
//!   shortcut = x
//!   x = norm1(x)
//!   if stage_transition:  shortcut = q_pool(proj_to_dim_out(x))
//!   x = window_partition(x, ws_old)
//!   x = attn(x)                            # qkv → [optional pool q] → SDPA → proj
//!   x = window_unpartition(x, ws_new)
//!   x = shortcut + x
//!   x = x + mlp(norm2(x))                  # plain GELU MLP, 4× expansion
//! ```
//!
//! At a stage transition (`q_pool_block`), three things change at once:
//!   1. The Q tensor is max-pooled spatially `q_stride=2`, halving each
//!      spatial axis and quartering the sequence length.
//!   2. The channel dim doubles (`dim_in → 2·dim_in`).
//!   3. The head count doubles (so head_dim is constant — `dh = 64`
//!      for every Hiera variant after enough Q-pools).
//!
//! All Q-pool blocks list `dim != dim_out`; the shortcut runs through
//! `Linear(dim_in → dim_out)` (using the *normed* x as input, matching
//! the reference) and then through the same max-pool the Q goes
//! through, so the residual add has matching shapes.
//!
//! ## Phase 1 parity status
//!
//! Graph builder is wired but **not yet parity-tested** — candle has no
//! SAM 2 implementation to bisect against, and the official
//! `sam2_hiera_*.pt` checkpoints need a pytorch reference to compare.
//! The `tests/sam2_parity.rs` skeleton lays out the harness; turning it
//! on is a follow-up (Phase 1B). The synthetic-weights test in
//! `mod.rs` exercises the full graph build + shape inference for every
//! Hiera variant.

use super::config::{SAM2_PATCH_GRID, SAM2_Q_POOL_COUNT, SAM2_Q_STRIDE, Sam2HieraConfig};
use super::fpn_neck::{FpnNeckWeights, extract_fpn_weights};
use super::preprocess::{Sam2PreprocessWeights, extract_preprocess_weights};
use anyhow::{Result, anyhow};
use rlx_core::weight_map::WeightMap;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::op::ReduceOp;
use rlx_ir::*;
use std::collections::HashMap;

struct Sam2Builder {
    hir: HirModule,
    params: HashMap<String, Vec<f32>>,
}

impl Sam2Builder {
    fn new(name: &str) -> Self {
        Self {
            hir: HirModule::new(name),
            params: HashMap::new(),
        }
    }

    fn m(&mut self) -> HirMut<'_> {
        HirMut::new(&mut self.hir)
    }
}

#[allow(dead_code)]
fn lower_hir(hir: HirModule) -> Result<Graph> {
    Graph::from_hir(hir).map_err(|e| anyhow!("{e}"))
}

fn max_reduce(
    sb: &mut Sam2Builder,
    x: HirNodeId,
    axes: Vec<usize>,
    keep_dim: bool,
    out_shape: Shape,
) -> HirNodeId {
    sb.m().add_node(
        Op::Reduce {
            op: ReduceOp::Max,
            axes,
            keep_dim,
        },
        vec![x],
        out_shape,
    )
}

/// Build the SAM 2 Hiera image-encoder graph.
///
/// Input: `"hidden"` shape `[1, grid, grid, E0]` BHWC — produced by the
/// host-side [`crate::sam2::preprocess::assemble_patch_tokens`] which
/// runs the overlapping Conv2d patch embed + adds the interpolated
/// position embedding.
///
/// Outputs (one per stage, in finest-to-coarsest order):
///   - `[1, 256, 256, E0]`  — stage 0 (stride 4)
///   - `[1, 128, 128, E1]`  — stage 1 (stride 8)
///   - `[1,  64,  64, E2]`  — stage 2 (stride 16)
///   - `[1,  32,  32, E3]`  — stage 3 (stride 32)
///
/// where `E_s = embed_dim · 2^s`.
/// Build the SAM2 Hiera image-encoder HIR module.
#[allow(clippy::type_complexity)]
pub fn build_sam2_image_encoder_hir(
    cfg: &Sam2HieraConfig,
    weights: &mut WeightMap,
) -> Result<(
    HirModule,
    HashMap<String, Vec<f32>>,
    Sam2PreprocessWeights,
    FpnNeckWeights,
)> {
    let mut b = Sam2Builder::new("sam2_hiera_image_encoder");
    let f = DType::F32;

    // Drain host-side preprocess weights before the body so they're
    // gone when we assert the WeightMap is empty at the end.
    let preprocess = extract_preprocess_weights(weights, cfg)?;

    let grid0 = SAM2_PATCH_GRID;
    let e0 = cfg.embed_dim;
    let eps = cfg.layer_norm_eps as f32;

    // BHWC input: [1, 256, 256, E0].
    let mut x = b.m().input("hidden", Shape::new(&[1, grid0, grid0, e0], f));

    let q_pool_blocks = cfg.q_pool_block_indices();
    let mut stage = 0usize;
    let mut h_curr = grid0;
    let mut w_curr = grid0;
    let mut dim_curr = e0;
    let mut stage_outputs: Vec<HirNodeId> = Vec::with_capacity(cfg.stages.len());

    let total = cfg.total_blocks();
    for i in 0..total {
        let lp = format!("image_encoder.trunk.blocks.{i}");

        let is_q_pool = q_pool_blocks.contains(&i);
        // Stage transition happens at q_pool boundary: stage++ and
        // dims/heads double *for this block*.
        let dim_in = dim_curr;
        let stage_after = if is_q_pool { stage + 1 } else { stage };
        let dim_out = cfg.embed_dim_at_stage(stage_after);
        let num_heads = cfg.num_heads_at_stage(stage_after);
        let head_dim = dim_out / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let is_global = cfg.global_att_blocks.contains(&i);
        // Window size at this block: zero if global. Note Hiera reads
        // window size from the *original* stage (pre-transition) at a
        // Q-pool block.
        let ws_old = if is_global {
            0
        } else {
            cfg.window_size_at_stage(stage)
        };

        x = multi_scale_block(
            &mut b,
            weights,
            &lp,
            x,
            h_curr,
            w_curr,
            dim_in,
            dim_out,
            num_heads,
            head_dim,
            scale,
            ws_old,
            is_q_pool,
            eps,
            cfg.mlp_ratio,
            cfg.qkv_bias,
        )?;

        // Update running spatial/channel state if this block Q-pooled.
        if is_q_pool {
            stage += 1;
            h_curr /= SAM2_Q_STRIDE;
            w_curr /= SAM2_Q_STRIDE;
            dim_curr = dim_out;
        }

        // Emit a feature map at the *last* block of every stage.
        let stage_end = (i + 1 == total) || q_pool_blocks.contains(&(i + 1));
        if stage_end {
            stage_outputs.push(x);
        }
    }

    debug_assert_eq!(stage_outputs.len(), cfg.stages.len());
    debug_assert_eq!(stage_outputs.len(), SAM2_Q_POOL_COUNT + 1);

    b.hir.set_outputs(stage_outputs);

    // Drain FPN weights before returning so the caller can assert the
    // WeightMap is empty.
    let fpn = extract_fpn_weights(weights, cfg)?;

    Ok((b.hir, b.params, preprocess, fpn))
}

/// Lowered graph wrapper for legacy callers (via [`super::flow::Sam2ImageEncoderFlow`]).
#[allow(clippy::type_complexity)]
pub fn build_sam2_image_encoder_graph(
    cfg: &Sam2HieraConfig,
    weights: &mut WeightMap,
) -> Result<(
    Graph,
    HashMap<String, Vec<f32>>,
    Sam2PreprocessWeights,
    FpnNeckWeights,
)> {
    let built = super::flow::build_sam2_image_encoder_built(cfg, weights)?;
    let (graph, params) = rlx_core::flow_util::graph_from_built(built.model)?;
    Ok((graph, params, built.preprocess, built.neck))
}

/// One Hiera `MultiScaleBlock` — see file-level docs for the structure.
#[allow(clippy::too_many_arguments)]
fn multi_scale_block(
    sb: &mut Sam2Builder,
    w: &mut WeightMap,
    lp: &str,
    x: HirNodeId, // [1, H, W, dim_in] BHWC
    h: usize,
    wd: usize,
    dim_in: usize,
    dim_out: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    ws_old: usize, // 0 = global
    is_q_pool: bool,
    eps: f32,
    mlp_ratio: f64,
    qkv_bias: bool,
) -> Result<HirNodeId> {
    // ── Pre-LN1 over channel dim ──
    let n1_g = load_p(sb, w, &format!("{lp}.norm1.weight"), false)?;
    let n1_b = load_p(sb, w, &format!("{lp}.norm1.bias"), false)?;
    let normed = sb.m().ln(x, n1_g, n1_b, eps);

    // ── Shortcut path ──
    // Hiera only projects the residual when dim_in != dim_out (which
    // happens at exactly the q_pool blocks). Both the projection and
    // the q_pool then apply to the projected, normed input.
    let shortcut = if dim_in != dim_out {
        let proj_w = load_p(sb, w, &format!("{lp}.proj.weight"), true)?;
        let proj_b = load_p(sb, w, &format!("{lp}.proj.bias"), false)?;
        let proj_mm = sb.m().mm(normed, proj_w);
        let projected = sb.m().add(proj_mm, proj_b);
        if is_q_pool {
            // [1, H, W, dim_out] → pool 2x2 → [1, H/2, W/2, dim_out]
            qpool_2x2(sb, projected, 1, h, wd, dim_out)
        } else {
            projected
        }
    } else {
        x
    };

    // ── Attention path ──
    let (attn_out, h_new, w_new) = if ws_old == 0 {
        // Global attention: no windowing, optional Q-pool.
        let out = multi_scale_attention_global(
            sb, w, lp, normed, h, wd, dim_in, dim_out, num_heads, head_dim, scale, is_q_pool,
            qkv_bias,
        )?;
        let (hh, ww) = if is_q_pool {
            (h / SAM2_Q_STRIDE, wd / SAM2_Q_STRIDE)
        } else {
            (h, wd)
        };
        (out, hh, ww)
    } else {
        // Windowed attention.
        let out = multi_scale_attention_windowed(
            sb, w, lp, normed, h, wd, dim_in, dim_out, num_heads, head_dim, scale, ws_old,
            is_q_pool, qkv_bias,
        )?;
        let (hh, ww) = if is_q_pool {
            (h / SAM2_Q_STRIDE, wd / SAM2_Q_STRIDE)
        } else {
            (h, wd)
        };
        (out, hh, ww)
    };
    let _ = (h_new, w_new); // future-proof for assertions

    // ── Residual (drop_path is identity at inference) ──
    let x = sb.m().add(shortcut, attn_out);

    // ── Pre-LN2 + MLP (4× expansion, plain GELU) ──
    let n2_g = load_p(sb, w, &format!("{lp}.norm2.weight"), false)?;
    let n2_b = load_p(sb, w, &format!("{lp}.norm2.bias"), false)?;
    let normed2 = sb.m().ln(x, n2_g, n2_b, eps);

    let hidden = (dim_out as f64 * mlp_ratio) as usize;
    // Reference's `MLP` uses `nn.ModuleList` indexed by `layers.{i}`.
    let fc1_w = load_p(sb, w, &format!("{lp}.mlp.layers.0.weight"), true)?;
    let fc1_b = load_p(sb, w, &format!("{lp}.mlp.layers.0.bias"), false)?;
    let fc2_w = load_p(sb, w, &format!("{lp}.mlp.layers.1.weight"), true)?;
    let fc2_b = load_p(sb, w, &format!("{lp}.mlp.layers.1.bias"), false)?;
    let _ = hidden; // dims live in the weight shapes; this asserts cfg preflight at load
    let up_mm = sb.m().mm(normed2, fc1_w);
    let up = sb.m().add(up_mm, fc1_b);
    let act = sb.m().gelu(up); // GELU(erf) — matches `nn.GELU()` reference default
    let down_mm = sb.m().mm(act, fc2_w);
    let down = sb.m().add(down_mm, fc2_b);

    Ok(sb.m().add(x, down))
}

/// Max-pool 2×2 stride 2 over the BHWC spatial dims via two
/// single-axis reduces (not a single reduce with axes=[2,4] — the
/// CPU executor's reduce op silently NOPs on non-contiguous axes,
/// which used to silently corrupt every Q-pool block in the encoder).
/// `b` is the leading batch dim — must be threaded explicitly because
/// the shortcut path uses `b=1` (full image) while the Q-attention
/// path uses `b=n_win` (per-window pooling).
fn qpool_2x2(
    sb: &mut Sam2Builder,
    x: HirNodeId,
    batch: usize,
    h: usize,
    w: usize,
    c: usize,
) -> HirNodeId {
    debug_assert!(
        h.is_multiple_of(2) && w.is_multiple_of(2),
        "Q-pool needs even spatial dims"
    );
    let f = DType::F32;
    // ── Pool over H ──
    // [B, H, W, C] → [B, H/2, 2, W, C] → reduce(Max, axis=2) → [B, H/2, W, C]
    let rs_h = sb
        .m()
        .reshape_(x, vec![batch as i64, (h / 2) as i64, 2, w as i64, c as i64]);
    let pool_h = max_reduce(
        sb,
        rs_h,
        vec![2],
        false,
        Shape::new(&[batch, h / 2, w, c], f),
    );
    // ── Pool over W ──
    // [B, H/2, W, C] → [B, H/2, W/2, 2, C] → reduce(Max, axis=3) → [B, H/2, W/2, C]
    let rs_w = sb.m().reshape_(
        pool_h,
        vec![batch as i64, (h / 2) as i64, (w / 2) as i64, 2, c as i64],
    );
    max_reduce(
        sb,
        rs_w,
        vec![3],
        false,
        Shape::new(&[batch, h / 2, w / 2, c], f),
    )
}

/// Mask-unit (windowed) multi-scale attention. Caller supplies the
/// *unpartitioned* `[1, H, W, dim_in]` BHWC tensor; this fn handles
/// window partition, qkv, optional q-pool, SDPA, proj, and unpartition.
#[allow(clippy::too_many_arguments)]
fn multi_scale_attention_windowed(
    sb: &mut Sam2Builder,
    w: &mut WeightMap,
    lp: &str,
    x: HirNodeId, // [1, H, W, dim_in] BHWC
    h: usize,
    wd: usize,
    dim_in: usize,
    dim_out: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    ws: usize,
    q_pool: bool,
    qkv_bias: bool,
) -> Result<HirNodeId> {
    // 1) Pad H, W up to a multiple of ws (concat-zero padding).
    let pad_h = (ws - h % ws) % ws;
    let pad_w = (ws - wd % ws) % ws;
    let hp = h + pad_h;
    let wp = wd + pad_w;
    let x_pad = if pad_h > 0 {
        let z = zero_param(sb, &format!("{lp}.attn._pad_h"), &[1, pad_h, wd, dim_in]);
        sb.m().concat_(vec![x, z], 1)
    } else {
        x
    };
    let x_pad = if pad_w > 0 {
        let z = zero_param(sb, &format!("{lp}.attn._pad_w"), &[1, hp, pad_w, dim_in]);
        sb.m().concat_(vec![x_pad, z], 2)
    } else {
        x_pad
    };

    // 2) Partition into windows: [1, Hp, Wp, C] → [1, nh, ws, nw, ws, C]
    //    → permute(0,1,3,2,4,5) → [1, nh, nw, ws, ws, C] → [nh·nw, ws, ws, C].
    let nh_w = hp / ws;
    let nw_w = wp / ws;
    let n_win = nh_w * nw_w;
    let rs = sb.m().reshape_(
        x_pad,
        vec![
            1,
            nh_w as i64,
            ws as i64,
            nw_w as i64,
            ws as i64,
            dim_in as i64,
        ],
    );
    let perm = sb.m().transpose_(rs, vec![0, 1, 3, 2, 4, 5]);
    let windowed = sb.m().reshape_(
        perm,
        vec![n_win as i64, ws as i64, ws as i64, dim_in as i64],
    );

    // 3) Run windowed attention.
    let attn_out = mask_unit_attention(
        sb, w, lp, windowed, n_win, ws, ws, dim_in, dim_out, num_heads, head_dim, scale, q_pool,
        qkv_bias,
    )?;
    // attn_out: [n_win, ws_new, ws_new, dim_out]
    let ws_new = if q_pool { ws / SAM2_Q_STRIDE } else { ws };

    // 4) Unpartition: [n_win, ws_new, ws_new, dim_out]
    //    → [1, nh, nw, ws_new, ws_new, dim_out]
    //    → permute(0,1,3,2,4,5) → [1, nh, ws_new, nw, ws_new, dim_out]
    //    → [1, Hp_new, Wp_new, dim_out]
    let hp_new = nh_w * ws_new;
    let wp_new = nw_w * ws_new;
    let r = sb.m().reshape_(
        attn_out,
        vec![
            1,
            nh_w as i64,
            nw_w as i64,
            ws_new as i64,
            ws_new as i64,
            dim_out as i64,
        ],
    );
    let p = sb.m().transpose_(r, vec![0, 1, 3, 2, 4, 5]);
    let unp = sb
        .m()
        .reshape_(p, vec![1, hp_new as i64, wp_new as i64, dim_out as i64]);

    // 5) Crop the padding back off if needed.
    let h_new = if q_pool { h / SAM2_Q_STRIDE } else { h };
    let w_new = if q_pool { wd / SAM2_Q_STRIDE } else { wd };
    let out = if hp_new != h_new {
        sb.m().narrow_(unp, 1, 0, h_new)
    } else {
        unp
    };
    let out = if wp_new != w_new {
        sb.m().narrow_(out, 2, 0, w_new)
    } else {
        out
    };

    Ok(out)
}

/// Global multi-scale attention: same logic as windowed but with the
/// whole `[1, H, W, dim_in]` treated as one window.
#[allow(clippy::too_many_arguments)]
fn multi_scale_attention_global(
    sb: &mut Sam2Builder,
    w: &mut WeightMap,
    lp: &str,
    x: HirNodeId,
    h: usize,
    wd: usize,
    dim_in: usize,
    dim_out: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    q_pool: bool,
    qkv_bias: bool,
) -> Result<HirNodeId> {
    mask_unit_attention(
        sb, w, lp, x, 1, h, wd, dim_in, dim_out, num_heads, head_dim, scale, q_pool, qkv_bias,
    )
}

/// Core mask-unit attention. Input `[B, H, W, dim_in]` where `B` is the
/// number of windows (or 1 for global). Q-pool, when active, max-pools
/// `q` over each (2×2) spatial group before SDPA — `k` and `v` keep
/// their full sequence length. Output `[B, H_out, W_out, dim_out]`.
#[allow(clippy::too_many_arguments)]
fn mask_unit_attention(
    sb: &mut Sam2Builder,
    w: &mut WeightMap,
    lp: &str,
    x: HirNodeId, // [B, H, W, dim_in]
    batch: usize,
    h: usize,
    wd: usize,
    _dim_in: usize,
    dim_out: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    q_pool: bool,
    qkv_bias: bool,
) -> Result<HirNodeId> {
    let s_kv = h * wd;

    // 1) qkv linear: [B, H, W, dim_in] → [B, H, W, 3·dim_out]
    let qkv_w = load_p(sb, w, &format!("{lp}.attn.qkv.weight"), true)?;
    let qkv_b = if qkv_bias {
        Some(load_p(sb, w, &format!("{lp}.attn.qkv.bias"), false)?)
    } else {
        None
    };
    let qkv = sb.m().mm(x, qkv_w);
    let qkv = if let Some(bnode) = qkv_b {
        sb.m().add(qkv, bnode)
    } else {
        qkv
    };

    // 2) Reshape to [B, S_kv, 3, nh, dh] and split into q, k, v of shape
    //    [B, S_kv, nh, dh] each.
    let qkv5 = sb.m().reshape_(
        qkv,
        vec![
            batch as i64,
            s_kv as i64,
            3,
            num_heads as i64,
            head_dim as i64,
        ],
    );
    // Permute to [3, B, S_kv, nh, dh] for clean split via narrow.
    let qkv_perm = sb.m().transpose_(qkv5, vec![2, 0, 1, 3, 4]);
    let q_full = {
        let s = sb.m().narrow_(qkv_perm, 0, 0, 1);
        sb.m().reshape_(
            s,
            vec![batch as i64, s_kv as i64, num_heads as i64, head_dim as i64],
        )
    };
    let k = {
        let s = sb.m().narrow_(qkv_perm, 0, 1, 1);
        sb.m().reshape_(
            s,
            vec![batch as i64, s_kv as i64, num_heads as i64, head_dim as i64],
        )
    };
    let v = {
        let s = sb.m().narrow_(qkv_perm, 0, 2, 1);
        sb.m().reshape_(
            s,
            vec![batch as i64, s_kv as i64, num_heads as i64, head_dim as i64],
        )
    };

    // 3) Optionally pool Q. The reference reshapes q back to [B, H, W,
    //    dim_out] before pooling, then back to [B, S_q, nh, dh] after.
    let (q, h_out, w_out, s_q) = if q_pool {
        let qspat = sb.m().reshape_(
            q_full,
            vec![batch as i64, h as i64, wd as i64, dim_out as i64],
        );
        // Pool with leading batch = n_win (per-window q pooling); not 1.
        let qpooled = qpool_2x2(sb, qspat, batch, h, wd, dim_out);
        let h2 = h / SAM2_Q_STRIDE;
        let w2 = wd / SAM2_Q_STRIDE;
        let s2 = h2 * w2;
        let qflat = sb.m().reshape_(
            qpooled,
            vec![batch as i64, s2 as i64, num_heads as i64, head_dim as i64],
        );
        (qflat, h2, w2, s2)
    } else {
        (q_full, h, wd, s_kv)
    };

    // 4) SDPA. Lower to explicit matmul-softmax-matmul (matching SAM
    //    v1's choice) for parity-debuggability and backend portability.
    //    Transpose [B, S, nh, dh] → [B, nh, S, dh], then merge heads
    //    into batch: [B·nh, S, dh].
    let q_t = sb.m().transpose_(q, vec![0, 2, 1, 3]);
    let k_t = sb.m().transpose_(k, vec![0, 2, 1, 3]);
    let v_t = sb.m().transpose_(v, vec![0, 2, 1, 3]);
    let q_flat = sb.m().reshape_(
        q_t,
        vec![(batch * num_heads) as i64, s_q as i64, head_dim as i64],
    );
    let k_flat = sb.m().reshape_(
        k_t,
        vec![(batch * num_heads) as i64, s_kv as i64, head_dim as i64],
    );
    let v_flat = sb.m().reshape_(
        v_t,
        vec![(batch * num_heads) as i64, s_kv as i64, head_dim as i64],
    );

    let scale_node = scalar_param(sb, &format!("{lp}.attn._scale"), scale);
    let q_scaled = sb.m().mul(q_flat, scale_node);
    let k_for_mm = sb.m().transpose_(k_flat, vec![0, 2, 1]); // [B·nh, dh, S_kv]
    let scores = sb.m().mm(q_scaled, k_for_mm); // [B·nh, S_q, S_kv]
    let attn_w = sb.m().sm(scores, -1);
    let attn_v = sb.m().mm(attn_w, v_flat); // [B·nh, S_q, dh]

    // 5) Merge heads back: [B·nh, S_q, dh] → [B, nh, S_q, dh]
    //    → [B, S_q, nh, dh] → [B, H_out, W_out, dim_out]
    let r = sb.m().reshape_(
        attn_v,
        vec![batch as i64, num_heads as i64, s_q as i64, head_dim as i64],
    );
    let r = sb.m().transpose_(r, vec![0, 2, 1, 3]);
    let merged = sb.m().reshape_(
        r,
        vec![batch as i64, h_out as i64, w_out as i64, dim_out as i64],
    );

    // 6) Output projection.
    let proj_w = load_p(sb, w, &format!("{lp}.attn.proj.weight"), true)?;
    let proj_b = load_p(sb, w, &format!("{lp}.attn.proj.bias"), false)?;
    let proj_mm = sb.m().mm(merged, proj_w);
    Ok(sb.m().add(proj_mm, proj_b))
}

// ─── Builder helpers ────────────────────────────────────────────────

fn load_p(
    sb: &mut Sam2Builder,
    weights: &mut WeightMap,
    key: &str,
    transpose: bool,
) -> Result<HirNodeId> {
    let (data, shape) = if transpose {
        weights
            .take_transposed(key)
            .map_err(|e| anyhow!("transpose-load `{key}`: {e}"))?
    } else {
        weights
            .take(key)
            .map_err(|e| anyhow!("load `{key}`: {e}"))?
    };
    let id = sb.m().param(key, Shape::new(&shape, DType::F32));
    sb.params.insert(key.to_string(), data);
    Ok(id)
}

fn scalar_param(sb: &mut Sam2Builder, name: &str, value: f32) -> HirNodeId {
    let id = sb.m().param(name, Shape::new(&[1], DType::F32));
    sb.params.insert(name.to_string(), vec![value]);
    id
}

fn zero_param(sb: &mut Sam2Builder, name: &str, shape: &[usize]) -> HirNodeId {
    let n: usize = shape.iter().product();
    let id = sb.m().param(name, Shape::new(shape, DType::F32));
    sb.params.insert(name.to_string(), vec![0f32; n]);
    id
}
