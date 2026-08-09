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

//! SAM v1 ViT image encoder HIR builder.
//!
//! Mirrors `candle-transformers/src/models/segment_anything/image_encoder.rs`.
//! Decomposes attention into primitives (rlx-ir's `attention_` op is a
//! black box and can't host the inline rel-pos add SAM uses).
//!
//! Two attention modes:
//!   - **Global** (window_size == 0): full S = hw·hw attention. Used by
//!     blocks listed in `global_attn_indexes`.
//!   - **Windowed** (window_size > 0): pad spatial dims to a multiple
//!     of `window_size` via concat-with-zeros, reshape into
//!     `[B·nW, ws, ws, C]`, attention within each window, reverse the
//!     reshape, narrow off the padding.
//!
//! The neck (Conv2d 1×1 + LN2d + Conv2d 3×3 + LN2d → `[B, 256, hw, hw]`)
//! is appended to the encoder HIR via [`rlx_core::vision_ops_ir`].

use super::config::{SAM_EMBED_HW, SamEncoderConfig};
use super::preprocess::{SamPreprocessWeights, extract_preprocess_weights};
use anyhow::{Result, anyhow, ensure};
use rlx_core::vision_ops_ir::{bhwc_to_nchw, conv2d_no_bias, layer_norm2d_nchw};
use rlx_core::weight_map::WeightMap;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::*;
use std::collections::HashMap;

struct SamBuilder {
    hir: HirModule,
    params: HashMap<String, Vec<f32>>,
}

impl SamBuilder {
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

/// Build the SAM ViT image-encoder HIR (body + neck).
///
/// Input: `"hidden"` shape `[1, hw·hw, embed_dim]` — patch tokens from
/// `crate::sam::preprocess::assemble_patch_tokens`.
///
/// Output: `[1, out_chans, hw, hw]` NCHW image embeddings.
pub fn build_sam_encoder_hir(
    cfg: &SamEncoderConfig,
    weights: &mut WeightMap,
) -> Result<(HirModule, HashMap<String, Vec<f32>>, SamPreprocessWeights)> {
    let mut b = SamBuilder::new("sam_image_encoder");
    let f = DType::F32;

    // Host-side preprocess weights (patch projection + abs pos embed).
    // Drain these *before* iterating blocks so the keys are gone when
    // we later assert the WeightMap is empty.
    let preprocess = extract_preprocess_weights(weights, cfg)?;

    let e = cfg.embed_dim;
    let nh = cfg.num_heads;
    let dh = cfg.head_dim();
    let scale = 1.0 / (dh as f32).sqrt();
    let eps = cfg.layer_norm_eps as f32;
    let hw = SAM_EMBED_HW;
    let s = hw * hw; // 64·64 = 4096

    // Input: pre-assembled patch tokens [1, 4096, E].
    let hidden_input = b.m().input("hidden", Shape::new(&[1, s, e], f));

    let mut x = hidden_input;
    for layer_idx in 0..cfg.depth {
        let lp = format!("image_encoder.blocks.{layer_idx}");
        let is_global = cfg.global_attn_indexes.contains(&layer_idx);
        let ws = if is_global { 0 } else { cfg.window_size };

        // ── Pre-LN1 ──
        let n1_g = load_p(&mut b, weights, &format!("{lp}.norm1.weight"), false)?;
        let n1_b = load_p(&mut b, weights, &format!("{lp}.norm1.bias"), false)?;
        let normed = b.m().ln(x, n1_g, n1_b, eps);

        // ── Attention (windowed or global) ──
        let attn_out = if ws == 0 {
            attention_global(
                &mut b,
                weights,
                &lp,
                normed,
                e,
                nh,
                dh,
                scale,
                hw,
                cfg.use_rel_pos,
                cfg.qkv_bias,
            )?
        } else {
            attention_windowed(
                &mut b,
                weights,
                &lp,
                normed,
                e,
                nh,
                dh,
                scale,
                hw,
                ws,
                cfg.use_rel_pos,
                cfg.qkv_bias,
            )?
        };

        // Residual
        x = b.m().add(x, attn_out);

        // ── Pre-LN2 + MLP (4× expansion, plain GELU) ──
        let n2_g = load_p(&mut b, weights, &format!("{lp}.norm2.weight"), false)?;
        let n2_b = load_p(&mut b, weights, &format!("{lp}.norm2.bias"), false)?;
        let normed2 = b.m().ln(x, n2_g, n2_b, eps);

        let fc1_w = load_p(&mut b, weights, &format!("{lp}.mlp.lin1.weight"), true)?;
        let fc1_b = load_p(&mut b, weights, &format!("{lp}.mlp.lin1.bias"), false)?;
        let fc2_w = load_p(&mut b, weights, &format!("{lp}.mlp.lin2.weight"), true)?;
        let fc2_b = load_p(&mut b, weights, &format!("{lp}.mlp.lin2.bias"), false)?;

        let up_mm = b.m().mm(normed2, fc1_w);
        let up = b.m().add(up_mm, fc1_b);
        // candle's `Activation::Gelu` dispatches to `Tensor::gelu_erf()`
        // — the exact erf form — for SAM's MlpBlock. Use the matching
        // erf kernel here.
        let act = b.m().gelu(up);
        let down_mm = b.m().mm(act, fc2_w);
        let ffn = b.m().add(down_mm, fc2_b);

        x = b.m().add(x, ffn);
    }

    // ── Neck: BHWC → NCHW, 1×1 conv, LN2d, 3×3 conv, LN2d ──
    // Meta's `segment_anything/modeling/image_encoder.py` uses
    // `bias=False` on both neck Conv2ds, so the official safetensors
    // (e.g. `sam_vit_b_01ec64`) only contain `neck.{0,2}.weight` — no
    // biases. We mirror that here with `conv2d_no_bias`.
    let oc = cfg.out_chans;
    let nchw = bhwc_to_nchw(&mut b.m(), x, 1, hw, hw, e);
    let c1_w = load_p(&mut b, weights, "image_encoder.neck.0.weight", false)?;
    let feat = conv2d_no_bias(&mut b.m(), nchw, c1_w, 1, oc, 1, 1, [1, 1], [0, 0], hw, hw);
    let ln1_g = load_p(&mut b, weights, "image_encoder.neck.1.weight", false)?;
    let ln1_b = load_p(&mut b, weights, "image_encoder.neck.1.bias", false)?;
    let feat = layer_norm2d_nchw(&mut b.m(), feat, ln1_g, ln1_b, eps);
    let c2_w = load_p(&mut b, weights, "image_encoder.neck.2.weight", false)?;
    let feat = conv2d_no_bias(&mut b.m(), feat, c2_w, 1, oc, 3, 3, [1, 1], [1, 1], hw, hw);
    let ln2_g = load_p(&mut b, weights, "image_encoder.neck.3.weight", false)?;
    let ln2_b = load_p(&mut b, weights, "image_encoder.neck.3.bias", false)?;
    let out = layer_norm2d_nchw(&mut b.m(), feat, ln2_g, ln2_b, eps);

    b.hir.set_outputs(vec![out]);

    Ok((b.hir, b.params, preprocess))
}

/// Lowered graph wrapper for legacy callers (via [`super::flow::SamEncoderFlow`]).
pub fn build_sam_encoder_graph(
    cfg: &SamEncoderConfig,
    weights: &mut WeightMap,
) -> Result<(Graph, HashMap<String, Vec<f32>>, SamPreprocessWeights)> {
    let built = super::flow::build_sam_encoder_built(cfg, weights)?;
    let (graph, params) = rlx_core::flow_util::graph_from_built(built.model)?;
    Ok((graph, params, built.preprocess))
}

/// Global-attention block: full self-attention over all `hw·hw` tokens.
#[allow(clippy::too_many_arguments)]
fn attention_global(
    sb: &mut SamBuilder,
    w: &mut WeightMap,
    lp: &str,
    x: HirNodeId, // [1, S, E]
    e: usize,
    nh: usize,
    dh: usize,
    scale: f32,
    hw: usize,
    use_rel_pos: bool,
    qkv_bias: bool,
) -> Result<HirNodeId> {
    let s = hw * hw;
    decomposed_attention(
        sb,
        w,
        lp,
        x,
        e,
        nh,
        dh,
        scale,
        hw,
        hw,
        s,
        1,
        use_rel_pos,
        qkv_bias,
    )
}

/// Windowed-attention block: pad → partition into `nW = (hw_p/ws)²`
/// windows → attention within each window → reverse partition → crop.
#[allow(clippy::too_many_arguments)]
fn attention_windowed(
    sb: &mut SamBuilder,
    w: &mut WeightMap,
    lp: &str,
    x: HirNodeId, // [1, S, E] flat (= [1, hw, hw, E] BHWC, flattened)
    e: usize,
    nh: usize,
    dh: usize,
    scale: f32,
    hw: usize,
    ws: usize,
    use_rel_pos: bool,
    qkv_bias: bool,
) -> Result<HirNodeId> {
    // Restore spatial: [1, S, E] → [1, hw, hw, E]
    let bhwc = sb.m().reshape_(x, vec![1, hw as i64, hw as i64, e as i64]);

    let pad = (ws - hw % ws) % ws;
    let hw_p = hw + pad;
    let n_win_per_side = hw_p / ws;
    let n_win = n_win_per_side * n_win_per_side;

    // Pad with concat-zeros along axes 1, 2.
    let padded = if pad > 0 {
        let z_h = pad_zero_param(sb, &format!("{lp}.attn._pad_h"), &[1, pad, hw, e]);
        let p1 = sb.m().concat_(vec![bhwc, z_h], 1); // [1, hw_p, hw, E]
        let z_w = pad_zero_param(sb, &format!("{lp}.attn._pad_w"), &[1, hw_p, pad, e]);
        sb.m().concat_(vec![p1, z_w], 2) // [1, hw_p, hw_p, E]
    } else {
        bhwc
    };

    // [1, hw_p, hw_p, E] → [1, nw, ws, nw, ws, E] → transpose(2,3)
    //   → [1, nw, nw, ws, ws, E] → reshape [nw², ws, ws, E]
    let reshaped = sb.m().reshape_(
        padded,
        vec![
            1,
            n_win_per_side as i64,
            ws as i64,
            n_win_per_side as i64,
            ws as i64,
            e as i64,
        ],
    );
    let transposed = sb.m().transpose_(reshaped, vec![0, 1, 3, 2, 4, 5]);
    let windowed = sb.m().reshape_(
        transposed,
        vec![n_win as i64, ws as i64, ws as i64, e as i64],
    );
    // Flatten spatial for the attention: [nw², ws², E]
    let win_flat = sb
        .m()
        .reshape_(windowed, vec![n_win as i64, (ws * ws) as i64, e as i64]);

    // Run decomposed attention. Window has spatial dims (ws, ws);
    // sequence length S = ws·ws; batch dim = n_win.
    let attn_out = decomposed_attention(
        sb,
        w,
        lp,
        win_flat,
        e,
        nh,
        dh,
        scale,
        ws,
        ws,
        ws * ws,
        n_win,
        use_rel_pos,
        qkv_bias,
    )?;
    // attn_out: [nw², ws·ws, E]

    // Reverse: [nw², ws², E] → [nw², ws, ws, E] → [1, nw, nw, ws, ws, E]
    //   → transpose(2,3) → [1, nw, ws, nw, ws, E] → [1, hw_p, hw_p, E]
    let un = sb
        .m()
        .reshape_(attn_out, vec![n_win as i64, ws as i64, ws as i64, e as i64]);
    let un = sb.m().reshape_(
        un,
        vec![
            1,
            n_win_per_side as i64,
            n_win_per_side as i64,
            ws as i64,
            ws as i64,
            e as i64,
        ],
    );
    let un = sb.m().transpose_(un, vec![0, 1, 3, 2, 4, 5]);
    let un = sb
        .m()
        .reshape_(un, vec![1, hw_p as i64, hw_p as i64, e as i64]);
    // Crop off the padding
    let un = if pad > 0 {
        let cropped_h = sb.m().narrow_(un, 1, 0, hw);
        sb.m().narrow_(cropped_h, 2, 0, hw)
    } else {
        un
    };
    // Flatten back to [1, S, E]
    Ok(sb.m().reshape_(un, vec![1, (hw * hw) as i64, e as i64]))
}

/// Decomposed multi-head attention with optional decomposed rel_pos.
/// Input `[B, S, E]`; output `[B, S, E]`.
///
/// `h, w` are the spatial dims of the attention window (S = h·w).
/// For windowed attention `B = n_win`, `h = w = ws`. For global,
/// `B = 1`, `h = w = hw`.
#[allow(clippy::too_many_arguments)]
fn decomposed_attention(
    sb: &mut SamBuilder,
    w: &mut WeightMap,
    lp: &str,
    x: HirNodeId, // [B, S, E]
    e: usize,
    nh: usize,
    dh: usize,
    scale: f32,
    h: usize,
    w_dim: usize,
    s: usize, // = h * w_dim
    batch: usize,
    use_rel_pos: bool,
    qkv_bias: bool,
) -> Result<HirNodeId> {
    // 1) QKV projection. Bias param is loaded *before* the mm so its
    //    HirNodeId is lower — `FuseMatMulBiasAct` walks nodes in topo
    //    order and assumes the bias has been copied into the new id
    //    map before the matmul is rewritten.
    let qkv_w_node = load_p(sb, w, &format!("{lp}.attn.qkv.weight"), true)?;
    let qkv_b_node = if qkv_bias {
        Some(load_p(sb, w, &format!("{lp}.attn.qkv.bias"), false)?)
    } else {
        None
    };
    let qkv_mm = sb.m().mm(x, qkv_w_node); // [B, S, 3E]
    let qkv = if let Some(b) = qkv_b_node {
        sb.m().add(qkv_mm, b)
    } else {
        qkv_mm
    };

    // 2) Reshape & permute to [3, B·nh, S, dh].
    //    [B, S, 3E] → [B, S, 3, nh, dh] → permute(2,0,3,1,4) → [3, B, nh, S, dh]
    //    → reshape [3, B·nh, S, dh].
    let qkv5 = sb
        .m()
        .reshape_(qkv, vec![batch as i64, s as i64, 3, nh as i64, dh as i64]);
    let qkv_perm = sb.m().transpose_(qkv5, vec![2, 0, 3, 1, 4]); // [3, B, nh, S, dh]
    let qkv_flat = sb
        .m()
        .reshape_(qkv_perm, vec![3, (batch * nh) as i64, s as i64, dh as i64]);
    let q = sb.m().narrow_(qkv_flat, 0, 0, 1);
    let q = sb
        .m()
        .reshape_(q, vec![(batch * nh) as i64, s as i64, dh as i64]);
    let k = sb.m().narrow_(qkv_flat, 0, 1, 1);
    let k = sb
        .m()
        .reshape_(k, vec![(batch * nh) as i64, s as i64, dh as i64]);
    let v = sb.m().narrow_(qkv_flat, 0, 2, 1);
    let v = sb
        .m()
        .reshape_(v, vec![(batch * nh) as i64, s as i64, dh as i64]);

    // 3) attn = (q * scale) @ k.T   shape [B·nh, S, S]
    let scale_node = scalar_param(sb, &format!("{lp}.attn._scale"), scale);
    let q_scaled = sb.m().mul(q, scale_node);
    let k_t = sb.m().transpose_(k, vec![0, 2, 1]); // [B·nh, dh, S]
    let scores = sb.m().mm(q_scaled, k_t); // [B·nh, S, S]

    // 4) Optionally add decomposed rel_pos.
    let scores = if use_rel_pos {
        // rel_pos_h: [2h-1, dh]  rel_pos_w: [2w-1, dh]
        // We pre-resolve get_rel_pos() host-side into r_h: [h, h, dh] and
        // r_w: [w, w, dh] indexed buffers (cheap, ≤ 27×27×64 elements).
        let (mut r_h_data, mut r_w_data) = extract_rel_pos(w, lp, h, w_dim, dh)?;
        // Bisect helpers:
        //   RLX_SAM_DEBUG_ZERO_RELPOS=1  zero both r_h and r_w
        //   RLX_SAM_DEBUG_ZERO_RELH=1    zero only r_h (keep rel_w)
        //   RLX_SAM_DEBUG_ZERO_RELW=1    zero only r_w (keep rel_h)
        if rlx_ir::env::flag("RLX_SAM_DEBUG_ZERO_RELPOS") {
            r_h_data.iter_mut().for_each(|v| *v = 0.0);
            r_w_data.iter_mut().for_each(|v| *v = 0.0);
        }
        if rlx_ir::env::flag("RLX_SAM_DEBUG_ZERO_RELH") {
            r_h_data.iter_mut().for_each(|v| *v = 0.0);
        }
        if rlx_ir::env::flag("RLX_SAM_DEBUG_ZERO_RELW") {
            r_w_data.iter_mut().for_each(|v| *v = 0.0);
        }
        let r_h_node = const_param(
            sb,
            &format!("{lp}.attn._rel_h_indexed"),
            &[h, h, dh],
            r_h_data,
        );
        let r_w_node = const_param(
            sb,
            &format!("{lp}.attn._rel_w_indexed"),
            &[w_dim, w_dim, dh],
            r_w_data,
        );
        add_decomposed_rel_pos(sb, scores, q, r_h_node, r_w_node, batch, nh, h, w_dim, dh)?
    } else {
        scores
    };

    // 5) softmax over last axis
    let attn_w = sb.m().sm(scores, -1);

    // 6) attn @ V → [B·nh, S, dh]
    let attn_v = sb.m().mm(attn_w, v);

    // 7) Reverse the head split: [B·nh, S, dh] → [B, nh, S, dh] → [B, S, nh, dh] → [B, S, E]
    let reshaped = sb
        .m()
        .reshape_(attn_v, vec![batch as i64, nh as i64, s as i64, dh as i64]);
    let perm = sb.m().transpose_(reshaped, vec![0, 2, 1, 3]); // [B, S, nh, dh]
    let merged = sb
        .m()
        .reshape_(perm, vec![batch as i64, s as i64, e as i64]);

    // 8) Output projection (always biased).
    let proj_w = load_p(sb, w, &format!("{lp}.attn.proj.weight"), true)?;
    let proj_b = load_p(sb, w, &format!("{lp}.attn.proj.bias"), false)?;
    let proj_mm = sb.m().mm(merged, proj_w);
    Ok(sb.m().add(proj_mm, proj_b))
}

/// Add decomposed relative positional bias to attention scores.
///
/// Math (per the SAM paper, candle's `add_decomposed_rel_pos`):
///   r_q = q.reshape(B·nh, h, w, dh)
///   rel_h[bhw,c] = sum_c r_q[bhw,c] · r_h_indexed[hq, hk, c]    → [B·nh, h, w, h]
///   rel_w[bhw,c] = sum_c r_q[bhw,c] · r_w_indexed[wq, wk, c]    → [B·nh, h, w, w]
///   scores += rel_h.unsqueeze(4) + rel_w.unsqueeze(3)           → [B·nh, h, w, h, w]
///   scores.reshape(B·nh, h·w, h·w)
#[allow(clippy::too_many_arguments)]
fn add_decomposed_rel_pos(
    sb: &mut SamBuilder,
    scores: HirNodeId, // [B·nh, S, S]
    q: HirNodeId,      // [B·nh, S, dh]
    r_h: HirNodeId,    // [h, h, dh]  (pre-indexed)
    r_w: HirNodeId,    // [w, w, dh]
    batch: usize,
    nh: usize,
    h: usize,
    w: usize,
    dh: usize,
) -> Result<HirNodeId> {
    let bh = batch * nh;
    // r_q: [bh, h, w, dh]
    let r_q = sb
        .m()
        .reshape_(q, vec![bh as i64, h as i64, w as i64, dh as i64]);

    // rel_h: "bhwc, hkc -> bhwk".
    // Unrolled-per-h_q: rlx-cpu's batched 3-D matmul gives subtly wrong
    // results in this exact shape regime, so we lower the einsum to
    // `h` independent 2-D matmuls (one per h_q index) and `sb.m().concat_`
    // them back. Each per-h_q matmul is `[bh, w, dh] @ [dh, h_k]`,
    // which uses the well-tested flat sgemm path (rhs has no batch
    // dim, only the lhs does — that's the case the Sgemm flatten
    // trick was designed for).
    let mut rel_h_slices: Vec<HirNodeId> = Vec::with_capacity(h);
    for h_q in 0..h {
        // r_q at h_q: narrow axis 1, then squeeze.
        let rq_slice = sb.m().narrow_(r_q, 1, h_q, 1); // [bh, 1, w, dh]
        let rq_slice = sb
            .m()
            .reshape_(rq_slice, vec![bh as i64, w as i64, dh as i64]);
        // r_h at h_q: narrow axis 0, then squeeze + transpose to [dh, h].
        let rh_slice = sb.m().narrow_(r_h, 0, h_q, 1); // [1, h, dh]
        let rh_slice = sb.m().reshape_(rh_slice, vec![h as i64, dh as i64]); // [h_k, dh]
        let rh_t = sb.m().transpose_(rh_slice, vec![1, 0]); // [dh, h_k]
        let mm = sb.m().mm(rq_slice, rh_t); // [bh, w, h_k]
        // Add a leading length-1 axis so we can concat into [bh, h, w, h_k].
        let mm5 = sb.m().reshape_(mm, vec![bh as i64, 1, w as i64, h as i64]);
        rel_h_slices.push(mm5);
    }
    let rel_h_4d = sb.m().concat_(rel_h_slices, 1); // [bh, h, w, h]

    // rel_w: same idea, w_q as the unrolled axis.
    let mut rel_w_slices: Vec<HirNodeId> = Vec::with_capacity(w);
    for w_q in 0..w {
        let rq_slice = sb.m().narrow_(r_q, 2, w_q, 1); // [bh, h, 1, dh]
        let rq_slice = sb
            .m()
            .reshape_(rq_slice, vec![bh as i64, h as i64, dh as i64]);
        let rw_slice = sb.m().narrow_(r_w, 0, w_q, 1); // [1, w, dh]
        let rw_slice = sb.m().reshape_(rw_slice, vec![w as i64, dh as i64]); // [w_k, dh]
        let rw_t = sb.m().transpose_(rw_slice, vec![1, 0]); // [dh, w_k]
        let mm = sb.m().mm(rq_slice, rw_t); // [bh, h, w_k]
        let mm5 = sb.m().reshape_(mm, vec![bh as i64, h as i64, 1, w as i64]);
        rel_w_slices.push(mm5);
    }
    let rel_w_4d = sb.m().concat_(rel_w_slices, 2); // [bh, h, w, w]

    // Broadcast-add into the [bh, h, w, h, w] view of scores.
    //
    // History: rlx-cpu's BiasAdd misroute for mid-shape singletons is
    // now fixed (`is_trailing_bias_broadcast`), so CPU uses simple
    // unsqueeze+add. The rlx-metal BinaryBroadcast MSL kernel exists
    // but produces wrong results on the SAM rel_pos pattern (suspect:
    // setBytes alignment of inline `constant uint*` for ranks > 4 —
    // needs focused debugging). Until then, materialise both rel
    // tensors to the full output shape via `concat`-tile so the add
    // is a same-shape op and works on every backend.
    let scores_5d = sb.m().reshape_(
        scores,
        vec![bh as i64, h as i64, w as i64, h as i64, w as i64],
    );
    let rel_h_5d = sb
        .m()
        .reshape_(rel_h_4d, vec![bh as i64, h as i64, w as i64, h as i64, 1]);
    let rel_h_tiled = {
        let mut copies = Vec::with_capacity(w);
        for _ in 0..w {
            copies.push(rel_h_5d);
        }
        sb.m().concat_(copies, 4) // [bh, h, w, h, w]
    };
    let rel_w_5d = sb
        .m()
        .reshape_(rel_w_4d, vec![bh as i64, h as i64, w as i64, 1, w as i64]);
    let rel_w_tiled = {
        let mut copies = Vec::with_capacity(h);
        for _ in 0..h {
            copies.push(rel_w_5d);
        }
        sb.m().concat_(copies, 3) // [bh, h, w, h, w]
    };
    let s1 = sb.m().add(scores_5d, rel_h_tiled);
    let s2 = sb.m().add(s1, rel_w_tiled);
    Ok(sb
        .m()
        .reshape_(s2, vec![bh as i64, (h * w) as i64, (h * w) as i64]))
}

/// Resolve candle's `get_rel_pos()` host-side into per-axis bias
/// tables of shape `[q_size, k_size, dh]` (here q_size == k_size).
///
/// Stored `rel_pos_h` has shape `[2·max(q,k)-1, dh]`; we gather along
/// axis 0 using `relative_coords[i,j] = i - j + (k-1)` (since q==k,
/// scale factors collapse to 1).
fn extract_rel_pos(
    weights: &mut WeightMap,
    lp: &str,
    h: usize,
    w: usize,
    dh: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let (rel_h_raw, rh_shape) = weights.take(&format!("{lp}.attn.rel_pos_h"))?;
    let (rel_w_raw, rw_shape) = weights.take(&format!("{lp}.attn.rel_pos_w"))?;
    ensure!(
        rh_shape == vec![2 * h - 1, dh],
        "{lp}.attn.rel_pos_h expected [{}, {dh}], got {rh_shape:?}",
        2 * h - 1
    );
    ensure!(
        rw_shape == vec![2 * w - 1, dh],
        "{lp}.attn.rel_pos_w expected [{}, {dh}], got {rw_shape:?}",
        2 * w - 1
    );

    let mut r_h = vec![0f32; h * h * dh];
    for q in 0..h {
        for k in 0..h {
            let idx = (q as isize - k as isize + (h as isize - 1)) as usize;
            let src = &rel_h_raw[idx * dh..(idx + 1) * dh];
            let dst = &mut r_h[(q * h + k) * dh..(q * h + k + 1) * dh];
            dst.copy_from_slice(src);
        }
    }
    let mut r_w = vec![0f32; w * w * dh];
    for q in 0..w {
        for k in 0..w {
            let idx = (q as isize - k as isize + (w as isize - 1)) as usize;
            let src = &rel_w_raw[idx * dh..(idx + 1) * dh];
            let dst = &mut r_w[(q * w + k) * dh..(q * w + k + 1) * dh];
            dst.copy_from_slice(src);
        }
    }
    Ok((r_h, r_w))
}

// ─── Neck (Conv2d 1×1 + LN2d + Conv2d 3×3 + LN2d) host-side ────────

/// Weights for the four neck layers, kept on the host because rlx-ir
/// doesn't have f32 forward Conv2d (and 3×3 padding=1 doesn't reduce
/// to matmul).
pub struct NeckWeights {
    pub conv1_w: Vec<f32>, // [out_chans, embed_dim] (1×1 conv = per-channel linear)
    pub ln1_g: Vec<f32>,   // [out_chans]
    pub ln1_b: Vec<f32>,
    pub conv2_w: Vec<f32>, // [out_chans, out_chans, 3, 3]
    pub ln2_g: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub embed_dim: usize,
    pub out_chans: usize,
    pub eps: f32,
}

#[allow(dead_code)]
fn extract_neck_weights(weights: &mut WeightMap, cfg: &SamEncoderConfig) -> Result<NeckWeights> {
    let (conv1_w_raw, c1_shape) = weights.take("image_encoder.neck.0.weight")?;
    ensure!(
        c1_shape == vec![cfg.out_chans, cfg.embed_dim, 1, 1],
        "neck.0.weight expected [{}, {}, 1, 1], got {c1_shape:?}",
        cfg.out_chans,
        cfg.embed_dim
    );
    let conv1_w = conv1_w_raw; // [out_chans, embed_dim] after flattening last two singleton dims
    let (ln1_g, _) = weights.take("image_encoder.neck.1.weight")?;
    let (ln1_b, _) = weights.take("image_encoder.neck.1.bias")?;
    let (conv2_w, c2_shape) = weights.take("image_encoder.neck.2.weight")?;
    ensure!(
        c2_shape == vec![cfg.out_chans, cfg.out_chans, 3, 3],
        "neck.2.weight expected [{}, {}, 3, 3], got {c2_shape:?}",
        cfg.out_chans,
        cfg.out_chans
    );
    let (ln2_g, _) = weights.take("image_encoder.neck.3.weight")?;
    let (ln2_b, _) = weights.take("image_encoder.neck.3.bias")?;
    Ok(NeckWeights {
        conv1_w,
        ln1_g,
        ln1_b,
        conv2_w,
        ln2_g,
        ln2_b,
        embed_dim: cfg.embed_dim,
        out_chans: cfg.out_chans,
        eps: cfg.layer_norm_eps as f32,
    })
}

/// Run the encoder neck on the host. `body_out` is the encoder body's
/// output reshaped to `[hw·hw, embed_dim]` (BHWC flattened). Returns
/// `[out_chans, hw, hw]` NCHW image embeddings.
pub fn apply_neck_host(neck: &NeckWeights, body_out: &[f32], hw: usize) -> Vec<f32> {
    let e = neck.embed_dim;
    let oc = neck.out_chans;
    let eps = neck.eps;

    // 1) Conv 1×1: per-pixel linear projection from embed_dim → out_chans.
    //    body_out is BHWC; treat as [hw·hw, embed_dim] and matmul by
    //    conv1_w.T (i.e. `out[s, oc] = sum_e body_out[s, e] * conv1_w[oc, e]`).
    let s = hw * hw;
    let mut feat = vec![0f32; s * oc]; // BHWC: [hw·hw, oc]
    for si in 0..s {
        for oi in 0..oc {
            let mut acc = 0f32;
            for ei in 0..e {
                acc += body_out[si * e + ei] * neck.conv1_w[oi * e + ei];
            }
            feat[si * oc + oi] = acc;
        }
    }

    // 2) LN2d: normalize over channel dim (per spatial position).
    layernorm2d_inplace(&mut feat, s, oc, &neck.ln1_g, &neck.ln1_b, eps);

    // 3) Conv 3×3 padding=1, stride=1. We compute it in NCHW. The input
    //    is currently BHWC = [hw·hw, oc]; convert to NCHW = [oc, hw, hw].
    let mut nchw = vec![0f32; oc * hw * hw];
    for y in 0..hw {
        for x in 0..hw {
            for c in 0..oc {
                nchw[c * hw * hw + y * hw + x] = feat[(y * hw + x) * oc + c];
            }
        }
    }
    let conv2_out = conv2d_3x3_pad1(&nchw, oc, oc, hw, hw, &neck.conv2_w);

    // 4) LN2d again. Convert back to BHWC for the LN, then back to NCHW.
    let mut bhwc = vec![0f32; s * oc];
    for c in 0..oc {
        for y in 0..hw {
            for x in 0..hw {
                bhwc[(y * hw + x) * oc + c] = conv2_out[c * hw * hw + y * hw + x];
            }
        }
    }
    layernorm2d_inplace(&mut bhwc, s, oc, &neck.ln2_g, &neck.ln2_b, eps);

    let mut out_nchw = vec![0f32; oc * hw * hw];
    for y in 0..hw {
        for x in 0..hw {
            for c in 0..oc {
                out_nchw[c * hw * hw + y * hw + x] = bhwc[(y * hw + x) * oc + c];
            }
        }
    }
    out_nchw
}

/// LN over channel dim of BHWC `[S, C]` (matches candle's LayerNorm2d).
fn layernorm2d_inplace(data: &mut [f32], s: usize, c: usize, g: &[f32], b: &[f32], eps: f32) {
    for si in 0..s {
        let row = &mut data[si * c..(si + 1) * c];
        let mean: f32 = row.iter().sum::<f32>() / c as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for k in 0..c {
            row[k] = (row[k] - mean) * inv * g[k] + b[k];
        }
    }
}

/// 3×3 Conv2d with stride=1, padding=1, no bias. NCHW in, NCHW out.
/// Reference implementation — not vectorized, fine for the SAM neck
/// (1 call per inference, 64×64×256).
fn conv2d_3x3_pad1(
    input: &[f32],
    in_c: usize,
    out_c: usize,
    h: usize,
    w: usize,
    weight: &[f32], // [out_c, in_c, 3, 3]
) -> Vec<f32> {
    let mut out = vec![0f32; out_c * h * w];
    for oc in 0..out_c {
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0f32;
                for ic in 0..in_c {
                    for ky in 0..3 {
                        let iy = y as isize + ky as isize - 1;
                        if iy < 0 || iy >= h as isize {
                            continue;
                        }
                        for kx in 0..3 {
                            let ix = x as isize + kx as isize - 1;
                            if ix < 0 || ix >= w as isize {
                                continue;
                            }
                            let v = input[ic * h * w + iy as usize * w + ix as usize];
                            let wi = ((oc * in_c + ic) * 3 + ky) * 3 + kx;
                            acc += v * weight[wi];
                        }
                    }
                }
                out[oc * h * w + y * w + x] = acc;
            }
        }
    }
    out
}

// ─── Small builder helpers ─────────────────────────────────────────

fn load_p(
    sb: &mut SamBuilder,
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
    let name = key.to_string();
    let id = sb.m().param(&name, Shape::new(&shape, DType::F32));
    sb.params.insert(name, data);
    Ok(id)
}

#[allow(dead_code)]
fn scalar_param(sb: &mut SamBuilder, name: &str, value: f32) -> HirNodeId {
    let id = sb.m().param(name, Shape::new(&[1], DType::F32));
    sb.params.insert(name.to_string(), vec![value]);
    id
}

fn const_param(sb: &mut SamBuilder, name: &str, shape: &[usize], data: Vec<f32>) -> HirNodeId {
    let id = sb.m().param(name, Shape::new(shape, DType::F32));
    sb.params.insert(name.to_string(), data);
    id
}

fn pad_zero_param(sb: &mut SamBuilder, name: &str, shape: &[usize]) -> HirNodeId {
    let n: usize = shape.iter().product();
    let id = sb.m().param(name, Shape::new(shape, DType::F32));
    sb.params.insert(name.to_string(), vec![0f32; n]);
    id
}
