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

//! One VLASH *joint* transformer layer: a Gemma-2B (VLM) layer and a
//! Gemma-300M (action-expert) layer that share a single attention over the
//! concatenated `[prefix ++ suffix]` sequence.
//!
//! ```text
//!   for each stream s in {vlm(2048), expert(1024)}:
//!     n_s      = norm_in_s(h_s [, cond])            # Gemma (1+w) or adaRMS
//!     q_s,k_s,v_s = n_s @ {q,k,v}_proj_s            # q: 8·256, kv: 1·256
//!   Q = concat(q_vlm, q_expert)  (along seq)        # RoPE(NeoX) applied here
//!   K = concat(k_vlm, k_expert)  → repeat_kv(×8)
//!   V = concat(v_vlm, v_expert)  → repeat_kv(×8)
//!   A = attention_bias(Q, K, V, block_causal_bias, 8 heads, 256)
//!   a_vlm, a_expert = split(A)
//!   h_s = h_s [+ gate_s ·] (a_s @ o_proj_s)         # gated residual (π₀.₅ expert)
//!     n2_s = norm_ff_s(h_s [, cond])
//!     h_s = h_s [+ gate_s ·] geglu_s(n2_s)          # GeGLU: down(gelu_tanh(gate)·up)
//! ```
//!
//! adaRMS (`cond = Some`, π₀.₅ expert): `dense(cond) → [scale, shift, gate]`,
//! `normed = _norm(x)·(1+scale) + shift`, residual `= r + sublayer·gate`
//! (`transformers` GemmaRMSNorm @ commit `dcddb97`). The VLM stream is always
//! standard (`cond = None`).

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::config::{GemmaConfig, VlashConfig};

/// Synthesize a constant vector `[dim]` filled with `val`, named by `key`.
fn synth_const(emit: &mut Emit<'_>, key: &str, val: f32, dim: usize) -> HirNodeId {
    emit.synth_param(key, vec![val; dim], Shape::new(&[dim], DType::F32))
}

/// `x @ Wᵀ` (no bias), loading HF `nn.Linear` weight `{key}` `[out, in]`.
fn linear_nb(emit: &mut Emit<'_>, key: &str, x: HirNodeId) -> Result<HirNodeId> {
    let w = emit.load_param(key, true)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.mm(x, w))
}

/// GQA replication along the last (`heads·head_dim`) axis: each of `num_kv_heads`
/// contiguous `head_dim` slices is repeated `group` times.
fn repeat_kv(
    gb: &mut HirMut,
    x: HirNodeId,
    num_kv_heads: usize,
    head_dim: usize,
    group: usize,
) -> HirNodeId {
    if group == 1 {
        return x;
    }
    let last = gb.shape(x).rank() - 1;
    let mut pieces = Vec::with_capacity(num_kv_heads * group);
    for h in 0..num_kv_heads {
        let slice = gb.narrow_(x, last, h * head_dim, head_dim);
        for _ in 0..group {
            pieces.push(slice);
        }
    }
    gb.concat_(pieces, last)
}

/// Gemma GeGLU MLP: `down(gelu_tanh(gate(x)) · up(x))` under `{prefix}.mlp.*`.
fn emit_geglu(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
    let gate = linear_nb(emit, &format!("{prefix}.mlp.gate_proj.weight"), x)?;
    let up = linear_nb(emit, &format!("{prefix}.mlp.up_proj.weight"), x)?;
    let act = {
        let mut gb = HirMut::new(emit.hir());
        let g = gb.gelu_approx(gate);
        gb.mul(g, up)
    };
    linear_nb(emit, &format!("{prefix}.mlp.down_proj.weight"), act)
}

/// A Gemma norm result: the normalized tensor and (adaRMS only) a per-feature
/// residual gate `[batch, 1, dim]`.
struct NormOut {
    normed: HirNodeId,
    gate: Option<HirNodeId>,
}

/// Emit a Gemma RMSNorm (standard `(1+w)` when `cond` is `None`, adaRMS
/// otherwise). `dim` is the stream hidden width; `batch` is needed to reshape
/// the adaRMS modulation to `[batch, 1, 3·dim]`.
fn emit_norm(
    emit: &mut Emit<'_>,
    key: &str,
    x: HirNodeId,
    dim: usize,
    eps: f32,
    batch: usize,
    cond: Option<HirNodeId>,
) -> Result<NormOut> {
    match cond {
        None => {
            // Standard Gemma: _norm(x) * (1 + weight).
            let w = emit.load_param(&format!("{key}.weight"), false)?;
            let ones = synth_const(emit, &format!("{key}.__ones"), 1.0, dim);
            let zero = synth_const(emit, &format!("{key}.__zero"), 0.0, dim);
            let mut gb = HirMut::new(emit.hir());
            let one_plus_w = gb.add(ones, w);
            let normed = gb.rms_norm(x, one_plus_w, zero, eps);
            Ok(NormOut { normed, gate: None })
        }
        Some(cond) => {
            // adaRMS: modulation = dense(cond) → [scale, shift, gate].
            let dw = emit.load_param(&format!("{key}.dense.weight"), true)?; // [cond_dim, 3·dim]
            let db = emit.load_param(&format!("{key}.dense.bias"), false)?; // [3·dim]
            let ones = synth_const(emit, &format!("{key}.__ones"), 1.0, dim);
            let zero = synth_const(emit, &format!("{key}.__zero"), 0.0, dim);
            let mut gb = HirMut::new(emit.hir());
            // modulation: [batch, cond_dim] @ [cond_dim, 3·dim] + bias → [batch, 3·dim].
            let m = gb.mm(cond, dw);
            let m = gb.add(m, db);
            let m = gb.reshape_(m, vec![batch as i64, 1, 3 * dim as i64]);
            let scale = gb.narrow_(m, 2, 0, dim);
            let shift = gb.narrow_(m, 2, dim, dim);
            let gate = gb.narrow_(m, 2, 2 * dim, dim);
            // normed = _norm(x) * (1 + scale) + shift   (broadcast over seq).
            let base = gb.rms_norm(x, ones, zero, eps); // pure _norm(x)
            // (1 + scale): add the [dim] ones vector, broadcast over [batch,1,dim].
            let scale1 = gb.add(scale, ones);
            let scaled = gb.mul(base, scale1);
            let normed = gb.add(scaled, shift);
            Ok(NormOut {
                normed,
                gate: Some(gate),
            })
        }
    }
}

/// Apply a (possibly gated) residual: `residual + sublayer` or
/// `residual + sublayer · gate` (broadcast `[batch,1,dim]` over seq).
fn gated_residual(
    gb: &mut HirMut,
    residual: HirNodeId,
    sublayer: HirNodeId,
    gate: Option<HirNodeId>,
) -> HirNodeId {
    match gate {
        None => gb.add(residual, sublayer),
        Some(g) => {
            let scaled = gb.mul(sublayer, g);
            gb.add(residual, scaled)
        }
    }
}

/// One joint layer over the two streams. `prefix`/`suffix` are the current
/// hidden states (`[batch, P, 2048]` / `[batch, S, expert_hidden]`); `cos`/`sin`
/// are the RoPE tables for the full `P+S` sequence; `bias` is the block-causal
/// additive mask `[batch, heads, P+S, P+S]`; `cond` is the adaRMS conditioning
/// for the expert stream (`None` for π₀). Returns the updated `(prefix, suffix)`.
#[allow(clippy::too_many_arguments)]
pub fn emit_joint_layer(
    emit: &mut Emit<'_>,
    cfg: &VlashConfig,
    idx: usize,
    prefix: HirNodeId,
    suffix: HirNodeId,
    cos: HirNodeId,
    sin: HirNodeId,
    bias: HirNodeId,
    cond: Option<HirNodeId>,
    batch: usize,
    p_len: usize,
    s_len: usize,
) -> Result<(HirNodeId, HirNodeId)> {
    let vlm = &cfg.vlm;
    let expert = &cfg.expert;
    let head_dim = cfg.head_dim();
    let heads = cfg.heads();
    let eps = vlm.rms_eps;
    let vk = format!("vlm.layers.{idx}");
    let ek = format!("expert.layers.{idx}");

    // --- pre-attention norm (VLM standard, expert std/adaRMS) ---
    let n_p = emit_norm(emit, &format!("{vk}.input_layernorm"), prefix, vlm.hidden, eps, batch, None)?;
    let n_s = emit_norm(
        emit,
        &format!("{ek}.input_layernorm"),
        suffix,
        expert.hidden,
        eps,
        batch,
        cond,
    )?;

    // --- per-stream Q/K/V ---
    let qp = linear_nb(emit, &format!("{vk}.self_attn.q_proj.weight"), n_p.normed)?;
    let kp = linear_nb(emit, &format!("{vk}.self_attn.k_proj.weight"), n_p.normed)?;
    let vp = linear_nb(emit, &format!("{vk}.self_attn.v_proj.weight"), n_p.normed)?;
    let qs = linear_nb(emit, &format!("{ek}.self_attn.q_proj.weight"), n_s.normed)?;
    let ks = linear_nb(emit, &format!("{ek}.self_attn.k_proj.weight"), n_s.normed)?;
    let vs = linear_nb(emit, &format!("{ek}.self_attn.v_proj.weight"), n_s.normed)?;

    // --- shared attention over [prefix ++ suffix] ---
    let attn = {
        let mut gb = HirMut::new(emit.hir());
        let q = gb.concat_(vec![qp, qs], 1);
        let k = gb.concat_(vec![kp, ks], 1);
        let v = gb.concat_(vec![vp, vs], 1);
        let q = gb.rope(q, cos, sin, head_dim);
        let k = gb.rope(k, cos, sin, head_dim);
        let group = vlm.kv_group();
        let k = repeat_kv(&mut gb, k, vlm.num_kv_heads, head_dim, group);
        let v = repeat_kv(&mut gb, v, vlm.num_kv_heads, head_dim, group);
        let attn_shape = rlx_ir::shape::attention_shape(gb.shape(q));
        gb.inner().mir(
            rlx_ir::ops::attention::attention_kind_op(
                heads,
                head_dim,
                rlx_ir::op::MaskKind::Bias,
                Some(cfg.vlm.score_scale()),
                None,
            ),
            vec![q, k, v, bias],
            attn_shape,
        )
    };

    // --- split back per stream ---
    let (attn_p, attn_s) = {
        let mut gb = HirMut::new(emit.hir());
        let ap = gb.narrow_(attn, 1, 0, p_len);
        let as_ = gb.narrow_(attn, 1, p_len, s_len);
        (ap, as_)
    };

    // --- o_proj + gated residual ---
    let out_p = linear_nb(emit, &format!("{vk}.self_attn.o_proj.weight"), attn_p)?;
    let out_s = linear_nb(emit, &format!("{ek}.self_attn.o_proj.weight"), attn_s)?;
    let (prefix, suffix) = {
        let mut gb = HirMut::new(emit.hir());
        let p = gated_residual(&mut gb, prefix, out_p, n_p.gate);
        let s = gated_residual(&mut gb, suffix, out_s, n_s.gate);
        (p, s)
    };

    // --- post-attention norm + GeGLU MLP + gated residual ---
    let n2_p = emit_norm(
        emit,
        &format!("{vk}.post_attention_layernorm"),
        prefix,
        vlm.hidden,
        eps,
        batch,
        None,
    )?;
    let n2_s = emit_norm(
        emit,
        &format!("{ek}.post_attention_layernorm"),
        suffix,
        expert.hidden,
        eps,
        batch,
        cond,
    )?;
    let mlp_p = emit_geglu(emit, &vk, n2_p.normed)?;
    let mlp_s = emit_geglu(emit, &ek, n2_s.normed)?;
    let (prefix, suffix) = {
        let mut gb = HirMut::new(emit.hir());
        let p = gated_residual(&mut gb, prefix, mlp_p, n2_p.gate);
        let s = gated_residual(&mut gb, suffix, mlp_s, n2_s.gate);
        (p, s)
    };

    Ok((prefix, suffix))
}

/// Final Gemma norm for one stream (standard or adaRMS), returning just the
/// normalized tensor (the final norm's gate is discarded, matching the
/// reference `self.norm(hidden, cond)[0]`).
pub fn emit_final_norm(
    emit: &mut Emit<'_>,
    key: &str,
    x: HirNodeId,
    dim: usize,
    eps: f32,
    batch: usize,
    cond: Option<HirNodeId>,
    _g: &GemmaConfig,
) -> Result<HirNodeId> {
    Ok(emit_norm(emit, key, x, dim, eps, batch, cond)?.normed)
}
