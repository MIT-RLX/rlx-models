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

//! Gated GQA attention — `Qwen4ExpTextAttention`, every fourth layer.
//!
//! ```text
//!   [q | gate] = x @ attn_qᵀ                 [s, n_head, 2·head_dim]  interleaved
//!   q          = RMSNorm(q);  k = RMSNorm(x @ attn_kᵀ)
//!   q, k       = M-RoPE(q), M-RoPE(k)        partial: 64 of 256 dims
//!   y          = SDPA(q, k, v, QSA mask)     24 q heads over 2 kv heads
//!   out        = (y · σ(gate)) @ attn_outputᵀ
//! ```
//!
//! Two layout details:
//!
//! * **`attn_q` is one projection holding query *and* gate, interleaved per
//!   head** — `[q₀ | g₀ | q₁ | g₁ | …]`, not `[all q | all gate]`. Splitting it
//!   the other way silently pairs each head's query with the next head's gate.
//! * **The GQA expansion interleaves** — key/value head `h / (n_head/n_kv)`
//!   serves query head `h`, matching `repeat_kv` and ggml's `mul_mat`
//!   broadcast. This is the *opposite* of the GatedDeltaNet layers' tiled
//!   repeat (see [`crate::gdn`]), because only the linear-attention tensors
//!   were permuted by the converter.
//!
//! The score scale is the default `head_dim^-0.5`, which `Op::Attention`
//! applies itself, so `score_scale` stays `None`.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::{MaskKind, Op};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::common::{linear, rms_gamma, sigmoid};
use crate::config::Qwen4ExpConfig;
use crate::indexer::{QsaDims, emit_qsa_bias};

/// The key/value cache a gated-attention layer reads at decode.
///
/// The new token's rotated key and value are appended at the **end** of the
/// cache and the mask decides what is live, so the cache does not have to be
/// kept in position order. That is safe because attention is
/// permutation-invariant over keys once the mask is applied — and the position
/// is already baked into the key by the rotation before it is stored.
///
/// It is *not* safe for the QSA indexer, whose blocks are groups of four
/// consecutive cache cells. [`emit_attention_with_cache`] therefore only
/// accepts a cache while the selection is provably the identity, where the
/// indexer is skipped entirely.
#[derive(Debug, Clone, Copy)]
pub struct AttnCache {
    /// `[1, cap, n_kv_heads · head_dim]` — rotated keys for the prefix.
    pub k: HirNodeId,
    /// `[1, cap, n_kv_heads · head_dim]`.
    pub v: HirNodeId,
    /// `[1, cap + 1]` — `1.0` live, `0.0` ignored. The last slot is this token,
    /// which is always live.
    pub mask: HirNodeId,
    pub cap: usize,
}

/// Additive penalty standing in for `-inf`; finite so an all-masked row cannot
/// produce NaN, large enough to vanish under softmax.
const MASK_PENALTY: f32 = -1.0e30;

/// The new token's cache row, to be written back by the caller.
#[derive(Debug, Clone, Copy)]
pub struct AttnCacheOut {
    /// `[1, 1, n_kv_heads · head_dim]` — **rotated**. Storing the unrotated key
    /// and rotating on read would need the whole cache re-rotated every step.
    pub k: HirNodeId,
    pub v: HirNodeId,
}

/// Static shape/hyper-parameters of one gated-attention layer.
#[derive(Debug, Clone, Copy)]
pub struct AttnDims {
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// `rope_dim_count` — only this many of `head_dim` rotate.
    pub n_rot: usize,
    pub eps: f32,
    pub norm_offset: bool,
    pub seq: usize,
}

impl AttnDims {
    /// Derive from the model config, for a step of `seq` positions.
    ///
    /// There is exactly one statement of how a config becomes these dimensions, so
    /// prefill, chunked prefill, the decode step, the MTP block and the tests
    /// cannot disagree about what model they are building. They used to hold four
    /// copies of it.
    pub fn from_config(cfg: &Qwen4ExpConfig, seq: usize) -> Self {
        AttnDims {
            hidden: cfg.hidden_size,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            n_rot: cfg.rope_dim_count,
            eps: cfg.rms_norm_eps,
            norm_offset: cfg.rms_norm_offset,
            seq,
        }
    }
}

/// Per-head RMSNorm on a packed `[1, seq, heads · head_dim]` tensor.
fn per_head_rms(
    emit: &mut Emit<'_>,
    key: &str,
    x: HirNodeId,
    heads: usize,
    head_dim: usize,
    eps: f32,
    offset: bool,
) -> Result<HirNodeId> {
    let gain = rms_gamma(emit, &format!("{key}.weight"), head_dim, offset)?;
    let zb = emit.synth_zeros(&format!("{key}.zb"), head_dim);
    let mut gb = HirMut::new(emit.hir());
    let seq = gb.shape(x).dim(1).unwrap_static() as i64;
    let r = gb.reshape_(x, vec![1, seq, heads as i64, head_dim as i64]);
    let n = gb.rms_norm(r, gain, zb, eps);
    Ok(gb.reshape_(n, vec![1, seq, (heads * head_dim) as i64]))
}

/// Widen `[1, seq, in_heads · head_dim]` to `out_heads` by repeating each head
/// `factor` times in place — the `repeat_kv` / ggml broadcast order.
fn repeat_kv(
    gb: &mut HirMut<'_>,
    x: HirNodeId,
    in_heads: usize,
    head_dim: usize,
    factor: usize,
) -> HirNodeId {
    if factor <= 1 {
        return x;
    }
    let last = gb.shape(x).rank() - 1;
    let mut pieces = Vec::with_capacity(in_heads * factor);
    for h in 0..in_heads {
        let slice = gb.narrow_(x, last, h * head_dim, head_dim);
        for _ in 0..factor {
            pieces.push(slice);
        }
    }
    gb.concat_(pieces, last)
}

/// Emit the gated-attention branch for GGUF block `prefix` over
/// `[1, seq, hidden]`, returning `[1, seq, hidden]`.
///
/// `qsa` is `None` on a layer with no indexer (nothing in the published
/// checkpoint, but the metadata permits it); otherwise the selection runs and
/// its bias replaces the causal mask — the bias already *contains* the causal
/// mask, because an unselected key and a future key are both `MASK_PENALTY`.
pub fn emit_attention(
    emit: &mut Emit<'_>,
    prefix: &str,
    x: HirNodeId,
    cos: HirNodeId,
    sin: HirNodeId,
    d: AttnDims,
    qsa: Option<QsaDims>,
) -> Result<HirNodeId> {
    let (out, _) = emit_attention_with_cache(emit, prefix, x, cos, sin, d, qsa, None)?;
    Ok(out)
}

/// [`emit_attention`] reading a key/value cache.
///
/// `None` is the stateless prefill. `Some(cache)` attends over
/// `[prefix | this token]` and returns the new token's rows; the QSA indexer
/// must then be `None`, because block membership is defined over consecutive
/// cache cells and this path appends out of order (see [`AttnCache`]). The
/// caller is responsible for only taking this path where selection is the
/// identity — [`crate::decode`] refuses a capacity past the budget rather than
/// quietly running dense attention, which would be a different model.
#[allow(clippy::too_many_arguments)]
pub fn emit_attention_with_cache(
    emit: &mut Emit<'_>,
    prefix: &str,
    x: HirNodeId,
    cos: HirNodeId,
    sin: HirNodeId,
    d: AttnDims,
    qsa: Option<QsaDims>,
    cache: Option<AttnCache>,
) -> Result<(HirNodeId, Option<AttnCacheOut>)> {
    if cache.is_some() && qsa.is_some_and(|q| !q.is_dense() || q.force_emit) {
        anyhow::bail!(
            "qwen4exp: the QSA indexer cannot run against a key/value cache — \
             its blocks are groups of {} consecutive cells and a cached decode \
             appends out of order. Only the dense short-circuit is cached.",
            qsa.map(|q| q.ratio).unwrap_or(0)
        );
    }
    let f = DType::F32;
    let (s, si) = (d.seq, d.seq as i64);
    let (nh, nkv, hd) = (d.n_heads, d.n_kv_heads, d.head_dim);
    let q_dim = nh * hd;

    // The indexer reads the same block input as q/k/v, so it must run before
    // anything consumes `x`.
    let bias = match qsa {
        Some(q) if cache.is_none() => emit_qsa_bias(emit, prefix, x, cos, sin, q, nh)?,
        _ => None,
    };

    // `[chunk, cap + chunk]`: free over a live prefix cell, causal among the
    // new tokens. Hoisted here because it needs `emit`, which the graph-builder
    // borrow below takes exclusively.
    let cached_bias = match cache.filter(|_| s > 1) {
        Some(c) => {
            let len = c.cap + s;
            let tri: Vec<f32> = (0..s)
                .flat_map(|q| {
                    (0..len).map(move |k| {
                        if k < c.cap || k - c.cap <= q {
                            0.0
                        } else {
                            MASK_PENALTY
                        }
                    })
                })
                .collect();
            let tri = emit.synth_param(
                &format!("{prefix}.attn.chunk_tri"),
                tri,
                Shape::new(&[s, len], f),
            );
            let one = crate::common::scalar(emit, &format!("{prefix}.attn.one"), 1.0);
            let pen = crate::common::scalar(emit, &format!("{prefix}.attn.neg"), MASK_PENALTY);
            let mut gb = HirMut::new(emit.hir());
            // The validity row covers the prefix; the new tokens are all real.
            let inv = gb.sub(one, c.mask);
            let key_pen = gb.mul(inv, pen); // [1, len]
            let b = gb.add(tri, key_pen);
            let b = gb.reshape_(b, vec![1, 1, si, len as i64]);
            Some(gb.expand_(b, vec![1, nh as i64, si, len as i64]))
        }
        None => None,
    };

    let x2d = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(x, vec![si, d.hidden as i64])
    };

    let q_gate = linear(emit, &format!("{prefix}.attn_q.weight"), x2d)?;
    let (q_packed, gate_packed) = {
        let mut gb = HirMut::new(emit.hir());
        // `[s, n_head, 2·head_dim]`: query first, gate second, per head.
        let qg = gb.reshape_(q_gate, vec![1, si, nh as i64, 2 * hd as i64]);
        let q = gb.narrow_(qg, 3, 0, hd);
        let g = gb.narrow_(qg, 3, hd, hd);
        (
            gb.reshape_(q, vec![1, si, q_dim as i64]),
            gb.reshape_(g, vec![1, si, q_dim as i64]),
        )
    };

    let k_proj = linear(emit, &format!("{prefix}.attn_k.weight"), x2d)?;
    let v_proj = linear(emit, &format!("{prefix}.attn_v.weight"), x2d)?;
    let (k_packed, v_packed) = {
        let mut gb = HirMut::new(emit.hir());
        (
            gb.reshape_(k_proj, vec![1, si, (nkv * hd) as i64]),
            gb.reshape_(v_proj, vec![1, si, (nkv * hd) as i64]),
        )
    };

    let q_normed = per_head_rms(
        emit,
        &format!("{prefix}.attn_q_norm"),
        q_packed,
        nh,
        hd,
        d.eps,
        d.norm_offset,
    )?;
    let k_normed = per_head_rms(
        emit,
        &format!("{prefix}.attn_k_norm"),
        k_packed,
        nkv,
        hd,
        d.eps,
        d.norm_offset,
    )?;

    let mut gb = HirMut::new(emit.hir());
    let q_rot = gb.rope_n(q_normed, cos, sin, hd, d.n_rot);
    let k_rot = gb.rope_n(k_normed, cos, sin, hd, d.n_rot);

    // With a cache, attend over `[prefix | these tokens]`.
    let (k_seq, v_seq, cache_out) = match cache {
        Some(c) => (
            gb.concat_(vec![c.k, k_rot], 1),
            gb.concat_(vec![c.v, v_packed], 1),
            Some(AttnCacheOut {
                k: k_rot,
                v: v_packed,
            }),
        ),
        None => (k_rot, v_packed, None),
    };

    let factor = nh / nkv.max(1);
    let k_full = repeat_kv(&mut gb, k_seq, nkv, hd, factor);
    let v_full = repeat_kv(&mut gb, v_seq, nkv, hd, factor);

    let attn_shape = Shape::new(&[1, s, q_dim], f);
    let y = match (bias, cache, cached_bias) {
        // A chunk of several tokens must be causal *among itself* as well as
        // masked over the prefix, which a per-key `[1, key_len]` row cannot
        // express — so chunk > 1 gets a real `[1, nh, chunk, key_len]` bias.
        (_, Some(_), Some(b)) => gb.attention_bias(q_rot, k_full, v_full, b, nh, hd, attn_shape),
        // One query is causal over everything already stored, so the per-key
        // validity row is the whole mask.
        (_, Some(c), None) => gb.attention(q_rot, k_full, v_full, c.mask, nh, hd, attn_shape),
        (Some(bias), None, _) => gb.attention_bias(q_rot, k_full, v_full, bias, nh, hd, attn_shape),
        (None, None, _) => gb.add_node(
            Op::Attention {
                num_heads: nh,
                head_dim: hd,
                v_head_dim: None,
                mask_kind: MaskKind::Causal,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q_rot, k_full, v_full],
            attn_shape,
        ),
    };

    let g = sigmoid(&mut gb, gate_packed);
    let gated = gb.mul(y, g);
    let gated2d = gb.reshape_(gated, vec![si, q_dim as i64]);
    let _ = gb;

    let out = linear(emit, &format!("{prefix}.attn_output.weight"), gated2d)?;
    let mut gb = HirMut::new(emit.hir());
    Ok((gb.reshape_(out, vec![1, si, d.hidden as i64]), cache_out))
}
