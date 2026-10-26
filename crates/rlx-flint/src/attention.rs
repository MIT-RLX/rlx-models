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

//! `FlintAttention` — GQA with QK-norm, optional RoPE and learned sinks.
//!
//! ```text
//!   q, k, v = x @ Wqᵀ, x @ Wkᵀ, x @ Wvᵀ
//!   q, k    = RMSNorm_head(q), RMSNorm_head(k)        one [head_dim] gain each
//!   q, k    = RoPE(q), RoPE(k)                        sliding layers only
//!   s       = q·kᵀ · head_dim^-0.5 + bias             [heads, seq, keys]
//!   p       = softmax([s | sink_h])[..keys]           sink: a logit, no value
//!   out     = (p · v) @ Woᵀ
//! ```
//!
//! **Why not `Op::Attention`.** The fused op has no sink input, and a sink is
//! not a mask: it is an extra logit in the softmax denominator per head, so
//! every row's probabilities sum to *less* than one. Dropping it renormalizes
//! the attention and changes every layer's output. The decomposition here —
//! append the sink as one more score column, softmax, drop the column — is the
//! same one `standard_decoder::build_v4_sink_attention` uses.
//!
//! **GQA without a repeat.** Query head `h` reads key/value head `h / group`
//! (`repeat_interleave`, the HF/SGLang order). Rather than materializing the
//! repeated keys, the queries are regrouped to `[kv_heads, group · seq,
//! head_dim]` so one batched matmul per kv head scores all of its query heads.
//!
//! **The mask is an input, not a kind.** `bias` is an additive `[1, 1, seq,
//! keys]` tensor (`0` visible, [`MASK_PENALTY`] hidden) built by
//! [`crate::mask`]. That covers full causal, the 513-token sliding window, a
//! ring-buffered sliding cache and right padding with one code path, and the
//! sink column keeps a fully-masked row finite.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::{HirGraphExt, HirNodeId};

use crate::common::{linear, per_head_rms, scalar};
use crate::config::FlintConfig;

/// Additive penalty for a hidden key. Finite, so a row with every key hidden
/// still has the sink to normalize against instead of producing NaN.
pub const MASK_PENALTY: f32 = -1.0e30;

/// Static shape of one attention layer for a step of `seq` new positions.
#[derive(Debug, Clone, Copy)]
pub struct AttnDims {
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub seq: usize,
    /// Rows of cached keys in front of this step's (`0` = no cache).
    pub cap: usize,
}

impl AttnDims {
    pub fn from_config(cfg: &FlintConfig, seq: usize, cap: usize) -> Self {
        AttnDims {
            hidden: cfg.hidden_size,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            eps: cfg.rms_norm_eps,
            seq,
            cap,
        }
    }

    /// Keys every query is scored against: the cache, then this step.
    pub fn keys(&self) -> usize {
        self.cap + self.seq
    }
}

/// Per-step tensors the attention reads besides the hidden state.
#[derive(Debug, Clone, Copy)]
pub struct AttnInputs {
    /// `(cos, sin)`, each `[seq, head_dim / 2]`; `None` on a no-RoPE layer.
    pub rope: Option<(HirNodeId, HirNodeId)>,
    /// `[1, 1, seq, cap + seq]` additive mask.
    pub bias: HirNodeId,
    /// `(kᵀ, v)` caches, already rotated and stored in the layouts the two
    /// matmuls read: `kᵀ` as `[kv_heads, head_dim, cap]`, `v` as
    /// `[kv_heads, cap, head_dim]`. `None` exactly when `AttnDims::cap == 0`.
    ///
    /// Storing them pre-arranged is what keeps a decode step O(new tokens) in
    /// data movement: concatenating `[cache | new]` and transposing the result
    /// copied the whole cache twice per layer per step, which profiled as most
    /// of a layer's time once the cache was a few hundred rows long.
    pub cache: Option<(HirNodeId, HirNodeId)>,
}

/// What the layer hands back.
#[derive(Debug, Clone, Copy)]
pub struct AttnOut {
    /// `[1, seq, hidden]` — after `o_proj`, before the residual add.
    pub out: HirNodeId,
    /// `[1, seq, kv_dim]` — this step's keys, **rotated**, for the cache.
    pub k: HirNodeId,
    /// `[1, seq, kv_dim]`.
    pub v: HirNodeId,
}

/// Emit `model.layers.{layer}.self_attn` over the normed input `x` `[1, seq, hidden]`.
pub fn emit_attention(
    emit: &mut Emit<'_>,
    layer: usize,
    x: HirNodeId,
    d: AttnDims,
    io: AttnInputs,
) -> Result<AttnOut> {
    if io.cache.is_some() != (d.cap > 0) {
        anyhow::bail!(
            "flint: layer {layer} has cap {} but a cache was {}",
            d.cap,
            if io.cache.is_some() {
                "given"
            } else {
                "not given"
            }
        );
    }
    let p = format!("model.layers.{layer}.self_attn");
    let (s, nh, nkv, hd) = (d.seq as i64, d.n_heads, d.n_kv_heads, d.head_dim);
    let (q_dim, kv_dim) = (nh * hd, nkv * hd);
    let group = nh / nkv;
    let keys = d.keys() as i64;

    let x2d = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(x, vec![s, d.hidden as i64])
    };
    let q = linear(emit, &format!("{p}.q_proj.weight"), x2d)?;
    let k = linear(emit, &format!("{p}.k_proj.weight"), x2d)?;
    let v = linear(emit, &format!("{p}.v_proj.weight"), x2d)?;
    let (q, k, v) = {
        let mut gb = HirMut::new(emit.hir());
        (
            gb.reshape_(q, vec![1, s, q_dim as i64]),
            gb.reshape_(k, vec![1, s, kv_dim as i64]),
            gb.reshape_(v, vec![1, s, kv_dim as i64]),
        )
    };
    let q = per_head_rms(emit, &format!("{p}.q_norm.weight"), q, nh, hd, d.eps)?;
    let k = per_head_rms(emit, &format!("{p}.k_norm.weight"), k, nkv, hd, d.eps)?;

    let sinks = emit.load_param(&format!("{p}.softmax_offset"), false)?;
    let scale = scalar(emit, &format!("{p}.score_scale"), (hd as f32).powf(-0.5));

    let mut gb = HirMut::new(emit.hir());
    let (q, k) = match io.rope {
        Some((cos, sin)) => (
            gb.rope_n(q, cos, sin, hd, hd),
            gb.rope_n(k, cos, sin, hd, hd),
        ),
        None => (q, k),
    };

    // q: [1, s, nh·hd] → [nkv, group·s, hd], head h = kv·group + g.
    let q4 = gb.reshape_(q, vec![s, nkv as i64, group as i64, hd as i64]);
    let q4 = gb.transpose_(q4, vec![1, 2, 0, 3]);
    let q3 = gb.reshape_(q4, vec![nkv as i64, group as i64 * s, hd as i64]);
    // This step's keys → [nkv, hd, s] and values → [nkv, s, hd]: the same
    // layouts the cache is stored in, so the two halves score alike.
    let k3 = gb.reshape_(k, vec![s, nkv as i64, hd as i64]);
    let k_new = gb.transpose_(k3, vec![1, 2, 0]);
    let v3 = gb.reshape_(v, vec![s, nkv as i64, hd as i64]);
    let v_new = gb.transpose_(v3, vec![1, 0, 2]);

    // Scores against [cache | new], concatenated as scores — small — rather
    // than as keys.
    let new_scores = gb.mm(q3, k_new); // [nkv, group·s, s]
    let scores = match io.cache {
        Some((kt, _)) => {
            let cached = gb.mm(q3, kt); // [nkv, group·s, cap]
            gb.concat_(vec![cached, new_scores], 2)
        }
        None => new_scores,
    };
    let scores = gb.mul(scores, scale);
    let scores = gb.reshape_(scores, vec![nkv as i64, group as i64, s, keys]);
    let scores = gb.add(scores, io.bias);

    let sink = gb.reshape_(sinks, vec![nkv as i64, group as i64, 1, 1]);
    let sink = gb.expand_(sink, vec![nkv as i64, group as i64, s, 1]);
    let ext = gb.concat_(vec![scores, sink], 3);
    let probs = gb.sm(ext, -1);
    let probs = gb.narrow_(probs, 3, 0, d.keys());
    let probs = gb.reshape_(probs, vec![nkv as i64, group as i64 * s, keys]);

    let p_new = gb.narrow_(probs, 2, d.cap, d.seq);
    let mut o = gb.mm(p_new, v_new); // [nkv, group·s, hd]
    if let Some((_, vc)) = io.cache {
        let p_cached = gb.narrow_(probs, 2, 0, d.cap);
        let o_cached = gb.mm(p_cached, vc);
        o = gb.add(o_cached, o);
    }
    let o = gb.reshape_(o, vec![nkv as i64, group as i64, s, hd as i64]);
    let o = gb.transpose_(o, vec![2, 0, 1, 3]); // [s, nkv, group, hd]
    let o = gb.reshape_(o, vec![s, q_dim as i64]);
    let _ = gb;

    let out = linear(emit, &format!("{p}.o_proj.weight"), o)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(AttnOut {
        out: gb.reshape_(out, vec![1, s, d.hidden as i64]),
        k,
        v,
    })
}
