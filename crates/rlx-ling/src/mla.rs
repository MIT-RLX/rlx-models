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

//! `BailingMoeV3MultiLatentAttention` — DeepSeek-style MLA with a sigmoid output
//! gate.
//!
//! ```text
//!   q      = q_b( rms(q_a(x)) )                → [.,h,qk_nope+qk_rope]
//!   ckv    = kv_a_with_mqa(x) → [k_lora | k_rot]   (k_rot: one shared head)
//!   kv_up  = kv_b( rms(k_lora) )               → [.,h,qk_nope+v]
//!   q = [q_nope | rope(q_rot)];  k = [k_nope | rope(k_rot)⇢h];  v = kv_up.v
//!   o = softmax(q·kᵀ·qk_head_dim^-0.5)·v
//!   o = o · σ(g_proj(x))                       ← Bailing-specific gate
//!   out = dense(o)
//! ```
//!
//! Differences from [`rlx_deepseek::mla`], which is otherwise the same layer:
//!
//! * a sigmoid **output gate** `g_proj` (`head_wise`: one scalar per head,
//!   broadcast over `v_head_dim`; `element_wise`: one per output element),
//! * the out-projection is named `dense`, not `o_proj`,
//! * the attention branch lives under `…layers.{i}.attention`, not `.self_attn`,
//! * V3 MLA has **no** q/k norm — the `use_qk_norm` config key is vestigial.
//!
//! RoPE is the interleaved (GPT-J) rotation over the `qk_rope_head_dim` slice.
//! The reference `apply_rotary_pos_emb_interleave` de-interleaves into half-split
//! order and leaves it that way; since q and k get the identical permutation and
//! only ever meet in a dot product, that is exactly GPT-J RoPE applied in place.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::{Activation, MaskKind, Op, PadMode};
use rlx_ir::{DType, HirGraphExt, HirNodeId, RopeStyle, Shape};

use crate::config::AttnGate;

pub const ROPE_COS: &str = "rope_cos";
pub const ROPE_SIN: &str = "rope_sin";

#[derive(Debug, Clone, Copy)]
pub struct MlaDims {
    pub hidden: usize,
    pub num_heads: usize,
    /// `None` projects Q directly (`q_proj`); `Some(r)` uses the `q_a`/`q_b` pair.
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub gate: AttnGate,
    pub eps: f32,
    pub seq: usize,
    /// Stored precision for this block's projections.
    pub quant: Quant,
}

impl MlaDims {
    fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
}

use crate::quant::{Quant, linear};

fn rmsnorm(
    emit: &mut Emit<'_>,
    key: &str,
    x: HirNodeId,
    dim: usize,
    eps: f32,
) -> Result<HirNodeId> {
    let g = emit.load_param(&format!("{key}.weight"), false)?;
    let zb = emit.synth_param(
        &format!("{key}.zb"),
        vec![0.0; dim],
        Shape::new(&[dim], DType::F32),
    );
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.rms_norm(x, g, zb, eps))
}

/// Emit MLA for `model.layers.{i}.attention` (`prefix`) on `[1, seq, hidden]`.
/// Returns the raw branch output — the caller owns the residual add.
pub fn emit_mla_attention(
    emit: &mut Emit<'_>,
    prefix: &str,
    hidden: HirNodeId,
    d: MlaDims,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let h = d.num_heads;
    let s = d.seq;
    let nope = d.qk_nope_head_dim;
    let rope = d.qk_rope_head_dim;
    let qk = d.qk_head_dim();
    let vd = d.v_head_dim;
    let (si, hi, ri, qki, vi) = (s as i64, h as i64, rope as i64, qk as i64, vd as i64);

    // ── Q: low-rank (q_a → rms → q_b) or direct ──
    let q = match d.q_lora_rank {
        Some(rank) => {
            let q_a = linear(emit, &format!("{prefix}.q_a_proj"), hidden, d.quant)?;
            let q_a = rmsnorm(emit, &format!("{prefix}.q_a_layernorm"), q_a, rank, d.eps)?;
            linear(emit, &format!("{prefix}.q_b_proj"), q_a, d.quant)?
        }
        None => linear(emit, &format!("{prefix}.q_proj"), hidden, d.quant)?,
    };

    // ── KV: compressed latent + one shared RoPE head ──
    let ckv = linear(
        emit,
        &format!("{prefix}.kv_a_proj_with_mqa"),
        hidden,
        d.quant,
    )?;
    let (k_lora, k_rot) = {
        let mut gb = HirMut::new(emit.hir());
        let kl = gb.narrow_(ckv, 2, 0, d.kv_lora_rank);
        let kr = gb.narrow_(ckv, 2, d.kv_lora_rank, rope);
        (kl, kr)
    };
    let k_lora = rmsnorm(
        emit,
        &format!("{prefix}.kv_a_layernorm"),
        k_lora,
        d.kv_lora_rank,
        d.eps,
    )?;
    let kv_up = linear(emit, &format!("{prefix}.kv_b_proj"), k_lora, d.quant)?;

    let cos = emit.flow_input(ROPE_COS)?.hir_id();
    let sin = emit.flow_input(ROPE_SIN)?.hir_id();
    let ones = emit.synth_param(
        &format!("{prefix}.kexp"),
        vec![1.0; h],
        Shape::new(&[1, 1, h, 1], f),
    );

    let attn = {
        let mut gb = HirMut::new(emit.hir());
        let q4 = gb.reshape_(q, vec![1, si, hi, qki]);
        let q_nope = gb.narrow_(q4, 3, 0, nope);
        let q_rot = gb.narrow_(q4, 3, nope, rope);
        let kv4 = gb.reshape_(kv_up, vec![1, si, hi, (nope + vd) as i64]);
        let k_nope = gb.narrow_(kv4, 3, 0, nope);
        let value = gb.narrow_(kv4, 3, nope, vd);

        // Rotate with heads packed in the last axis ([1,s,h*rope]) — the portable
        // multi-head RoPE path; folding heads onto the seq axis is not lowerable
        // on MLX (cos[seq] can't broadcast over packed heads).
        let q_rot_3d = gb.reshape_(q_rot, vec![1, si, hi * ri]);
        let q_rot_3d = gb.rope_styled(q_rot_3d, cos, sin, rope, RopeStyle::GptJ);
        let q_rot = gb.reshape_(q_rot_3d, vec![1, si, hi, ri]);
        let k_rot = gb.rope_styled(k_rot, cos, sin, rope, RopeStyle::GptJ); // [1,s,rope]
        let k_rot = gb.reshape_(k_rot, vec![1, si, 1, ri]);
        let k_rot = gb.mul(k_rot, ones); // broadcast the shared head → [1,s,h,rope]

        let query = gb.concat_(vec![q_nope, q_rot], 3);
        let key = gb.concat_(vec![k_nope, k_rot], 3);
        // The fused attention op wants matching q/k/v widths; pad v to qk and slice
        // back after (what HF's flash path does).
        let v_pad = gb.pad_(
            value,
            vec![[0, 0], [0, 0], [0, 0], [0, qk - vd]],
            PadMode::Constant(0.0),
        );

        let qf = gb.reshape_(query, vec![1, si, hi * qki]);
        let kf = gb.reshape_(key, vec![1, si, hi * qki]);
        let vf = gb.reshape_(v_pad, vec![1, si, hi * qki]);

        // Default score scale is qk_head_dim^-0.5, which is exactly
        // `BailingMoeV3MultiLatentAttention.scaling` (no rope_scaling ⇒ no mscale).
        let out = gb.attention_kind(
            qf,
            kf,
            vf,
            h,
            qk,
            MaskKind::Causal,
            Shape::new(&[1, s, h * qk], f),
        );
        let out4 = gb.reshape_(out, vec![1, si, hi, qki]);
        gb.narrow_(out4, 3, 0, vd) // [1,s,h,vd]
    };

    // ── Bailing output gate ──
    let attn = match d.gate {
        AttnGate::None => attn,
        AttnGate::HeadWise | AttnGate::ElementWise => {
            let g = linear(emit, &format!("{prefix}.g_proj"), hidden, d.quant)?;
            let mut gb = HirMut::new(emit.hir());
            let g4 = match d.gate {
                // [1,s,h] → [1,s,h,1], broadcast across v_head_dim
                AttnGate::HeadWise => gb.reshape_(g, vec![1, si, hi, 1]),
                _ => gb.reshape_(g, vec![1, si, hi, vi]),
            };
            let gshape = match d.gate {
                AttnGate::HeadWise => Shape::new(&[1, s, h, 1], f),
                _ => Shape::new(&[1, s, h, vd], f),
            };
            let sig = gb.add_node(Op::Activation(Activation::Sigmoid), vec![g4], gshape);
            gb.mul(attn, sig)
        }
    };

    let flat = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(attn, vec![si, hi * vi])
    };
    let out = linear(emit, &format!("{prefix}.dense"), flat, d.quant)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.reshape_(out, vec![1, si, d.hidden as i64]))
}

/// KV cache for [`emit_mla_decode`], all host-maintained graph inputs.
///
/// The cache holds `cap` slots; the harness writes each step's returned
/// `k_new`/`v_new` into slot `pos` and flips `mask[pos]` on. The current token's
/// own key/value are appended *after* the cache inside the graph, so the mask is
/// `cap + 1` wide with the last entry always valid.
///
/// Appending at the end rather than at `pos` is safe because attention is
/// permutation-invariant over keys once the mask is applied — and RoPE has
/// already baked position into `k_new`. That avoids needing a scatter.
#[derive(Debug, Clone, Copy)]
pub struct MlaCache {
    /// `[1, cap, num_heads * qk_head_dim]`
    pub k: HirNodeId,
    /// `[1, cap, num_heads * v_head_dim]`
    pub v: HirNodeId,
    /// `[1, cap + 1]` — `1.0` valid, `0.0` ignored.
    pub mask: HirNodeId,
    pub cap: usize,
}

/// Emit one MLA **decode step** for a single new token.
///
/// Same arithmetic as [`emit_mla_attention`]; the difference is that keys and
/// values come from the cache plus this token instead of the whole prompt, and
/// the causal mask is replaced by the explicit validity mask (`MaskKind::Custom`)
/// — with one query there is nothing left to mask causally.
///
/// Returns `(out [1,1,hidden], k_new [1,1,h*qk], v_new [1,1,h*vd])`; the caller
/// stores `k_new`/`v_new` into the cache at the current position.
pub fn emit_mla_decode(
    emit: &mut Emit<'_>,
    prefix: &str,
    hidden: HirNodeId,
    cache: MlaCache,
    d: MlaDims,
) -> Result<(HirNodeId, HirNodeId, HirNodeId)> {
    let f = DType::F32;
    let h = d.num_heads;
    let nope = d.qk_nope_head_dim;
    let rope = d.qk_rope_head_dim;
    let qk = d.qk_head_dim();
    let vd = d.v_head_dim;
    let cap = cache.cap;
    let (hi, ri, qki, vi) = (h as i64, rope as i64, qk as i64, vd as i64);
    debug_assert_eq!(d.seq, 1, "emit_mla_decode processes exactly one token");

    // ── Q / KV projections for the new token (identical to prefill) ──
    let q = match d.q_lora_rank {
        Some(rank) => {
            let q_a = linear(emit, &format!("{prefix}.q_a_proj"), hidden, d.quant)?;
            let q_a = rmsnorm(emit, &format!("{prefix}.q_a_layernorm"), q_a, rank, d.eps)?;
            linear(emit, &format!("{prefix}.q_b_proj"), q_a, d.quant)?
        }
        None => linear(emit, &format!("{prefix}.q_proj"), hidden, d.quant)?,
    };
    let ckv = linear(
        emit,
        &format!("{prefix}.kv_a_proj_with_mqa"),
        hidden,
        d.quant,
    )?;
    let (k_lora, k_rot) = {
        let mut gb = HirMut::new(emit.hir());
        let kl = gb.narrow_(ckv, 2, 0, d.kv_lora_rank);
        let kr = gb.narrow_(ckv, 2, d.kv_lora_rank, rope);
        (kl, kr)
    };
    let k_lora = rmsnorm(
        emit,
        &format!("{prefix}.kv_a_layernorm"),
        k_lora,
        d.kv_lora_rank,
        d.eps,
    )?;
    let kv_up = linear(emit, &format!("{prefix}.kv_b_proj"), k_lora, d.quant)?;

    let cos = emit.flow_input(ROPE_COS)?.hir_id();
    let sin = emit.flow_input(ROPE_SIN)?.hir_id();
    let ones = emit.synth_param(
        &format!("{prefix}.dec.kexp"),
        vec![1.0; h],
        Shape::new(&[1, 1, h, 1], f),
    );

    let (k_new, v_new, query) = {
        let mut gb = HirMut::new(emit.hir());
        let q4 = gb.reshape_(q, vec![1, 1, hi, qki]);
        let q_nope = gb.narrow_(q4, 3, 0, nope);
        let q_rot = gb.narrow_(q4, 3, nope, rope);
        let kv4 = gb.reshape_(kv_up, vec![1, 1, hi, (nope + vd) as i64]);
        let k_nope = gb.narrow_(kv4, 3, 0, nope);
        let value = gb.narrow_(kv4, 3, nope, vd);

        let q_rot_3d = gb.reshape_(q_rot, vec![1, 1, hi * ri]);
        let q_rot_3d = gb.rope_styled(q_rot_3d, cos, sin, rope, RopeStyle::GptJ);
        let q_rot = gb.reshape_(q_rot_3d, vec![1, 1, hi, ri]);
        let k_rot = gb.rope_styled(k_rot, cos, sin, rope, RopeStyle::GptJ);
        let k_rot = gb.reshape_(k_rot, vec![1, 1, 1, ri]);
        let k_rot = gb.mul(k_rot, ones);

        let query = gb.concat_(vec![q_nope, q_rot], 3);
        let key = gb.concat_(vec![k_nope, k_rot], 3);
        let k_new = gb.reshape_(key, vec![1, 1, hi * qki]);
        let v_new = gb.reshape_(value, vec![1, 1, hi * vi]);
        let query = gb.reshape_(query, vec![1, 1, hi * qki]);
        (k_new, v_new, query)
    };

    // ── attend over [cache | this token] under the validity mask ──
    let attn = {
        let mut gb = HirMut::new(emit.hir());
        let k_all = gb.concat_(vec![cache.k, k_new], 1); // [1, cap+1, h*qk]
        let v_all = gb.concat_(vec![cache.v, v_new], 1); // [1, cap+1, h*vd]
        // The fused attention op wants matching q/k/v widths, so pad v per head
        // to qk and slice back after — same trick as prefill.
        let v4 = gb.reshape_(v_all, vec![1, (cap + 1) as i64, hi, vi]);
        let v_pad = gb.pad_(
            v4,
            vec![[0, 0], [0, 0], [0, 0], [0, qk - vd]],
            PadMode::Constant(0.0),
        );
        let v_pad = gb.reshape_(v_pad, vec![1, (cap + 1) as i64, hi * qki]);
        let out = gb.attention(
            query,
            k_all,
            v_pad,
            cache.mask,
            h,
            qk,
            Shape::new(&[1, 1, h * qk], f),
        );
        let out4 = gb.reshape_(out, vec![1, 1, hi, qki]);
        gb.narrow_(out4, 3, 0, vd)
    };

    // ── output gate + dense (identical to prefill) ──
    let attn = match d.gate {
        AttnGate::None => attn,
        AttnGate::HeadWise | AttnGate::ElementWise => {
            let g = linear(emit, &format!("{prefix}.g_proj"), hidden, d.quant)?;
            let mut gb = HirMut::new(emit.hir());
            let g4 = match d.gate {
                AttnGate::HeadWise => gb.reshape_(g, vec![1, 1, hi, 1]),
                _ => gb.reshape_(g, vec![1, 1, hi, vi]),
            };
            let gshape = match d.gate {
                AttnGate::HeadWise => Shape::new(&[1, 1, h, 1], f),
                _ => Shape::new(&[1, 1, h, vd], f),
            };
            let sig = gb.add_node(Op::Activation(Activation::Sigmoid), vec![g4], gshape);
            gb.mul(attn, sig)
        }
    };
    let flat = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(attn, vec![1, hi * vi])
    };
    let out = linear(emit, &format!("{prefix}.dense"), flat, d.quant)?;
    let mut gb = HirMut::new(emit.hir());
    let out = gb.reshape_(out, vec![1, 1, d.hidden as i64]);
    Ok((out, k_new, v_new))
}
