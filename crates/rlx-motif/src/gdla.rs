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

//! GDLA — Grouped Differential Latent Attention (`MotifGDLAttention`, V2).
//!
//! DeepSeek-style multi-head **latent** attention (low-rank Q and KV, one shared
//! RoPE head) crossed with **differential** attention: heads come in groups of
//! `grouped_ratio + 1`, the last head of each group is a *noise* head, and its
//! output is subtracted from every signal head in the group with an
//! input-dependent λ.
//!
//! ```text
//!   q_lat = RMSNorm(x·Wq_aᵀ)                              [.,q_lora_rank]
//!   q     = q_lat·Wq_bᵀ            → [.,H,  qk_nope+qk_rope]
//!   gate  = q_lat·Wq_b_gateᵀ       → [.,S,  v_head_dim]     (S = signal heads)
//!   ckv   = x·Wkv_aᵀ = [kv_lat | k_pe]
//!   kv    = RMSNorm(kv_lat)·Wkv_bᵀ → [.,KV, qk_nope+v]
//!   q = [q_nope | RoPE(q_pe)];  k = [k_nope | RoPE(k_pe)⇢KV];  v = kv.v
//!   o = softmax(q·kᵀ·scale)·v      → [.,H,v_head_dim]       (GQA: H/KV per group)
//!   o = o[signal] − σ(x·Wλᵀ)·o[noise ⇢ signal]
//!   out = (o · σ(gate))·Woᵀ
//! ```
//!
//! Notes that cost correctness if missed:
//!
//! * The head regroup is `(g gs) → g gs` with `g = num_noise_heads`, i.e. heads
//!   are **contiguous** per group and the noise head is last. That is the same
//!   partition GQA uses (`num_key_value_heads == num_noise_heads`), so KV head
//!   `g` serves exactly group `g`.
//! * `repeat_kv` here is `repeat_interleave`, not tiling: noise head `g` is
//!   broadcast to the `grouped_ratio` signal heads *of its own group*.
//! * V is `v_head_dim` wide but the scores are `head_dim` wide. As in
//!   `rlx_deepseek::mla` we zero-pad V and slice the output back rather than
//!   using asymmetric SDPA, which the CPU interpreter rejects.
//! * Sliding-window layers use their own `swa_rope_theta` table with no YaRN
//!   interpolation, and drop the YaRN `mscale` from the softmax scale.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::{Activation, MaskKind, Op, PadMode};
use rlx_ir::{DType, HirGraphExt, HirNodeId, RopeStyle, Shape};

/// Flow input names for the two RoPE tables.
pub const ROPE_COS: &str = "rope_cos";
pub const ROPE_SIN: &str = "rope_sin";
pub const SWA_ROPE_COS: &str = "swa_rope_cos";
pub const SWA_ROPE_SIN: &str = "swa_rope_sin";

/// Everything one GDLA layer needs that is not a weight.
#[derive(Debug, Clone, Copy)]
pub struct GdlaDims {
    /// Model width — the branch's input and output.
    pub hidden: usize,
    /// Query heads.
    pub num_heads: usize,
    /// `num_key_value_heads`, which GDLA also uses as the group count.
    pub num_kv_heads: usize,
    /// Signal heads per group — `(H − noise) / noise`.
    pub grouped_ratio: usize,
    /// Q/K width per head (`qk_nope + qk_rope`).
    pub head_dim: usize,
    /// Trailing slice of Q/K that carries RoPE.
    pub qk_rope_head_dim: usize,
    /// V width per head; may be narrower than `head_dim`.
    pub v_head_dim: usize,
    /// Rank of the Q bottleneck.
    pub q_lora_rank: usize,
    /// Rank of the KV bottleneck (excludes the shared RoPE head).
    pub kv_lora_rank: usize,
    /// `None` ⇒ full causal; `Some(w)` ⇒ keys `[q − w, q]`.
    pub window: Option<usize>,
    /// `MotifGDLAttention.scaling` for this layer.
    pub score_scale: f32,
    /// RMSNorm epsilon for the Q/KV latent norms.
    pub eps: f32,
    /// Prompt length this graph is built for.
    pub seq: usize,
}

impl GdlaDims {
    pub fn qk_nope_head_dim(&self) -> usize {
        self.head_dim - self.qk_rope_head_dim
    }
    /// Heads that survive the differential subtraction.
    pub fn n_signal_heads(&self) -> usize {
        self.grouped_ratio * self.num_kv_heads
    }
    fn mask(&self) -> MaskKind {
        match self.window {
            Some(w) => MaskKind::SlidingWindow(w),
            None => MaskKind::Causal,
        }
    }
}

fn linear(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
    let w = emit.load_param(&format!("{prefix}.weight"), true)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.mm(x, w))
}

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

/// Emit GDLA for `model.layers.{i}.self_attn` (`prefix`) on `[1, seq, hidden]`.
/// Returns the branch output; the caller owns the residual/MHC combine.
pub fn emit_gdla_attention(
    emit: &mut Emit<'_>,
    prefix: &str,
    hidden: HirNodeId,
    d: GdlaDims,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let (h, kv, s) = (d.num_heads, d.num_kv_heads, d.seq);
    let nope = d.qk_nope_head_dim();
    let rope = d.qk_rope_head_dim;
    let (hd, vd) = (d.head_dim, d.v_head_dim);
    let gs = d.grouped_ratio;
    let sig = d.n_signal_heads();
    let (si, hi, kvi) = (s as i64, h as i64, kv as i64);
    let (ri, hdi, vdi) = (rope as i64, hd as i64, vd as i64);

    // ── Q: low-rank with a normed latent shared by the output gate ──
    let q_lat = linear(emit, &format!("{prefix}.wq_a"), hidden)?;
    let q_lat = rmsnorm(
        emit,
        &format!("{prefix}.q_norm"),
        q_lat,
        d.q_lora_rank,
        d.eps,
    )?;
    let q = linear(emit, &format!("{prefix}.wq_b"), q_lat)?;
    let gate = linear(emit, &format!("{prefix}.wq_b_gate"), q_lat)?;

    // ── KV: compressed latent + one shared RoPE head ──
    let ckv = linear(emit, &format!("{prefix}.wkv_a"), hidden)?;
    let (kv_lat, k_pe) = {
        let mut gb = HirMut::new(emit.hir());
        let lat = gb.narrow_(ckv, 2, 0, d.kv_lora_rank);
        let pe = gb.narrow_(ckv, 2, d.kv_lora_rank, rope);
        (lat, pe)
    };
    let kv_lat = rmsnorm(
        emit,
        &format!("{prefix}.kv_norm"),
        kv_lat,
        d.kv_lora_rank,
        d.eps,
    )?;
    let kv_up = linear(emit, &format!("{prefix}.wkv_b"), kv_lat)?;

    let lambda = linear(emit, &format!("{prefix}.lambda_proj"), hidden)?;

    let (cos_name, sin_name) = if d.window.is_some() {
        (SWA_ROPE_COS, SWA_ROPE_SIN)
    } else {
        (ROPE_COS, ROPE_SIN)
    };
    let cos = emit.flow_input(cos_name)?.hir_id();
    let sin = emit.flow_input(sin_name)?.hir_id();

    let attn = {
        let mut gb = HirMut::new(emit.hir());
        let q4 = gb.reshape_(q, vec![1, si, hi, hdi]);
        let q_nope = gb.narrow_(q4, 3, 0, nope);
        let q_pe = gb.narrow_(q4, 3, nope, rope);
        let kv4 = gb.reshape_(kv_up, vec![1, si, kvi, (nope + vd) as i64]);
        let k_nope = gb.narrow_(kv4, 3, 0, nope);
        let value = gb.narrow_(kv4, 3, nope, vd);

        // Rotate with the heads packed into the last axis — the portable
        // multi-head RoPE path (folding heads onto seq is not lowerable on MLX).
        // Motif's `rotate_half` is the half-split (NeoX) pairing.
        let q_pe = gb.reshape_(q_pe, vec![1, si, hi * ri]);
        let q_pe = gb.rope_styled(q_pe, cos, sin, rope, RopeStyle::NeoX);
        let q_pe = gb.reshape_(q_pe, vec![1, si, hi, ri]);
        let k_pe = gb.rope_styled(k_pe, cos, sin, rope, RopeStyle::NeoX); // [1,s,rope]
        let k_pe = gb.reshape_(k_pe, vec![1, si, 1, ri]);
        let k_pe = gb.expand_(k_pe, vec![1, si, kvi, ri]); // shared across KV heads

        let query = gb.concat_(vec![q_nope, q_pe], 3);
        let key = gb.concat_(vec![k_nope, k_pe], 3);
        // The fused attention op wants matching q/k/v widths; zero-pad V to
        // head_dim and slice back after (what the flash path does).
        let v_pad = gb.pad_(
            value,
            vec![[0, 0], [0, 0], [0, 0], [0, hd - vd]],
            PadMode::Constant(0.0),
        );

        // GQA: repeat_interleave each KV head over its `H / KV` query heads.
        let reps = (h / kv) as i64;
        let key = gb.reshape_(key, vec![1, si, kvi, 1, hdi]);
        let key = gb.expand_(key, vec![1, si, kvi, reps, hdi]);
        let key = gb.reshape_(key, vec![1, si, hi * hdi]);
        let v_pad = gb.reshape_(v_pad, vec![1, si, kvi, 1, hdi]);
        let v_pad = gb.expand_(v_pad, vec![1, si, kvi, reps, hdi]);
        let v_pad = gb.reshape_(v_pad, vec![1, si, hi * hdi]);
        let query = gb.reshape_(query, vec![1, si, hi * hdi]);

        let out = gb.add_node(
            Op::Attention {
                num_heads: h,
                head_dim: hd,
                v_head_dim: None,
                mask_kind: d.mask(),
                score_scale: Some(d.score_scale),
                attn_logit_softcap: None,
            },
            vec![query, key, v_pad],
            Shape::new(&[1, s, h * hd], f),
        );
        let out = gb.reshape_(out, vec![1, si, hi, hdi]);
        gb.narrow_(out, 3, 0, vd) // [1,s,H,v_head_dim]
    };

    // ── Differential recombination ──
    let out = {
        let mut gb = HirMut::new(emit.hir());
        let grouped = gb.reshape_(attn, vec![1, si, kvi, (gs + 1) as i64, vdi]);
        let signal = gb.narrow_(grouped, 3, 0, gs);
        let signal = gb.reshape_(signal, vec![1, si, sig as i64, vdi]);
        let noise = gb.narrow_(grouped, 3, gs, 1);
        let noise = gb.expand_(noise, vec![1, si, kvi, gs as i64, vdi]);
        let noise = gb.reshape_(noise, vec![1, si, sig as i64, vdi]);

        let lam = gb.add_node(
            Op::Activation(Activation::Sigmoid),
            vec![lambda],
            Shape::new(&[1, s, sig], f),
        );
        let lam = gb.reshape_(lam, vec![1, si, sig as i64, 1]);
        let scaled = gb.mul(noise, lam);
        let diff = gb.sub(signal, scaled);

        let g4 = gb.reshape_(gate, vec![1, si, sig as i64, vdi]);
        let g4 = gb.add_node(
            Op::Activation(Activation::Sigmoid),
            vec![g4],
            Shape::new(&[1, s, sig, vd], f),
        );
        let gated = gb.mul(diff, g4);
        gb.reshape_(gated, vec![si, (sig * vd) as i64])
    };

    let out = linear(emit, &format!("{prefix}.wo"), out)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.reshape_(out, vec![1, si, d.hidden as i64]))
}
