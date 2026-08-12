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

//! DiffusionGemma attention — Gemma 4's, with three properties that make the
//! full-attention layers unusual:
//!
//! * **Per-layer geometry.** Sliding layers are 16 heads × 256 with 8 KV heads;
//!   full-attention layers are 16 heads × 512 with 2 KV heads.
//! * **V aliased to K.** Full-attention layers ship no `v_proj`; V is the *raw*
//!   `k_proj` output — taken before `k_norm`, and never RoPE'd.
//! * **`scaling = 1.0`.** Q/K are RMS-normed per head, so there is no
//!   `1/sqrt(head_dim)` factor. V additionally gets a scale-free RMS norm.
//!
//! The encoder and the diffusion decoder share these weights. The encoder runs
//! causally and taps its post-RoPE K / post-norm V; the decoder re-projects the
//! canvas, concatenates the tapped encoder K/V in front of its own, and attends
//! bidirectionally over the result without writing back.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::{MaskKind, RopeStyle};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

/// Per-layer attention geometry.
#[derive(Debug, Clone, Copy)]
pub struct AttnDims {
    pub hidden: usize,
    pub num_heads: usize,
    /// Per-layer KV heads (8 sliding / 2 full).
    pub num_kv_heads: usize,
    /// Per-layer head width (256 sliding / 512 full).
    pub head_dim: usize,
    /// Full-attention layers alias V to the pre-`k_norm` K projection.
    pub k_eq_v: bool,
    pub eps: f32,
    /// Query length.
    pub seq: usize,
}

impl AttnDims {
    pub fn q_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }
    pub fn group(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }
}

/// Post-RoPE K and post-`v_norm` V, in compact `[1, seq, kv_dim]` layout — what
/// the encoder writes into the cache and the decoder reads back.
#[derive(Debug, Clone, Copy)]
pub struct KvTap {
    pub k: HirNodeId,
    pub v: HirNodeId,
}

fn rms(emit: &mut Emit<'_>, key: &str, x: HirNodeId, dim: usize, eps: f32) -> Result<HirNodeId> {
    let gamma = emit.load_param(&format!("{key}.weight"), false)?;
    let beta = emit.synth_param(
        &format!("{key}.beta"),
        vec![0.0; dim],
        Shape::new(&[dim], DType::F32),
    );
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.rms_norm(x, gamma, beta, eps))
}

fn rms_no_scale(emit: &mut Emit<'_>, tag: &str, x: HirNodeId, dim: usize, eps: f32) -> HirNodeId {
    let ones = emit.synth_param(
        &format!("{tag}.ones"),
        vec![1.0; dim],
        Shape::new(&[dim], DType::F32),
    );
    let zeros = emit.synth_param(
        &format!("{tag}.zeros"),
        vec![0.0; dim],
        Shape::new(&[dim], DType::F32),
    );
    let mut gb = HirMut::new(emit.hir());
    gb.rms_norm(x, ones, zeros, eps)
}

/// Repeat each KV head `group` times along the packed last axis, so the fused
/// attention op sees `num_heads` K/V heads (rlx `Op::Attention` has no GQA arm).
fn repeat_kv(
    gb: &mut HirMut<'_>,
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

/// Q/K/V projections + per-head norms + RoPE, shared by both stacks.
///
/// Returns `(q, KvTap)` with `q` packed `[1, seq, q_dim]` and the tap compact
/// `[1, seq, kv_dim]`.
fn emit_qkv(
    emit: &mut Emit<'_>,
    prefix: &str,
    x: HirNodeId,
    d: AttnDims,
    cos: HirNodeId,
    sin: HirNodeId,
) -> Result<(HirNodeId, KvTap)> {
    let (s, dh, nh, nkv) = (d.seq, d.head_dim, d.num_heads, d.num_kv_heads);
    let (si, dhi) = (s as i64, dh as i64);

    let q_w = emit.load_param(&format!("{prefix}.q_proj.weight"), true)?;
    let k_w = emit.load_param(&format!("{prefix}.k_proj.weight"), true)?;
    let v_w = if d.k_eq_v {
        None
    } else {
        Some(emit.load_param(&format!("{prefix}.v_proj.weight"), true)?)
    };

    let (q, k_raw, v_raw) = {
        let mut gb = HirMut::new(emit.hir());
        let q = gb.mm(x, q_w);
        let k = gb.mm(x, k_w);
        // V aliases the *pre-`k_norm`* K projection on full-attention layers.
        let v = match v_w {
            Some(w) => gb.mm(x, w),
            None => k,
        };
        (q, k, v)
    };

    let q4 = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(q, vec![1, si, nh as i64, dhi])
    };
    let q4 = rms(emit, &format!("{prefix}.q_norm"), q4, dh, d.eps)?;
    let k4 = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(k_raw, vec![1, si, nkv as i64, dhi])
    };
    let k4 = rms(emit, &format!("{prefix}.k_norm"), k4, dh, d.eps)?;
    let v4 = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(v_raw, vec![1, si, nkv as i64, dhi])
    };
    // `v_norm` is scale-free, so it carries no checkpoint weight.
    let v4 = rms_no_scale(emit, &format!("{prefix}.v_norm"), v4, dh, d.eps);

    let mut gb = HirMut::new(emit.hir());
    let q = gb.reshape_(q4, vec![1, si, d.q_dim() as i64]);
    let k = gb.reshape_(k4, vec![1, si, d.kv_dim() as i64]);
    let v = gb.reshape_(v4, vec![1, si, d.kv_dim() as i64]);
    // Full-width NeoX RoPE. Proportional (partial) layers get their identity
    // tail straight from the tables — `inv_freq` is zero there, so cos = 1 and
    // sin = 0 and those channels pass through untouched. Q and K only; V never
    // gets RoPE.
    let q = gb.rope_styled(q, cos, sin, dh, RopeStyle::NeoX);
    let k = gb.rope_styled(k, cos, sin, dh, RopeStyle::NeoX);
    Ok((q, KvTap { k, v }))
}

fn emit_out_proj(
    emit: &mut Emit<'_>,
    prefix: &str,
    attn: HirNodeId,
    _d: AttnDims,
) -> Result<HirNodeId> {
    let o_w = emit.load_param(&format!("{prefix}.o_proj.weight"), true)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.mm(attn, o_w))
}

/// Encoder self-attention: causal over the prompt, windowed on sliding layers.
///
/// Also returns the K/V tap the diffusion decoder consumes.
pub fn emit_encoder_attention(
    emit: &mut Emit<'_>,
    prefix: &str,
    x: HirNodeId,
    d: AttnDims,
    cos: HirNodeId,
    sin: HirNodeId,
    mask: MaskKind,
) -> Result<(HirNodeId, KvTap)> {
    let f = DType::F32;
    let (q, tap) = emit_qkv(emit, prefix, x, d, cos, sin)?;
    let attn = {
        let mut gb = HirMut::new(emit.hir());
        let group = d.group();
        let k_rep = repeat_kv(&mut gb, tap.k, d.num_kv_heads, d.head_dim, group);
        let v_rep = repeat_kv(&mut gb, tap.v, d.num_kv_heads, d.head_dim, group);
        gb.attention_kind_opts(
            q,
            k_rep,
            v_rep,
            d.num_heads,
            d.head_dim,
            mask,
            Shape::new(&[1, d.seq, d.q_dim()], f),
            // Q/K are per-head RMS-normed, so HF sets `scaling = 1.0`.
            Some(1.0),
            None,
        )
    };
    let out = emit_out_proj(emit, prefix, attn, d)?;
    Ok((out, tap))
}

/// Decoder (denoiser) attention: the canvas attends bidirectionally over the
/// read-only encoder K/V prefix followed by its own K/V.
///
/// `enc_k` / `enc_v` are `[1, enc_len, kv_dim]` — already post-RoPE and
/// post-`v_norm`, exactly as the encoder tapped them. Nothing is written back:
/// the encoder cache stays read-only across denoising steps.
#[allow(clippy::too_many_arguments)]
pub fn emit_decoder_attention(
    emit: &mut Emit<'_>,
    prefix: &str,
    x: HirNodeId,
    d: AttnDims,
    cos: HirNodeId,
    sin: HirNodeId,
    enc_k: HirNodeId,
    enc_v: HirNodeId,
    enc_len: usize,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let (q, tap) = emit_qkv(emit, prefix, x, d, cos, sin)?;
    let attn = {
        let mut gb = HirMut::new(emit.hir());
        let k_full = gb.concat_(vec![enc_k, tap.k], 1); // [1, enc_len + seq, kv_dim]
        let v_full = gb.concat_(vec![enc_v, tap.v], 1);
        let group = d.group();
        let k_rep = repeat_kv(&mut gb, k_full, d.num_kv_heads, d.head_dim, group);
        let v_rep = repeat_kv(&mut gb, v_full, d.num_kv_heads, d.head_dim, group);
        // `MaskKind::None`: the denoiser is bidirectional over both the cached
        // prefix and the canvas (`is_causal = False`). Locality on sliding
        // layers comes from the cache itself being windowed, not from a mask —
        // so a cache sliced to the wrong length silently changes what the
        // canvas can see instead of failing.
        debug_assert_eq!(
            gb.shape(k_full).dim(1).unwrap_static(),
            enc_len + d.seq,
            "decoder KV is enc_len + canvas"
        );
        gb.attention_kind_opts(
            q,
            k_rep,
            v_rep,
            d.num_heads,
            d.head_dim,
            MaskKind::None,
            Shape::new(&[1, d.seq, d.q_dim()], f),
            Some(1.0),
            None,
        )
    };
    emit_out_proj(emit, prefix, attn, d)
}
