// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Kimi-K3 Multi-head Latent Attention (MLA), **NoPE** variant.
//!
//! DeepSeek-style low-rank Q/KV compression, but with `mla_use_nope = true`:
//! there is **no rotary embedding** — the `qk_rope_head_dim` slice is carried
//! through unrotated. A sigmoid **output gate** (`mla_use_output_gate`) scales the
//! attention output before `o_proj`. Value is zero-padded to `qk_head_dim` for the
//! fused attention op, then sliced back to `v_head_dim` (matching HF's flash path).

use crate::common::{linear, reg, sigmoid};
use anyhow::Result;
use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::op::{MaskKind, PadMode};
use rlx_ir::{DType, HirGraphExt, Shape};
use std::collections::HashMap;

type Params = HashMap<String, Vec<f32>>;

#[derive(Debug, Clone, Copy)]
pub struct MlaDims {
    pub hidden: usize,
    pub num_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub eps: f32,
    pub batch: usize,
    pub seq: usize,
}

impl MlaDims {
    pub fn qk(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
}

/// Use the asymmetric-V attention op (`attention_kind_vdim`) instead of
/// zero-padding V up to `qk_head_dim`. Default ON; `RLX_MLA_VDIM=0` falls back
/// to the pad path (for backends whose attention kernel doesn't yet implement
/// `v_head_dim != head_dim`). The vdim path also shrinks the V KV-cache
/// (`v_head_dim` vs `qk_head_dim`).
pub(crate) fn mla_vdim() -> bool {
    std::env::var("RLX_MLA_VDIM")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// Dense MLA weights (`[in, out]` projections; loader transposes HF `[out, in]`).
#[derive(Debug, Clone, Default)]
pub struct MlaWeights {
    pub q_a_proj: Vec<f32>,           // [hidden, q_lora]
    pub q_a_layernorm: Vec<f32>,      // [q_lora]
    pub q_b_proj: Vec<f32>,           // [q_lora, heads*qk]
    pub kv_a_proj_with_mqa: Vec<f32>, // [hidden, kv_lora + rope]
    pub kv_a_layernorm: Vec<f32>,     // [kv_lora]
    pub kv_b_proj: Vec<f32>,          // [kv_lora, heads*(nope+v)]
    pub g_proj: Vec<f32>,             // [hidden, heads*v]  (output gate)
    pub o_proj: Vec<f32>,             // [heads*v, hidden]
}

/// Build one NoPE MLA layer on the (already input-normed) `h_in`
/// `[batch, seq, hidden]`; returns the **raw** attention output (no residual).
/// Shared MLA q/k/v assembly (NoPE) — `x2d [rows,hidden]` → `(qf, kf, vf)` each
/// `[b, s, h·qk]` (value zero-padded to `qk`). Used by prefill AND the KV-cache
/// decode step so both compute keys/values identically.
/// Fuse the two MLA down-projections that read the normed input `x2d`
/// (`q_a_proj`, `kv_a_proj_with_mqa`) into one `[hidden, ql+ckvl]` matmul + narrows
/// — 2 GEMVs → 1. **Bit-exact** (each output column is the same independent
/// dot-product over `hidden`); the per-row weight repack bakes into the (cached)
/// graph, done once. Returns `(q_a [rows,ql], ckv [rows,ckvl])`.
fn fused_down_proj(
    g: &mut HirMut,
    params: &mut Params,
    prefix: &str,
    x2d: HirNodeId,
    w: &MlaWeights,
    hidden: usize,
    ql: usize,
    ckvl: usize,
) -> (HirNodeId, HirNodeId) {
    let parts: [(&[f32], usize); 2] = [(&w.q_a_proj, ql), (&w.kv_a_proj_with_mqa, ckvl)];
    let total = ql + ckvl;
    // Prequant-load: q_a came in empty; the fused int8 codes are mmapped by name in
    // `emit_int8_resident` → skip the f32 assembly, pass an empty placeholder.
    let load = crate::common::prequant_load_active();
    let mut fused_w = if load {
        Vec::new()
    } else {
        vec![0f32; hidden * total]
    };
    let mut offs = [0usize; 2];
    let mut off = 0usize;
    for (i, (wi, out)) in parts.iter().enumerate() {
        offs[i] = off;
        if !load && !wi.is_empty() {
            for r in 0..hidden {
                fused_w[r * total + off..r * total + off + out]
                    .copy_from_slice(&wi[r * out..r * out + out]);
            }
        }
        off += out;
    }
    let fused = linear(
        g,
        params,
        prefix,
        "qkv_a_fused",
        x2d,
        &fused_w,
        hidden,
        total,
    );
    (
        g.narrow_(fused, 1, offs[0], ql),
        g.narrow_(fused, 1, offs[1], ckvl),
    )
}

fn mla_qkv(
    g: &mut HirMut,
    params: &mut Params,
    prefix: &str,
    x2d: HirNodeId,
    w: &MlaWeights,
    d: MlaDims,
) -> (HirNodeId, HirNodeId, HirNodeId) {
    let (b, s, hidden, h) = (d.batch, d.seq, d.hidden, d.num_heads);
    let (ql, kvl, nope, rope, vd) = (
        d.q_lora_rank,
        d.kv_lora_rank,
        d.qk_nope_head_dim,
        d.qk_rope_head_dim,
        d.v_head_dim,
    );
    let qk = d.qk();
    let rows = b * s;
    let zero = |g: &mut HirMut, params: &mut Params, name: &str, n: usize| {
        reg(
            g,
            params,
            &format!("{prefix}.{name}.zero_beta"),
            vec![0f32; n],
            &[n],
        )
    };

    // fuse the two down-projections (q_a, kv_a) that both read x2d into one matmul
    let (q, ckv) = fused_down_proj(g, params, prefix, x2d, w, hidden, ql, kvl + rope);

    // ── Q: q_a → RMSNorm(q_a_layernorm) → q_b_proj → [b,s,h,qk] ──
    let qln = reg(
        g,
        params,
        &format!("{prefix}.q_a_layernorm"),
        w.q_a_layernorm.clone(),
        &[ql],
    );
    let qzb = zero(g, params, "q_a_layernorm", ql);
    let q = g.rms_norm(q, qln, qzb, d.eps);
    let q = linear(g, params, prefix, "q_b_proj", q, &w.q_b_proj, ql, h * qk);
    let q4 = g.reshape_(q, vec![b as i64, s as i64, h as i64, qk as i64]);
    let q_pass = g.narrow_(q4, 3, 0, nope);
    let q_rot = g.narrow_(q4, 3, nope, rope); // NoPE: carried unrotated

    // ── KV: ckv (from the fused proj) → split(k_pass, k_rot) ──
    let ckv3 = g.reshape_(ckv, vec![b as i64, s as i64, (kvl + rope) as i64]);
    let k_pass = g.narrow_(ckv3, 2, 0, kvl); // [b,s,kvl]
    let k_rot = g.narrow_(ckv3, 2, kvl, rope); // [b,s,rope]

    // k_pass → RMSNorm(kv_a_layernorm) → kv_b_proj → [b,s,h,nope+v] → split
    let kpass2d = g.reshape_(k_pass, vec![rows as i64, kvl as i64]);
    let kvln = reg(
        g,
        params,
        &format!("{prefix}.kv_a_layernorm"),
        w.kv_a_layernorm.clone(),
        &[kvl],
    );
    let kvzb = zero(g, params, "kv_a_layernorm", kvl);
    let kpass2d = g.rms_norm(kpass2d, kvln, kvzb, d.eps);
    let kb = linear(
        g,
        params,
        prefix,
        "kv_b_proj",
        kpass2d,
        &w.kv_b_proj,
        kvl,
        h * (nope + vd),
    );
    let kv4 = g.reshape_(kb, vec![b as i64, s as i64, h as i64, (nope + vd) as i64]);
    let k_nope = g.narrow_(kv4, 3, 0, nope);
    let value = g.narrow_(kv4, 3, nope, vd);

    // k_rot shared across heads: [b,s,rope] → [b,s,1,rope] → * ones[1,1,h,1]
    let k_rot4 = g.reshape_(k_rot, vec![b as i64, s as i64, 1, rope as i64]);
    let ones = reg(
        g,
        params,
        &format!("{prefix}.k_rot_ones"),
        vec![1f32; h],
        &[1, 1, h, 1],
    );
    let k_rot_h = g.mul(k_rot4, ones); // [b,s,h,rope]

    // ── assemble query/key [.,h,qk]; value stays v-wide (vdim) or padded ──
    let query = g.concat_(vec![q_pass, q_rot], 3);
    let key = g.concat_(vec![k_nope, k_rot_h], 3);
    let qf = g.reshape_(query, vec![b as i64, s as i64, (h * qk) as i64]);
    let kf = g.reshape_(key, vec![b as i64, s as i64, (h * qk) as i64]);
    let vf = if mla_vdim() {
        // asymmetric V: keep the native v_head_dim width (no pad).
        g.reshape_(value, vec![b as i64, s as i64, (h * vd) as i64])
    } else {
        let v_pad = g.pad_(
            value,
            vec![[0, 0], [0, 0], [0, 0], [0, rope]],
            PadMode::Constant(0.0),
        );
        g.reshape_(v_pad, vec![b as i64, s as i64, (h * qk) as i64])
    };
    (qf, kf, vf)
}

/// Shared MLA post-attention: value slice (`qk`→`v`), sigmoid output gate, o_proj
/// → `[b, s, hidden]`. `attn_raw` is the fused-attention output `[b, s, h·qk]`;
/// `s` is the QUERY length (prefill `seq`, or `s_new` in decode).
fn mla_out(
    g: &mut HirMut,
    params: &mut Params,
    prefix: &str,
    x2d: HirNodeId,
    attn_raw: HirNodeId,
    w: &MlaWeights,
    d: MlaDims,
) -> HirNodeId {
    let (b, s, hidden, h) = (d.batch, d.seq, d.hidden, d.num_heads);
    let (qk, vd) = (d.qk(), d.v_head_dim);
    let rows = b * s;
    let f = DType::F32;
    // vdim: attn_raw is already [b,s,h·v]. pad path: [b,s,h·qk] → slice qk→v.
    let attn = if mla_vdim() {
        g.reshape_(attn_raw, vec![rows as i64, (h * vd) as i64])
    } else {
        let out4 = g.reshape_(attn_raw, vec![b as i64, s as i64, h as i64, qk as i64]);
        let out_v = g.narrow_(out4, 3, 0, vd); // [b,s,h,v]
        g.reshape_(out_v, vec![rows as i64, (h * vd) as i64])
    };
    // ── sigmoid output gate + o_proj ──
    let gate = linear(g, params, prefix, "g_proj", x2d, &w.g_proj, hidden, h * vd);
    let gate = sigmoid(g, gate, Shape::new(&[rows, h * vd], f));
    let attn = g.mul(attn, gate);
    let out = linear(g, params, prefix, "o_proj", attn, &w.o_proj, h * vd, hidden);
    g.reshape_(out, vec![b as i64, s as i64, hidden as i64])
}

pub fn build_mla_layer(
    g: &mut HirMut,
    params: &mut Params,
    prefix: &str,
    h_in: HirNodeId,
    w: &MlaWeights,
    d: MlaDims,
) -> Result<HirNodeId> {
    let (b, s, hidden, h) = (d.batch, d.seq, d.hidden, d.num_heads);
    let qk = d.qk();
    let rows = b * s;
    let vd = d.v_head_dim;
    let x2d = g.reshape_(h_in, vec![rows as i64, hidden as i64]);
    let (qf, kf, vf) = mla_qkv(g, params, prefix, x2d, w, d);
    let attn_raw = if mla_vdim() {
        g.attention_kind_vdim(
            qf,
            kf,
            vf,
            h,
            qk,
            vd,
            MaskKind::Causal,
            Shape::new(&[b, s, h * vd], DType::F32),
        )
    } else {
        g.attention_kind(
            qf,
            kf,
            vf,
            h,
            qk,
            MaskKind::Causal,
            Shape::new(&[b, s, h * qk], DType::F32),
        )
    };
    // Raw attention output — the residual is added by the AttnRes accumulation.
    Ok(mla_out(g, params, prefix, x2d, attn_raw, w, d))
}

/// MLA **decode step** with a KV cache: compute q/k/v for `d.seq` new tokens,
/// CONCAT the cached key/value `[b, s_past, h·qk]`, attend (`q_len < kv_len`,
/// absolute-position causal), and return `(attn_out [b,s_new,hidden], new_k,
/// new_v)` — the grown cache to carry to the next step. Same math as
/// [`build_mla_layer`] when `s_past == 0`; the O(1) decode path for the 24 MLA
/// layers (`d.seq` is the NEW-token count, typically 1).
pub fn build_mla_decode_step(
    g: &mut HirMut,
    params: &mut Params,
    prefix: &str,
    h_in: HirNodeId,
    cached_k: HirNodeId,
    cached_v: HirNodeId,
    w: &MlaWeights,
    d: MlaDims,
) -> Result<(HirNodeId, HirNodeId, HirNodeId)> {
    let (b, s, hidden, h) = (d.batch, d.seq, d.hidden, d.num_heads);
    let (qk, vd) = (d.qk(), d.v_head_dim);
    let rows = b * s;
    let x2d = g.reshape_(h_in, vec![rows as i64, hidden as i64]);
    let (qf, kf, vf) = mla_qkv(g, params, prefix, x2d, w, d);
    // grow the cache: [b, s_past, h·w] ⊕ [b, s_new, h·w] → [b, s_total, h·w]
    // (w = v_head_dim in the vdim path — a smaller V cache — else qk).
    let full_k = g.concat_(vec![cached_k, kf], 1);
    let full_v = g.concat_(vec![cached_v, vf], 1);
    // new queries attend over the whole cache (absolute-position causal decode).
    let attn_raw = if mla_vdim() {
        g.attention_kind_vdim(
            qf,
            full_k,
            full_v,
            h,
            qk,
            vd,
            MaskKind::Causal,
            Shape::new(&[b, s, h * vd], DType::F32),
        )
    } else {
        g.attention_kind(
            qf,
            full_k,
            full_v,
            h,
            qk,
            MaskKind::Causal,
            Shape::new(&[b, s, h * qk], DType::F32),
        )
    };
    let out = mla_out(g, params, prefix, x2d, attn_raw, w, d);
    Ok((out, full_k, full_v))
}
