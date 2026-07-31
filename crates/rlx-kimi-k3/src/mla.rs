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
pub fn build_mla_layer(
    g: &mut HirMut,
    params: &mut Params,
    prefix: &str,
    h_in: HirNodeId,
    w: &MlaWeights,
    d: MlaDims,
) -> Result<HirNodeId> {
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
    let f = DType::F32;
    let zero = |g: &mut HirMut, params: &mut Params, name: &str, n: usize| {
        reg(
            g,
            params,
            &format!("{prefix}.{name}.zero_beta"),
            vec![0f32; n],
            &[n],
        )
    };

    let x2d = g.reshape_(h_in, vec![rows as i64, hidden as i64]);

    // ── Q: q_a_proj → RMSNorm(q_a_layernorm) → q_b_proj → [b,s,h,qk] ──
    let q = linear(g, params, prefix, "q_a_proj", x2d, &w.q_a_proj, hidden, ql);
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

    // ── KV: kv_a_proj_with_mqa → split(k_pass, k_rot) ──
    let ckv = linear(
        g,
        params,
        prefix,
        "kv_a_proj_with_mqa",
        x2d,
        &w.kv_a_proj_with_mqa,
        hidden,
        kvl + rope,
    );
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

    // ── assemble query/key [.,h,qk]; pad value to qk; fused causal attention ──
    let query = g.concat_(vec![q_pass, q_rot], 3);
    let key = g.concat_(vec![k_nope, k_rot_h], 3);
    let v_pad = g.pad_(
        value,
        vec![[0, 0], [0, 0], [0, 0], [0, rope]],
        PadMode::Constant(0.0),
    );
    let qf = g.reshape_(query, vec![b as i64, s as i64, (h * qk) as i64]);
    let kf = g.reshape_(key, vec![b as i64, s as i64, (h * qk) as i64]);
    let vf = g.reshape_(v_pad, vec![b as i64, s as i64, (h * qk) as i64]);
    let out = g.attention_kind(
        qf,
        kf,
        vf,
        h,
        qk,
        MaskKind::Causal,
        Shape::new(&[b, s, h * qk], f),
    );
    let out4 = g.reshape_(out, vec![b as i64, s as i64, h as i64, qk as i64]);
    let out_v = g.narrow_(out4, 3, 0, vd); // [b,s,h,v]
    let attn = g.reshape_(out_v, vec![rows as i64, (h * vd) as i64]);

    // ── sigmoid output gate + o_proj + residual ──
    let gate = linear(g, params, prefix, "g_proj", x2d, &w.g_proj, hidden, h * vd);
    let gate = sigmoid(g, gate, Shape::new(&[rows, h * vd], f));
    let attn = g.mul(attn, gate);
    let out = linear(g, params, prefix, "o_proj", attn, &w.o_proj, h * vd, hidden);
    // Raw attention output — the residual is added by the AttnRes accumulation.
    Ok(g.reshape_(out, vec![b as i64, s as i64, hidden as i64]))
}
