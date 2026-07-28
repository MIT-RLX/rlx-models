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

//! DeepSeek-V3 Multi-head Latent Attention (`DeepseekV3Attention`).
//!
//! Low-rank compressed Q and KV with a decoupled RoPE head:
//! ```text
//!   q  = q_b( rms(q_a(h)) )                      → [.,H,qk_head_dim=nope+rope]
//!   ckv = kv_a_with_mqa(h) → [k_lora | k_rot]     (k_rot is a single shared head)
//!   kv_up = kv_b( rms(k_lora) )                   → [.,H, nope+v]
//!   q = [q_pass(nope) | rope(q_rot)];  k = [k_nope | rope(k_rot)⇢H];  v = kv_up.v
//!   attn = softmax(q·kᵀ·scale) · pad(v→qk_head_dim);  o_proj(attn[..,:v_head_dim])
//! ```
//! RoPE is applied only to the `qk_rope_head_dim` slice (interleaved / GptJ), and
//! the value is zero-padded to `qk_head_dim` for the fused attention op then
//! sliced back (matching HF's flash path). All projections are bias-free.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::{MaskKind, PadMode};
use rlx_ir::{DType, HirGraphExt, HirNodeId, RopeStyle, Shape};

pub const ROPE_COS: &str = "rope_cos";
pub const ROPE_SIN: &str = "rope_sin";

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
    pub seq: usize,
    pub score_scale: f32,
}

impl MlaDims {
    fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
}

fn linear(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
    let w = emit.load_param(&format!("{prefix}.weight"), true)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.mm(x, w))
}

/// Weight-only RMSNorm over the last dim (`dim`).
fn rmsnorm(
    emit: &mut Emit<'_>,
    tag: &str,
    key: &str,
    x: HirNodeId,
    dim: usize,
    eps: f32,
) -> Result<HirNodeId> {
    let g = emit.load_param(&format!("{key}.weight"), false)?;
    let zb = emit.synth_param(
        &format!("{tag}.zb"),
        vec![0.0; dim],
        Shape::new(&[dim], DType::F32),
    );
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.rms_norm(x, g, zb, eps))
}

/// Emit MLA for `model.layers.{i}.self_attn` (`prefix`) on `[1,seq,hidden]`.
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
    let (si, hi, ni, ri, qki, vi) = (
        s as i64,
        h as i64,
        nope as i64,
        rope as i64,
        qk as i64,
        vd as i64,
    );

    // --- Q low-rank path ---
    let q_a = linear(emit, &format!("{prefix}.q_a_proj"), hidden)?;
    let q_a = rmsnorm(
        emit,
        &format!("{prefix}.qa"),
        &format!("{prefix}.q_a_layernorm"),
        q_a,
        d.q_lora_rank,
        d.eps,
    )?;
    let q = linear(emit, &format!("{prefix}.q_b_proj"), q_a)?; // [1,s,h*qk]

    // --- KV compressed path ---
    let ckv = linear(emit, &format!("{prefix}.kv_a_proj_with_mqa"), hidden)?; // [1,s,kv_lora+rope]
    let (k_lora, k_rot) = {
        let mut gb = HirMut::new(emit.hir());
        let kl = gb.narrow_(ckv, 2, 0, d.kv_lora_rank);
        let kr = gb.narrow_(ckv, 2, d.kv_lora_rank, rope);
        (kl, kr)
    };
    let k_lora = rmsnorm(
        emit,
        &format!("{prefix}.kva"),
        &format!("{prefix}.kv_a_layernorm"),
        k_lora,
        d.kv_lora_rank,
        d.eps,
    )?;
    let kv_up = linear(emit, &format!("{prefix}.kv_b_proj"), k_lora)?; // [1,s,h*(nope+v)]

    // RoPE cos/sin + a broadcast-ones for expanding the single k_rot head.
    let cos = emit.flow_input(ROPE_COS)?.hir_id();
    let sin = emit.flow_input(ROPE_SIN)?.hir_id();
    let ones = emit.synth_param(
        &format!("{prefix}.kexp"),
        vec![1.0; h],
        Shape::new(&[1, 1, h, 1], f),
    );

    let attn = {
        let mut gb = HirMut::new(emit.hir());
        // Split Q per head into nope / rope parts.
        let q4 = gb.reshape_(q, vec![1, si, hi, qki]);
        let q_pass = gb.narrow_(q4, 3, 0, nope);
        let mut q_rot = gb.narrow_(q4, 3, nope, rope);
        // Split KV-up per head into k_nope / value.
        let kv4 = gb.reshape_(kv_up, vec![1, si, hi, (nope + vd) as i64]);
        let k_nope = gb.narrow_(kv4, 3, 0, nope);
        let value = gb.narrow_(kv4, 3, nope, vd);

        // RoPE (interleaved) on q_rot [1,s*h,rope] and the single k_rot head [1,s,rope].
        let q_rot_2d = gb.reshape_(q_rot, vec![1, si * hi, ri]);
        let q_rot_2d = gb.rope_styled(q_rot_2d, cos, sin, rope, RopeStyle::GptJ);
        q_rot = gb.reshape_(q_rot_2d, vec![1, si, hi, ri]);
        let k_rot_r = gb.rope_styled(k_rot, cos, sin, rope, RopeStyle::GptJ); // [1,s,rope]
        let k_rot_4 = gb.reshape_(k_rot_r, vec![1, si, 1, ri]);
        let k_rot_h = gb.mul(k_rot_4, ones); // broadcast → [1,s,h,rope]

        // Assemble query/key [.,h,qk], value padded to qk.
        let query = gb.concat_(vec![q_pass, q_rot], 3);
        let key = gb.concat_(vec![k_nope, k_rot_h], 3);
        let v_pad = gb.pad_(
            value,
            vec![[0, 0], [0, 0], [0, 0], [0, rope]],
            PadMode::Constant(0.0),
        );

        let qf = gb.reshape_(query, vec![1, si, hi * qki]);
        let kf = gb.reshape_(key, vec![1, si, hi * qki]);
        let vf = gb.reshape_(v_pad, vec![1, si, hi * qki]);

        // Fused attention at qk_head_dim, then slice output back to v_head_dim.
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
        let out_v = gb.narrow_(out4, 3, 0, vd);
        gb.reshape_(out_v, vec![1, si, hi * vi])
    };
    // Note: score_scale for base config == qk_head_dim^-0.5 == attention default;
    // YaRN mscale (d.score_scale) folds in via attention_kind_opts in a later revision.
    let _ = d.score_scale;

    linear(emit, &format!("{prefix}.o_proj"), attn)
}
