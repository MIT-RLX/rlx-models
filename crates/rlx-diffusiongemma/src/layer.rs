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

//! One DiffusionGemma transformer layer.
//!
//! Structurally this is `Gemma4TextDecoderLayer` with the MoE block enabled and
//! the per-layer-input (PLE) path absent. The FFN half is a *two-branch* block,
//! which is easy to get subtly wrong:
//!
//! ```text
//! residual ─┬─ pre_feedforward_layernorm  → mlp     → post_feedforward_layernorm_1 ─┐
//!           │                                                                      (+)
//!           ├─ pre_feedforward_layernorm_2 → experts → post_feedforward_layernorm_2 ─┘
//!           │                                   ↑                    │
//!           └───────────── router (unnormed!) ──┘                    ↓
//!                                                    post_feedforward_layernorm → + residual
//! ```
//!
//! Both branches read the *same* post-attention residual, but the router scores
//! the raw residual while the experts consume the `_2`-normed copy. The result
//! is scaled by the per-layer `layer_scalar`, which is the one weight the
//! encoder and decoder stacks do **not** share.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::attention::{AttnDims, KvTap, emit_decoder_attention, emit_encoder_attention};
use crate::moe::{MoeDims, emit_moe};

/// Everything one layer needs that isn't derivable from the layer prefix.
#[derive(Debug, Clone)]
pub struct LayerDims {
    pub attn: AttnDims,
    pub moe: MoeDims,
    /// Shared-expert (`mlp`) width — `intermediate_size` (2112), distinct from
    /// [`MoeDims::moe_inter`] (704), which sizes each *routed* expert.
    pub intermediate: usize,
    pub hidden: usize,
    pub eps: f32,
    pub seq: usize,
    /// Where `layer_scalar` is read from. The encoder stack points at
    /// `model.encoder.language_model.layers.{i}.layer_scalar`, the decoder at
    /// `model.decoder.layers.{i}.layer_scalar`; every other weight is shared.
    pub layer_scalar_key: String,
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

/// Gated FFN with `gelu_pytorch_tanh`: `down(gelu_tanh(gate(x)) · up(x))`.
pub(crate) fn emit_gated_mlp(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
    let gate_w = emit.load_param(&format!("{prefix}.gate_proj.weight"), true)?;
    let up_w = emit.load_param(&format!("{prefix}.up_proj.weight"), true)?;
    let down_w = emit.load_param(&format!("{prefix}.down_proj.weight"), true)?;
    let mut gb = HirMut::new(emit.hir());
    let gate = gb.mm(x, gate_w);
    let up = gb.mm(x, up_w);
    let act = gb.gelu_approx(gate);
    let hx = gb.mul(act, up);
    Ok(gb.mm(hx, down_w))
}

/// The FFN half: shared `mlp` branch + routed-expert branch, combined and
/// added back to `residual`.
fn emit_ffn_block(
    emit: &mut Emit<'_>,
    prefix: &str,
    residual: HirNodeId,
    d: &LayerDims,
) -> Result<HirNodeId> {
    let hidden = d.hidden;

    // Branch 1 — always-on shared expert.
    let normed = rms(
        emit,
        &format!("{prefix}.pre_feedforward_layernorm"),
        residual,
        hidden,
        d.eps,
    )?;
    let mlp_out = emit_gated_mlp(emit, &format!("{prefix}.mlp"), normed)?;
    debug_assert_eq!(
        HirMut::new(emit.hir())
            .shape(mlp_out)
            .dim(2)
            .unwrap_static(),
        hidden,
        "the shared expert projects back to hidden"
    );
    let branch1 = rms(
        emit,
        &format!("{prefix}.post_feedforward_layernorm_1"),
        mlp_out,
        hidden,
        d.eps,
    )?;

    // Branch 2 — routed experts. The router reads the *unnormalized* residual;
    // only the expert input goes through `pre_feedforward_layernorm_2`.
    let flat = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(residual, vec![d.moe.rows as i64, hidden as i64])
    };
    let expert_in = rms(
        emit,
        &format!("{prefix}.pre_feedforward_layernorm_2"),
        flat,
        hidden,
        d.eps,
    )?;
    let moe_out = emit_moe(emit, prefix, flat, expert_in, d.moe)?;
    let moe_out = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(moe_out, vec![1, d.seq as i64, hidden as i64])
    };
    let branch2 = rms(
        emit,
        &format!("{prefix}.post_feedforward_layernorm_2"),
        moe_out,
        hidden,
        d.eps,
    )?;

    let combined = {
        let mut gb = HirMut::new(emit.hir());
        gb.add(branch1, branch2)
    };
    let out = rms(
        emit,
        &format!("{prefix}.post_feedforward_layernorm"),
        combined,
        hidden,
        d.eps,
    )?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.add(residual, out))
}

/// `hidden *= layer_scalar` — a single scalar buffer per layer per stack.
fn apply_layer_scalar(emit: &mut Emit<'_>, key: &str, x: HirNodeId) -> Result<HirNodeId> {
    let s = emit.load_param(key, false)?;
    let mut gb = HirMut::new(emit.hir());
    let s = gb.reshape_(s, vec![1]);
    Ok(gb.mul(x, s))
}

/// Encoder layer — causal (or windowed-causal) self-attention, taps K/V.
pub fn emit_encoder_layer(
    emit: &mut Emit<'_>,
    prefix: &str,
    x: HirNodeId,
    d: &LayerDims,
    cos: HirNodeId,
    sin: HirNodeId,
    mask: MaskKind,
) -> Result<(HirNodeId, KvTap)> {
    let normed = rms(
        emit,
        &format!("{prefix}.input_layernorm"),
        x,
        d.hidden,
        d.eps,
    )?;
    let (attn, tap) = emit_encoder_attention(
        emit,
        &format!("{prefix}.self_attn"),
        normed,
        d.attn,
        cos,
        sin,
        mask,
    )?;
    let attn = rms(
        emit,
        &format!("{prefix}.post_attention_layernorm"),
        attn,
        d.hidden,
        d.eps,
    )?;
    let residual = {
        let mut gb = HirMut::new(emit.hir());
        gb.add(x, attn)
    };
    let out = emit_ffn_block(emit, prefix, residual, d)?;
    let out = apply_layer_scalar(emit, &d.layer_scalar_key, out)?;
    Ok((out, tap))
}

/// Decoder layer — bidirectional attention over `[encoder K/V ; canvas K/V]`.
#[allow(clippy::too_many_arguments)]
pub fn emit_decoder_layer(
    emit: &mut Emit<'_>,
    prefix: &str,
    x: HirNodeId,
    d: &LayerDims,
    cos: HirNodeId,
    sin: HirNodeId,
    enc_k: HirNodeId,
    enc_v: HirNodeId,
    enc_len: usize,
) -> Result<HirNodeId> {
    let normed = rms(
        emit,
        &format!("{prefix}.input_layernorm"),
        x,
        d.hidden,
        d.eps,
    )?;
    let attn = emit_decoder_attention(
        emit,
        &format!("{prefix}.self_attn"),
        normed,
        d.attn,
        cos,
        sin,
        enc_k,
        enc_v,
        enc_len,
    )?;
    let attn = rms(
        emit,
        &format!("{prefix}.post_attention_layernorm"),
        attn,
        d.hidden,
        d.eps,
    )?;
    let residual = {
        let mut gb = HirMut::new(emit.hir());
        gb.add(x, attn)
    };
    let out = emit_ffn_block(emit, prefix, residual, d)?;
    apply_layer_scalar(emit, &d.layer_scalar_key, out)
}

/// `DiffusionGemmaSelfConditioning` — folds the previous denoising step's soft
/// embeddings into the canvas embeddings.
///
/// `inputs_embeds + down(gelu_tanh(gate(pre_norm(sc))) · up(pre_norm(sc)))`,
/// then a scale-free RMS norm. On the first step `sc` is all zeros, which makes
/// the gated branch zero and leaves `post_norm(inputs_embeds)`.
pub fn emit_self_conditioning(
    emit: &mut Emit<'_>,
    prefix: &str,
    inputs_embeds: HirNodeId,
    sc_signal: HirNodeId,
    hidden: usize,
    eps: f32,
) -> Result<HirNodeId> {
    let normed = rms(emit, &format!("{prefix}.pre_norm"), sc_signal, hidden, eps)?;
    let sc = emit_gated_mlp(emit, prefix, normed)?;
    let combined = {
        let mut gb = HirMut::new(emit.hir());
        gb.add(inputs_embeds, sc)
    };
    // `post_norm` is `with_scale=False`, so it has no checkpoint weight.
    let ones = emit.synth_param(
        &format!("{prefix}.post_norm.ones"),
        vec![1.0; hidden],
        Shape::new(&[hidden], DType::F32),
    );
    let zeros = emit.synth_param(
        &format!("{prefix}.post_norm.zeros"),
        vec![0.0; hidden],
        Shape::new(&[hidden], DType::F32),
    );
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.rms_norm(combined, ones, zeros, eps))
}
