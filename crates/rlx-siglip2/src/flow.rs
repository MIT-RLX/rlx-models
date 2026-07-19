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

//! SigLIP 2 flows — native [`ModelFlow`] assembly of the vision and text
//! transformer towers.
//!
//! Both towers are stacks of pre-LayerNorm SigLIP encoder layers:
//! ```text
//!   x = x + out_proj(MHSA(layer_norm1(x)))
//!   x = x + fc2(gelu_tanh(fc1(layer_norm2(x))))
//! ```
//! with separate `q/k/v/out_proj` projections (each with bias) and
//! `gelu_pytorch_tanh` activation. Attention is bidirectional in both
//! towers ([`MaskKind::None`]); NaFlex passes an additive padding bias.
//!
//! The vision tower ends with `post_layernorm` and a **MAP head**
//! (`MultiheadAttentionPoolingHead`): a learned probe cross-attends over
//! the patch sequence, then a residual LayerNorm + MLP; the probe row is
//! the pooled image embedding. The text tower ends with `final_layer_norm`
//! and returns the full sequence — last-token pooling and the linear head
//! run on host (see [`crate::runner`]).

use anyhow::Result;
use rlx_flow::{BuiltModel, CompileProfile, Emit, FlowValue, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::config::{LN_EPS, Siglip2Config};
use crate::preprocess::PoolingWeights;
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;

/// A linear layer `x @ Wᵀ + b`, loading HF `nn.Linear` params under `prefix`
/// (`{prefix}.weight` `[out, in]`, `{prefix}.bias` `[out]`).
fn emit_linear(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
    let w = emit.load_param(&format!("{prefix}.weight"), true)?;
    let b = emit.load_param(&format!("{prefix}.bias"), false)?;
    let mut gb = HirMut::new(emit.hir());
    let mm = gb.mm(x, w);
    Ok(gb.add(mm, b))
}

/// Emit one pre-LN SigLIP encoder layer under `prefix` (e.g.
/// `"vision_model.encoder.layers.3"`). `mask`, if present, is a binary
/// key-padding mask `[batch, seq]` (1 = attend, 0 = padded); NaFlex only.
fn emit_encoder_layer(
    emit: &mut Emit<'_>,
    prefix: &str,
    width: usize,
    heads: usize,
    eps: f32,
    mask: Option<HirNodeId>,
    input: &FlowValue,
) -> Result<FlowValue> {
    let head_dim = width / heads;
    let shape = input.shape.clone();
    let x = input.hir_id();

    // --- Attention sub-block (pre-norm) ---
    let ln1_g = emit.load_param(&format!("{prefix}.layer_norm1.weight"), false)?;
    let ln1_b = emit.load_param(&format!("{prefix}.layer_norm1.bias"), false)?;
    let normed1 = {
        let mut gb = HirMut::new(emit.hir());
        gb.ln(x, ln1_g, ln1_b, eps)
    };
    let q = emit_linear(emit, &format!("{prefix}.self_attn.q_proj"), normed1)?;
    let k = emit_linear(emit, &format!("{prefix}.self_attn.k_proj"), normed1)?;
    let v = emit_linear(emit, &format!("{prefix}.self_attn.v_proj"), normed1)?;
    let attn = {
        let mut gb = HirMut::new(emit.hir());
        let attn_shape = rlx_ir::shape::attention_shape(gb.shape(q));
        match mask {
            Some(m) => gb.attention(q, k, v, m, heads, head_dim, attn_shape),
            None => gb.attention_kind(q, k, v, heads, head_dim, MaskKind::None, attn_shape),
        }
    };
    let attn_out = emit_linear(emit, &format!("{prefix}.self_attn.out_proj"), attn)?;
    let res1 = {
        let mut gb = HirMut::new(emit.hir());
        gb.add(x, attn_out)
    };

    // --- MLP sub-block (pre-norm) ---
    let ln2_g = emit.load_param(&format!("{prefix}.layer_norm2.weight"), false)?;
    let ln2_b = emit.load_param(&format!("{prefix}.layer_norm2.bias"), false)?;
    let normed2 = {
        let mut gb = HirMut::new(emit.hir());
        gb.ln(res1, ln2_g, ln2_b, eps)
    };
    let fc1 = emit_linear(emit, &format!("{prefix}.mlp.fc1"), normed2)?;
    let act = {
        let mut gb = HirMut::new(emit.hir());
        gb.gelu_approx(fc1)
    };
    let fc2 = emit_linear(emit, &format!("{prefix}.mlp.fc2"), act)?;
    let out = {
        let mut gb = HirMut::new(emit.hir());
        gb.add(res1, fc2)
    };
    Ok(emit.wrap(out, shape))
}

/// Build the vision tower flow. Input `"hidden"` is `[batch, seq, width]`
/// (patch projections + position embeddings, assembled on host). For NaFlex
/// an extra input `"attn_bias"` `[batch, heads, seq, seq]` is bound. Output
/// `"image_embeds"` is `[batch, embed_dim]` (embed_dim == width).
pub fn build_vision_flow(
    cfg: &Siglip2Config,
    weights: &mut WeightMap,
    batch: usize,
    pooling: PoolingWeights,
) -> Result<BuiltModel> {
    let v = cfg.vision;
    let width = v.width;
    let heads = v.heads;
    let seq = match cfg.variant {
        crate::config::Variant::Fixed => v.seq_len(),
        crate::config::Variant::NaFlex => v.num_positions, // == max_num_patches
    };
    let eps = LN_EPS;
    let f = DType::F32;
    let masked = cfg.variant == crate::config::Variant::NaFlex;

    let mut flow = ModelFlow::new("siglip2_vision")
        .with_profile(CompileProfile::encoder())
        .input("hidden", Shape::new(&[batch, seq, width], f));
    if masked {
        // Binary key-padding mask (1 = attend, 0 = padded patch), shared by
        // the encoder and the MAP head (both mask the same keys).
        flow = flow.input("key_mask", Shape::new(&[batch, seq], f));
    }

    // Transformer encoder layers.
    for i in 0..v.layers {
        let prefix = format!("vision_model.encoder.layers.{i}");
        flow = flow.plugin_named(format!("vision.layer{i}"), move |emit, prev| {
            let input = prev.ok_or_else(|| anyhow::anyhow!("vision layer requires hidden"))?;
            let mask = if masked {
                Some(emit.flow_input("key_mask")?.hir_id())
            } else {
                None
            };
            let out = emit_encoder_layer(emit, &prefix, width, heads, eps, mask, &input)?;
            Ok(Some(out))
        });
    }

    // post_layernorm over every token.
    let post_shape = Shape::new(&[batch, seq, width], f);
    flow = flow.plugin_named("vision.post_ln", move |emit, prev| {
        let encoded = prev.ok_or_else(|| anyhow::anyhow!("post_ln requires hidden"))?;
        let g = emit.load_param("vision_model.post_layernorm.weight", false)?;
        let b = emit.load_param("vision_model.post_layernorm.bias", false)?;
        let mut gb = HirMut::new(emit.hir());
        let out = gb.ln(encoded.hir_id(), g, b, eps);
        Ok(Some(emit.wrap(out, post_shape.clone())))
    });

    // MAP head: probe cross-attends over the sequence → pooled image embed.
    flow = flow.plugin_named("vision.head", move |emit, prev| {
        let seq_val = prev.ok_or_else(|| anyhow::anyhow!("head requires hidden"))?;
        let mask = if masked {
            Some(emit.flow_input("key_mask")?.hir_id())
        } else {
            None
        };
        let out = emit_pooling_head(emit, &pooling, width, heads, eps, batch, mask, &seq_val)?;
        Ok(Some(out))
    });

    flow.output("image_embeds")
        .build_with(&mut WeightMapSource(weights), None)
}

/// SigLIP `MultiheadAttentionPoolingHead`. `seq_val` is the post-LN patch
/// sequence `[batch, seq, width]`; returns `[batch, width]`.
///
/// The packed `nn.MultiheadAttention` `in_proj_weight`/`in_proj_bias` are
/// pre-split host-side into q/k/v (see [`PoolingWeights`]) and injected as
/// synthetic params — the probe projects to Q, the sequence to K/V.
fn emit_pooling_head(
    emit: &mut Emit<'_>,
    pw: &PoolingWeights,
    width: usize,
    heads: usize,
    eps: f32,
    batch: usize,
    mask: Option<HirNodeId>,
    seq_val: &FlowValue,
) -> Result<FlowValue> {
    let head_dim = width / heads;
    let f = DType::F32;
    let wi = width as i64;

    // Probe [batch, 1, width] and the split q/k/v projections.
    let probe = emit.synth_param(
        "head.probe",
        pw.probe.clone(),
        Shape::new(&[batch, 1, width], f),
    );
    let qw = emit.synth_param("head.q_w", pw.q_w.clone(), Shape::new(&[width, width], f));
    let kw = emit.synth_param("head.k_w", pw.k_w.clone(), Shape::new(&[width, width], f));
    let vw = emit.synth_param("head.v_w", pw.v_w.clone(), Shape::new(&[width, width], f));
    let qb = emit.synth_param("head.q_b", pw.q_b.clone(), Shape::new(&[width], f));
    let kb = emit.synth_param("head.k_b", pw.k_b.clone(), Shape::new(&[width], f));
    let vb = emit.synth_param("head.v_b", pw.v_b.clone(), Shape::new(&[width], f));

    let seq_id = seq_val.hir_id();
    let (q, k, v) = {
        let mut gb = HirMut::new(emit.hir());
        let q = {
            let m = gb.mm(probe, qw);
            gb.add(m, qb)
        };
        let k = {
            let m = gb.mm(seq_id, kw);
            gb.add(m, kb)
        };
        let v = {
            let m = gb.mm(seq_id, vw);
            gb.add(m, vb)
        };
        (q, k, v)
    };
    let attn = {
        let mut gb = HirMut::new(emit.hir());
        // Query length 1 (the probe); the K/V length is implied by k's shape.
        let attn_shape = Shape::new(&[batch, 1, width], f);
        match mask {
            Some(m) => gb.attention(q, k, v, m, heads, head_dim, attn_shape),
            None => gb.attention_kind(q, k, v, heads, head_dim, MaskKind::None, attn_shape),
        }
    };
    // out_proj, then residual LayerNorm + MLP; return the (single) probe row.
    let attn_out = emit_linear(emit, "vision_model.head.attention.out_proj", attn)?;
    let ln_g = emit.load_param("vision_model.head.layernorm.weight", false)?;
    let ln_b = emit.load_param("vision_model.head.layernorm.bias", false)?;
    let normed = {
        let mut gb = HirMut::new(emit.hir());
        gb.ln(attn_out, ln_g, ln_b, eps)
    };
    let fc1 = emit_linear(emit, "vision_model.head.mlp.fc1", normed)?;
    let act = {
        let mut gb = HirMut::new(emit.hir());
        gb.gelu_approx(fc1)
    };
    let fc2 = emit_linear(emit, "vision_model.head.mlp.fc2", act)?;
    let out = {
        let mut gb = HirMut::new(emit.hir());
        let pooled = gb.add(attn_out, fc2); // residual is the attention output
        gb.reshape_(pooled, vec![batch as i64, wi])
    };
    Ok(emit.wrap(out, Shape::new(&[batch, width], f)))
}

/// Build the text tower flow. Input `"hidden"` is `[batch, ctx, width]`
/// (token + position embeddings assembled on host). Attention is
/// bidirectional with no padding mask (SigLIP pads to a fixed 64 and
/// attends to every position). Output `"text_hidden"` is the full
/// post-`final_layer_norm` sequence `[batch, ctx, width]`; last-token
/// pooling + the linear head run on host.
pub fn build_text_flow(
    cfg: &Siglip2Config,
    weights: &mut WeightMap,
    batch: usize,
) -> Result<BuiltModel> {
    let t = cfg.text;
    let width = t.width;
    let heads = t.heads;
    let seq = t.context_length;
    let eps = LN_EPS;
    let f = DType::F32;

    let mut flow = ModelFlow::new("siglip2_text")
        .with_profile(CompileProfile::encoder())
        .input("hidden", Shape::new(&[batch, seq, width], f));

    flow = flow.plugin_named("text.input", move |emit, _prev| {
        let x = emit.flow_input("hidden")?;
        Ok(Some(x))
    });

    for i in 0..t.layers {
        let prefix = format!("text_model.encoder.layers.{i}");
        flow = flow.plugin_named(format!("text.layer{i}"), move |emit, prev| {
            let input = prev.ok_or_else(|| anyhow::anyhow!("text layer requires hidden"))?;
            let out = emit_encoder_layer(emit, &prefix, width, heads, eps, None, &input)?;
            Ok(Some(out))
        });
    }

    flow = flow.plugin_named("text.final_ln", move |emit, prev| {
        let encoded = prev.ok_or_else(|| anyhow::anyhow!("final_ln requires hidden"))?;
        let g = emit.load_param("text_model.final_layer_norm.weight", false)?;
        let b = emit.load_param("text_model.final_layer_norm.bias", false)?;
        let mut gb = HirMut::new(emit.hir());
        let out = gb.ln(encoded.hir_id(), g, b, eps);
        Ok(Some(emit.wrap(out, Shape::new(&[batch, seq, width], f))))
    });

    flow.output("text_hidden")
        .build_with(&mut WeightMapSource(weights), None)
}
