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

//! Llama-4 text-decoder flow assembly (prefill).
//!
//! `token_embed → N × (RMSNorm → attention → +res → RMSNorm → FFN → +res) →
//! final RMSNorm → lm_head`. Every layer's FFN is MoE for Scout; Maverick
//! interleaves dense `Llama4TextMLP` layers. RoPE cos/sin `[seq, head_dim/2]`
//! and `input_ids [1, seq]` are graph inputs.

use anyhow::{Result, anyhow};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, Emit, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::attention::{AttnDims, ROPE_COS, ROPE_SIN, emit_attention};
use crate::config::Llama4TextConfig;
use crate::moe::emit_moe_ffn;

fn linear(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
    let w = emit.load_param(&format!("{prefix}.weight"), true)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.mm(x, w))
}

/// Weight-only RMSNorm (`gamma·x/rms(x)`), zero bias.
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

/// Dense SwiGLU FFN (`Llama4TextMLP`, non-MoE layers).
fn emit_dense_mlp(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
    let gate = linear(emit, &format!("{prefix}.gate_proj"), x)?;
    let up = linear(emit, &format!("{prefix}.up_proj"), x)?;
    let swiglu = {
        let mut gb = HirMut::new(emit.hir());
        let a = gb.silu(gate);
        gb.mul(a, up)
    };
    linear(emit, &format!("{prefix}.down_proj"), swiglu)
}

/// Build the Llama-4 text prefill graph for a fixed `seq`. Weights use the
/// HF `LlamaForCausalLM`-style names (`model.*`, `lm_head.weight`). With
/// `inputs_embeds`, the graph takes a host-assembled `inputs_embeds [1,seq,hidden]`
/// (for VLM image splicing) instead of `input_ids` + token embedding.
pub fn build_llama4_text_flow(
    cfg: &Llama4TextConfig,
    weights: &mut WeightMap,
    seq: usize,
    with_lm_head: bool,
    inputs_embeds: bool,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;
    let head_dim = cfg.head_dim();
    let half = head_dim / 2;
    let inter = cfg.intermediate_size;
    let inter_mlp = cfg.intermediate_size_mlp;
    let top_k = cfg.num_experts_per_tok;
    let use_qk_norm = cfg.use_qk_norm;
    let attn = AttnDims {
        hidden,
        num_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim,
        eps,
        seq,
    };

    let base = ModelFlow::new("llama4_text").with_profile(CompileProfile::llama32_prefill());
    let mut flow = if inputs_embeds {
        base.input("inputs_embeds", Shape::new(&[1, seq, hidden], f))
            .input(ROPE_COS, Shape::new(&[seq, half], f))
            .input(ROPE_SIN, Shape::new(&[seq, half], f))
            .zero_beta_named("llama4.zero_beta.hidden", hidden)
            .plugin_named("embeds_in", |emit, _prev| {
                Ok(Some(emit.flow_input("inputs_embeds")?))
            })
    } else {
        base.input("input_ids", Shape::new(&[1, seq], f))
            .input(ROPE_COS, Shape::new(&[seq, half], f))
            .input(ROPE_SIN, Shape::new(&[seq, half], f))
            .zero_beta_named("llama4.zero_beta.hidden", hidden)
            .token_embed()
    };

    let hidden_shape = Shape::new(&[1, seq, hidden], f);
    for i in 0..cfg.num_hidden_layers {
        let prefix = format!("model.layers.{i}");
        let use_rope = cfg.uses_rope(i);
        let is_moe = cfg.is_moe_layer(i);
        let hs = hidden_shape.clone();
        flow = flow.plugin_named(format!("layer{i}"), move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("layer{i} needs a hidden input"))?
                .hir_id();

            let normed = rmsnorm(emit, &format!("{prefix}.input_layernorm"), x, hidden, eps)?;
            let attn_out = emit_attention(
                emit,
                &format!("{prefix}.self_attn"),
                normed,
                attn,
                use_rope,
                use_qk_norm,
            )?;
            let h = {
                let mut gb = HirMut::new(emit.hir());
                gb.add(x, attn_out)
            };

            let normed2 = rmsnorm(
                emit,
                &format!("{prefix}.post_attention_layernorm"),
                h,
                hidden,
                eps,
            )?;
            let ffn = if is_moe {
                emit_moe_ffn(
                    emit,
                    &format!("{prefix}.feed_forward"),
                    normed2,
                    seq,
                    hidden,
                    inter,
                    top_k,
                )?
            } else {
                emit_dense_mlp(emit, &format!("{prefix}.feed_forward"), normed2)?
            };
            let out = {
                let mut gb = HirMut::new(emit.hir());
                gb.add(h, ffn)
            };
            Ok(Some(emit.wrap(out, hs.clone())))
        });
        let _ = inter_mlp; // used inside emit_dense_mlp via config intermediate_size_mlp naming
    }

    flow = flow.final_norm(eps);
    let flow = if with_lm_head {
        flow.lm_head(cfg.vocab_size, hidden, cfg.tie_word_embeddings)
            .output("logits")
    } else {
        flow.output("hidden")
    };
    flow.build_with(&mut WeightMapSource(weights), None)
}
