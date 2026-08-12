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

//! GLM-4.6 text decoder flow. Standard pre-norm decoder with partial-RoPE GQA
//! attention and (after `first_k_dense_replace`) DeepSeek-style fine-grained MoE
//! — the MoE is byte-for-byte the DeepSeek router + expert layout, so it reuses
//! [`rlx_deepseek::moe::emit_deepseek_moe`].

use anyhow::{Result, anyhow};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_deepseek::moe::{DeepseekMoeDims, emit_deepseek_moe};
use rlx_flow::{BuiltModel, CompileProfile, Emit, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::attention::{GlmAttnDims, ROPE_COS, ROPE_SIN, emit_glm_attention};
use crate::config::Glm4MoeConfig;

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

fn dense_mlp(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
    let gate = linear(emit, &format!("{prefix}.gate_proj"), x)?;
    let up = linear(emit, &format!("{prefix}.up_proj"), x)?;
    let swiglu = {
        let mut gb = HirMut::new(emit.hir());
        let a = gb.silu(gate);
        gb.mul(a, up)
    };
    linear(emit, &format!("{prefix}.down_proj"), swiglu)
}

/// Build the GLM-4.6 text prefill graph for a fixed `seq`.
pub fn build_glm_text_flow(
    cfg: &Glm4MoeConfig,
    weights: &mut WeightMap,
    seq: usize,
    with_lm_head: bool,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;
    let half = cfg.rotary_dim() / 2;

    let attn = GlmAttnDims {
        hidden,
        num_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim(),
        rotary_dim: cfg.rotary_dim(),
        eps,
        seq,
        attention_bias: cfg.attention_bias,
        use_qk_norm: cfg.use_qk_norm,
    };
    let moe = DeepseekMoeDims {
        hidden,
        moe_inter: cfg.moe_intermediate_size,
        n_routed: cfg.n_routed_experts,
        top_k: cfg.num_experts_per_tok,
        n_group: cfg.n_group,
        topk_group: cfg.topk_group,
        routed_scaling: cfg.routed_scaling_factor,
        shared_inter: cfg.moe_intermediate_size * cfg.n_shared_experts,
        seq,
        experts_pretransposed: false,
        mxfp4_group: None,
    };

    let mut flow = ModelFlow::new("glm4_moe")
        .with_profile(CompileProfile::llama32_prefill())
        .input("input_ids", Shape::new(&[1, seq], f))
        .input(ROPE_COS, Shape::new(&[seq, half], f))
        .input(ROPE_SIN, Shape::new(&[seq, half], f))
        .zero_beta_named("glm4moe.zero_beta.hidden", hidden)
        .token_embed();

    let hs = Shape::new(&[1, seq, hidden], f);
    for i in 0..cfg.num_hidden_layers {
        let prefix = format!("model.layers.{i}");
        let is_moe = cfg.is_moe_layer(i);
        let hs = hs.clone();
        flow = flow.plugin_named(format!("layer{i}"), move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("layer{i} needs a hidden input"))?
                .hir_id();
            let normed = rmsnorm(emit, &format!("{prefix}.input_layernorm"), x, hidden, eps)?;
            let a = emit_glm_attention(emit, &format!("{prefix}.self_attn"), normed, attn)?;
            let h = {
                let mut gb = HirMut::new(emit.hir());
                gb.add(x, a)
            };
            let normed2 = rmsnorm(
                emit,
                &format!("{prefix}.post_attention_layernorm"),
                h,
                hidden,
                eps,
            )?;
            let ffn = if is_moe {
                emit_deepseek_moe(emit, &format!("{prefix}.mlp"), normed2, moe)?
            } else {
                dense_mlp(emit, &format!("{prefix}.mlp"), normed2)?
            };
            let out = {
                let mut gb = HirMut::new(emit.hir());
                gb.add(h, ffn)
            };
            Ok(Some(emit.wrap(out, hs.clone())))
        });
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
