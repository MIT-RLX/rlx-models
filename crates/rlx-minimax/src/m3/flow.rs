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

//! MiniMax-M3 text decoder flow:
//! `embed → N × (GemmaRMSNorm → attn(full|MSA) → +res → GemmaRMSNorm →
//! (dense|MoE) → +res) → GemmaRMSNorm → lm_head`.
//!
//! Layers `0..first_dense` (per `moe_layer_freq`/`sparse_attention_freq`) use a
//! dense MLP + full causal attention; the rest use the MoE + MSA sparse
//! attention. RoPE cos/sin are graph inputs (`[seq, n_rot/2]`).

use anyhow::{Result, anyhow};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Shape};

use super::attention::{M3AttnDims, emit_m3_attention};
use super::config::MiniMaxM3Config;
use super::mlp::emit_dense_mlp;
use super::moe::{M3MoeDims, emit_m3_moe};
use super::ops::gemma_rmsnorm;
use super::{ROPE_COS, ROPE_SIN};

/// Build the MiniMax-M3 text prefill graph for a fixed `seq`, gathering token
/// embeddings from `input_ids`.
pub fn build_m3_text_flow(
    cfg: &MiniMaxM3Config,
    weights: &mut WeightMap,
    seq: usize,
    with_lm_head: bool,
) -> Result<BuiltModel> {
    build_m3_text_flow_opts(cfg, weights, seq, with_lm_head, false)
}

/// Build the MiniMax-M3 text prefill graph consuming a precomputed
/// `inputs_embeds [1, seq, hidden]` input (the VL path: image features spliced
/// into the token embeddings host-side).
pub fn build_m3_text_embeds_flow(
    cfg: &MiniMaxM3Config,
    weights: &mut WeightMap,
    seq: usize,
    with_lm_head: bool,
) -> Result<BuiltModel> {
    build_m3_text_flow_opts(cfg, weights, seq, with_lm_head, true)
}

fn build_m3_text_flow_opts(
    cfg: &MiniMaxM3Config,
    weights: &mut WeightMap,
    seq: usize,
    with_lm_head: bool,
    embeds: bool,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;
    let half = cfg.n_rot() / 2;
    let hd = cfg.head_dim();

    let mut flow = ModelFlow::new("minimax_m3").with_profile(CompileProfile::llama32_prefill());
    if embeds {
        flow = flow
            .input("inputs_embeds", Shape::new(&[1, seq, hidden], f))
            .input(ROPE_COS, Shape::new(&[seq, half], f))
            .input(ROPE_SIN, Shape::new(&[seq, half], f));
        flow = flow.plugin_named("embeds_in", |emit, _prev| {
            Ok(Some(emit.flow_input("inputs_embeds")?))
        });
    } else {
        flow = flow
            .input("input_ids", Shape::new(&[1, seq], f))
            .input(ROPE_COS, Shape::new(&[seq, half], f))
            .input(ROPE_SIN, Shape::new(&[seq, half], f))
            .token_embed();
    }

    let hs = Shape::new(&[1, seq, hidden], f);
    for i in 0..cfg.num_hidden_layers {
        let prefix = format!("model.layers.{i}");
        let is_moe = cfg.is_moe_layer(i);
        let attn = M3AttnDims {
            hidden,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: hd,
            n_rot: cfg.n_rot(),
            eps,
            seq,
            sparse: cfg.is_sparse_layer(i),
            index_head_dim: cfg.sparse.index_head_dim,
            block_size: cfg.sparse.block_size,
            topk_blocks: cfg.sparse.topk_blocks,
            local_blocks: cfg.sparse.local_blocks,
        };
        let moe = M3MoeDims {
            hidden,
            moe_inter: cfg.moe_intermediate_size,
            shared_inter: cfg.shared_inter(),
            n_routed: cfg.num_local_experts,
            top_k: cfg.num_experts_per_tok,
            routed_scaling: cfg.routed_scaling_factor,
            alpha: cfg.swiglu_alpha,
            limit: cfg.swiglu_limit,
            seq,
        };
        let dense_inter = cfg.dense_intermediate_size;
        let (alpha, limit) = (cfg.swiglu_alpha, cfg.swiglu_limit);
        let hs = hs.clone();
        flow = flow.plugin_named(format!("layer{i}"), move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("layer{i} needs a hidden input"))?
                .hir_id();
            let normed = gemma_rmsnorm(emit, &format!("{prefix}.input_layernorm"), x, hidden, eps)?;
            let a = emit_m3_attention(emit, &format!("{prefix}.self_attn"), normed, attn)?;
            let h = {
                let mut gb = HirMut::new(emit.hir());
                gb.add(x, a)
            };
            let normed2 = gemma_rmsnorm(
                emit,
                &format!("{prefix}.post_attention_layernorm"),
                h,
                hidden,
                eps,
            )?;
            let ffn = if is_moe {
                emit_m3_moe(emit, &format!("{prefix}.block_sparse_moe"), normed2, moe)?
            } else {
                emit_dense_mlp(
                    emit,
                    &format!("{prefix}.mlp"),
                    normed2,
                    seq,
                    hidden,
                    dense_inter,
                    alpha,
                    limit,
                )?
            };
            let out = {
                let mut gb = HirMut::new(emit.hir());
                gb.add(h, ffn)
            };
            Ok(Some(emit.wrap(out, hs.clone())))
        });
    }

    // Final Gemma RMSNorm + (untied) LM head, as a plugin so the (1+w) norm is used.
    let vocab = cfg.vocab_size;
    let tie = cfg.tie_word_embeddings;
    if with_lm_head {
        flow = flow.plugin_named("lm_head", move |emit, prev| {
            let h = prev
                .ok_or_else(|| anyhow!("lm_head needs a hidden input"))?
                .hir_id();
            let normed = gemma_rmsnorm(emit, "model.norm", h, hidden, eps)?;
            let key = if tie {
                "model.embed_tokens.weight"
            } else {
                "lm_head.weight"
            };
            let lm_w = emit.load_param(key, true)?;
            let mut gb = HirMut::new(emit.hir());
            let logits = gb.mm(normed, lm_w);
            Ok(Some(emit.wrap(logits, Shape::new(&[1, seq, vocab], f))))
        });
        flow = flow.output("logits");
    } else {
        flow = flow.plugin_named("final_norm", move |emit, prev| {
            let h = prev
                .ok_or_else(|| anyhow!("final_norm needs a hidden input"))?
                .hir_id();
            let normed = gemma_rmsnorm(emit, "model.norm", h, hidden, eps)?;
            Ok(Some(emit.wrap(normed, hs.clone())))
        });
        flow = flow.output("hidden");
    }

    flow.build_with(&mut WeightMapSource(weights), None)
}
