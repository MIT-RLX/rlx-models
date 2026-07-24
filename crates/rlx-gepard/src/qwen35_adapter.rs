// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: Apache-2.0

//! Map Gepard HF backbone config + safetensors into [`Qwen35Config`] /
//! [`Qwen35Weights`] for compiled multi-backend inference.

use anyhow::{Context, Result};
use rlx_qwen35::{
    MatWeight, Qwen35Config, Qwen35FullAttnLayer, Qwen35LayerFfn, Qwen35TrunkLayer, Qwen35Weights,
};
use safetensors::SafeTensors;
use std::sync::Arc;

use crate::config::BackboneConfig;
use crate::weights::{backbone_embed_key, backbone_final_norm_key, backbone_layer_key, read_f32};

/// Convert Gepard backbone metadata into a Qwen3.5 config (14 full-attn layers).
pub fn gepard_backbone_to_qwen35_config(bb: &BackboneConfig) -> Qwen35Config {
    let head_dim = bb.effective_head_dim();
    // Gepard eager backbone uses full-head RoPE (not partial MRoPE). Match that
    // in the compiled Qwen3.5 graphs so AR hidden states align with training.
    Qwen35Config {
        vocab_size: bb.vocab_size,
        hidden_size: bb.hidden_size,
        intermediate_size: bb.intermediate_size,
        num_hidden_layers: bb.num_hidden_layers,
        nextn_predict_layers: 0,
        num_attention_heads: bb.num_attention_heads,
        num_key_value_heads: bb.num_key_value_heads,
        key_length: head_dim,
        value_length: head_dim,
        max_position_embeddings: bb.max_position_embeddings,
        rms_norm_eps: bb.rms_norm_eps,
        rope_theta: bb.rope_theta,
        rope_dim_count: head_dim,
        rope_dim_sections: Vec::new(),
        mrope_interleaved: false,
        full_attention_interval: 1,
        ssm_conv_kernel: 4,
        ssm_group_count: 1,
        ssm_inner_size: 8,
        ssm_state_size: 4,
        ssm_time_step_rank: 1,
        tie_word_embeddings: true,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        shared_expert_ffn_size: 0,
        expert_weights_scale: 1.0,
        rms_norm_offset: false,
    }
}

fn mat_f32(data: Vec<f32>) -> MatWeight {
    MatWeight::F32(data)
}

/// HF Qwen3.5 stores zero-init RMSNorm weights for `(1 + w)`; RLX Qwen3.5 graphs
/// multiply by `gamma` directly — fold the `+1` into loaded norm tensors.
fn qwen35_rms_weight(w: Vec<f32>) -> Vec<f32> {
    w.into_iter().map(|wi| 1.0 + wi).collect()
}

fn load_full_attn_layer(st: &SafeTensors<'_>, layer: usize) -> Result<Qwen35FullAttnLayer> {
    let k = |s: &str| backbone_layer_key(layer, s);
    Ok(Qwen35FullAttnLayer {
        attn_norm: qwen35_rms_weight(read_f32(st, &k("input_layernorm.weight"))?),
        attn_post_norm: qwen35_rms_weight(read_f32(st, &k("post_attention_layernorm.weight"))?),
        attn_q_gate: mat_f32(read_f32(st, &k("self_attn.q_proj.weight"))?),
        attn_k: mat_f32(read_f32(st, &k("self_attn.k_proj.weight"))?),
        attn_v: mat_f32(read_f32(st, &k("self_attn.v_proj.weight"))?),
        attn_output: mat_f32(read_f32(st, &k("self_attn.o_proj.weight"))?),
        attn_q_norm: qwen35_rms_weight(read_f32(st, &k("self_attn.q_norm.weight"))?),
        attn_k_norm: qwen35_rms_weight(read_f32(st, &k("self_attn.k_norm.weight"))?),
        ffn: Qwen35LayerFfn::Dense {
            gate: mat_f32(read_f32(st, &k("mlp.gate_proj.weight"))?),
            up: mat_f32(read_f32(st, &k("mlp.up_proj.weight"))?),
            down: mat_f32(read_f32(st, &k("mlp.down_proj.weight"))?),
        },
    })
}

/// Load Gepard backbone tensors from `model.safetensors` into Qwen3.5 weights.
pub fn load_qwen35_weights_from_gepard(
    st: &SafeTensors<'_>,
    bb: &BackboneConfig,
) -> Result<Qwen35Weights> {
    let token_embd = Arc::<[f32]>::from(read_f32(st, backbone_embed_key())?.into_boxed_slice());
    let output_norm = qwen35_rms_weight(read_f32(st, backbone_final_norm_key())?);

    let mut trunk_layers = Vec::with_capacity(bb.num_hidden_layers);
    for il in 0..bb.num_hidden_layers {
        trunk_layers.push(Qwen35TrunkLayer::FullAttn(load_full_attn_layer(st, il)?));
    }

    Ok(Qwen35Weights {
        token_embd,
        output_norm,
        output: None,
        token_embd_lm: None,
        trunk_layers,
        mtp_layers: Vec::new(),
    })
}

/// Load config + weights from parsed safetensors.
pub fn load_gepard_qwen35_bundle(
    st: &SafeTensors<'_>,
    bb: &BackboneConfig,
) -> Result<(Qwen35Config, Qwen35Weights)> {
    let cfg = gepard_backbone_to_qwen35_config(bb);
    let weights = load_qwen35_weights_from_gepard(st, bb)
        .context("map Gepard safetensors to Qwen35Weights")?;
    Ok((cfg, weights))
}
