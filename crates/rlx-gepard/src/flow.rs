// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: Apache-2.0

//! Gepard IR graph builders — backbone prefill/decode, Q-Former, audio heads.
//!
//! Builds static IR graphs for compilation across metal/mlx/cuda/rocm/wgpu/vulkan.
//! Uses RLX IR primitives to assemble models.
//!
//! **Note**: These are simplified placeholders. Production would use rlx_flow
//! ModelFlow builder (like rlx-qwen3/src/flow.rs) to assemble transformer layers
//! with proper attention + FFN + normalization stages.

use anyhow::Result;
use rlx_core::weight_loader::WeightLoader;
use rlx_ir::{DType, Graph, GraphExt, NodeId, Shape};

use crate::config::GepardConfig;

/// Helper to load a weight parameter and store it in params HashMap.
fn load_param(
    g: &mut Graph,
    params: &mut std::collections::HashMap<String, Vec<f32>>,
    _weights: &mut dyn WeightLoader,
    name: &str,
    shape: &[usize],
    _transpose: bool,
) -> Result<NodeId> {
    // Create placeholder data (in production: load from weights file)
    let size: usize = shape.iter().product();
    let data = vec![0.0f32; size];

    params.insert(name.to_string(), data);
    Ok(g.param(name, Shape::new(shape, DType::F32)))
}

/// Options for backbone prefill graph.
#[derive(Debug, Clone)]
pub struct BackbonePrefillOpts {
    pub batch: usize,
    pub seq: usize,
    pub with_lm_head: bool,
    pub with_kv_outputs: bool,
}

impl BackbonePrefillOpts {
    pub fn new(batch: usize, seq: usize) -> Self {
        Self {
            batch,
            seq,
            with_lm_head: false,
            with_kv_outputs: true,
        }
    }
}

/// Options for backbone decode graph.
#[derive(Debug, Clone)]
pub struct BackboneDecodeOpts {
    pub batch: usize,
    pub past_seq: usize,
    pub use_custom_mask: bool,
}

impl Default for BackboneDecodeOpts {
    fn default() -> Self {
        Self {
            batch: 1,
            past_seq: 0,
            use_custom_mask: false,
        }
    }
}

/// Build Qwen3.5 backbone prefill graph: `[batch, seq, hidden]` → hidden states.
///
/// When `with_kv_outputs` is true, outputs K and V per layer for cache seeding.
pub fn build_backbone_prefill_graph(
    cfg: &GepardConfig,
    weights: &mut dyn WeightLoader,
    opts: &BackbonePrefillOpts,
) -> Result<(Graph, std::collections::HashMap<String, Vec<f32>>)> {
    let mut params = std::collections::HashMap::new();
    let mut g = Graph::new("gepard_prefill");

    let hidden = cfg.hidden_size();
    let batch = opts.batch;
    let seq = opts.seq;
    let f = DType::F32;
    let num_heads = cfg.backbone.num_attention_heads;
    let _head_dim = hidden / num_heads;
    let inter_dim = cfg.backbone.intermediate_size;
    let eps = 1e-5f32;

    // Input: text embeddings [batch, seq, hidden]
    let mut h = g.input("hidden_states", Shape::new(&[batch, seq, hidden], f));

    // Transformer layers with attention + FFN + residuals
    for layer_idx in 0..cfg.backbone.num_hidden_layers {
        let layer_name = format!("backbone.layers.{}", layer_idx);

        // Pre-norm layer norm
        let ln1_w = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{}.ln1.weight", layer_name),
            &[hidden],
            false,
        )?;
        let ln1_b = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{}.ln1.bias", layer_name),
            &[hidden],
            false,
        )?;
        let h_norm = g.ln(h, ln1_w, ln1_b, eps);

        // Attention: QKV projections
        let q_w = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{}.attn_q.weight", layer_name),
            &[hidden, hidden],
            true,
        )?;
        let k_w = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{}.attn_k.weight", layer_name),
            &[hidden, hidden],
            true,
        )?;
        let v_w = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{}.attn_v.weight", layer_name),
            &[hidden, hidden],
            true,
        )?;

        let q = g.mm(h_norm, q_w);
        let k = g.mm(h_norm, k_w);
        let _v = g.mm(h_norm, v_w);

        // Simplified attention (in production: reshape to num_heads, apply scaled dot-product)
        let attn_out_w = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{}.attn_out.weight", layer_name),
            &[hidden, hidden],
            true,
        )?;
        let attn_combined = g.add(q, k); // Placeholder: should be proper attention
        let attn_out = g.mm(attn_combined, attn_out_w);

        // Residual connection
        let h_attn = g.add(h, attn_out);

        // FFN with pre-norm
        let ln2_w = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{}.ln2.weight", layer_name),
            &[hidden],
            false,
        )?;
        let ln2_b = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{}.ln2.bias", layer_name),
            &[hidden],
            false,
        )?;
        let h_ffn_norm = g.ln(h_attn, ln2_w, ln2_b, eps);

        // SwiGLU gate
        let fc1_w = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{}.ffn.w1.weight", layer_name),
            &[hidden, inter_dim],
            true,
        )?;
        let fc_gate_w = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{}.ffn.w3.weight", layer_name),
            &[hidden, inter_dim],
            true,
        )?;

        let fc1_out = g.mm(h_ffn_norm, fc1_w);
        let gate_out = g.mm(h_ffn_norm, fc_gate_w);
        let gate_activated = g.silu(gate_out);
        let swiglu = g.mul(fc1_out, gate_activated);

        // FC2 projection
        let fc2_w = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{}.ffn.w2.weight", layer_name),
            &[inter_dim, hidden],
            true,
        )?;
        let ffn_out = g.mm(swiglu, fc2_w);

        // Residual connection
        h = g.add(h_attn, ffn_out);
    }

    // Final layer norm
    let final_ln_w = load_param(
        &mut g,
        &mut params,
        weights,
        "backbone.norm.weight",
        &[hidden],
        false,
    )?;
    let final_ln_b = load_param(
        &mut g,
        &mut params,
        weights,
        "backbone.norm.bias",
        &[hidden],
        false,
    )?;
    let _output = g.ln(h, final_ln_w, final_ln_b, eps);

    Ok((g, params))
}

/// Build Qwen3.5 backbone decode graph: `[batch, 1, hidden]` + caches → logits + new cache.
pub fn build_backbone_decode_graph(
    cfg: &GepardConfig,
    weights: &mut dyn WeightLoader,
    opts: &BackboneDecodeOpts,
) -> Result<(Graph, std::collections::HashMap<String, Vec<f32>>)> {
    let mut params = std::collections::HashMap::new();
    let mut g = Graph::new("gepard_decode");

    let hidden = cfg.hidden_size();
    let batch = opts.batch;
    let inter_dim = cfg.backbone.intermediate_size;
    let eps = 1e-5f32;
    let f = DType::F32;

    // Input: [batch, 1, hidden] (single token)
    let mut h = g.input("hidden", Shape::new(&[batch, 1, hidden], f));

    // Simplified decode: single-token step through layers with same weights
    for layer_idx in 0..cfg.backbone.num_hidden_layers {
        let layer_name = format!("backbone.layers.{}", layer_idx);

        // Pre-norm
        let ln1_w = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{}.ln1.weight", layer_name),
            &[hidden],
            false,
        )?;
        let ln1_b = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{}.ln1.bias", layer_name),
            &[hidden],
            false,
        )?;
        let h_norm = g.ln(h, ln1_w, ln1_b, eps);

        // Attention (simplified for single token)
        let q_w = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{}.attn_q.weight", layer_name),
            &[hidden, hidden],
            true,
        )?;
        let v_w = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{}.attn_v.weight", layer_name),
            &[hidden, hidden],
            true,
        )?;

        let q = g.mm(h_norm, q_w);
        let v = g.mm(h_norm, v_w);
        let attn_combined = g.add(q, v); // Simplified placeholder

        let attn_out_w = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{}.attn_out.weight", layer_name),
            &[hidden, hidden],
            true,
        )?;
        let attn_out = g.mm(attn_combined, attn_out_w);
        let h_attn = g.add(h, attn_out);

        // FFN with pre-norm
        let ln2_w = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{}.ln2.weight", layer_name),
            &[hidden],
            false,
        )?;
        let ln2_b = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{}.ln2.bias", layer_name),
            &[hidden],
            false,
        )?;
        let h_ffn_norm = g.ln(h_attn, ln2_w, ln2_b, eps);

        // SwiGLU
        let fc1_w = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{}.ffn.w1.weight", layer_name),
            &[hidden, inter_dim],
            true,
        )?;
        let fc_gate_w = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{}.ffn.w3.weight", layer_name),
            &[hidden, inter_dim],
            true,
        )?;

        let fc1_out = g.mm(h_ffn_norm, fc1_w);
        let gate_out = g.mm(h_ffn_norm, fc_gate_w);
        let gate_activated = g.silu(gate_out);
        let swiglu = g.mul(fc1_out, gate_activated);

        let fc2_w = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{}.ffn.w2.weight", layer_name),
            &[inter_dim, hidden],
            true,
        )?;
        let ffn_out = g.mm(swiglu, fc2_w);

        h = g.add(h_attn, ffn_out);
    }

    let final_ln_w = load_param(
        &mut g,
        &mut params,
        weights,
        "backbone.norm.weight",
        &[hidden],
        false,
    )?;
    let final_ln_b = load_param(
        &mut g,
        &mut params,
        weights,
        "backbone.norm.bias",
        &[hidden],
        false,
    )?;
    let _output = g.ln(h, final_ln_w, final_ln_b, eps);

    Ok((g, params))
}

/// Build Q-Former compressor graph: codec codes → [8, hidden] speaker prefix.
pub fn build_qformer_graph(
    cfg: &GepardConfig,
    weights: &mut dyn WeightLoader,
) -> Result<(Graph, std::collections::HashMap<String, Vec<f32>>)> {
    let mut params = std::collections::HashMap::new();
    let mut g = Graph::new("gepard_qformer");

    let hidden = cfg.hidden_size();
    let num_queries = 8;
    let _f = DType::F32;
    let num_attn_heads = 8;
    let _head_dim = hidden / num_attn_heads;
    let eps = 1e-5f32;

    let _codes = g.input("codes", Shape::new(&[32], DType::I64));

    // Learnable query tokens: [num_queries, hidden]
    let query_tokens = load_param(
        &mut g,
        &mut params,
        weights,
        "qformer.query_tokens",
        &[num_queries, hidden],
        false,
    )?;

    // Self-attention block on query tokens
    let self_attn_q_w = load_param(
        &mut g,
        &mut params,
        weights,
        "qformer.self_attn.q_proj.weight",
        &[hidden, hidden],
        true,
    )?;
    let self_attn_k_w = load_param(
        &mut g,
        &mut params,
        weights,
        "qformer.self_attn.k_proj.weight",
        &[hidden, hidden],
        true,
    )?;
    let self_attn_v_w = load_param(
        &mut g,
        &mut params,
        weights,
        "qformer.self_attn.v_proj.weight",
        &[hidden, hidden],
        true,
    )?;

    let q = g.mm(query_tokens, self_attn_q_w);
    let k = g.mm(query_tokens, self_attn_k_w);
    let _v = g.mm(query_tokens, self_attn_v_w);

    // Simplified self-attention
    let attn_combined = g.add(q, k);
    let self_attn_out_w = load_param(
        &mut g,
        &mut params,
        weights,
        "qformer.self_attn.out_proj.weight",
        &[hidden, hidden],
        true,
    )?;
    let self_attn_out = g.mm(attn_combined, self_attn_out_w);

    // Residual + layer norm
    let attn_result = g.add(query_tokens, self_attn_out);
    let qformer_ln_w = load_param(
        &mut g,
        &mut params,
        weights,
        "qformer.ln.weight",
        &[hidden],
        false,
    )?;
    let qformer_ln_b = load_param(
        &mut g,
        &mut params,
        weights,
        "qformer.ln.bias",
        &[hidden],
        false,
    )?;
    let _output = g.ln(attn_result, qformer_ln_w, qformer_ln_b, eps);

    Ok((g, params))
}

/// Build audio embedding + codebook heads graph: hidden → [32 vocab logits].
pub fn build_audio_heads_graph(
    cfg: &GepardConfig,
    weights: &mut dyn WeightLoader,
) -> Result<(Graph, std::collections::HashMap<String, Vec<f32>>)> {
    let mut params = std::collections::HashMap::new();
    let mut g = Graph::new("gepard_audio_heads");

    let hidden = cfg.hidden_size();
    let f = DType::F32;
    let batch = 1; // Typical audio head batch

    // Input: hidden states [batch, hidden]
    let hidden_input = g.input("hidden", Shape::new(&[batch, hidden], f));

    // Load codebook head params and compute projections
    let vocabs = cfg.codec.channel_vocabs();
    let mut _logits = Vec::new();

    for (ch, &vocab_size) in vocabs.iter().enumerate() {
        let head_weight = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("codebook_heads.{}.weight", ch),
            &[hidden, vocab_size as usize],
            true,
        )?;
        let head_bias = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("codebook_heads.{}.bias", ch),
            &[vocab_size as usize],
            false,
        )?;

        // Compute logits: mm(hidden, weight) + bias
        let logits_raw = g.mm(hidden_input, head_weight);
        let _head_logits = g.add(logits_raw, head_bias);
        _logits.push(_head_logits);
    }

    // Stop head: scalar prediction for stopping criterion
    let stop_weight = load_param(
        &mut g,
        &mut params,
        weights,
        "stop_head.weight",
        &[hidden],
        false,
    )?;
    let _stop_bias = load_param(&mut g, &mut params, weights, "stop_head.bias", &[1], false)?;

    // Compute stop logit (simplified: use stop_weight as scalar multiplier)
    let _stop_logit = g.mul(hidden_input, stop_weight);

    Ok((g, params))
}
