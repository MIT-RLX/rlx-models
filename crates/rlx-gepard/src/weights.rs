// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: Apache-2.0

//! Gepard weight key definitions and safetensors loading helpers.
//!
//! # Parameter-name prefixes (from MODEL_GUIDE §10.1)
//!
//! | prefix                    | module                              |
//! |---------------------------|-------------------------------------|
//! | `model.*`                 | Qwen3_5TextModel backbone           |
//! | `audio_embeddings.{i}.*`  | 32 × Embedding(L_i, 32)             |
//! | `audio_embed_proj.*`      | 2-layer GELU MLP + affine-free LN   |
//! | `audio_embed_scale`       | scalar buffer (not a parameter)     |
//! | `codebook_heads.{i}.*`    | 32 × Linear(1024, L_i)              |
//! | `stop_head.*`             | Linear(1024, 1)                     |
//! | `ref_compressor.*`        | Q-Former (voice cloning, optional)  |
//! | `null_prefix`             | Parameter[K=8, 1024] (optional)     |
//!
//! Backbone key naming (HF transformers Qwen3 layout):
//! ```text
//! model.embed_tokens.weight          [vocab, 1024]
//! model.norm.weight                  [1024]
//! model.layers.{i}.input_layernorm.weight
//! model.layers.{i}.self_attn.q_proj.weight / .bias
//! model.layers.{i}.self_attn.k_proj.weight / .bias
//! model.layers.{i}.self_attn.v_proj.weight / .bias
//! model.layers.{i}.self_attn.o_proj.weight
//! model.layers.{i}.post_attention_layernorm.weight
//! model.layers.{i}.mlp.gate_proj.weight
//! model.layers.{i}.mlp.up_proj.weight
//! model.layers.{i}.mlp.down_proj.weight
//! ```
//! Note: `lm_head` is absent — it was discarded during training (§10.1).

use crate::qformer::QFormerWeights;
use anyhow::{Context, Result};
use safetensors::SafeTensors;
use std::path::Path;

// ── key helpers ─────────────────────────────────────────────────────────────

/// Key for a per-channel audio embedding table (`[L_i, 32]`).
pub fn audio_emb_key(channel: usize) -> String {
    format!("audio_embeddings.{channel}.weight")
}

/// Keys for the audio-embed MLP (input projection, then output projection).
/// The MLP is `Sequential([Linear, GELU, Linear, LayerNorm])`, so layer
/// indices 0 and 2 hold the Linear layers.
pub fn audio_proj_key(layer_idx: usize, kind: &str) -> String {
    format!("audio_embed_proj.{layer_idx}.{kind}")
}

pub fn audio_embed_scale_key() -> &'static str {
    "audio_embed_scale"
}

/// Key for a codebook head weight/bias (`Linear(1024, L_i)`).
pub fn codebook_head_key(channel: usize, kind: &str) -> String {
    format!("codebook_heads.{channel}.{kind}")
}

pub fn stop_head_key(kind: &str) -> String {
    format!("stop_head.{kind}")
}

/// Key for a backbone transformer layer tensor.
pub fn backbone_layer_key(layer: usize, suffix: &str) -> String {
    format!("model.layers.{layer}.{suffix}")
}

pub fn backbone_embed_key() -> &'static str {
    "model.embed_tokens.weight"
}
pub fn backbone_final_norm_key() -> &'static str {
    "model.norm.weight"
}

// Q-Former keys
pub fn ref_compressor_key(suffix: &str) -> String {
    format!("ref_compressor.{suffix}")
}

pub fn null_prefix_key() -> &'static str {
    "null_prefix"
}

// ── tensor reader ────────────────────────────────────────────────────────────

/// Load the raw bytes of a safetensors file into memory.
pub fn load_safetensors_bytes(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).with_context(|| format!("read {}", path.display()))
}

/// Read a named tensor from an already-parsed SafeTensors view as `Vec<f32>`.
pub fn read_f32(st: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
    let view = st
        .tensor(name)
        .with_context(|| format!("tensor '{name}' not found in safetensors"))?;
    let bytes = view.data();
    use safetensors::tensor::Dtype;
    match view.dtype() {
        Dtype::F32 => Ok(f32_from_bytes_le(bytes)),
        Dtype::BF16 => Ok(bf16_to_f32(bytes)),
        Dtype::F16 => Ok(f16_to_f32(bytes)),
        other => anyhow::bail!("unsupported dtype {other:?} for tensor '{name}'"),
    }
}

/// Read a named scalar (single value) from a safetensors file.
pub fn read_scalar_f32(st: &SafeTensors<'_>, name: &str) -> Result<f32> {
    let data = read_f32(st, name)?;
    data.into_iter()
        .next()
        .with_context(|| format!("scalar tensor '{name}' is empty"))
}

/// Read tensor shape from the safetensors metadata.
pub fn read_shape(st: &SafeTensors<'_>, name: &str) -> Result<Vec<usize>> {
    let view = st
        .tensor(name)
        .with_context(|| format!("tensor '{name}' not found in safetensors"))?;
    Ok(view.shape().to_vec())
}

/// Check whether a tensor is present (used for optional modules).
pub fn has_tensor(st: &SafeTensors<'_>, name: &str) -> bool {
    st.tensor(name).is_ok()
}

// ── dtype conversion ─────────────────────────────────────────────────────────

fn f32_from_bytes_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

fn f16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            half_bits_to_f32(bits)
        })
        .collect()
}

fn half_bits_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) >> 15) << 31;
    let exp = ((h as u32 >> 10) & 0x1f) as i32 - 15 + 127;
    let mant = (h as u32 & 0x3ff) << 13;
    if exp <= 0 {
        return f32::from_bits(sign);
    }
    if exp >= 255 {
        return f32::from_bits(sign | 0x7f80_0000 | mant);
    }
    f32::from_bits(sign | ((exp as u32) << 23) | mant)
}

// ── overlay weight bundle ────────────────────────────────────────────────────

/// Everything from the non-backbone overlay that lives outside the transformer.
pub struct GepardOverlay {
    /// Per-channel embedding tables, channel order [0..32].
    /// Each table is `[L_i, audio_embed_dim]` = `[L_i, 32]`, row-major f32.
    pub audio_embeddings: Vec<Vec<f32>>,
    /// First MLP linear weights: `[hidden, 1024]` f32 (hidden = 1024).
    pub audio_proj_w0: Vec<f32>,
    pub audio_proj_b0: Vec<f32>,
    /// Second MLP linear weights: `[hidden, hidden]` f32.
    pub audio_proj_w2: Vec<f32>,
    pub audio_proj_b2: Vec<f32>,
    /// Scalar scale buffer (≈ text embedding std).
    pub audio_embed_scale: f32,
    /// Codebook head weights: 32 × `[L_i, 1024]`.
    pub codebook_weights: Vec<Vec<f32>>,
    /// Codebook head biases: 32 × `[L_i]`.
    pub codebook_biases: Vec<Vec<f32>>,
    /// Stop-head weight `[1, 1024]` and bias `[1]`.
    pub stop_weight: Vec<f32>,
    pub stop_bias: Vec<f32>,
    /// Speaker prefix from Q-Former (optional, `[K, hidden]`).
    pub null_prefix: Option<Vec<f32>>,
    /// Q-Former reference compressor weights (optional, for voice cloning).
    pub qformer: Option<QFormerWeights>,
}

impl GepardOverlay {
    /// Load overlay tensors from a `SafeTensors` view.
    pub fn load(st: &SafeTensors<'_>, num_channels: usize) -> Result<Self> {
        // Audio embeddings
        let mut audio_embeddings = Vec::with_capacity(num_channels);
        for i in 0..num_channels {
            let key = audio_emb_key(i);
            audio_embeddings.push(read_f32(st, &key).with_context(|| format!("loading {key}"))?);
        }

        // MLP projection
        let audio_proj_w0 = read_f32(st, &audio_proj_key(0, "weight"))?;
        let audio_proj_b0 = read_f32(st, &audio_proj_key(0, "bias"))?;
        let audio_proj_w2 = read_f32(st, &audio_proj_key(2, "weight"))?;
        let audio_proj_b2 = read_f32(st, &audio_proj_key(2, "bias"))?;

        // Scale buffer
        let audio_embed_scale = read_scalar_f32(st, audio_embed_scale_key()).unwrap_or(0.02); // sensible fallback if buffer absent

        // Codebook heads
        let mut codebook_weights = Vec::with_capacity(num_channels);
        let mut codebook_biases = Vec::with_capacity(num_channels);
        for i in 0..num_channels {
            codebook_weights.push(read_f32(st, &codebook_head_key(i, "weight"))?);
            codebook_biases.push(read_f32(st, &codebook_head_key(i, "bias"))?);
        }

        // Stop head
        let stop_weight = read_f32(st, &stop_head_key("weight"))?;
        let stop_bias = read_f32(st, &stop_head_key("bias"))?;

        // Optional null_prefix
        let null_prefix = if has_tensor(st, null_prefix_key()) {
            Some(read_f32(st, null_prefix_key())?)
        } else {
            None
        };

        // Optional Q-Former (voice cloning)
        let qformer = if has_tensor(st, "ref_compressor.input_proj.weight") {
            Some(QFormerWeights::load(st, 8, 2, 8, 1024, 32, 4)?)
        } else {
            None
        };

        Ok(Self {
            audio_embeddings,
            audio_proj_w0,
            audio_proj_b0,
            audio_proj_w2,
            audio_proj_b2,
            audio_embed_scale,
            codebook_weights,
            codebook_biases,
            stop_weight,
            stop_bias,
            null_prefix,
            qformer,
        })
    }

    /// List all expected overlay tensor keys (for validation).
    pub fn expected_keys(num_channels: usize) -> Vec<String> {
        let mut keys = Vec::new();
        for i in 0..num_channels {
            keys.push(audio_emb_key(i));
        }
        keys.push(audio_proj_key(0, "weight"));
        keys.push(audio_proj_key(0, "bias"));
        keys.push(audio_proj_key(2, "weight"));
        keys.push(audio_proj_key(2, "bias"));
        keys.push(audio_embed_scale_key().to_string());
        for i in 0..num_channels {
            keys.push(codebook_head_key(i, "weight"));
            keys.push(codebook_head_key(i, "bias"));
        }
        keys.push(stop_head_key("weight"));
        keys.push(stop_head_key("bias"));
        // Q-Former keys are optional
        keys
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_names() {
        assert_eq!(audio_emb_key(0), "audio_embeddings.0.weight");
        assert_eq!(audio_emb_key(31), "audio_embeddings.31.weight");
        assert_eq!(audio_proj_key(0, "weight"), "audio_embed_proj.0.weight");
        assert_eq!(audio_proj_key(2, "bias"), "audio_embed_proj.2.bias");
        assert_eq!(codebook_head_key(31, "weight"), "codebook_heads.31.weight");
        assert_eq!(stop_head_key("bias"), "stop_head.bias");
        assert_eq!(
            backbone_layer_key(0, "self_attn.q_proj.weight"),
            "model.layers.0.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn expected_keys_count() {
        let keys = GepardOverlay::expected_keys(32);
        // 32 audio_emb + 4 proj + 1 scale + 64 codebook + 2 stop = 103
        assert_eq!(keys.len(), 103);
    }
}
