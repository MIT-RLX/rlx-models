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

//! NLLB-200 / M2M100 configuration.
//!
//! Preset matches [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M):
//! `d_model=1024`, 12/12 layers, 16 heads, FFN 4096, `relu`, `scale_embedding=true`.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// LayerNorm epsilon — PyTorch `nn.LayerNorm` default.
pub const LN_EPS: f32 = 1e-5;

/// Distilled 600M HF id.
pub const HF_DISTILLED_600M: &str = "facebook/nllb-200-distilled-600M";

/// M2M100 / NLLB text encoder–decoder configuration.
#[derive(Debug, Clone)]
pub struct NllbConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub decoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub decoder_ffn_dim: usize,
    pub max_position_embeddings: usize,
    /// When true, token embeds are multiplied by `sqrt(d_model)`.
    pub scale_embedding: bool,
    /// Activation in the FFN (`"relu"` for NLLB/M2M100).
    pub activation_function: String,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub decoder_start_token_id: u32,
    pub num_beams: usize,
    pub no_repeat_ngram_size: usize,
}

impl NllbConfig {
    /// Learned positional embeddings carry an offset of 2 (BART / M2M100 hack).
    pub const POS_OFFSET: usize = 2;

    pub fn enc_head_dim(&self) -> usize {
        self.d_model / self.encoder_attention_heads
    }
    pub fn dec_head_dim(&self) -> usize {
        self.d_model / self.decoder_attention_heads
    }

    /// Embedding scale (`sqrt(d_model)` when `scale_embedding`, else 1).
    pub fn embed_scale(&self) -> f32 {
        if self.scale_embedding {
            (self.d_model as f32).sqrt()
        } else {
            1.0
        }
    }

    /// M2M100 / NLLB sinusoidal positional table `[num_embeddings, d_model]`.
    /// Matches HuggingFace `M2M100SinusoidalPositionalEmbedding.get_embedding`.
    pub fn m2m100_sinusoidal_embedding(
        num_embeddings: usize,
        dim: usize,
        padding_idx: usize,
    ) -> Vec<f32> {
        let half = dim / 2;
        let log_base = 10000f32.ln();
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| {
                let exp = (i as f32) * (-log_base / (half as f32 - 1.0));
                exp.exp()
            })
            .collect();
        let mut out = vec![0f32; num_embeddings * dim];
        for pos in 0..num_embeddings {
            for i in 0..half {
                let angle = pos as f32 * inv_freq[i];
                out[pos * dim + i] = angle.sin();
                out[pos * dim + half + i] = angle.cos();
            }
            if dim % 2 == 1 {
                out[pos * dim + dim - 1] = 0.0;
            }
        }
        if padding_idx < num_embeddings {
            for v in &mut out[padding_idx * dim..(padding_idx + 1) * dim] {
                *v = 0.0;
            }
        }
        out
    }

    /// `facebook/nllb-200-distilled-600M` preset.
    pub fn distilled_600m() -> Self {
        Self {
            vocab_size: 256_206,
            d_model: 1024,
            encoder_layers: 12,
            decoder_layers: 12,
            encoder_attention_heads: 16,
            decoder_attention_heads: 16,
            encoder_ffn_dim: 4096,
            decoder_ffn_dim: 4096,
            max_position_embeddings: 1024,
            scale_embedding: true,
            activation_function: "relu".into(),
            bos_token_id: 0,
            eos_token_id: 2,
            pad_token_id: 1,
            decoder_start_token_id: 2,
            num_beams: 5,
            no_repeat_ngram_size: 0,
        }
    }

    /// Tiny config for unit / synthetic graph tests.
    pub fn tiny() -> Self {
        Self {
            vocab_size: 64,
            d_model: 32,
            encoder_layers: 2,
            decoder_layers: 2,
            encoder_attention_heads: 4,
            decoder_attention_heads: 4,
            encoder_ffn_dim: 64,
            decoder_ffn_dim: 64,
            max_position_embeddings: 32,
            scale_embedding: true,
            activation_function: "relu".into(),
            bos_token_id: 0,
            eos_token_id: 2,
            pad_token_id: 1,
            decoder_start_token_id: 2,
            num_beams: 1,
            no_repeat_ngram_size: 0,
        }
    }

    /// Parse an HF `config.json` (`model_type`: `m2m_100` / NLLB).
    pub fn from_hf_config_json(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("read nllb config {}", path.display()))?;
        let hf: HfConfig = serde_json::from_str(&raw)
            .with_context(|| format!("parse nllb config {}", path.display()))?;
        Ok(hf.into_config())
    }
}

#[derive(Debug, Deserialize)]
struct HfConfig {
    vocab_size: usize,
    d_model: usize,
    encoder_layers: usize,
    decoder_layers: usize,
    encoder_attention_heads: usize,
    decoder_attention_heads: usize,
    encoder_ffn_dim: usize,
    decoder_ffn_dim: usize,
    max_position_embeddings: usize,
    #[serde(default = "default_scale")]
    scale_embedding: bool,
    #[serde(default = "default_activation")]
    activation_function: String,
    bos_token_id: u32,
    eos_token_id: u32,
    pad_token_id: u32,
    decoder_start_token_id: u32,
    #[serde(default = "default_beams")]
    num_beams: usize,
    #[serde(default)]
    no_repeat_ngram_size: usize,
}

fn default_scale() -> bool {
    true
}
fn default_activation() -> String {
    "relu".into()
}
fn default_beams() -> usize {
    5
}

impl HfConfig {
    fn into_config(self) -> NllbConfig {
        NllbConfig {
            vocab_size: self.vocab_size,
            d_model: self.d_model,
            encoder_layers: self.encoder_layers,
            decoder_layers: self.decoder_layers,
            encoder_attention_heads: self.encoder_attention_heads,
            decoder_attention_heads: self.decoder_attention_heads,
            encoder_ffn_dim: self.encoder_ffn_dim,
            decoder_ffn_dim: self.decoder_ffn_dim,
            max_position_embeddings: self.max_position_embeddings,
            scale_embedding: self.scale_embedding,
            activation_function: self.activation_function,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            pad_token_id: self.pad_token_id,
            decoder_start_token_id: self.decoder_start_token_id,
            num_beams: self.num_beams,
            no_repeat_ngram_size: self.no_repeat_ngram_size,
        }
    }
}
