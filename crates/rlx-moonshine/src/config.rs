// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//! HF `config.json` fields for `MoonshineForConditionalGeneration`.

use anyhow::{Result, bail};
use serde::Deserialize;
use std::path::Path;

/// Native sample rate (Wav2Vec2-style feature extractor).
pub const SAMPLE_RATE: u32 = 16_000;

pub(crate) const LN_EPS: f32 = 1e-5;
pub(crate) const GROUP_NORM_EPS: f32 = 1e-5;

/// Known Moonshine sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoonshineVariant {
    Tiny,
    Base,
}

impl MoonshineVariant {
    pub fn repo_id(self) -> &'static str {
        match self {
            Self::Tiny => "UsefulSensors/moonshine-tiny",
            Self::Base => "UsefulSensors/moonshine-base",
        }
    }
}

/// HF `config.json` for Moonshine encoder–decoder ASR.
#[derive(Debug, Clone, Deserialize)]
pub struct MoonshineConfig {
    #[serde(default = "d_hidden")]
    pub hidden_size: usize,
    #[serde(default = "d_layers")]
    pub encoder_num_hidden_layers: usize,
    #[serde(default = "d_layers")]
    pub decoder_num_hidden_layers: usize,
    #[serde(default = "d_heads")]
    pub encoder_num_attention_heads: usize,
    #[serde(default = "d_heads")]
    pub decoder_num_attention_heads: usize,
    #[serde(default = "d_ff")]
    pub encoder_intermediate_size: usize,
    #[serde(default = "d_ff")]
    pub decoder_intermediate_size: usize,
    #[serde(default = "d_vocab")]
    pub vocab_size: usize,
    #[serde(default = "d_rope")]
    pub rope_theta: f64,
    #[serde(default = "d_partial")]
    pub partial_rotary_factor: f64,
    #[serde(default = "d_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default = "d_false")]
    pub attention_bias: bool,
    /// Pad per-head dim to a multiple of this value inside SDPA (HF
    /// `pad_head_dim_to_multiple_of`). Required for Metal flash/MMA paths when
    /// `hidden/heads` is not already aligned (tiny: 36 → 40).
    #[serde(default = "d_pad_hd")]
    pub pad_head_dim_to_multiple_of: Option<usize>,
    #[serde(default = "d_bos")]
    pub bos_token_id: u32,
    #[serde(default = "d_eos")]
    pub eos_token_id: u32,
    #[serde(default = "d_eos")]
    pub pad_token_id: u32,
    #[serde(default = "d_bos")]
    pub decoder_start_token_id: u32,
}

fn d_hidden() -> usize {
    288
}
fn d_layers() -> usize {
    6
}
fn d_heads() -> usize {
    8
}
fn d_ff() -> usize {
    1152
}
fn d_vocab() -> usize {
    32_768
}
fn d_rope() -> f64 {
    10_000.0
}
fn d_partial() -> f64 {
    0.9
}
fn d_max_pos() -> usize {
    194
}
fn d_false() -> bool {
    false
}
fn d_pad_hd() -> Option<usize> {
    Some(8)
}
fn d_bos() -> u32 {
    1
}
fn d_eos() -> u32 {
    2
}

impl MoonshineConfig {
    /// Published `moonshine-tiny` dims.
    pub fn tiny() -> Self {
        Self {
            hidden_size: 288,
            encoder_num_hidden_layers: 6,
            decoder_num_hidden_layers: 6,
            encoder_num_attention_heads: 8,
            decoder_num_attention_heads: 8,
            encoder_intermediate_size: 1152,
            decoder_intermediate_size: 1152,
            vocab_size: 32_768,
            rope_theta: 10_000.0,
            partial_rotary_factor: 0.9,
            max_position_embeddings: 194,
            attention_bias: false,
            pad_head_dim_to_multiple_of: Some(8),
            bos_token_id: 1,
            eos_token_id: 2,
            pad_token_id: 2,
            decoder_start_token_id: 1,
        }
    }

    /// Tiny-scale synthetic config for graph smoke tests (no real weights).
    pub fn synth_tiny() -> Self {
        Self {
            hidden_size: 32,
            encoder_num_hidden_layers: 1,
            decoder_num_hidden_layers: 1,
            encoder_num_attention_heads: 4,
            decoder_num_attention_heads: 4,
            encoder_intermediate_size: 64,
            decoder_intermediate_size: 64,
            vocab_size: 64,
            rope_theta: 10_000.0,
            partial_rotary_factor: 0.9,
            max_position_embeddings: 32,
            attention_bias: false,
            // head_dim=8 already aligned; keep padding off for synth graphs.
            pad_head_dim_to_multiple_of: None,
            bos_token_id: 1,
            eos_token_id: 2,
            pad_token_id: 2,
            decoder_start_token_id: 1,
        }
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&raw)?)
    }

    /// Load `config.json` from a weights directory.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let p = dir.join("config.json");
        if !p.is_file() {
            bail!("no config.json in {}", dir.display());
        }
        Self::from_file(&p)
    }

    pub fn enc_head_dim(&self) -> usize {
        self.hidden_size / self.encoder_num_attention_heads
    }

    pub fn dec_head_dim(&self) -> usize {
        self.hidden_size / self.decoder_num_attention_heads
    }

    /// Rotary width: `(head_dim * partial_rotary_factor)` rounded to even.
    pub fn n_rot(&self, head_dim: usize) -> usize {
        let n = ((head_dim as f64) * self.partial_rotary_factor).round() as usize;
        n & !1
    }

    /// SDPA head dim after optional padding: `(padded_hd, pad_amount)`.
    ///
    /// Matches HF `MoonshineAttention`: pad Q/K/V on the last dim, run SDPA at
    /// `padded_hd`, then slice back. Softmax scale stays `1/sqrt(raw head_dim)`.
    pub fn attn_padded_head_dim(&self, head_dim: usize) -> (usize, usize) {
        let Some(m) = self.pad_head_dim_to_multiple_of.filter(|&m| m > 0) else {
            return (head_dim, 0);
        };
        let padded = m * head_dim.div_ceil(m);
        (padded, padded - head_dim)
    }

    /// Conv frontend output length (HF `_get_feat_extract_output_lengths`).
    pub fn feat_extract_output_length(input_lengths: usize) -> usize {
        if input_lengths < 127 {
            return 0;
        }
        let c1 = (input_lengths - 127) / 64 + 1;
        if c1 < 7 {
            return 0;
        }
        let c2 = (c1 - 7) / 3 + 1;
        if c2 < 3 {
            return 0;
        }
        (c2 - 3) / 2 + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_preset() {
        let c = MoonshineConfig::tiny();
        assert_eq!(c.hidden_size, 288);
        assert_eq!(c.enc_head_dim(), 36);
        assert_eq!(c.n_rot(36), 32);
        assert_eq!(c.attn_padded_head_dim(36), (40, 4));
        assert!((c.partial_rotary_factor - 0.9).abs() < 1e-6);
    }

    #[test]
    fn feat_length_formula() {
        // ((L-127)/64+1 → (·-7)/3+1 → (·-3)/2+1) — integer floor division
        assert_eq!(MoonshineConfig::feat_extract_output_length(895), 1);
        assert_eq!(MoonshineConfig::feat_extract_output_length(16_000), 40);
        assert_eq!(MoonshineConfig::feat_extract_output_length(100), 0);
    }
}
