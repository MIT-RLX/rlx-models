//! CSM-1B configuration (HF `CsmConfig` / `config.json`).

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    pub factor: f32,
    #[serde(default = "default_low_freq")]
    pub low_freq_factor: f32,
    #[serde(default = "default_high_freq")]
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    #[serde(default)]
    pub rope_type: String,
}

fn default_low_freq() -> f32 {
    1.0
}
fn default_high_freq() -> f32 {
    4.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct DepthDecoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub num_codebooks: usize,
    pub backbone_hidden_size: usize,
    #[serde(default = "default_rms")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SesameConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub text_vocab_size: usize,
    pub num_codebooks: usize,
    #[serde(default = "default_rms")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default = "default_audio_token")]
    pub audio_token_id: u32,
    #[serde(default = "default_audio_eos")]
    pub audio_eos_token_id: u32,
    #[serde(default)]
    pub codebook_eos_token_id: u32,
    #[serde(default = "default_codebook_pad")]
    pub codebook_pad_token_id: u32,
    #[serde(default = "default_bos")]
    pub bos_token_id: u32,
    pub depth_decoder_config: DepthDecoderConfig,
    #[serde(default = "default_tie")]
    pub tie_codebooks_embeddings: bool,
}

fn default_rms() -> f64 {
    1e-5
}
fn default_theta() -> f64 {
    500_000.0
}
fn default_audio_token() -> u32 {
    128_002
}
fn default_audio_eos() -> u32 {
    128_003
}
fn default_codebook_pad() -> u32 {
    2050
}
fn default_bos() -> u32 {
    128_000
}
fn default_tie() -> bool {
    true
}

impl SesameConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("read config {}", path.display()))?;
        serde_json::from_str(&data).with_context(|| format!("parse {}", path.display()))
    }

    pub fn audio_sample_rate(&self) -> u32 {
        24_000
    }

    /// Frame width = num_codebooks + 1 text column (Sesame packing).
    pub fn frame_width(&self) -> usize {
        self.num_codebooks + 1
    }
}

impl Default for SesameConfig {
    fn default() -> Self {
        Self {
            hidden_size: 2048,
            intermediate_size: 8192,
            num_hidden_layers: 16,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 64,
            max_position_embeddings: 2048,
            vocab_size: 2051,
            text_vocab_size: 128_256,
            num_codebooks: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            rope_scaling: Some(RopeScaling {
                factor: 32.0,
                low_freq_factor: 0.125,
                high_freq_factor: 0.5,
                original_max_position_embeddings: 1024,
                rope_type: "llama3".into(),
            }),
            audio_token_id: 128_002,
            audio_eos_token_id: 128_003,
            codebook_eos_token_id: 0,
            codebook_pad_token_id: 2050,
            bos_token_id: 128_000,
            depth_decoder_config: DepthDecoderConfig {
                hidden_size: 1024,
                intermediate_size: 8192,
                num_hidden_layers: 4,
                num_attention_heads: 8,
                num_key_value_heads: 2,
                head_dim: 128,
                max_position_embeddings: 33,
                vocab_size: 2051,
                num_codebooks: 32,
                backbone_hidden_size: 2048,
                rms_norm_eps: 1e-5,
                rope_theta: 500_000.0,
                rope_scaling: Some(RopeScaling {
                    factor: 32.0,
                    low_freq_factor: 0.001_953_125,
                    high_freq_factor: 0.007_812_5,
                    original_max_position_embeddings: 16,
                    rope_type: "llama3".into(),
                }),
            },
            tie_codebooks_embeddings: true,
        }
    }
}
