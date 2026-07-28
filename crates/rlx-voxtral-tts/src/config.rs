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

//! Voxtral-4B-TTS config (`params.json` / HF layout).

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::path::{Path, PathBuf};

pub const HF_MODEL_ID: &str = "mistralai/Voxtral-4B-TTS-2603";
pub const CONSOLIDATED_WEIGHTS: &str = "consolidated.safetensors";
pub const PARAMS_FILE: &str = "params.json";
pub const TEKKEN_FILE: &str = "tekken.json";

#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralTtsConfig {
    pub text_config: TextConfig,
    pub audio_config: AudioConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub rope_theta: f64,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
}

impl TextConfig {
    pub fn llama_config(&self) -> rlx_llama32::Llama32Config {
        rlx_llama32::Llama32Config {
            embedding_scale: None,
            residual_scale: None,
            attention_scale: None,
            logit_scale: None,
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size.unwrap_or(self.hidden_size * 3),
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: if self.rope_theta > 0.0 {
                self.rope_theta
            } else {
                1_000_000.0
            },
            hidden_act: "silu".into(),
            tie_word_embeddings: true,
            attention_bias: false,
            head_dim: Some(self.head_dim),
            rope_scaling: None,
            num_loops: 1,
            skip_loop_final_norm: false,
            rope_style: rlx_ir::RopeStyle::NeoX,
            gguf_arch: None,
            rope_dim: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    pub codec_args: CodecArgs,
    pub audio_model_args: AudioModelArgs,
    #[serde(default)]
    pub speaker_id: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CodecArgs {
    pub channels: usize,
    pub sampling_rate: usize,
    pub pretransform_patch_size: usize,
    #[serde(default = "default_patch_proj_kernel_size")]
    pub patch_proj_kernel_size: usize,
    pub semantic_codebook_size: usize,
    pub semantic_dim: usize,
    pub acoustic_codebook_size: usize,
    pub acoustic_dim: usize,
    pub dim: usize,
    pub hidden_dim: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub attn_sliding_window_size: usize,
    pub encoder_transformer_lengths_str: String,
    pub encoder_convs_kernels_str: String,
    pub encoder_convs_strides_str: String,
    pub decoder_transformer_lengths_str: String,
    pub decoder_convs_kernels_str: String,
    pub decoder_convs_strides_str: String,
}

fn default_norm_eps() -> f64 {
    1e-5
}

#[derive(Debug, Clone, Deserialize)]
pub struct AcousticTransformerArgs {
    /// LM hidden size (`llm_projection` in/out).
    pub input_dim: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    pub sigma: f64,
    #[serde(default)]
    pub rope_theta: f64,
    #[serde(default)]
    pub use_biases: bool,
    pub n_decoding_steps: Option<usize>,
}

impl AcousticTransformerArgs {
    /// Sinusoidal time embedding base frequency (vLLM `TimeEmbedding` theta).
    pub fn time_theta(&self) -> f64 {
        if self.rope_theta > 0.0 {
            self.rope_theta
        } else {
            10_000.0
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioModelArgs {
    pub semantic_codebook_size: usize,
    pub acoustic_codebook_size: usize,
    pub n_acoustic_codebook: usize,
    pub acoustic_transformer_args: AcousticTransformerArgs,
}

impl VoxtralTtsConfig {
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let path = dir.join(PARAMS_FILE);
        let raw =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        parse_params_json(&raw).with_context(|| format!("parse {}", path.display()))
    }

    pub fn frame_rate(&self) -> f64 {
        let patch = self.audio_config.codec_args.pretransform_patch_size as f64;
        let stride_prod: usize = self
            .audio_config
            .codec_args
            .encoder_convs_strides_str
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .product();
        self.audio_config.codec_args.sampling_rate as f64 / (patch * stride_prod as f64)
    }

    pub fn downsample_factor(&self) -> usize {
        let fr = self.frame_rate();
        (self.audio_config.codec_args.sampling_rate as f64 / fr).round() as usize
    }
}

/// Accept either nested `VoxtralTtsConfig` JSON or Mistral flat `params.json`.
fn parse_params_json(raw: &str) -> Result<VoxtralTtsConfig> {
    if let Ok(cfg) = serde_json::from_str::<VoxtralTtsConfig>(raw) {
        return Ok(cfg);
    }

    let root: MistralParamsRoot =
        serde_json::from_str(raw).context("parse params.json as Mistral or VoxtralTtsConfig")?;
    root.into_voxtral_config()
}

#[derive(Debug, Deserialize)]
struct MistralParamsRoot {
    dim: usize,
    n_layers: usize,
    head_dim: usize,
    hidden_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    norm_eps: f64,
    rope_theta: f64,
    max_position_embeddings: Option<usize>,
    max_seq_len: Option<usize>,
    multimodal: MistralMultimodal,
}

#[derive(Debug, Deserialize)]
struct MistralMultimodal {
    audio_model_args: AudioModelArgs,
    audio_tokenizer_args: MistralCodecArgs,
}

#[derive(Debug, Deserialize)]
struct MistralCodecArgs {
    channels: usize,
    sampling_rate: usize,
    pretransform_patch_size: usize,
    #[serde(default = "default_patch_proj_kernel_size")]
    patch_proj_kernel_size: usize,
    semantic_codebook_size: usize,
    semantic_dim: usize,
    acoustic_codebook_size: usize,
    acoustic_dim: usize,
    dim: usize,
    hidden_dim: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    attn_sliding_window_size: usize,
    decoder_transformer_lengths_str: String,
    decoder_convs_kernels_str: String,
    decoder_convs_strides_str: String,
    #[serde(default)]
    encoder_transformer_lengths_str: String,
    #[serde(default)]
    encoder_convs_kernels_str: String,
    #[serde(default)]
    encoder_convs_strides_str: String,
}

impl MistralParamsRoot {
    fn into_voxtral_config(self) -> Result<VoxtralTtsConfig> {
        let max_pos = self
            .max_position_embeddings
            .or(self.max_seq_len)
            .unwrap_or(128_000);
        ensure!(self.dim > 0, "dim must be > 0");
        let tok = self.multimodal.audio_tokenizer_args;
        Ok(VoxtralTtsConfig {
            text_config: TextConfig {
                hidden_size: self.dim,
                num_hidden_layers: self.n_layers,
                num_attention_heads: self.n_heads,
                num_key_value_heads: self.n_kv_heads,
                head_dim: self.head_dim,
                vocab_size: self.vocab_size,
                rms_norm_eps: self.norm_eps,
                max_position_embeddings: max_pos,
                rope_theta: self.rope_theta,
                intermediate_size: Some(self.hidden_dim),
            },
            audio_config: AudioConfig {
                codec_args: CodecArgs {
                    channels: tok.channels,
                    sampling_rate: tok.sampling_rate,
                    pretransform_patch_size: tok.pretransform_patch_size,
                    patch_proj_kernel_size: tok.patch_proj_kernel_size,
                    semantic_codebook_size: tok.semantic_codebook_size,
                    semantic_dim: tok.semantic_dim,
                    acoustic_codebook_size: tok.acoustic_codebook_size,
                    acoustic_dim: tok.acoustic_dim,
                    dim: tok.dim,
                    hidden_dim: tok.hidden_dim,
                    head_dim: tok.head_dim,
                    n_heads: tok.n_heads,
                    n_kv_heads: tok.n_kv_heads,
                    attn_sliding_window_size: tok.attn_sliding_window_size,
                    encoder_transformer_lengths_str: default_encoder_lens(
                        &tok.encoder_transformer_lengths_str,
                    ),
                    encoder_convs_kernels_str: default_encoder_kernels(
                        &tok.encoder_convs_kernels_str,
                    ),
                    encoder_convs_strides_str: default_encoder_strides(
                        &tok.encoder_convs_strides_str,
                    ),
                    decoder_transformer_lengths_str: tok.decoder_transformer_lengths_str,
                    decoder_convs_kernels_str: tok.decoder_convs_kernels_str,
                    decoder_convs_strides_str: tok.decoder_convs_strides_str,
                },
                audio_model_args: self.multimodal.audio_model_args,
                speaker_id: std::collections::HashMap::new(),
            },
        })
    }
}

pub fn resolve_model_dir(path: &Path) -> Result<PathBuf> {
    if path.is_dir() {
        return Ok(path.to_path_buf());
    }
    path.parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| anyhow::anyhow!("weights path has no parent: {path:?}"))
}

impl CodecArgs {
    pub fn decoder_transformer_lengths(&self) -> Vec<usize> {
        parse_csv_usize(&self.decoder_transformer_lengths_str)
    }

    pub fn decoder_convs_kernels(&self) -> Vec<usize> {
        parse_csv_usize(&self.decoder_convs_kernels_str)
    }

    pub fn decoder_convs_strides(&self) -> Vec<usize> {
        parse_csv_usize(&self.decoder_convs_strides_str)
    }

    pub fn encoder_transformer_lengths(&self) -> Vec<usize> {
        parse_csv_usize(&self.encoder_transformer_lengths_str)
    }

    pub fn encoder_convs_kernels(&self) -> Vec<usize> {
        parse_csv_usize(&self.encoder_convs_kernels_str)
    }

    pub fn encoder_convs_strides(&self) -> Vec<usize> {
        parse_csv_usize(&self.encoder_convs_strides_str)
    }

    pub fn latent_dim(&self) -> usize {
        self.semantic_dim + self.acoustic_dim
    }
}

fn parse_csv_usize(s: &str) -> Vec<usize> {
    s.split(',')
        .filter_map(|p| p.trim().parse::<usize>().ok())
        .collect()
}

fn default_patch_proj_kernel_size() -> usize {
    7
}

fn default_encoder_lens(s: &str) -> String {
    if s.trim().is_empty() {
        "2,2,2,2".into()
    } else {
        s.to_string()
    }
}

fn default_encoder_kernels(s: &str) -> String {
    if s.trim().is_empty() {
        "4,4,4,3".into()
    } else {
        s.to_string()
    }
}

fn default_encoder_strides(s: &str) -> String {
    if s.trim().is_empty() {
        "2,2,2,1".into()
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_mistral_flat_params_json() {
        let raw = r#"{
            "dim": 3072,
            "n_layers": 26,
            "head_dim": 128,
            "hidden_dim": 9216,
            "n_heads": 32,
            "n_kv_heads": 8,
            "vocab_size": 131072,
            "norm_eps": 1e-5,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 128000,
            "multimodal": {
                "audio_model_args": {
                    "semantic_codebook_size": 8192,
                    "acoustic_codebook_size": 21,
                    "n_acoustic_codebook": 36,
                    "acoustic_transformer_args": {
                        "input_dim": 3072,
                        "dim": 3072,
                        "n_layers": 3,
                        "head_dim": 128,
                        "hidden_dim": 9216,
                        "n_heads": 32,
                        "n_kv_heads": 8,
                        "norm_eps": 1e-5,
                        "rope_theta": 10000.0,
                        "sigma": 1e-5
                    }
                },
                "audio_tokenizer_args": {
                    "channels": 1,
                    "sampling_rate": 24000,
                    "pretransform_patch_size": 240,
                    "semantic_codebook_size": 8192,
                    "semantic_dim": 256,
                    "acoustic_codebook_size": 21,
                    "acoustic_dim": 36,
                    "dim": 1024,
                    "hidden_dim": 4096,
                    "head_dim": 128,
                    "n_heads": 8,
                    "n_kv_heads": 8,
                    "attn_sliding_window_size": 16,
                    "decoder_transformer_lengths_str": "2,2,2,2",
                    "decoder_convs_kernels_str": "3,4,4,4",
                    "decoder_convs_strides_str": "1,2,2,2"
                }
            }
        }"#;
        let cfg = parse_params_json(raw).expect("parse sample params");
        assert_eq!(cfg.text_config.hidden_size, 3072);
        assert_eq!(cfg.text_config.rope_theta, 1_000_000.0);
        assert_eq!(
            cfg.audio_config
                .audio_model_args
                .acoustic_transformer_args
                .rope_theta,
            10_000.0
        );
        assert_eq!(cfg.audio_config.audio_model_args.n_acoustic_codebook, 36);
    }
}
