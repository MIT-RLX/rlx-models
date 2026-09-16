// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Configuration for microsoft/VibeVoice-ASR (BitNet GGUF + Streaming safetensors).
//
// BitNet: Qwen2-1.5B (I2_S) + I8_S ConvNeXt VAEs.
// Streaming-7B: Qwen2.5-7B BF16 + BF16 ConvNeXt VAEs (GELU), chunked generate.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// Special token ids (canonical HuggingFace Qwen2.5 + VibeVoice ASR layout).
/// ASR speech markers reuse Qwen2.5-VL object/box slots:
///   speech_start = `<|object_ref_start|>`, speech_end = `<|object_ref_end|>`,
///   speech_pad   = `<|box_start|>`. Streaming also uses `<|text_chunk_end|>`.
pub const TOK_ENDOFTEXT: i64 = 151643; // <|endoftext|> (bos == eos)
pub const TOK_IM_START: i64 = 151644; // <|im_start|>
pub const TOK_IM_END: i64 = 151645; // <|im_end|>
pub const TOK_SPEECH_START: i64 = 151646; // <|object_ref_start|> / <|speech_start|>
pub const TOK_SPEECH_END: i64 = 151647; // <|object_ref_end|> / <|speech_end|>
pub const TOK_SPEECH_PAD: i64 = 151648; // <|box_start|> / <|speech_pad|>
pub const TOK_TEXT_CHUNK_END: i64 = 151665; // <|text_chunk_end|>

/// Audio front-end constants.
pub const TARGET_SR: usize = 24_000;
/// One speech frame per this many input samples (3200 → 7.5 Hz at 24 kHz).
pub const COMPRESS_RATIO: usize = 3_200;
/// RMS-normalization target (dBFS). BitNet path uses this; Streaming-7B ships
/// `normalize_audio: false` in preprocessor_config.json.
pub const TARGET_DBFS: f32 = -25.0;

/// The transcription system prompt used by VibeASR.cpp / non-streaming ASR.
pub const SYSTEM_PROMPT: &str =
    "You are a helpful assistant that transcribes audio input into text output in JSON format.";

/// Streaming prompt prefix (matches `streaming_generate` in modeling_vibevoice_asr.py).
pub const STREAMING_PROMPT_PREFIX: &str = "You are a helpful assistant that transcribes audio input into text output. \
Please transcribe the following audios streamingly with these keys: speaker, content";

/// Qwen2 decoder hyper-parameters.
#[derive(Debug, Clone)]
pub struct LmConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    /// Qwen2 uses biased Q/K/V projections and no per-head QK-norm.
    pub attention_bias: bool,
    pub qk_norm: bool,
    pub tie_word_embeddings: bool,
}

impl Default for LmConfig {
    /// BitNet / VibeVoice-ASR-1.5B defaults.
    fn default() -> Self {
        Self {
            hidden_size: 1536,
            num_hidden_layers: 28,
            num_attention_heads: 12,
            num_key_value_heads: 2,
            head_dim: 128, // 1536 / 12
            intermediate_size: 8960,
            vocab_size: 151_936,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 65_536,
            attention_bias: true,
            qk_norm: false,
            tie_word_embeddings: true,
        }
    }
}

impl LmConfig {
    /// Qwen2.5-7B layout used by VibeVoice-ASR-Streaming-7B.
    pub fn streaming_7b() -> Self {
        Self {
            hidden_size: 3584,
            num_hidden_layers: 28,
            num_attention_heads: 28,
            num_key_value_heads: 4,
            head_dim: 128, // 3584 / 28
            intermediate_size: 18944,
            vocab_size: 152_064,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 131_072,
            attention_bias: true,
            qk_norm: false,
            tie_word_embeddings: false,
        }
    }

    pub fn kv_proj_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
    pub fn q_proj_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }
}

/// Config of one ConvNeXt VAE encoder (acoustic or semantic).
#[derive(Debug, Clone)]
pub struct VaeEncoderConfig {
    /// Tensor prefix, `"acoustic"` or `"semantic"` (GGUF) / HF path stem.
    pub prefix: &'static str,
    /// Latent (VAE) dim out of the head conv (64 acoustic / 128 semantic).
    pub vae_dim: usize,
    /// Connector output dim (== LM hidden size).
    pub connector_dim: usize,
}

/// Downsample strides (index 0 is the stem, stride 1). Product = 3200.
pub const DOWNSAMPLE_STRIDES: [usize; 7] = [1, 2, 2, 4, 5, 5, 8];
/// Output channels after each downsample conv (num_filters doubling).
pub const DOWNSAMPLE_DIMS: [usize; 7] = [32, 64, 128, 256, 512, 1024, 2048];
/// ConvNeXt block depth per stage.
pub const STAGE_DEPTHS: [usize; 7] = [3, 3, 3, 3, 3, 3, 8];
/// RMSNorm epsilon used throughout the VAE (matches VibeASR.cpp `ggml_nn_rms_norm`).
pub const VAE_EPS: f32 = 1e-5;

/// FFN activation inside each ConvNeXt block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VaeFfnAct {
    /// BitNet I8_S path (VibeASR.cpp `ggml_nn_linear_relu`).
    #[default]
    Relu,
    /// Official BF16 / Streaming safetensors checkpoint.
    Gelu,
}

/// Streaming chunk schedule (from `preprocessor_config.json`).
#[derive(Debug, Clone)]
pub struct StreamingSchedule {
    pub chunk_frames: usize,
    pub lookahead_frames: usize,
    pub normalize_audio: bool,
}

impl Default for StreamingSchedule {
    fn default() -> Self {
        // microsoft/VibeVoice-ASR-Streaming-7B preprocessor_config.json
        Self {
            chunk_frames: 22,
            lookahead_frames: 4,
            normalize_audio: false,
        }
    }
}

impl StreamingSchedule {
    pub fn chunk_samples(&self) -> usize {
        self.chunk_frames * COMPRESS_RATIO
    }
    pub fn lookahead_samples(&self) -> usize {
        self.lookahead_frames * COMPRESS_RATIO
    }
    pub fn chunk_duration_sec(&self) -> f32 {
        self.chunk_samples() as f32 / TARGET_SR as f32
    }
    pub fn lookahead_duration_sec(&self) -> f32 {
        self.lookahead_samples() as f32 / TARGET_SR as f32
    }
}

/// Full model config.
#[derive(Debug, Clone)]
pub struct VibeAsrConfig {
    pub lm: LmConfig,
    pub acoustic: VaeEncoderConfig,
    pub semantic: VaeEncoderConfig,
    pub vae_ffn: VaeFfnAct,
    pub streaming: StreamingSchedule,
}

impl Default for VibeAsrConfig {
    fn default() -> Self {
        Self::bitnet_1_5b()
    }
}

impl VibeAsrConfig {
    pub fn bitnet_1_5b() -> Self {
        Self {
            lm: LmConfig::default(),
            acoustic: VaeEncoderConfig {
                prefix: "acoustic",
                vae_dim: 64,
                connector_dim: 1536,
            },
            semantic: VaeEncoderConfig {
                prefix: "semantic",
                vae_dim: 128,
                connector_dim: 1536,
            },
            vae_ffn: VaeFfnAct::Relu,
            streaming: StreamingSchedule::default(),
        }
    }

    pub fn streaming_7b() -> Self {
        let lm = LmConfig::streaming_7b();
        let h = lm.hidden_size;
        Self {
            lm,
            acoustic: VaeEncoderConfig {
                prefix: "acoustic",
                vae_dim: 64,
                connector_dim: h,
            },
            semantic: VaeEncoderConfig {
                prefix: "semantic",
                vae_dim: 128,
                connector_dim: h,
            },
            vae_ffn: VaeFfnAct::Gelu,
            streaming: StreamingSchedule::default(),
        }
    }

    /// Load from a HuggingFace model directory (`config.json` + optional
    /// `preprocessor_config.json`).
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let cfg_path = dir.join("config.json");
        let raw = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("read {}", cfg_path.display()))?;
        let mut cfg = Self::from_config_json(&raw)?;
        let prep = dir.join("preprocessor_config.json");
        if prep.is_file() {
            let p = std::fs::read_to_string(&prep)
                .with_context(|| format!("read {}", prep.display()))?;
            cfg.streaming = StreamingSchedule::from_preprocessor_json(&p)?;
        }
        Ok(cfg)
    }

    pub fn from_config_json(data: &str) -> Result<Self> {
        let raw: RawHfConfig = serde_json::from_str(data).context("parse vibevoice config.json")?;
        let dec = raw
            .decoder_config
            .ok_or_else(|| anyhow::anyhow!("config.json missing decoder_config"))?;
        let hidden = dec.hidden_size;
        let heads = dec.num_attention_heads.max(1);
        let lm = LmConfig {
            hidden_size: hidden,
            num_hidden_layers: dec.num_hidden_layers,
            num_attention_heads: heads,
            num_key_value_heads: dec.num_key_value_heads.max(1),
            head_dim: hidden / heads,
            intermediate_size: dec.intermediate_size,
            vocab_size: dec.vocab_size,
            rms_norm_eps: dec.rms_norm_eps.unwrap_or(1e-6),
            rope_theta: dec.rope_theta.unwrap_or(1_000_000.0),
            max_position_embeddings: dec.max_position_embeddings.unwrap_or(131_072),
            attention_bias: true,
            qk_norm: false,
            tie_word_embeddings: false,
        };
        let acoustic_dim = raw
            .acoustic_tokenizer_config
            .as_ref()
            .and_then(|c| c.vae_dim)
            .or(raw.acoustic_vae_dim)
            .unwrap_or(64);
        let semantic_dim = raw
            .semantic_tokenizer_config
            .as_ref()
            .and_then(|c| c.vae_dim)
            .or(raw.semantic_vae_dim)
            .unwrap_or(128);
        Ok(Self {
            lm,
            acoustic: VaeEncoderConfig {
                prefix: "acoustic",
                vae_dim: acoustic_dim,
                connector_dim: hidden,
            },
            semantic: VaeEncoderConfig {
                prefix: "semantic",
                vae_dim: semantic_dim,
                connector_dim: hidden,
            },
            // Safetensors BF16 checkpoints use GELU; BitNet GGUF stays Relu via
            // `bitnet_1_5b()`.
            vae_ffn: VaeFfnAct::Gelu,
            streaming: StreamingSchedule::default(),
        })
    }
}

impl StreamingSchedule {
    pub fn from_preprocessor_json(data: &str) -> Result<Self> {
        let raw: RawPreprocessor = serde_json::from_str(data).context("parse preprocessor")?;
        Ok(Self {
            chunk_frames: raw.chunk_frames.unwrap_or(22),
            lookahead_frames: raw.lookahead_frames.unwrap_or(4),
            normalize_audio: raw.normalize_audio.unwrap_or(false),
        })
    }
}

#[derive(Debug, Deserialize)]
struct RawHfConfig {
    decoder_config: Option<RawDecoder>,
    acoustic_vae_dim: Option<usize>,
    semantic_vae_dim: Option<usize>,
    acoustic_tokenizer_config: Option<RawTok>,
    semantic_tokenizer_config: Option<RawTok>,
}

#[derive(Debug, Deserialize)]
struct RawTok {
    vae_dim: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct RawDecoder {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    intermediate_size: usize,
    vocab_size: usize,
    rms_norm_eps: Option<f32>,
    rope_theta: Option<f32>,
    max_position_embeddings: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct RawPreprocessor {
    chunk_frames: Option<usize>,
    lookahead_frames: Option<usize>,
    normalize_audio: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_7b_dims() {
        let c = VibeAsrConfig::streaming_7b();
        assert_eq!(c.lm.hidden_size, 3584);
        assert_eq!(c.lm.num_key_value_heads, 4);
        assert_eq!(c.acoustic.connector_dim, 3584);
        assert_eq!(c.vae_ffn, VaeFfnAct::Gelu);
        assert_eq!(c.streaming.chunk_samples(), 22 * 3200);
        assert_eq!(c.streaming.lookahead_samples(), 4 * 3200);
    }

    #[test]
    fn parse_streaming_config_json() {
        let json = r#"{
          "acoustic_vae_dim": 64,
          "semantic_vae_dim": 128,
          "decoder_config": {
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "num_attention_heads": 28,
            "num_hidden_layers": 28,
            "num_key_value_heads": 4,
            "vocab_size": 152064,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 131072
          }
        }"#;
        let c = VibeAsrConfig::from_config_json(json).unwrap();
        assert_eq!(c.lm.head_dim, 128);
        assert!(!c.lm.tie_word_embeddings);
        assert_eq!(c.vae_ffn, VaeFfnAct::Gelu);
    }

    #[test]
    fn parse_preprocessor() {
        let json = r#"{"chunk_frames":22,"lookahead_frames":4,"normalize_audio":false}"#;
        let s = StreamingSchedule::from_preprocessor_json(json).unwrap();
        assert!((s.chunk_duration_sec() - 2.9333).abs() < 1e-3);
        assert!(!s.normalize_audio);
    }
}
