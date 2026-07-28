// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Configuration for microsoft/VibeVoice-ASR-BitNet.
//
// The LM decoder is Qwen2-1.5B (BitNet-quantized: I2_S ternary projections,
// Q6_K token embeddings, F16 output head). The two VAE encoders (acoustic +
// semantic) are ConvNeXt down-samplers shipped as I8_S. All numbers below are
// taken from config.json and verified against the shipped GGUF headers.

/// Special token ids (canonical HuggingFace Qwen2.5 + VibeVoice layout). These
/// are inserted by id directly — the GGUF BPE vocab may not round-trip them via
/// `parse_special`, but the trained embedding rows are always at these ids.
pub const TOK_ENDOFTEXT: i64 = 151643; // <|endoftext|> (bos == eos)
pub const TOK_IM_START: i64 = 151644; // <|im_start|>
pub const TOK_IM_END: i64 = 151645; // <|im_end|>
pub const TOK_SPEECH_START: i64 = 151646; // <|speech_start|>
pub const TOK_SPEECH_END: i64 = 151647; // <|speech_end|>
pub const TOK_SPEECH_PAD: i64 = 151648; // <|speech_pad|>

/// Audio front-end constants.
pub const TARGET_SR: usize = 24_000;
/// One speech frame per this many input samples (3200 → 7.5 Hz at 24 kHz).
pub const COMPRESS_RATIO: usize = 3_200;
/// RMS-normalization target (dBFS).
pub const TARGET_DBFS: f32 = -25.0;

/// The transcription system prompt used by VibeASR.cpp.
pub const SYSTEM_PROMPT: &str =
    "You are a helpful assistant that transcribes audio input into text output in JSON format.";

/// Qwen2 decoder hyper-parameters for the 1.5B ASR LM.
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
    /// gguf tensor prefix, `"acoustic"` or `"semantic"`.
    pub prefix: &'static str,
    /// Latent (VAE) dim out of the head conv (64 acoustic / 128 semantic).
    pub vae_dim: usize,
    /// Connector output dim (== LM hidden size, 1536).
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

/// Full model config.
#[derive(Debug, Clone)]
pub struct VibeAsrConfig {
    pub lm: LmConfig,
    pub acoustic: VaeEncoderConfig,
    pub semantic: VaeEncoderConfig,
}

impl Default for VibeAsrConfig {
    fn default() -> Self {
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
        }
    }
}
