//! Kyutai Moshi weight checkpoint presets and HuggingFace routing.
//!
//! Each [`MoshiCheckpoint`] selects a quantization / file layout. Pair it with
//! [`MoshiVoice`] (via [`crate::config::MoshiVariant`]) to resolve the correct
//! `kyutai/moshiko-*` or `kyutai/moshika-*` repo.

use crate::config::MoshiVoice;
use std::path::{Path, PathBuf};

/// Published Moshi LM weight preset (quantization + on-disk layout).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoshiCheckpoint {
    /// `kyutai/moshiko-candle-bf16` — `model.safetensors` (~14 GB).
    Bf16Safetensors,
    /// `kyutai/moshiko-candle-q8` — `model.q8.gguf` (~8 GB).
    Q8Gguf,
    /// `kyutai/moshiko-mlx-q4` — `model.q4.safetensors` (~4 GB).
    Q4MlxSafetensors,
    /// `kyutai/moshiko-mlx-q8` — `model.q8.safetensors` (~7 GB).
    Q8MlxSafetensors,
    /// `kyutai/moshiko-mlx-bf16` — `model.safetensors` (~14 GB, MLX key layout).
    MlxBf16Safetensors,
}

impl MoshiCheckpoint {
    /// HuggingFace model repo for this preset and voice family.
    pub fn hf_repo(self, voice: MoshiVoice) -> &'static str {
        match (voice, self) {
            (MoshiVoice::Moshiko, Self::Bf16Safetensors) => "kyutai/moshiko-candle-bf16",
            (MoshiVoice::Moshiko, Self::Q8Gguf) => "kyutai/moshiko-candle-q8",
            (MoshiVoice::Moshiko, Self::Q4MlxSafetensors) => "kyutai/moshiko-mlx-q4",
            (MoshiVoice::Moshiko, Self::Q8MlxSafetensors) => "kyutai/moshiko-mlx-q8",
            (MoshiVoice::Moshiko, Self::MlxBf16Safetensors) => "kyutai/moshiko-mlx-bf16",
            (MoshiVoice::Moshika, Self::Bf16Safetensors) => "kyutai/moshika-candle-bf16",
            (MoshiVoice::Moshika, Self::Q8Gguf) => "kyutai/moshika-candle-q8",
            (MoshiVoice::Moshika, Self::Q4MlxSafetensors) => "kyutai/moshika-mlx-q4",
            (MoshiVoice::Moshika, Self::Q8MlxSafetensors) => "kyutai/moshika-mlx-q8",
            (MoshiVoice::Moshika, Self::MlxBf16Safetensors) => "kyutai/moshika-mlx-bf16",
        }
    }

    /// Default `.cache/…` directory for a voice + preset pair (unless `RLX_MOSHI_DIR` overrides).
    pub fn default_cache_dir(self, voice: MoshiVoice) -> PathBuf {
        let prefix = match voice {
            MoshiVoice::Moshiko => "moshiko",
            MoshiVoice::Moshika => "moshika",
        };
        let suffix = match self {
            Self::Bf16Safetensors => "",
            Self::Q8Gguf => "-q8",
            Self::Q4MlxSafetensors => "-mlx-q4",
            Self::Q8MlxSafetensors => "-mlx-q8",
            Self::MlxBf16Safetensors => "-mlx-bf16",
        };
        PathBuf::from(".cache").join(format!("{prefix}{suffix}"))
    }

    /// Primary LM weights filename inside a model directory.
    pub fn lm_filename(self) -> &'static str {
        match self {
            Self::Bf16Safetensors | Self::MlxBf16Safetensors => "model.safetensors",
            Self::Q8Gguf => "model.q8.gguf",
            Self::Q4MlxSafetensors => "model.q4.safetensors",
            Self::Q8MlxSafetensors => "model.q8.safetensors",
        }
    }

    /// True when weights are Kyutai Q8 GGUF (`model.q8.gguf`).
    pub fn is_gguf(self) -> bool {
        matches!(self, Self::Q8Gguf)
    }

    /// MLX key layout (Q4/Q8/bf16 safetensors).
    pub fn is_mlx(self) -> bool {
        matches!(
            self,
            Self::Q4MlxSafetensors | Self::Q8MlxSafetensors | Self::MlxBf16Safetensors
        )
    }

    /// Affine-quantized MLX weights (U32 packed + scales/biases).
    pub fn is_mlx_quantized(self) -> bool {
        matches!(self, Self::Q4MlxSafetensors | Self::Q8MlxSafetensors)
    }

    /// Bits per weight and group size for affine MLX quants; `(0, 0)` for full-precision MLX bf16.
    pub fn mlx_quant_params(self) -> (u32, usize) {
        match self {
            Self::Q4MlxSafetensors => (4, 32),
            Self::Q8MlxSafetensors => (8, 64),
            _ => (0, 0),
        }
    }

    /// Whether the Candle `moshi` GPU backend can load this preset.
    pub fn gpu_loadable(self) -> bool {
        self.candle_compatible() || self.is_mlx()
    }

    /// Candle-native tensor names (bf16 safetensors or Q8 GGUF).
    pub fn candle_compatible(self) -> bool {
        matches!(self, Self::Bf16Safetensors | Self::Q8Gguf)
    }

    /// Approximate published size class for docs / fetch hints.
    pub fn size_class_gb(self) -> f32 {
        match self {
            Self::Bf16Safetensors | Self::MlxBf16Safetensors => 14.0,
            Self::Q8Gguf => 8.0,
            Self::Q8MlxSafetensors => 7.0,
            Self::Q4MlxSafetensors => 4.0,
        }
    }

    /// Parse CLI / env preset names (`bf16`, `q8`, `q4`, `q8-mlx`, `mlx-bf16`, …).
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "bf16" | "f32" | "safetensors" | "default" => Some(Self::Bf16Safetensors),
            "q8" | "q8-gguf" | "gguf" => Some(Self::Q8Gguf),
            "q4" | "q4-mlx" | "mlx-q4" => Some(Self::Q4MlxSafetensors),
            "q8-mlx" | "mlx-q8" => Some(Self::Q8MlxSafetensors),
            "mlx-bf16" | "bf16-mlx" | "mlx-bf16-safetensors" => Some(Self::MlxBf16Safetensors),
            _ => None,
        }
    }

    /// Read `RLX_MOSHI_CHECKPOINT`, defaulting to Candle bf16 safetensors.
    pub fn from_env_or_default() -> Self {
        std::env::var("RLX_MOSHI_CHECKPOINT")
            .ok()
            .and_then(|s| Self::parse(&s))
            .unwrap_or(Self::Bf16Safetensors)
    }

    /// Absolute path to LM weights inside `model_dir`.
    pub fn lm_weights_path(self, model_dir: &Path) -> PathBuf {
        model_dir.join(self.lm_filename())
    }

    /// SentencePiece model path (shared across Kyutai repos).
    pub fn tokenizer_path(self, model_dir: &Path) -> PathBuf {
        let _ = self;
        model_dir.join("tokenizer_spm_32k_3.model")
    }

    /// Shared tokenizer + mimi sidecar files (same across Kyutai repos).
    pub const SHARED_FILES: &'static [&'static str] = &[
        "tokenizer_spm_32k_3.model",
        "tokenizer-e351c8d8-checkpoint125.safetensors",
    ];
}
