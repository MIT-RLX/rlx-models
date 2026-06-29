//! Kyutai TTS checkpoint identification and voice families.
//!
//! Today only one checkpoint ships publicly: `kyutai/tts-1.6b-en_fr`. Voice
//! conditioning is provided by pre-computed embeddings in
//! [`tts-voices`](https://huggingface.co/kyutai/tts-voices), looked up at
//! inference time via the cross-attention `speaker_wavs` conditioner — so
//! "voices" here are not separate repos like Moshiko/Moshika.

use std::path::{Path, PathBuf};

/// Published Kyutai TTS checkpoint preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KyutaiTtsCheckpoint {
    /// `kyutai/tts-1.6b-en_fr` — `dsm_tts_1e68beda@240.safetensors` (~3.68 GB).
    V1_6bEnFr,
}

impl KyutaiTtsCheckpoint {
    /// HuggingFace repo for this preset.
    pub fn hf_repo(self) -> &'static str {
        match self {
            Self::V1_6bEnFr => crate::download::HF_KYUTAI_TTS_REPO,
        }
    }

    /// Primary LM weights filename inside the model directory.
    pub fn lm_filename(self) -> &'static str {
        match self {
            Self::V1_6bEnFr => crate::download::TTS_WEIGHTS_FILE,
        }
    }

    /// SentencePiece tokenizer filename.
    pub fn tokenizer_filename(self) -> &'static str {
        match self {
            Self::V1_6bEnFr => crate::download::SPM_TOKENIZER_FILE,
        }
    }

    /// Mimi sidecar filename (Candle / `moshi`-format weights).
    pub fn mimi_sidecar_filename(self) -> &'static str {
        match self {
            Self::V1_6bEnFr => crate::download::MIMI_SIDECAR_FILE,
        }
    }

    /// Default `.cache/…` directory for this preset.
    pub fn default_cache_dir(self) -> PathBuf {
        PathBuf::from(".cache").join(match self {
            Self::V1_6bEnFr => "kyutai-tts-1.6b-en_fr",
        })
    }

    /// Absolute path to LM weights inside `model_dir`.
    pub fn lm_weights_path(self, model_dir: &Path) -> PathBuf {
        model_dir.join(self.lm_filename())
    }

    /// Absolute path to SentencePiece tokenizer inside `model_dir`.
    pub fn tokenizer_path(self, model_dir: &Path) -> PathBuf {
        model_dir.join(self.tokenizer_filename())
    }

    /// Approximate published checkpoint size, GB.
    pub fn size_class_gb(self) -> f32 {
        match self {
            Self::V1_6bEnFr => 3.7,
        }
    }

    /// Parse CLI / env preset names (`1.6b`, `1.6b-en_fr`, `default`).
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "1.6b" | "1.6b-en_fr" | "1.6b-en-fr" | "tts-1.6b-en_fr" | "default" => {
                Some(Self::V1_6bEnFr)
            }
            _ => None,
        }
    }

    /// Read `RLX_KYUTAI_TTS_CHECKPOINT`, defaulting to the 1.6B en/fr preset.
    pub fn from_env_or_default() -> Self {
        std::env::var("RLX_KYUTAI_TTS_CHECKPOINT")
            .ok()
            .and_then(|s| Self::parse(&s))
            .unwrap_or(Self::V1_6bEnFr)
    }

    /// HuggingFace repo for pre-computed voice embeddings.
    pub fn voice_repo(self) -> &'static str {
        crate::download::HF_KYUTAI_TTS_VOICES_REPO
    }

    /// Filename suffix for voice `.safetensors` (matches upstream `model_id` in the LM checkpoint).
    pub fn voice_embedding_suffix(self) -> &'static str {
        match self {
            Self::V1_6bEnFr => ".1e68beda@240.safetensors",
        }
    }

    /// HF path for a voice name (e.g. `alba-mackenna/casual.wav.1e68beda@240.safetensors`).
    pub fn voice_hf_filename(self, voice_name: &str) -> String {
        format!("{}{}", voice_name, self.voice_embedding_suffix())
    }
}

/// Voice identifier — resolved against `kyutai/tts-voices` (pre-computed 512-D embeddings).
///
/// Voices are not separate model checkpoints; they are tensor conditioners loaded into
/// the `speaker_wavs` cross-attention slot at inference time.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KyutaiTtsVoice {
    /// Voice name (matches a file under `kyutai/tts-voices`, e.g. `expresso/ex03-ex01_happy_001`).
    pub name: String,
}

impl KyutaiTtsVoice {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    /// Default unconditional / "zero" voice — no embedding loaded.
    pub fn unconditional() -> Self {
        Self::new("")
    }

    /// True for the unconditional / empty voice.
    pub fn is_unconditional(&self) -> bool {
        self.name.is_empty()
    }
}
