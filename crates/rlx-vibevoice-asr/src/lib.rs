// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
//! `rlx-vibevoice-asr` — native RLX port of **microsoft/VibeVoice-ASR-BitNet**,
//! a CPU-first speech-recognition model.
//!
//! Pipeline: 24 kHz mono audio → RMS-normalize → two ConvNeXt VAE encoders
//! (acoustic + semantic, shipped `I8_S`) → per-encoder `SpeechConnector`
//! (Linear→RMSNorm→Linear → 1536) → element-wise sum spliced into the
//! `<|speech_pad|>` rows of a Qwen2.5 chat prompt → BitNet Qwen2-1.5B LM
//! (`I2_S` ternary projections + `Q6_K` embeddings) → greedy decode →
//! transcription.
//!
//! Microsoft's shipped GGUFs load directly: the `I2_S` (ggml 36) and `I8_S`
//! (ggml 37) formats were added to `rlx-gguf`. The LM ternary projections and
//! `Q6_K` embeddings are currently dequantized to f32 on load (correctness-first
//! dense path). Transcoding the `I2_S` projections to rlx's packed `TQ2_0`
//! DequantMatMul scheme (numerically exact for ternary, ~7× smaller working set)
//! is the tracked next step — see `TODO(bitnet)` in `lm.rs`.

pub mod audio;
pub mod config;
pub mod embed;
pub mod lm;
pub mod prompt;
pub mod vae;
pub mod weights;

#[cfg(feature = "tokenizer")]
pub mod runner;
#[cfg(feature = "tokenizer")]
pub mod tokenizer;

pub use audio::AudioData;
pub use config::{LmConfig, VaeEncoderConfig, VibeAsrConfig};
pub use lm::VibeLm;
pub use prompt::{PromptTokens, build_prompt, build_prompt_default};
pub use vae::{VaeEncoderGraph, build_connector_graph, build_latent_graph, pad_to_multiple};
pub use weights::{VaeEncoderWeights, load_vae};

#[cfg(feature = "tokenizer")]
pub use runner::VibeAsr;
#[cfg(feature = "tokenizer")]
pub use tokenizer::VibeTokenizer;
