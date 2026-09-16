// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//!
//! `rlx-vibevoice-asr` — native RLX port of Microsoft VibeVoice-ASR:
//!
//! - **BitNet** ([VibeVoice-ASR-BitNet](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet)):
//!   I8_S ConvNeXt VAEs + I2_S Qwen2-1.5B GGUFs.
//! - **Streaming** ([VibeVoice-ASR-Streaming-7B](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B)):
//!   BF16 safetensors, dual ConvNeXt encoders + Qwen2.5-7B, chunked KV generate
//!   with `<|text_chunk_end|>`.

pub mod audio;
pub mod config;
pub mod embed;
pub mod lm;
pub mod load_streaming;
pub mod prompt;
pub mod stream_lm;
pub mod vae;
pub mod weights;

#[cfg(feature = "tokenizer")]
pub mod runner;
#[cfg(feature = "tokenizer")]
pub mod streaming;
#[cfg(feature = "tokenizer")]
pub mod tokenizer;

pub use audio::AudioData;
pub use config::{
    COMPRESS_RATIO, DOWNSAMPLE_DIMS, LmConfig, STAGE_DEPTHS, StreamingSchedule, TOK_TEXT_CHUNK_END,
    VaeEncoderConfig, VaeFfnAct, VibeAsrConfig,
};
pub use lm::VibeLm;
pub use load_streaming::StreamingWeightStore;
pub use prompt::{PromptTokens, build_prompt, build_prompt_default};
pub use stream_lm::StreamLm;
pub use vae::{
    VaeEncoderGraph, build_connector_graph, build_latent_graph, build_latent_graph_with_act,
    pad_to_multiple,
};
pub use weights::{VaeEncoderWeights, load_vae, load_vae_from_map};

#[cfg(feature = "tokenizer")]
pub use runner::VibeAsr;
#[cfg(feature = "tokenizer")]
pub use streaming::{EncodeMode, StreamChunk, StreamingAsr};
#[cfg(feature = "tokenizer")]
pub use tokenizer::VibeTokenizer;
