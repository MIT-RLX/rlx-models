// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
//! Native Rust inference for [Kyutai Moshi](https://github.com/kyutai-labs/moshi) speech-to-speech.
//!
//! - **LM**: eager CPU ndarray ([`LmModel`]) or native RLX graphs ([`backend::MoshiLm`])
//! - **Codec**: [`rlx_mimi`] Mimi (CPU eager or native RLX graph on GPU)
//! - **Voices**: Moshiko / Moshika with per-preset HuggingFace routing ([`MoshiVoice`], [`MoshiCheckpoint`])
//! - **Presets**: bf16, Q8 GGUF, MLX Q4/Q8/bf16 ([`MoshiCheckpoint`])
//!
//! Entry points: [`MoshiSession`], CLI (`feature = "cli"`), streaming ([`spawn_duplex_stream`]).
//!
//! https://github.com/kyutai-labs/moshi

pub mod backend;
pub mod checkpoint;
pub mod config;
pub mod depformer;
pub mod device;
pub mod download;
pub mod generate;
pub mod gguf;
pub mod lm;
pub mod mlx_dequant;
pub mod mlx_weights;
pub mod nn;
pub mod rlx_gen;
pub mod rlx_lm;
pub mod sampling;
pub mod session;
pub mod stream;
pub mod tokenizer;
pub mod transformer;
pub mod weights;

#[cfg(feature = "cli")]
pub mod cli;

pub use crate::gguf::{gguf_tensor_count, load_gguf_weight_map};
pub use backend::{MoshiGenState, MoshiLm, resolve_lm_device};
pub use checkpoint::MoshiCheckpoint;
pub use config::{
    DepFormerConfig, GenerateConfig, LmConfig, MoshiVariant, MoshiVoice, TransformerConfig,
};
pub use device::{device_ready, parse_moshi_device, test_devices};
pub use download::{
    HF_MOSHI_MOSHIKO_REPO, default_mimi_dir, default_moshi_dir, default_moshi_dir_for,
    ensure_weights, ensure_weights_checkpoint, fetch_moshi, fetch_moshi_checkpoint,
    fetch_moshi_voice_checkpoint, resolve_moshi_dir,
};
pub use generate::GenerateState;
pub use lm::LmModel;
pub use mlx_weights::mlx_to_candle_key;
pub use session::{GenerationConfig, GenerationResult, MoshiSession, MoshiSessionParts};
pub use stream::{
    DuplexStreamEngine, FRAME_SAMPLES, StreamCommand, StreamEvent, StreamHandle, StreamStats,
    StreamStepOutput, WsMsgType, decode_ws_message, encode_ws_audio, encode_ws_handshake,
    encode_ws_text, spawn_duplex_stream,
};
pub use tokenizer::MoshiTokenizer;
pub use weights::{
    expected_lm_keys, load_lm_weights, load_lm_weights_from_checkpoint, load_weight_map, open_lm,
    open_lm_from_checkpoint, open_lm_from_weights,
};

#[cfg(feature = "tokio")]
pub use stream::{TokioStreamHandle, spawn_duplex_tokio};

#[cfg(feature = "cli")]
pub use cli::run;
