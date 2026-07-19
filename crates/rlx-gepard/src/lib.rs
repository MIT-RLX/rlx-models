// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: Apache-2.0

//! Gepard TTS — autoregressive decoder-only text-to-speech.
//!
//! # Architecture
//!
//! Gepard is a single-pass autoregressive TTS model built for real-time
//! voice synthesis:
//!
//! - **Backbone**: Qwen3.5 full-attention-only transformer (14 layers,
//!   hidden 1024, 8 heads, 2 KV heads, ~500 M params).
//! - **Audio interface (input)**: 32 per-channel FSQ embedding tables
//!   (`audio_embeddings.{0..31}`, each `[L_i, 32]`) concatenated to
//!   `[B, T, 1024]`, projected through a 2-layer GELU MLP
//!   (`audio_embed_proj`), L2-normalised (affine-free LayerNorm),
//!   and scaled by `audio_embed_scale` to match text embedding std.
//! - **Audio output**: 32 independent codebook heads
//!   (`codebook_heads.{0..31}`) predicting mixed-radix FSQ channels;
//!   one stop head (`stop_head`) predicts end-of-sequence.
//! - **Voice cloning** (optional): 8-query Q-Former compressor
//!   (`ref_compressor.*`) compresses reference codec codes into a
//!   speaker prefix prepended at prefill; the null-reference path uses
//!   a learnable `null_prefix` parameter.
//! - **Codec**: NVIDIA NeMo NanoCodec — 8 FSQ groups, levels `[8,7,6,6]`,
//!   22.05 kHz, 21.5 frames/s, 1.89 kbps.
//!   Each group produces 4 orthogonal channels via little-endian
//!   mixed-radix decomposition; the 8 groups yield 32 channels/frame.
//!
//! # Weights
//!
//! Weights live in `model.safetensors` next to `gepard_config.json`.
//! Download from `nineninesix/gepard-1.0` on Hugging Face.
//!
//! # References
//!
//! - Model card: <https://huggingface.co/nineninesix/gepard-1.0>
//! - Design guide: <https://github.com/nineninesix-ai/gepard-train/blob/main/docs/MODEL_GUIDE.md>

pub mod backbone;
pub mod cli;
pub mod codec_ops;
pub mod compiled_session;
pub mod config;
pub mod flow;
pub mod gepard_decoder;
pub mod qformer;
pub mod qwen35_adapter;
pub mod synthesis;
pub mod tokenizer;
pub mod training;
pub mod weights;

pub use compiled_session::{GepardCompiledSession, GepardTiming};
pub use config::GepardConfig;
pub use synthesis::{GepardSynthesizer, InferOpts, default_seed_for_text};
pub use tokenizer::GepardTokenizer;
pub use training::{AdamOptimizer, DataLoader, TrainingBatch, TrainingConfig, training_loop};
