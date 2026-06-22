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

//! Native Rust inference for [Kyutai TTS](https://huggingface.co/kyutai/tts-1.6b-en_fr).
//!
//! ## Architecture (mirrors the published `config.json`)
//!
//! | Block | Spec |
//! |-------|------|
//! | Backbone temporal LM | 1B Helium-style, 16 layers × 16 heads, `d_model=2048`, `context=500`, RoPE, RMSNorm f32, SwiGLU (`hidden_scale=4.125`) |
//! | DepFormer (depth decoder) | 600M, 4 layers × 16 heads, `d_model=1024`, **per-step weights** (33-entry sharing schedule → 11 unique heads), low-rank codebook embeddings (rank 128) |
//! | Audio codec | Mimi via [`rlx_mimi`] — 12.5 Hz, 24 kHz mono, 32 codebooks (`n_q = dep_q = 32`) |
//! | Conditioners | `speaker_wavs` (512-D tensor, cross-attn) + `cfg` (7-bin LUT) + `control` (1-bin LUT) |
//! | Streams | Demuxed second stream (`demux_second_stream = true`); audio shifted 1.28 s (16 frames) behind text |
//! | Sampling | Distilled CFG (single-pass — no batch doubling), `temp = text_temp = 0.6` |
//!
//! Per-codebook delays and the per-step head schedule are exposed as
//! [`DELAYS_DEFAULT`] and [`DEPFORMER_WEIGHTS_PER_STEP_SCHEDULE`].
//!
//! ## Files in `kyutai/tts-1.6b-en_fr`
//!
//! | File | Size | Role |
//! |------|------|------|
//! | `config.json` | 2.4 kB | Static architecture |
//! | `dsm_tts_1e68beda@240.safetensors` | 3.68 GB | Backbone + DepFormer weights |
//! | `tokenizer-e351c8d8-checkpoint125.safetensors` | 385 MB | Mimi codec sidecar (same as `kyutai/moshiko-*`) |
//! | `tokenizer_spm_8k_en_fr_audio.model` | 120 kB | SentencePiece text tokenizer (8k en/fr + audio) |
//!
//! ## Crate layout — all primitives are native Rust
//!
//! | Module | Native role |
//! |--------|-------------|
//! | [`nn`] | `linear`, `rms_norm`, `silu`, `swiglu_mlp`, `softmax_last_dim`, `rope_tables`, `apply_rope_vec`, `sin_pos_embed` |
//! | [`low_rank_embedding`] | Factorized codebook embedding `E ≈ A · B` (rank 128) |
//! | [`conditioner`] | [`LutConditioner`] (cfg, control), [`TensorConditioner`] (speaker_wavs) |
//! | [`fuser`] | `sum` / `prepend` / `cross` routing into the backbone |
//! | [`cross_attention`] | Multi-head cross-attn for speaker conditioning (Q from hidden, K/V from speaker context) |
//! | [`transformer`] | Streaming temporal backbone with optional cross-attn per layer, KV cache ring buffer |
//! | [`depformer`] | Per-step DepFormer (head selection via schedule, low-rank input embeddings) |
//! | [`sampling`] | Temperature + top-k [`LogitsProcessor`] + distilled [`CfgSampler`] |
//! | [`delays`] | [`StreamLayout`] — per-codebook delay padding + demuxed-stream offsets |
//! | [`weights`] | Native safetensors loader (F32 / F16 / BF16 → f32) + expected-key inventory |
//! | [`session`] | [`KyutaiTtsSession`] — high-level fetch / load / generate entry point |
//!
//! ## Features
//!
//! | Feature | Stack |
//! |---------|-------|
//! | `cli` (default) | CLI for `--fetch` / `--info` / `--prompt` |
//! | `hf-download` | HuggingFace weight + Mimi sidecar fetching |
//! | `metal` / `cuda` / `mlx` / `gpu` / `vulkan` / `rocm` | Device backends (forward to `rlx-runtime` + `rlx-mimi`) |
//!
//! The upstream Kyutai `moshi` 0.6.4 crate + Candle are wired in as
//! **dev-dependencies only** — used by parity / validation tests
//! (e.g. Whisper round-trip in `tests/whisper_validate.rs`) and never enter
//! the runtime dep graph.
//!
//! Generation state — `KyutaiTtsSession::generate` is currently a stub; every
//! architectural primitive above is implemented and unit-tested, but the
//! orchestration loop that composes them with loaded weights is not yet
//! wired. The crate already produces real audio through the Mimi codec
//! (see `examples/generate_wav.rs`); end-to-end native TTS is the next step.
//!
//! See [`KyutaiTtsConfig`] for the static architecture preset and
//! [`download::fetch_kyutai_tts`] for weight fetching.

pub mod checkpoint;
pub mod cli;
pub mod conditioner;
pub mod config;
pub mod cross_attention;
pub mod delays;
pub mod depformer;
pub mod device;
pub mod download;
pub mod fuser;
pub mod low_rank_embedding;
pub mod nn;
pub mod sampling;
pub mod session;
pub mod transformer;
pub mod weights;

pub use checkpoint::{KyutaiTtsCheckpoint, KyutaiTtsVoice};
pub use conditioner::{LutConditioner, TensorConditioner};
pub use config::{
    ConditionerConfig, ConditionerKind, DELAYS_DEFAULT, DEPFORMER_WEIGHTS_PER_STEP_SCHEDULE,
    DepFormerConfig, FuserConfig, KyutaiTtsConfig, PositionalEmbedding, TransformerConfig,
    TtsConfig,
};
pub use cross_attention::{CrossAttention, CrossKvCache};
pub use delays::StreamLayout;
pub use depformer::{DepFormer, DepFormerHead};
pub use fuser::{ConditionerOutputs, FusedConditioning, SumOffset, fuse};
pub use low_rank_embedding::LowRankEmbedding;
pub use nn::{
    Embedding, RMS_EPS, apply_rope_vec, linear, rms_norm, rope_tables, silu, sin_pos_embed,
    softmax_last_dim, swiglu_mlp,
};
pub use sampling::{CfgSampler, LogitsProcessor, argmax};
pub use transformer::{
    AttnWeights, LayerWeights, StreamingTransformer, TransformerConfig as BackboneConfig,
};
pub use weights::{
    WeightEntry, WeightMap, decode_tensor, expected_kyutai_tts_keys, load_weight_map, missing_keys,
    strip_prefix,
};
// The config-side spec types `config::LutConditioner` and
// `config::TensorConditioner` (Kyutai TTS `config.json` shape) are NOT
// re-exported because they collide with the runtime kernels
// `conditioner::LutConditioner` and `conditioner::TensorConditioner`. Access
// the config-side spec types via the `config` submodule, or destructure
// [`ConditionerKind`] variants.
pub use device::{device_ready, parse_kyutai_tts_device, test_devices};
pub use download::{
    HF_KYUTAI_TTS_REPO, MIMI_SIDECAR_FILE, SPM_TOKENIZER_FILE, TTS_WEIGHTS_FILE,
    default_kyutai_tts_dir, default_mimi_dir, ensure_weights, fetch_kyutai_tts,
    resolve_kyutai_tts_dir, tokenizer_path, tts_weights_path,
};
pub use session::{GenerationConfig, GenerationResult, KyutaiTtsSession};

#[cfg(feature = "cli")]
pub use cli::run;
