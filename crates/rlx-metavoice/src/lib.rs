// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! MetaVoice-1B — zero-shot voice-cloning TTS for RLX (Apache-2.0).
//!
//! Architecture (from [`metavoiceio/metavoice-src`](https://github.com/metavoiceio/metavoice-src)):
//! 1. **Speaker encoder** (3-layer LSTM) → 256-d embedding from reference audio
//! 2. **First-stage GPT** (24×2048, RMSNorm + SwiGLU, vocab 2562) — interleaved
//!    text + EnCodec codebook-0/1 tokens with speaker conditioning + CFG
//! 3. **Second-stage** (6×384 non-causal) — fills EnCodec codebooks 2–7
//! 4. **EnCodec 24 kHz** decoder (via `rlx-encodec`) → PCM
//!
//! Weights live under `weights/tts/metavoice/` (`*.safetensors` converted from the
//! HF `.pt` checkpoints). First/second-stage run eagerly on CPU with optional
//! speaker LSTM conditioning from a reference wav; EnCodec decode follows the
//! requested RLX [`Device`].

pub mod config;
pub mod first_stage;
pub mod native;
pub mod second_stage;
pub mod speaker;
pub mod tokenize;

pub use config::{
    DEFAULT_HF_REPO, DEFAULT_LOCAL_DIR, FirstStageArgs, MetaVoiceConfig, SAMPLE_RATE,
    SecondStageArgs,
};
pub use first_stage::{EOS_AUDIO, extract_codebooks};
pub use native::{
    DEFAULT_ENCODEC_PATH, DEFAULT_REFERENCE, FOX_WORDS, InferOpts, MetaVoice, normalize_text,
    postprocess_pcm,
};
pub use rlx_runtime::{Device, parse_device};
pub use second_stage::{PAD as SECOND_PAD, TEXT_OFFSET as SECOND_TEXT_OFFSET};
pub use speaker::SpeakerEncoder;
pub use tokenize::MetaTokenizer;

/// Peak absolute amplitude (audibility check).
pub fn peak_amplitude(audio: &[f32]) -> f32 {
    audio.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
}
