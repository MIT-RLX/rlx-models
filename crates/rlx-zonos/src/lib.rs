// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Zonos v0.1 transformer TTS (Apache-2.0) — Zyphra.
//!
//! Pipeline:
//! ```text
//! text → espeak IPA → PrefixConditioner → compiled 26×2048 GQA AR (delay-pattern DAC)
//!      → rlx-dac decode → PCM 44.1 kHz
//! ```
//!
//! `--device` selects both the compiled backbone and DAC decode backend.
//! Eager host fallback: `RLX_ZONOS_EAGER=1`.
//!
//! ```bash
//! just fetch-zonos
//! just fetch-parler-dac
//! just zonos-demo
//! just zonos-backends
//! ```

pub mod backbone;
pub mod compile_opts;
pub mod conditioner;
pub mod config;
pub mod delay;
pub mod engine;
pub mod flow;
pub mod generate;
pub mod native;
pub mod ops;
pub mod phonemes;
pub mod weights;

pub use config::{
    DEFAULT_DAC_DIR, DEFAULT_HF_REPO, DEFAULT_LOCAL_DIR, EOS_TOKEN_ID, MASKED_TOKEN_ID,
    N_CODEBOOKS, SAMPLE_RATE, ZonosFileConfig,
};
pub use engine::suggest_max_tokens;
pub use native::{InferOpts, NativeZonos, load_speaker_emb, peak_amplitude};
pub use rlx_runtime::{Device, parse_device};
