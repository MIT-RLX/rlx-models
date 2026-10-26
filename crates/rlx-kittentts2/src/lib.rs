// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! KittenTTS 2 — [`KittenML/kitten-tts-2`](https://huggingface.co/KittenML/kitten-tts-2)
//! on RLX.
//!
//! A 1.7B Qwen3 speech language model with ternary linears reads text and
//! writes 25 Hz S3 codec tokens; ResembleAI's chatterbox-turbo S3Gen
//! (meanflow conformer + U-Net flow, two Euler steps) turns them into a mel
//! spectrogram and its HiFT vocoder into 24 kHz audio. A voice conditions both
//! halves: a speaker vector and the reference clip's own tokens prompt the LM,
//! the clip's mel and CAM++ x-vector prompt the flow.
//!
//! | piece | module |
//! |---|---|
//! | `config.json`, token layout, presets | [`config`] |
//! | built-in voices and their conditioning | [`voices`] |
//! | ternary checkpoint → `Q2_0` GGUF (lossless) | [`ternary`] |
//! | speech LM + `spk_proj` | [`lm`] |
//! | sampler (run / repetition penalties, top-k/p, min-p) | [`sampling`] |
//! | written → spoken English (NeMo grammars) | [`normalize`] |
//! | markup, clean-up, chunking, prompt, joining | [`text`] |
//! | S3Gen flow + HiFT vocoder | [`s3gen`] |
//! | voice cloning from a reference clip | [`reference`](mod@reference) |
//! | text → audio | [`pipeline`] |

pub mod config;
#[cfg(feature = "export")]
pub mod export;
pub mod lm;
pub mod normalize;
pub mod pipeline;
pub mod reference;
pub mod s3gen;
pub mod sampling;
pub mod ternary;
pub mod text;
pub mod voices;

pub use config::{DecodePreset, Kitten2Config, TokenMap};
pub use lm::{SpeakerProjection, SpeechLm, StopReason};
pub use pipeline::{GenerateOpts, KittenTts2, resolve_s3gen, write_wav};
pub use rlx_runtime::{Device, parse_device};
pub use sampling::{Rng, SamplerOpts};
pub use voices::{S3Conditioning, Voice, VoiceLibrary};

/// Output sample rate of the S3Gen vocoder.
pub const SAMPLE_RATE: u32 = 24_000;
