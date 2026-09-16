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
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! HumeAI **TADA** — Text-Acoustic Dual Alignment — as native rlx graphs.
//!
//! TADA is a zero-shot voice cloner built on a Llama-3.2 backbone. What makes
//! it unusual is that text and audio ride the *same* autoregressive stream at
//! 1:1: a forced aligner assigns every text token exactly one 50 Hz frame, and
//! at each step the model emits both the next token's acoustic latent and how
//! many frames that token should occupy. The latent is not sampled from a
//! codebook — a small DiT-style head runs a flow-matching ODE conditioned on
//! the backbone's hidden state, so the acoustic stream stays continuous.
//!
//! ```text
//!   reference wav ─┬─► aligner (wav2vec2-CTC) ─► token↔frame assignment ─┐
//!                  └─► codec encoder ──────────► 50 Hz latents ──────────┤
//!                                                                        ▼
//!   target text ──► Llama-3.2 backbone ──► hidden ──► flow-matching head ──►
//!                        ▲                                    │
//!                        └──────── acoustic feedback ◄─────────┘
//!                                                              │
//!                                     latents + durations ─────┴─► codec
//!                                                        decoder ─► 24 kHz wav
//! ```
//!
//! Checkpoints: `HumeAI/tada-1b` (English) and `HumeAI/tada-3b-ml`
//! (multilingual) for the backbone, `HumeAI/tada-codec` for the shared codec
//! and the per-language aligners. The `tada-1b` file bundles the codec decoder
//! under a `_decoder.` prefix, so synthesis from a cached voice prompt needs
//! that one file and nothing else.
//!
//! Every stage compiles to rlx HIR and runs on any rlx backend. The
//! convolutional codec stacks are DAC's, so they reuse `rlx-dac`'s graph
//! builders rather than re-deriving them here; the backbone reuses
//! `rlx-llama32`'s flow with an `inputs_embeds` entry point.

pub mod align;
pub mod aligner;
pub mod backbone;
pub mod builder;
pub mod codec;
pub mod config;
pub mod gray;
pub mod head;
pub mod local_attn;
pub mod mask;
pub mod model;
pub mod prof;
pub mod prompt;
pub mod prompt_builder;
pub mod resample;
pub mod rng;
pub mod synth;
pub mod text;
pub mod tokenizer;
pub mod weights;

pub use prof::rss_mb;

pub use align::{Alignment, align_tokens};
pub use config::{
    ALIGNER_SAMPLE_RATE, AlignerConfig, DecoderConfig, EncoderConfig, FRAME_RATE, SAMPLE_RATE,
    TadaConfig,
};
pub use text::normalize_text;
pub use weights::TensorStore;
