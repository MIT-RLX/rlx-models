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

//! TrOCR — Microsoft's transformer text recognizer — natively in RLX.
//!
//! A `VisionEncoderDecoderModel`: a ViT (base, large) or DeiT (small)
//! encoder over a 384×384 line image and a post-norm transformer decoder
//! that writes the text autoregressively. `microsoft/trocr-{small,base,
//! large}-handwritten` are trained on IAM (English); community fine-tunes
//! of the same architecture cover other languages.
//!
//! - [`TrOcrConfig`] — `config.json` + `preprocessor_config.json`.
//! - [`graph`] — the encoder (with every layer's cross-attention K/V) and a
//!   KV-cached one-token decoder step.
//! - [`TrOcrRunner`] — image → text, with [`TrOcrRunner::encode`] /
//!   [`TrOcrRunner::step`] as the parity entry points.

/// Command-line entry point (`rlx-trocr`).
pub mod cli;
/// Model configuration.
pub mod config;
/// Encoder and decoder-step graphs.
pub mod graph;
/// Image → text.
pub mod runner;
/// Decode-only tokenizers.
pub mod tokenizer;
/// Checkpoint loading.
pub mod weights;

pub use config::TrOcrConfig;
pub use runner::{DEFAULT_MAX_TOKENS, Recognition, TrOcrRunner};
pub use tokenizer::Tokenizer;
