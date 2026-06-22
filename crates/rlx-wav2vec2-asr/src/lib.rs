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

//! Classic Wav2Vec2 + CTC forced alignment (WhisperX-style).
//!
//! Separate from [`rlx-wav2vec2-bert`] (Conformer W2V-BERT). This crate provides
//! phoneme/frame alignment for transcript text against 16 kHz PCM.

pub mod align;
pub mod config;
pub mod registry;

pub use align::{AlignSession, AlignedWord};
pub use config::Wav2Vec2AsrConfig;
pub use registry::align_model_for_language;
