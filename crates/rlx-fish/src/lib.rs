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

//! # rlx-fish
//!
//! Fish Audio text-to-speech and voice cloning on RLX, native Rust:
//!
//! - **S2-Pro** ([`fishaudio/s2-pro`](https://huggingface.co/fishaudio/s2-pro)) —
//!   a dual-AR model. A Qwen3-4B-shaped *slow* backbone emits one semantic
//!   token per audio frame; a 4-layer *fast* transformer then fills that
//!   frame's nine residual codes one at a time.
//! - **Drama 3** ([`fishaudio/drama-3`](https://huggingface.co/fishaudio/drama-3),
//!   demo Space `fishaudio/Drama-3`) — a Qwen3.5-MoE hybrid backbone (Gated
//!   DeltaNet + gated attention, 256 experts) with a codebook *tower* that
//!   attends across frames.
//!
//! Both emit 10 codes per frame for the same codec ([`codec::FishCodec`]):
//! a causal 44.1 kHz DAC with a transformer-wrapped RVQ, 2048 samples per
//! frame.

pub mod audio;
pub mod codec;
pub mod config;
pub mod drama3;
#[cfg(feature = "export")]
pub mod drama3_export;
pub mod drama3_text;
#[cfg(feature = "export")]
pub mod export;
pub mod moe;
pub mod nn;
pub mod prompt;
pub mod s2;
pub mod sampling;
pub mod store;

pub use codec::FishCodec;
pub use config::S2Config;
pub use drama3::{DramaModel, DramaOptions};
pub use s2::{S2Model, S2Options};
pub use sampling::{Rng, SampleParams};
