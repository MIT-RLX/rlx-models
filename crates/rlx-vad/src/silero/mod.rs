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

//! Silero VAD on RLX (embedded `silero_vad_16k.safetensors`, no external files).
//!
//! Architecture reference: <https://github.com/snakers4/silero-vad>

mod embedded;
mod model;
mod session;
mod weights;

pub use model::forward_frame;
pub use session::{SileroConfig, SileroSession};
pub use weights::SileroWeights;

pub const FRAME_SAMPLES_16K: usize = 512;
pub const FRAME_SAMPLES_8K: usize = 256;
pub const CONTEXT_16K: usize = 64;
pub const CONTEXT_8K: usize = 32;
