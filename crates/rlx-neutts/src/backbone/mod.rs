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

//! GGUF text backbone for NeuTTS.
//!
//! | Feature              | Implementation                          |
//! |----------------------|-----------------------------------------|
//! | `llama` (default)    | [`rlx::BackboneModel`] — `rlx-llama32`   |
//! | `parity-llama-cpp`   | [`llama_cpp::LlamaCppBackbone`] — ref   |

#[cfg(feature = "llama")]
mod rlx;

#[cfg(feature = "parity-llama-cpp")]
pub mod llama_cpp;

#[cfg(feature = "llama")]
pub use rlx::{BackboneModel, DEFAULT_N_CTX};
