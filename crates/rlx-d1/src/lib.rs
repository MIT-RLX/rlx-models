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

//! LiquidAI **d1** decision models ([`LiquidAI/d1-3B`](https://huggingface.co/LiquidAI/d1-3B)).
//!
//! d1-3B is LFM2.5-VL-3B post-trained for *System One* decisions: a state
//! (text, JSON, pictures, or a mix) and typed questions in, calibrated
//! answers out, in one forward pass with zero output tokens. Each answer is a
//! softmax over the logits of its options' tokens at the answer slot.
//!
//! * [`prompt`] — the prompt and the readout, ported from d1's `prompt.py`;
//! * [`vision`] — LFM2-VL's image processor, SigLIP2 NaFlex tower and
//!   pixel-unshuffle projector;
//! * [`model::D1`] — the engine (`system_one`, [`model::D1::decide`]).
//!
//! The checkpoint is a stock LFM2-VL one: the text model runs through
//! [`rlx_lfm::Lfm2GgufRunner`], which reads LFM2-VL directories directly.

pub mod cli;
#[cfg(feature = "export")]
pub mod export;
pub mod model;
pub mod processor;
pub mod prompt;
pub mod vision;

pub use model::{D1, Decision, HfTokenizer, OpenOptions};
pub use prompt::{PromptStyle, Question, TextTokenizer, answer};
pub use rlx_core::ordered_json::Json;
pub use rlx_core::weight_storage::WeightStorage;
pub use vision::RgbImage;
