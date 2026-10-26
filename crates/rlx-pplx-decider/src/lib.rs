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

//! Perplexity **pplx-decider** classifiers
//! ([`perplexity-ai/pplx-decider-flash-gguf-v1.2`](https://huggingface.co/perplexity-ai/pplx-decider-flash-gguf-v1.2)).
//!
//! A decider answers typed questions about a *state* (text or JSON) in one
//! forward pass with no generated tokens: `choice` (one of up to 255 named
//! options), `noul` (P(true)) and `score` (expected level on an ordered
//! scale), each with calibrated probabilities. Every question of a request
//! shares one sequence (protocol v3, "joint-question-inline-readout-v1").
//!
//! The backbone is Qwen3.5-4B with **non-causal** full-attention layers
//! (`qwen35.attention.causal = false`); the Gated DeltaNet layers stay
//! causal. It runs through `rlx-qwen35` with
//! [`rlx_qwen35::Qwen35Flow::bidirectional_attention`].
//!
//! * [`protocol`] — request schema, prompt and answers, ported from the
//!   checkpoint's `pplx_decider/protocol.py`;
//! * [`model::PplxDecider`] — the engine.
//!
//! ```no_run
//! use rlx_pplx_decider::{Json, PplxDecider};
//! # fn main() -> anyhow::Result<()> {
//! let mut d = PplxDecider::open("decider-flash-v1.2-q8_0.gguf", rlx_runtime::Device::Cpu)?;
//! let req = d.request(&Json::parse(r#"{"state": "I was charged twice.",
//!     "questions": {"team": {"type": "choice",
//!         "criteria": {"billing": null, "support": null}}}}"#)?)?;
//! println!("{}", d.decide(&req, None)?.to_python_json(Some(2)));
//! # Ok(()) }
//! ```

pub mod cli;
#[cfg(feature = "export")]
pub mod export;
pub mod model;
pub mod protocol;
pub mod vision;

pub use model::{
    DeciderMeta, Encoded, OpenOptions, Plan, PplxDecider, QuestionLogits, gguf_tokenizer,
    resolve_gguf,
};
pub use protocol::{Kind, Question, Request, RequestError};
pub use rlx_core::ordered_json::Json;
pub use vision::{Preprocessed, RgbImage};
