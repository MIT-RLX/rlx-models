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

//! EAGLE3 speculative decoding for RLX.
//!
//! # What's in this crate
//!
//! | Module | Status | Notes |
//! |---|---|---|
//! | [`config`] | ✅ tested | Parses RedHatAI / vLLM-speculators `Eagle3SpeculatorConfig`. |
//! | [`d2t`] | ✅ tested | Draft-vocab → target-vocab scatter (LUT-based). |
//! | [`weights`] | ✅ tested on synthetic | Reads draft `model.safetensors` and surfaces named tensors. |
//! | [`speculator`] | ⚠️ scaffold | `Eagle3Speculator<H>` implements `Speculator`, but its `propose()` requires the draft graph from [`draft`] to be wired. |
//! | [`draft`] | ⚠️ not yet built | The HIR graph builder for the 1-layer Llama draft + fc fusion + lm_head over draft vocab. See module docs for the spec we need to match against `speculators/core.py`. |
//!
//! # The architecture (per RedHatAI/gemma-4-31B-it-speculator.eagle3 + vLLM docs)
//!
//! The verifier emits hidden states from `eagle_aux_hidden_state_layer_ids`
//! (typically 3 layers, low/mid/high). For each speculation round:
//!
//! ```text
//!   h_aux   = concat(h_low, h_mid, h_high)         # [B, 1, 3*H_target]
//!   x_fused = fc(h_aux)                            # [B, 1, H_draft]   — Linear, no bias
//!   x       = concat(x_fused, embed(next_token))   # [B, 1, 2*H_draft]
//!   y       = draft_decoder_layer(x)               # 1-layer Llama, accepts 2*H_draft
//!   logits  = lm_head(verifier_norm(y))            # [B, 1, draft_vocab]
//! ```
//!
//! After sampling from `logits`, [`d2t`] scatters the draft token into the
//! target vocab (RedHatAI/Gemma 4: 32K → 262K).
//!
//! # Status & caveats
//!
//! Real-weight verification (RedHatAI/gemma-4-31B-it-speculator.eagle3 vs
//! llama.cpp b9606) is gated by:
//!
//! 1. The `draft` module's HIR builder being grounded in the actual
//!    `speculators/core.py` source so the modified-decoder-layer input
//!    handling is bit-exact (specifically: whether `input_layernorm`
//!    operates on the full 2H or only the embed half).
//! 2. A llama-cpp-2 release pinned at or past b9606 (workspace is on
//!    0.1.146; pre-b9606).
//! 3. The verifier-side runner exposing aux hidden states per
//!    speculation step (see `rlx_gemma::flow::GemmaFlow::with_aux_hidden_outputs`).

pub mod config;
pub mod d2t;
pub mod draft;
pub mod hir_draft;
pub mod hir_runner;
pub mod reference;
pub mod speculator;
pub mod weights;

// ── Per-verifier integration modules ───────────────────────────────
// These pull in heavier model-side deps and only compile when the
// relevant verifier feature is enabled. The bridge primitives
// themselves (callback-based `VerifierHiddenSource` adapter +
// `AuxStateBuffer`) are generic — gated under `gemma` because that's
// the first verifier we target, but they work for any model.
#[cfg(feature = "gemma")]
pub mod gemma_bridge;

pub use config::{Eagle3Config, Eagle3DraftTransformerConfig};
pub use d2t::D2tMap;
pub use speculator::{Eagle3Speculator, VerifierHiddenSource};
pub use weights::{DraftTensor, Eagle3DraftWeights};
