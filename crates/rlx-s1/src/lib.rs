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

//! **S1-mini** by Superwhisper ([superwhisper/s1-mini](https://huggingface.co/superwhisper/s1-mini))
//! — a 0.6 B text normalizer for speech-to-text output.
//!
//! It takes a raw ASR transcript and rewrites it as clean written text: fillers
//! removed, false starts and self-corrections resolved to whatever the speaker
//! landed on, punctuation and capitalization applied, and spoken numbers,
//! dates, times, currency and email addresses rendered in written form. It sits
//! after an ASR stage, not in front of a user:
//!
//! ```text
//! audio ──▶ ASR (Whisper, Parakeet, …) ──▶ S1-mini ──▶ clean text
//! ```
//!
//! # What this crate is
//!
//! S1-mini is a fine-tune of `Qwen/Qwen3-0.6B` and its `config.json` is
//! byte-identical to the base model's — same 28 layers, 16 Q / 8 KV heads,
//! hidden 1024, RoPE θ 1e6, tied embeddings. So there is **no new architecture
//! here**: the forward pass is [`rlx_qwen3`], and every backend that already
//! runs Qwen3 runs this (CPU, Metal, MLX, CUDA, ROCm, wgpu, Vulkan, CoreML).
//!
//! What this crate adds is the part that is easy to get wrong, and that the
//! model card spends most of its length on:
//!
//! * [`prompt::SYSTEM_PROMPT`] — the exact system prompt, verbatim. Re-wording
//!   it degrades the output.
//! * [`Controls`] — the `[Styling: …] [Structure: …] [Context: …]` control
//!   line, as three enums, so untrained axis values are unrepresentable.
//! * The `enable_thinking=False` assistant prefix (`<think>\n\n</think>\n\n`).
//!   Omit it and the model emits an empty think block and stops — the single
//!   most common way to get a blank result.
//! * Greedy decoding, pinned. `generation_config.json` ships `do_sample: false`.
//! * `max_new_tokens` sized per call as `1.3 × prompt + 32` instead of a flat
//!   1024, and stop on `<|im_end|>` / `<|endoftext|>`.
//! * Sentence-boundary chunking for transcripts past the ~1,000-token design
//!   point, with a word-boundary fallback (raw ASR often has no punctuation to
//!   break on).
//!
//! # Quickstart
//!
//! ```no_run
//! use rlx_s1::{Controls, S1Runner, Styling};
//!
//! let mut s1 = S1Runner::builder()
//!     .weights("/path/to/s1-mini")            // HF dir, .safetensors, or .gguf
//!     .build()?;
//!
//! let raw = "so um i need to like send the the report by uh friday no wait make that thursday";
//! assert_eq!(s1.normalize(raw)?, "I need to send the report by Thursday.");
//!
//! // Steer with the control line; every combination of the three axes was trained.
//! let casual = Controls::new().styling(Styling::Casual);
//! let _ = s1.normalize_with(raw, casual)?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! An empty result is a valid answer: filler-only input ("um") normalizes to
//! the empty string by design, and a pipeline should treat that as success.
//!
//! # CLI
//!
//! ```text
//! rlx-s1 --weights /path/to/s1-mini --transcript "so um send the report by uh friday"
//! cat notes.txt | rlx-s1 --weights /path/to/s1-mini --stdin --styling formal --context email
//! ```
//!
//! # License note
//!
//! S1-mini is Apache-2.0 (inherited from Qwen3-0.6B) plus one additional term:
//! wherever it is used it must keep its name, "S1-mini" by "Superwhisper", with
//! that exact capitalization. Read the upstream `LICENSE` before shipping it.

pub mod cli;
pub mod prompt;
pub mod runner;

pub use cli::cli_run;
pub use prompt::{
    Context, Controls, EMPTY_THINK_BLOCK, SYSTEM_PROMPT, Structure, Styling, chunk_transcript,
    recommended_max_new_tokens, render_prompt, render_user_turn, sentence_units, shared_prefix,
    word_units,
};
pub use runner::{
    DEFAULT_CHUNK_TOKENS, DEFAULT_PREFILL_BUCKET, EOS_TOKENS, REFERENCE_SHAPE, S1Runner,
    S1RunnerBuilder, S1Shape, decode_ids, encode_ids, join_chunk_outputs,
};

/// Re-exported so callers can pick a device / precision without also depending
/// on `rlx-qwen3` and `rlx-runtime` directly.
pub use rlx_qwen3::{Precision, Qwen3Config};
pub use rlx_runtime::Device;

/// One-import surface: `use rlx_s1::prelude::*;`
pub mod prelude {
    pub use crate::prompt::{Context, Controls, Structure, Styling};
    pub use crate::runner::{S1Runner, S1RunnerBuilder};
    pub use rlx_runtime::Device;
}
