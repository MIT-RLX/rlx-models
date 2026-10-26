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

//! OpenBMB [MiniCPM-SALA](https://huggingface.co/openbmb/MiniCPM-SALA) for RLX:
//! a 9B hybrid of sparse attention (InfLLM-v2, 8 of 32 layers) and Lightning
//! linear attention (24 layers), with muP scaling.
//!
//! - [`config`] — `config.json` (`model_type = minicpm_sala`);
//! - [`graph`] — one compiled step graph for prefill chunks and decode;
//! - [`SalaRunner`] — the K/V cache and Lightning state around it.
//!
//! InfLLM-v2 is dense attention below `sparse_config.dense_len` keys (8192 in
//! the release), so contexts up to that length are exact here; longer ones are
//! refused rather than run dense, which would be a different model.

pub mod cli;
pub mod config;
pub mod graph;
pub mod runner;
pub mod sparse;

pub use config::{Mixer, SalaConfig};
pub use runner::{SalaRunner, SalaRunnerBuilder};

/// `<s><|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n`.
pub fn chat_prompt(question: &str) -> String {
    format!("<s><|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n")
}
