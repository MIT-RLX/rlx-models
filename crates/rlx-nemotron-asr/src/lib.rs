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

//! NVIDIA **Nemotron 3.5 ASR Streaming 0.6B** — a cache-aware
//! FastConformer encoder with a prompt-conditioned RNN-T decoder — running
//! on RLX, loaded natively from the distributed `.nemo` via [`rlx_nemo`].
//!
//! Pipeline: log-mel frontend ([`mel`]) → FastConformer encoder graph
//! ([`encoder`]) → host-side RNN-T greedy decode ([`decoder`]) → text.

pub mod cli;
pub mod config;
pub mod decoder;
pub mod encoder;
pub mod mel;
pub mod runner;
pub mod tokenizer;
pub mod wav;
pub mod weights;

pub use config::AsrConfig;
pub use runner::NemotronAsr;

/// Model family identifier for CLI/registry dispatch.
pub const FAMILY: &str = "nemotron-asr";
