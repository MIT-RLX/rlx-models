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

//! Llama-4 (`meta-llama/Llama-4-Scout/Maverick`) for RLX.
//!
//! Text tower: MoE decoder (top-1 experts + shared expert) with **iRoPE** —
//! RoPE layers do chunked-window attention, periodic NoPE layers do full
//! attention with temperature-tuned scaling (a no-op below the chunk size).
//! Vision is early-fusion (embed-splice), added later.
//!
//! Modules present: [`config`], [`moe`]. Attention + full text flow + runner
//! follow.

pub mod attention;
pub mod cli;
pub mod config;
pub mod flow;
pub mod moe;
pub mod preprocess;
pub mod rope;
pub mod runner;
pub mod vision;
pub mod vl_runner;

pub use attention::{AttnDims, emit_attention};
pub use config::{Llama4TextConfig, Llama4VisionConfig};
pub use flow::build_llama4_text_flow;
pub use moe::emit_moe_ffn;
pub use runner::Llama4Runner;
pub use vision::build_llama4_vision_flow;
pub use vl_runner::Llama4VlRunner;
