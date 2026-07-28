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

//! MiniMax runners for RLX.
//!
//! Two unrelated MiniMax architectures live here:
//!
//! * **M2.5 / M2.7** — a Lightning-Attention (linear-attention) LM. Top-level
//!   [`MiniMaxConfig`] + [`minimax_decode_layer_plugin`] emit the per-layer
//!   `LightningAttentionStepStage` + state-out side output, driven by
//!   [`MiniMaxRunner`].
//!
//! * **M3 (MSA — MiniMax Sparse Attention)** — a natively-multimodal
//!   128-expert MoE (`minimax_m3_vl` / `MiniMaxM3SparseForCausalLM`). See the
//!   [`m3`] module: config, text decoder (GQA + per-head Gemma QK-norm + partial
//!   RoPE + SwiGLU-OAI MoE + block-sparse MSA), CLIP-style vision tower +
//!   projector, weights loader, prefill and KV-cache decode runners
//!   ([`m3::MiniMaxM3Runner`]), and the VL runner ([`m3::MiniMaxM3VlRunner`]).

use anyhow::{Context, Result, bail};
use rlx_llama_base::LlamaBaseConfig;
use std::path::Path;

pub mod config;
pub mod flow;
pub mod m3;
pub mod runner;

pub use config::MiniMaxConfig;
pub use flow::{minimax_decode_layer_plugin, minimax_decode_layer_plugin_with_sink};
pub use runner::{MiniMaxRunner, MiniMaxRunnerBuilder};

pub const PLAN_MILESTONE: &str = "M5";
pub const FAMILY: &str = "MiniMax";

const ACCEPTED_ARCHES: &[&str] = &["minimax-m2", "minimax_m2", "minimax"];

// Runner now lives in `runner` module — see `MiniMaxRunner`.

pub fn cli_run(args: &[String]) -> Result<()> {
    if let Some(first) = args.iter().position(|a| a == "--weights") {
        if let Some(path) = args.get(first + 1) {
            let cfg = LlamaBaseConfig::from_gguf_path(Path::new(path))
                .with_context(|| format!("rlx-minimax: parse {path}"))?;
            if !ACCEPTED_ARCHES.contains(&cfg.arch.as_str()) {
                bail!(
                    "rlx-minimax: {path}: GGUF arch = `{}`, expected one of {ACCEPTED_ARCHES:?}",
                    cfg.arch
                );
            }
        }
    }
    bail!("rlx-minimax: runner-level state plumbing still TODO")
}
