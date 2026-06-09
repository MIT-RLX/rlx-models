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

//! MiniMax M2.5 / M2.7 runner — Lightning Attention LM.
//!
//! Provides:
//!   * [`MiniMaxConfig`] — GGUF + HF config parsing.
//!   * [`minimax_decode_layer_plugin`] — per-layer decode block
//!     emitting `LightningAttentionStepStage` + state-out side output.
//!
//! Still pending in this crate:
//!   * `MiniMaxRunner` that allocates per-layer state buffers, binds
//!     them as model inputs each step, and reads back the
//!     `minimax.state_out_{layer}` side outputs into the buffers.
//!   * `MiniMaxWeights` loader for embedding + LM head + per-layer
//!     projections via the existing GGUF/safetensors infrastructure.

use anyhow::{Context, Result, bail};
use rlx_llama_base::LlamaBaseConfig;
use std::path::Path;

pub mod config;
pub mod flow;
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
