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

//! LiquidAI LFM2.5 runner — text variants.
//!
//! Provides:
//!   * [`LfmConfig`] — GGUF + HF config parsing.
//!   * [`lfm_decode_layer_plugin`] — per-layer decode block emitting
//!     `LfmSsmStepStage` + state-out side output.
//!
//! Still pending: `LfmRunner` with state-buffer binding across decode
//! calls (mirrors `MiniMaxRunner` follow-up).

use anyhow::{Context, Result, bail};
use rlx_llama_base::LlamaBaseConfig;
use std::path::Path;

pub mod config;
pub mod flow;
pub mod runner;

pub use config::LfmConfig;
pub use flow::{lfm_decode_layer_plugin, lfm_decode_layer_plugin_with_sink};
pub use runner::{LfmRunner, LfmRunnerBuilder};

pub const PLAN_MILESTONE: &str = "M5";
pub const FAMILY: &str = "LFM2.5 (text)";

const ACCEPTED_ARCHES: &[&str] = &["lfm2", "lfm", "lfm25", "lfm2_5", "lfm2moe"];

// Runner now lives in `runner` module — see `LfmRunner`.

pub fn cli_run(args: &[String]) -> Result<()> {
    if let Some(first) = args.iter().position(|a| a == "--weights") {
        if let Some(path) = args.get(first + 1) {
            let cfg = LlamaBaseConfig::from_gguf_path(Path::new(path))
                .with_context(|| format!("rlx-lfm: parse {path}"))?;
            if !ACCEPTED_ARCHES.contains(&cfg.arch.as_str()) {
                bail!(
                    "rlx-lfm: {path}: GGUF arch = `{}`, expected one of {ACCEPTED_ARCHES:?}",
                    cfg.arch
                );
            }
        }
    }
    bail!("rlx-lfm: runner-level state plumbing still TODO")
}
