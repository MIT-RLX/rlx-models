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

//! Bonsai small-reasoning family runner.
//!
//! Bonsai (1.7B / 2B / 4B / 8B) ships as `general.architecture = llama`
//! in its GGUF converters — it's a standard Llama-shaped decoder with
//! tuned hyperparameters for small-context reasoning. This crate is a
//! thin wrapper over [`rlx_llama32::Llama32Runner`] that:
//!
//! * validates the GGUF actually has `arch = llama` (catches users
//!   pointing at the wrong file);
//! * gives `rlx-run bonsai` its own subcommand for discoverability;
//! * keeps the door open for Bonsai-specific tuning later (e.g.
//!   per-family compile profile, eval prompts) without changing the
//!   user-facing API.

use anyhow::{Context, Result, bail};
use rlx_llama_base::LlamaBaseConfig;
use std::path::{Path, PathBuf};

pub use rlx_llama32::{Llama32ConfigSource, Llama32Runner, Llama32RunnerBuilder};

pub const PLAN_MILESTONE: &str = "M4";
pub const FAMILY: &str = "Bonsai";

/// Per-family runner. Wraps [`Llama32Runner`] with a Bonsai-specific
/// arch validation so misrouted GGUFs fail loudly at `build()` instead
/// of silently producing garbage.
pub struct BonsaiRunner {
    inner: Llama32Runner,
    config: LlamaBaseConfig,
}

impl BonsaiRunner {
    pub fn builder() -> BonsaiRunnerBuilder {
        BonsaiRunnerBuilder::default()
    }

    /// Borrow the parsed LLaMA-base config (dims, RoPE, GQA, etc.).
    pub fn config(&self) -> &LlamaBaseConfig {
        &self.config
    }

    /// Read-only access to the underlying Llama-3.2 runner for advanced
    /// callers that need to drive prefill/decode directly.
    pub fn inner(&self) -> &Llama32Runner {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut Llama32Runner {
        &mut self.inner
    }

    /// Packed-decode generation. Mirrors
    /// [`Llama32Runner::generate_packed`] so callers don't need to
    /// reach through `.inner_mut()` for the common case.
    pub fn generate_packed(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        self.inner.generate_packed(prompt_ids, n_new, on_token)
    }
}

/// Builder. Same surface as [`Llama32RunnerBuilder`] but builds a
/// [`BonsaiRunner`] and enforces the arch tag.
#[derive(Debug, Clone, Default)]
pub struct BonsaiRunnerBuilder {
    weights: Option<PathBuf>,
    inner: Llama32RunnerBuilder,
}

impl BonsaiRunnerBuilder {
    pub fn weights(mut self, path: impl Into<PathBuf>) -> Self {
        let p: PathBuf = path.into();
        self.weights = Some(p.clone());
        self.inner = self.inner.weights(p);
        self
    }

    pub fn max_seq(mut self, n: usize) -> Self {
        self.inner = self.inner.max_seq(n);
        self
    }

    pub fn packed_weights(mut self, on: bool) -> Self {
        self.inner = self.inner.packed_weights(on);
        self
    }

    /// Build the runner. Returns `Err` when the GGUF doesn't carry
    /// `general.architecture = llama` (Bonsai's canonical arch tag).
    pub fn build(self) -> Result<BonsaiRunner> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weights path required (call .weights(...))"))?
            .clone();
        let config = LlamaBaseConfig::from_gguf_path(&weights)
            .with_context(|| format!("rlx-bonsai: parse {weights:?}"))?;
        if config.arch != "llama" {
            bail!(
                "rlx-bonsai: expected `general.architecture = llama` (Bonsai's GGUF tag); \
                 got `{}` at {weights:?}",
                config.arch
            );
        }
        let inner = self
            .inner
            .build()
            .context("rlx-bonsai: building underlying Llama32Runner")?;
        Ok(BonsaiRunner { inner, config })
    }
}

/// CLI entry point — delegates straight to `rlx_llama32::cli::run` so
/// every Llama-shaped flag works (`--weights`, `--tokenizer`,
/// `--prompt`, `--packed`, …).
pub fn cli_run(args: &[String]) -> Result<()> {
    if let Some(first) = args.iter().position(|a| a == "--weights") {
        if let Some(path) = args.get(first + 1) {
            let cfg = LlamaBaseConfig::from_gguf_path(Path::new(path))
                .with_context(|| format!("rlx-bonsai: parse {path}"))?;
            if cfg.arch != "llama" {
                bail!(
                    "rlx-bonsai: {path}: GGUF arch = `{}`, expected `llama` (Bonsai)",
                    cfg.arch
                );
            }
        }
    }
    rlx_llama32::cli::run(args)
}
