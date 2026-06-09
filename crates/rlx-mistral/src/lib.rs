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

//! Mistral 3+ / Ministral runner.
//!
//! Mistral 3 / 3.5 / 4 and Ministral 3 ship as `general.architecture =
//! mistral3` or `mistral4` in their GGUF converters — Llama-shaped with
//! per-arch deltas. This crate is a thin wrapper over
//! [`rlx_llama32::Llama32Runner`] with arch validation.
//!
//! **Caveat:** The underlying `rlx-llama32` builder doesn't yet apply
//! Mistral 3's per-layer sliding-window mask or Mistral-specific RoPE
//! base — runs will produce *some* tokens but won't match the upstream
//! reference until those land in `rlx-llama32`. PLAN.md M4 follow-up.

use anyhow::{Context, Result, bail};
use rlx_llama_base::LlamaBaseConfig;
use std::path::{Path, PathBuf};

pub use rlx_llama32::{Llama32ConfigSource, Llama32Runner, Llama32RunnerBuilder};

pub const PLAN_MILESTONE: &str = "M4";
pub const FAMILY: &str = "Mistral 3+ / Ministral";

const ACCEPTED_ARCHES: &[&str] = &["mistral3", "mistral4"];

pub struct MistralRunner {
    inner: Llama32Runner,
    config: LlamaBaseConfig,
}

impl MistralRunner {
    pub fn builder() -> MistralRunnerBuilder {
        MistralRunnerBuilder::default()
    }
    pub fn config(&self) -> &LlamaBaseConfig {
        &self.config
    }
    pub fn inner(&self) -> &Llama32Runner {
        &self.inner
    }
    pub fn inner_mut(&mut self) -> &mut Llama32Runner {
        &mut self.inner
    }
    pub fn generate_packed(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        self.inner.generate_packed(prompt_ids, n_new, on_token)
    }
}

#[derive(Debug, Clone, Default)]
pub struct MistralRunnerBuilder {
    weights: Option<PathBuf>,
    inner: Llama32RunnerBuilder,
}

impl MistralRunnerBuilder {
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
    pub fn build(self) -> Result<MistralRunner> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weights path required"))?
            .clone();
        let config = LlamaBaseConfig::from_gguf_path(&weights)
            .with_context(|| format!("rlx-mistral: parse {weights:?}"))?;
        if !ACCEPTED_ARCHES.contains(&config.arch.as_str()) {
            bail!(
                "rlx-mistral: expected `general.architecture` ∈ {ACCEPTED_ARCHES:?}; \
                 got `{}` at {weights:?} (Mistral 1/2 ship as `llama` — use rlx-llama32 directly)",
                config.arch
            );
        }
        let inner = self
            .inner
            .build()
            .context("rlx-mistral: building underlying Llama32Runner")?;
        Ok(MistralRunner { inner, config })
    }
}

pub fn cli_run(args: &[String]) -> Result<()> {
    if let Some(first) = args.iter().position(|a| a == "--weights") {
        if let Some(path) = args.get(first + 1) {
            let cfg = LlamaBaseConfig::from_gguf_path(Path::new(path))
                .with_context(|| format!("rlx-mistral: parse {path}"))?;
            if !ACCEPTED_ARCHES.contains(&cfg.arch.as_str()) {
                bail!(
                    "rlx-mistral: {path}: GGUF arch = `{}`, expected one of {ACCEPTED_ARCHES:?}",
                    cfg.arch
                );
            }
        }
    }
    rlx_llama32::cli::run(args)
}
