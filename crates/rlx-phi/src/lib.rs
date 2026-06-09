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

//! Phi 3 / Phi 4 runner.
//!
//! Phi-3 and Phi-4 ship as `general.architecture = phi3` in their GGUF
//! converters (Phi-4 reuses the Phi-3 arch tag upstream — there's no
//! separate `phi4` enum in llama.cpp). This crate is a thin wrapper
//! over [`rlx_llama32::Llama32Runner`] with arch validation.
//!
//! **Caveat:** Phi-3's per-layer LayerNorm placement and partial-RoPE
//! split aren't yet implemented in `rlx-llama32` — runs will produce
//! *some* tokens but won't match the upstream reference until those
//! land. PLAN.md M4 follow-up.

use anyhow::{Context, Result, bail};
use rlx_llama_base::LlamaBaseConfig;
use std::path::{Path, PathBuf};

pub use rlx_llama32::{Llama32ConfigSource, Llama32Runner, Llama32RunnerBuilder};

pub const PLAN_MILESTONE: &str = "M4";
pub const FAMILY: &str = "Phi 3 / Phi 4";

const ACCEPTED_ARCHES: &[&str] = &["phi3"];

pub struct PhiRunner {
    inner: Llama32Runner,
    config: LlamaBaseConfig,
}

impl PhiRunner {
    pub fn builder() -> PhiRunnerBuilder {
        PhiRunnerBuilder::default()
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
pub struct PhiRunnerBuilder {
    weights: Option<PathBuf>,
    inner: Llama32RunnerBuilder,
}

impl PhiRunnerBuilder {
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
    pub fn build(self) -> Result<PhiRunner> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weights path required"))?
            .clone();
        let config = LlamaBaseConfig::from_gguf_path(&weights)
            .with_context(|| format!("rlx-phi: parse {weights:?}"))?;
        if !ACCEPTED_ARCHES.contains(&config.arch.as_str()) {
            bail!(
                "rlx-phi: expected `general.architecture` ∈ {ACCEPTED_ARCHES:?}; got `{}` at {weights:?}",
                config.arch
            );
        }
        let inner = self
            .inner
            .build()
            .context("rlx-phi: building underlying Llama32Runner")?;
        Ok(PhiRunner { inner, config })
    }
}

pub fn cli_run(args: &[String]) -> Result<()> {
    if let Some(first) = args.iter().position(|a| a == "--weights") {
        if let Some(path) = args.get(first + 1) {
            let cfg = LlamaBaseConfig::from_gguf_path(Path::new(path))
                .with_context(|| format!("rlx-phi: parse {path}"))?;
            if !ACCEPTED_ARCHES.contains(&cfg.arch.as_str()) {
                bail!(
                    "rlx-phi: {path}: GGUF arch = `{}`, expected one of {ACCEPTED_ARCHES:?}",
                    cfg.arch
                );
            }
        }
    }
    rlx_llama32::cli::run(args)
}
