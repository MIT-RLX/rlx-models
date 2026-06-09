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

//! Granite (IBM) runner.
//!
//! Granite ships as `general.architecture = granite` / `granitemoe` /
//! `granitehybrid` in its GGUF converters — Llama-shaped with
//! attention-scale and embedding-scale multipliers. This crate is a
//! thin wrapper over [`rlx_llama32::Llama32Runner`] with arch
//! validation.
//!
//! **Caveat:** Granite's `attention.scale` and `embedding.scale` keys
//! aren't yet read or applied in `rlx-llama32` — runs will produce
//! *some* tokens but won't match the upstream reference until those
//! land. `granitemoe`/`granitehybrid` additionally need MoE / hybrid
//! wiring (PLAN.md M4 + M5).

use anyhow::{Context, Result, bail};
use rlx_llama_base::LlamaBaseConfig;
use std::path::{Path, PathBuf};

pub use rlx_llama32::{Llama32ConfigSource, Llama32Runner, Llama32RunnerBuilder};

pub const PLAN_MILESTONE: &str = "M4";
pub const FAMILY: &str = "Granite (IBM)";

const ACCEPTED_ARCHES: &[&str] = &["granite", "granitemoe", "granitehybrid"];

pub struct GraniteRunner {
    inner: Llama32Runner,
    config: LlamaBaseConfig,
}

impl GraniteRunner {
    pub fn builder() -> GraniteRunnerBuilder {
        GraniteRunnerBuilder::default()
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
pub struct GraniteRunnerBuilder {
    weights: Option<PathBuf>,
    inner: Llama32RunnerBuilder,
}

impl GraniteRunnerBuilder {
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
    pub fn build(self) -> Result<GraniteRunner> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weights path required"))?
            .clone();
        let config = LlamaBaseConfig::from_gguf_path(&weights)
            .with_context(|| format!("rlx-granite: parse {weights:?}"))?;
        if !ACCEPTED_ARCHES.contains(&config.arch.as_str()) {
            bail!(
                "rlx-granite: expected `general.architecture` ∈ {ACCEPTED_ARCHES:?}; got `{}` at {weights:?}",
                config.arch
            );
        }
        let inner = self
            .inner
            .build()
            .context("rlx-granite: building underlying Llama32Runner")?;
        Ok(GraniteRunner { inner, config })
    }
}

pub fn cli_run(args: &[String]) -> Result<()> {
    if let Some(first) = args.iter().position(|a| a == "--weights") {
        if let Some(path) = args.get(first + 1) {
            let cfg = LlamaBaseConfig::from_gguf_path(Path::new(path))
                .with_context(|| format!("rlx-granite: parse {path}"))?;
            if !ACCEPTED_ARCHES.contains(&cfg.arch.as_str()) {
                bail!(
                    "rlx-granite: {path}: GGUF arch = `{}`, expected one of {ACCEPTED_ARCHES:?}",
                    cfg.arch
                );
            }
        }
    }
    rlx_llama32::cli::run(args)
}
