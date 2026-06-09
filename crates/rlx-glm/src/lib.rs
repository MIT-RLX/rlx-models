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

//! GLM 5.1 runner.
//!
//! GLM ships as `general.architecture = glm4` / `glm5` / `chatglm` /
//! `glm4moe` in its GGUF converters. Llama-shaped with a GLM-specific
//! RoPE layout and RMSNorm placement. This crate is a thin wrapper
//! over [`rlx_llama32::Llama32Runner`] with arch validation.
//!
//! **Caveat:** GLM-specific RoPE (`rope_ratio`) and the
//! post-LayerNorm-on-FFN-input placement aren't yet wired in
//! `rlx-llama32` — runs produce *some* tokens but won't match the
//! upstream reference for the first few prompts until those deltas
//! land (PLAN.md M5 follow-up).

use anyhow::{Context, Result, bail};
use rlx_llama_base::LlamaBaseConfig;
use std::path::{Path, PathBuf};

pub use rlx_llama32::{Llama32ConfigSource, Llama32Runner, Llama32RunnerBuilder};

pub const PLAN_MILESTONE: &str = "M5";
pub const FAMILY: &str = "GLM 5.1";

const ACCEPTED_ARCHES: &[&str] = &["glm4", "glm5", "chatglm", "glm4moe"];

pub struct GlmRunner {
    inner: Llama32Runner,
    config: LlamaBaseConfig,
}

impl GlmRunner {
    pub fn builder() -> GlmRunnerBuilder {
        GlmRunnerBuilder::default()
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
}

#[derive(Debug, Default)]
pub struct GlmRunnerBuilder {
    weights: Option<PathBuf>,
    inner: Llama32RunnerBuilder,
}

impl GlmRunnerBuilder {
    pub fn weights(mut self, path: impl Into<PathBuf>) -> Self {
        let p = path.into();
        self.weights = Some(p.clone());
        self.inner = self.inner.weights(p);
        self
    }
    pub fn inner(mut self, f: impl FnOnce(Llama32RunnerBuilder) -> Llama32RunnerBuilder) -> Self {
        self.inner = f(self.inner);
        self
    }

    pub fn build(self) -> Result<GlmRunner> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weights path required (call .weights(...))"))?
            .clone();
        let config = LlamaBaseConfig::from_gguf_path(&weights)
            .with_context(|| format!("rlx-glm: parse {weights:?}"))?;
        if !ACCEPTED_ARCHES.contains(&config.arch.as_str()) {
            bail!(
                "rlx-glm: expected `general.architecture` ∈ {ACCEPTED_ARCHES:?}; \
                 got `{}` at {weights:?}",
                config.arch
            );
        }
        let inner = self
            .inner
            .build()
            .context("rlx-glm: building underlying Llama32Runner")?;
        Ok(GlmRunner { inner, config })
    }
}

pub fn cli_run(args: &[String]) -> Result<()> {
    if let Some(first) = args.iter().position(|a| a == "--weights") {
        if let Some(path) = args.get(first + 1) {
            let cfg = LlamaBaseConfig::from_gguf_path(Path::new(path))
                .with_context(|| format!("rlx-glm: parse {path}"))?;
            if !ACCEPTED_ARCHES.contains(&cfg.arch.as_str()) {
                bail!(
                    "rlx-glm: {path}: GGUF arch = `{}`, expected one of {ACCEPTED_ARCHES:?}",
                    cfg.arch
                );
            }
        }
    }
    rlx_llama32::cli::run(args)
}
