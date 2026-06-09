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

//! OmniCoder family runner.
//!
//! Tesslate/OmniCoder-9B and siblings are Qwen3-coder-shaped and ship
//! as `general.architecture = qwen3` in their GGUF converters. This
//! crate is a thin wrapper over [`rlx_qwen3::Qwen3Runner`] with arch
//! validation so users get a discoverable per-family entry point and
//! misrouted weights fail loudly at `build()`.

use anyhow::{Context, Result, bail};
use rlx_llama_base::LlamaBaseConfig;
use std::path::{Path, PathBuf};

pub use rlx_qwen3::{Qwen3ConfigSource, Qwen3Runner, Qwen3RunnerBuilder};

pub const PLAN_MILESTONE: &str = "M4";
pub const FAMILY: &str = "OmniCoder";

pub struct OmniCoderRunner {
    inner: Qwen3Runner,
    config: LlamaBaseConfig,
}

impl OmniCoderRunner {
    pub fn builder() -> OmniCoderRunnerBuilder {
        OmniCoderRunnerBuilder::default()
    }
    pub fn config(&self) -> &LlamaBaseConfig {
        &self.config
    }
    pub fn inner(&self) -> &Qwen3Runner {
        &self.inner
    }
    pub fn inner_mut(&mut self) -> &mut Qwen3Runner {
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
pub struct OmniCoderRunnerBuilder {
    weights: Option<PathBuf>,
    inner: Qwen3RunnerBuilder,
}

impl OmniCoderRunnerBuilder {
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
    pub fn build(self) -> Result<OmniCoderRunner> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weights path required"))?
            .clone();
        let config = LlamaBaseConfig::from_gguf_path(&weights)
            .with_context(|| format!("rlx-omnicoder: parse {weights:?}"))?;
        if !matches!(config.arch.as_str(), "qwen3" | "qwen2") {
            bail!(
                "rlx-omnicoder: expected `general.architecture = qwen3` (OmniCoder uses \
                 Qwen3 arch); got `{}` at {weights:?}",
                config.arch
            );
        }
        let inner = self
            .inner
            .build()
            .context("rlx-omnicoder: building underlying Qwen3Runner")?;
        Ok(OmniCoderRunner { inner, config })
    }
}

pub fn cli_run(args: &[String]) -> Result<()> {
    if let Some(first) = args.iter().position(|a| a == "--weights") {
        if let Some(path) = args.get(first + 1) {
            let cfg = LlamaBaseConfig::from_gguf_path(Path::new(path))
                .with_context(|| format!("rlx-omnicoder: parse {path}"))?;
            if !matches!(cfg.arch.as_str(), "qwen3" | "qwen2") {
                bail!(
                    "rlx-omnicoder: {path}: GGUF arch = `{}`, expected `qwen3` (OmniCoder)",
                    cfg.arch
                );
            }
        }
    }
    rlx_qwen3::cli::run(args)
}
