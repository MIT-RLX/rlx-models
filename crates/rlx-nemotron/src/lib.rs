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

//! NVIDIA Nemotron 3 Nano runner.
//!
//! Nemotron ships as several GGUF arch tags:
//! * `nemotron` — text-only, Llama-shaped attention stack; runs via the
//!   [`rlx_llama32::Llama32Runner`] delegate below.
//! * `nemotron_h` / `nemotron_h_moe` — hybrid Mamba2 + attention; the
//!   [`NemotronHybridRunner`] in `runner.rs` drives it via per-layer
//!   `Mamba2StepStage` interleaved with stateless attention blocks.
//!
//! The Omni 30B variant (vision + audio) lives in `rlx-nemotron-omni`
//! and is wired independently.

use anyhow::{Context, Result, bail};
use rlx_llama_base::LlamaBaseConfig;
use std::path::{Path, PathBuf};

pub use rlx_llama32::{Llama32ConfigSource, Llama32Runner, Llama32RunnerBuilder};

pub mod config;
pub mod flow;
pub mod runner;

pub use config::{NemotronHybridConfig, NemotronLayerKind};
pub use flow::{mamba2_decode_layer_plugin_with_sink, stateless_attention_layer_plugin};
pub use runner::{NemotronHybridRunner, NemotronHybridRunnerBuilder};

pub const PLAN_MILESTONE: &str = "M5";
pub const FAMILY: &str = "Nemotron (text)";

const ACCEPTED_ARCHES: &[&str] = &["nemotron", "nemotron_h", "nemotron_h_moe"];
const ATTN_ONLY_ARCHES: &[&str] = &["nemotron"];

pub struct NemotronRunner {
    inner: Llama32Runner,
    config: LlamaBaseConfig,
}

impl NemotronRunner {
    pub fn builder() -> NemotronRunnerBuilder {
        NemotronRunnerBuilder::default()
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
pub struct NemotronRunnerBuilder {
    weights: Option<PathBuf>,
    inner: Llama32RunnerBuilder,
}

impl NemotronRunnerBuilder {
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

    pub fn build(self) -> Result<NemotronRunner> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weights path required (call .weights(...))"))?
            .clone();
        let config = LlamaBaseConfig::from_gguf_path(&weights)
            .with_context(|| format!("rlx-nemotron: parse {weights:?}"))?;
        if !ACCEPTED_ARCHES.contains(&config.arch.as_str()) {
            bail!(
                "rlx-nemotron: expected `general.architecture` ∈ {ACCEPTED_ARCHES:?}; \
                 got `{}` at {weights:?}",
                config.arch
            );
        }
        if !ATTN_ONLY_ARCHES.contains(&config.arch.as_str()) {
            bail!(
                "rlx-nemotron: arch `{}` is hybrid Mamba2+attention — use \
                 `NemotronHybridRunner::builder()` (this builder is attention-only \
                 via the Llama32Runner delegate). The hybrid runner reads the same \
                 `--weights` path, picks layer kinds from \
                 `{0}.{{layer_kinds, attn_layer_period}}` metadata, and drives \
                 per-layer Mamba2 state buffers across decode calls.",
                config.arch
            );
        }
        let inner = self
            .inner
            .build()
            .context("rlx-nemotron: building underlying Llama32Runner")?;
        Ok(NemotronRunner { inner, config })
    }
}

pub fn cli_run(args: &[String]) -> Result<()> {
    if let Some(first) = args.iter().position(|a| a == "--weights") {
        if let Some(path) = args.get(first + 1) {
            let cfg = LlamaBaseConfig::from_gguf_path(Path::new(path))
                .with_context(|| format!("rlx-nemotron: parse {path}"))?;
            if !ACCEPTED_ARCHES.contains(&cfg.arch.as_str()) {
                bail!(
                    "rlx-nemotron: {path}: GGUF arch = `{}`, expected one of {ACCEPTED_ARCHES:?}",
                    cfg.arch
                );
            }
        }
    }
    rlx_llama32::cli::run(args)
}
