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
//!
//! Multimodal (Pixtral mmproj) lives in `rlx-mistral-vl`.

use anyhow::{Context, Result, bail};
use rlx_cli::LmRunner;
use rlx_llama_base::LlamaBaseConfig;
use rlx_runtime::Device;
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
    pub fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        self.inner.predict_logits(prompt_ids)
    }
    /// Register a one-shot multimodal embed splice for the next `generate`
    /// (packed `input_embeddings` path). See
    /// [`rlx_llama32::Llama32Runner::set_multimodal_embed_override`].
    pub fn set_multimodal_embed_override(&mut self, start: usize, embeds: Vec<f32>) {
        self.inner.set_multimodal_embed_override(start, embeds);
    }
    /// Whether a registered multimodal splice is still unconsumed.
    pub fn multimodal_override_pending(&self) -> bool {
        self.inner.multimodal_override_pending()
    }
    /// Drop any unconsumed multimodal splice.
    pub fn clear_multimodal_embed_override(&mut self) {
        self.inner.clear_multimodal_embed_override();
    }
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        self.inner.generate(prompt_ids, n_new, on_token)
    }
}

impl LmRunner for MistralRunner {
    fn family(&self) -> &'static str {
        "mistral"
    }
    fn vocab_size(&self) -> usize {
        self.inner.config().vocab_size
    }
    fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        MistralRunner::predict_logits(self, prompt_ids)
    }
    fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        MistralRunner::generate(self, prompt_ids, n_new, |tok| {
            let _ = on_token(tok);
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct MistralRunnerBuilder {
    weights: Option<PathBuf>,
    inner: Llama32RunnerBuilder,
    accept_llama_arch: bool,
}

impl MistralRunnerBuilder {
    pub fn weights(mut self, path: impl Into<PathBuf>) -> Self {
        let p: PathBuf = path.into();
        self.weights = Some(p.clone());
        self.inner = self.inner.weights(p);
        self
    }
    /// Also accept `general.architecture = llama`. Mistral-Small-3.x / Ministral
    /// checkpoints are frequently converted with the legacy `llama` tag (they
    /// are Llama-shaped), so an arch check alone can't tell them from genuine
    /// Mistral-1/2. Opt in only when the Mistral-3 identity is already confirmed
    /// out-of-band — e.g. a paired Pixtral mmproj, which never accompanies a
    /// plain Mistral-1/2 text model. See [`crate::MistralRunner`] and
    /// `rlx-mistral-vl`.
    pub fn accept_llama_arch(mut self, on: bool) -> Self {
        self.accept_llama_arch = on;
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
    pub fn device(mut self, d: Device) -> Self {
        self.inner = self.inner.device(d);
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
        let arch_ok = ACCEPTED_ARCHES.contains(&config.arch.as_str())
            || (self.accept_llama_arch && config.arch == "llama");
        if !arch_ok {
            bail!(
                "rlx-mistral: expected `general.architecture` ∈ {ACCEPTED_ARCHES:?}; \
                 got `{}` at {weights:?} (Mistral 1/2 ship as `llama` — use rlx-llama32 directly; \
                 for a Mistral-3 VL checkpoint tagged `llama`, build with `.accept_llama_arch(true)`)",
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
