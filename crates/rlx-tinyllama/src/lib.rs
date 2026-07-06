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

//! TinyLlama-1.1B — [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0).
//!
//! Standard Llama decoder: GQA + RoPE + SwiGLU + RMSNorm, `LlamaForCausalLM`
//! weight layout. This crate wraps [`rlx_llama32::Llama32Runner`] with:
//!
//! * GGUF `general.architecture = llama` validation plus 1.1B shape checks;
//! * HF `config.json` checks (`model_type = llama`, `LlamaForCausalLM`) for safetensors;
//! * a typed [`TinyLlamaRunner`] surface and `rlx-tinyllama` CLI binary.
//!
//! **How to run:** see [README.md](README.md) (download, CLI, GGUF, examples).

pub mod config;

#[cfg(feature = "hf-download")]
pub mod download;

/// Transformers-style one-liner API — see [`pipeline::TextGeneration`].
#[cfg(feature = "pipeline")]
pub mod pipeline;

#[cfg(feature = "pipeline")]
pub use pipeline::{ChatMessage, GenerationConfig, TextGeneration, TextGenerationBuilder};

use anyhow::{Context, Result};
use config::validate_weights_kind;
use rlx_cli::WeightFormat;
use rlx_llama_base::LlamaBaseConfig;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

pub use config::{
    TINYLLAMA_1_1B_HIDDEN_SIZE, TINYLLAMA_1_1B_NUM_LAYERS, config_json_path, llama_config_from_hf,
    tinyllama_1_1b_preset,
};
#[cfg(feature = "hf-download")]
pub use download::{
    default_hf_cache_dir, download_tinyllama_1_1b, download_tinyllama_gguf, fetch_tinyllama_1_1b,
    fetch_tinyllama_gguf, materialize_tinyllama_1_1b, materialize_tinyllama_gguf,
};
pub use rlx_llama32::{Llama32Config, Llama32ConfigSource, Llama32Runner, Llama32RunnerBuilder};

/// Human-readable family label (`"TinyLlama"`).
pub const FAMILY: &str = "TinyLlama";

/// Hugging Face model id for the 1.1B chat reference checkpoint.
pub const HF_MODEL_ID_1_1B: &str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";

/// GGUF quants on Hugging Face ([TheBloke](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF); public).
pub const HF_MODEL_ID_GGUF: &str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF";

/// Published GGUF filenames on Hugging Face (`TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`).
pub const TINYLLAMA_GGUF_FILES: &[(&str, &str)] = &[
    ("Q4_K_M", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
    ("Q8_0", "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"),
    ("Q6_K", "tinyllama-1.1b-chat-v1.0.Q6_K.gguf"),
];

/// Typed runner for TinyLlama-1.1B checkpoints.
///
/// Wraps [`Llama32Runner`] and validates weight metadata at [`TinyLlamaRunnerBuilder::build`].
pub struct TinyLlamaRunner {
    inner: Llama32Runner,
    /// Parsed GGUF metadata when weights are GGUF; otherwise derived from HF config.
    base: LlamaBaseConfig,
}

impl TinyLlamaRunner {
    /// Start building a runner (requires [`.weights(...)`](TinyLlamaRunnerBuilder::weights)).
    pub fn builder() -> TinyLlamaRunnerBuilder {
        TinyLlamaRunnerBuilder::default()
    }

    /// Shared Llama-shaped arch fields (dims, RoPE, GQA, …).
    pub fn base_config(&self) -> &LlamaBaseConfig {
        &self.base
    }

    /// Underlying [`Llama32Config`] used by the inner runner.
    pub fn llama_config(&self) -> &Llama32Config {
        self.inner.config()
    }

    /// Borrow the inner [`Llama32Runner`] for advanced prefill/decode control.
    pub fn inner(&self) -> &Llama32Runner {
        &self.inner
    }

    /// Mutable access to the inner [`Llama32Runner`].
    pub fn inner_mut(&mut self) -> &mut Llama32Runner {
        &mut self.inner
    }

    /// Packed-decode generation (GGUF K-quants via [`TinyLlamaRunnerBuilder::packed_weights`]).
    pub fn generate_packed(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        self.inner.generate_packed(prompt_ids, n_new, on_token)
    }

    /// KV-cached greedy generation (F32 weights; safetensors or GGUF dequant).
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        self.inner.generate(prompt_ids, n_new, on_token)
    }

    /// KV-cached generation that stops early when `keep_going` returns
    /// `false` (e.g. on an end-of-sequence id). See
    /// [`Llama32Runner::generate_until`].
    pub fn generate_until(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        keep_going: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        self.inner.generate_until(prompt_ids, n_new, keep_going)
    }

    /// Swap the sampling options (greedy / temperature / top-p) without
    /// rebuilding. See [`Llama32Runner::set_sample`].
    pub fn set_sample(&mut self, opts: rlx_llama32::SampleOpts) {
        self.inner.set_sample(opts);
    }

    /// Last-position logits after prefill.
    pub fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        self.inner.predict_logits(prompt_ids)
    }
}

/// Builder for [`TinyLlamaRunner`]. Same surface as [`Llama32RunnerBuilder`].
#[derive(Debug, Clone, Default)]
pub struct TinyLlamaRunnerBuilder {
    weights: Option<PathBuf>,
    inner: Llama32RunnerBuilder,
}

impl TinyLlamaRunnerBuilder {
    /// Path to a safetensors shard, model directory, or GGUF file.
    pub fn weights(mut self, path: impl Into<PathBuf>) -> Self {
        let p: PathBuf = path.into();
        self.weights = Some(p.clone());
        self.inner = self.inner.weights(p);
        self
    }

    /// Maximum sequence length for compile / KV cache (default 512 in the inner runner).
    pub fn max_seq(mut self, n: usize) -> Self {
        self.inner = self.inner.max_seq(n);
        self
    }

    /// Enable packed GGUF matmul (`Op::DequantMatMul`) for K-quant checkpoints.
    pub fn packed_weights(mut self, on: bool) -> Self {
        self.inner = self.inner.packed_weights(on);
        self
    }

    /// Execution device for decode (prefill may route differently for packed GGUF).
    pub fn device(mut self, d: Device) -> Self {
        self.inner = self.inner.device(d);
        self
    }

    /// Build the runner after validating TinyLlama-1.1B weight metadata.
    pub fn build(self) -> Result<TinyLlamaRunner> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weights path required (call .weights(...))"))?
            .clone();

        validate_weights_kind(&weights)?;

        let base = match WeightFormat::from_path(&weights)? {
            WeightFormat::Gguf => LlamaBaseConfig::from_gguf_path(&weights)
                .with_context(|| format!("rlx-tinyllama: parse GGUF {weights:?}"))?,
            WeightFormat::Safetensors => llama_base_from_hf(&weights)?,
        };

        let inner = self
            .inner
            .build()
            .context("rlx-tinyllama: building underlying Llama32Runner")?;

        Ok(TinyLlamaRunner { inner, base })
    }
}

fn llama_base_from_hf(weights_or_dir: &Path) -> Result<LlamaBaseConfig> {
    let cfg = config::llama_config_from_hf(weights_or_dir)?;
    Ok(LlamaBaseConfig {
        arch: "llama".into(),
        vocab_size: cfg.vocab_size,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        rms_norm_eps: cfg.rms_norm_eps,
        rope_theta: cfg.rope_theta,
        rope_scaling: None,
        sliding_window: None,
        max_position_embeddings: cfg.max_position_embeddings,
    })
}

/// CLI entry — delegates to [`rlx_llama32::cli::run`] after weight-kind checks.
pub fn cli_run(args: &[String]) -> Result<()> {
    if let Some(first) = args.iter().position(|a| a == "--weights") {
        if let Some(path) = args.get(first + 1) {
            validate_weights_kind(Path::new(path))?;
        }
    }
    rlx_llama32::cli::run(args)
}
