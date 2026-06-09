// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// MiniCPM5 — OpenBMB edge LMs (e.g. [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)).
//
// Standard Llama decoder: GQA + RoPE + SwiGLU + RMSNorm, `LlamaForCausalLM`
// weight layout. This crate wraps [`rlx_llama32::Llama32Runner`] with:
//
// * GGUF `general.architecture = llama` validation;
// * HF `config.json` checks (`model_type = llama`, `LlamaForCausalLM`) for safetensors;
// * a typed [`MiniCpm5Runner`] surface and `rlx-minicpm5` CLI binary.
//
// **How to run:** see [README.md](README.md) (download, CLI, chat, GGUF, examples).

pub mod config;

#[cfg(feature = "hf-download")]
pub mod download;

use anyhow::{Context, Result};
use config::validate_weights_kind;
use rlx_cli::WeightFormat;
use rlx_llama_base::LlamaBaseConfig;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

pub use config::{config_json_path, llama_config_from_hf, minicpm5_1b_preset};
#[cfg(feature = "hf-download")]
pub use download::{
    default_hf_cache_dir, download_minicpm5_1b, download_minicpm5_gguf, fetch_minicpm5_1b,
    fetch_minicpm5_gguf, materialize_minicpm5_1b, materialize_minicpm5_gguf,
};
pub use rlx_llama32::{Llama32Config, Llama32ConfigSource, Llama32Runner, Llama32RunnerBuilder};

pub const FAMILY: &str = "MiniCPM5";
/// HF model id for the 1B reference checkpoint.
pub const HF_MODEL_ID_1B: &str = "openbmb/MiniCPM5-1B";
/// GGUF quants (Q4_K_M, Q8_0, F16) on Hugging Face.
pub const HF_MODEL_ID_GGUF: &str = "openbmb/MiniCPM5-1B-GGUF";

/// Published GGUF filenames on Hugging Face (`openbmb/MiniCPM5-1B-GGUF`).
pub const MINICPM5_GGUF_FILES: &[(&str, &str)] = &[
    ("Q4_K_M", "MiniCPM5-1B-Q4_K_M.gguf"),
    ("Q8_0", "MiniCPM5-1B-Q8_0.gguf"),
    ("F16", "MiniCPM5-1B-F16.gguf"),
];

pub struct MiniCpm5Runner {
    inner: Llama32Runner,
    /// Parsed GGUF metadata when weights are GGUF; otherwise derived from HF config.
    base: LlamaBaseConfig,
}

impl MiniCpm5Runner {
    pub fn builder() -> MiniCpm5RunnerBuilder {
        MiniCpm5RunnerBuilder::default()
    }

    pub fn base_config(&self) -> &LlamaBaseConfig {
        &self.base
    }

    pub fn llama_config(&self) -> &Llama32Config {
        self.inner.config()
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

    /// KV-cached greedy generation (F32 weights; safetensors or GGUF dequant).
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        self.inner.generate(prompt_ids, n_new, on_token)
    }

    /// Last-position logits after prefill.
    pub fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        self.inner.predict_logits(prompt_ids)
    }
}

#[derive(Debug, Clone, Default)]
pub struct MiniCpm5RunnerBuilder {
    weights: Option<PathBuf>,
    inner: Llama32RunnerBuilder,
}

impl MiniCpm5RunnerBuilder {
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

    pub fn device(mut self, d: Device) -> Self {
        self.inner = self.inner.device(d);
        self
    }

    pub fn build(self) -> Result<MiniCpm5Runner> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weights path required (call .weights(...))"))?
            .clone();

        validate_weights_kind(&weights)?;

        let base = match WeightFormat::from_path(&weights)? {
            WeightFormat::Gguf => LlamaBaseConfig::from_gguf_path(&weights)
                .with_context(|| format!("rlx-minicpm5: parse GGUF {weights:?}"))?,
            WeightFormat::Safetensors => llama_base_from_hf(&weights)?,
        };

        let inner = self
            .inner
            .build()
            .context("rlx-minicpm5: building underlying Llama32Runner")?;

        Ok(MiniCpm5Runner { inner, base })
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

/// CLI entry — delegates to `rlx_llama32::cli::run` after weight-kind checks.
pub fn cli_run(args: &[String]) -> Result<()> {
    if let Some(first) = args.iter().position(|a| a == "--weights") {
        if let Some(path) = args.get(first + 1) {
            validate_weights_kind(Path::new(path))?;
        }
    }
    rlx_llama32::cli::run(args)
}
