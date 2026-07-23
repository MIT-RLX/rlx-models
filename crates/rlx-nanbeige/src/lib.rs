// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Nanbeige4.2 — Looped Transformer causal LM
// ([Nanbeige/Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B)).
//
// Llama-shaped decoder (GQA + RoPE + SwiGLU + RMSNorm) with `num_loops > 1`:
// the same physical layers are applied multiple times with shared weights and
// per-loop KV caches. Inference is implemented in [`rlx_llama32::Llama32Runner`]
// once HF `config.json` is validated.
//
// **How to run:** see [README.md](README.md).

pub mod config;
pub mod device_policy;

#[cfg(feature = "hf-download")]
pub mod download;

use anyhow::{Context, Result};
use config::validate_weights_kind;
use rlx_cli::WeightFormat;
use rlx_llama_base::LlamaBaseConfig;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

pub use config::{config_json_path, llama_config_from_hf, nanbeige42_3b_preset};
pub use device_policy::{
    BackendPlan, approx_param_bytes_f32, assert_full_model_fits, clamp_max_seq, kv_cache_bytes,
    prepare as prepare_device, working_set_bytes,
};
#[cfg(feature = "hf-download")]
pub use download::{
    default_hf_cache_dir, download_nanbeige42_3b, fetch_nanbeige42_3b, materialize_nanbeige42_3b,
};
pub use rlx_llama32::{Llama32Config, Llama32ConfigSource, Llama32Runner, Llama32RunnerBuilder};

pub const FAMILY: &str = "Nanbeige";
/// HF model id for the 3B instruct / agentic checkpoint.
pub const HF_MODEL_ID_3B: &str = "Nanbeige/Nanbeige4.2-3B";
/// HF model id for the base (pre-SFT) checkpoint.
pub const HF_MODEL_ID_3B_BASE: &str = "Nanbeige/Nanbeige4.2-3B-Base";

fn weight_format(weights: &Path) -> Result<WeightFormat> {
    if weights.is_dir() {
        WeightFormat::detect(weights)
    } else {
        WeightFormat::from_path(weights)
    }
}

pub struct NanbeigeRunner {
    inner: Llama32Runner,
    /// Parsed GGUF metadata when weights are GGUF; otherwise derived from HF config.
    base: LlamaBaseConfig,
}

impl NanbeigeRunner {
    pub fn builder() -> NanbeigeRunnerBuilder {
        NanbeigeRunnerBuilder::default()
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
pub struct NanbeigeRunnerBuilder {
    weights: Option<PathBuf>,
    device: Option<Device>,
    inner: Llama32RunnerBuilder,
}

impl NanbeigeRunnerBuilder {
    pub fn weights(mut self, path: impl Into<PathBuf>) -> Self {
        let p: PathBuf = path.into();
        self.weights = Some(p.clone());
        self.inner = self.inner.weights(p);
        self
    }

    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self.inner = self.inner.device(d);
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

    pub fn max_memory_gb(mut self, gb: f32) -> Self {
        self.inner = self.inner.max_memory_gb(gb);
        self
    }

    pub fn bucketed_decode_cache(mut self, on: bool) -> Self {
        self.inner = self.inner.bucketed_decode_cache(on);
        self
    }

    pub fn config(mut self, src: Llama32ConfigSource) -> Self {
        self.inner = self.inner.config(src);
        self
    }

    /// Apply [`BackendPlan`] for `device` (max_seq, bucketed decode, …).
    pub fn with_device_plan(mut self, cfg: &Llama32Config, device: Device) -> Self {
        device_policy::prepare(device);
        let plan = BackendPlan::for_device(cfg, device);
        self.device = Some(device);
        self.inner = self
            .inner
            .device(device)
            .max_seq(plan.max_seq)
            .bucketed_decode_cache(plan.bucketed_decode);
        self
    }

    pub fn build(self) -> Result<NanbeigeRunner> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weights path required"))?
            .clone();
        validate_weights_kind(&weights)?;
        let device = self.device.unwrap_or(Device::Cpu);
        let fmt = weight_format(&weights)?;

        let mut inner_builder = self.inner.format(fmt);
        let (base, hf_cfg) = match fmt {
            WeightFormat::Gguf => {
                let base = LlamaBaseConfig::from_gguf_path(&weights)
                    .with_context(|| format!("rlx-nanbeige: parse GGUF {weights:?}"))?;
                (base, None)
            }
            WeightFormat::Safetensors => {
                let cfg = llama_config_from_hf(&weights)?;
                device_policy::assert_full_model_fits(&cfg, device, 64)?;
                let base = LlamaBaseConfig {
                    arch: "nanbeige".into(),
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
                };
                inner_builder = inner_builder.config(Llama32ConfigSource::Explicit(cfg.clone()));
                (base, Some(cfg))
            }
        };
        let _ = hf_cfg;

        let inner = inner_builder
            .build()
            .context("rlx-nanbeige: building underlying Llama32Runner")?;
        Ok(NanbeigeRunner { inner, base })
    }
}

/// CLI entry — validates weights, applies [`BackendPlan`] defaults, then
/// delegates to `rlx_llama32::cli::run`.
pub fn cli_run(args: &[String]) -> Result<()> {
    if let Some(first) = args.iter().position(|a| a == "--weights") {
        if let Some(path) = args.get(first + 1) {
            validate_weights_kind(Path::new(path))?;
        }
    }

    let mut args = args.to_vec();
    let device_str = args
        .iter()
        .position(|a| a == "--device")
        .and_then(|i| args.get(i + 1).cloned())
        .unwrap_or_else(|| "cpu".into());
    let device = rlx_cli::parse_llama32_device(&device_str)?;

    device_policy::prepare(device);
    let cfg = nanbeige42_3b_preset();
    let plan = BackendPlan::for_device(&cfg, device);

    if !args.iter().any(|a| a == "--max-seq") {
        args.push("--max-seq".into());
        args.push(plan.max_seq.to_string());
    }
    if !plan.bucketed_decode && !args.iter().any(|a| a == "--no-bucketed-decode") {
        args.push("--no-bucketed-decode".into());
    }

    eprintln!(
        "[rlx-nanbeige] device={device:?} max_seq={} bucketed={} ({})",
        plan.max_seq, plan.bucketed_decode, plan.note
    );

    rlx_llama32::cli::run(&args)
}
