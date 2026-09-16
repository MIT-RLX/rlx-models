// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! [TranslateGemma](https://huggingface.co/google/translategemma-4b-it) — Google’s
//! open translation models (Gemma 3 backbone).
//!
//! mlx-community catalogs list `translategemma-*-it-*` as **gemma3** / generic
//! dense GeGLU+RMSNorm — already validated on the [`rlx_gemma`] path. This crate
//! brands the family, validates `gemma3` / `gemma3_text` checkpoints, and adds
//! MT prompt helpers. Inference delegates to [`GemmaRunner`].

pub mod prompt;

use anyhow::{Context, Result, bail};
use rlx_cli::WeightFormat;
use rlx_core::gguf_architecture_from_path;
use rlx_gemma::{GemmaRunner, GemmaRunnerBuilder};
use rlx_runtime::Device;
use serde::Deserialize;
use std::path::{Path, PathBuf};

pub use prompt::{
    TranslatePrompt, encode_translate_prompt, format_official_translate_prompt, language_code,
    target_language_name, wrap_gemma_user_turn,
};
// Deliberately re-exported while deprecated: removing it is a breaking change for
// downstreams. Callers should move to `format_official_translate_prompt`.
#[allow(deprecated)]
pub use prompt::format_translate_prompt;
pub use rlx_gemma::{GemmaConfig, GemmaRunner as TranslateGemmaInner};

/// Family label.
pub const FAMILY: &str = "TranslateGemma";
/// Reference HF id (4B instruction-tuned).
pub const HF_MODEL_ID_4B: &str = "google/translategemma-4b-it";
/// Accepted GGUF / HF arch tags (Gemma 3 text).
pub const GGUF_ARCHES: &[&str] = &["gemma3", "gemma3_text", "gemma"];
pub const HF_MODEL_TYPES: &[&str] = &["gemma3", "gemma3_text", "gemma2", "gemma"];

#[derive(Debug, Clone, Deserialize)]
struct HfProbe {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    architectures: Option<Vec<String>>,
    #[serde(default)]
    text_config: Option<serde_json::Value>,
}

fn config_json_path(weights_or_dir: &Path) -> PathBuf {
    if weights_or_dir.is_dir() {
        return weights_or_dir.join("config.json");
    }
    weights_or_dir
        .parent()
        .map(|p| p.join("config.json"))
        .unwrap_or_else(|| PathBuf::from("config.json"))
}

fn is_gemma3_family(mt: &str) -> bool {
    let a = mt.to_ascii_lowercase();
    HF_MODEL_TYPES.iter().any(|t| a == *t) || a.starts_with("gemma3")
}

/// Ensure weights look like Gemma 3 / TranslateGemma.
pub fn validate_weights_kind(weights: &Path) -> Result<()> {
    match WeightFormat::from_path(weights)? {
        WeightFormat::Gguf => {
            let arch = gguf_architecture_from_path(weights)?;
            if !GGUF_ARCHES.iter().any(|a| {
                arch.eq_ignore_ascii_case(a) || arch.to_ascii_lowercase().starts_with("gemma3")
            }) {
                bail!(
                    "rlx-translategemma: expected Gemma 3 GGUF arch ({GGUF_ARCHES:?}); got `{arch}`"
                );
            }
        }
        WeightFormat::Safetensors => {
            let cfg_path = config_json_path(weights);
            let raw = std::fs::read_to_string(&cfg_path)
                .with_context(|| format!("reading {cfg_path:?}"))?;
            let probe: HfProbe = serde_json::from_str(&raw)?;
            let mt = probe
                .model_type
                .or_else(|| {
                    probe
                        .text_config
                        .as_ref()
                        .and_then(|t| t.get("model_type"))
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                })
                .unwrap_or_default();
            if !is_gemma3_family(&mt) {
                bail!(
                    "rlx-translategemma: {cfg_path:?} model_type={mt:?}; expected gemma3 / gemma3_text"
                );
            }
            if let Some(archs) = &probe.architectures {
                let ok = archs.iter().any(|a| {
                    let l = a.to_ascii_lowercase();
                    l.contains("gemma") || l.contains("translategemma")
                });
                if !ok {
                    bail!("rlx-translategemma: unexpected architectures {archs:?}");
                }
            }
        }
    }
    Ok(())
}

/// Typed TranslateGemma runner (wraps [`GemmaRunner`]).
pub struct TranslateGemmaRunner {
    inner: GemmaRunner,
}

impl TranslateGemmaRunner {
    pub fn builder() -> TranslateGemmaRunnerBuilder {
        TranslateGemmaRunnerBuilder::default()
    }

    pub fn inner(&self) -> &GemmaRunner {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut GemmaRunner {
        &mut self.inner
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

#[derive(Debug, Clone, Default)]
pub struct TranslateGemmaRunnerBuilder {
    weights: Option<PathBuf>,
    inner: GemmaRunnerBuilder,
}

impl TranslateGemmaRunnerBuilder {
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

    pub fn build(self) -> Result<TranslateGemmaRunner> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weights path required"))?
            .clone();
        validate_weights_kind(&weights)?;
        let inner = self
            .inner
            .build()
            .context("rlx-translategemma: building GemmaRunner")?;
        Ok(TranslateGemmaRunner { inner })
    }
}

/// CLI — validate then delegate to `rlx-gemma`.
pub fn cli_run(args: &[String]) -> Result<()> {
    if let Some(i) = args.iter().position(|a| a == "--weights")
        && let Some(path) = args.get(i + 1)
    {
        validate_weights_kind(Path::new(path))?;
    }
    rlx_gemma::cli::run(args)
}
