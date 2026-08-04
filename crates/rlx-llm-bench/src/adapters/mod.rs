// RLX models — LLM benchmark harness.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Per-model adapters: turn a [`BuildSpec`] into a ready [`BenchModel`].
//!
//! Each model lives behind its own cargo feature and a match arm here. Adding a
//! model is: implement `LmRunner` in the model crate (most already do), add an
//! `adapters/<model>.rs` that builds it, gate it on a `<model>` feature, and add
//! one arm to [`build_model`]. Nothing else in the harness changes.

use std::path::PathBuf;

use anyhow::{Result, bail};
use rlx_runtime::Device;

use crate::model::BenchModel;

#[cfg(feature = "qwen3")]
pub mod qwen3;

/// Everything an adapter needs to construct a model.
#[derive(Debug, Clone)]
pub struct BuildSpec {
    /// Model family, e.g. `"qwen3"`. Selects the adapter.
    pub model_kind: String,
    /// Weights path (file or mlx-community directory).
    pub weights: PathBuf,
    /// Inference device.
    pub device: Device,
    /// Prefill/decode bucket hint for the runner.
    pub max_seq: usize,
    /// `tokenizer.json` path (needed for text tasks: MMLU/GSM8K).
    pub tokenizer: Option<PathBuf>,
    /// EOS ids that stop generation. Empty ⇒ the adapter's family default.
    pub eos_ids: Vec<u32>,
    /// Force the F32 (unpacked) path. Quality tasks need it because
    /// `prefill_logits`/`decode_logits` — the log-prob scorers — are F32-only;
    /// the packed path exposes only whole-sequence `predict_logits`.
    pub force_f32: bool,
    /// Optional display name; defaults to the weights file stem.
    pub name: Option<String>,
}

/// Lowercase device label for the leaderboard.
pub fn device_label(d: Device) -> &'static str {
    match d {
        Device::Cpu => "cpu",
        Device::Metal => "metal",
        Device::Mlx => "mlx",
        Device::Ane => "coreml",
        Device::Cuda => "cuda",
        Device::Rocm => "rocm",
        Device::Gpu => "wgpu",
        Device::Vulkan => "vulkan",
        other => other.name(),
    }
}

/// Parse a `--device` string into a [`Device`].
pub fn parse_device(s: &str) -> Result<Device> {
    let d = match s.trim().to_ascii_lowercase().as_str() {
        "cpu" => Device::Cpu,
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "coreml" | "ane" => Device::Ane,
        "cuda" | "nvidia" => Device::Cuda,
        "rocm" | "amd" => Device::Rocm,
        "gpu" | "wgpu" => Device::Gpu,
        "vulkan" => Device::Vulkan,
        other => bail!(
            "unknown device {other:?}; expected one of: cpu, metal, mlx, coreml, cuda, rocm, wgpu, vulkan"
        ),
    };
    Ok(d)
}

/// Build a [`BenchModel`] for `spec.model_kind`, dispatching to the adapter
/// compiled in.
pub fn build_model(spec: &BuildSpec) -> Result<BenchModel> {
    match spec.model_kind.as_str() {
        "qwen3" => {
            #[cfg(feature = "qwen3")]
            {
                qwen3::build(spec)
            }
            #[cfg(not(feature = "qwen3"))]
            {
                bail!("qwen3 adapter not compiled in; rebuild with `--features qwen3`")
            }
        }
        other => bail!(
            "unknown model kind {other:?}; compiled adapters: {}",
            compiled_adapters()
        ),
    }
}

/// Comma-separated list of adapters compiled into this binary. Each adapter
/// contributes one cfg-gated entry, so the list stays correct as models are
/// added without a mutable accumulator.
pub fn compiled_adapters() -> String {
    let v: &[&str] = &[
        #[cfg(feature = "qwen3")]
        "qwen3",
    ];
    if v.is_empty() {
        "<none>".to_string()
    } else {
        v.join(", ")
    }
}
