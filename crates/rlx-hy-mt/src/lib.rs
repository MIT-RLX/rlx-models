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

//! Tencent [HY-MT1.5](https://huggingface.co/tencent/HY-MT1.5-1.8B) — on-device
//! multilingual translation.
//!
//! Architecture is **Hunyuan dense** (`hunyuan_v1_dense` / GGUF `hunyuan-dense`):
//! GQA + QK-norm + SwiGLU + RMSNorm — the same shape as Qwen3. This crate
//! validates HY-MT checkpoints and delegates inference to [`rlx_qwen3::Qwen3Runner`].
//!
//! Official prompts (from the model card):
//! - ZH↔XX: Chinese instruction template
//! - XX↔XX (incl. EN→FR): `Translate the following segment into {lang}, without additional explanation.`

pub mod config;
pub mod prompt;

use anyhow::{Context, Result};
use config::validate_weights_kind;
use rlx_cli::WeightFormat;
use rlx_qwen3::{Qwen3Config, Qwen3ConfigSource, Qwen3Runner, Qwen3RunnerBuilder, SampleOpts};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

pub use config::{
    GGUF_ARCHES, HF_MODEL_TYPES, HY_MT_1_8B_HIDDEN, HY_MT_1_8B_LAYERS, HY_MT_1_8B_ROPE_THETA_GGUF,
    HY_MT_7B_HIDDEN, HY_MT_7B_LAYERS, config_json_path, hy_mt_1_8b_preset, qwen3_config_from_hf,
    validate_hf_config,
};
pub use prompt::{
    HY_EOT_TOKEN_ID, TranslatePrompt, format_xx_to_xx_prompt, format_zh_to_xx_prompt,
    target_language_name, wrap_hy_mt_user_turn,
};
pub use rlx_qwen3::{Qwen3Config as HyMtLmConfig, SampleOpts as HyMtSampleOpts};

/// Human-readable family label.
pub const FAMILY: &str = "HY-MT";

/// Hugging Face id for the 1.8B translation checkpoint.
pub const HF_MODEL_ID_1_8B: &str = "tencent/HY-MT1.5-1.8B";
/// Official GGUF quants (Q4_K_M / Q6_K / Q8_0).
pub const HF_MODEL_ID_1_8B_GGUF: &str = "tencent/HY-MT1.5-1.8B-GGUF";
/// 7B sibling.
pub const HF_MODEL_ID_7B: &str = "tencent/HY-MT1.5-7B";

/// Published GGUF filenames on `tencent/HY-MT1.5-1.8B-GGUF`.
pub const HY_MT_1_8B_GGUF_FILES: &[(&str, &str)] = &[
    ("Q4_K_M", "HY-MT1.5-1.8B-Q4_K_M.gguf"),
    ("Q6_K", "HY-MT1.5-1.8B-Q6_K.gguf"),
    ("Q8_0", "HY-MT1.5-1.8B-Q8_0.gguf"),
];

/// Typed runner for HY-MT1.5 dense checkpoints.
pub struct HyMtRunner {
    inner: Qwen3Runner,
}

impl HyMtRunner {
    /// Start building a runner (requires [`.weights(...)`](HyMtRunnerBuilder::weights)).
    pub fn builder() -> HyMtRunnerBuilder {
        HyMtRunnerBuilder::default()
    }

    /// Underlying Qwen3-shaped config.
    pub fn config(&self) -> &Qwen3Config {
        self.inner.config()
    }

    /// Borrow the inner [`Qwen3Runner`].
    pub fn inner(&self) -> &Qwen3Runner {
        &self.inner
    }

    /// Mutable access to the inner runner.
    pub fn inner_mut(&mut self) -> &mut Qwen3Runner {
        &mut self.inner
    }

    /// Packed-decode generation (GGUF K-quants).
    pub fn generate_packed(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        self.inner.generate_packed(prompt_ids, n_new, on_token)
    }

    /// KV-cached greedy / sampled generation.
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

/// Builder for [`HyMtRunner`]. Same surface as [`Qwen3RunnerBuilder`].
#[derive(Debug, Clone, Default)]
pub struct HyMtRunnerBuilder {
    weights: Option<PathBuf>,
    inner: Qwen3RunnerBuilder,
}

impl HyMtRunnerBuilder {
    /// Path to a safetensors shard, model directory, or GGUF file.
    pub fn weights(mut self, path: impl Into<PathBuf>) -> Self {
        let p: PathBuf = path.into();
        self.weights = Some(p.clone());
        self.inner = self.inner.weights(p);
        self
    }

    /// Maximum sequence length for compile / KV cache.
    pub fn max_seq(mut self, n: usize) -> Self {
        self.inner = self.inner.max_seq(n);
        self
    }

    /// Enable packed GGUF matmul (`Op::DequantMatMul`).
    pub fn packed_weights(mut self, on: bool) -> Self {
        self.inner = self.inner.packed_weights(on);
        self
    }

    /// Execution device.
    pub fn device(mut self, d: Device) -> Self {
        self.inner = self.inner.device(d);
        self
    }

    /// Sampling options.
    pub fn sample(mut self, opts: SampleOpts) -> Self {
        self.inner = self.inner.sample(opts);
        self
    }

    /// Build after validating HY-MT weight metadata.
    pub fn build(self) -> Result<HyMtRunner> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weights path required (call .weights(...))"))?
            .clone();

        validate_weights_kind(&weights)?;

        let inner = match WeightFormat::from_path(&weights)? {
            WeightFormat::Gguf => self
                .inner
                .build()
                .context("rlx-hy-mt: building Qwen3Runner for GGUF")?,
            WeightFormat::Safetensors => {
                let cfg = qwen3_config_from_hf(&weights)?;
                self.inner
                    .config(Qwen3ConfigSource::Explicit(cfg))
                    .build()
                    .context("rlx-hy-mt: building Qwen3Runner for safetensors")?
            }
        };

        Ok(HyMtRunner { inner })
    }
}

/// CLI entry — validates HY-MT weights then delegates to [`rlx_qwen3::cli::run`].
pub fn cli_run(args: &[String]) -> Result<()> {
    if let Some(first) = args.iter().position(|a| a == "--weights")
        && let Some(path) = args.get(first + 1)
    {
        validate_weights_kind(Path::new(path))?;
    }
    rlx_qwen3::cli::run(args)
}
