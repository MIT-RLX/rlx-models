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

//! Fara runner — thin wrapper over [`Qwen35Runner`] multimodal generate.

use crate::config::{FaraSize, is_model_dir};
use crate::prompt::format_fara_multimodal_prompt;
use crate::tools::{ToolCall, parse_tool_calls, text_before_tool_calls};
use anyhow::{Context, Result, bail};
use rlx_qwen35::{Qwen35ConfigSource, Qwen35Runner, Qwen35RunnerBuilder};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

/// One Fara agent step (screenshot + goal → text + tool calls).
#[derive(Debug, Clone)]
pub struct FaraStep {
    pub raw_text: String,
    pub thinking: String,
    pub tool_calls: Vec<ToolCall>,
}

pub struct FaraRunner {
    inner: Qwen35Runner,
    size: FaraSize,
    model_dir: PathBuf,
}

#[derive(Default)]
pub struct FaraRunnerBuilder {
    model_dir: Option<PathBuf>,
    size: Option<FaraSize>,
    device: Option<Device>,
    max_seq: Option<usize>,
    prefill_seq: Option<usize>,
}

impl FaraRunnerBuilder {
    pub fn model_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_dir = Some(path.into());
        self
    }
    pub fn size(mut self, size: FaraSize) -> Self {
        self.size = Some(size);
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }
    pub fn max_seq(mut self, n: usize) -> Self {
        self.max_seq = Some(n);
        self
    }
    pub fn prefill_seq(mut self, n: usize) -> Self {
        self.prefill_seq = Some(n);
        self
    }

    pub fn build(self) -> Result<FaraRunner> {
        let model_dir = self
            .model_dir
            .ok_or_else(|| anyhow::anyhow!("FaraRunner: model_dir required"))?;
        if !is_model_dir(&model_dir) {
            bail!(
                "FaraRunner: {model_dir:?} is not a Fara model dir \
                 (need config.json + safetensors)"
            );
        }
        let size = self.size.unwrap_or(FaraSize::B4);
        let device = self.device.unwrap_or(Device::Cpu);
        let max_seq = self.max_seq.unwrap_or(2048);
        // Prefer the size preset (nested HF `text_config` is also supported
        // via JsonFile). Explicit keeps Fara defaults like mrope_interleaved
        // + rms_norm_offset aligned with the Microsoft checkpoint.
        let mut b = Qwen35RunnerBuilder::default()
            .weights(&model_dir)
            .config(Qwen35ConfigSource::Explicit(
                crate::config::fara_qwen35_config(size),
            ))
            .device(device)
            .max_seq(max_seq)
            .runtime_mrope(true)
            .force_host_embed(true)
            .skip_warm(true);
        // Prefer streaming / host-release defaults for BF16 safetensors on
        // unified memory without overriding an explicit user env.
        if std::env::var_os("RLX_LOW_MEM_COMPILE").is_none() {
            unsafe { std::env::set_var("RLX_LOW_MEM_COMPILE", "1") };
        }
        if std::env::var_os("RLX_QWEN35_RELEASE_HOST_WEIGHTS").is_none() {
            unsafe { std::env::set_var("RLX_QWEN35_RELEASE_HOST_WEIGHTS", "1") };
        }
        if std::env::var_os("RLX_QWEN35_KEEP_PREFILL").is_none() {
            unsafe { std::env::set_var("RLX_QWEN35_KEEP_PREFILL", "0") };
        }
        eprintln!(
            "[rlx-fara] low-mem build: skip_warm=true max_seq={max_seq} host_embed=true \
             release_host_weights=1 (BF16→F32; prefer quantized GGUF when available)"
        );
        if let Some(ps) = self.prefill_seq {
            b = b.prefill_seq(ps);
        }
        let inner = b
            .build()
            .with_context(|| format!("FaraRunner: build from {}", model_dir.display()))?;
        if !inner.has_vision() {
            bail!(
                "FaraRunner: vision tower not loaded from {} — \
                 expected model.visual.* in safetensors",
                model_dir.display()
            );
        }
        Ok(FaraRunner {
            inner,
            size,
            model_dir,
        })
    }
}

impl FaraRunner {
    pub fn builder() -> FaraRunnerBuilder {
        FaraRunnerBuilder::default()
    }

    pub fn from_model_dir(dir: impl AsRef<Path>, size: FaraSize, device: Device) -> Result<Self> {
        Self::builder()
            .model_dir(dir.as_ref())
            .size(size)
            .device(device)
            .build()
    }

    pub fn size(&self) -> FaraSize {
        self.size
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn inner(&self) -> &Qwen35Runner {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut Qwen35Runner {
        &mut self.inner
    }

    /// Generate the next agent action from a screenshot RGB buffer.
    pub fn step(
        &mut self,
        goal: &str,
        rgb: &[u8],
        img_w: usize,
        img_h: usize,
        max_tokens: usize,
        tokenizer_path: Option<&Path>,
    ) -> Result<FaraStep> {
        let prompt = format_fara_multimodal_prompt(self.size, goal);
        let tok = tokenizer_path.map(PathBuf::from).or_else(|| {
            let p = self.model_dir.join("tokenizer.json");
            p.is_file().then_some(p)
        });
        let new_ids = self.inner.generate_multimodal_with_opts(
            &prompt,
            rgb,
            img_w,
            img_h,
            tok.as_deref(),
            max_tokens,
            rlx_qwen35::SampleOpts::greedy(),
            |_| true,
        )?;
        eprintln!(
            "[rlx-fara] generated {} token ids: {:?}",
            new_ids.len(),
            &new_ids[..new_ids.len().min(32)]
        );
        let raw_text = decode_ids(&self.model_dir, &new_ids)?;
        let tool_calls = parse_tool_calls(&raw_text).unwrap_or_default();
        let thinking = text_before_tool_calls(&raw_text).to_string();
        Ok(FaraStep {
            raw_text,
            thinking,
            tool_calls,
        })
    }
}

fn decode_ids(model_dir: &Path, ids: &[u32]) -> Result<String> {
    #[cfg(feature = "tokenizer")]
    {
        let path = model_dir.join("tokenizer.json");
        if path.is_file() {
            let tok = tokenizers::Tokenizer::from_file(&path)
                .map_err(|e| anyhow::anyhow!("load tokenizer {path:?}: {e}"))?;
            return tok
                .decode(ids, true)
                .map_err(|e| anyhow::anyhow!("decode: {e}"));
        }
    }
    Ok(ids
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(" "))
}
