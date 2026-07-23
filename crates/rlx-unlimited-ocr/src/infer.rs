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

//! High-level inference session.

use crate::device::resolve_device;
use crate::generation::SampleOpts;
use crate::hub::default_model_dir;
use crate::lm_precision::LmWeightPrecision;
use crate::preprocess::{ImageMode, pdf_to_page_images, preprocess_path, preprocess_paths};
use crate::runner::{RunnerOptions, UnlimitedOcrRunner};
use anyhow::{Context, Result};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    Single,
    Multi,
    Pdf,
}

#[derive(Debug, Clone)]
pub struct InferenceOptions {
    pub device: Device,
    pub sample: SampleOpts,
    pub mode: ImageMode,
    pub prompt: String,
    pub preload: bool,
    /// MoE / large-mat host storage. Default [`LmWeightPrecision::Auto`].
    pub weight_precision: LmWeightPrecision,
}

impl InferenceOptions {
    pub fn for_ocr() -> Self {
        Self {
            device: resolve_device(None).unwrap_or(Device::Cpu),
            sample: SampleOpts::default(),
            mode: ImageMode::default(),
            prompt: "<image>document parsing.".into(),
            preload: false,
            weight_precision: LmWeightPrecision::Auto,
        }
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn device_name(mut self, name: &str) -> Result<Self> {
        self.device = resolve_device(Some(name))?;
        Ok(self)
    }

    pub fn mode(mut self, mode: ImageMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    pub fn max_new_tokens(mut self, n: usize) -> Self {
        self.sample.max_new_tokens = n;
        self
    }

    pub fn weight_precision(mut self, p: LmWeightPrecision) -> Self {
        self.weight_precision = p;
        self
    }

    pub fn preload(mut self, preload: bool) -> Self {
        self.preload = preload;
        self
    }
}

#[derive(Debug, Clone, Default)]
pub struct OcrResult {
    pub text: String,
    pub pages: Vec<String>,
    pub prompt_len: usize,
    pub new_tokens: usize,
    pub token_ids: Vec<u32>,
}

pub struct UnlimitedOcrSession {
    runner: UnlimitedOcrRunner,
    options: InferenceOptions,
}

impl UnlimitedOcrSession {
    pub fn open(model_dir: impl AsRef<Path>, options: InferenceOptions) -> Result<Self> {
        let mut runner = UnlimitedOcrRunner::open_with(
            model_dir.as_ref(),
            RunnerOptions::new(options.device).weight_precision(options.weight_precision),
        )?;
        if options.preload {
            runner.load_weights()?;
        }
        Ok(Self { runner, options })
    }

    pub fn open_default() -> Result<Self> {
        Self::open(default_model_dir()?, InferenceOptions::for_ocr())
    }

    pub fn device(&self) -> Device {
        self.runner.device()
    }

    pub fn model_dir(&self) -> &Path {
        self.runner.model_dir()
    }

    pub fn runner(&self) -> &UnlimitedOcrRunner {
        &self.runner
    }

    pub fn runner_mut(&mut self) -> &mut UnlimitedOcrRunner {
        &mut self.runner
    }

    pub fn run_single(&mut self, image_path: impl AsRef<Path>) -> Result<OcrResult> {
        let image = preprocess_path(image_path.as_ref(), self.options.mode)?;
        let (text, ids, prompt_len) =
            self.runner
                .generate(&self.options.prompt, &[image], &self.options.sample)?;
        Ok(OcrResult {
            text: text.clone(),
            pages: vec![text],
            prompt_len,
            new_tokens: ids.len().saturating_sub(prompt_len),
            token_ids: ids,
        })
    }

    pub fn run_multi(&mut self, image_paths: &[PathBuf]) -> Result<OcrResult> {
        let mode = ImageMode::Multi { size: 1024 };
        let batch = preprocess_paths(image_paths, mode)?;
        let mut sample = SampleOpts::multi_page();
        sample.max_new_tokens = self.options.sample.max_new_tokens;
        let prompt = if self.options.prompt.contains("<image>") {
            self.options.prompt.clone()
        } else {
            format!("<image>{}", self.options.prompt)
        };
        let (text, ids, prompt_len) = self.runner.generate(&prompt, &batch.images, &sample)?;
        let pages: Vec<String> = text
            .split("<PAGE>")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        Ok(OcrResult {
            text,
            pages,
            prompt_len,
            new_tokens: ids.len().saturating_sub(prompt_len),
            token_ids: ids,
        })
    }

    pub fn run_pdf(&mut self, pdf_path: impl AsRef<Path>) -> Result<OcrResult> {
        let tmp = std::env::temp_dir().join(format!(
            "rlx-unlimited-ocr-{}",
            pdf_path
                .as_ref()
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("doc")
        ));
        let pages = pdf_to_page_images(pdf_path.as_ref(), &tmp)
            .with_context(|| format!("rasterize PDF {:?}", pdf_path.as_ref()))?;
        self.run_multi(&pages)
    }
}
