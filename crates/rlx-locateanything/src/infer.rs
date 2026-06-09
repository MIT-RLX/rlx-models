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

//! High-level inference session — open once, ground many images with minimal code.
//!
//! Default image for quick runs: [`crate::fixtures::sample_image_path`].
//! Full CLI and env documentation: `README.md` in this crate.

use crate::config::LocateAnythingConfig;
use crate::device::resolve_device;
use crate::generation::{GenerationMode, SampleOpts};
use crate::hub::{default_model_dir, resolve_weights_path};
use crate::parse::{GroundingParse, parse_grounding};
use crate::preprocess::{PreprocessedImage, preprocess_image};
use crate::prompts;
use crate::runner::{GenerateProfile, LocateAnythingRunner};
use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView};
use rlx_runtime::Device;
use std::path::Path;

/// How user text + vision placeholders are assembled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PromptStyle {
    /// HF `LocateAnythingProcessor` layout (recommended for boxes).
    #[default]
    Processor,
    /// RLX minimal Qwen chat (`<|im_start|>user` + raw image slots).
    Rlx,
}

/// Inference settings (device, decode, prompt, optional resize for speed).
#[derive(Debug, Clone)]
pub struct InferenceOptions {
    pub device: Device,
    pub generation_mode: GenerationMode,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub top_p: f32,
    pub prompt_style: PromptStyle,
    /// Resize so the longest edge is at most this many pixels before patchify (faster).
    pub max_image_side: Option<u32>,
    /// Compile LM graphs on [`LocateAnythingSession::open`].
    pub preload_language_model: bool,
}

impl InferenceOptions {
    /// Greedy hybrid defaults suited to grounding (processor prompt).
    pub fn for_grounding() -> Self {
        Self {
            device: resolve_device(None).unwrap_or(Device::Cpu),
            generation_mode: GenerationMode::Hybrid,
            max_new_tokens: 64,
            temperature: 0.0,
            repetition_penalty: 1.0,
            top_p: 1.0,
            prompt_style: PromptStyle::Processor,
            max_image_side: None,
            preload_language_model: false,
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

    pub fn max_new_tokens(mut self, n: usize) -> Self {
        self.max_new_tokens = n;
        self
    }

    pub fn generation_mode(mut self, mode: GenerationMode) -> Self {
        self.generation_mode = mode;
        self
    }

    pub fn prompt_style(mut self, style: PromptStyle) -> Self {
        self.prompt_style = style;
        self
    }

    pub fn max_image_side(mut self, side: u32) -> Self {
        self.max_image_side = Some(side);
        self
    }

    pub fn preload_language_model(mut self, yes: bool) -> Self {
        self.preload_language_model = yes;
        self
    }

    fn sample_opts(&self) -> SampleOpts {
        SampleOpts {
            temperature: self.temperature,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            max_new_tokens: self.max_new_tokens,
            mode: self.generation_mode,
        }
    }
}

/// Parsed grounding output (boxes, points, ref labels).
pub type GroundingResult = GroundingParse;

/// Loaded model + compile caches for repeated grounding calls.
pub struct LocateAnythingSession {
    runner: LocateAnythingRunner,
    cfg: LocateAnythingConfig,
    options: InferenceOptions,
    #[cfg(feature = "tokenizer")]
    tokenizer: tokenizers::Tokenizer,
}

impl LocateAnythingSession {
    /// Open from [`default_model_dir`] (HF cache, env, or local fetch layout).
    pub fn open_default() -> Result<Self> {
        Self::open(default_model_dir()?)
    }

    /// Open checkpoint at `model_dir` (path, `hf`, or Hub id `nvidia/LocateAnything-3B`).
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_options(model_dir, InferenceOptions::for_grounding())
    }

    /// Open with full control over device, decode, and prompt layout.
    pub fn open_with_options(
        model_dir: impl AsRef<Path>,
        options: InferenceOptions,
    ) -> Result<Self> {
        let dir = resolve_weights_path(model_dir.as_ref())?;
        let cfg = LocateAnythingConfig::from_file(&dir.join("config.json"))
            .with_context(|| format!("load config from {dir:?}"))?;
        cfg.validate()?;

        let sample = options.sample_opts();
        let mut runner = LocateAnythingRunner::builder()
            .weights(&dir)
            .device(options.device)
            .max_new_tokens(sample.max_new_tokens)
            .generation_mode(sample.mode)
            .temperature(sample.temperature)
            .repetition_penalty(sample.repetition_penalty)
            .build()?;

        if options.preload_language_model {
            runner.preload_language_model()?;
        }

        #[cfg(feature = "tokenizer")]
        let tokenizer = crate::tokenizer::load_tokenizer(&dir)?;

        Ok(Self {
            runner,
            cfg,
            options,
            #[cfg(feature = "tokenizer")]
            tokenizer,
        })
    }

    pub fn model_dir(&self) -> &Path {
        self.runner.model_dir()
    }

    pub fn device(&self) -> Device {
        self.options.device
    }

    pub fn options(&self) -> &InferenceOptions {
        &self.options
    }

    pub fn runner(&self) -> &LocateAnythingRunner {
        &self.runner
    }

    pub fn runner_mut(&mut self) -> &mut LocateAnythingRunner {
        &mut self.runner
    }

    pub fn preprocess_dynamic(&self, img: &DynamicImage) -> Result<PreprocessedImage> {
        let img = maybe_resize(img, self.options.max_image_side);
        preprocess_image(&img, &self.cfg)
    }

    pub fn preprocess_file(&self, path: impl AsRef<Path>) -> Result<PreprocessedImage> {
        let img = image::open(path.as_ref())?;
        self.preprocess_dynamic(&img)
    }

    /// Compile MoonViT + projector + LM prefill for this image/prompt shape (amortize first real query).
    pub fn warmup(&mut self, image: &PreprocessedImage, phrase: &str) -> Result<()> {
        let prompt_ids = self.build_prompt_ids(image, phrase)?;
        self.runner.warmup_compile(&prompt_ids, image)
    }

    /// Ground a single phrase in one image (returns boxes when the model emits them).
    #[cfg(feature = "tokenizer")]
    pub fn ground(&mut self, image: &PreprocessedImage, phrase: &str) -> Result<GroundingResult> {
        self.ground_with_profile(image, phrase).map(|(r, _)| r)
    }

    /// Like [`Self::ground`] but also returns per-phase generation timings.
    #[cfg(feature = "tokenizer")]
    pub fn ground_with_profile(
        &mut self,
        image: &PreprocessedImage,
        phrase: &str,
    ) -> Result<(GroundingResult, GenerateProfile)> {
        let (w, h) = (image.pixel_w, image.pixel_h);
        let prompt_ids = self.build_prompt_ids(image, phrase)?;
        let (tokens, profile) = self.runner.generate_with_profile(&prompt_ids, image)?;
        let result = self.decode_grounding(&tokens, prompt_ids.len(), w, h)?;
        Ok((result, profile))
    }

    #[cfg(feature = "tokenizer")]
    pub fn ground_path(&mut self, path: impl AsRef<Path>, phrase: &str) -> Result<GroundingResult> {
        let prep = self.preprocess_file(path)?;
        self.ground(&prep, phrase)
    }

    #[cfg(feature = "tokenizer")]
    pub fn ground_dynamic(&mut self, img: &DynamicImage, phrase: &str) -> Result<GroundingResult> {
        let prep = self.preprocess_dynamic(img)?;
        self.ground(&prep, phrase)
    }

    #[cfg(feature = "tokenizer")]
    pub fn detect(
        &mut self,
        image: &PreprocessedImage,
        categories: &[&str],
    ) -> Result<GroundingResult> {
        self.ground(image, &prompts::detect(categories))
    }

    #[cfg(feature = "tokenizer")]
    fn build_prompt_ids(&self, image: &PreprocessedImage, user_text: &str) -> Result<Vec<u32>> {
        let kh = self.cfg.vision_config.merge_kernel_size[0];
        let kw = self.cfg.vision_config.merge_kernel_size[1];
        let n_image = (image.grid_h / kh) * (image.grid_w / kw);
        match self.options.prompt_style {
            PromptStyle::Processor => {
                let with_ph = if user_text.starts_with("<image-1>") {
                    user_text.to_string()
                } else {
                    format!("<image-1>{user_text}")
                };
                crate::processor_prompt::build_processor_prompt_ids(
                    self.runner.model_dir(),
                    &self.cfg,
                    &self.tokenizer,
                    &with_ph,
                    n_image,
                )
            }
            PromptStyle::Rlx => crate::tokenizer::build_user_prompt_ids(
                &self.cfg,
                &self.tokenizer,
                user_text,
                n_image,
            ),
        }
    }

    #[cfg(feature = "tokenizer")]
    fn decode_grounding(
        &self,
        tokens: &[u32],
        prompt_len: usize,
        width: u32,
        height: u32,
    ) -> Result<GroundingResult> {
        let new = &tokens[prompt_len..];
        let text = crate::tokenizer::decode(&self.tokenizer, new)?;
        let raw = self
            .tokenizer
            .decode(new, false)
            .unwrap_or_else(|_| text.clone());
        let mut parsed = parse_grounding(&text, width, height);
        if parsed.boxes.is_empty() && raw != text {
            let from_raw = parse_grounding(&raw, width, height);
            if !from_raw.boxes.is_empty() || !from_raw.refs.is_empty() {
                parsed = from_raw;
            }
        }
        parsed.text = text;
        parsed.raw = raw;
        parsed.new_tokens = new.len();
        parsed.prompt_len = prompt_len;
        Ok(parsed)
    }
}

fn maybe_resize(img: &DynamicImage, max_side: Option<u32>) -> DynamicImage {
    let Some(max_side) = max_side else {
        return img.clone();
    };
    let (w, h) = img.dimensions();
    let longest = w.max(h);
    if longest <= max_side {
        return img.clone();
    }
    let scale = max_side as f32 / longest as f32;
    let nw = ((w as f32 * scale).round() as u32).max(1);
    let nh = ((h as f32 * scale).round() as u32).max(1);
    img.resize_exact(nw, nh, image::imageops::FilterType::Triangle)
}
