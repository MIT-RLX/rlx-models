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

//! High-level end-to-end Florence-2 session: image + task prompt → parsed
//! result, wiring preprocessing, the model graphs, generation, and the task
//! post-processor.

use crate::config::Florence2Config;
use crate::generate::GenerateConfig;
use crate::postprocess::{Florence2Result, post_process};
use crate::preprocess::preprocess_rgb;
use crate::runner::Florence2Model;
use crate::tokenizer::Florence2Tokenizer;
use anyhow::{Context, Result};
use rlx_runtime::Device;
use std::path::Path;

/// Image side length the checkpoint expects (768²).
pub const IMAGE_SIZE: usize = 768;

pub struct Florence2Session {
    model: Florence2Model,
    tokenizer: Florence2Tokenizer,
    image_size: usize,
}

impl Florence2Session {
    /// Load model weights + tokenizer from a checkpoint directory.
    pub fn from_dir(dir: &Path, device: Device) -> Result<Self> {
        let cfg = Florence2Config::from_hf_config_json(&dir.join("config.json"))
            .with_context(|| format!("load config from {}", dir.display()))?;
        let model = Florence2Model::load(dir, cfg, device)?;
        let tokenizer = Florence2Tokenizer::from_file(&dir.join("tokenizer.json"))?;
        Ok(Self {
            model,
            tokenizer,
            image_size: IMAGE_SIZE,
        })
    }

    pub fn config(&self) -> &Florence2Config {
        self.model.config()
    }
    pub fn tokenizer(&self) -> &Florence2Tokenizer {
        &self.tokenizer
    }

    /// Run a task on an RGB-HWC `u8` image of arbitrary size. Returns the
    /// generated token ids (including start/EOS).
    pub fn run_tokens(
        &mut self,
        rgb: &[u8],
        in_h: usize,
        in_w: usize,
        task: &str,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>> {
        let pixel = preprocess_rgb(rgb, in_h, in_w, self.image_size);
        self.run_pixel_values(&pixel, task, max_new_tokens)
    }

    /// Run a task given preprocessed `pixel_values` (NCHW `[3·size·size]`).
    pub fn run_pixel_values(
        &mut self,
        pixel: &[f32],
        task: &str,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>> {
        let feats = self.model.encode_image(pixel, self.image_size)?;
        let input_ids = self.tokenizer.encode_prompt(task)?;
        let text = self.model.embed_text(&input_ids);
        let merged = self.model.merge_embeds(&feats, &text);
        let seq = feats.len() / self.model.config().text.d_model + input_ids.len();
        let enc = self.model.encode(&merged, seq)?;
        let gcfg = GenerateConfig::from_text(&self.model.config().text, max_new_tokens);
        if gcfg.num_beams > 1 {
            self.model.generate_beam(&enc, seq, &gcfg)
        } else {
            self.model.generate_greedy(&enc, seq, &gcfg)
        }
    }

    /// Full pipeline → parsed task result (caption / boxes / quads / polygons).
    pub fn run(
        &mut self,
        rgb: &[u8],
        in_h: usize,
        in_w: usize,
        task: &str,
        max_new_tokens: usize,
    ) -> Result<Florence2Result> {
        let ids = self.run_tokens(rgb, in_h, in_w, task, max_new_tokens)?;
        // Strip the leading start/BOS tokens before post-processing; HF parses
        // the decoded generation text including its location tokens.
        post_process(task, &ids, &self.tokenizer, (in_w as f64, in_h as f64))
    }
}
