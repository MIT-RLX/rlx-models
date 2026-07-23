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

//! Wires deep encoder + projector + compiled MoE LM for generation.
//!
//! Vision (SAM/CLIP/projector pack) stays host-f32; the MoE LM runs as a
//! compiled RLX graph on [`Self::device`] (including CPU).

use crate::config::{EOS_TOKEN_ID, IMAGE_TOKEN_ID, UnlimitedOcrConfig};
use crate::deep_encoder::DeepEncoder;
use crate::expert_pack::PackedLmWeights;
use crate::generation::{SampleOpts, sample_token};
use crate::lm_device::CompiledLm;
use crate::lm_precision::{LmWeightPrecision, ResolvedLmPrecision};
use crate::preprocess::PreprocessedImage;
use crate::projector::Projector;
use crate::weights::UnlimitedOcrWeightStore;
use anyhow::Result;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Runner construction options (device + LM weight storage precision).
#[derive(Debug, Clone)]
pub struct RunnerOptions {
    pub device: Device,
    /// Host pack precision for MoE / large mats. Default [`LmWeightPrecision::Auto`].
    pub weight_precision: LmWeightPrecision,
}

impl RunnerOptions {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            weight_precision: LmWeightPrecision::Auto,
        }
    }

    pub fn weight_precision(mut self, p: LmWeightPrecision) -> Self {
        self.weight_precision = p;
        self
    }
}

pub struct UnlimitedOcrRunner {
    model_dir: PathBuf,
    device: Device,
    weight_precision: LmWeightPrecision,
    config: UnlimitedOcrConfig,
    store: UnlimitedOcrWeightStore,
    encoder: DeepEncoder,
    projector: Projector,
    lm: Option<CompiledLm>,
    packed: Option<Arc<PackedLmWeights>>,
    loaded: bool,
}

impl UnlimitedOcrRunner {
    pub fn open(model_dir: &Path, device: Device) -> Result<Self> {
        Self::open_with(model_dir, RunnerOptions::new(device))
    }

    pub fn open_with(model_dir: &Path, opts: RunnerOptions) -> Result<Self> {
        let config = UnlimitedOcrConfig::from_model_dir(model_dir)?;
        config.validate()?;
        let store = UnlimitedOcrWeightStore::open(model_dir)?;
        let encoder = DeepEncoder::from_config(&config);
        let projector = Projector::from_config(&config.projector);
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            device: opts.device,
            weight_precision: opts.weight_precision,
            config,
            store,
            encoder,
            projector,
            lm: None,
            packed: None,
            loaded: false,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn weight_precision(&self) -> LmWeightPrecision {
        self.weight_precision
    }

    pub fn resolved_weight_precision(&self) -> Option<ResolvedLmPrecision> {
        self.packed.as_ref().map(|p| p.resolved_precision)
    }

    pub fn config(&self) -> &UnlimitedOcrConfig {
        &self.config
    }

    pub fn store(&self) -> &UnlimitedOcrWeightStore {
        &self.store
    }

    pub fn set_weight_precision(&mut self, p: LmWeightPrecision) {
        if self.weight_precision != p {
            self.weight_precision = p;
            // Force reload so the next generate() re-packs.
            self.loaded = false;
            self.lm = None;
            self.packed = None;
        }
    }

    pub fn load_weights(&mut self) -> Result<()> {
        if self.loaded {
            return Ok(());
        }
        self.encoder.load(&self.store)?;
        self.projector.load(&self.store)?;
        let packed = Arc::new(PackedLmWeights::from_store_with_precision(
            &self.store,
            &self.config,
            self.weight_precision,
        )?);
        eprintln!(
            "[rlx-unlimited-ocr] packed LM host cache ≈ {:.1} GiB ({})",
            packed.host_nbytes() as f64 / (1024.0 * 1024.0 * 1024.0),
            packed.resolved_precision,
        );
        self.lm = Some(CompiledLm::new(self.device, Arc::clone(&packed)));
        self.packed = Some(packed);
        self.loaded = true;
        Ok(())
    }

    pub fn build_prompt_ids(&self, prompt: &str, images: &[PreprocessedImage]) -> Result<Vec<u32>> {
        #[cfg(feature = "tokenizer")]
        {
            crate::tokenizer::build_prompt_ids(
                &self.model_dir,
                prompt,
                images,
                self.config.bos_token_id,
                IMAGE_TOKEN_ID,
            )
        }
        #[cfg(not(feature = "tokenizer"))]
        {
            let _ = (prompt, images);
            anyhow::bail!("enable feature `tokenizer` to build prompt ids")
        }
    }

    /// Returns `(decoded_text, full_token_ids, prompt_len)`.
    pub fn generate(
        &mut self,
        prompt: &str,
        images: &[PreprocessedImage],
        opts: &SampleOpts,
    ) -> Result<(String, Vec<u32>, usize)> {
        self.load_weights()?;
        let prompt_ids = self.build_prompt_ids(prompt, images)?;
        let prompt_len = prompt_ids.len();

        let vision = self.encoder.encode_and_project(images, &self.projector)?;
        let lm = self.lm.as_mut().expect("lm loaded");
        let mut inputs_embeds = lm.embed_tokens(&prompt_ids)?;
        crate::embed::fuse_inputs_embeds(
            &prompt_ids,
            &mut inputs_embeds,
            self.config.hidden_size,
            IMAGE_TOKEN_ID,
            &vision,
        )?;

        let mut token_ids = prompt_ids;
        let (mut logits, mut kv) = lm.prefill(&inputs_embeds, token_ids.len())?;

        for _ in 0..opts.max_new_tokens {
            let next = sample_token(&logits, opts, &token_ids);
            token_ids.push(next);
            if next == self.config.eos_token_id || next == EOS_TOKEN_ID {
                break;
            }
            let step_embed = lm.embed_tokens(&[next])?;
            let pos = token_ids.len() - 1;
            logits = lm.decode_step(&step_embed, pos, &mut kv)?;
        }

        #[cfg(feature = "tokenizer")]
        let text = crate::tokenizer::decode(&self.model_dir, &token_ids[prompt_len..])?;
        #[cfg(not(feature = "tokenizer"))]
        let text = format!("<tokens:{}>", token_ids.len() - prompt_len);

        Ok((text, token_ids, prompt_len))
    }
}
