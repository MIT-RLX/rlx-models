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

//! MiniMax-M3 vision-language runner — image → tower → projector → splice into
//! text embeddings → prefill.
//!
//! This is the "embeds-driven" VL prefill (à la Qwen2.5-VL / Llama-4): the
//! vision tower + projector produce `[n_img, hidden]` image features, the host
//! gathers the prompt's token embeddings, overwrites the `image_token_index`
//! placeholder rows with the image features, and feeds the assembled matrix as
//! `inputs_embeds` to the text decoder. Correctness-first (re-prefill, no
//! KV-cache), matching [`super::runner::MiniMaxM3Runner`].

use anyhow::{Result, anyhow};
use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;

use super::config::{M3VisionConfig, MiniMaxM3Config};
use super::flow::build_m3_text_embeds_flow;
use super::rope_tables;
use super::vision::{build_m3_projector_flow, build_m3_vision_flow, vision_rope_tables};
use super::weights::Snapshot;

/// A single image: pre-patchified pixels `[num_patches, patch_dim]` (flattened)
/// plus its `(t, h, w)` patch grid.
#[derive(Debug, Clone)]
pub struct ImageInput {
    /// Row-major `[num_patches · patch_dim]` pre-patchified pixels.
    pub pixel_values: Vec<f32>,
    /// Temporal grid extent (`1` for a single image).
    pub grid_t: usize,
    /// Patch-grid height.
    pub grid_h: usize,
    /// Patch-grid width.
    pub grid_w: usize,
}

impl ImageInput {
    pub fn num_patches(&self) -> usize {
        self.grid_t * self.grid_h * self.grid_w
    }
}

pub struct MiniMaxM3VlRunner {
    text_cfg: MiniMaxM3Config,
    vision_cfg: M3VisionConfig,
    snapshot: Snapshot,
    device: Device,
    text_cache: HashMap<usize, CompiledGraph>,
    vision_cache: HashMap<usize, CompiledGraph>,
    projector_cache: HashMap<usize, CompiledGraph>,
}

impl MiniMaxM3VlRunner {
    pub fn from_snapshot(
        text_cfg: MiniMaxM3Config,
        vision_cfg: M3VisionConfig,
        snapshot: Snapshot,
        device: Device,
    ) -> Self {
        Self {
            text_cfg,
            vision_cfg,
            snapshot,
            device,
            text_cache: HashMap::new(),
            vision_cache: HashMap::new(),
            projector_cache: HashMap::new(),
        }
    }

    pub fn text_config(&self) -> &MiniMaxM3Config {
        &self.text_cfg
    }

    /// Host-gather token embeddings for `ids` → `[seq · hidden]` row-major.
    pub fn embed_text(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let hidden = self.text_cfg.hidden_size;
        let (w, shape) = self
            .snapshot
            .get("model.embed_tokens.weight")
            .ok_or_else(|| anyhow!("missing model.embed_tokens.weight"))?;
        if shape.len() != 2 || shape[1] != hidden {
            return Err(anyhow!(
                "embed_tokens.weight shape {shape:?} != [vocab, {hidden}]"
            ));
        }
        let vocab = shape[0];
        let mut out = vec![0f32; ids.len() * hidden];
        for (i, &t) in ids.iter().enumerate() {
            let t = t as usize;
            if t >= vocab {
                return Err(anyhow!("token id {t} >= vocab {vocab}"));
            }
            out[i * hidden..(i + 1) * hidden].copy_from_slice(&w[t * hidden..(t + 1) * hidden]);
        }
        Ok(out)
    }

    /// Run the vision tower + projector for one image → `[n_out · hidden]`
    /// image features (`n_out = num_patches / spatial_merge_size²`).
    pub fn encode_image(&mut self, img: &ImageInput) -> Result<Vec<f32>> {
        let np = img.num_patches();
        let embed = self.vision_cfg.hidden_size;
        let axis_dim = self.vision_cfg.axis_dim();
        let theta = self.vision_cfg.rope_theta;

        // Vision tower.
        if !self.vision_cache.contains_key(&np) {
            let mut wm = WeightMap::from_tensors(self.snapshot.clone());
            let built = build_m3_vision_flow(&self.vision_cfg, &mut wm, np)?;
            self.vision_cache
                .insert(np, compile_built(built, self.device)?);
        }
        let (vcos, vsin) = vision_rope_tables(img.grid_t, img.grid_h, img.grid_w, axis_dim, theta);
        let vision_hidden = {
            let g = self.vision_cache.get_mut(&np).expect("vision graph");
            let mut out = g.run(&[
                ("pixel_values", img.pixel_values.as_slice()),
                ("vcos", vcos.as_slice()),
                ("vsin", vsin.as_slice()),
            ]);
            out.drain(..)
                .next()
                .ok_or_else(|| anyhow!("vision produced no output"))?
        };
        if vision_hidden.len() != np * embed {
            return Err(anyhow!(
                "vision hidden len {} != np·embed {}",
                vision_hidden.len(),
                np * embed
            ));
        }

        // Projector.
        if !self.projector_cache.contains_key(&np) {
            let mut wm = WeightMap::from_tensors(self.snapshot.clone());
            let built = build_m3_projector_flow(&self.vision_cfg, &mut wm, np)?;
            self.projector_cache
                .insert(np, compile_built(built, self.device)?);
        }
        let feats = {
            let g = self.projector_cache.get_mut(&np).expect("projector graph");
            let mut out = g.run(&[("vision_hidden", vision_hidden.as_slice())]);
            out.drain(..)
                .next()
                .ok_or_else(|| anyhow!("projector produced no output"))?
        };
        Ok(feats)
    }

    fn ensure_text_compiled(&mut self, seq: usize) -> Result<()> {
        if self.text_cache.contains_key(&seq) {
            return Ok(());
        }
        let mut wm = WeightMap::from_tensors(self.snapshot.clone());
        let built = build_m3_text_embeds_flow(&self.text_cfg, &mut wm, seq, true)?;
        self.text_cache
            .insert(seq, compile_built(built, self.device)?);
        Ok(())
    }

    /// Prefill over a precomputed `inputs_embeds [seq · hidden]`; last-token logits.
    pub fn forward_embeds(&mut self, embeds: &[f32], seq: usize) -> Result<Vec<f32>> {
        let hidden = self.text_cfg.hidden_size;
        let vocab = self.text_cfg.vocab_size;
        if embeds.len() != seq * hidden {
            return Err(anyhow!(
                "embeds len {} != seq·hidden {}",
                embeds.len(),
                seq * hidden
            ));
        }
        let n_rot = self.text_cfg.n_rot();
        let theta = self.text_cfg.rope_theta;
        self.ensure_text_compiled(seq)?;
        let (cos, sin) = rope_tables(seq, n_rot, theta);
        let g = self.text_cache.get_mut(&seq).expect("text graph");
        let mut out = g.run(&[
            ("inputs_embeds", embeds),
            ("rope_cos", cos.as_slice()),
            ("rope_sin", sin.as_slice()),
        ]);
        let logits = out
            .drain(..)
            .next()
            .ok_or_else(|| anyhow!("text produced no output"))?;
        if logits.len() != seq * vocab {
            return Err(anyhow!(
                "logits len {} != seq·vocab {}",
                logits.len(),
                seq * vocab
            ));
        }
        Ok(logits[(seq - 1) * vocab..seq * vocab].to_vec())
    }

    /// Assemble `inputs_embeds` for `prompt_ids`, overwriting the rows at
    /// `image_positions` with the (row-major `[n·hidden]`) `image_features`.
    pub fn assemble_embeds(
        &self,
        prompt_ids: &[u32],
        image_positions: &[usize],
        image_features: &[f32],
    ) -> Result<Vec<f32>> {
        let hidden = self.text_cfg.hidden_size;
        let mut embeds = self.embed_text(prompt_ids)?;
        if image_features.len() != image_positions.len() * hidden {
            return Err(anyhow!(
                "image_features len {} != positions {}·hidden {}",
                image_features.len(),
                image_positions.len(),
                hidden
            ));
        }
        for (i, &pos) in image_positions.iter().enumerate() {
            if pos >= prompt_ids.len() {
                return Err(anyhow!(
                    "image position {pos} >= prompt len {}",
                    prompt_ids.len()
                ));
            }
            embeds[pos * hidden..(pos + 1) * hidden]
                .copy_from_slice(&image_features[i * hidden..(i + 1) * hidden]);
        }
        Ok(embeds)
    }

    /// Prefill a multimodal prompt: gather text embeds, splice `image` features
    /// at `image_positions`, run the decoder → last-token logits.
    pub fn prefill_multimodal(
        &mut self,
        prompt_ids: &[u32],
        image_positions: &[usize],
        image: &ImageInput,
    ) -> Result<Vec<f32>> {
        let feats = self.encode_image(image)?;
        let embeds = self.assemble_embeds(prompt_ids, image_positions, &feats)?;
        self.forward_embeds(&embeds, prompt_ids.len())
    }

    /// Greedy multimodal generation: prefill the spliced prompt, then re-prefill
    /// with each new token's text embedding appended. Returns produced ids.
    pub fn generate_multimodal(
        &mut self,
        prompt_ids: &[u32],
        image_positions: &[usize],
        image: &ImageInput,
        n_new: usize,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        let hidden = self.text_cfg.hidden_size;
        let feats = self.encode_image(image)?;
        let mut embeds = self.assemble_embeds(prompt_ids, image_positions, &feats)?;
        let mut seq = prompt_ids.len();
        let mut produced = Vec::with_capacity(n_new);
        for _ in 0..n_new {
            let logits = self.forward_embeds(&embeds, seq)?;
            let next = argmax(&logits);
            produced.push(next);
            if !on_token(next) {
                break;
            }
            // Append the new token's text embedding (subsequent tokens are text).
            let row = self.embed_text(&[next])?;
            embeds.extend_from_slice(&row);
            seq += 1;
            debug_assert_eq!(embeds.len(), seq * hidden);
        }
        Ok(produced)
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best as u32
}
