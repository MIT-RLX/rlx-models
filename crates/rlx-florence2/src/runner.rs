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

//! Staged execution of the Florence-2 pipeline: vision encode → host-side
//! text embedding + merge → BART encode → decoder logits → generation.

use crate::config::Florence2Config;
use crate::weight_source::CloningWeightSource;
use crate::weights::lang as lk;
use anyhow::{Result, anyhow};
use rayon::prelude::*;
use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;

/// Loaded Florence-2 model with lazily compiled graphs.
///
/// The decoder is compiled **once** at the generation cap (not per length) and
/// reused for every step by padding the prefix to that length — the causal mask
/// keeps each real position independent of trailing pad. The token embedding and
/// LM head run host-side from `shared_table`, so the heavy `[vocab,d]` matrix is
/// never cloned into a graph and per-step output is the small `[T,d]` hidden
/// state rather than a `[T,vocab]` logits tensor.
pub struct Florence2Model {
    cfg: Florence2Config,
    device: Device,
    weights: WeightMap,
    /// Host `shared.weight` `[vocab, d]` — text/decoder embedding + tied LM head.
    shared_table: Vec<f32>,
    /// Host `final_logits_bias` `[vocab]`.
    final_logits_bias: Vec<f32>,
    vision: Option<CompiledGraph>,
    encoders: HashMap<usize, CompiledGraph>,
    /// Decoder hidden-state graphs keyed by `(bucket, enc_seq)`.
    decoders: HashMap<usize, CompiledGraph>,
    /// Reused `[cap·d]` host buffer for decoder input embeddings.
    embed_scratch: Vec<f32>,
    /// Image side length the vision graph was compiled for.
    img_size: usize,
}

impl Florence2Model {
    pub fn config(&self) -> &Florence2Config {
        &self.cfg
    }
    pub fn device(&self) -> Device {
        self.device
    }

    /// Load from a safetensors checkpoint directory or file.
    pub fn load(
        weights_path: &std::path::Path,
        cfg: Florence2Config,
        device: Device,
    ) -> Result<Self> {
        rlx_core::validate_standard_device("florence2", device)?;
        let mut weights = if weights_path.is_dir() {
            WeightMap::from_safetensors_dir(weights_path)?
        } else {
            WeightMap::from_file(
                weights_path
                    .to_str()
                    .ok_or_else(|| anyhow!("non-UTF8 weights path"))?,
            )?
        };
        // Take (not clone) the embedding + bias host-side so the heavy
        // `[vocab,d]` matrix is stored once, not duplicated into the WeightMap
        // and a graph param.
        let (shared_table, _) = weights.take(lk::SHARED)?;
        let final_logits_bias = weights
            .take(lk::FINAL_LOGITS_BIAS)
            .map(|(d, _)| d)
            .unwrap_or_else(|_| vec![0.0; cfg.vocab_size]);
        Ok(Self {
            cfg,
            device,
            weights,
            shared_table,
            final_logits_bias,
            vision: None,
            encoders: HashMap::new(),
            decoders: HashMap::new(),
            embed_scratch: Vec::new(),
            img_size: 768,
        })
    }

    /// Run the DaViT vision tower on `pixel_values` (NCHW `[1,3,H,W]`).
    /// Returns image features `[image_seq · projection_dim]` (577 × 1024).
    pub fn encode_image(&mut self, pixel_values: &[f32], img_size: usize) -> Result<Vec<f32>> {
        if self.vision.is_none() || self.img_size != img_size {
            let mut src = CloningWeightSource(&self.weights);
            let built =
                crate::flow::build_vision_built(&self.cfg, &mut src, 1, img_size, img_size)?;
            self.vision = Some(compile_built(built, self.device)?);
            self.img_size = img_size;
        }
        let g = self.vision.as_mut().unwrap();
        let out = g.run(&[("pixel", pixel_values)]);
        out.into_iter()
            .next()
            .ok_or_else(|| anyhow!("vision graph produced no output"))
    }

    /// Build + compile a fresh (uncached) vision graph on `device` and run it.
    /// Honors `RLX_FLORENCE2_VISION_STOP_STAGE` for cross-backend bisection.
    pub fn debug_vision(&self, pixel: &[f32], img_size: usize, device: Device) -> Result<Vec<f32>> {
        let mut src = CloningWeightSource(&self.weights);
        let built = crate::flow::build_vision_built(&self.cfg, &mut src, 1, img_size, img_size)?;
        let mut g = compile_built(built, device)?;
        Ok(g.run(&[("pixel", pixel)]).into_iter().next().unwrap())
    }

    /// Embed text token ids host-side from the shared table (`embed_scale`).
    pub fn embed_text(&self, token_ids: &[u32]) -> Vec<f32> {
        let d = self.cfg.text.d_model;
        let scale = self.cfg.embed_scale();
        let mut out = vec![0f32; token_ids.len() * d];
        for (i, &tok) in token_ids.iter().enumerate() {
            let src = (tok as usize) * d;
            let dst = i * d;
            for j in 0..d {
                out[dst + j] = self.shared_table[src + j] * scale;
            }
        }
        out
    }

    /// Concatenate image features (`[img_seq, d]`) with text embeddings
    /// (`[txt_seq, d]`) → `inputs_embeds [(img_seq+txt_seq), d]`.
    pub fn merge_embeds(&self, image_features: &[f32], text_embeds: &[f32]) -> Vec<f32> {
        let mut merged = Vec::with_capacity(image_features.len() + text_embeds.len());
        merged.extend_from_slice(image_features);
        merged.extend_from_slice(text_embeds);
        merged
    }

    /// BART encoder over `inputs_embeds [seq · d]` → `encoder_hidden [seq · d]`.
    pub fn encode(&mut self, inputs_embeds: &[f32], seq: usize) -> Result<Vec<f32>> {
        if !self.encoders.contains_key(&seq) {
            let mut src = CloningWeightSource(&self.weights);
            let built = crate::flow::build_encoder_built(&self.cfg, &mut src, 1, seq)?;
            self.encoders
                .insert(seq, compile_built(built, self.device)?);
        }
        let g = self.encoders.get_mut(&seq).unwrap();
        let out = g.run(&[("inputs_embeds", inputs_embeds)]);
        out.into_iter()
            .next()
            .ok_or_else(|| anyhow!("encoder graph produced no output"))
    }

    /// Next-token logits for the last position of `token_ids`, using a decoder
    /// graph compiled once per bucket and reused across steps (the prefix is
    /// padded; the causal mask makes the last real position independent of the
    /// pad). The LM head is applied host-side to that one row. `encoder_hidden`
    /// is `[enc_seq · d]`. Returns `[vocab]`.
    pub fn decode_logits(
        &mut self,
        token_ids: &[u32],
        encoder_hidden: &[f32],
        enc_seq: usize,
        cap: usize,
    ) -> Result<Vec<f32>> {
        let d = self.cfg.text.d_model;
        let cur = token_ids.len();
        debug_assert!(cur >= 1 && cur <= cap);
        // Bucket the graph length to the smallest size ≥ the current prefix
        // (clamped to the generation cap): early steps don't pay for the full
        // `max_new_tokens` of padding and we compile only ~log(N) graphs.
        let cap = bucket_len(cur).min(cap).max(cur);

        let key = cap * 1_000_000 + enc_seq;
        if !self.decoders.contains_key(&key) {
            let mut src = CloningWeightSource(&self.weights);
            let built =
                crate::flow::build_decoder_hidden_built(&self.cfg, &mut src, 1, cap, enc_seq)?;
            self.decoders
                .insert(key, compile_built(built, self.device)?);
        }

        // Host token embedding into the reused scratch buffer; pad rows carry
        // the pad-token embedding (masked out, but keeps the buffer well-formed).
        let pad = self.cfg.text.pad_token_id as usize;
        self.embed_scratch.resize(cap * d, 0.0);
        for i in 0..cap {
            let tok = if i < cur { token_ids[i] as usize } else { pad };
            let src = tok * d;
            self.embed_scratch[i * d..(i + 1) * d]
                .copy_from_slice(&self.shared_table[src..src + d]);
        }

        let hidden = {
            let g = self.decoders.get_mut(&key).unwrap();
            g.run(&[
                ("decoder_inputs_embeds", self.embed_scratch.as_slice()),
                ("encoder_hidden", encoder_hidden),
            ])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("decoder graph produced no output"))?
        };
        let row = &hidden[(cur - 1) * d..cur * d];
        Ok(self.lm_head(row))
    }

    /// Convenience: next-token logits for a prefix (one graph sized to the
    /// prefix). Used by staged tests.
    pub fn decoder_logits(
        &mut self,
        token_ids: &[u32],
        encoder_hidden: &[f32],
        enc_seq: usize,
    ) -> Result<Vec<f32>> {
        self.decode_logits(token_ids, encoder_hidden, enc_seq, token_ids.len())
    }

    /// Tied LM head + `final_logits_bias` for a single hidden row `[d]`,
    /// parallelized over the vocabulary.
    fn lm_head(&self, hidden_row: &[f32]) -> Vec<f32> {
        let d = self.cfg.text.d_model;
        let vocab = self.cfg.vocab_size;
        let table = &self.shared_table;
        let bias = &self.final_logits_bias;
        let mut logits = vec![0f32; vocab];
        logits.par_iter_mut().enumerate().for_each(|(v, out)| {
            let row = &table[v * d..v * d + d];
            let mut acc = 0f32;
            for j in 0..d {
                acc += hidden_row[j] * row[j];
            }
            *out = acc + bias[v];
        });
        logits
    }
}

/// Decode graph length buckets: smallest entry ≥ `n` (else next power of two).
/// Keeps the number of compiled decoder graphs logarithmic in output length
/// and bounds per-step padding to roughly 2× the live prefix.
fn bucket_len(n: usize) -> usize {
    const BUCKETS: [usize; 11] = [4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256];
    for &b in &BUCKETS {
        if b >= n {
            return b;
        }
    }
    n.next_power_of_two()
}
