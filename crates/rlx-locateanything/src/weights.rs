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

//! Checkpoint tensor names for `nvidia/LocateAnything-3B` safetensors.

use anyhow::{Context, Result};
use rlx_core::weight_loader::WeightLoader;
use std::sync::Arc;

use crate::load::{LocateAnythingWeightStore, WeightSnapshot};

/// HF weight prefix helpers (`vision_model.*`, `mlp1.*`, `language_model.*`).
#[derive(Debug, Clone)]
pub struct LocateAnythingWeightPrefix;

impl LocateAnythingWeightPrefix {
    pub fn vision_block(i: usize, suffix: &str) -> String {
        format!("vision_model.encoder.blocks.{i}.{suffix}")
    }

    pub fn vision_patch_proj_w() -> &'static str {
        "vision_model.patch_embed.proj.weight"
    }

    pub fn vision_patch_proj_b() -> &'static str {
        "vision_model.patch_embed.proj.bias"
    }

    pub fn vision_pos_emb() -> &'static str {
        "vision_model.patch_embed.pos_emb.weight"
    }

    pub fn vision_final_ln_w() -> &'static str {
        "vision_model.encoder.final_layernorm.weight"
    }

    pub fn vision_final_ln_b() -> &'static str {
        "vision_model.encoder.final_layernorm.bias"
    }

    pub fn projector_ln_w() -> &'static str {
        "mlp1.0.weight"
    }

    pub fn projector_ln_b() -> &'static str {
        "mlp1.0.bias"
    }

    pub fn projector_fc1_w() -> &'static str {
        "mlp1.1.weight"
    }

    pub fn projector_fc1_b() -> &'static str {
        "mlp1.1.bias"
    }

    pub fn projector_fc2_w() -> &'static str {
        "mlp1.3.weight"
    }

    pub fn projector_fc2_b() -> &'static str {
        "mlp1.3.bias"
    }

    pub fn lm_embed_tokens() -> &'static str {
        "language_model.model.embed_tokens.weight"
    }

    pub fn lm_head() -> &'static str {
        "language_model.lm_head.weight"
    }

    pub fn lm_layer(i: usize, suffix: &str) -> String {
        format!("language_model.model.layers.{i}.{suffix}")
    }

    pub fn lm_norm() -> &'static str {
        "language_model.model.norm.weight"
    }
}

fn map_lm_key(key: &str) -> String {
    match key {
        "model.embed_tokens.weight" => LocateAnythingWeightPrefix::lm_embed_tokens().into(),
        "model.norm.weight" => LocateAnythingWeightPrefix::lm_norm().into(),
        "lm_head.weight" => LocateAnythingWeightPrefix::lm_head().into(),
        k if k.starts_with("model.layers.") => format!("language_model.{k}"),
        other => other.into(),
    }
}

/// Maps Qwen-shaped keys (`model.*`, `lm_head.*`) to HF `language_model.*` names.
pub struct LanguageModelPrefixLoader<'a> {
    inner: &'a mut dyn WeightLoader,
}

impl<'a> LanguageModelPrefixLoader<'a> {
    pub fn new(inner: &'a mut dyn WeightLoader) -> Self {
        Self { inner }
    }
}

impl WeightLoader for LanguageModelPrefixLoader<'_> {
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.inner.take(&map_lm_key(key))
    }

    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.inner.take_transposed(&map_lm_key(key))
    }

    fn remaining_keys(&self) -> Vec<String> {
        self.inner.remaining_keys()
    }
}

/// LM weights loaded on demand from mmap-backed safetensors (no full-RAM snapshot).
pub struct CheckpointLmWeightLoader {
    store: Arc<LocateAnythingWeightStore>,
}

impl CheckpointLmWeightLoader {
    pub fn new(store: Arc<LocateAnythingWeightStore>) -> Self {
        Self { store }
    }

    fn take_hf(&self, hf: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let mut wm = self
            .store
            .load_keys(&[hf])
            .with_context(|| format!("load LM weight {hf}"))?;
        wm.take(hf)
            .with_context(|| format!("missing LM weight {hf} after load"))
    }
}

impl WeightLoader for CheckpointLmWeightLoader {
    fn len(&self) -> usize {
        self.store
            .count_keys_with_prefix(crate::load::PREFIX_LANGUAGE_MODEL)
    }

    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.take_hf(&map_lm_key(key))
    }

    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let hf = map_lm_key(key);
        let (data, shape) = self.take_hf(&hf)?;
        if shape.len() != 2 {
            anyhow::bail!("transpose requires rank-2 weight: {key}");
        }
        let rows = shape[0];
        let cols = shape[1];
        let mut out = vec![0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = data[r * cols + c];
            }
        }
        Ok((out, vec![cols, rows]))
    }

    fn remaining_keys(&self) -> Vec<String> {
        self.store
            .keys()
            .iter()
            .filter(|k| k.starts_with(crate::load::PREFIX_LANGUAGE_MODEL))
            .cloned()
            .collect()
    }
}

/// LM weights from a shared snapshot — one tensor cloned per `take`, not the full map.
pub struct ArcLmWeightLoader {
    snapshot: Arc<WeightSnapshot>,
}

impl ArcLmWeightLoader {
    pub fn new(snapshot: Arc<WeightSnapshot>) -> Self {
        Self { snapshot }
    }
}

impl WeightLoader for ArcLmWeightLoader {
    fn len(&self) -> usize {
        self.snapshot.len()
    }

    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let hf = map_lm_key(key);
        let (data, shape) = self
            .snapshot
            .get(&hf)
            .with_context(|| format!("missing weight {hf}"))?;
        Ok((data.clone(), shape.clone()))
    }

    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let hf = map_lm_key(key);
        let (data, shape) = self
            .snapshot
            .get(&hf)
            .with_context(|| format!("missing weight {hf}"))?;
        if shape.len() != 2 {
            anyhow::bail!("transpose requires rank-2 weight: {key}");
        }
        let rows = shape[0];
        let cols = shape[1];
        let mut out = vec![0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = data[r * cols + c];
            }
        }
        Ok((out, vec![cols, rows]))
    }

    fn remaining_keys(&self) -> Vec<String> {
        self.snapshot.keys().cloned().collect()
    }
}
