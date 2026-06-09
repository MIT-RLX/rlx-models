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

//! Checkpoint tensor names for `mistralai/Voxtral-*` safetensors.

use anyhow::Result;
use rlx_core::weight_loader::WeightLoader;
use rlx_core::weight_map::WeightMap;

/// HF weight prefix helpers (`audio_tower.*`, `language_model.*`, `multi_modal_projector.*`).
#[derive(Debug, Clone)]
pub struct VoxtralWeightPrefix;

impl VoxtralWeightPrefix {
    pub fn enc_layer(i: usize, suffix: &str) -> String {
        format!("audio_tower.layers.{i}.{suffix}")
    }

    pub fn enc_conv1_w() -> &'static str {
        "audio_tower.conv1.weight"
    }

    pub fn enc_conv1_b() -> &'static str {
        "audio_tower.conv1.bias"
    }

    pub fn enc_conv2_w() -> &'static str {
        "audio_tower.conv2.weight"
    }

    pub fn enc_conv2_b() -> &'static str {
        "audio_tower.conv2.bias"
    }

    pub fn enc_embed_positions() -> &'static str {
        "audio_tower.embed_positions.weight"
    }

    pub fn enc_ln_post_w() -> &'static str {
        "audio_tower.layer_norm.weight"
    }

    pub fn enc_ln_post_b() -> &'static str {
        "audio_tower.layer_norm.bias"
    }

    pub fn projector_linear1() -> &'static str {
        "multi_modal_projector.linear_1.weight"
    }

    pub fn projector_linear2() -> &'static str {
        "multi_modal_projector.linear_2.weight"
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
        "model.embed_tokens.weight" => VoxtralWeightPrefix::lm_embed_tokens().to_string(),
        "model.norm.weight" => VoxtralWeightPrefix::lm_norm().to_string(),
        "lm_head.weight" => VoxtralWeightPrefix::lm_head().to_string(),
        k if k.starts_with("model.layers.") => format!("language_model.{k}"),
        other => other.to_string(),
    }
}

/// Maps Llama-shaped keys (`model.*`, `lm_head.*`) to Voxtral safetensor names.
pub struct LanguageModelPrefixLoader<'a> {
    inner: &'a mut WeightMap,
}

impl<'a> LanguageModelPrefixLoader<'a> {
    pub fn new(inner: &'a mut WeightMap) -> Self {
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
