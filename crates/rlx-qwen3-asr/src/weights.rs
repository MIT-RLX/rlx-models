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

//! Checkpoint tensor names for `Qwen/Qwen3-ASR-*` safetensors (`thinker.*`).

use anyhow::Result;
use rlx_core::weight_loader::WeightLoader;
use rlx_core::weight_map::WeightMap;

pub const PREFIX_AUDIO_TOWER: &str = "thinker.audio_tower.";
pub const PREFIX_LANGUAGE_MODEL: &str = "thinker.model.";
pub const KEY_LM_HEAD: &str = "thinker.lm_head.weight";
pub const KEY_EMBED_TOKENS: &str = "thinker.model.embed_tokens.weight";

/// HF weight-name helpers for the audio tower.
pub struct AsrWeightPrefix;

impl AsrWeightPrefix {
    pub const CONV2D1_W: &'static str = "thinker.audio_tower.conv2d1.weight";
    pub const CONV2D1_B: &'static str = "thinker.audio_tower.conv2d1.bias";
    pub const CONV2D2_W: &'static str = "thinker.audio_tower.conv2d2.weight";
    pub const CONV2D2_B: &'static str = "thinker.audio_tower.conv2d2.bias";
    pub const CONV2D3_W: &'static str = "thinker.audio_tower.conv2d3.weight";
    pub const CONV2D3_B: &'static str = "thinker.audio_tower.conv2d3.bias";
    pub const CONV_OUT_W: &'static str = "thinker.audio_tower.conv_out.weight";
    pub const LN_POST_W: &'static str = "thinker.audio_tower.ln_post.weight";
    pub const LN_POST_B: &'static str = "thinker.audio_tower.ln_post.bias";
    pub const PROJ1_W: &'static str = "thinker.audio_tower.proj1.weight";
    pub const PROJ1_B: &'static str = "thinker.audio_tower.proj1.bias";
    pub const PROJ2_W: &'static str = "thinker.audio_tower.proj2.weight";
    pub const PROJ2_B: &'static str = "thinker.audio_tower.proj2.bias";

    pub fn audio_layer(i: usize, suffix: &str) -> String {
        format!("thinker.audio_tower.layers.{i}.{suffix}")
    }
}

/// Maps Qwen3 decoder keys (`model.*`, `lm_head.*`) to the `thinker.*` names
/// used in the Qwen3-ASR checkpoint.
pub struct LanguageModelPrefixLoader<'a> {
    inner: &'a mut WeightMap,
}

impl<'a> LanguageModelPrefixLoader<'a> {
    pub fn new(inner: &'a mut WeightMap) -> Self {
        Self { inner }
    }
}

fn map_lm_key(key: &str) -> String {
    format!("thinker.{key}")
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
