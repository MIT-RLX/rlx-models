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

//! Checkpoint key prefixes — HuggingFace (`model.encoder`) vs OpenAI (`.pt`).

use rlx_core::weight_map::WeightMap;

#[derive(Debug, Clone)]
pub struct WhisperWeightPrefix {
    pub encoder: String,
    pub decoder: String,
    pub hf_embed_names: bool,
}

impl WhisperWeightPrefix {
    pub fn detect(weights: &WeightMap) -> Self {
        Self::detect_with(|k| weights.has(k))
    }

    /// Detect prefixes from a key-existence predicate, avoiding a full [`WeightMap`]
    /// clone when only key names are available (e.g. a raw snapshot `HashMap`).
    pub fn detect_with(has: impl Fn(&str) -> bool) -> Self {
        let (encoder, decoder) = if has("model.encoder.conv1.weight") {
            ("model.encoder".into(), "model.decoder".into())
        } else if has("encoder.conv1.weight") {
            ("encoder".into(), "decoder".into())
        } else {
            ("model.encoder".into(), "model.decoder".into())
        };
        let hf_embed_names = has(&format!("{decoder}.embed_tokens.weight"));
        Self {
            encoder,
            decoder,
            hf_embed_names,
        }
    }

    pub fn enc_layer(&self, i: usize, suffix: &str) -> String {
        format!("{}.layers.{i}.{suffix}", self.encoder)
    }

    pub fn dec_layer(&self, i: usize, suffix: &str) -> String {
        format!("{}.layers.{i}.{suffix}", self.decoder)
    }

    pub fn enc_conv1_w(&self) -> String {
        format!("{}.conv1.weight", self.encoder)
    }

    pub fn enc_conv1_b(&self) -> String {
        format!("{}.conv1.bias", self.encoder)
    }

    pub fn enc_conv2_w(&self) -> String {
        format!("{}.conv2.weight", self.encoder)
    }

    pub fn enc_conv2_b(&self) -> String {
        format!("{}.conv2.bias", self.encoder)
    }

    pub fn enc_ln_post_w(&self) -> String {
        format!("{}.layer_norm.weight", self.encoder)
    }

    pub fn enc_ln_post_b(&self) -> String {
        format!("{}.layer_norm.bias", self.encoder)
    }

    pub fn dec_embed_tokens(&self) -> String {
        if self.hf_embed_names {
            format!("{}.embed_tokens.weight", self.decoder)
        } else {
            format!("{}.token_embedding.weight", self.decoder)
        }
    }

    pub fn dec_embed_positions(&self) -> String {
        if self.hf_embed_names {
            format!("{}.embed_positions.weight", self.decoder)
        } else {
            format!("{}.positional_embedding", self.decoder)
        }
    }

    pub fn dec_ln_w(&self) -> String {
        format!("{}.layer_norm.weight", self.decoder)
    }

    pub fn dec_ln_b(&self) -> String {
        format!("{}.layer_norm.bias", self.decoder)
    }
}
