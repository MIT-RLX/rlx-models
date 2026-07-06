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

//! Tier-0 BERT encoder flow — native [`ModelFlow`] assembly.

use anyhow::Result;
use rlx_flow::{BertQkvStyle, BuiltModel, CompileProfile, ModelFlow};
use rlx_ir::{DType, Shape};
use std::borrow::Cow;

use rlx_core::config::BertConfig;
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;

/// Fully-qualified weight-key naming for a BERT checkpoint.
///
/// Owns the detected `bert.` / `""` prefix and the Q/K/V layout, and hands out
/// the embedding keys the flow needs. Centralizing the HF names here keeps a
/// rename to a single edit (and makes detection unit-testable); returning a
/// [`Cow`] avoids allocating in the common no-prefix (bare-encoder) case.
#[derive(Debug, Clone)]
struct BertKeys {
    prefix: &'static str,
    qkv_style: BertQkvStyle,
}

impl BertKeys {
    // Weight-name suffixes (relative to the optional `bert.` prefix), each
    // defined once and shared by `detect` + the accessors below.
    const WORD_EMBEDDINGS: &'static str = "embeddings.word_embeddings.weight";
    const POSITION_EMBEDDINGS: &'static str = "embeddings.position_embeddings.weight";
    const TOKEN_TYPE_EMBEDDINGS: &'static str = "embeddings.token_type_embeddings.weight";
    const LN_WEIGHT: &'static str = "embeddings.LayerNorm.weight";
    const LN_BIAS: &'static str = "embeddings.LayerNorm.bias";
    /// Presence of this key (under the detected prefix) selects the BERT
    /// (vs MPNet) Q/K/V layout.
    const QKV_PROBE: &'static str = "encoder.layer.0.attention.self.query.weight";

    /// Detect the `bert.` prefix and Q/K/V weight layout from the checkpoint.
    fn detect(weights: &WeightMap) -> Self {
        let prefix = if weights.has(&format!("bert.{}", Self::WORD_EMBEDDINGS)) {
            "bert."
        } else {
            ""
        };
        let qkv_style = if weights.has(&format!("{prefix}{}", Self::QKV_PROBE)) {
            BertQkvStyle::Bert
        } else {
            BertQkvStyle::Mpnet
        };
        Self { prefix, qkv_style }
    }

    /// `prefix` + `suffix`, borrowing the static literal when there's no prefix.
    fn k(&self, suffix: &'static str) -> Cow<'static, str> {
        if self.prefix.is_empty() {
            Cow::Borrowed(suffix)
        } else {
            Cow::Owned(format!("{}{suffix}", self.prefix))
        }
    }

    fn word_embeddings(&self) -> Cow<'static, str> {
        self.k(Self::WORD_EMBEDDINGS)
    }
    fn position_embeddings(&self) -> Cow<'static, str> {
        self.k(Self::POSITION_EMBEDDINGS)
    }
    fn token_type_embeddings(&self) -> Cow<'static, str> {
        self.k(Self::TOKEN_TYPE_EMBEDDINGS)
    }
    fn embeddings_ln_weight(&self) -> Cow<'static, str> {
        self.k(Self::LN_WEIGHT)
    }
    fn embeddings_ln_bias(&self) -> Cow<'static, str> {
        self.k(Self::LN_BIAS)
    }

    /// Prefix passed to `repeat_bert_layers` (no trailing dot): `"bert"` or `""`.
    fn layer_prefix(&self) -> &'static str {
        self.prefix.trim_end_matches('.')
    }
}

#[derive(Debug, Clone)]
pub struct BertFlow<'a> {
    cfg: &'a BertConfig,
    batch: usize,
    seq: usize,
    profile: CompileProfile,
}

impl<'a> BertFlow<'a> {
    pub fn new(cfg: &'a BertConfig, batch: usize, seq: usize) -> Self {
        Self {
            cfg,
            batch,
            seq,
            profile: CompileProfile::encoder(),
        }
    }

    pub fn with_profile(mut self, profile: CompileProfile) -> Self {
        self.profile = profile;
        self
    }

    pub fn build(self, weights: &mut WeightMap) -> Result<BuiltModel> {
        let keys = BertKeys::detect(weights);

        let f = DType::F32;
        let eps = self.cfg.layer_norm_eps as f32;

        let flow = ModelFlow::new("bert")
            .with_profile(self.profile)
            .input("input_ids", Shape::new(&[self.batch, self.seq], f))
            .input("attention_mask", Shape::new(&[self.batch, self.seq], f))
            .input("token_type_ids", Shape::new(&[self.batch, self.seq], f))
            .input("position_ids", Shape::new(&[self.batch, self.seq], f))
            .embed(keys.word_embeddings())
            .gather_add("position_ids", keys.position_embeddings())
            .gather_add("token_type_ids", keys.token_type_embeddings())
            .layer_norm(keys.embeddings_ln_weight(), keys.embeddings_ln_bias(), eps)
            .repeat_bert_layers(
                self.cfg.num_hidden_layers,
                keys.layer_prefix(),
                keys.qkv_style,
                self.cfg.hidden_size,
                self.cfg.num_attention_heads,
                eps,
            )
            .output("hidden_states");

        flow.build(&mut WeightMapSource(weights))
    }
}

pub fn build_bert_built(
    cfg: &BertConfig,
    weights: &mut WeightMap,
    batch: usize,
    seq: usize,
) -> Result<BuiltModel> {
    BertFlow::new(cfg, batch, seq).build(weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{tiny_bert_weights, weights_with_keys};
    use rlx_core::config::BertConfig;

    #[test]
    fn bert_keys_detects_prefix_and_qkv_style() {
        // Bare encoder, BERT-style Q/K/V.
        let k = BertKeys::detect(&weights_with_keys(&[
            "embeddings.word_embeddings.weight",
            "encoder.layer.0.attention.self.query.weight",
        ]));
        assert_eq!(k.layer_prefix(), "");
        assert_eq!(
            k.word_embeddings().as_ref(),
            "embeddings.word_embeddings.weight"
        );
        assert!(matches!(k.qkv_style, BertQkvStyle::Bert));

        // `bert.`-prefixed, no BERT-style query key → Mpnet layout. The prefix
        // is applied to every key — checked structurally, not by re-pinning a
        // second full key literal.
        let k2 = BertKeys::detect(&weights_with_keys(&[
            "bert.embeddings.word_embeddings.weight",
        ]));
        assert_eq!(k2.layer_prefix(), "bert");
        assert!(k2.embeddings_ln_bias().starts_with("bert."));
        assert!(matches!(k2.qkv_style, BertQkvStyle::Mpnet));
    }

    #[test]
    fn bert_flow_builds() {
        let cfg = BertConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            intermediate_size: 32,
            max_position_embeddings: 32,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            hidden_act: "gelu".into(),
        };
        let mut wm = tiny_bert_weights(&cfg);
        let built = BertFlow::new(&cfg, 1, 4).build(&mut wm).unwrap();
        assert!(built.into_hir().unwrap().len() > 10);
    }
}
