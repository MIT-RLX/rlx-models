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

use rlx_core::config::BertConfig;
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;

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
        let prefix = if weights.has("bert.embeddings.word_embeddings.weight") {
            "bert."
        } else {
            ""
        };
        let qkv_style = if weights.has(&format!(
            "{prefix}encoder.layer.0.attention.self.query.weight"
        )) {
            BertQkvStyle::Bert
        } else {
            BertQkvStyle::Mpnet
        };

        let h = self.cfg.hidden_size;
        let f = DType::F32;
        let eps = self.cfg.layer_norm_eps as f32;

        let flow = ModelFlow::new("bert")
            .with_profile(self.profile)
            .input("input_ids", Shape::new(&[self.batch, self.seq], DType::F32))
            .input("attention_mask", Shape::new(&[self.batch, self.seq], f))
            .input(
                "token_type_ids",
                Shape::new(&[self.batch, self.seq], DType::F32),
            )
            .input(
                "position_ids",
                Shape::new(&[self.batch, self.seq], DType::F32),
            )
            .embed(format!("{prefix}embeddings.word_embeddings.weight"))
            .gather_add(
                "position_ids",
                format!("{prefix}embeddings.position_embeddings.weight"),
            )
            .gather_add(
                "token_type_ids",
                format!("{prefix}embeddings.token_type_embeddings.weight"),
            )
            .layer_norm(
                format!("{prefix}embeddings.LayerNorm.weight"),
                format!("{prefix}embeddings.LayerNorm.bias"),
                eps,
            )
            .repeat_bert_layers(
                self.cfg.num_hidden_layers,
                prefix.trim_end_matches('.'),
                qkv_style,
                h,
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
    use rlx_core::config::BertConfig;
    use std::collections::HashMap;

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
        let h = cfg.hidden_size;
        let int_dim = cfg.intermediate_size;
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let z = |n: usize| vec![0.0f32; n];
        t.insert(
            "embeddings.word_embeddings.weight".into(),
            (z(cfg.vocab_size * h), vec![cfg.vocab_size, h]),
        );
        t.insert(
            "embeddings.position_embeddings.weight".into(),
            (
                z(cfg.max_position_embeddings * h),
                vec![cfg.max_position_embeddings, h],
            ),
        );
        t.insert(
            "embeddings.token_type_embeddings.weight".into(),
            (z(cfg.type_vocab_size * h), vec![cfg.type_vocab_size, h]),
        );
        t.insert("embeddings.LayerNorm.weight".into(), (z(h), vec![h]));
        t.insert("embeddings.LayerNorm.bias".into(), (z(h), vec![h]));
        let lp = "encoder.layer.0";
        t.insert(
            format!("{lp}.attention.self.query.weight"),
            (z(h * h), vec![h, h]),
        );
        t.insert(format!("{lp}.attention.self.query.bias"), (z(h), vec![h]));
        t.insert(
            format!("{lp}.attention.self.key.weight"),
            (z(h * h), vec![h, h]),
        );
        t.insert(format!("{lp}.attention.self.key.bias"), (z(h), vec![h]));
        t.insert(
            format!("{lp}.attention.self.value.weight"),
            (z(h * h), vec![h, h]),
        );
        t.insert(format!("{lp}.attention.self.value.bias"), (z(h), vec![h]));
        t.insert(
            format!("{lp}.attention.output.dense.weight"),
            (z(h * h), vec![h, h]),
        );
        t.insert(format!("{lp}.attention.output.dense.bias"), (z(h), vec![h]));
        t.insert(
            format!("{lp}.attention.output.LayerNorm.weight"),
            (z(h), vec![h]),
        );
        t.insert(
            format!("{lp}.attention.output.LayerNorm.bias"),
            (z(h), vec![h]),
        );
        t.insert(
            format!("{lp}.intermediate.dense.weight"),
            (z(int_dim * h), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.intermediate.dense.bias"),
            (z(int_dim), vec![int_dim]),
        );
        t.insert(
            format!("{lp}.output.dense.weight"),
            (z(int_dim * h), vec![h, int_dim]),
        );
        t.insert(format!("{lp}.output.dense.bias"), (z(h), vec![h]));
        t.insert(format!("{lp}.output.LayerNorm.weight"), (z(h), vec![h]));
        t.insert(format!("{lp}.output.LayerNorm.bias"), (z(h), vec![h]));
        let mut wm = WeightMap::from_tensors(t);
        let built = BertFlow::new(&cfg, 1, 4).build(&mut wm).unwrap();
        assert!(built.into_hir().unwrap().len() > 10);
    }
}
