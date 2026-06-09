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

//! Tier-0 NomicBERT encoder flow — native [`ModelFlow`] assembly.

use anyhow::Result;
use rlx_flow::{BuiltModel, CompileProfile, ModelFlow, RopeTablesStage};
use rlx_ir::{DType, Shape};

use rlx_core::config::NomicBertConfig;
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;

#[derive(Debug, Clone)]
pub struct NomicFlow<'a> {
    cfg: &'a NomicBertConfig,
    batch: usize,
    seq: usize,
    profile: CompileProfile,
}

impl<'a> NomicFlow<'a> {
    pub fn new(cfg: &'a NomicBertConfig, batch: usize, seq: usize) -> Self {
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
        let h = self.cfg.hidden_size;
        let nh = self.cfg.num_attention_heads;
        let dh = self.cfg.head_dim;
        let eps = self.cfg.layer_norm_eps as f32;
        let f = DType::F32;

        let (cos_data, sin_data) = rope_tables(self.cfg);

        let flow = ModelFlow::new("nomic_bert")
            .with_profile(self.profile)
            .input("input_ids", Shape::new(&[self.batch, self.seq], DType::F32))
            .input("attention_mask", Shape::new(&[self.batch, self.seq], f))
            .input(
                "token_type_ids",
                Shape::new(&[self.batch, self.seq], DType::F32),
            )
            .rope_tables(RopeTablesStage::param(
                self.cfg.max_position_embeddings,
                dh / 2,
                cos_data,
                sin_data,
            ))
            .embed("embeddings.word_embeddings.weight")
            .gather_add("token_type_ids", "embeddings.token_type_embeddings.weight")
            .layer_norm("emb_ln.weight", "emb_ln.bias", eps)
            .repeat_nomic_layers(self.cfg.num_hidden_layers, h, nh, dh, eps)
            .output("hidden_states");

        flow.build(&mut WeightMapSource(weights))
    }
}

fn rope_tables(cfg: &NomicBertConfig) -> (Vec<f32>, Vec<f32>) {
    let dh = cfg.head_dim;
    let half = dh / 2;
    let mut cos_data = vec![0f32; cfg.max_position_embeddings * half];
    let mut sin_data = vec![0f32; cfg.max_position_embeddings * half];
    for pos in 0..cfg.max_position_embeddings {
        for i in 0..half {
            let freq = 1.0 / cfg.rotary_emb_base.powf((2 * i) as f64 / dh as f64);
            let angle = pos as f64 * freq;
            let (s, c) = angle.sin_cos();
            cos_data[pos * half + i] = c as f32;
            sin_data[pos * half + i] = s as f32;
        }
    }
    (cos_data, sin_data)
}

pub fn build_nomic_built(
    cfg: &NomicBertConfig,
    weights: &mut WeightMap,
    batch: usize,
    seq: usize,
) -> Result<BuiltModel> {
    NomicFlow::new(cfg, batch, seq).build(weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn nomic_flow_builds() {
        let cfg = NomicBertConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            intermediate_size: 32,
            max_position_embeddings: 32,
            type_vocab_size: 2,
            layer_norm_eps: 1e-5,
            head_dim: 4,
            rotary_emb_base: 1000.0,
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
            "embeddings.token_type_embeddings.weight".into(),
            (z(cfg.type_vocab_size * h), vec![cfg.type_vocab_size, h]),
        );
        t.insert("emb_ln.weight".into(), (z(h), vec![h]));
        t.insert("emb_ln.bias".into(), (z(h), vec![h]));
        let lp = "encoder.layers.0";
        t.insert(
            format!("{lp}.attn.Wqkv.weight"),
            (z(h * 3 * h), vec![3 * h, h]),
        );
        t.insert(format!("{lp}.attn.out_proj.weight"), (z(h * h), vec![h, h]));
        t.insert(format!("{lp}.norm1.weight"), (z(h), vec![h]));
        t.insert(format!("{lp}.norm1.bias"), (z(h), vec![h]));
        t.insert(
            format!("{lp}.mlp.fc11.weight"),
            (z(h * int_dim), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.fc12.weight"),
            (z(h * int_dim), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.fc2.weight"),
            (z(int_dim * h), vec![h, int_dim]),
        );
        t.insert(format!("{lp}.norm2.weight"), (z(h), vec![h]));
        t.insert(format!("{lp}.norm2.bias"), (z(h), vec![h]));
        let mut wm = WeightMap::from_tensors(t);
        let built = NomicFlow::new(&cfg, 1, 4).build(&mut wm).unwrap();
        assert!(built.into_hir().unwrap().len() > 10);
    }
}
