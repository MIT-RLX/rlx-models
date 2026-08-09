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

//! Native Wav2Vec2-BERT flow — Conformer encoder via [`ModelFlow`] + shared `super::builder::W2vBuilder`.

use anyhow::Result;
use rlx_flow::{BuiltModel, CompileProfile, ModelFlow};
use rlx_ir::{DType, Shape};

use super::builder::W2vBuilder;
use super::config::Wav2Vec2BertConfig;
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;

#[derive(Debug, Clone)]
pub struct Wav2Vec2BertFlow<'a> {
    cfg: &'a Wav2Vec2BertConfig,
    batch: usize,
    seq: usize,
}

impl<'a> Wav2Vec2BertFlow<'a> {
    pub fn new(cfg: &'a Wav2Vec2BertConfig, batch: usize, seq: usize) -> Self {
        Self { cfg, batch, seq }
    }

    pub fn encoder(cfg: &'a Wav2Vec2BertConfig, batch: usize, seq: usize) -> Self {
        Self::new(cfg, batch, seq)
    }

    pub fn build(self, weights: &mut WeightMap) -> Result<BuiltModel> {
        build_wav2vec2_bert_built(self.cfg, weights, self.batch, self.seq)
    }
}

pub fn build_wav2vec2_bert_built(
    cfg: &Wav2Vec2BertConfig,
    weights: &mut WeightMap,
    batch: usize,
    seq: usize,
) -> Result<BuiltModel> {
    let feat_dim = cfg.feature_projection_input_dim;
    let h = cfg.hidden_size;
    let f = DType::F32;
    let hidden_shape = Shape::new(&[batch, seq, h], f);
    let cfg = cfg.clone();

    ModelFlow::new("wav2vec2_bert")
        .with_profile(CompileProfile::encoder())
        .input("input_features", Shape::new(&[batch, seq, feat_dim], f))
        .input("attention_mask", Shape::new(&[batch, seq], f))
        .plugin_named("w2v.encoder", move |emit, _| {
            let feats = emit.flow_input("input_features")?.hir_id();
            let mask = emit.flow_input("attention_mask")?.hir_id();
            let hir = emit
                .module
                .as_hir_mut()
                .expect("wav2vec2 flow requires HIR stage");
            let mut b = W2vBuilder::from_emit_parts(hir, emit.params, emit.weights, batch, seq);
            let hidden = b.emit_encoder(feats, mask, None, &cfg)?;
            Ok(Some(emit.wrap(hidden, hidden_shape.clone())))
        })
        .output("hidden")
        .build(&mut WeightMapSource(weights))
}

pub fn build_wav2vec2_bert_hir(
    cfg: &Wav2Vec2BertConfig,
    weights: &mut WeightMap,
    batch: usize,
    seq: usize,
) -> Result<(
    rlx_ir::hir::HirModule,
    std::collections::HashMap<String, Vec<f32>>,
)> {
    super::builder::build_wav2vec2_bert_hir_sized(cfg, weights, batch, seq)
}

#[cfg(test)]
mod tests {
    use super::super::builder::build_wav2vec2_bert_hir_sized;
    use super::super::config::Wav2Vec2BertConfig;
    use super::*;

    fn tiny_cfg() -> Wav2Vec2BertConfig {
        Wav2Vec2BertConfig {
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            feature_projection_input_dim: 4,
            layer_norm_eps: 1e-5,
            hidden_act: "swish".into(),
            position_embeddings_type: "relative_key".into(),
            left_max_position_embeddings: 4,
            right_max_position_embeddings: 2,
            conv_depthwise_kernel_size: 3,
            add_adapter: false,
            apply_spec_augment: false,
            use_intermediate_ffn_before_adapter: false,
            model_type: None,
        }
    }

    fn synth_weights(cfg: &Wav2Vec2BertConfig, seq: usize) -> WeightMap {
        use std::collections::HashMap;

        let h = cfg.hidden_size;
        let feat = cfg.feature_projection_input_dim;
        let ff = cfg.intermediate_size;
        let dh = cfg.head_dim();
        let k = cfg.conv_depthwise_kernel_size;
        let num_pos = cfg.num_relative_positions();
        let z = |n: usize| vec![0.01f32; n];

        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        t.insert(
            "feature_projection.layer_norm.weight".into(),
            (z(feat), vec![feat]),
        );
        t.insert(
            "feature_projection.layer_norm.bias".into(),
            (z(feat), vec![feat]),
        );
        t.insert(
            "feature_projection.projection.weight".into(),
            (z(h * feat), vec![h, feat]),
        );
        t.insert("feature_projection.projection.bias".into(), (z(h), vec![h]));

        let lp = "encoder.layers.0";
        t.insert(format!("{lp}.ffn1_layer_norm.weight"), (z(h), vec![h]));
        t.insert(format!("{lp}.ffn1_layer_norm.bias"), (z(h), vec![h]));
        t.insert(
            format!("{lp}.ffn1.intermediate_dense.weight"),
            (z(ff * h), vec![ff, h]),
        );
        t.insert(
            format!("{lp}.ffn1.intermediate_dense.bias"),
            (z(ff), vec![ff]),
        );
        t.insert(
            format!("{lp}.ffn1.output_dense.weight"),
            (z(h * ff), vec![h, ff]),
        );
        t.insert(format!("{lp}.ffn1.output_dense.bias"), (z(h), vec![h]));
        t.insert(format!("{lp}.self_attn_layer_norm.weight"), (z(h), vec![h]));
        t.insert(format!("{lp}.self_attn_layer_norm.bias"), (z(h), vec![h]));
        for name in ["linear_q", "linear_k", "linear_v", "linear_out"] {
            t.insert(
                format!("{lp}.self_attn.{name}.weight"),
                (z(h * h), vec![h, h]),
            );
            t.insert(format!("{lp}.self_attn.{name}.bias"), (z(h), vec![h]));
        }
        t.insert(
            format!("{lp}.self_attn.distance_embedding.weight"),
            (z(num_pos * dh), vec![num_pos, dh]),
        );
        t.insert(
            format!("{lp}.conv_module.layer_norm.weight"),
            (z(h), vec![h]),
        );
        t.insert(format!("{lp}.conv_module.layer_norm.bias"), (z(h), vec![h]));
        t.insert(
            format!("{lp}.conv_module.pointwise_conv1.weight"),
            (z(2 * h * h), vec![2 * h, h, 1]),
        );
        t.insert(
            format!("{lp}.conv_module.depthwise_conv.weight"),
            (z(h * k), vec![h, 1, k]),
        );
        t.insert(
            format!("{lp}.conv_module.depthwise_layer_norm.weight"),
            (z(h), vec![h]),
        );
        t.insert(
            format!("{lp}.conv_module.depthwise_layer_norm.bias"),
            (z(h), vec![h]),
        );
        t.insert(
            format!("{lp}.conv_module.pointwise_conv2.weight"),
            (z(h * h), vec![h, h, 1]),
        );
        t.insert(format!("{lp}.ffn2_layer_norm.weight"), (z(h), vec![h]));
        t.insert(format!("{lp}.ffn2_layer_norm.bias"), (z(h), vec![h]));
        t.insert(
            format!("{lp}.ffn2.intermediate_dense.weight"),
            (z(ff * h), vec![ff, h]),
        );
        t.insert(
            format!("{lp}.ffn2.intermediate_dense.bias"),
            (z(ff), vec![ff]),
        );
        t.insert(
            format!("{lp}.ffn2.output_dense.weight"),
            (z(h * ff), vec![h, ff]),
        );
        t.insert(format!("{lp}.ffn2.output_dense.bias"), (z(h), vec![h]));
        t.insert(format!("{lp}.final_layer_norm.weight"), (z(h), vec![h]));
        t.insert(format!("{lp}.final_layer_norm.bias"), (z(h), vec![h]));

        let _ = seq;
        WeightMap::from_tensors(t)
    }

    #[test]
    fn encoder_flow_matches_hir_node_count() {
        let cfg = tiny_cfg();
        let batch = 1;
        let seq = 4;
        let mut weights = synth_weights(&cfg, seq);

        let ref_hir = build_wav2vec2_bert_hir_sized(&cfg, &mut weights, batch, seq)
            .unwrap()
            .0;
        let mut weights2 = synth_weights(&cfg, seq);
        let built = Wav2Vec2BertFlow::encoder(&cfg, batch, seq)
            .build(&mut weights2)
            .unwrap();
        let flow_hir = built.into_hir().unwrap();

        assert_eq!(
            flow_hir.len(),
            ref_hir.len(),
            "wav2vec2 flow should match hir_builder node count"
        );
    }
}
