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

pub mod builder;
pub mod cli;
pub mod config;
pub mod flow;
pub mod preprocess;
pub mod runner;

pub use builder::{
    W2vLayerStop, build_wav2vec2_bert_graph_probe, build_wav2vec2_bert_graph_sized,
    build_wav2vec2_bert_hir_sized,
};
pub use config::Wav2Vec2BertConfig;
pub use flow::{Wav2Vec2BertFlow, build_wav2vec2_bert_built, build_wav2vec2_bert_hir};
pub use preprocess::{
    LogMelExtractor, LogMelFeatures, Wav2Vec2BertPreprocessConfig, load_wav_mono_f32,
    parse_wav_mono_f32,
};
pub use runner::{Wav2Vec2BertRunner, Wav2Vec2BertRunnerBuilder};

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_core::weight_map::WeightMap;
    use rlx_runtime::Device;
    use std::collections::HashMap;

    fn synthetic_weights(cfg: &Wav2Vec2BertConfig, seq: usize) -> WeightMap {
        let h = cfg.hidden_size;
        let feat = cfg.feature_projection_input_dim;
        let int_dim = cfg.intermediate_size;
        let _nh = cfg.num_attention_heads;
        let dh = cfg.head_dim();
        let k = cfg.conv_depthwise_kernel_size;
        let num_pos = cfg.num_relative_positions();

        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let z = |n: usize| vec![0.01f32; n];

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

        for i in 0..cfg.num_hidden_layers {
            let lp = format!("encoder.layers.{i}");
            t.insert(format!("{lp}.ffn1_layer_norm.weight"), (z(h), vec![h]));
            t.insert(format!("{lp}.ffn1_layer_norm.bias"), (z(h), vec![h]));
            t.insert(
                format!("{lp}.ffn1.intermediate_dense.weight"),
                (z(int_dim * h), vec![int_dim, h]),
            );
            t.insert(
                format!("{lp}.ffn1.intermediate_dense.bias"),
                (z(int_dim), vec![int_dim]),
            );
            t.insert(
                format!("{lp}.ffn1.output_dense.weight"),
                (z(h * int_dim), vec![h, int_dim]),
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
                (z(int_dim * h), vec![int_dim, h]),
            );
            t.insert(
                format!("{lp}.ffn2.intermediate_dense.bias"),
                (z(int_dim), vec![int_dim]),
            );
            t.insert(
                format!("{lp}.ffn2.output_dense.weight"),
                (z(h * int_dim), vec![h, int_dim]),
            );
            t.insert(format!("{lp}.ffn2.output_dense.bias"), (z(h), vec![h]));

            t.insert(format!("{lp}.final_layer_norm.weight"), (z(h), vec![h]));
            t.insert(format!("{lp}.final_layer_norm.bias"), (z(h), vec![h]));
        }

        let _ = seq;
        WeightMap::from_tensors(t)
    }

    #[test]
    fn w2v_bert_graph_builds_and_runs() {
        let cfg = Wav2Vec2BertConfig::w2v_bert_2_0();
        let batch = 1;
        let seq = 8;
        let mut wm = synthetic_weights(&cfg, seq);
        let (g, params) = build_wav2vec2_bert_graph_sized(&cfg, &mut wm, batch, seq).unwrap();
        assert_eq!(g.outputs.len(), 1);

        let mut wm_flow = synthetic_weights(&cfg, seq);
        let built = build_wav2vec2_bert_built(&cfg, &mut wm_flow, batch, seq).unwrap();
        assert_eq!(built.primary_shape().rank(), 3);

        let mut compiled = rlx_core::flow_util::compile_built(built, Device::Cpu).unwrap();
        for (name, data) in &params {
            compiled.set_param(name, data);
        }

        let feat_dim = cfg.feature_projection_input_dim;
        let h = cfg.hidden_size;
        let features = vec![0.05f32; batch * seq * feat_dim];
        let mask = vec![1.0f32; batch * seq];
        let out = compiled
            .run(&[("input_features", &features), ("attention_mask", &mask)])
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(out.len(), batch * seq * h);
    }

    #[test]
    fn parse_hf_config_json() {
        let json = r#"{
            "model_type": "wav2vec2-bert",
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "feature_projection_input_dim": 160,
            "position_embeddings_type": "relative_key",
            "left_max_position_embeddings": 64,
            "right_max_position_embeddings": 8,
            "conv_depthwise_kernel_size": 31,
            "hidden_act": "swish"
        }"#;
        let cfg: Wav2Vec2BertConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.num_relative_positions(), 73);
    }
}
