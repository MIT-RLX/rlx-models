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

//! Shared helpers for Wav2Vec2-BERT cross-backend basic tests.

#![allow(dead_code)]

use rlx_models::WeightMap;
use rlx_models::wav2vec2_bert::{Wav2Vec2BertConfig, build_wav2vec2_bert_graph_sized};
use rlx_runtime::Device;
use std::collections::HashMap;

/// Tiny Conformer config — 1 layer, small dims, fast compile on GPU backends.
pub fn tiny_config() -> Wav2Vec2BertConfig {
    Wav2Vec2BertConfig {
        hidden_size: 32,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        intermediate_size: 64,
        feature_projection_input_dim: 16,
        layer_norm_eps: 1e-5,
        hidden_act: "swish".into(),
        position_embeddings_type: "relative_key".into(),
        left_max_position_embeddings: 4,
        right_max_position_embeddings: 2,
        conv_depthwise_kernel_size: 5,
        add_adapter: false,
        apply_spec_augment: false,
        use_intermediate_ffn_before_adapter: false,
        model_type: Some("wav2vec2-bert".into()),
    }
}

pub fn synthetic_weights(cfg: &Wav2Vec2BertConfig) -> WeightMap {
    let h = cfg.hidden_size;
    let feat = cfg.feature_projection_input_dim;
    let int_dim = cfg.intermediate_size;
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

    WeightMap::from_tensors(t)
}

/// Build, compile, and run the tiny graph on `device`. Returns output hidden states.
pub fn run_tiny_graph(device: Device) -> Vec<f32> {
    let cfg = tiny_config();
    let batch = 1;
    let seq = 8;
    let mut wm = synthetic_weights(&cfg);
    let (graph, params) =
        build_wav2vec2_bert_graph_sized(&cfg, &mut wm, batch, seq).expect("graph build");

    let mut compiled =
        rlx_models::flow_util::compile_graph_encoder_with_params(device, graph, params)
            .expect("compile wav2vec2 graph");

    let feat_dim = cfg.feature_projection_input_dim;
    let h = cfg.hidden_size;
    let features = vec![0.05f32; batch * seq * feat_dim];
    let mask = vec![1.0f32; batch * seq];
    let outs = compiled.run(&[("input_features", &features), ("attention_mask", &mask)]);
    let out = outs.into_iter().next().expect("one output");
    assert_eq!(out.len(), batch * seq * h);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "non-finite values on {device:?}"
    );
    out
}
