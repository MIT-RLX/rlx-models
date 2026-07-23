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

//! Tiny synthetic Laguna configs + deterministic weight tensors.

use crate::config::{
    AttnGating, AttnLayerType, LagunaConfig, MlpLayerType, MODEL_TYPE, RopeLayerParams,
};
use crate::eager::TextWeights;
use std::collections::HashMap;

fn ramp(n: usize, seed: u32) -> Vec<f32> {
    let s = seed as f32 * 0.001 + 0.01;
    (0..n)
        .map(|i| ((i as f32 + 1.0) * s).sin() * 0.02)
        .collect()
}

fn ones(n: usize) -> Vec<f32> {
    vec![1.0; n]
}

/// Tiny config: 4 layers (dense + 3 MoE), full / SWA×3, per-head gate.
pub fn tiny_cfg() -> LagunaConfig {
    let layers = 4;
    let layer_types = vec![
        AttnLayerType::Full,
        AttnLayerType::Sliding,
        AttnLayerType::Sliding,
        AttnLayerType::Sliding,
    ];
    let mlp_layer_types = vec![
        MlpLayerType::Dense,
        MlpLayerType::Sparse,
        MlpLayerType::Sparse,
        MlpLayerType::Sparse,
    ];
    let num_attention_heads_per_layer = vec![4, 6, 6, 6];
    LagunaConfig {
        model_type: MODEL_TYPE.into(),
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: layers,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 4,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        num_experts: 4,
        num_experts_per_tok: 2,
        moe_intermediate_size: 8,
        shared_expert_intermediate_size: 8,
        norm_topk_prob: true,
        moe_routed_scaling_factor: 2.5,
        moe_router_logit_softcapping: 0.0,
        sliding_window: 4,
        gating: AttnGating::PerHead,
        layer_types,
        mlp_layer_types,
        num_attention_heads_per_layer,
        rope_full: RopeLayerParams {
            rope_type: "yarn".into(),
            rope_theta: 10_000.0,
            partial_rotary_factor: 0.5,
            yarn_factor: 2.0,
            original_max_position_embeddings: 32,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attention_factor: 1.0,
        },
        rope_sliding: RopeLayerParams {
            rope_type: "default".into(),
            rope_theta: 10_000.0,
            partial_rotary_factor: 1.0,
            ..RopeLayerParams::default()
        },
        bos_token_id: 2,
        eos_token_id: 2,
        pad_token_id: 0,
        tie_word_embeddings: false,
    }
}

pub fn synthetic_text_weights(cfg: &LagunaConfig) -> TextWeights {
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;
    let mut tensors = HashMap::new();
    tensors.insert("embed".into(), ramp(v * h, 1));
    tensors.insert("norm".into(), ones(h));
    tensors.insert("unembed".into(), ramp(v * h, 2));

    for layer in 0..cfg.num_hidden_layers {
        let n_h = cfg.n_heads(layer);
        let n_kv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let q_dim = n_h * hd;
        let kv_dim = n_kv * hd;
        let seed = 100 + layer as u32 * 17;
        let gate_out = match cfg.gating {
            AttnGating::Off => 0,
            AttnGating::PerHead => n_h,
            AttnGating::PerElement => q_dim,
        };

        tensors.insert(format!("layers.{layer}.attn_norm"), ones(h));
        tensors.insert(format!("layers.{layer}.ffn_norm"), ones(h));
        tensors.insert(format!("layers.{layer}.q_norm"), ones(hd));
        tensors.insert(format!("layers.{layer}.k_norm"), ones(hd));
        tensors.insert(format!("layers.{layer}.wq"), ramp(q_dim * h, seed));
        tensors.insert(format!("layers.{layer}.wk"), ramp(kv_dim * h, seed + 1));
        tensors.insert(format!("layers.{layer}.wv"), ramp(kv_dim * h, seed + 2));
        tensors.insert(format!("layers.{layer}.wo"), ramp(h * q_dim, seed + 3));
        if gate_out > 0 {
            tensors.insert(format!("layers.{layer}.wg"), ramp(gate_out * h, seed + 4));
        }

        if cfg.is_dense_mlp(layer) {
            let inter = cfg.intermediate_size;
            tensors.insert(format!("layers.{layer}.gate"), ramp(inter * h, seed + 10));
            tensors.insert(format!("layers.{layer}.up"), ramp(inter * h, seed + 11));
            tensors.insert(format!("layers.{layer}.down"), ramp(h * inter, seed + 12));
        } else {
            let inter = cfg.moe_intermediate_size;
            let ne = cfg.num_experts;
            let ns = cfg.shared_expert_intermediate_size;
            tensors.insert(format!("layers.{layer}.gate_weight"), ramp(ne * h, seed + 20));
            tensors.insert(format!("layers.{layer}.gate_bias"), ramp(ne, seed + 21));
            tensors.insert(
                format!("layers.{layer}.expert_gate"),
                ramp(ne * inter * h, seed + 22),
            );
            tensors.insert(
                format!("layers.{layer}.expert_up"),
                ramp(ne * inter * h, seed + 23),
            );
            tensors.insert(
                format!("layers.{layer}.expert_down"),
                ramp(ne * h * inter, seed + 24),
            );
            tensors.insert(format!("layers.{layer}.shared_gate"), ramp(ns * h, seed + 25));
            tensors.insert(format!("layers.{layer}.shared_up"), ramp(ns * h, seed + 26));
            tensors.insert(
                format!("layers.{layer}.shared_down"),
                ramp(h * ns, seed + 27),
            );
        }
    }

    TextWeights { tensors }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_has_expected_keys() {
        let cfg = tiny_cfg();
        let w = synthetic_text_weights(&cfg);
        assert!(w.tensors.contains_key("embed"));
        assert!(w.tensors.contains_key("layers.0.gate"));
        assert!(w.tensors.contains_key("layers.1.expert_gate"));
        assert!(w.tensors.contains_key("layers.1.wg"));
    }
}
