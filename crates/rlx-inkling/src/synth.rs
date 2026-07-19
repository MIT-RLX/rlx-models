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

//! Tiny synthetic Inkling text configs + deterministic weight tensors.

use crate::config::{
    AttnLayerType, InklingAudioConfig, InklingConfig, InklingTextConfig, InklingVisionConfig,
    MODEL_TYPE, MlpLayerType,
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

/// Tiny config: 3 layers (dense, MoE, MoE), sliding / sliding / global.
pub fn tiny_cfg() -> InklingTextConfig {
    let layers = 3;
    let dense_mlp_idx = 1;
    let local = vec![0, 1];
    let layer_types = (0..layers)
        .map(|i| {
            if local.contains(&i) {
                AttnLayerType::Sliding
            } else {
                AttnLayerType::Global
            }
        })
        .collect();
    let mlp_layer_types = (0..layers)
        .map(|i| {
            if i < dense_mlp_idx {
                MlpLayerType::Dense
            } else {
                MlpLayerType::Sparse
            }
        })
        .collect();
    InklingTextConfig {
        vocab_size: 32,
        unpadded_vocab_size: Some(32),
        hidden_size: 16,
        num_hidden_layers: layers,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 4,
        swa_num_attention_heads: 4,
        swa_num_key_value_heads: 2,
        swa_head_dim: 4,
        sliding_window_size: 4,
        d_rel: 4,
        rel_extent: 8,
        log_scaling_n_floor: None,
        log_scaling_alpha: 0.1,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        conv_kernel_size: 3,
        use_embed_norm: true,
        dense_intermediate_size: 32,
        moe_intermediate_size: 16,
        n_routed_experts: 4,
        num_experts_per_tok: 2,
        n_shared_experts: 1,
        shared_expert_sink: true,
        route_scale: 1.0,
        logits_mup_width_multiplier: 1.0,
        dense_mlp_idx,
        local_layer_ids: local,
        layer_types,
        mlp_layer_types,
        num_mtp_layers: 0,
        mtp_local_layer_ids: vec![],
        eos_token_id: 2,
    }
}

pub fn tiny_mm_cfg() -> InklingConfig {
    let text = tiny_cfg();
    let h = text.hidden_size;
    InklingConfig {
        model_type: MODEL_TYPE.into(),
        text,
        audio: InklingAudioConfig {
            n_mel_bins: 4,
            mel_vocab_size: 4,
            text_hidden_size: h,
            dmel_min_value: -7.0,
            dmel_max_value: 2.0,
            audio_mode: "dmel".into(),
        },
        vision: InklingVisionConfig {
            patch_size: 8,
            temporal_patch_size: 2,
            n_channels: 3,
            n_layers: 2,
            text_hidden_size: h,
            vision_encoder_type: "hmlp".into(),
        },
        image_token_id: 30,
        audio_token_id: 31,
        image_bos_token_id: 28,
        audio_bos_token_id: 29,
    }
}

/// Canonical (split) text weights for [`crate::eager`].
pub fn synthetic_text_weights(cfg: &InklingTextConfig) -> TextWeights {
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;
    let k = cfg.conv_kernel_size;
    let mut tensors = HashMap::new();

    tensors.insert("embed".into(), ramp(v * h, 1));
    tensors.insert("embed_norm".into(), ones(h));
    tensors.insert("norm".into(), ones(h));
    tensors.insert("unembed".into(), ramp(v * h, 2));

    for layer in 0..cfg.num_hidden_layers {
        let (n_h, n_kv, hd) = cfg.attn_heads(layer);
        let rel_ext = cfg.rel_extent_for_layer(layer);
        let q_dim = n_h * hd;
        let kv_dim = n_kv * hd;
        let r_dim = n_h * cfg.d_rel;
        let seed = 100 + layer as u32 * 17;

        tensors.insert(format!("layers.{layer}.attn_norm"), ones(h));
        tensors.insert(format!("layers.{layer}.mlp_norm"), ones(h));
        tensors.insert(format!("layers.{layer}.attn_sconv"), ramp(h * k, seed));
        tensors.insert(format!("layers.{layer}.mlp_sconv"), ramp(h * k, seed + 1));
        tensors.insert(format!("layers.{layer}.q_norm"), ones(hd));
        tensors.insert(format!("layers.{layer}.k_norm"), ones(hd));
        tensors.insert(
            format!("layers.{layer}.k_sconv"),
            ramp(kv_dim * k, seed + 2),
        );
        tensors.insert(
            format!("layers.{layer}.v_sconv"),
            ramp(kv_dim * k, seed + 3),
        );
        tensors.insert(
            format!("layers.{layer}.rel_proj"),
            ramp(cfg.d_rel * rel_ext, seed + 4),
        );
        tensors.insert(format!("layers.{layer}.wq"), ramp(q_dim * h, seed + 5));
        tensors.insert(format!("layers.{layer}.wk"), ramp(kv_dim * h, seed + 6));
        tensors.insert(format!("layers.{layer}.wv"), ramp(kv_dim * h, seed + 7));
        tensors.insert(format!("layers.{layer}.wr"), ramp(r_dim * h, seed + 8));
        tensors.insert(format!("layers.{layer}.wo"), ramp(h * q_dim, seed + 9));

        if cfg.is_dense_mlp(layer) {
            let inter = cfg.dense_intermediate_size;
            tensors.insert(format!("layers.{layer}.gate"), ramp(inter * h, seed + 10));
            tensors.insert(format!("layers.{layer}.up"), ramp(inter * h, seed + 11));
            tensors.insert(format!("layers.{layer}.down"), ramp(h * inter, seed + 12));
            tensors.insert(format!("layers.{layer}.mlp_global_scale"), vec![1.0]);
        } else {
            let inter = cfg.moe_intermediate_size;
            let ne = cfg.n_routed_experts;
            let ns = cfg.n_shared_experts;
            tensors.insert(
                format!("layers.{layer}.expert_w13"),
                ramp(ne * 2 * inter * h, seed + 10),
            );
            tensors.insert(
                format!("layers.{layer}.expert_w2"),
                ramp(ne * h * inter, seed + 11),
            );
            tensors.insert(
                format!("layers.{layer}.gate_weight"),
                ramp((ne + ns) * h, seed + 12),
            );
            tensors.insert(format!("layers.{layer}.gate_bias"), ramp(ne, seed + 13));
            tensors.insert(format!("layers.{layer}.gate_global_scale"), vec![1.0]);
            tensors.insert(
                format!("layers.{layer}.shared_gate"),
                ramp(ns * inter * h, seed + 14),
            );
            tensors.insert(
                format!("layers.{layer}.shared_up"),
                ramp(ns * inter * h, seed + 15),
            );
            tensors.insert(
                format!("layers.{layer}.shared_down"),
                ramp(ns * h * inter, seed + 16),
            );
        }
    }

    TextWeights { tensors }
}
