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

//! Synthetic configs for unit tests.

use crate::config::Qwen25VlLmConfig;
use rlx_qwen3::Qwen3Config;
use std::collections::HashMap;

pub fn tiny_lm_cfg() -> Qwen25VlLmConfig {
    Qwen25VlLmConfig {
        lm: Qwen3Config {
            vocab_size: 128,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            attention_bias: true,
            qk_norm: false,
            sliding_window: None,
            max_window_layers: 0,
            use_sliding_window: false,
            num_experts: 0,
            num_experts_used: 0,
            expert_ffn_size: 0,
            shared_expert_ffn_size: 0,
            expert_weights_scale: 1.0,
        },
        mrope_sections: [4, 4, 4, 0],
        rope_dim_count: 8,
    }
}

pub fn tiny_mmproj_cfg() -> crate::vision::MmProjConfig {
    crate::vision::MmProjConfig {
        patch_size: 2,
        n_embd: 16,
        n_head: 2,
        n_layer: 1,
        image_size: 4,
        image_min_pixels: 16,
        image_max_pixels: 256,
        n_merge: 2,
        eps: 1e-6,
        projector_type: "qwen2.5vl_merger".into(),
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
        spatial_merge_size: 2,
        llm_hidden_size: 32,
        n_ff: 32,
        n_wa_pattern: 0,
        use_silu: true,
        use_rms_norm: true,
    }
}

fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| 0.001 + scale * (i as f32) * 0.01).collect()
}

/// Synthetic Qwen2.5 LM weights (`model.*` keys) for quick-check tests.
pub fn synth_lm_weight_map(cfg: &Qwen25VlLmConfig) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let lm = &cfg.lm;
    let h = lm.hidden_size;
    let q_dim = lm.q_proj_dim();
    let kv_dim = lm.kv_proj_dim();
    let int_dim = lm.intermediate_size;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    t.insert(
        "model.embed_tokens.weight".into(),
        (ramp(lm.vocab_size * h, 0.001), vec![lm.vocab_size, h]),
    );
    t.insert("model.norm.weight".into(), (vec![1.0; h], vec![h]));
    t.insert(
        "lm_head.weight".into(),
        (ramp(lm.vocab_size * h, 0.005), vec![lm.vocab_size, h]),
    );
    for layer in 0..lm.num_hidden_layers {
        let lp = format!("model.layers.{layer}");
        t.insert(
            format!("{lp}.input_layernorm.weight"),
            (vec![1.0; h], vec![h]),
        );
        t.insert(
            format!("{lp}.post_attention_layernorm.weight"),
            (vec![1.0; h], vec![h]),
        );
        t.insert(
            format!("{lp}.self_attn.q_proj.weight"),
            (ramp(q_dim * h, 0.01), vec![q_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.k_proj.weight"),
            (ramp(kv_dim * h, 0.01), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.v_proj.weight"),
            (ramp(kv_dim * h, 0.01), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.o_proj.weight"),
            (ramp(h * q_dim, 0.01), vec![h, q_dim]),
        );
        if lm.attention_bias {
            t.insert(
                format!("{lp}.self_attn.q_proj.bias"),
                (ramp(q_dim, 0.02), vec![q_dim]),
            );
            t.insert(
                format!("{lp}.self_attn.k_proj.bias"),
                (ramp(kv_dim, 0.02), vec![kv_dim]),
            );
            t.insert(
                format!("{lp}.self_attn.v_proj.bias"),
                (ramp(kv_dim, 0.02), vec![kv_dim]),
            );
            t.insert(
                format!("{lp}.self_attn.o_proj.bias"),
                (ramp(h, 0.02), vec![h]),
            );
        }
        t.insert(
            format!("{lp}.mlp.gate_proj.weight"),
            (ramp(int_dim * h, 0.01), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.up_proj.weight"),
            (ramp(int_dim * h, 0.01), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.down_proj.weight"),
            (ramp(h * int_dim, 0.01), vec![h, int_dim]),
        );
    }
    t
}
