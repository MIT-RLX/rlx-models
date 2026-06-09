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

//! Synthetic LLaDA2 weights for CPU integration tests.

use crate::config::LLaDA2MoeConfig;
use crate::weights::{DenseFfnWeights, LLaDA2Weights, LayerFfn, LayerWeights, MoeLayerWeights};

pub fn tiny_cfg() -> LLaDA2MoeConfig {
    LLaDA2MoeConfig {
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: Some(32),
        num_hidden_layers: 3,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: Some(4),
        num_experts: 4,
        num_experts_per_tok: 2,
        num_shared_experts: Some(1),
        moe_intermediate_size: Some(8),
        n_group: 2,
        topk_group: 1,
        routed_scaling_factor: 2.5,
        first_k_dense_replace: 1,
        max_position_embeddings: 64,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        partial_rotary_factor: 0.5,
        use_qk_norm: true,
        use_qkv_bias: false,
        use_bias: false,
        hidden_act: "silu".into(),
        attention_dropout: 0.0,
        embedding_dropout: 0.0,
        output_dropout: 0.0,
        tie_word_embeddings: false,
        norm_topk_prob: true,
        moe_router_enable_expert_bias: true,
        pad_token_id: 0,
        mask_token_id: 31,
        eos_token_id: 30,
    }
}

fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| 0.001 + scale * (i as f32) * 0.01).collect()
}

pub fn tiny_weights(cfg: &LLaDA2MoeConfig) -> LLaDA2Weights {
    let h = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let ff_dense = cfg.intermediate_size();
    let ff_moe = cfg.expert_ffn_dim();
    let e = cfg.num_experts;
    let hd = cfg.head_dim();
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_kv_heads();
    let qkv_out = (nh + 2 * nkv) * hd;

    let mut layers = Vec::new();
    for il in 0..cfg.num_hidden_layers {
        let ffn = if cfg.is_moe_layer(il) {
            LayerFfn::Moe(MoeLayerWeights {
                router: ramp(h * e, 0.1 + il as f32),
                expert_bias: ramp(e, 0.01),
                gate_exps: ramp(e * h * ff_moe, 0.2),
                up_exps: ramp(e * h * ff_moe, 0.3),
                down_exps: ramp(e * ff_moe * h, 0.4),
                shared_gate: Some(ramp(h * ff_moe, 0.5)),
                shared_up: Some(ramp(h * ff_moe, 0.6)),
                shared_down: Some(ramp(ff_moe * h, 0.7)),
            })
        } else {
            LayerFfn::Dense(DenseFfnWeights {
                gate: ramp(h * ff_dense, 0.1),
                up: ramp(h * ff_dense, 0.2),
                down: ramp(ff_dense * h, 0.3),
            })
        };
        layers.push(LayerWeights {
            input_norm: ramp(h, 1.0),
            post_attn_norm: ramp(h, 1.0),
            qkv: ramp(h * qkv_out, 0.15),
            q_norm: Some(ramp(hd, 1.0)),
            k_norm: Some(ramp(hd, 1.0)),
            o_proj: ramp(nh * hd * h, 0.16),
            ffn,
        });
    }

    LLaDA2Weights {
        embed: ramp(vocab * h, 0.05),
        final_norm: ramp(h, 1.0),
        lm_head: ramp(h * vocab, 0.08),
        layers,
    }
}
