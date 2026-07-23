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

//! Synthetic Qwen3.5 weights for integration tests and criterion benches.

use super::{
    MatWeight, Qwen35Config, Qwen35FullAttnLayer, Qwen35LayerFfn, Qwen35LinearLayer, Qwen35MoeFfn,
    Qwen35MtpLayer, Qwen35TrunkLayer, Qwen35Weights,
};

pub fn mat(data: Vec<f32>) -> MatWeight {
    MatWeight::F32(data)
}

pub fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| 0.001 + scale * (i as f32) * 0.01).collect()
}

/// Tiny config (3 trunk + 1 MTP) — matches `qwen35_forward_check`.
pub fn tiny_cfg() -> Qwen35Config {
    Qwen35Config {
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 4,
        nextn_predict_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        key_length: 4,
        value_length: 4,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        rope_dim_count: 4,
        rope_dim_sections: vec![],
        mrope_interleaved: false,
        rms_norm_offset: false,
        full_attention_interval: 3,
        ssm_conv_kernel: 4,
        ssm_group_count: 2,
        ssm_inner_size: 8,
        ssm_state_size: 4,
        ssm_time_step_rank: 2,
        tie_word_embeddings: true,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        shared_expert_ffn_size: 0,
        expert_weights_scale: 1.0,
    }
}

/// Medium config for shape-stress tests.
pub fn medium_cfg() -> Qwen35Config {
    Qwen35Config {
        vocab_size: 64,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 6,
        nextn_predict_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        key_length: 16,
        value_length: 16,
        max_position_embeddings: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        rope_dim_count: 16,
        rope_dim_sections: vec![],
        mrope_interleaved: false,
        rms_norm_offset: false,
        full_attention_interval: 3,
        ssm_conv_kernel: 4,
        ssm_group_count: 4,
        ssm_inner_size: 128,
        ssm_state_size: 16,
        ssm_time_step_rank: 8,
        tie_word_embeddings: true,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        shared_expert_ffn_size: 0,
        expert_weights_scale: 1.0,
    }
}

/// Larger config for criterion benches.
pub fn bench_cfg() -> Qwen35Config {
    Qwen35Config {
        vocab_size: 128,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 6,
        nextn_predict_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        key_length: 16,
        value_length: 16,
        max_position_embeddings: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        rope_dim_count: 16,
        rope_dim_sections: vec![],
        mrope_interleaved: false,
        rms_norm_offset: false,
        full_attention_interval: 3,
        ssm_conv_kernel: 4,
        ssm_group_count: 4,
        ssm_inner_size: 128,
        ssm_state_size: 16,
        ssm_time_step_rank: 8,
        tie_word_embeddings: true,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        shared_expert_ffn_size: 0,
        expert_weights_scale: 1.0,
    }
}

pub fn linear_layer(cfg: &Qwen35Config) -> Qwen35LinearLayer {
    let n_embd = cfg.hidden_size;
    let n_state = cfg.ssm_state_size;
    let n_k_heads = cfg.ssm_group_count;
    let n_v_heads = cfg.ssm_time_step_rank;
    let key_dim = n_state * n_k_heads;
    let value_dim = n_state * n_v_heads;
    let conv_channels = key_dim * 2 + value_dim;
    let n_ff = cfg.intermediate_size;
    let k_conv = cfg.ssm_conv_kernel;
    Qwen35LinearLayer {
        attn_norm: vec![1.0; n_embd],
        attn_post_norm: vec![1.0; n_embd],
        attn_qkv: mat(ramp(n_embd * conv_channels, 0.01)),
        attn_gate: mat(ramp(n_embd * value_dim, 0.01)),
        ssm_conv1d: ramp(k_conv * conv_channels, 0.02),
        ssm_dt_bias: ramp(n_v_heads, 0.05),
        ssm_a: vec![-1.0; n_v_heads],
        ssm_beta: mat(ramp(n_embd * n_v_heads, 0.01)),
        ssm_alpha: mat(ramp(n_embd * n_v_heads, 0.01)),
        ssm_norm: vec![1.0; n_state],
        ssm_out: mat(ramp(value_dim * n_embd, 0.01)),
        ffn: Qwen35LayerFfn::Dense {
            gate: mat(ramp(n_embd * n_ff, 0.01)),
            down: mat(ramp(n_ff * n_embd, 0.01)),
            up: mat(ramp(n_embd * n_ff, 0.01)),
        },
    }
}

pub fn full_attn_layer(cfg: &Qwen35Config) -> Qwen35FullAttnLayer {
    let n_embd = cfg.hidden_size;
    let n_head = cfg.num_attention_heads;
    let n_kv_head = cfg.num_key_value_heads;
    let head_dim = cfg.key_length;
    let q_gate_cols = n_head * head_dim * 2;
    let kv_cols = n_kv_head * head_dim;
    let n_ff = cfg.intermediate_size;
    Qwen35FullAttnLayer {
        attn_norm: vec![1.0; n_embd],
        attn_post_norm: vec![1.0; n_embd],
        attn_q_gate: mat(ramp(n_embd * q_gate_cols, 0.01)),
        attn_k: mat(ramp(n_embd * kv_cols, 0.01)),
        attn_v: mat(ramp(n_embd * kv_cols, 0.01)),
        attn_output: mat(ramp(n_head * head_dim * n_embd, 0.01)),
        attn_q_norm: vec![1.0; head_dim],
        attn_k_norm: vec![1.0; head_dim],
        ffn: Qwen35LayerFfn::Dense {
            gate: mat(ramp(n_embd * n_ff, 0.01)),
            down: mat(ramp(n_ff * n_embd, 0.01)),
            up: mat(ramp(n_embd * n_ff, 0.01)),
        },
    }
}

/// Dense MoE quick check config (3 trunk layers, no MTP).
pub fn moe_cfg() -> Qwen35Config {
    Qwen35Config {
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 3,
        nextn_predict_layers: 0,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        key_length: 4,
        value_length: 4,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        rope_dim_count: 4,
        rope_dim_sections: vec![],
        mrope_interleaved: false,
        rms_norm_offset: false,
        full_attention_interval: 3,
        ssm_conv_kernel: 4,
        ssm_group_count: 2,
        ssm_inner_size: 8,
        ssm_state_size: 4,
        ssm_time_step_rank: 2,
        tie_word_embeddings: true,
        num_experts: 4,
        num_experts_used: 2,
        expert_ffn_size: 16,
        shared_expert_ffn_size: 16,
        expert_weights_scale: 1.0,
    }
}

pub fn moe_ffn(cfg: &Qwen35Config) -> Qwen35LayerFfn {
    let n_embd = cfg.hidden_size;
    let n_ff = cfg.expert_ffn_dim();
    let n_ff_s = cfg.shared_expert_ffn_dim();
    let e = cfg.num_experts;
    Qwen35LayerFfn::Moe(Qwen35MoeFfn {
        router: mat(ramp(n_embd * e, 0.001)),
        gate_exps: mat(ramp(e * n_embd * n_ff, 0.002)),
        up_exps: mat(ramp(e * n_embd * n_ff, 0.003)),
        down_exps: mat(ramp(e * n_ff * n_embd, 0.004)),
        shared_router: ramp(n_embd, 0.005),
        shared_gate: mat(ramp(n_embd * n_ff_s, 0.006)),
        shared_up: mat(ramp(n_embd * n_ff_s, 0.007)),
        shared_down: mat(ramp(n_ff_s * n_embd, 0.008)),
    })
}

pub fn moe_linear_layer(cfg: &Qwen35Config) -> Qwen35LinearLayer {
    let mut layer = linear_layer(cfg);
    layer.ffn = moe_ffn(cfg);
    layer
}

pub fn moe_full_attn_layer(cfg: &Qwen35Config) -> Qwen35FullAttnLayer {
    let mut layer = full_attn_layer(cfg);
    layer.ffn = moe_ffn(cfg);
    layer
}

pub fn moe_synth_weights(cfg: &Qwen35Config) -> Qwen35Weights {
    let n_embd = cfg.hidden_size;
    let n_vocab = cfg.vocab_size;
    Qwen35Weights {
        token_embd: std::sync::Arc::from(ramp(n_vocab * n_embd, 0.0001)),
        output_norm: vec![1.0; n_embd],
        output: None,
        token_embd_lm: None,
        trunk_layers: vec![
            Qwen35TrunkLayer::Linear(moe_linear_layer(cfg)),
            Qwen35TrunkLayer::Linear(moe_linear_layer(cfg)),
            Qwen35TrunkLayer::FullAttn(moe_full_attn_layer(cfg)),
        ],
        mtp_layers: vec![],
    }
}

pub fn synth_weights(cfg: &Qwen35Config) -> Qwen35Weights {
    let n_embd = cfg.hidden_size;
    let n_vocab = cfg.vocab_size;
    let n_main = cfg.num_hidden_layers - cfg.nextn_predict_layers;
    let interval = cfg.full_attention_interval.max(1);

    let mut trunk = Vec::new();
    for il in 0..n_main {
        let is_full = ((il + 1) % interval) == 0;
        trunk.push(if is_full {
            Qwen35TrunkLayer::FullAttn(full_attn_layer(cfg))
        } else {
            Qwen35TrunkLayer::Linear(linear_layer(cfg))
        });
    }
    let mtp = Qwen35MtpLayer {
        base: full_attn_layer(cfg),
        eh_proj: mat(ramp(2 * n_embd * n_embd, 0.01)),
        enorm: vec![1.0; n_embd],
        hnorm: vec![1.0; n_embd],
        embed_tokens: None,
        shared_head_head: None,
        shared_head_norm: None,
    };

    Qwen35Weights {
        token_embd: std::sync::Arc::from(ramp(n_vocab * n_embd, 0.001)),
        output_norm: vec![1.0; n_embd],
        output: None,
        token_embd_lm: None,
        trunk_layers: trunk,
        mtp_layers: vec![mtp],
    }
}
