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
    MatWeight, PestleFactor, Proj, Qwen35Config, Qwen35FullAttnLayer, Qwen35LayerFfn,
    Qwen35LinearLayer, Qwen35MoeFfn, Qwen35MtpLayer, Qwen35TrunkLayer, Qwen35Weights,
};

pub fn mat(data: Vec<f32>) -> MatWeight {
    MatWeight::F32(data)
}

/// Dense [`Proj`] for synthesized layers. Pestle factorization has its
/// own fixtures ([`pestle_and_dense`], [`synth_weights_pestle`]) so the
/// default synth path stays a plain matmul.
pub fn proj(data: Vec<f32>) -> Proj {
    Proj::Dense(MatWeight::F32(data))
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
        attn_qkv: proj(ramp(n_embd * conv_channels, 0.01)),
        attn_gate: proj(ramp(n_embd * value_dim, 0.01)),
        ssm_conv1d: ramp(k_conv * conv_channels, 0.02),
        ssm_dt_bias: ramp(n_v_heads, 0.05),
        ssm_a: vec![-1.0; n_v_heads],
        ssm_beta: proj(ramp(n_embd * n_v_heads, 0.01)),
        ssm_alpha: proj(ramp(n_embd * n_v_heads, 0.01)),
        ssm_norm: vec![1.0; n_state],
        ssm_out: proj(ramp(value_dim * n_embd, 0.01)),
        ffn: Qwen35LayerFfn::Dense {
            gate: proj(ramp(n_embd * n_ff, 0.01)),
            down: proj(ramp(n_ff * n_embd, 0.01)),
            up: proj(ramp(n_embd * n_ff, 0.01)),
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
        attn_q_gate: proj(ramp(n_embd * q_gate_cols, 0.01)),
        attn_k: proj(ramp(n_embd * kv_cols, 0.01)),
        attn_v: proj(ramp(n_embd * kv_cols, 0.01)),
        attn_output: proj(ramp(n_head * head_dim * n_embd, 0.01)),
        attn_q_norm: vec![1.0; head_dim],
        attn_k_norm: vec![1.0; head_dim],
        ffn: Qwen35LayerFfn::Dense {
            gate: proj(ramp(n_embd * n_ff, 0.01)),
            down: proj(ramp(n_ff * n_embd, 0.01)),
            up: proj(ramp(n_embd * n_ff, 0.01)),
        },
    }
}

// ── Pestle fixtures ──────────────────────────────────────────────

/// Deterministic pseudo-random value in `[-1, 1)`.
fn prand(seed: u64, i: usize) -> f32 {
    let mut x = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    ((x >> 40) as f32) / 8_388_608.0 - 1.0
}

/// A Pestle factor pair and the dense `[out, n_in]` matrix it is
/// **exactly** equivalent to.
///
/// `emit_linear` computes
/// `y = scale_post ⊙ (U (scale_mid ⊙ (V (scale_pre ⊙ x))))`, so
///
/// ```text
///   W[o,i] = scale_post[o] · (Σ_r U[o,r] · scale_mid[r] · V[r,i]) · scale_pre[i]
/// ```
///
/// Building one model from each half and requiring identical logits
/// pins the whole convention at once: factor order (`V` then `U`), the
/// `[out, in]` transpose `proj_mat` applies, and — because `scale_pre`
/// and `scale_post` have different lengths — the axis each of the three
/// scales broadcasts along. Any of those wrong and the two diverge.
pub fn pestle_and_dense(out: usize, n_in: usize, rank: usize, seed: u64) -> (Proj, Proj) {
    let v: Vec<f32> = (0..rank * n_in).map(|i| 0.3 * prand(seed, i)).collect();
    let u: Vec<f32> = (0..out * rank)
        .map(|i| 0.3 * prand(seed ^ 0xA1, i))
        .collect();
    let scale_pre: Vec<f32> = (0..n_in)
        .map(|i| 0.5 + 0.4 * prand(seed ^ 0xB2, i))
        .collect();
    let scale_mid: Vec<f32> = (0..rank)
        .map(|i| 0.5 + 0.4 * prand(seed ^ 0xC3, i))
        .collect();
    let scale_post: Vec<f32> = (0..out)
        .map(|i| 0.5 + 0.4 * prand(seed ^ 0xD4, i))
        .collect();

    let mut w = vec![0f32; out * n_in];
    for o in 0..out {
        for r in 0..rank {
            let ur = u[o * rank + r] * scale_mid[r];
            for i in 0..n_in {
                w[o * n_in + i] += ur * v[r * n_in + i];
            }
        }
        for i in 0..n_in {
            w[o * n_in + i] *= scale_post[o] * scale_pre[i];
        }
    }

    (
        Proj::Pestle(Box::new(PestleFactor {
            v: MatWeight::F32(v),
            u: MatWeight::F32(u),
            scale_pre,
            scale_mid,
            scale_post,
            rank,
        })),
        Proj::Dense(MatWeight::F32(w)),
    )
}

/// Pick the Pestle or the dense half of [`pestle_and_dense`].
fn paired(out: usize, n_in: usize, rank: usize, seed: u64, pestle: bool) -> Proj {
    let (p, d) = pestle_and_dense(out, n_in, rank, seed);
    if pestle { p } else { d }
}

/// [`synth_weights`] with every trunk projection replaced by a
/// Pestle factor pair (`pestle = true`) or by its exact dense
/// equivalent (`pestle = false`). The two bundles are the *same model*
/// numerically, so their logits must match.
///
/// MTP layers stay dense either way — no Pestle checkpoint ships an MTP
/// head, and the MTP builders take the [`Proj::dense`] path.
pub fn synth_weights_pestle(cfg: &Qwen35Config, pestle: bool) -> Qwen35Weights {
    let n_embd = cfg.hidden_size;
    let n_vocab = cfg.vocab_size;
    let n_main = cfg.num_hidden_layers - cfg.nextn_predict_layers;
    let interval = cfg.full_attention_interval.max(1);
    let n_ff = cfg.intermediate_size;
    let n_state = cfg.ssm_state_size;
    let n_v_heads = cfg.ssm_time_step_rank;
    let value_dim = n_state * n_v_heads;
    let conv_channels = n_state * cfg.ssm_group_count * 2 + value_dim;
    let head_dim = cfg.key_length;
    let q_gate_cols = cfg.num_attention_heads * head_dim * 2;
    let kv_cols = cfg.num_key_value_heads * head_dim;
    let q_dim = cfg.num_attention_heads * head_dim;

    // Seed per (layer, slot) so both halves stay in lockstep. Ranks are
    // deliberately not all equal — Pestle's own ranks vary 128…3968 per
    // slot, and a builder that assumed one rank per layer would pass a
    // uniform-rank fixture.
    let seed = |il: usize, slot: usize| ((il as u64) << 8 | slot as u64).wrapping_add(0x51ED);
    let ffn = |il: usize| Qwen35LayerFfn::Dense {
        gate: paired(n_ff, n_embd, 5, seed(il, 5), pestle),
        up: paired(n_ff, n_embd, 6, seed(il, 6), pestle),
        down: paired(n_embd, n_ff, 7, seed(il, 7), pestle),
    };

    let mut trunk = Vec::new();
    for il in 0..n_main {
        trunk.push(if ((il + 1) % interval) == 0 {
            Qwen35TrunkLayer::FullAttn(Qwen35FullAttnLayer {
                attn_norm: vec![1.0; n_embd],
                attn_post_norm: vec![1.0; n_embd],
                attn_q_gate: paired(q_gate_cols, n_embd, 3, seed(il, 0), pestle),
                attn_k: paired(kv_cols, n_embd, 4, seed(il, 1), pestle),
                attn_v: paired(kv_cols, n_embd, 5, seed(il, 2), pestle),
                attn_output: paired(n_embd, q_dim, 6, seed(il, 3), pestle),
                attn_q_norm: vec![1.0; head_dim],
                attn_k_norm: vec![1.0; head_dim],
                ffn: ffn(il),
            })
        } else {
            Qwen35TrunkLayer::Linear(Qwen35LinearLayer {
                attn_norm: vec![1.0; n_embd],
                attn_post_norm: vec![1.0; n_embd],
                attn_qkv: paired(conv_channels, n_embd, 3, seed(il, 0), pestle),
                attn_gate: paired(value_dim, n_embd, 4, seed(il, 1), pestle),
                ssm_conv1d: ramp(cfg.ssm_conv_kernel * conv_channels, 0.02),
                ssm_dt_bias: ramp(n_v_heads, 0.05),
                ssm_a: vec![-1.0; n_v_heads],
                ssm_beta: paired(n_v_heads, n_embd, 5, seed(il, 2), pestle),
                ssm_alpha: paired(n_v_heads, n_embd, 6, seed(il, 3), pestle),
                ssm_norm: vec![1.0; n_state],
                ssm_out: paired(n_embd, value_dim, 7, seed(il, 4), pestle),
                ffn: ffn(il),
            })
        });
    }

    Qwen35Weights {
        token_embd: std::sync::Arc::from(ramp(n_vocab * n_embd, 0.001)),
        output_norm: vec![1.0; n_embd],
        output: None,
        token_embd_lm: None,
        trunk_layers: trunk,
        mtp_layers: (0..cfg.nextn_predict_layers)
            .map(|_| Qwen35MtpLayer {
                base: full_attn_layer(cfg),
                eh_proj: mat(ramp(2 * n_embd * n_embd, 0.01)),
                enorm: vec![1.0; n_embd],
                hnorm: vec![1.0; n_embd],
                embed_tokens: None,
                shared_head_head: None,
                shared_head_norm: None,
            })
            .collect(),
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
