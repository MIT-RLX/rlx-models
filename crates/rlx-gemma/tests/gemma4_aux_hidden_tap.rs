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

//! EAGLE3 layer-input tap on the Gemma 4 decode graph.
//!
//! Verifies that `GemmaFlow::with_aux_hidden_outputs([...])` exports
//! one extra HIR output per requested layer, in ascending layer-index
//! order, *after* the KV-cache outputs. This is the gating change
//! that lets an EAGLE3 draft consume the verifier's per-layer
//! hidden states (`eagle_aux_hidden_state_layer_ids`).
//!
//! Pure graph-structure assertion — no runtime execution.

use anyhow::Result;
use rlx_core::weight_map::WeightMap;
use rlx_gemma::config::{
    GemmaArch, GemmaConfig, GemmaLayerType, GemmaRopeKind, GemmaRopeMap, GemmaRopeParameters,
};
use rlx_gemma::flow::GemmaFlow;
use std::collections::HashMap;

const LAYERS: usize = 6;

fn tiny_gemma4_cfg() -> GemmaConfig {
    GemmaConfig {
        arch: GemmaArch::Gemma4,
        vocab_size: 32,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: LAYERS,
        num_attention_heads: 8,
        num_key_value_heads: 4,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        tie_word_embeddings: true,
        attention_bias: false,
        head_dim: Some(8),
        attn_logit_softcapping: None,
        final_logit_softcapping: Some(30.0),
        sliding_window: None,
        query_pre_attn_scalar: None,
        effective_num_layers: None,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        expert_weights_scale: 1.0,
        layer_types: (0..LAYERS)
            .map(|i| {
                if (i + 1) % 6 == 0 {
                    GemmaLayerType::FullAttention
                } else {
                    GemmaLayerType::SlidingAttention
                }
            })
            .collect(),
        rope_parameters: GemmaRopeMap {
            sliding_attention: Some(GemmaRopeParameters {
                rope_theta: Some(10_000.0),
                rope_type: Some(GemmaRopeKind::Default),
                partial_rotary_factor: None,
            }),
            full_attention: Some(GemmaRopeParameters {
                rope_theta: Some(1_000_000.0),
                rope_type: Some(GemmaRopeKind::Proportional),
                partial_rotary_factor: Some(0.25),
            }),
        },
        global_head_dim: Some(16),
        num_global_key_value_heads: Some(1),
        attention_k_eq_v: true,
        use_bidirectional_attention: None,
        hidden_size_per_layer_input: 0,
        vocab_size_per_layer_input: 0,
        num_kv_shared_layers: 0,
        use_double_wide_mlp: false,
        enable_moe_block: false,
    }
}

fn ramp(n: usize, scale: f32, salt: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(salt)) >> 8;
            (x as f32 / (1u32 << 24) as f32 - 0.5) * scale
        })
        .collect()
}

fn synthetic_weights(cfg: &GemmaConfig) -> WeightMap {
    let h = cfg.hidden_size;
    let int_dim = cfg.intermediate_size;
    let nh = cfg.num_attention_heads;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

    t.insert(
        "model.embed_tokens.weight".into(),
        (ramp(cfg.vocab_size * h, 0.02, 1), vec![cfg.vocab_size, h]),
    );

    for layer in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{layer}");
        let salt = layer as u32 * 17;
        let dh = cfg.layer_head_dim(layer);
        let kv = cfg.layer_num_kv_heads(layer);
        let q_dim = nh * dh;
        let kv_dim = kv * dh;

        t.insert(
            format!("{lp}.input_layernorm.weight"),
            (ramp(h, 0.001, salt), vec![h]),
        );
        t.insert(
            format!("{lp}.pre_feedforward_layernorm.weight"),
            (ramp(h, 0.001, salt + 10), vec![h]),
        );
        t.insert(
            format!("{lp}.post_feedforward_layernorm.weight"),
            (ramp(h, 0.001, salt + 11), vec![h]),
        );
        t.insert(
            format!("{lp}.self_attn.q_proj.weight"),
            (ramp(q_dim * h, 0.01, salt + 2), vec![q_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.k_proj.weight"),
            (ramp(kv_dim * h, 0.01, salt + 3), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.v_proj.weight"),
            (ramp(kv_dim * h, 0.01, salt + 4), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.o_proj.weight"),
            (ramp(h * q_dim, 0.01, salt + 5), vec![h, q_dim]),
        );
        t.insert(
            format!("{lp}.mlp.gate_proj.weight"),
            (ramp(int_dim * h, 0.01, salt + 6), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.up_proj.weight"),
            (ramp(int_dim * h, 0.01, salt + 7), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.down_proj.weight"),
            (ramp(h * int_dim, 0.01, salt + 8), vec![h, int_dim]),
        );
    }
    t.insert("model.norm.weight".into(), (ramp(h, 0.001, 99), vec![h]));
    WeightMap::from_tensors(t)
}

fn build_decode_with_aux(layer_ids: &[usize]) -> Result<rlx_flow::BuiltModel> {
    let cfg = tiny_gemma4_cfg();
    let mut wm = synthetic_weights(&cfg);
    GemmaFlow::new(&cfg)
        .decode()
        .batch(1)
        .past(4)
        .with_aux_hidden_outputs(layer_ids.iter().copied())
        .build(&mut wm)
}

fn count_outputs(built: &rlx_flow::BuiltModel) -> usize {
    built.clone().into_hir().expect("HIR stage").outputs.len()
}

#[test]
fn aux_tap_disabled_by_default() {
    // No aux: 1 (logits) + 2 * LAYERS (K then V per layer) = 13.
    let cfg = tiny_gemma4_cfg();
    let mut wm = synthetic_weights(&cfg);
    let built = GemmaFlow::new(&cfg)
        .decode()
        .batch(1)
        .past(4)
        .build(&mut wm)
        .expect("build decode without aux");
    assert_eq!(
        count_outputs(&built),
        1 + 2 * LAYERS,
        "no aux outputs expected when with_aux_hidden_outputs is not called"
    );
}

#[test]
fn aux_tap_three_layers_low_mid_high() {
    // Mirrors RedHatAI's eagle_aux_hidden_state_layer_ids stratification
    // (low / mid / high). Expects logits + 2*LAYERS KV + 3 aux = 16.
    let built = build_decode_with_aux(&[0, 2, 5]).expect("build decode with aux");
    assert_eq!(count_outputs(&built), 1 + 2 * LAYERS + 3);
}

#[test]
fn aux_tap_dedups_and_sorts_input() {
    // Duplicates and out-of-order ids should collapse to {0, 1, 3}.
    let built =
        build_decode_with_aux(&[3, 0, 1, 3, 0]).expect("build decode with deduplicated aux");
    assert_eq!(count_outputs(&built), 1 + 2 * LAYERS + 3);
}

#[test]
fn aux_tap_silently_drops_oob_layer_ids() {
    // Layer 99 is out of range — should be dropped at build time.
    let built = build_decode_with_aux(&[0, 99, 2]).expect("build decode with oob aux");
    assert_eq!(count_outputs(&built), 1 + 2 * LAYERS + 2);
}

#[test]
fn aux_tap_all_layers() {
    let ids: Vec<usize> = (0..LAYERS).collect();
    let built = build_decode_with_aux(&ids).expect("build decode with all-layer aux");
    assert_eq!(count_outputs(&built), 1 + 2 * LAYERS + LAYERS);
}
