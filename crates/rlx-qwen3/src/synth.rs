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

//! A small Qwen3 with made-up weights, for tests that need a whole model.
//!
//! Public, not `#[cfg(test)]`, because the things worth testing here are
//! *equivalences between crates* — a pipeline stage against the monolithic
//! graph, a runner against the loop it replaces — and an integration test
//! cannot reach a private test helper. `rlx-qwen4exp::synth` exists for the
//! same reason.
//!
//! ## The weights are deliberately not zeros
//!
//! `flow.rs`'s own test helper fills everything with `0.0`, which is fine for
//! checking that a graph builds and has the right shapes. It is useless for
//! checking that two *paths* agree: with zero weights every path produces the
//! same constant, so a pipeline split that drops a layer, swaps two ranks, or
//! mis-slices the KV cache passes just as well as a correct one. Everything
//! here is non-zero and position-dependent, so a structural mistake changes
//! the output.

use std::collections::HashMap;

use crate::config::Qwen3Config;

/// One tensor: values plus shape, as a weight map wants them.
pub type Tensors = HashMap<String, (Vec<f32>, Vec<usize>)>;

/// Deterministic, non-zero, and different per `seed`.
///
/// Small magnitudes keep a few layers of attention + SwiGLU away from
/// saturation, so a difference in the output reflects a difference in the
/// computation rather than two different flavours of `inf`.
pub fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.25
        })
        .collect()
}

/// A Qwen3 small enough to run in a unit test, with `layers` decoder layers.
///
/// GQA is real here (4 query heads over 2 KV heads) and QK-norm is on, because
/// both are places a layer-range split can go wrong: the KV cache is narrower
/// than the query projection, and the per-head norms are indexed by head.
pub fn tiny_config(layers: usize) -> Qwen3Config {
    Qwen3Config {
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: layers,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        max_position_embeddings: 32,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        hidden_act: "silu".into(),
        tie_word_embeddings: false,
        attention_bias: false,
        qk_norm: true,
        sliding_window: None,
        max_window_layers: usize::MAX,
        use_sliding_window: false,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        shared_expert_ffn_size: 0,
        expert_weights_scale: 1.0,
    }
}

/// Every tensor a dense Qwen3 of `cfg`'s shape reads, in HF naming.
///
/// The per-layer seed is derived from the layer index, so layer `i`'s weights
/// differ from layer `j`'s — which is what makes a reordered or dropped layer
/// visible in the logits.
pub fn synth_tensors(cfg: &Qwen3Config) -> Tensors {
    let h = cfg.hidden_size;
    let q_dim = cfg.q_proj_dim();
    let kv_dim = cfg.kv_proj_dim();
    let int_dim = cfg.intermediate_size;
    let dh = cfg.head_dim;
    let mut t: Tensors = HashMap::new();

    t.insert(
        "model.embed_tokens.weight".into(),
        (fill(cfg.vocab_size * h, 1), vec![cfg.vocab_size, h]),
    );
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        // Offset the seed by the layer so no two layers share weights.
        let s = |k: u64| 100 + (i as u64) * 20 + k;
        t.insert(
            format!("{lp}.input_layernorm.weight"),
            (fill(h, s(0)), vec![h]),
        );
        t.insert(
            format!("{lp}.post_attention_layernorm.weight"),
            (fill(h, s(1)), vec![h]),
        );
        t.insert(
            format!("{lp}.self_attn.q_proj.weight"),
            (fill(q_dim * h, s(2)), vec![q_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.k_proj.weight"),
            (fill(kv_dim * h, s(3)), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.v_proj.weight"),
            (fill(kv_dim * h, s(4)), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.o_proj.weight"),
            (fill(h * q_dim, s(5)), vec![h, q_dim]),
        );
        if cfg.qk_norm {
            t.insert(
                format!("{lp}.self_attn.q_norm.weight"),
                (fill(dh, s(6)), vec![dh]),
            );
            t.insert(
                format!("{lp}.self_attn.k_norm.weight"),
                (fill(dh, s(7)), vec![dh]),
            );
        }
        t.insert(
            format!("{lp}.mlp.gate_proj.weight"),
            (fill(int_dim * h, s(8)), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.up_proj.weight"),
            (fill(int_dim * h, s(9)), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.down_proj.weight"),
            (fill(h * int_dim, s(10)), vec![h, int_dim]),
        );
    }
    t.insert("model.norm.weight".into(), (fill(h, 2), vec![h]));
    if !cfg.tie_word_embeddings {
        t.insert(
            "lm_head.weight".into(),
            (fill(cfg.vocab_size * h, 3), vec![cfg.vocab_size, h]),
        );
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Layers get different weights — the property the equivalence tests rest
    /// on.
    #[test]
    fn each_layer_has_its_own_weights() {
        let cfg = tiny_config(3);
        let t = synth_tensors(&cfg);
        let g = |i: usize| {
            t[&format!("model.layers.{i}.mlp.gate_proj.weight")]
                .0
                .clone()
        };
        assert_ne!(g(0), g(1), "layer 0 and 1 must differ");
        assert_ne!(g(1), g(2), "layer 1 and 2 must differ");
        assert!(
            g(0).iter().any(|v| *v != 0.0),
            "all-zero weights would make every path agree"
        );
    }

    /// The tensor set is exactly what the layer count implies.
    #[test]
    fn the_tensor_set_scales_with_the_layer_count() {
        let per_layer = 11; // 2 norms + 4 projections + 2 qk-norms + 3 mlp
        for layers in [1usize, 4] {
            let cfg = tiny_config(layers);
            let t = synth_tensors(&cfg);
            // embed + final norm + lm_head, plus the per-layer block.
            assert_eq!(t.len(), 3 + layers * per_layer, "{layers} layers");
        }
    }
}
