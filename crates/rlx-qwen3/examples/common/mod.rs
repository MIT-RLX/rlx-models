// RLX models — distributed inference.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Shared fixtures for the `rlx-qwen3` examples: a parameterized synthetic
//! Qwen3 config, deterministic weight synthesis, and greedy `argmax`.
//!
//! Examples pull this in with:
//!
//! ```ignore
//! #[path = "common/mod.rs"]
//! mod common;
//! ```
//!
//! Not every example uses every helper, hence the module-wide `dead_code`
//! allow.
#![allow(dead_code)]

use rlx_qwen3::Qwen3Config;
use std::collections::HashMap;

/// `name -> (flat data, shape)` — the in-memory weight map the pipeline
/// stages filter down to their block.
pub type Tensors = HashMap<String, (Vec<f32>, Vec<usize>)>;

/// The size knobs that distinguish one synthetic Qwen3 from another. Every
/// other field is the usual Qwen3 default (see [`qwen3`]).
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub vocab: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub max_pos: usize,
}

impl Shape {
    /// Smallest useful shape — multi-process / correctness smoke tests.
    pub const TINY: Shape = Shape {
        vocab: 256,
        hidden: 64,
        intermediate: 128,
        layers: 6,
        heads: 4,
        kv_heads: 2,
        head_dim: 16,
        max_pos: 64,
    };

    /// Mid-size shape for decode/throughput benchmarks.
    pub const SMALL: Shape = Shape {
        vocab: 1000,
        hidden: 256,
        intermediate: 512,
        layers: 12,
        heads: 8,
        kv_heads: 2,
        head_dim: 32,
        max_pos: 128,
    };

    /// Larger shape for transport / pipeline-relay benchmarks.
    pub const MEDIUM: Shape = Shape {
        vocab: 2048,
        hidden: 512,
        intermediate: 1024,
        layers: 8,
        heads: 8,
        kv_heads: 2,
        head_dim: 64,
        max_pos: 64,
    };

    /// Same shape with a different layer count (some benches sweep depth).
    pub const fn with_layers(mut self, layers: usize) -> Shape {
        self.layers = layers;
        self
    }
}

/// Build a synthetic Qwen3 config of the given [`Shape`]; all non-shape
/// fields take standard Qwen3 values (RoPE, qk-norm, no experts).
pub fn qwen3(s: Shape) -> Qwen3Config {
    Qwen3Config {
        vocab_size: s.vocab,
        hidden_size: s.hidden,
        intermediate_size: s.intermediate,
        num_hidden_layers: s.layers,
        num_attention_heads: s.heads,
        num_key_value_heads: s.kv_heads,
        head_dim: s.head_dim,
        max_position_embeddings: s.max_pos,
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

/// Deterministic, name-seeded weights in a small symmetric range — same
/// `name` always yields the same tensor, so every rank/process agrees.
pub fn fill(name: &str, n: usize) -> Vec<f32> {
    let seed = name
        .bytes()
        .fold(2166136261u32, |a, b| (a ^ b as u32).wrapping_mul(16777619));
    (0..n)
        .map(|i| {
            let x = seed.wrapping_add((i as u32).wrapping_mul(2654435761));
            ((x % 2000) as f32 / 1000.0 - 1.0) * 0.05
        })
        .collect()
}

/// Synthesize a full set of Qwen3 weights for `c`.
pub fn synth(c: &Qwen3Config) -> Tensors {
    fn put(t: &mut Tensors, key: String, shape: Vec<usize>) {
        let n: usize = shape.iter().product();
        let d = fill(&key, n);
        t.insert(key, (d, shape));
    }
    let h = c.hidden_size;
    let q = c.num_attention_heads * c.head_dim;
    let kv = c.num_key_value_heads * c.head_dim;
    let im = c.intermediate_size;
    let dh = c.head_dim;
    let mut t = Tensors::new();
    put(
        &mut t,
        "model.embed_tokens.weight".into(),
        vec![c.vocab_size, h],
    );
    for l in 0..c.num_hidden_layers {
        let lp = format!("model.layers.{l}");
        put(&mut t, format!("{lp}.input_layernorm.weight"), vec![h]);
        put(
            &mut t,
            format!("{lp}.post_attention_layernorm.weight"),
            vec![h],
        );
        put(&mut t, format!("{lp}.self_attn.q_proj.weight"), vec![q, h]);
        put(&mut t, format!("{lp}.self_attn.k_proj.weight"), vec![kv, h]);
        put(&mut t, format!("{lp}.self_attn.v_proj.weight"), vec![kv, h]);
        put(&mut t, format!("{lp}.self_attn.o_proj.weight"), vec![h, q]);
        put(&mut t, format!("{lp}.self_attn.q_norm.weight"), vec![dh]);
        put(&mut t, format!("{lp}.self_attn.k_norm.weight"), vec![dh]);
        put(&mut t, format!("{lp}.mlp.gate_proj.weight"), vec![im, h]);
        put(&mut t, format!("{lp}.mlp.up_proj.weight"), vec![im, h]);
        put(&mut t, format!("{lp}.mlp.down_proj.weight"), vec![h, im]);
    }
    put(&mut t, "model.norm.weight".into(), vec![h]);
    put(&mut t, "lm_head.weight".into(), vec![c.vocab_size, h]);
    t
}

/// Greedy token: index of the max logit.
pub fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}
