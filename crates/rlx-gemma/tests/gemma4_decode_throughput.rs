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

#![allow(dead_code)]

//! Decode-step throughput micro-benchmark (Gemma 4-shape tiny graph).
//!
//! ```bash
//! cargo test -p rlx-gemma --release --features apple-silicon \
//!     --test gemma4_decode_throughput -- --nocapture
//! ```
//!
//! Measures wall-clock per-decode-step time on each compiled
//! backend. These are **tiny-graph** numbers (6 layers, hidden=64);
//! they reflect kernel-launch + dispatch overhead more than steady-
//! state Gemma 4 12B throughput. Real-model throughput would need
//! the actual weights and would be 4–5 orders of magnitude slower
//! per token. Use this for **regression detection** (alert when a
//! backend's per-step jumps unexpectedly), not absolute claims.

use anyhow::Result;
use rlx_core::weight_map::WeightMap;
use rlx_gemma::config::{
    GemmaArch, GemmaConfig, GemmaLayerType, GemmaRopeKind, GemmaRopeMap, GemmaRopeParameters,
};
use rlx_gemma::generator::GemmaGenerator;
use rlx_runtime::Device;
use std::collections::HashMap;
use std::time::Instant;

const PREFILL: &[u32] = &[1, 2, 3, 4];
const STEPS: usize = 32;

fn tiny_cfg() -> GemmaConfig {
    GemmaConfig {
        arch: GemmaArch::Gemma4,
        vocab_size: 32,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 6,
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
        layer_types: (0..6)
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
        eog_token_ids: Vec::new(),
        activation_sparsity_pattern: Vec::new(),
        altup_num_inputs: 0,
        altup_active_idx: 0,
        altup_coef_clip: None,
        altup_correct_scale: false,
        laurel_rank: 0,
        rope_local_base_freq: 10_000.0,
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

fn synthetic_weights_gemma4(cfg: &GemmaConfig) -> WeightMap {
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
    t.insert(
        "lm_head.weight".into(),
        (
            ramp(cfg.vocab_size * h, 0.02, 3000),
            vec![cfg.vocab_size, h],
        ),
    );
    WeightMap::from_tensors(t)
}

fn ensure_metal_precise() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        unsafe { std::env::set_var("RLX_METAL_PRECISE", "1") };
    });
}

fn bench_decode(device: Device) -> Result<(f64, f64)> {
    ensure_metal_precise();
    let cfg = tiny_cfg();
    let mut wm = synthetic_weights(&cfg);
    let mut g = GemmaGenerator::from_loader(cfg, &mut wm, device)?;

    // Warmup prefill + 4 decode steps to amortize compile.
    g.prefill_get_last_logits(PREFILL)?;
    for s in 0..4 {
        g.decode_get_logits((s as u32) + 5)?;
    }

    // Reset to clean cache and measure.
    g.prefill_get_last_logits(PREFILL)?;
    let start = Instant::now();
    for s in 0..STEPS {
        g.decode_get_logits((s as u32) + 5)?;
    }
    let elapsed = start.elapsed();
    let total_s = elapsed.as_secs_f64();
    let per_step_us = (total_s * 1e6) / STEPS as f64;
    let toks_per_s = STEPS as f64 / total_s;
    eprintln!(
        "[gemma4 decode bench] {device:?} {STEPS} steps in {total_s:.4}s — {per_step_us:.1} µs/step ({toks_per_s:.0} tok/s synthetic)"
    );
    Ok((per_step_us, toks_per_s))
}

#[test]
fn bench_decode_cpu() {
    let _ = bench_decode(Device::Cpu).expect("CPU decode bench");
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn bench_decode_metal() {
    if !is_available(Device::Metal) {
        return;
    }
    let _ = bench_decode(Device::Metal).expect("Metal decode bench");
}

/// Bucketed `step_cached` throughput at Gemma4-shaped hidden=1024 (6 layers).
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn bench_step_cached_metal_1024() {
    use rlx_gemma::config::{
        GemmaArch, GemmaConfig, GemmaLayerType, GemmaRopeKind, GemmaRopeMap, GemmaRopeParameters,
    };
    if !is_available(Device::Metal) {
        return;
    }
    use rlx_qwen3::SampleOpts;
    let cfg = GemmaConfig {
        arch: GemmaArch::Gemma4,
        vocab_size: 8192,
        hidden_size: 1024,
        intermediate_size: 4096,
        num_hidden_layers: 6,
        num_attention_heads: 8,
        num_key_value_heads: 4,
        max_position_embeddings: 4096,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        tie_word_embeddings: true,
        attention_bias: false,
        head_dim: Some(128),
        attn_logit_softcapping: None,
        final_logit_softcapping: Some(30.0),
        sliding_window: Some(1024),
        query_pre_attn_scalar: None,
        effective_num_layers: None,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        expert_weights_scale: 1.0,
        layer_types: (0..6)
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
        global_head_dim: Some(256),
        num_global_key_value_heads: Some(2),
        attention_k_eq_v: true,
        use_bidirectional_attention: Some("vision".into()),
        hidden_size_per_layer_input: 0,
        vocab_size_per_layer_input: 0,
        num_kv_shared_layers: 0,
        use_double_wide_mlp: false,
        enable_moe_block: false,
        eog_token_ids: Vec::new(),
        activation_sparsity_pattern: Vec::new(),
        altup_num_inputs: 0,
        altup_active_idx: 0,
        altup_coef_clip: None,
        altup_correct_scale: false,
        laurel_rank: 0,
        rope_local_base_freq: 10_000.0,
    };
    let mut wm = synthetic_weights_gemma4(&cfg);
    let horizon = 32 + 128;
    let mut g = GemmaGenerator::from_loader(cfg, &mut wm, Device::Metal)
        .expect("generator")
        .with_inference_caches(horizon);
    let prefill: Vec<u32> = (0..32).collect();
    g.prefill(&prefill);
    let _ = g.generate_cached(1, SampleOpts::greedy()).expect("seed");
    for _ in 0..64 {
        g.step_cached(SampleOpts::greedy()).expect("warm");
    }
    g.prefill(&prefill);
    let _ = g.generate_cached(1, SampleOpts::greedy()).expect("seed");
    let steps = 128usize;
    let t0 = Instant::now();
    for _ in 0..steps {
        g.step_cached(SampleOpts::greedy()).expect("step");
    }
    let dt = t0.elapsed();
    let tok_s = steps as f64 / dt.as_secs_f64();
    eprintln!(
        "[gemma4 step_cached] Metal hidden=1024 layers=6: {steps} steps in {:.3}s ({tok_s:.1} tok/s)",
        dt.as_secs_f64()
    );
    assert!(
        tok_s > 10.0,
        "Metal step_cached unexpectedly slow ({tok_s:.1} tok/s)"
    );
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn bench_decode_mlx() {
    if !is_available(Device::Mlx) {
        return;
    }
    let _ = bench_decode(Device::Mlx).expect("MLX decode bench");
}

#[cfg(feature = "gpu")]
#[test]
fn bench_decode_wgpu() {
    if !is_available(Device::Gpu) {
        return;
    }
    let _ = bench_decode(Device::Gpu).expect("wgpu decode bench");
}
