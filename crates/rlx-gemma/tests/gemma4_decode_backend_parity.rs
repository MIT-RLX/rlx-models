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

//! Cross-backend numerical parity for the **Gemma 4 decode path** —
//! the per-step decode graph uses different code than prefill
//! (`build_gemma_decode_flow`, dynamic cos/sin row, named secondary
//! rope binding via `CustomStage`), so it gets its own parity
//! coverage.
//!
//! ```bash
//! cargo test -p rlx-gemma --test gemma4_decode_backend_parity --features apple-silicon
//! ```
//!
//! Runs `GemmaGenerator::prefill` followed by `decode_get_logits` on
//! each backend and compares the decode-step logits to CPU.
//!
//! Tolerances mirror the prefill suite. The Metal tests set
//! `RLX_METAL_PRECISE=1` so the scalar fp32 sgemm path is used and
//! `simdgroup_float8x8`'s reduced-precision accumulators don't
//! dominate the delta (see `gemma4_lm_backend_parity.rs` header).
//!
//! ## Known limitation: per-layer KV cache dim
//!
//! The shared `LayerKvCache` infrastructure (used by Whisper, Gemma,
//! and several other crates) currently assumes a single per-layer
//! KV dim. Gemma 4 12B's full-attention layers want `kv_dim = 512`
//! while sliding layers want `2048`. Wiring per-layer kv_dim into
//! the cache infrastructure is a separate multi-crate workstream;
//! this test uses a uniform-shape Gemma 4 variant
//! (`global_head_dim == head_dim`, `num_global_kv_heads ==
//! num_kv_heads`) so the new ops `attention_k_eq_v` + split RoPE +
//! partial rotary factor are exercised on the decode path while
//! the cache layout stays compatible.

use anyhow::Result;
use rlx_core::weight_map::WeightMap;
use rlx_gemma::config::{
    GemmaArch, GemmaConfig, GemmaLayerType, GemmaRopeKind, GemmaRopeMap, GemmaRopeParameters,
};
use rlx_gemma::generator::GemmaGenerator;
use rlx_runtime::Device;
use std::collections::HashMap;

const PREFILL: &[u32] = &[1, 2, 3, 4];
const DECODE_TOKEN: u32 = 5;

// ── Gemma 4-shape config for decode parity ───────────────────────

fn tiny_gemma4_decode_cfg() -> GemmaConfig {
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
        // Full per-layer divergence (mirrors real Gemma 4 12B):
        // full-attention layers use head_dim=16, kv_heads=1; sliding
        // layers use head_dim=8, kv_heads=4. End-to-end works now
        // that the LayerKvCache + KvTapStage are per-layer-aware.
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
    }
}

// ── Deterministic synthetic weights with per-layer shape ─────────

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
        // v_proj.weight is aliased to k via attention_k_eq_v but we
        // ship it for completeness (unused by the flow).
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

// ── Metal precise mode setup ─────────────────────────────────────

fn ensure_metal_precise() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // SAFETY: set once before any backend thread reads it.
        unsafe { std::env::set_var("RLX_METAL_PRECISE", "1") };
    });
}

// ── One prefill + one decode step on `device`, return decode logits

fn run_decode_step(device: Device) -> Result<Vec<f32>> {
    ensure_metal_precise();
    let cfg = tiny_gemma4_decode_cfg();
    let mut wm = synthetic_weights(&cfg);
    let mut generator = GemmaGenerator::from_loader(cfg, &mut wm, device)?;
    generator.prefill_get_last_logits(PREFILL)?;
    generator.decode_get_logits(DECODE_TOKEN)
}

fn max_abs_delta(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "shape mismatch: cpu={} dev={}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

fn assert_decode_parity(device: Device, tol: f32) {
    let cpu = run_decode_step(Device::Cpu).expect("decode on CPU");
    let dev = run_decode_step(device).expect("decode on device");
    assert!(
        cpu.iter().all(|v| v.is_finite()),
        "CPU produced non-finite decode logits"
    );
    assert!(
        dev.iter().all(|v| v.is_finite()),
        "{device:?} produced non-finite decode logits"
    );
    let d = max_abs_delta(&cpu, &dev);
    eprintln!("[gemma4 decode parity] {device:?} max|Δ|={d:.3e} (tol={tol:.1e})");
    assert!(
        d < tol,
        "Gemma 4 decode parity {device:?} vs CPU: max|Δ|={d:.3e} > tol={tol:.1e}",
    );
}

// ── CPU baseline ─────────────────────────────────────────────────

#[test]
fn gemma4_decode_runs_on_cpu() {
    let logits = run_decode_step(Device::Cpu).expect("decode on CPU");
    assert_eq!(logits.len(), tiny_gemma4_decode_cfg().vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

// ── Cross-backend parity ────────────────────────────────────────

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn gemma4_decode_matches_cpu_on_metal() {
    if !is_available(Device::Metal) {
        return;
    }
    assert_decode_parity(Device::Metal, 1e-3);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn gemma4_decode_matches_cpu_on_mlx() {
    if !is_available(Device::Mlx) {
        return;
    }
    assert_decode_parity(Device::Mlx, 1e-5);
}

#[cfg(feature = "gpu")]
#[test]
fn gemma4_decode_matches_cpu_on_wgpu() {
    if !is_available(Device::Gpu) {
        return;
    }
    assert_decode_parity(Device::Gpu, 1e-3);
}

#[cfg(feature = "cuda")]
#[test]
fn gemma4_decode_matches_cpu_on_cuda() {
    if !is_available(Device::Cuda) {
        return;
    }
    assert_decode_parity(Device::Cuda, 1e-2);
}

#[cfg(feature = "rocm")]
#[test]
fn gemma4_decode_matches_cpu_on_rocm() {
    if !is_available(Device::Rocm) {
        return;
    }
    assert_decode_parity(Device::Rocm, 1e-2);
}

#[cfg(feature = "vulkan")]
#[test]
fn gemma4_decode_matches_cpu_on_vulkan() {
    if !is_available(Device::Vulkan) {
        return;
    }
    assert_decode_parity(Device::Vulkan, 1e-2);
}
