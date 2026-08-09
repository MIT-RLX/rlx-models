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

//! Cross-backend numerical parity for the **Gemma 4 LM graph** —
//! exercises k_eq_v + split RoPE (sliding theta=10k, full theta=1M,
//! partial rotary factor 0.25) + per-layer head_dim / num_kv_heads
//! divergence.
//!
//! ```bash
//! cargo test -p rlx-gemma --test gemma4_lm_backend_parity --features apple-silicon
//! ```
//!
//! Per-backend tolerance versus CPU (measured on a tiny 6-layer
//! graph; the test sets `RLX_METAL_SGEMM_VARIANT=naive` for Metal
//! to bypass `simdgroup_float8x8`'s reduced-precision tensor units):
//!
//! | Backend | Tolerance | Measured | Notes |
//! |---|---|---|---|
//! | MLX | `1e-5` | ~`8e-8` | essentially bit-exact |
//! | wgpu | `1e-3` | ~`1e-6` | fp32-precise |
//! | Metal causal | `1e-3` | ~`1e-6` | precise scalar sgemm path |
//! | Metal sliding | `1e-3` | ~`1e-6` | SDPA `mask_kind=4` wired in `rlx-metal/src/kernels.rs` |
//! | CUDA / ROCm / Vulkan | `1e-2` | not run on this host | hardware-gated |
//!
//! ## Background on the Metal precision path
//!
//! Apple Silicon `simdgroup_float8x8` matmul instructions use
//! reduced-precision internal accumulators (~fp16 class for the
//! tensor-multiply step). For production inference of large LMs the
//! resulting ~1e-3 relative error is well-bounded and the 10–100×
//! speed win is dominant; for parity testing it shows up as ~1e-1
//! absolute drift in last-token logits. Setting
//! `RLX_METAL_SGEMM_VARIANT=naive` routes through the scalar fp32
//! `sgemm` kernel, which brings Metal to within fp32 reduction
//! noise of CPU. Tests do this automatically via `MetalPreciseGuard`.
//!
//! `rlx-metal/src/kernels.rs::sdpa` was also tightened in this work
//! to use `precise::sqrt` + `precise::exp` (avoiding the
//! reduced-precision defaults in the `metal::` namespace).

use anyhow::Result;
use rlx_core::flow_util::compile_graph_gemma_prefill_with_params;
use rlx_core::weight_map::WeightMap;
use rlx_gemma::config::{
    GemmaArch, GemmaConfig, GemmaLayerType, GemmaRopeKind, GemmaRopeMap, GemmaRopeParameters,
};
use rlx_gemma::flow::{GemmaPrefillOpts, build_gemma_prefill_flow};
use rlx_runtime::Device;
#[cfg(any(
    feature = "metal",
    feature = "mlx",
    feature = "gpu",
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan"
))]
use rlx_runtime::is_available;
use std::collections::HashMap;

const SEQ: usize = 4;

// ── Tiny Gemma 4 config (mirrors the 12B's per-layer split shape) ──

/// Gemma 4-shape config with **all causal attention** (no sliding
/// window). Exercises the new ops Gemma 4 specifically introduces:
/// `attention_k_eq_v`, split RoPE per layer type, partial rotary
/// factor on full layers, and per-layer head_dim / num_kv_heads
/// divergence. Works on every backend.
fn tiny_gemma4_causal_cfg() -> GemmaConfig {
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
        // No sliding window — every layer uses causal attention.
        // The "sliding"/"full" layer_types distinction below still
        // drives per-layer head_dim and RoPE-table dispatch, just
        // without changing the mask kind.
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
        // Full-attention layers: head_dim=16, kv_heads=1. Sliding:
        // head_dim=8, kv_heads=4. This exercises the per-layer
        // shape divergence the new flow code must handle.
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

/// Same as [`tiny_gemma4_causal_cfg`] but with sliding-window=8 to
/// exercise the strided alternating attention pattern. Used to
/// drive a separate Metal-aware test that's allowed to skip on
/// backends without a sliding-window SDPA kernel (Metal as of
/// this writing).
fn tiny_gemma4_sliding_cfg() -> GemmaConfig {
    let mut cfg = tiny_gemma4_causal_cfg();
    cfg.sliding_window = Some(8);
    cfg
}

// ── Deterministic synthetic weights with per-layer shape variation ─

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
        // Layer-norm names depend on Gemma version. Gemma 1 ships
        // `post_attention_layernorm`; Gemma 2/3/4 ship both
        // pre+post feedforward variants. Emit the right set so the
        // weight loader doesn't 404.
        if cfg.arch == GemmaArch::Gemma {
            t.insert(
                format!("{lp}.post_attention_layernorm.weight"),
                (ramp(h, 0.001, salt + 12), vec![h]),
            );
        } else {
            t.insert(
                format!("{lp}.pre_feedforward_layernorm.weight"),
                (ramp(h, 0.001, salt + 10), vec![h]),
            );
            t.insert(
                format!("{lp}.post_feedforward_layernorm.weight"),
                (ramp(h, 0.001, salt + 11), vec![h]),
            );
        }

        // Q / K / O — per-layer head_dim.
        t.insert(
            format!("{lp}.self_attn.q_proj.weight"),
            (ramp(q_dim * h, 0.01, salt + 2), vec![q_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.k_proj.weight"),
            (ramp(kv_dim * h, 0.01, salt + 3), vec![kv_dim, h]),
        );
        // attention_k_eq_v=true ⇒ V is aliased to K at runtime, so
        // the flow does NOT load v_proj.weight. We still publish it
        // for completeness; the WeightLoader leaves it untouched.
        t.insert(
            format!("{lp}.self_attn.v_proj.weight"),
            (ramp(kv_dim * h, 0.01, salt + 4), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.o_proj.weight"),
            (ramp(h * q_dim, 0.01, salt + 5), vec![h, q_dim]),
        );

        // MLP — uniform across layers.
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

// ── Run the prefill on `device` and return last-token logits ──────

/// Force Metal's scalar `sgemm` for the duration of this test binary
/// (bypasses `simdgroup_float8x8`'s reduced-precision tensor units).
/// Uses the public `RLX_METAL_PRECISE=1` knob so this test exercises
/// the documented path. Set once at process start via `Once` so
/// concurrent tests can't race the env var on/off.
fn ensure_metal_precise() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // SAFETY: this runs in the test process before any thread
        // has called into the Metal backend. After this point the
        // var stays set for the lifetime of the test binary, so
        // there is no concurrent read/write race.
        unsafe { std::env::set_var("RLX_METAL_PRECISE", "1") };
    });
}

fn run_prefill_on(device: Device) -> Result<Vec<f32>> {
    run_prefill_on_cfg(device, tiny_gemma4_causal_cfg())
}

fn run_prefill_on_cfg(device: Device, cfg: GemmaConfig) -> Result<Vec<f32>> {
    // Apple Silicon's `simdgroup_float8x8` matmul units use reduced
    // internal-accumulator precision (~fp16 class) — fine for
    // production but visible (~1e-1 absolute) on tiny parity-test
    // logits. The scalar `naive` sgemm variant restores full fp32
    // precision. Setting this is harmless for non-Metal devices
    // (they don't read it).
    let _ = device;
    ensure_metal_precise();
    let mut wm = synthetic_weights(&cfg);
    let opts = GemmaPrefillOpts {
        batch: 1,
        seq: SEQ,
        dynamic_seq: false,
        prefill_hidden: false,
        media_attn_bias: false,
        with_lm_head: true,
        with_kv_outputs: false,
        last_logits_only: true,
        profile: None,
    };
    let (hir, params) = build_gemma_prefill_flow(&cfg, &mut wm, &opts)?;
    let graph = rlx_core::flow_util::graph_from_hir(hir, params.clone())
        .map_err(|e| anyhow::anyhow!("hir → graph lower: {e}"))?
        .0;
    let mut compiled = compile_graph_gemma_prefill_with_params(device, graph, params)
        .map_err(|e| anyhow::anyhow!("compile on {device:?}: {e}"))?;
    let ids: Vec<f32> = (0..SEQ).map(|i| (i + 1) as f32).collect();
    let outs = compiled.run(&[("input_ids", ids.as_slice())]);
    Ok(outs.into_iter().next().expect("logits output"))
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

fn assert_parity_with_cpu_cfg(device: Device, cfg: GemmaConfig, tol: f32) {
    let cpu = run_prefill_on_cfg(Device::Cpu, cfg.clone()).expect("Gemma 4 LM prefill on CPU");
    let dev = run_prefill_on_cfg(device, cfg).expect("Gemma 4 LM prefill on device");
    assert!(
        cpu.iter().all(|v| v.is_finite()),
        "CPU produced non-finite logits"
    );
    assert!(
        dev.iter().all(|v| v.is_finite()),
        "{device:?} produced non-finite logits"
    );
    let d = max_abs_delta(&cpu, &dev);
    eprintln!("[gemma4 LM parity] {device:?} max|Δ|={d:.3e} (tol={tol:.1e})");
    assert!(
        d < tol,
        "Gemma 4 LM parity {device:?} vs CPU: max|Δ|={d:.3e} > tol={tol:.1e}",
    );
}

fn assert_causal_parity(device: Device, tol: f32) {
    assert_parity_with_cpu_cfg(device, tiny_gemma4_causal_cfg(), tol);
}

// ── Legacy Gemma 1 / 2 / 3 regression configs ────────────────────
//
// Gemma 1 / 2 / 3 don't use any of the per-layer accessors that
// Gemma 4 added (layer_types is empty, no global_head_dim, no
// attention_k_eq_v). These regression tests confirm the older
// variants still produce CPU-matching output after the Gemma 4
// flow changes.

fn tiny_gemma_legacy_base() -> GemmaConfig {
    GemmaConfig {
        arch: GemmaArch::Gemma,
        vocab_size: 32,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        tie_word_embeddings: true,
        attention_bias: false,
        head_dim: Some(8),
        attn_logit_softcapping: None,
        final_logit_softcapping: None,
        sliding_window: None,
        query_pre_attn_scalar: None,
        effective_num_layers: None,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        expert_weights_scale: 1.0,
        layer_types: Vec::new(),
        rope_parameters: GemmaRopeMap::default(),
        global_head_dim: None,
        num_global_key_value_heads: None,
        attention_k_eq_v: false,
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

fn tiny_gemma1_cfg() -> GemmaConfig {
    tiny_gemma_legacy_base()
}

fn tiny_gemma2_cfg() -> GemmaConfig {
    let mut cfg = tiny_gemma_legacy_base();
    cfg.arch = GemmaArch::Gemma2;
    cfg.attn_logit_softcapping = Some(50.0);
    cfg.final_logit_softcapping = Some(30.0);
    cfg
}

fn tiny_gemma3_cfg() -> GemmaConfig {
    let mut cfg = tiny_gemma_legacy_base();
    cfg.arch = GemmaArch::Gemma3;
    cfg.num_hidden_layers = 6; // stride-6 pattern needs ≥6 layers
    cfg.final_logit_softcapping = Some(30.0);
    cfg.sliding_window = None; // causal-only — bypasses Metal sliding kernel for clean baseline
    cfg
}

fn tiny_gemma3_sliding_cfg() -> GemmaConfig {
    let mut cfg = tiny_gemma3_cfg();
    cfg.sliding_window = Some(8); // exercises strided alternating mask
    cfg
}

fn assert_sliding_parity(device: Device, tol: f32) {
    assert_parity_with_cpu_cfg(device, tiny_gemma4_sliding_cfg(), tol);
}

// ── CPU baseline (sanity: the graph actually compiles + runs) ─────
//
// Both the causal-only and sliding-window configs must run on CPU.
// The sliding case explicitly exercises stride-6 alternating
// attention dispatch.

#[test]
fn gemma4_lm_prefill_runs_on_cpu_causal() {
    let logits =
        run_prefill_on_cfg(Device::Cpu, tiny_gemma4_causal_cfg()).expect("causal LM on CPU");
    assert_eq!(logits.len(), tiny_gemma4_causal_cfg().vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn gemma4_lm_prefill_runs_on_cpu_sliding() {
    let logits = run_prefill_on_cfg(Device::Cpu, tiny_gemma4_sliding_cfg())
        .expect("sliding-window LM on CPU");
    assert_eq!(logits.len(), tiny_gemma4_sliding_cfg().vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

// ── Cross-backend parity (causal — every accelerator) ────────────

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn gemma4_lm_prefill_matches_cpu_on_metal() {
    if !is_available(Device::Metal) {
        eprintln!("[gemma4 LM parity] Metal unavailable — skip");
        return;
    }
    // 1e-3 — Metal with the precise scalar sgemm path (forced via
    // env in run_prefill_on_cfg). Drops from 1.4e-1 with the
    // default simdgroup_float8x8 tensor units.
    assert_causal_parity(Device::Metal, 1e-3);
}

// ── Metal feature-isolation probes ───────────────────────────────
//
// When the full causal config fails on Metal but passes on
// MLX/wgpu, these probes narrow which Gemma-4-specific feature is
// to blame. They build progressively-richer configs and assert
// parity after each addition.

#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_probe_cfg_baseline() -> GemmaConfig {
    // Plain Gemma-3-like (causal everywhere, no k_eq_v, no split
    // rope, uniform head_dim). Establishes the Metal CPU-vs-Metal
    // floor without any Gemma-4 additions.
    let mut cfg = tiny_gemma4_causal_cfg();
    cfg.arch = GemmaArch::Gemma3;
    cfg.attention_k_eq_v = false;
    cfg.global_head_dim = None;
    cfg.num_global_key_value_heads = None;
    cfg.rope_parameters = GemmaRopeMap::default();
    cfg.layer_types = Vec::new();
    cfg
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_probe_baseline_gemma3() {
    if !is_available(Device::Metal) {
        return;
    }
    assert_parity_with_cpu_cfg(Device::Metal, metal_probe_cfg_baseline(), 1e-3);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_probe_plus_k_eq_v() {
    if !is_available(Device::Metal) {
        return;
    }
    let mut cfg = metal_probe_cfg_baseline();
    cfg.attention_k_eq_v = true;
    assert_parity_with_cpu_cfg(Device::Metal, cfg, 1e-3);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_probe_plus_split_rope() {
    if !is_available(Device::Metal) {
        return;
    }
    let mut cfg = metal_probe_cfg_baseline();
    cfg.arch = GemmaArch::Gemma4;
    cfg.layer_types = (0..cfg.num_hidden_layers)
        .map(|i| {
            if (i + 1) % 6 == 0 {
                GemmaLayerType::FullAttention
            } else {
                GemmaLayerType::SlidingAttention
            }
        })
        .collect();
    cfg.rope_parameters = GemmaRopeMap {
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
    };
    assert_parity_with_cpu_cfg(Device::Metal, cfg, 1e-3);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_probe_plus_per_layer_kv_shape() {
    if !is_available(Device::Metal) {
        return;
    }
    let mut cfg = metal_probe_cfg_baseline();
    cfg.arch = GemmaArch::Gemma4;
    cfg.layer_types = (0..cfg.num_hidden_layers)
        .map(|i| {
            if (i + 1) % 6 == 0 {
                GemmaLayerType::FullAttention
            } else {
                GemmaLayerType::SlidingAttention
            }
        })
        .collect();
    cfg.global_head_dim = Some(16);
    cfg.num_global_key_value_heads = Some(1);
    assert_parity_with_cpu_cfg(Device::Metal, cfg, 1e-3);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn gemma4_lm_prefill_matches_cpu_on_mlx() {
    if !is_available(Device::Mlx) {
        eprintln!("[gemma4 LM parity] MLX unavailable — skip");
        return;
    }
    // 1e-5 — MLX is essentially fp32-bit-exact with CPU on this
    // graph (measured ~8e-8).
    assert_causal_parity(Device::Mlx, 1e-5);
}

#[cfg(feature = "cuda")]
#[test]
fn gemma4_lm_prefill_matches_cpu_on_cuda() {
    if !is_available(Device::Cuda) {
        eprintln!("[gemma4 LM parity] CUDA unavailable — skip");
        return;
    }
    assert_causal_parity(Device::Cuda, 1e-2);
}

#[cfg(feature = "rocm")]
#[test]
fn gemma4_lm_prefill_matches_cpu_on_rocm() {
    if !is_available(Device::Rocm) {
        eprintln!("[gemma4 LM parity] ROCm unavailable — skip");
        return;
    }
    assert_causal_parity(Device::Rocm, 1e-2);
}

#[cfg(feature = "gpu")]
#[test]
fn gemma4_lm_prefill_matches_cpu_on_wgpu() {
    if !is_available(Device::Gpu) {
        eprintln!("[gemma4 LM parity] wgpu unavailable — skip");
        return;
    }
    // 1e-3 — wgpu measured at ~1e-6 on this graph, give some
    // headroom for driver/shader variance across hosts.
    assert_causal_parity(Device::Gpu, 1e-3);
}

#[cfg(feature = "vulkan")]
#[test]
fn gemma4_lm_prefill_matches_cpu_on_vulkan() {
    if !is_available(Device::Vulkan) {
        eprintln!("[gemma4 LM parity] Vulkan unavailable — skip");
        return;
    }
    assert_causal_parity(Device::Vulkan, 1e-2);
}

// ── Sliding-window parity (Gemma 3/4 strided alternation) ────────
//
// Metal SDPA does not yet lower `MaskKind::SlidingWindow` — that's
// an rlx-metal kernel gap, not a Gemma-4 issue. These tests assert
// parity on every other backend.

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn gemma4_lm_sliding_matches_cpu_on_mlx() {
    if !is_available(Device::Mlx) {
        return;
    }
    assert_sliding_parity(Device::Mlx, 1e-5);
}

#[cfg(feature = "gpu")]
#[test]
fn gemma4_lm_sliding_matches_cpu_on_wgpu() {
    if !is_available(Device::Gpu) {
        return;
    }
    assert_sliding_parity(Device::Gpu, 1e-3);
}

#[cfg(feature = "cuda")]
#[test]
fn gemma4_lm_sliding_matches_cpu_on_cuda() {
    if !is_available(Device::Cuda) {
        return;
    }
    assert_sliding_parity(Device::Cuda, 1e-2);
}

#[cfg(feature = "rocm")]
#[test]
fn gemma4_lm_sliding_matches_cpu_on_rocm() {
    if !is_available(Device::Rocm) {
        return;
    }
    assert_sliding_parity(Device::Rocm, 1e-2);
}

#[cfg(feature = "vulkan")]
#[test]
fn gemma4_lm_sliding_matches_cpu_on_vulkan() {
    if !is_available(Device::Vulkan) {
        return;
    }
    assert_sliding_parity(Device::Vulkan, 1e-2);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn gemma4_lm_sliding_matches_cpu_on_metal() {
    if !is_available(Device::Metal) {
        return;
    }
    assert_sliding_parity(Device::Metal, 1e-3);
}

// ── Legacy Gemma regression — CPU baselines + cross-backend parity

#[test]
fn legacy_gemma1_prefill_runs_on_cpu() {
    let logits = run_prefill_on_cfg(Device::Cpu, tiny_gemma1_cfg()).expect("Gemma 1 CPU");
    assert_eq!(logits.len(), tiny_gemma1_cfg().vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn legacy_gemma2_prefill_runs_on_cpu() {
    let logits = run_prefill_on_cfg(Device::Cpu, tiny_gemma2_cfg()).expect("Gemma 2 CPU");
    assert_eq!(logits.len(), tiny_gemma2_cfg().vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn legacy_gemma3_prefill_runs_on_cpu() {
    let logits = run_prefill_on_cfg(Device::Cpu, tiny_gemma3_cfg()).expect("Gemma 3 CPU");
    assert_eq!(logits.len(), tiny_gemma3_cfg().vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn legacy_gemma3_sliding_prefill_runs_on_cpu() {
    let logits =
        run_prefill_on_cfg(Device::Cpu, tiny_gemma3_sliding_cfg()).expect("Gemma 3 sliding CPU");
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn legacy_gemma1_matches_cpu_on_metal() {
    if !is_available(Device::Metal) {
        return;
    }
    assert_parity_with_cpu_cfg(Device::Metal, tiny_gemma1_cfg(), 1e-3);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn legacy_gemma2_matches_cpu_on_metal() {
    if !is_available(Device::Metal) {
        return;
    }
    assert_parity_with_cpu_cfg(Device::Metal, tiny_gemma2_cfg(), 1e-3);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn legacy_gemma3_matches_cpu_on_metal() {
    if !is_available(Device::Metal) {
        return;
    }
    assert_parity_with_cpu_cfg(Device::Metal, tiny_gemma3_cfg(), 1e-3);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn legacy_gemma3_sliding_matches_cpu_on_metal() {
    if !is_available(Device::Metal) {
        return;
    }
    assert_parity_with_cpu_cfg(Device::Metal, tiny_gemma3_sliding_cfg(), 1e-3);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn legacy_gemma1_matches_cpu_on_mlx() {
    if !is_available(Device::Mlx) {
        return;
    }
    assert_parity_with_cpu_cfg(Device::Mlx, tiny_gemma1_cfg(), 1e-5);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn legacy_gemma2_matches_cpu_on_mlx() {
    if !is_available(Device::Mlx) {
        return;
    }
    assert_parity_with_cpu_cfg(Device::Mlx, tiny_gemma2_cfg(), 1e-5);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn legacy_gemma3_matches_cpu_on_mlx() {
    if !is_available(Device::Mlx) {
        return;
    }
    assert_parity_with_cpu_cfg(Device::Mlx, tiny_gemma3_cfg(), 1e-5);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn legacy_gemma3_sliding_matches_cpu_on_mlx() {
    if !is_available(Device::Mlx) {
        return;
    }
    assert_parity_with_cpu_cfg(Device::Mlx, tiny_gemma3_sliding_cfg(), 1e-5);
}

#[cfg(feature = "gpu")]
#[test]
fn legacy_gemma1_matches_cpu_on_wgpu() {
    if !is_available(Device::Gpu) {
        return;
    }
    assert_parity_with_cpu_cfg(Device::Gpu, tiny_gemma1_cfg(), 1e-3);
}

#[cfg(feature = "gpu")]
#[test]
fn legacy_gemma2_matches_cpu_on_wgpu() {
    if !is_available(Device::Gpu) {
        return;
    }
    assert_parity_with_cpu_cfg(Device::Gpu, tiny_gemma2_cfg(), 1e-3);
}

#[cfg(feature = "gpu")]
#[test]
fn legacy_gemma3_matches_cpu_on_wgpu() {
    if !is_available(Device::Gpu) {
        return;
    }
    assert_parity_with_cpu_cfg(Device::Gpu, tiny_gemma3_cfg(), 1e-3);
}

#[cfg(feature = "gpu")]
#[test]
fn legacy_gemma3_sliding_matches_cpu_on_wgpu() {
    if !is_available(Device::Gpu) {
        return;
    }
    assert_parity_with_cpu_cfg(Device::Gpu, tiny_gemma3_sliding_cfg(), 1e-3);
}
