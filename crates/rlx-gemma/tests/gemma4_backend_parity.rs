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

//! Cross-backend parity for the Gemma 4 12B-shape LM graph + multimodal
//! projector graphs. Each `#[test]` is gated on a Cargo feature so the
//! test only runs when that backend's crate is compiled in. The
//! runtime probe (`is_available`) additionally skips when the host
//! lacks the matching live driver (e.g. macOS CI without an MLX-able
//! GPU).
//!
//! ```bash
//! # All Apple backends + CPU on macOS:
//! cargo test -p rlx-gemma --test gemma4_backend_parity --features apple-silicon
//!
//! # CPU-only baseline (always runs):
//! cargo test -p rlx-gemma --test gemma4_backend_parity
//! ```
//!
//! Tolerance: every backend should agree with CPU to within
//! `max|Δ| < 1e-3` for the synthesized tiny weights used here (FP32
//! throughout; the gap comes from associativity of reductions, not
//! quantization).

use anyhow::Result;
use rlx_gemma::multimodal::{
    GemmaAudioConfig, GemmaVisionConfig, build_audio_projection_graph,
    build_vision_projection_graph,
};
use rlx_gemma::unified_projector::{build_unified_audio_graph, build_unified_vision_graph};
use rlx_runtime::{Device, Session};

// ── Tiny projector configs ───────────────────────────────────────

fn tiny_vision_cfg() -> GemmaVisionConfig {
    GemmaVisionConfig {
        patch_size: 2,
        model_patch_size: 4,
        mm_embed_dim: 8,
        mm_posemb_size: 16,
        num_soft_tokens: 4,
        output_proj_dims: 8,
        pooling_kernel_size: 1,
        rms_norm_eps: 1e-6,
    }
}

fn tiny_audio_cfg() -> GemmaAudioConfig {
    GemmaAudioConfig {
        hidden_size: 4,
        audio_embed_dim: 4,
        audio_samples_per_token: 8,
        output_proj_dims: 4,
        rms_norm_eps: 1e-6,
    }
}

/// Unified 12B-shaped tiny config (48×48 patches, d=3840 scaled down).
fn tiny_unified_vision_cfg() -> GemmaVisionConfig {
    GemmaVisionConfig {
        patch_size: 16,
        model_patch_size: 48,
        mm_embed_dim: 8,
        mm_posemb_size: 16,
        num_soft_tokens: 4,
        output_proj_dims: 8,
        pooling_kernel_size: 3,
        rms_norm_eps: 1e-6,
    }
}

fn tiny_unified_audio_cfg() -> GemmaAudioConfig {
    GemmaAudioConfig {
        hidden_size: 8,
        audio_embed_dim: 8,
        audio_samples_per_token: 8,
        output_proj_dims: 8,
        rms_norm_eps: 1e-6,
    }
}

// ── Reproducible deterministic weights ────────────────────────────

fn det_buf(seed: u32, len: usize) -> Vec<f32> {
    // tiny LCG so tests are byte-stable but not all-zeros (which would
    // make the comparison trivially pass through zero matmuls).
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        let v = ((s >> 16) & 0x7fff) as f32 / 32_767.0 * 0.1 - 0.05;
        out.push(v);
    }
    out
}

// ── Vision projector parity ───────────────────────────────────────

fn run_vision_projector_on(device: Device, num_patches: usize) -> Result<Vec<f32>> {
    let cfg = tiny_vision_cfg();
    let g = build_vision_projection_graph(1, num_patches, &cfg)?;
    let session = Session::new(device);
    let mut compiled = session
        .compile_hir(g.hir)
        .map_err(|e| anyhow::anyhow!("compile vision projector on {device:?}: {e:?}"))?;
    let patch_feats = cfg.patch_size * cfg.patch_size * 3;
    compiled.set_param(
        "vision_tower.embed.weight",
        &det_buf(1, patch_feats * cfg.mm_embed_dim),
    );
    compiled.set_param(
        "vision_tower.pos_embed.weight",
        &det_buf(2, num_patches * cfg.mm_embed_dim),
    );
    compiled.set_param("vision_tower.norm.weight", &det_buf(3, cfg.mm_embed_dim));
    compiled.set_param("vision_tower.ones", &vec![1.0f32; cfg.mm_embed_dim]);
    compiled.set_param("vision_tower.zero_beta", &vec![0.0f32; cfg.mm_embed_dim]);
    compiled.set_param(
        "vision_tower.soft_token.weight",
        // Patch-axis reducer: [num_patches, num_soft_tokens].
        &det_buf(4, num_patches * cfg.num_soft_tokens),
    );
    compiled.set_param(
        "vision_tower.lm_proj.weight",
        // Feature-axis projection: [mm_embed_dim, output_proj_dims].
        &det_buf(5, cfg.mm_embed_dim * cfg.output_proj_dims),
    );
    let patches = det_buf(42, num_patches * patch_feats);
    let outs = compiled.run(&[("patches", patches.as_slice())]);
    Ok(outs.into_iter().next().expect("vision projector output"))
}

fn run_audio_projector_on(device: Device, num_frames: usize) -> Result<Vec<f32>> {
    let cfg = tiny_audio_cfg();
    let lm_hidden = 8;
    let g = build_audio_projection_graph(1, num_frames, &cfg, lm_hidden)?;
    let session = Session::new(device);
    let mut compiled = session
        .compile_hir(g.hir)
        .map_err(|e| anyhow::anyhow!("compile audio projector on {device:?}: {e:?}"))?;
    compiled.set_param(
        "audio_tower.embed.weight",
        &det_buf(11, cfg.audio_samples_per_token * cfg.audio_embed_dim),
    );
    compiled.set_param("audio_tower.norm.weight", &det_buf(12, cfg.audio_embed_dim));
    compiled.set_param("audio_tower.ones", &vec![1.0f32; cfg.audio_embed_dim]);
    compiled.set_param("audio_tower.zero_beta", &vec![0.0f32; cfg.audio_embed_dim]);
    compiled.set_param(
        "audio_tower.lm_proj.weight",
        &det_buf(13, cfg.audio_embed_dim * lm_hidden),
    );
    let frames = det_buf(43, num_frames * cfg.audio_samples_per_token);
    let outs = compiled.run(&[("frames", frames.as_slice())]);
    Ok(outs.into_iter().next().expect("audio projector output"))
}

fn run_unified_vision_projector_on(device: Device, num_slots: usize) -> Result<Vec<f32>> {
    let cfg = tiny_unified_vision_cfg();
    let g = build_unified_vision_graph(num_slots, &cfg)?;
    let session = Session::new(device);
    let mut compiled = session
        .compile_hir(g.hir)
        .map_err(|e| anyhow::anyhow!("compile unified vision on {device:?}: {e:?}"))?;
    let patch_dim = cfg.model_patch_size * cfg.model_patch_size * 3;
    let d = cfg.mm_embed_dim;
    compiled.set_param(
        "model.vision_embedder.patch_ln1.weight",
        &det_buf(21, patch_dim),
    );
    compiled.set_param(
        "model.vision_embedder.patch_ln1.bias",
        &det_buf(22, patch_dim),
    );
    compiled.set_param(
        "model.vision_embedder.patch_dense.weight",
        &det_buf(23, patch_dim * d),
    );
    compiled.set_param("model.vision_embedder.patch_dense.bias", &det_buf(24, d));
    compiled.set_param("model.vision_embedder.patch_ln2.weight", &det_buf(25, d));
    compiled.set_param("model.vision_embedder.patch_ln2.bias", &det_buf(26, d));
    compiled.set_param("model.vision_embedder.pos_norm.weight", &det_buf(27, d));
    compiled.set_param("model.vision_embedder.pos_norm.bias", &det_buf(28, d));
    compiled.set_param(
        "model.embed_vision.embedding_projection.weight",
        &det_buf(29, d * d),
    );
    compiled.set_param("unified.ones", &vec![1.0f32; d]);
    compiled.set_param("unified.zero_beta", &vec![0.0f32; d]);
    let patches = det_buf(44, num_slots * patch_dim);
    let pos_bias = det_buf(45, num_slots * d);
    let outs = compiled.run(&[
        ("patches", patches.as_slice()),
        ("pos_bias", pos_bias.as_slice()),
    ]);
    Ok(outs.into_iter().next().expect("unified vision output"))
}

fn run_unified_audio_projector_on(device: Device, num_frames: usize) -> Result<Vec<f32>> {
    let cfg = tiny_unified_audio_cfg();
    let lm_hidden = 8;
    let g = build_unified_audio_graph(num_frames, &cfg, lm_hidden)?;
    let session = Session::new(device);
    let mut compiled = session
        .compile_hir(g.hir)
        .map_err(|e| anyhow::anyhow!("compile unified audio on {device:?}: {e:?}"))?;
    let d = cfg.audio_embed_dim;
    compiled.set_param(
        "model.embed_audio.embedding_projection.weight",
        &det_buf(31, d * lm_hidden),
    );
    compiled.set_param("unified.audio.ones", &vec![1.0f32; d]);
    compiled.set_param("unified.audio.zero_beta", &vec![0.0f32; d]);
    let frames = det_buf(46, num_frames * cfg.audio_samples_per_token);
    let outs = compiled.run(&[("frames", frames.as_slice())]);
    Ok(outs.into_iter().next().expect("unified audio output"))
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

fn assert_parity_with_cpu_vision(device: Device, tol: f32) {
    let num_patches = 4;
    let cpu = run_vision_projector_on(Device::Cpu, num_patches).expect("vision on CPU");
    let dev = run_vision_projector_on(device, num_patches).expect("vision on device");
    let d = max_abs_delta(&cpu, &dev);
    assert!(
        d < tol,
        "vision projector parity {device:?} vs CPU: max|Δ|={d:.3e} > tol={tol:.1e}",
    );
}

fn assert_parity_with_cpu_audio(device: Device, tol: f32) {
    let num_frames = 3;
    let cpu = run_audio_projector_on(Device::Cpu, num_frames).expect("audio on CPU");
    let dev = run_audio_projector_on(device, num_frames).expect("audio on device");
    let d = max_abs_delta(&cpu, &dev);
    assert!(
        d < tol,
        "audio projector parity {device:?} vs CPU: max|Δ|={d:.3e} > tol={tol:.1e}",
    );
}

fn assert_parity_with_cpu_unified_vision(device: Device, tol: f32) {
    let num_slots = 4;
    let cpu = run_unified_vision_projector_on(Device::Cpu, num_slots).expect("unified vision CPU");
    let dev = run_unified_vision_projector_on(device, num_slots).expect("unified vision device");
    let d = max_abs_delta(&cpu, &dev);
    eprintln!("[gemma4 unified parity] vision {device:?} max|Δ|={d:.3e}");
    assert!(
        d < tol,
        "unified vision projector parity {device:?} vs CPU: max|Δ|={d:.3e} > tol={tol:.1e}",
    );
}

fn assert_parity_with_cpu_unified_audio(device: Device, tol: f32) {
    let num_frames = 3;
    let cpu = run_unified_audio_projector_on(Device::Cpu, num_frames).expect("unified audio CPU");
    let dev = run_unified_audio_projector_on(device, num_frames).expect("unified audio device");
    let d = max_abs_delta(&cpu, &dev);
    eprintln!("[gemma4 unified parity] audio {device:?} max|Δ|={d:.3e}");
    assert!(
        d < tol,
        "unified audio projector parity {device:?} vs CPU: max|Δ|={d:.3e} > tol={tol:.1e}",
    );
}

// ── CPU baseline ─────────────────────────────────────────────────

#[test]
fn unified_vision_projector_baseline_runs_on_cpu() {
    let _ = run_unified_vision_projector_on(Device::Cpu, 4).expect("unified vision on CPU");
}

#[test]
fn unified_audio_projector_baseline_runs_on_cpu() {
    let _ = run_unified_audio_projector_on(Device::Cpu, 3).expect("unified audio on CPU");
}

// ── CPU baseline (legacy pool) ─────────────────────────────────

#[test]
fn vision_projector_baseline_runs_on_cpu() {
    let _ = run_vision_projector_on(Device::Cpu, 4).expect("vision projector on CPU");
}

#[test]
fn audio_projector_baseline_runs_on_cpu() {
    let _ = run_audio_projector_on(Device::Cpu, 3).expect("audio projector on CPU");
}

// ── Metal ────────────────────────────────────────────────────────

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_vision_projector_matches_cpu() {
    if !is_available(Device::Metal) {
        eprintln!("[gemma4 parity] Metal unavailable at runtime — skip");
        return;
    }
    assert_parity_with_cpu_vision(Device::Metal, 1e-3);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_audio_projector_matches_cpu() {
    if !is_available(Device::Metal) {
        eprintln!("[gemma4 parity] Metal unavailable at runtime — skip");
        return;
    }
    assert_parity_with_cpu_audio(Device::Metal, 1e-3);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_unified_vision_projector_matches_cpu() {
    if !is_available(Device::Metal) {
        return;
    }
    assert_parity_with_cpu_unified_vision(Device::Metal, 1e-3);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_unified_audio_projector_matches_cpu() {
    if !is_available(Device::Metal) {
        return;
    }
    assert_parity_with_cpu_unified_audio(Device::Metal, 1e-3);
}

// ── MLX ──────────────────────────────────────────────────────────

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn mlx_vision_projector_matches_cpu() {
    if !is_available(Device::Mlx) {
        eprintln!("[gemma4 parity] MLX unavailable at runtime — skip");
        return;
    }
    assert_parity_with_cpu_vision(Device::Mlx, 1e-3);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn mlx_audio_projector_matches_cpu() {
    if !is_available(Device::Mlx) {
        eprintln!("[gemma4 parity] MLX unavailable at runtime — skip");
        return;
    }
    assert_parity_with_cpu_audio(Device::Mlx, 1e-3);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn mlx_unified_vision_projector_matches_cpu() {
    if !is_available(Device::Mlx) {
        return;
    }
    assert_parity_with_cpu_unified_vision(Device::Mlx, 1e-3);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn mlx_unified_audio_projector_matches_cpu() {
    if !is_available(Device::Mlx) {
        return;
    }
    assert_parity_with_cpu_unified_audio(Device::Mlx, 1e-3);
}

// ── CUDA ─────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
#[test]
fn cuda_vision_projector_matches_cpu() {
    if !is_available(Device::Cuda) {
        eprintln!("[gemma4 parity] CUDA unavailable at runtime — skip");
        return;
    }
    assert_parity_with_cpu_vision(Device::Cuda, 1e-3);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_audio_projector_matches_cpu() {
    if !is_available(Device::Cuda) {
        eprintln!("[gemma4 parity] CUDA unavailable at runtime — skip");
        return;
    }
    assert_parity_with_cpu_audio(Device::Cuda, 1e-3);
}

// ── ROCm ─────────────────────────────────────────────────────────

#[cfg(feature = "rocm")]
#[test]
fn rocm_vision_projector_matches_cpu() {
    if !is_available(Device::Rocm) {
        eprintln!("[gemma4 parity] ROCm unavailable at runtime — skip");
        return;
    }
    assert_parity_with_cpu_vision(Device::Rocm, 1e-3);
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_audio_projector_matches_cpu() {
    if !is_available(Device::Rocm) {
        eprintln!("[gemma4 parity] ROCm unavailable at runtime — skip");
        return;
    }
    assert_parity_with_cpu_audio(Device::Rocm, 1e-3);
}

// ── wgpu (portable GPU) ──────────────────────────────────────────

#[cfg(feature = "gpu")]
#[test]
fn wgpu_vision_projector_matches_cpu() {
    if !is_available(Device::Gpu) {
        eprintln!("[gemma4 parity] wgpu unavailable at runtime — skip");
        return;
    }
    assert_parity_with_cpu_vision(Device::Gpu, 1e-3);
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_audio_projector_matches_cpu() {
    if !is_available(Device::Gpu) {
        eprintln!("[gemma4 parity] wgpu unavailable at runtime — skip");
        return;
    }
    assert_parity_with_cpu_audio(Device::Gpu, 1e-3);
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_unified_vision_projector_matches_cpu() {
    if !is_available(Device::Gpu) {
        return;
    }
    assert_parity_with_cpu_unified_vision(Device::Gpu, 1e-3);
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_unified_audio_projector_matches_cpu() {
    if !is_available(Device::Gpu) {
        return;
    }
    assert_parity_with_cpu_unified_audio(Device::Gpu, 1e-3);
}

// ── Vulkan ───────────────────────────────────────────────────────

#[cfg(feature = "vulkan")]
#[test]
fn vulkan_vision_projector_matches_cpu() {
    if !is_available(Device::Vulkan) {
        eprintln!("[gemma4 parity] Vulkan unavailable at runtime — skip");
        return;
    }
    assert_parity_with_cpu_vision(Device::Vulkan, 1e-3);
}

#[cfg(feature = "vulkan")]
#[test]
fn vulkan_audio_projector_matches_cpu() {
    if !is_available(Device::Vulkan) {
        eprintln!("[gemma4 parity] Vulkan unavailable at runtime — skip");
        return;
    }
    assert_parity_with_cpu_audio(Device::Vulkan, 1e-3);
}
