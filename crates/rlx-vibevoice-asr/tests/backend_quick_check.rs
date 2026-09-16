// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//!
//! Backend quick-check for VibeVoice-ASR: ConvNeXt VAE encode (BitNet ReLU +
//! Streaming GELU) on every standard RLX device. Unavailable backends skip.
//!
//! ```bash
//! cargo test -p rlx-vibevoice-asr --test backend_quick_check --features all-backends --release
//! just features=all-backends test-vibevoice-asr-backends
//! ```

use rlx_runtime::{Device, is_available};
use rlx_vibevoice_asr::config::VaeFfnAct;
use rlx_vibevoice_asr::vae::VaeEncoderGraph;
use rlx_vibevoice_asr::weights::{BlockW, ConnectorW, ConvW, VaeEncoderWeights};

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.1
        })
        .collect()
}

fn conv(c_out: usize, c_in: usize, k: usize, seed: u64) -> ConvW {
    ConvW {
        weight: fill(c_out * c_in * k, seed),
        bias: fill(c_out, seed + 1),
        c_out,
        c_in,
        k,
    }
}

fn block(dim: usize, seed: u64) -> BlockW {
    let inter = 4 * dim;
    BlockW {
        norm_w: fill(dim, seed),
        mixer: ConvW {
            weight: fill(dim * 3, seed + 1),
            bias: fill(dim, seed + 2),
            c_out: dim,
            c_in: 1,
            k: 3,
        },
        gamma: fill(dim, seed + 3),
        ffn_norm_w: fill(dim, seed + 4),
        l1_w: fill(inter * dim, seed + 5),
        l1_b: fill(inter, seed + 6),
        l2_w: fill(dim * inter, seed + 7),
        l2_b: fill(dim, seed + 8),
        ffn_gamma: fill(dim, seed + 9),
        dim,
    }
}

/// Tiny 2-stage encoder (strides 1, 2) — same shape as `vae_encoder_smoke`.
fn tiny_weights() -> VaeEncoderWeights {
    let (c0, c1, vae_dim, connector_dim) = (4usize, 8usize, 6usize, 10usize);
    VaeEncoderWeights {
        downsamples: vec![conv(c0, 1, 3, 1), conv(c1, c0, 3, 10)],
        stages: vec![vec![block(c0, 100)], vec![block(c1, 200)]],
        head: conv(vae_dim, c1, 3, 30),
        connector: ConnectorW {
            fc1_w: fill(connector_dim * vae_dim, 40),
            fc1_b: fill(connector_dim, 41),
            norm_w: fill(connector_dim, 42),
            fc2_w: fill(connector_dim * connector_dim, 43),
            fc2_b: fill(connector_dim, 44),
            in_dim: vae_dim,
            out_dim: connector_dim,
        },
        vae_dim,
        connector_dim,
    }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn run_vae(device: Device, act: VaeFfnAct, normalize: bool) -> Vec<f32> {
    let w = tiny_weights();
    let padded_len = 16usize;
    let mut g = VaeEncoderGraph::compile_for_opts(device, &w, padded_len, act, normalize)
        .unwrap_or_else(|e| panic!("compile VAE on {device:?} act={act:?}: {e:#}"));
    let audio = fill(padded_len, 7);
    let feats = g
        .run(&audio)
        .unwrap_or_else(|e| panic!("run VAE on {device:?} act={act:?}: {e:#}"));
    assert_eq!(
        feats.len(),
        g.n_frames * g.connector_dim,
        "{device:?} feat len"
    );
    assert!(
        feats.iter().all(|v| v.is_finite()),
        "{device:?} non-finite features"
    );
    feats
}

fn run_if_available(device: Device) {
    if device != Device::Cpu && !is_available(device) {
        eprintln!("skip vibevoice-asr on {device:?}: backend not available");
        return;
    }
    eprintln!("vibevoice-asr backend {device:?} …");

    // BitNet path: ReLU + whole-clip latent normalize.
    let bitnet = run_vae(device, VaeFfnAct::Relu, true);
    // Streaming path: GELU, no whole-clip normalize.
    let streaming = run_vae(device, VaeFfnAct::Gelu, false);

    if device != Device::Cpu && is_available(Device::Cpu) {
        let cpu_b = run_vae(Device::Cpu, VaeFfnAct::Relu, true);
        let cpu_s = run_vae(Device::Cpu, VaeFfnAct::Gelu, false);
        let eb = max_abs(&bitnet, &cpu_b);
        let es = max_abs(&streaming, &cpu_s);
        eprintln!("  BitNet ReLU  vs CPU max|Δ| = {eb:.3e}");
        eprintln!("  Stream GELU  vs CPU max|Δ| = {es:.3e}");
        // Audio graphs should match tightly across backends; leave headroom for
        // GPU/ANE reduction order.
        assert!(
            eb < 2e-3,
            "{device:?} BitNet vs CPU max|Δ|={eb} (want < 2e-3)"
        );
        assert!(
            es < 2e-3,
            "{device:?} Streaming vs CPU max|Δ|={es} (want < 2e-3)"
        );
    } else {
        eprintln!("  BitNet ReLU  ok ({} feats)", bitnet.len());
        eprintln!("  Stream GELU  ok ({} feats)", streaming.len());
    }
}

#[test]
fn vibevoice_asr_on_cpu() {
    run_if_available(Device::Cpu);
}

#[cfg(feature = "metal")]
#[test]
fn vibevoice_asr_on_metal() {
    run_if_available(Device::Metal);
}

#[cfg(feature = "mlx")]
#[test]
fn vibevoice_asr_on_mlx() {
    run_if_available(Device::Mlx);
}

#[cfg(feature = "cuda")]
#[test]
fn vibevoice_asr_on_cuda() {
    run_if_available(Device::Cuda);
}

#[cfg(feature = "rocm")]
#[test]
fn vibevoice_asr_on_rocm() {
    run_if_available(Device::Rocm);
}

#[cfg(feature = "gpu")]
#[test]
fn vibevoice_asr_on_wgpu() {
    run_if_available(Device::Gpu);
}

#[cfg(feature = "vulkan")]
#[test]
fn vibevoice_asr_on_vulkan() {
    run_if_available(Device::Vulkan);
}

#[cfg(feature = "coreml")]
#[test]
fn vibevoice_asr_on_coreml() {
    run_if_available(Device::Ane);
}

/// Matrix entry: every compiled-in backend that reports available must pass.
#[test]
fn vibevoice_asr_all_available_backends() {
    #[allow(unused_mut)] // pushed under backend feature cfgs
    let mut devices = vec![Device::Cpu];
    #[cfg(feature = "metal")]
    devices.push(Device::Metal);
    #[cfg(feature = "mlx")]
    devices.push(Device::Mlx);
    #[cfg(feature = "cuda")]
    devices.push(Device::Cuda);
    #[cfg(feature = "rocm")]
    devices.push(Device::Rocm);
    #[cfg(feature = "gpu")]
    devices.push(Device::Gpu);
    #[cfg(feature = "vulkan")]
    devices.push(Device::Vulkan);
    #[cfg(feature = "coreml")]
    devices.push(Device::Ane);

    let mut ran = 0usize;
    for d in devices {
        if d != Device::Cpu && !is_available(d) {
            eprintln!("skip matrix {d:?}");
            continue;
        }
        run_if_available(d);
        ran += 1;
    }
    assert!(ran >= 1, "expected at least CPU");
    eprintln!("vibevoice-asr backend matrix: {ran} device(s) passed");
}
