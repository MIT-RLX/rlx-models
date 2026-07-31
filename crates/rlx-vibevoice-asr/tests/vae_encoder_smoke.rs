// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
//! Smoke test: the ConvNeXt VAE encoder graph (strided causal convs + ConvNeXt
//! blocks + head conv + SpeechConnector) with a tiny synthetic weight set,
//! compiled + run on the `RLX_TEST_DEVICE` backend; finite features out.
//! Set `RLX_TEST_DEVICE=metal|mlx|gpu|coreml|cuda|vulkan` (default CPU) and build
//! the matching cargo feature to exercise a backend.

use rlx_runtime::Device;
use rlx_vibevoice_asr::vae::VaeEncoderGraph;
use rlx_vibevoice_asr::weights::{BlockW, ConnectorW, ConvW, VaeEncoderWeights};

fn dev() -> Device {
    match std::env::var("RLX_TEST_DEVICE").ok().as_deref() {
        Some("metal") | Some("mtl") => Device::Metal,
        Some("mlx") => Device::Mlx,
        Some("gpu") | Some("wgpu") => Device::Gpu,
        Some("coreml") | Some("ane") => Device::Ane,
        Some("cuda") => Device::Cuda,
        Some("vulkan") | Some("vk") => Device::Vulkan,
        _ => Device::Cpu,
    }
}

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

/// Dense conv weight `[c_out, c_in, k]` + bias `[c_out]`.
fn conv(c_out: usize, c_in: usize, k: usize, seed: u64) -> ConvW {
    ConvW {
        weight: fill(c_out * c_in * k, seed),
        bias: fill(c_out, seed + 1),
        c_out,
        c_in,
        k,
    }
}

/// One ConvNeXt block for channel width `dim`: depthwise mixer conv `[dim,1,k]`,
/// FFN `l1 [4·dim, dim]` / `l2 [dim, 4·dim]`, plus per-channel norms + layer-scales.
fn block(dim: usize, seed: u64) -> BlockW {
    let inter = 4 * dim;
    BlockW {
        norm_w: fill(dim, seed),
        mixer: ConvW {
            weight: fill(dim * 3, seed + 1), // depthwise: [dim, 1, 3]
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

#[test]
fn vae_encoder_compiles_and_runs() {
    // Minimal 2-downsample encoder (strides [1, 2] from DOWNSAMPLE_STRIDES),
    // one ConvNeXt block per stage, small channel widths.
    let (c0, c1, vae_dim, connector_dim) = (4usize, 8usize, 6usize, 10usize);
    let w = VaeEncoderWeights {
        downsamples: vec![
            conv(c0, 1, 3, 1),   // stem  (stride 1): audio 1ch -> c0
            conv(c1, c0, 3, 10), // ds #1 (stride 2): c0 -> c1
        ],
        stages: vec![vec![block(c0, 100)], vec![block(c1, 200)]],
        head: conv(vae_dim, c1, 3, 30), // c1 -> vae_dim
        connector: ConnectorW {
            fc1_w: fill(connector_dim * vae_dim, 40), // [out, in]
            fc1_b: fill(connector_dim, 41),
            norm_w: fill(connector_dim, 42),
            fc2_w: fill(connector_dim * connector_dim, 43),
            fc2_b: fill(connector_dim, 44),
            in_dim: vae_dim,
            out_dim: connector_dim,
        },
        vae_dim,
        connector_dim,
    };

    // padded_len must be divisible by the stride product (2 here).
    let padded_len = 16usize;
    let mut graph = VaeEncoderGraph::compile_for(dev(), &w, padded_len).expect("compile VAE encoder");
    let audio = fill(padded_len, 7);
    let feats = graph.run(&audio).expect("run VAE encoder");

    assert_eq!(feats.len(), graph.n_frames * connector_dim);
    assert!(
        feats.iter().all(|v| v.is_finite()),
        "VAE features must be finite"
    );
}
