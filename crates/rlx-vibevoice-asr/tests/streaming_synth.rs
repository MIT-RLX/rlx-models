// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//!
//! Synthetic Streaming VAE encode (GELU) + config checks.
//! Real-weight presence: set `RLX_VIBEVOICE_ASR_STREAMING_DIR`.

use rlx_runtime::Device;
use rlx_vibevoice_asr::config::{VaeFfnAct, VibeAsrConfig};
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

#[test]
fn streaming_config_defaults() {
    let c = VibeAsrConfig::streaming_7b();
    assert_eq!(c.lm.hidden_size, 3584);
    assert_eq!(c.vae_ffn, VaeFfnAct::Gelu);
    assert_eq!(c.streaming.chunk_frames, 22);
    assert_eq!(c.streaming.chunk_samples(), 22 * 3200);
}

#[test]
fn gelu_vae_encode_cpu() {
    // Tiny 2-stage encoder (strides 1, 2) — same shape as vae_encoder_smoke,
    // but compiled with GELU / no whole-clip normalize (Streaming path).
    let (c0, c1, vae_dim, connector_dim) = (4usize, 8usize, 6usize, 10usize);
    let w = VaeEncoderWeights {
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
    };
    let padded_len = 16usize;
    let mut g =
        VaeEncoderGraph::compile_for_streaming(Device::Cpu, &w, padded_len, VaeFfnAct::Gelu)
            .expect("compile");
    let feats = g.run(&fill(padded_len, 7)).expect("run");
    assert_eq!(feats.len(), g.n_frames * connector_dim);
    assert!(feats.iter().all(|v| v.is_finite()));
}

#[test]
fn env_gated_streaming_dir_present() {
    let Ok(dir) = std::env::var("RLX_VIBEVOICE_ASR_STREAMING_DIR") else {
        eprintln!("skip: RLX_VIBEVOICE_ASR_STREAMING_DIR unset");
        return;
    };
    let p = std::path::Path::new(&dir);
    assert!(
        p.join("config.json").is_file(),
        "expected config.json under {dir}"
    );
    let cfg = VibeAsrConfig::from_model_dir(p).expect("parse config");
    assert!(cfg.lm.hidden_size >= 1536);
    assert_eq!(cfg.vae_ffn, VaeFfnAct::Gelu);
}
