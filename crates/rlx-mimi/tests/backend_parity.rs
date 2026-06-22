//! End-to-end parity: the rlx-runtime graph (Metal/MLX/wgpu) vs the CPU eager
//! reference, on the real kyutai/mimi weights. Skips backends not available.

use rlx_mimi::{MimiCodec, MimiCodes, SAMPLE_RATE, default_mimi_dir};
use rlx_runtime::{Device, is_available};
use std::path::PathBuf;

fn model_dir() -> Option<PathBuf> {
    let dir = default_mimi_dir();
    if dir.join("model.safetensors").is_file() {
        Some(dir)
    } else {
        std::env::var("RLX_MIMI_DIR")
            .ok()
            .map(PathBuf::from)
            .filter(|p| p.join("model.safetensors").is_file())
    }
}

fn gpu_devices() -> Vec<Device> {
    [Device::Metal, Device::Mlx, Device::Gpu]
        .into_iter()
        .filter(|&d| is_available(d))
        .collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn code_match(a: &MimiCodes, b: &MimiCodes) -> (usize, usize) {
    let mut ok = 0;
    let mut total = 0;
    for (ra, rb) in a.frames.iter().zip(b.frames.iter()) {
        for (ca, cb) in ra.iter().zip(rb.iter()) {
            total += 1;
            if ca == cb {
                ok += 1;
            }
        }
    }
    (ok, total)
}

#[test]
fn graph_matches_eager_on_real_weights() {
    // Slow (compiles the full graph on real weights per backend) and drives
    // every GPU at once; opt-in so it doesn't contend with the lib's fast
    // synthetic cross-backend tests under a parallel `cargo test`.
    if std::env::var("RLX_MIMI_BACKEND_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip graph_matches_eager_on_real_weights (set RLX_MIMI_BACKEND_PARITY=1)");
        return;
    }
    let Some(dir) = model_dir() else {
        eprintln!("skip graph_matches_eager_on_real_weights: no mimi weights");
        return;
    };
    let devices = gpu_devices();
    if devices.is_empty() {
        eprintln!("skip: no GPU backend available");
        return;
    }

    let cpu = MimiCodec::open(&dir).expect("open cpu");
    assert_eq!(cpu.device(), Device::Cpu);
    let n = SAMPLE_RATE as usize / 4;
    let pcm: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / SAMPLE_RATE as f32).sin() * 0.3)
        .collect();
    let codes_cpu = cpu.encode_pcm(&pcm, Some(8)).expect("cpu encode");
    let wav_cpu = cpu.decode_codes(&codes_cpu).expect("cpu decode");

    for dev in devices {
        let g = MimiCodec::open_on(&dir, dev).expect("open gpu");
        assert_eq!(g.device(), dev, "{dev:?} should be active");

        // Decode the *same* codes on the graph: waveforms must match closely.
        let wav_g = g.decode_codes(&codes_cpu).expect("gpu decode");
        let derr = max_abs(&wav_g, &wav_cpu);
        assert!(derr < 1e-2, "decode on {dev:?}: max|Δ| = {derr}");

        // Encode on the graph: the discrete codes should agree with the CPU
        // reference almost everywhere (argmin can flip on near-ties).
        let codes_g = g.encode_pcm(&pcm, Some(8)).expect("gpu encode");
        let (ok, total) = code_match(&codes_cpu, &codes_g);
        let frac = ok as f32 / total.max(1) as f32;
        assert!(
            frac > 0.97,
            "encode codes on {dev:?}: only {ok}/{total} match ({frac:.3})"
        );
        eprintln!("{dev:?}: decode max|Δ| = {derr:.2e}, code match {ok}/{total} ({frac:.3})");
    }
}
