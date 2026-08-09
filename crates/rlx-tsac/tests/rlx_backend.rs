//! TSAC decode on RLX backends, verified against the C reference decoder.
//!
//! ```bash
//! RLX_TSAC_E2E=1 cargo test -p rlx-tsac --test rlx_backend --release -- --nocapture
//! # GPU backends:
//! RLX_TSAC_E2E=1 cargo test -p rlx-tsac --test rlx_backend --release \
//!   --features apple-silicon -- --nocapture
//! ```

use rlx_runtime::{Device, is_available};
use rlx_tsac::rlx_decode::{RlxDecoder, read_codes};
use rlx_tsac::{TsacBackendKind, TsacCodec, TsacOptions, audio, default_tsac_dir, ensure_tsac};
use std::path::PathBuf;

fn e2e_enabled() -> bool {
    std::env::var("RLX_TSAC_E2E").ok().as_deref() == Some("1")
}

fn fixture_wav() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../rlx-qwen3-tts/examples/audio/ask_not.wav")
}

// Elements are cfg-gated per backend, so `vec![..]` is not an option.
#[allow(clippy::vec_init_then_push)]
fn gpu_devices() -> Vec<Device> {
    // `mut` is only exercised by the backend-feature pushes below.
    #[allow(unused_mut)]
    let mut v = Vec::new();
    #[cfg(feature = "metal")]
    v.push(Device::Metal);
    #[cfg(feature = "mlx")]
    v.push(Device::Mlx);
    #[cfg(feature = "gpu")]
    v.push(Device::Gpu);
    v
}

fn corr(a: &[f32], b: &[f32]) -> f32 {
    audio::correlation(a, b)
}
fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    audio::max_abs_error(a, b)
}

#[test]
fn rlx_decode_matches_c_reference() {
    if !e2e_enabled() {
        eprintln!("skip rlx_decode_matches_c_reference (set RLX_TSAC_E2E=1)");
        return;
    }
    let dir = default_tsac_dir();
    ensure_tsac(&dir).expect("fetch TSAC with `just fetch-tsac`");
    let wav = fixture_wav();
    assert!(wav.is_file(), "missing fixture {}", wav.display());

    let tag = std::process::id();
    let tsac = std::env::temp_dir().join(format!("rlx-tsac-bk-{tag}.tsac"));
    let oracle = std::env::temp_dir().join(format!("rlx-tsac-bk-{tag}-c.wav"));

    // 1. Encode + C-decode (oracle) via the vendored reference.
    let c = TsacCodec::open_with_options(
        &dir,
        TsacOptions {
            device: Device::Cpu,
            backend: TsacBackendKind::Native,
            quality: Some(6),
            ..Default::default()
        },
    )
    .unwrap();
    c.encode(&wav, &tsac).expect("encode");
    c.decode(&tsac, &oracle).expect("c decode");
    let oracle_pcm = audio::load_pcm_from_wav(&oracle).expect("load oracle");

    // 2. RLX decode (CPU ndarray path) of the same bitstream.
    let (codes, n_frames, n_cb) = read_codes(&tsac).expect("read codes");
    eprintln!("codes: n_frames={n_frames} n_cb={n_cb}");
    let cpu = RlxDecoder::open(&dir, Device::Cpu).expect("open cpu decoder");
    let cpu_pcm = cpu
        .decode_codes(&codes, n_frames, n_cb)
        .expect("cpu decode");
    let ch0: Vec<f32> = cpu_pcm.row(0).to_vec();

    let n = ch0.len().min(oracle_pcm.len());
    let cc = corr(&ch0[..n], &oracle_pcm[..n]);
    let mx = max_abs(&ch0[..n], &oracle_pcm[..n]);
    eprintln!(
        "RLX(cpu) vs C: corr={cc:.5} max_abs={mx:.5} (rlx_len={}, c_len={})",
        ch0.len(),
        oracle_pcm.len()
    );
    // CPU RLX decode is bit-exact with the C reference (same batching + snake/
    // conv-transpose conventions).
    assert!(
        cc > 0.999,
        "RLX cpu decode vs C reference correlation {cc} too low"
    );
    assert!(
        mx < 1e-3,
        "RLX cpu decode vs C reference max_abs {mx} too high"
    );

    // 3. Each available GPU backend must match the CPU graph result closely.
    let cpu_flat: Vec<f32> = cpu_pcm.iter().copied().collect();
    for dev in gpu_devices() {
        if !is_available(dev) {
            eprintln!("skip {dev:?}: not available");
            continue;
        }
        let g = RlxDecoder::open(&dir, dev).expect("open gpu decoder");
        let gpu_pcm = g.decode_codes(&codes, n_frames, n_cb).expect("gpu decode");
        let gpu_flat: Vec<f32> = gpu_pcm.iter().copied().collect();
        let m = gpu_flat.len().min(cpu_flat.len());
        let err = max_abs(&gpu_flat[..m], &cpu_flat[..m]);
        eprintln!("RLX {dev:?} vs cpu: max_abs={err:.2e}");
        assert!(err < 1e-2, "{dev:?} decode vs cpu max|Δ|={err}");
    }

    let _ = std::fs::remove_file(&tsac);
    let _ = std::fs::remove_file(&oracle);
}
