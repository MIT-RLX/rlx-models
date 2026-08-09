// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Encode/decode (file ⇄ .tsac) throughput for TSAC across every compiled rlx
// backend.
//
//   RLX_DAC_DIR=.cache/dac44 cargo run -p rlx-tsac --example bench --release \
//     --features metal,mlx,gpu -- --dur 4 --iters 3
//
// Backend choice (see TsacBackendKind):
//   * `Correct` (used here) = Descript-DAC-44 kHz RVQ codec, encode+decode run
//     entirely on the chosen rlx backend (cpu/metal/mlx/wgpu). This is the
//     GPU-native path — no CPU transformer in the loop.
//   * `Native` = Bellard's libnc transformer-entropy codec. Its only GPU backend
//     is CUDA (`libnc_cuda.so`); on Apple GPUs the transformer falls back to CPU,
//     so it is *not* benchmarked here. Use `examples/perf_bench.rs` for that path.
//
// `Correct` needs Descript-DAC-44 kHz weights (`model.safetensors` + a 44 kHz
// `config.json`) under `RLX_DAC_DIR` / `.cache/dac44`; skips cleanly if absent.

use anyhow::Result;
use rlx_core::codec_bench as cb;
use rlx_runtime::Device;
use rlx_tsac::correct;
use rlx_tsac::{TsacBackendKind, TsacCodec, TsacOptions};

const CRATE: &str = "rlx-tsac";

// Elements are cfg-gated per backend, so `vec![..]` is not an option.
#[allow(clippy::vec_init_then_push)]
fn main() -> Result<()> {
    let (_dur, iters) = cb::parse_dur_iters();
    let dir = correct::default_dir();
    if !correct::weights_available(&dir) {
        eprintln!(
            "skip {CRATE}: Descript-DAC-44kHz weights not at {} (set RLX_DAC_DIR / .cache/dac44)",
            dir.display()
        );
        return Ok(());
    }

    let in_wav = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../rlx-qwen3-tts/examples/audio/ask_not.wav");
    if !in_wav.is_file() {
        eprintln!("skip {CRATE}: missing input wav {}", in_wav.display());
        return Ok(());
    }
    let audio_s = wav_seconds(&in_wav).unwrap_or(0.0);

    // `mut` is only exercised by the backend-feature pushes below.
    #[allow(unused_mut)]
    let mut candidates = vec![];
    #[cfg(feature = "metal")]
    candidates.push(Device::Metal);
    #[cfg(feature = "mlx")]
    candidates.push(Device::Mlx);
    #[cfg(feature = "gpu")]
    candidates.push(Device::Gpu);

    for dev in cb::available(&candidates) {
        if let Err(e) = bench_device(&dir, dev, &in_wav, audio_s, iters) {
            cb::report_fail(CRATE, dev, "all", &e.to_string());
        }
    }
    Ok(())
}

fn bench_device(
    dir: &std::path::Path,
    dev: Device,
    in_wav: &std::path::Path,
    audio_s: f64,
    iters: usize,
) -> Result<()> {
    let opts = TsacOptions {
        device: dev,
        backend: TsacBackendKind::Correct,
        quality: Some(6),
        ..Default::default()
    };
    let codec = TsacCodec::open_with_options(dir, opts)?;
    // The codec resolves its own device; report what it actually ran on.
    let ran = codec.device();
    let tmp = std::env::temp_dir().join(format!("rlx-tsac-bench-{}.tsac", std::process::id()));
    let out = std::env::temp_dir().join(format!("rlx-tsac-bench-{}.wav", std::process::id()));

    match cb::time_median_ms(1, iters, || codec.encode(in_wav, &tmp).map(|_| ())) {
        Ok(run_ms) => cb::report(CRATE, ran, "encode", audio_s, 0, 0.0, run_ms),
        Err(e) => cb::report_fail(CRATE, ran, "encode", &e.to_string()),
    }
    match cb::time_median_ms(1, iters, || codec.decode(&tmp, &out)) {
        Ok(run_ms) => cb::report(CRATE, ran, "decode", audio_s, 0, 0.0, run_ms),
        Err(e) => cb::report_fail(CRATE, ran, "decode", &e.to_string()),
    }
    let _ = std::fs::remove_file(&tmp);
    let _ = std::fs::remove_file(&out);
    Ok(())
}

/// Seconds of a 16-bit PCM WAV from its header (sample rate @24, channels @22,
/// data size = file - 44-byte header).
fn wav_seconds(path: &std::path::Path) -> Option<f64> {
    let b = std::fs::read(path).ok()?;
    let ch = (*b.get(22)? as usize | ((*b.get(23)? as usize) << 8)).max(1);
    let sr = u32::from_le_bytes([*b.get(24)?, *b.get(25)?, *b.get(26)?, *b.get(27)?]).max(1);
    let data = b.len().saturating_sub(44);
    Some(data as f64 / (sr as f64 * ch as f64 * 2.0))
}
