// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Decode/encode throughput for EnCodec across every compiled rlx backend.
//
//   cargo run -p rlx-encodec --example bench --release \
//     --features cpu,metal,mlx,gpu -- --dur 4 --iters 5
//
// Emits one `BENCH crate=... backend=... op=...` line per (backend, op) for the
// top-level coverage matrix. Uses the real decoder weights from the parity
// fixture; inputs are synthesized at the requested duration (cost tracks length,
// not code values).

use anyhow::Result;
use rlx_core::AudioCodec;
use rlx_core::codec_bench as cb;
use rlx_encodec::EncodecCodec;
use rlx_runtime::Device;

const CRATE: &str = "rlx-encodec";

fn main() -> Result<()> {
    let (dur, iters) = cb::parse_dur_iters();

    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/encodec24.safetensors");
    if !fixture.is_file() {
        eprintln!(
            "skip {CRATE}: missing {} (run scripts/gen_fixture.py)",
            fixture.display()
        );
        return Ok(());
    }

    let candidates = vec![];
    #[cfg(feature = "metal")]
    candidates.push(Device::Metal);
    #[cfg(feature = "mlx")]
    candidates.push(Device::Mlx);
    #[cfg(feature = "gpu")]
    candidates.push(Device::Gpu);
    let devices = cb::available(&candidates);

    for dev in devices {
        if let Err(e) = bench_device(&fixture, dev, dur, iters) {
            cb::report_fail(CRATE, dev, "all", &e.to_string());
        }
    }
    Ok(())
}

fn bench_device(fixture: &std::path::Path, dev: Device, dur: f64, iters: usize) -> Result<()> {
    let codec = EncodecCodec::from_safetensors_path(fixture, dev)?;
    let info = codec.info();
    let n_samples = (dur * info.sample_rate as f64) as usize;
    let t = info.frames_for_samples(n_samples).max(1);
    let audio_s = (t * info.hop_length) as f64 / info.sample_rate as f64;

    // decode: quantizer-major codes [n_q][t]
    let codes = cb::synth_codes_qmajor(info.max_quantizers, t, info.codebook_size, 0xDEC0);
    match cb::time_once_ms(|| codec.decode(&codes)) {
        Ok((_, compile_ms)) => {
            let run_ms = cb::time_median_ms(0, iters, || codec.decode(&codes))?;
            cb::report(CRATE, dev, "decode", audio_s, t, compile_ms, run_ms);
        }
        Err(e) => cb::report_fail(CRATE, dev, "decode", &e.to_string()),
    }

    // encode: mono PCM -> codes
    let pcm = cb::synth_pcm(n_samples, 0.3, 0xE5C0);
    match cb::time_once_ms(|| codec.encode(&pcm, info.max_quantizers)) {
        Ok((_, compile_ms)) => {
            let run_ms = cb::time_median_ms(0, iters, || codec.encode(&pcm, info.max_quantizers))?;
            cb::report(CRATE, dev, "encode", audio_s, t, compile_ms, run_ms);
        }
        Err(e) => cb::report_fail(CRATE, dev, "encode", &e.to_string()),
    }
    Ok(())
}
