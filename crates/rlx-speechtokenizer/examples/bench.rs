// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Decode/encode throughput for SpeechTokenizer across every compiled rlx backend.
//
//   cargo run -p rlx-speechtokenizer --example bench --release \
//     --features cpu,metal,mlx,gpu -- --dur 4 --iters 5
//
// Uses the real codec weights from the parity fixture; real codes come from
// encoding synthetic PCM.

use anyhow::Result;
use rlx_core::AudioCodec;
use rlx_core::codec_bench as cb;
use rlx_runtime::Device;
use rlx_speechtokenizer::SpeechTokenizerCodec;

const CRATE: &str = "rlx-speechtokenizer";

fn main() -> Result<()> {
    let (dur, iters) = cb::parse_dur_iters();
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/speechtokenizer.safetensors");
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

    for dev in cb::available(&candidates) {
        if let Err(e) = bench_device(&fixture, dev, dur, iters) {
            cb::report_fail(CRATE, dev, "all", &e.to_string());
        }
    }
    Ok(())
}

fn bench_device(fixture: &std::path::Path, dev: Device, dur: f64, iters: usize) -> Result<()> {
    let codec = SpeechTokenizerCodec::from_safetensors_path(fixture, dev)?;
    let info = codec.info();
    let nq = info.max_quantizers;
    let n_samples = (dur * info.sample_rate as f64) as usize;
    let pcm = cb::synth_pcm(n_samples, 0.3, 0x57E0);

    let codes = match cb::time_once_ms(|| codec.encode(&pcm, nq)) {
        Ok((codes, compile_ms)) => {
            let t = codes.first().map(|c| c.len()).unwrap_or(0);
            let audio_s = (t * info.hop_length) as f64 / info.sample_rate as f64;
            let run_ms = cb::time_median_ms(0, iters, || codec.encode(&pcm, nq))?;
            cb::report(CRATE, dev, "encode", audio_s, t, compile_ms, run_ms);
            codes
        }
        Err(e) => {
            cb::report_fail(CRATE, dev, "encode", &e.to_string());
            return Ok(());
        }
    };

    let t = codes.first().map(|c| c.len()).unwrap_or(0);
    let audio_s = (t * info.hop_length) as f64 / info.sample_rate as f64;
    match cb::time_once_ms(|| codec.decode(&codes)) {
        Ok((_, compile_ms)) => {
            let run_ms = cb::time_median_ms(0, iters, || codec.decode(&codes))?;
            cb::report(CRATE, dev, "decode", audio_s, t, compile_ms, run_ms);
        }
        Err(e) => cb::report_fail(CRATE, dev, "decode", &e.to_string()),
    }
    Ok(())
}
