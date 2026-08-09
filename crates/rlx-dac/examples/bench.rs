// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Decode/encode throughput for DAC across every compiled rlx backend.
//
//   RLX_DAC_DIR=/path/to/dac cargo run -p rlx-dac --example bench --release \
//     --features cpu,metal,mlx,gpu -- --dur 4 --iters 5
//
// Weights: a dir with `model.safetensors` (config.json optional; falls back to
// the 24 kHz default). Real codes come from encoding synthetic PCM.

use anyhow::Result;
use rlx_core::AudioCodec;
use rlx_core::codec_bench as cb;
use rlx_dac::DacCodec;
use rlx_runtime::Device;

const CRATE: &str = "rlx-dac";

// Elements are cfg-gated per backend, so `vec![..]` is not an option.
#[allow(clippy::vec_init_then_push)]
fn main() -> Result<()> {
    let (dur, iters) = cb::parse_dur_iters();
    let dir = rlx_dac::resolve_model_dir(None);
    if !dir.join("model.safetensors").is_file() {
        eprintln!(
            "skip {CRATE}: missing {}/model.safetensors (set RLX_DAC_DIR)",
            dir.display()
        );
        return Ok(());
    }

    #[allow(unused_mut)]
    let mut candidates = vec![];
    #[cfg(feature = "metal")]
    candidates.push(Device::Metal);
    #[cfg(feature = "mlx")]
    candidates.push(Device::Mlx);
    #[cfg(feature = "gpu")]
    candidates.push(Device::Gpu);

    for dev in cb::available(&candidates) {
        if let Err(e) = bench_device(&dir, dev, dur, iters) {
            cb::report_fail(CRATE, dev, "all", &e.to_string());
        }
    }
    Ok(())
}

fn bench_device(dir: &std::path::Path, dev: Device, dur: f64, iters: usize) -> Result<()> {
    let codec = DacCodec::open_on(dir, dev)?;
    let info = codec.info();
    let n_samples = (dur * info.sample_rate as f64) as usize;
    let pcm = cb::synth_pcm(n_samples, 0.3, 0xDAC0);

    // encode produces real codes used by the decode bench.
    let codes = match cb::time_once_ms(|| codec.encode_pcm(&pcm, None)) {
        Ok((codes, compile_ms)) => {
            let t = codes.num_frames();
            let audio_s = (t * info.hop_length) as f64 / info.sample_rate as f64;
            let run_ms = cb::time_median_ms(0, iters, || codec.encode_pcm(&pcm, None))?;
            cb::report(CRATE, dev, "encode", audio_s, t, compile_ms, run_ms);
            codes
        }
        Err(e) => {
            cb::report_fail(CRATE, dev, "encode", &e.to_string());
            return Ok(());
        }
    };

    let t = codes.num_frames();
    let audio_s = (t * info.hop_length) as f64 / info.sample_rate as f64;
    match cb::time_once_ms(|| codec.decode_codes(&codes)) {
        Ok((_, compile_ms)) => {
            let run_ms = cb::time_median_ms(0, iters, || codec.decode_codes(&codes))?;
            cb::report(CRATE, dev, "decode", audio_s, t, compile_ms, run_ms);
        }
        Err(e) => cb::report_fail(CRATE, dev, "decode", &e.to_string()),
    }
    Ok(())
}
