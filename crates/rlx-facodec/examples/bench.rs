// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Decode throughput for FACodec (factorized HiFi-GAN decoder) across every
// compiled rlx backend.
//
//   cargo run -p rlx-facodec --example bench --release \
//     --features cpu,metal,mlx,gpu -- --dur 4 --iters 5
//
// Decode-only codec: input is a latent `[256, T]` channel-major + a speaker
// embedding (AdaIN timbre). Uses the real decoder weights from the parity
// fixture; latent + speaker vector are synthesized at the requested duration.

use anyhow::Result;
use rlx_core::codec_bench as cb;
use rlx_facodec::{FacodecDecoderGraph, FacodecWeights, model};
#[cfg(any(feature = "metal", feature = "mlx", feature = "gpu"))]
use rlx_runtime::Device;

const CRATE: &str = "rlx-facodec";
const SAMPLE_RATE: usize = 16_000;

// Elements are cfg-gated per backend, so `vec![..]` is not an option.
#[allow(clippy::vec_init_then_push)]
fn main() -> Result<()> {
    let (dur, iters) = cb::parse_dur_iters();
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/facodec_dec.safetensors");
    if !fixture.is_file() {
        eprintln!(
            "skip {CRATE}: missing {} (run scripts/gen_fixture.py)",
            fixture.display()
        );
        return Ok(());
    }
    let w = FacodecWeights::from_safetensors(&std::fs::read(&fixture)?)?;

    let upsample: usize = model::UP_RATIOS.iter().product();
    let t = ((dur * SAMPLE_RATE as f64) / upsample as f64).max(1.0) as usize;
    let audio_s = (t * upsample) as f64 / SAMPLE_RATE as f64;
    let emb = cb::synth_f32(model::IN_CH * t, 0xFAC0);
    let spk = cb::synth_f32(model::IN_CH, 0x5CA0);
    let (gamma, beta) = w.timbre_affine(&spk);

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
        match cb::time_once_ms(|| FacodecDecoderGraph::compile_for(dev, &w, &gamma, &beta, t)) {
            Ok((mut g, compile_ms)) => match cb::time_median_ms(1, iters, || g.run(&emb)) {
                Ok(run_ms) => cb::report(CRATE, dev, "decode", audio_s, t, compile_ms, run_ms),
                Err(e) => cb::report_fail(CRATE, dev, "decode", &e.to_string()),
            },
            Err(e) => cb::report_fail(CRATE, dev, "decode", &e.to_string()),
        }
    }
    Ok(())
}
