// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Decode throughput for NVIDIA NanoCodec (Group-FSQ) across every compiled rlx
// backend.
//
//   cargo run -p rlx-nanocodec --example bench --release \
//     --features cpu,metal,mlx,gpu -- --dur 4 --iters 5
//
// Decode-only codec. Uses the real decoder weights from the parity fixture; FSQ
// group codes are synthesized at the requested duration.

use anyhow::Result;
use rlx_core::codec_bench as cb;
use rlx_nanocodec::{NanoDecoderGraph, NanoWeights, model};

const CRATE: &str = "rlx-nanocodec";
const SAMPLE_RATE: usize = 22_050;

fn main() -> Result<()> {
    let (dur, iters) = cb::parse_dur_iters();
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/nano_dec.safetensors");
    if !fixture.is_file() {
        eprintln!(
            "skip {CRATE}: missing {} (run scripts/gen_fixture.py)",
            fixture.display()
        );
        return Ok(());
    }
    let w = NanoWeights::from_safetensors(&std::fs::read(&fixture)?)?;

    let t = ((dur * SAMPLE_RATE as f64) / model::SAMPLES_PER_FRAME as f64).max(1.0) as usize;
    let audio_s = (t * model::SAMPLES_PER_FRAME) as f64 / SAMPLE_RATE as f64;
    let range: usize = model::LEVELS.iter().product::<i64>() as usize;
    let codes = cb::synth_codes_i64(model::NUM_GROUPS, t, range, 0x4A40);

    let candidates = vec![];
    #[cfg(feature = "metal")]
    candidates.push(Device::Metal);
    #[cfg(feature = "mlx")]
    candidates.push(Device::Mlx);
    #[cfg(feature = "gpu")]
    candidates.push(Device::Gpu);

    for dev in cb::available(&candidates) {
        match cb::time_once_ms(|| NanoDecoderGraph::compile_for(dev, &w, t)) {
            Ok((mut g, compile_ms)) => {
                match cb::time_median_ms(1, iters, || g.decode_codes(&codes)) {
                    Ok(run_ms) => cb::report(CRATE, dev, "decode", audio_s, t, compile_ms, run_ms),
                    Err(e) => cb::report_fail(CRATE, dev, "decode", &e.to_string()),
                }
            }
            Err(e) => cb::report_fail(CRATE, dev, "decode", &e.to_string()),
        }
    }
    Ok(())
}
