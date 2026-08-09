// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Decode throughput for XCodec2 (RoFormer-Vocos) across every compiled rlx
// backend.
//
//   cargo run -p rlx-xcodec --example bench --release \
//     --features cpu,metal,mlx,gpu -- --dur 4 --iters 5
//
// Decode-only codec: input is a quantized latent `[T, 1024]` → head → ISTFT wav.
// Uses the real decoder weights from the parity fixture; the latent is
// synthesized at the requested duration. The timed op is the backend graph
// (`run`); the host ISTFT (`head_to_wav`) is identical across backends.

use anyhow::Result;
use rlx_core::codec_bench as cb;
#[cfg(any(feature = "metal", feature = "mlx", feature = "gpu"))]
use rlx_runtime::Device;
use rlx_xcodec::{XcodecDecoderGraph, XcodecWeights, head_to_wav, model};

const CRATE: &str = "rlx-xcodec";
const SAMPLE_RATE: usize = 16_000;

// Elements are cfg-gated per backend, so `vec![..]` is not an option.
#[allow(clippy::vec_init_then_push)]
fn main() -> Result<()> {
    let (dur, iters) = cb::parse_dur_iters();
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/xcodec_dec.safetensors");
    if !fixture.is_file() {
        eprintln!(
            "skip {CRATE}: missing {} (run scripts/gen_fixture.py)",
            fixture.display()
        );
        return Ok(());
    }
    let w = XcodecWeights::from_safetensors(&std::fs::read(&fixture)?)?;

    let t = ((dur * SAMPLE_RATE as f64) / model::HOP as f64).max(1.0) as usize;
    let audio_s = (t * model::HOP) as f64 / SAMPLE_RATE as f64;
    let emb = cb::synth_f32(t * model::DIM, 0xC0DEC2);

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
        match cb::time_once_ms(|| XcodecDecoderGraph::compile_for(dev, &w, t)) {
            Ok((mut g, compile_ms)) => {
                // sanity: produce a wav once so the op is end-to-end valid.
                let _ = g.run(&emb).map(|h| head_to_wav(&h, t, &w.window));
                match cb::time_median_ms(1, iters, || g.run(&emb)) {
                    Ok(run_ms) => cb::report(CRATE, dev, "decode", audio_s, t, compile_ms, run_ms),
                    Err(e) => cb::report_fail(CRATE, dev, "decode", &e.to_string()),
                }
            }
            Err(e) => cb::report_fail(CRATE, dev, "decode", &e.to_string()),
        }
    }
    Ok(())
}
