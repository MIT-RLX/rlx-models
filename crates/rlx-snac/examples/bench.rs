// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Decode throughput for SNAC (multi-scale RVQ) across every compiled rlx backend.
//
//   cargo run -p rlx-snac --example bench --release \
//     --features cpu,metal,mlx,gpu -- --dur 4 --iters 5
//
// Uses the real decoder weights from the parity fixture. Decode-only: the
// encoder needs separate weights not shipped in the decoder fixture. Multi-scale
// codes are synthesized at the requested duration per the 24 kHz vq strides.

use anyhow::Result;
use rlx_core::audio_codec::HierarchicalCodes;
use rlx_core::codec_bench as cb;
#[cfg(any(feature = "metal", feature = "mlx", feature = "gpu"))]
use rlx_runtime::Device;
use rlx_snac::{SnacConfig, SnacDecoder};

const CRATE: &str = "rlx-snac";

// Elements are cfg-gated per backend, so `vec![..]` is not an option.
#[allow(clippy::vec_init_then_push)]
fn main() -> Result<()> {
    let (dur, iters) = cb::parse_dur_iters();
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/snac24_decoder.safetensors");
    if !fixture.is_file() {
        eprintln!(
            "skip {CRATE}: missing {} (run scripts/gen_fixture.py)",
            fixture.display()
        );
        return Ok(());
    }

    let cfg = SnacConfig::snac_24khz();
    let upsample: usize = cfg.decoder_rates.iter().product();
    let max_stride = *cfg.vq_strides.iter().max().unwrap_or(&1);
    // finest-rate frame count, divisible by every stride.
    let mut t_base = ((dur * cfg.sampling_rate as f64) / upsample as f64) as usize;
    t_base = (t_base / max_stride).max(1) * max_stride;
    let audio_s = (t_base * upsample) as f64 / cfg.sampling_rate as f64;

    let mut r = cb::Lcg::new(0x57AC);
    let levels: Vec<Vec<u32>> = cfg
        .vq_strides
        .iter()
        .map(|&s| (0..t_base / s).map(|_| r.idx(cfg.codebook_size)).collect())
        .collect();
    let codes = HierarchicalCodes::new(levels);

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
        match SnacDecoder::from_safetensors_path(&fixture, dev) {
            Ok(dec) => match cb::time_once_ms(|| dec.decode(&codes, Some(7))) {
                Ok((_, compile_ms)) => {
                    let run_ms = cb::time_median_ms(0, iters, || dec.decode(&codes, Some(7)))?;
                    cb::report(CRATE, dev, "decode", audio_s, t_base, compile_ms, run_ms);
                }
                Err(e) => cb::report_fail(CRATE, dev, "decode", &e.to_string()),
            },
            Err(e) => cb::report_fail(CRATE, dev, "decode", &e.to_string()),
        }
    }
    Ok(())
}
