//! Native vs Bellard binary parity on Linux x86_64.
//! For other hosts use `perf_bench` (RLX on host, refs in Docker).
//!
//! `RLX_TSAC_PARITY=1 cargo test -p rlx-tsac --test bellard_parity --release -- --nocapture`

use anyhow::Result;
use rlx_runtime::Device;
use rlx_tsac::{
    ParityOptions, bench_bellard_parity, default_tsac_dir, ensure_tsac, tsac_binary_supported,
};

fn parity_enabled() -> bool {
    std::env::var("RLX_TSAC_PARITY").ok().as_deref() == Some("1")
}

fn fixture_wav() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../rlx-qwen3-tts/examples/audio/ask_not.wav")
}

#[test]
fn native_vs_bellard_roundtrip_and_cross_decode() -> Result<()> {
    if !parity_enabled() {
        eprintln!("skip bellard_parity (set RLX_TSAC_PARITY=1 on Linux x86_64)");
        return Ok(());
    }
    if !tsac_binary_supported() {
        eprintln!("skip bellard_parity: requires Linux x86_64");
        return Ok(());
    }

    let dir = default_tsac_dir();
    ensure_tsac(&dir)?;
    let wav = fixture_wav();
    assert!(wav.is_file(), "missing {}", wav.display());

    let opts = ParityOptions {
        quality: 8,
        fast: true,
        native_device: Device::Cpu,
        ..ParityOptions::default()
    };
    let report = bench_bellard_parity(&dir, &wav, &opts)?;
    report.print_summary(&opts);
    assert!(
        report.passes(&opts),
        "native vs Bellard PCM diverged beyond tolerance (corr>={} mse<={})",
        opts.min_correlation,
        opts.max_mse
    );
    Ok(())
}
