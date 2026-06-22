//! Compare native RLX TSAC vs the public Bellard Linux binary (encode/decode + cross-decode).
//!
//! On macOS or non-x86_64 hosts, use Docker refs + host RLX instead:
//! ```bash
//! bash crates/rlx-tsac/docker/run.sh build
//! cargo run -p rlx-tsac --example perf_bench --release --features native-codec -- --in-wav speech.wav
//! ```

use anyhow::Result;
use rlx_runtime::Device;
use rlx_tsac::{
    ParityOptions, bench_bellard_parity, default_tsac_dir, ensure_tsac, tsac_binary_supported,
};

fn main() -> Result<()> {
    let mut install_dir = default_tsac_dir();
    let mut in_wav = None;
    let mut quality = 9u8;
    let mut fast = true;
    let mut native_device = Device::Cpu;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--install-dir" => {
                i += 1;
                install_dir = std::path::PathBuf::from(&args[i]);
            }
            "--in-wav" | "--wav" => {
                i += 1;
                in_wav = Some(std::path::PathBuf::from(&args[i]));
            }
            "--quality" | "-q" => {
                i += 1;
                quality = args[i].parse().expect("--quality");
            }
            "--full" => fast = false,
            "--fast" | "-f" => fast = true,
            "--device" => {
                i += 1;
                native_device = rlx_tsac::parse_tsac_device(&args[i])?;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => anyhow::bail!("unknown arg `{other}`"),
        }
        i += 1;
    }

    if !tsac_binary_supported() {
        anyhow::bail!("bellard_parity_bench requires Linux x86_64 with the public tsac binary");
    }

    ensure_tsac(&install_dir)?;
    let in_wav = in_wav.unwrap_or_else(|| {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../rlx-qwen3-tts/examples/audio/ask_not.wav")
    });

    let opts = ParityOptions {
        quality,
        fast,
        native_device,
        ..ParityOptions::default()
    };
    let report = bench_bellard_parity(&install_dir, &in_wav, &opts)?;
    report.print_summary(&opts);
    if !report.passes(&opts) {
        anyhow::bail!("parity tolerances not met");
    }
    Ok(())
}

fn print_help() {
    eprintln!(
        "bellard_parity_bench — native RLX TSAC vs public Bellard binary

Usage:
  cargo run -p rlx-tsac --example bellard_parity_bench --release --features fetch -- \\
    --in-wav speech.wav

Options:
  --install-dir DIR   Weight + binary dir (default: RLX_TSAC_DIR / .cache/tsac)
  --in-wav PATH       Input audio (resampled to 44.1 kHz)
  --quality N         Codebooks / quality (default 9)
  --fast              Disable transformer (default)
  --full              Enable transformer (Bellard default quality path)
  --device NAME       Native RLX backend tag (default cpu)

Env:
  RLX_TSAC_PARITY_MIN_CORR  Min Pearson correlation (default 0.92)
  RLX_TSAC_PARITY_MAX_MSE   Max MSE vs reference PCM (default 0.0025)
"
    );
}
