//! RLX native codec on the host vs Bellard + tsac-ng reference binaries in Docker.
//!
//! ```bash
//! bash crates/rlx-tsac/docker/run.sh build
//! cargo run -p rlx-tsac --example perf_bench --release --features native-codec -- \
//!   --in-wav speech.wav --quality 9 --fast
//! ```

use anyhow::Result;
use rlx_runtime::Device;
use rlx_tsac::{PerfOptions, bench_perf, default_tsac_dir};

fn main() -> Result<()> {
    let mut in_wav = None;
    let mut quality = 9u8;
    let mut fast = true;
    let mut native_device = Device::Cpu;
    let mut docker_image = None;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
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
            "--docker-image" => {
                i += 1;
                docker_image = Some(args[i].clone());
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => anyhow::bail!("unknown arg `{other}`"),
        }
        i += 1;
    }

    let in_wav = in_wav.unwrap_or_else(|| {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../rlx-qwen3-tts/examples/audio/ask_not.wav")
    });

    let mut opts = PerfOptions {
        quality,
        fast,
        native_device,
        ..Default::default()
    };
    if let Some(image) = docker_image {
        opts.docker.image = image;
    }

    let report = bench_perf(default_tsac_dir(), &in_wav, &opts)?;
    report.print_summary(&opts);
    Ok(())
}

fn print_help() {
    eprintln!(
        "perf_bench — RLX on host vs Docker reference binaries (bellard + tsac-ng)

Usage:
  bash crates/rlx-tsac/docker/run.sh build
  cargo run -p rlx-tsac --example perf_bench --release --features native-codec -- \\
    --in-wav speech.wav

Options:
  --in-wav PATH       Input audio (resampled to 44.1 kHz)
  --quality N         Codebooks / quality (default 9)
  --fast              Disable transformer (default)
  --full              Enable transformer
  --device NAME       RLX native backend tag (default cpu)
  --docker-image TAG  Reference image (default rlx-tsac-ref)

Env:
  RLX_TSAC_REF_IMAGE          Docker image tag
  RLX_TSAC_DOCKER_PLATFORM    Platform (default linux/amd64)
  RLX_TSAC_PARITY_MIN_CORR    Min correlation vs refs (default 0.92)
  RLX_TSAC_PARITY_MAX_MSE     Max MSE vs refs (default 0.0025)
"
    );
}
