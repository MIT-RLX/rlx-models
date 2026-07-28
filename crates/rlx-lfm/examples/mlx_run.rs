// RLX — versatile ML compiler + runtime. GPLv3.
//! Run an mlx-community LFM2 (affine 4-bit) checkpoint end-to-end and print the
//! generated text — the coherence of the output is the parity signal that the
//! `MlxLoader` (affine→F32) + config parse + hybrid conv/attn flow all line up.
//!
//! Usage:
//!   cargo run --release -p rlx-lfm --example mlx_run -- .mlx-test/lfm2-1.2b-4bit \
//!       [device] [--prompt "..."] [--ngen 40]

use std::path::Path;

use anyhow::{Context, Result, bail};
use rlx_lfm::runner::LfmRunner;
use rlx_runtime::Device;
use tokenizers::Tokenizer;

fn pick_device(arg: Option<&str>) -> Result<Device> {
    match arg.unwrap_or("cpu").to_ascii_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "metal" => Ok(Device::Metal),
        "mlx" => Ok(Device::Mlx),
        "gpu" | "wgpu" => Ok(Device::Gpu),
        "cuda" => Ok(Device::Cuda),
        other => bail!("unknown device {other:?}"),
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = args
        .first()
        .cloned()
        .unwrap_or_else(|| ".mlx-test/lfm2-1.2b-4bit".to_string());
    let device = pick_device(
        args.get(1)
            .map(String::as_str)
            .filter(|s| !s.starts_with("--")),
    )?;
    let prompt = args
        .iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "The capital of France is".to_string());
    let ngen: usize = args
        .iter()
        .position(|a| a == "--ngen")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);

    let tok = Tokenizer::from_file(Path::new(&dir).join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
    let enc = tok
        .encode(prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let ids: Vec<u32> = enc.get_ids().to_vec();
    eprintln!("[lfm mlx] dir={dir} device={device:?} prompt={prompt:?} ids={ids:?}");

    let t0 = std::time::Instant::now();
    let mut runner = LfmRunner::builder()
        .weights(&dir)
        .device(device)
        .build()
        .with_context(|| format!("build LfmRunner from {dir}"))?;
    eprintln!("[lfm mlx] loaded+built in {:.2?}", t0.elapsed());

    let mut out_ids = ids.clone();
    let new_ids = runner.generate(&ids, ngen, |_| {});
    out_ids.extend_from_slice(&new_ids);

    // Sanity: every generated id is in-vocab and finite decode.
    let text = tok
        .decode(&out_ids, true)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    println!("\n── generated ──────────────────────────────────────────");
    println!("{text}");
    println!("───────────────────────────────────────────────────────");
    println!("gen ids: {new_ids:?}");
    Ok(())
}
