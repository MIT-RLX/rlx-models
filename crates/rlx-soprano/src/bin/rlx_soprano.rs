// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! `rlx-soprano` — Soprano 1.1 CLI.
//!
//! ```text
//! rlx-soprano --text "Hello." --output out.wav --device metal
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_runtime::Device;
use rlx_soprano::{
    DEFAULT_GGUF_NAME, DEFAULT_LOCAL_DIR, DEFAULT_RLXP_NAME, InferOpts, NativeSoprano,
    pack_directory, pack_rlxp, parse_device, peak_amplitude,
};

fn main() -> Result<()> {
    let mut text: Option<String> = None;
    let mut output = PathBuf::from("soprano.wav");
    let mut model_dir = PathBuf::from(DEFAULT_LOCAL_DIR);
    let mut device = Device::Cpu;
    let mut opts = InferOpts::default();
    let mut pack_gguf: Option<PathBuf> = None;
    let mut pack_rlxp_out: Option<PathBuf> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--text" => text = Some(args.next().context("missing --text")?),
            "--output" | "-o" => output = PathBuf::from(args.next().context("missing --output")?),
            "--model-dir" | "--weights" => {
                model_dir = PathBuf::from(args.next().context("missing --model-dir")?)
            }
            "--pack-rlxp" => {
                pack_rlxp_out =
                    Some(PathBuf::from(args.next().context("missing --pack-rlxp PATH")?))
            }
            "--pack-gguf" => {
                pack_gguf = Some(PathBuf::from(args.next().context("missing --pack-gguf PATH")?))
            }
            "--max-tokens" => {
                opts.max_new_tokens = args
                    .next()
                    .context("missing --max-tokens")?
                    .parse()
                    .context("--max-tokens")?
            }
            "--temperature" => {
                opts.temperature = args
                    .next()
                    .context("missing --temperature")?
                    .parse()
                    .context("--temperature")?
            }
            "--seed" => {
                opts.seed = args
                    .next()
                    .context("missing --seed")?
                    .parse()
                    .context("--seed")?
            }
            "--greedy" => opts.greedy = true,
            "--device" => {
                device = parse_device(&args.next().context("missing --device")?)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            "-h" | "--help" => {
                println!("Usage: rlx-soprano --text TEXT [--output FILE]");
                println!("  --model-dir DIR   default: {DEFAULT_LOCAL_DIR}");
                println!("  --pack-rlxp PATH  pack loose dir → {DEFAULT_RLXP_NAME} (or PATH)");
                println!("  --pack-gguf PATH  pack loose dir → {DEFAULT_GGUF_NAME} (legacy)");
                println!("  --max-tokens N    AR steps (default 256)");
                println!("  --device NAME     cpu|metal|mlx|wgpu|coreml|cuda");
                println!("  --greedy          argmax sampling");
                return Ok(());
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    if let Some(out) = pack_rlxp_out {
        let report = pack_rlxp(&model_dir, &out)?;
        println!(
            "packed {} ({} bytes, {} blobs)",
            report.path.display(),
            report.bytes,
            report.blob_count
        );
        return Ok(());
    }
    if let Some(out) = pack_gguf {
        let report = pack_directory(&model_dir, &out)?;
        println!(
            "packed {} ({} bytes, {} text KV, {} blobs)",
            report.path.display(),
            report.bytes,
            report.file_kv,
            report.blob_count
        );
        return Ok(());
    }

    let text = text.context("--text is required (or use --pack-rlxp)")?;

    let t0 = Instant::now();
    let model = NativeSoprano::open(&model_dir, device)
        .with_context(|| format!("open Soprano at {}", model_dir.display()))?;
    eprintln!("[load {:?}] device={device:?}", t0.elapsed());

    let t1 = Instant::now();
    let pcm = model.synthesize(&text, &opts)?;
    let secs = pcm.len() as f32 / model.sample_rate() as f32;
    eprintln!(
        "[synth {:?}] {} samples = {:.2}s @ {}Hz, peak {:.3}",
        t1.elapsed(),
        pcm.len(),
        secs,
        model.sample_rate(),
        peak_amplitude(&pcm)
    );
    NativeSoprano::write_wav(&pcm, &output, model.sample_rate())?;
    println!("Wrote {} samples to {}", pcm.len(), output.display());
    Ok(())
}
