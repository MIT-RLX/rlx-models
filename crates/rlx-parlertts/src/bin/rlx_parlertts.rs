// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! `rlx-parlertts` — Parler-TTS Mini v1 CLI.
//!
//! ```text
//! rlx-parlertts --text "Hello." --voice "A clear female voice." --output out.wav
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_parlertts::{
    DEFAULT_DAC_DIR, DEFAULT_DESCRIPTION, DEFAULT_LOCAL_DIR, InferOpts, NativeParler,
    peak_amplitude,
};
use rlx_runtime::{Device, parse_device};

fn main() -> Result<()> {
    let mut text: Option<String> = None;
    let mut voice = DEFAULT_DESCRIPTION.to_string();
    let mut output = PathBuf::from("parlertts.wav");
    let mut model_dir = PathBuf::from(DEFAULT_LOCAL_DIR);
    let mut dac_dir = PathBuf::from(DEFAULT_DAC_DIR);
    let mut device = Device::Cpu;
    let mut opts = InferOpts::default();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--text" => {
                text = Some(args.next().context("missing value for --text")?);
            }
            "--voice" | "--description" => {
                voice = args.next().context("missing value for --voice")?;
            }
            "--output" | "-o" => {
                output = PathBuf::from(args.next().context("missing value for --output")?);
            }
            "--model-dir" | "--weights" => {
                model_dir = PathBuf::from(args.next().context("missing value for --model-dir")?);
            }
            "--dac-dir" => {
                dac_dir = PathBuf::from(args.next().context("missing value for --dac-dir")?);
            }
            "--device" => {
                device = parse_device(&args.next().context("missing value for --device")?)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            "--max-steps" => {
                opts.max_steps = args
                    .next()
                    .context("missing value for --max-steps")?
                    .parse()
                    .context("--max-steps")?;
            }
            "--temperature" => {
                opts.temperature = args
                    .next()
                    .context("missing value for --temperature")?
                    .parse()
                    .context("--temperature")?;
            }
            "--top-k" => {
                opts.top_k = args
                    .next()
                    .context("missing value for --top-k")?
                    .parse()
                    .context("--top-k")?;
            }
            "--seed" => {
                opts.seed = args
                    .next()
                    .context("missing value for --seed")?
                    .parse()
                    .context("--seed")?;
            }
            "--greedy" => opts.greedy = true,
            "-h" | "--help" => {
                print_help();
                return Ok(());
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    let text = text.context("--text is required")?;
    if !model_dir.join("onnx/text_encoder.onnx").is_file() {
        anyhow::bail!(
            "missing Parler ONNX under {} (expected onnx/text_encoder.onnx + onnx/decoder.onnx)",
            model_dir.display()
        );
    }
    if !dac_dir.join("model.safetensors").is_file() {
        anyhow::bail!(
            "missing DAC weights under {} (expected model.safetensors + config.json)",
            dac_dir.display()
        );
    }

    let t0 = Instant::now();
    let tts = NativeParler::open(&model_dir, &dac_dir, device)
        .with_context(|| format!("open parler at {}", model_dir.display()))?;
    eprintln!("[load {:?}] device={device:?}", t0.elapsed());

    let t1 = Instant::now();
    let pcm = tts.synthesize(&text, &voice, &opts)?;
    let secs = pcm.len() as f32 / tts.sample_rate() as f32;
    let elapsed = t1.elapsed().as_secs_f32().max(1e-6);
    eprintln!(
        "[synth {:?}] {} samples = {:.2}s @ {}Hz, peak {:.3}, RTF {:.2}×",
        t1.elapsed(),
        pcm.len(),
        secs,
        tts.sample_rate(),
        peak_amplitude(&pcm),
        secs / elapsed
    );

    tts.write_wav(&pcm, &output)?;
    println!("Wrote {} samples to {}", pcm.len(), output.display());
    Ok(())
}

fn print_help() {
    println!("Usage: rlx-parlertts [options]");
    println!("  --text <TEXT>           Transcript to synthesize (required)");
    println!("  --voice <DESC>          Voice description (default: clear female)");
    println!("  --output <FILE>         Output WAV path (default: parlertts.wav)");
    println!("  --model-dir <DIR>       Parler ONNX dir (default: {DEFAULT_LOCAL_DIR})");
    println!("  --dac-dir <DIR>         Descript DAC dir (default: {DEFAULT_DAC_DIR})");
    println!("  --device <NAME>         cpu|metal|mlx|cuda|… (default: cpu)");
    println!("  --max-steps <N>         AR steps (default: 172)");
    println!("  --temperature <F>       Sampling temperature (default: 1.0)");
    println!("  --top-k <N>             Top-k sampling (default: 50)");
    println!("  --seed <U64>            RNG seed");
    println!("  --greedy                Deterministic argmax (ignore temp/top-k)");
}
