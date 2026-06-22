// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

// Whisper greedy pipeline stage timings (encoder / cross / prefill / decode).
//
// ```bash
// just fetch-whisper-base
// just bench-whisper -- --device metal
// ```

use anyhow::Context;
use rlx_cli::parse_device;
use rlx_models::whisper::{WhisperRunner, jfk_wav_path, load_wav_mono_f32};
use rlx_runtime::{Device, is_available};
use std::env;
use std::path::{Path, PathBuf};

fn default_model_dir() -> PathBuf {
    env::var("RLX_WHISPER_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/whisper-tiny"))
}

fn default_wav() -> PathBuf {
    jfk_wav_path()
}

#[allow(clippy::vec_init_then_push)]
fn all_backend_devices() -> Vec<Device> {
    let mut out = Vec::new();
    #[cfg(feature = "cuda")]
    out.push(Device::Cuda);
    #[cfg(feature = "metal")]
    out.push(Device::Metal);
    #[cfg(feature = "mlx")]
    out.push(Device::Mlx);
    #[cfg(feature = "rocm")]
    out.push(Device::Rocm);
    #[cfg(feature = "gpu")]
    out.push(Device::Gpu);
    #[cfg(feature = "vulkan")]
    out.push(Device::Vulkan);
    out.push(Device::Cpu);
    out
}

fn print_report(device: Device, report: &rlx_whisper::WhisperBenchReport, transcript: &str) {
    println!(
        "device={device:?} encode_ms={:.2} cross_ms={:.2} prefill_ms={:.2} decode_ms={:.2} \
         greedy_ms={:.2} decode_steps={}",
        report.encode_ms,
        report.cross_ms,
        report.prefill_ms,
        report.decode_ms,
        report.greedy_ms,
        report.decode_steps,
    );
    println!("  transcript={transcript:?}");
}

fn run_one(
    model_dir: &Path,
    pcm: &[f32],
    device: Device,
    decode_steps: usize,
    warmup: usize,
    runs: usize,
    precision: bool,
) -> anyhow::Result<()> {
    if !is_available(device) {
        eprintln!("skip: {device:?} not available");
        return Ok(());
    }
    let weights = model_dir.join("model.safetensors");
    anyhow::ensure!(weights.is_file(), "missing {}", weights.display());
    let mut builder = WhisperRunner::builder()
        .weights(&weights)
        .config_path(model_dir.join("config.json"))
        .tokenizer_path(model_dir.join("tokenizer.json"))
        .device(device)
        .language("en");
    if precision {
        builder = builder.activation_dtype(rlx_ir::DType::F32);
    }
    let mut runner = builder.build()?;
    for _ in 0..runs {
        let (report, transcript) = runner.bench_greedy_pipeline(pcm, decode_steps, warmup)?;
        print_report(device, &report, &transcript);
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = env::args().skip(1).filter(|a| a != "--").collect();
    let model_dir = match args.first() {
        Some(p) if !p.starts_with('-') => PathBuf::from(args.remove(0)),
        _ => default_model_dir(),
    };
    let mut device = Device::Cpu;
    let mut wav = default_wav();
    let mut decode_steps = 64usize;
    let mut warmup = 1usize;
    let mut runs = 1usize;
    let mut all_backends = false;
    let mut precision = false;

    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--device" => device = parse_device(&it.next().context("--device")?)?,
            other if other.starts_with("--device=") => {
                device = parse_device(other.trim_start_matches("--device="))?;
            }
            "--wav" => wav = PathBuf::from(it.next().context("--wav")?),
            "--decode-steps" => decode_steps = it.next().context("value")?.parse()?,
            "--warmup" => warmup = it.next().context("value")?.parse()?,
            "--runs" => runs = it.next().context("value")?.parse()?,
            "--all-backends" => all_backends = true,
            "--precision" => precision = true,
            "--help" | "-h" => {
                eprintln!(
                    "whisper_bench [MODEL_DIR] [--device NAME] [--wav PATH] [--decode-steps N] \
                     [--warmup N] [--runs N] [--all-backends] [--precision]"
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown flag: {other}"),
        }
    }

    anyhow::ensure!(
        model_dir.is_dir(),
        "model dir not found: {}",
        model_dir.display()
    );

    let pcm = if wav.is_file() {
        load_wav_mono_f32(&wav)?
    } else {
        eprintln!("wav not found ({}); using 10 s silence", wav.display());
        vec![0.0f32; 16_000 * 10]
    };

    if all_backends {
        for d in all_backend_devices() {
            println!("--- {d:?} ---");
            run_one(&model_dir, &pcm, d, decode_steps, warmup, runs, precision)?;
        }
    } else {
        run_one(
            &model_dir,
            &pcm,
            device,
            decode_steps,
            warmup,
            runs,
            precision,
        )?;
    }
    Ok(())
}
