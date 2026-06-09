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

// Stage breakdown on one Qwen3-TTS checkpoint (talker micro-benchmark).
//
// ```bash
// just fetch-qwen3-tts
// just bench-qwen3-tts -- --device metal
// ```

use anyhow::Context;
use rlx_cli::parse_device;
use rlx_models::qwen3_tts::Qwen3TtsRunnerBuilder;
use rlx_runtime::Device;
use std::env;
use std::path::PathBuf;

fn default_model_dir() -> PathBuf {
    env::var("RLX_QWEN3_TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-CustomVoice"))
}

fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = env::args().skip(1).filter(|a| a != "--").collect();
    let model_dir = match args.first() {
        Some(p) if !p.starts_with('-') => PathBuf::from(args.remove(0)),
        _ => default_model_dir(),
    };
    let mut device = Device::Cpu;
    let mut prefill_seq = 128usize;
    let mut decode_steps = 64usize;
    let mut warmup = 1usize;
    let mut runs = 3usize;

    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--device" => device = parse_device(&it.next().context("--device")?)?,
            other if other.starts_with("--device=") => {
                device = parse_device(other.trim_start_matches("--device="))?;
            }
            "--prefill-seq" => prefill_seq = it.next().context("value")?.parse()?,
            "--decode-steps" => decode_steps = it.next().context("value")?.parse()?,
            "--warmup" => warmup = it.next().context("value")?.parse()?,
            "--runs" => runs = it.next().context("value")?.parse()?,
            "--help" | "-h" => {
                eprintln!(
                    "qwen3_tts_bench [MODEL_DIR] [--device NAME] [--prefill-seq N] [--decode-steps N] [--warmup N] [--runs N]"
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

    let runner = Qwen3TtsRunnerBuilder::default()
        .model_dir(&model_dir)
        .device(device)
        .build()?;

    for _ in 0..warmup {
        let _ = runner.bench_talker_synthetic(prefill_seq, decode_steps)?;
    }
    let mut acc = Vec::new();
    for _ in 0..runs {
        acc.push(runner.bench_talker_synthetic(prefill_seq, decode_steps)?);
    }
    let last = acc.last().unwrap();
    last.print_line();
    Ok(())
}
