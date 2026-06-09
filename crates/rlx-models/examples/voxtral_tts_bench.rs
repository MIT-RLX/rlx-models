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

// Stage timing on one Voxtral-4B-TTS checkpoint (RLX_VOXTRAL_TTS_DIR).
//
// ```bash
// just fetch-voxtral-tts   # if needed
// just bench-voxtral-tts -- --device metal
// just bench-voxtral-tts -- --compare
// ```

use anyhow::Context;
use rlx_cli::parse_device;
use rlx_runtime::Device;
use rlx_voxtral_tts::speech_tokenizer::SpeechTokenizer;
use rlx_voxtral_tts::{
    GenerationConfig, VoxtralTtsOptions, VoxtralTtsRunner, VoxtralTtsRunnerBuilder,
};
use std::env;
use std::path::PathBuf;

fn default_model_dir() -> PathBuf {
    env::var("RLX_VOXTRAL_TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/voxtral/Voxtral-4B-TTS"))
}

fn run_bench(
    runner: &mut VoxtralTtsRunner,
    prompt: &[u32],
    voice: &str,
    gen_cfg: &GenerationConfig,
    options: &VoxtralTtsOptions,
    warmup: usize,
    runs: usize,
) -> anyhow::Result<()> {
    for _ in 0..warmup {
        let _ = runner.bench_native_profiled(prompt, voice, gen_cfg, options)?;
    }
    let mut last = None;
    for _ in 0..runs {
        last = Some(runner.bench_native_profiled(prompt, voice, gen_cfg, options)?);
    }
    if let Some(report) = last {
        report.print_line();
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
    let mut voice = "neutral_female".to_string();
    let mut text = "Hello.".to_string();
    let mut warmup = 1usize;
    let mut runs = 1usize;
    let mut compare = false;
    let mut eager_lm = false;
    let mut eager_acoustic = false;

    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--device" => device = parse_device(&it.next().context("--device")?)?,
            other if other.starts_with("--device=") => {
                device = parse_device(other.trim_start_matches("--device="))?;
            }
            "--voice" => voice = it.next().context("--voice")?,
            "--text" => text = it.next().context("--text")?,
            "--warmup" => warmup = it.next().context("value")?.parse()?,
            "--runs" => runs = it.next().context("value")?.parse()?,
            "--compare" => compare = true,
            "--eager-lm" => eager_lm = true,
            "--eager-acoustic" => eager_acoustic = true,
            "--help" | "-h" => {
                eprintln!(
                    "voxtral_tts_bench [MODEL_DIR] [--device NAME] [--voice NAME] [--text TEXT] \
                     [--warmup N] [--runs N] [--compare] [--eager-lm] [--eager-acoustic]"
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

    let tok = SpeechTokenizer::from_model_dir(&model_dir)?;
    let prompt = tok.encode_speech(&text, &voice)?;
    let gen_cfg = GenerationConfig::default();

    if compare {
        let configs = [
            ("compiled+compiled", false, false),
            ("eager_lm", true, false),
            ("eager_acoustic", false, true),
            ("eager+both", true, true),
        ];
        for (label, lm, acoustic) in configs {
            println!("--- {label} ---");
            let options = VoxtralTtsOptions {
                device,
                eager_lm: lm,
                eager_acoustic: acoustic,
            };
            let mut runner = VoxtralTtsRunnerBuilder::default()
                .model_dir(&model_dir)
                .options(options.clone())
                .build()?;
            run_bench(
                &mut runner,
                &prompt,
                &voice,
                &gen_cfg,
                &options,
                warmup,
                runs,
            )?;
        }
        return Ok(());
    }

    let options = VoxtralTtsOptions {
        device,
        eager_lm,
        eager_acoustic,
    };
    let mut runner = VoxtralTtsRunner::open_with_options(&model_dir, options.clone())?;
    run_bench(
        &mut runner,
        &prompt,
        &voice,
        &gen_cfg,
        &options,
        warmup,
        runs,
    )
}
