// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! `rlx-zonos` — Zonos v0.1 transformer TTS CLI.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rlx_runtime::{Device, parse_device};
use rlx_zonos::{
    DEFAULT_DAC_DIR, DEFAULT_LOCAL_DIR, InferOpts, NativeZonos, load_speaker_emb, peak_amplitude,
};

fn main() -> Result<()> {
    let mut text = String::from("Hello from Zonos.");
    let mut output = PathBuf::from("zonos.wav");
    let mut model_dir = PathBuf::from(DEFAULT_LOCAL_DIR);
    let mut dac_dir = PathBuf::from(DEFAULT_DAC_DIR);
    let mut device = Device::Cpu;
    let mut opts = InferOpts::default();
    let mut phonemes_only = false;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--text" => text = args.next().context("--text value")?,
            "--output" | "-o" => output = PathBuf::from(args.next().context("--output")?),
            "--model-dir" => model_dir = PathBuf::from(args.next().context("--model-dir")?),
            "--dac-dir" => dac_dir = PathBuf::from(args.next().context("--dac-dir")?),
            "--device" => {
                device = parse_device(&args.next().context("--device")?)?;
            }
            "--max-tokens" => {
                opts.max_new_tokens = Some(
                    args.next()
                        .context("--max-tokens")?
                        .parse()
                        .context("parse --max-tokens")?,
                );
            }
            "--speaking-rate" => {
                opts.speaking_rate = args
                    .next()
                    .context("--speaking-rate")?
                    .parse()
                    .context("parse --speaking-rate")?;
            }
            "--cfg" => {
                opts.cfg_scale = args
                    .next()
                    .context("--cfg")?
                    .parse()
                    .context("parse --cfg")?;
            }
            "--min-p" => {
                opts.min_p = args
                    .next()
                    .context("--min-p")?
                    .parse()
                    .context("parse --min-p")?;
            }
            "--temperature" | "--temp" => {
                opts.temperature = args
                    .next()
                    .context("--temperature")?
                    .parse()
                    .context("parse --temperature")?;
            }
            "--seed" => {
                opts.seed = args
                    .next()
                    .context("--seed")?
                    .parse()
                    .context("parse --seed")?;
            }
            "--speaker-emb" => {
                let path = PathBuf::from(args.next().context("--speaker-emb path")?);
                opts.speaker = Some(load_speaker_emb(&path)?);
            }
            "--greedy" => opts.greedy = true,
            "--sample" => opts.greedy = false,
            "--phonemes" => phonemes_only = true,
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => bail!("unknown arg {other} (try --help)"),
        }
    }

    if !model_dir.join("model.safetensors").is_file() {
        bail!(
            "missing Zonos weights under {} — run `just fetch-zonos`",
            model_dir.display()
        );
    }
    if !dac_dir.join("model.safetensors").is_file() {
        bail!(
            "missing DAC under {} — run `just fetch-parler-dac`",
            dac_dir.display()
        );
    }

    let t0 = Instant::now();
    let model = NativeZonos::open(&model_dir, &dac_dir, device)?;
    let ids = model.encode_text(&text)?;
    println!(
        "device={device:?} phonemes={} (ids={:?}…)",
        ids.len(),
        &ids[..ids.len().min(12)]
    );
    if phonemes_only {
        println!(
            "ok phonemes-only in {:.0}ms",
            t0.elapsed().as_secs_f64() * 1000.0
        );
        return Ok(());
    }

    let pcm = model.synthesize(&text, &opts)?;
    NativeZonos::write_wav(&pcm, &output, model.sample_rate())?;
    println!(
        "wrote {} ({:.2}s audio, peak={:.3}) in {:.0}ms",
        output.display(),
        pcm.len() as f64 / model.sample_rate() as f64,
        peak_amplitude(&pcm),
        t0.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

fn print_help() {
    println!(
        "\
rlx-zonos — Zonos v0.1 transformer TTS (44.1 kHz DAC)

  rlx-zonos --text \"Hello.\" [--device cpu|metal|mlx] [--output out.wav]

Options:
  --text <str>            Text to speak
  --output / -o <path>    WAV path (default: zonos.wav)
  --model-dir <DIR>       Zonos weights (default: {DEFAULT_LOCAL_DIR})
  --dac-dir <DIR>         Descript DAC 44kHz (default: {DEFAULT_DAC_DIR})
  --device <name>         cpu / metal / mlx / …
  --max-tokens <n>        AR budget (omit = adaptive from phoneme length)
  --speaking-rate <f>     Conditioner dial (default 15)
  --cfg <f>               Classifier-free guidance scale (default 2)
  --greedy                Argmax codes (short prompts; can mush long endings)
  --sample                min_p multinomial sampling (default; matches Zyphra)
  --min-p <f>             Nucleus floor for sampling (default 0.1)
  --temperature <f>       Softmax temperature for sampling (default 1)
  --seed <u64>            RNG seed for sampling
  --speaker-emb <path>    128×f32 LE binary (512 bytes); else uncond speaker
  --phonemes              Only run espeak → ids (no synth)
"
    );
}
