// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Orpheus zero-shot voice clone walkthrough.
//!
//! Encode a reference clip once, then synthesize arbitrary text in that voice
//! with the **pretrained** Orpheus checkpoint (in-context text + SNAC tokens).
//!
//! ## 1 — Encode reference audio (Python, one-time)
//!
//! ```bash
//! python3 -m venv /tmp/rlx-venv-orpheus
//! /tmp/rlx-venv-orpheus/bin/pip install snac torch safetensors soundfile
//!
//! python3 scripts/orpheus_encode_reference.py \
//!   --wav assets/jfk/jfk_voice_clone.wav \
//!   --transcript "Ask not what your country can do for you." \
//!   --out /tmp/jfk_orpheus_ref.json
//! ```
//!
//! ## 2 — Synthesize with RLX (needs pretrained GGUF + SNAC decoder)
//!
//! ```bash
//! just fetch-orpheus-snac
//! export ORPHEUS_SNAC_PATH=/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors
//!
//! cargo run --release -p rlx-orpheus --example voice_clone --features apple-silicon -- \
//!   --weights /path/to/orpheus-3b-0.1-pretrained.Q4_K_M.gguf \
//!   --ref-json /tmp/jfk_orpheus_ref.json \
//!   --target-text "I write my software in Rust because it is fast and safe." \
//!   --out /tmp/orpheus_clone.wav \
//!   --device metal
//! ```
//!
//! The finetune-prod GGUF (`just fetch-orpheus`) uses named voices (`tara`, …)
//! and is not trained for zero-shot cloning — use
//! [`canopylabs/orpheus-3b-0.1-pretrained`](https://huggingface.co/canopylabs/orpheus-3b-0.1-pretrained)
//! converted to GGUF for this example.

use anyhow::{Context, Result, bail};
use rlx_orpheus::{GenerationConfig, OrpheusTts, VoiceCloneReference};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn parse_args() -> Result<(PathBuf, PathBuf, String, PathBuf, Device)> {
    let mut weights = PathBuf::from(std::env::var("ORPHEUS_PRETRAINED_GGUF").unwrap_or_else(
        |_| "/tmp/rlx-weights/orpheus/orpheus-3b-0.1-pretrained.Q4_K_M.gguf".into(),
    ));
    let mut ref_json = PathBuf::from("/tmp/jfk_orpheus_ref.json");
    let mut target_text =
        "I write my software in Rust because it is fast, safe, and predictable.".to_string();
    let mut out_wav = PathBuf::from("/tmp/orpheus_voice_clone.wav");
    let mut device = Device::Metal;

    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        let take = |i: &mut usize, raw: &[String]| -> Result<String> {
            *i += 1;
            raw.get(*i)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing value for {}", raw[*i - 1]))
        };
        match raw[i].as_str() {
            "--weights" => weights = PathBuf::from(take(&mut i, &raw)?),
            "--ref-json" => ref_json = PathBuf::from(take(&mut i, &raw)?),
            "--target-text" => target_text = take(&mut i, &raw)?,
            "--out" | "--wav" => out_wav = PathBuf::from(take(&mut i, &raw)?),
            "--device" => {
                device = rlx_cli::parse_device(&take(&mut i, &raw)?)?;
            }
            "-h" | "--help" => {
                eprintln!(
                    "Usage: voice_clone \\
  --weights PRETRAINED.gguf \\
  --ref-json ref.json \\
  --target-text \"…\" \\
  --out out.wav \\
  [--device metal|cpu|mlx|…]"
                );
                std::process::exit(0);
            }
            other => bail!("unknown arg {other:?}"),
        }
        i += 1;
    }
    Ok((weights, ref_json, target_text, out_wav, device))
}

fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create wav {}", path.display()))?;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        writer.write_sample((clamped * 32767.0).round() as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

fn main() -> Result<()> {
    let (weights, ref_json, target_text, out_wav, device) = parse_args()?;

    if !weights.is_file() {
        bail!(
            "missing weights {} — set ORPHEUS_PRETRAINED_GGUF or pass --weights\n\
             Zero-shot clone needs the *pretrained* Orpheus GGUF, not the finetune-prod bundle from `just fetch-orpheus`.",
            weights.display()
        );
    }
    if !ref_json.is_file() {
        bail!(
            "missing reference {} — run scripts/orpheus_encode_reference.py first",
            ref_json.display()
        );
    }

    let reference = VoiceCloneReference::load_json(&ref_json)?;
    eprintln!("┌─ Orpheus voice clone ───────────────────────────────────────────");
    eprintln!("│ weights:  {}", weights.display());
    eprintln!(
        "│ ref:      {} ({} audio tokens)",
        ref_json.display(),
        reference.token_ids.len()
    );
    eprintln!("│ transcript: {:?}", reference.transcript);
    eprintln!("│ target:   {target_text:?}");
    eprintln!("│ device:   {device:?}");
    eprintln!("└────────────────────────────────────────────────────────────────\n");

    let t0 = Instant::now();
    let mut tts = OrpheusTts::load_with_env_decoder_on(&weights, device)?;
    eprintln!(
        "✓ loaded backbone + SNAC in {:.2}s",
        t0.elapsed().as_secs_f64()
    );

    tts.config = GenerationConfig {
        max_new_tokens: 1200,
        ..GenerationConfig::default()
    };

    let t = Instant::now();
    let result =
        tts.synthesize_voice_clone(&reference.transcript, &reference.token_ids, &target_text)?;
    eprintln!(
        "✓ synthesized {} codes -> {} samples ({:.2}s audio) in {:.2}s wall",
        result.code_count,
        result.samples.len(),
        result.samples.len() as f64 / result.sample_rate as f64,
        t.elapsed().as_secs_f64()
    );

    write_wav(&out_wav, &result.samples, result.sample_rate)?;
    eprintln!("\n wrote {}", out_wav.display());
    eprintln!(" listen:  afplay {}", out_wav.display());
    Ok(())
}
