// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! `rlx-kittentts2` — KittenTTS 2 command line.
//!
//! ```text
//! rlx-kittentts2 --text "Hello there." --voice Bruno -o out.wav --device metal
//! rlx-kittentts2 --list-voices
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_kittentts2::pipeline::{GenerateOpts, KittenTts2, resolve_s3gen, write_wav};
use rlx_kittentts2::{Device, parse_device};

const DEFAULT_DIR: &str = "weights/tts/kitten-tts-2";

fn usage() {
    println!(
        "Usage: rlx-kittentts2 --text TEXT [options]\n\
         \n  --voice NAME          built-in voice (default: the repo's, Bruno)\n\
         \x20 --out, -o FILE        output wav (default kittentts2.wav)\n\
         \x20 --model-dir DIR       local KittenML/kitten-tts-2 (default {DEFAULT_DIR})\n\
         \x20 --s3gen FILE          chatterbox-turbo s3gen_meanflow.safetensors\n\
         \x20 --device NAME         cpu|metal|mlx|wgpu|cuda|… (default cpu)\n\
         \x20 --preset NAME         stable|expressive\n\
         \x20 --decoder NAME        default | student_w4 (4-bit one-step flow, smaller, lower fidelity)\n\
         \x20 --temperature F --top-k N --top-p F --min-p F\n\
         \x20 --max-tokens N        per-chunk budget (25 ≈ 1 s; default 1000)\n\
         \x20 --seed N              sampler seed (default 0)\n\
         \x20 --greedy              argmax decoding\n\
         \x20 --no-normalize        keep numbers/dates as written (use for non-English)\n\
         \x20 --no-reference-prompt drop the reference clip from the prompt\n\
         \x20 --reference WAV       clone a voice from 5–30 s of one speaker\n\
         \x20 --reference-text T    what that clip says (else Whisper transcribes it)\n\
         \x20 --whisper DIR         Whisper checkpoint for that (default: RLX_WHISPER_DIR,\n\
         \x20                       then .cache/whisper-large-v3, -small.en, -base.en)\n\
         \x20 --dump-tokens FILE    write each chunk's codec tokens as JSON\n\
         \x20 --list-voices         print the built-in voices and exit"
    );
}

fn main() -> Result<()> {
    let mut text: Option<String> = None;
    let mut out = PathBuf::from("kittentts2.wav");
    let mut dir = PathBuf::from(DEFAULT_DIR);
    let mut s3gen: Option<PathBuf> = None;
    let mut device = Device::Cpu;
    let mut opts = GenerateOpts::default();
    let mut list = false;
    let mut dump_tokens: Option<PathBuf> = None;
    let mut reference: Option<PathBuf> = None;
    let mut reference_text: Option<String> = None;
    let mut whisper: Option<PathBuf> = None;
    let mut decoder = rlx_kittentts2::pipeline::Decoder::Default;

    let mut args = std::env::args().skip(1);
    let next = |name: &str, args: &mut dyn Iterator<Item = String>| -> Result<String> {
        args.next()
            .with_context(|| format!("missing value for {name}"))
    };
    while let Some(a) = args.next() {
        match a.as_str() {
            "--text" => text = Some(next(&a, &mut args)?),
            "--voice" => opts.voice = Some(next(&a, &mut args)?),
            "--out" | "-o" | "--output" => out = PathBuf::from(next(&a, &mut args)?),
            "--model-dir" | "--weights" => dir = PathBuf::from(next(&a, &mut args)?),
            "--s3gen" => s3gen = Some(PathBuf::from(next(&a, &mut args)?)),
            "--device" => {
                device = parse_device(&next(&a, &mut args)?).map_err(|e| anyhow::anyhow!("{e}"))?
            }
            "--preset" => opts.preset = Some(next(&a, &mut args)?),
            "--temperature" => opts.temperature = Some(next(&a, &mut args)?.parse()?),
            "--top-k" => opts.top_k = Some(next(&a, &mut args)?.parse()?),
            "--top-p" => opts.top_p = Some(next(&a, &mut args)?.parse()?),
            "--min-p" => opts.min_p = Some(next(&a, &mut args)?.parse()?),
            "--max-tokens" => opts.max_new_tokens = next(&a, &mut args)?.parse()?,
            "--seed" => opts.seed = next(&a, &mut args)?.parse()?,
            "--greedy" => opts.greedy = true,
            "--no-normalize" => opts.normalize = false,
            "--no-reference-prompt" => opts.use_reference_prompt = Some(false),
            "--list-voices" => list = true,
            "--dump-tokens" => dump_tokens = Some(PathBuf::from(next(&a, &mut args)?)),
            "--reference" => reference = Some(PathBuf::from(next(&a, &mut args)?)),
            "--reference-text" => reference_text = Some(next(&a, &mut args)?),
            "--decoder" => decoder = next(&a, &mut args)?.parse().unwrap(),
            "--whisper" => whisper = Some(PathBuf::from(next(&a, &mut args)?)),
            "-h" | "--help" => {
                usage();
                return Ok(());
            }
            other if text.is_none() && !other.starts_with('-') => text = Some(other.to_string()),
            other => anyhow::bail!("unknown argument {other:?} (see --help)"),
        }
    }

    let s3gen = match s3gen {
        Some(p) => p,
        None => resolve_s3gen(&dir)?,
    };
    let t0 = Instant::now();
    let mut tts = KittenTts2::open_with_decoder(&dir, &s3gen, device, decoder)?;
    eprintln!(
        "[kittentts2] loaded in {:.2}s on {device:?}",
        t0.elapsed().as_secs_f32()
    );
    if list {
        for name in tts.voices().names() {
            println!("{name}");
        }
        return Ok(());
    }
    let text = text.context("no --text given (see --help)")?;

    let t0 = Instant::now();
    let voice = match reference {
        Some(wav) => {
            let t = Instant::now();
            let transcript = match reference_text {
                Some(t) => t,
                None => transcribe(&tts, &wav, whisper.as_deref())?,
            };
            let v = tts.clone_voice(&wav, &transcript)?;
            eprintln!(
                "[kittentts2] cloned {} ({} reference tokens) in {:.2}s",
                wav.display(),
                v.reference_tokens.len(),
                t.elapsed().as_secs_f32()
            );
            v
        }
        None => tts
            .voices()
            .load(opts.voice.as_deref().unwrap_or(&tts.config().default_voice))?,
    };
    let (wav, chunks) = tts.generate_with_voice(&text, &voice, &opts, |c| {
        eprintln!(
            "[kittentts2] chunk: {} audio tokens ({:?}) — {:?}",
            c.audio_tokens, c.stop, c.text
        )
    })?;
    let dt = t0.elapsed().as_secs_f32();
    let secs = wav.len() as f32 / tts.sample_rate() as f32;
    write_wav(&out, &wav, tts.sample_rate())?;
    if let Some(p) = dump_tokens {
        let all: Vec<&[u32]> = chunks.iter().map(|c| c.audio.as_slice()).collect();
        std::fs::write(&p, serde_json::to_string(&all)?)?;
    }
    eprintln!(
        "[kittentts2] {secs:.2}s of audio in {dt:.2}s (RTF {:.2}) → {}",
        dt / secs.max(1e-6),
        out.display()
    );
    Ok(())
}

/// Whisper checkpoint for transcribing a reference: `--whisper`, then
/// `RLX_WHISPER_DIR`, then the first of a few conventional `.cache` dirs.
#[cfg(feature = "transcribe")]
fn transcribe(
    tts: &KittenTts2,
    wav: &std::path::Path,
    dir: Option<&std::path::Path>,
) -> Result<String> {
    let ready = |d: &std::path::Path| {
        d.join("model.safetensors").is_file() && d.join("tokenizer.json").is_file()
    };
    let dir = match dir {
        Some(d) => d.to_path_buf(),
        None => rlx_ir::env::var_result("RLX_WHISPER_DIR")
            .map(PathBuf::from)
            .ok()
            .into_iter()
            .chain(
                ["whisper-large-v3", "whisper-large-v3-turbo", "whisper-small.en", "whisper-base.en"]
                    .iter()
                    .map(|n| PathBuf::from(".cache").join(n)),
            )
            .find(|d| ready(d))
            .context("no --reference-text, and no Whisper checkpoint (--whisper DIR) to transcribe the reference")?,
    };
    let t = Instant::now();
    let text = tts.transcribe_reference(wav, &dir, "en")?;
    eprintln!(
        "[kittentts2] reference transcript ({:.2}s): {text}",
        t.elapsed().as_secs_f32()
    );
    Ok(text)
}

#[cfg(not(feature = "transcribe"))]
fn transcribe(_: &KittenTts2, _: &std::path::Path, _: Option<&std::path::Path>) -> Result<String> {
    anyhow::bail!("--reference needs --reference-text (built without the `transcribe` feature)")
}
