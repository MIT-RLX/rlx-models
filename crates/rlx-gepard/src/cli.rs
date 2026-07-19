// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use crate::synthesis::GepardSynthesizer;

const USAGE: &str = "\
rlx-gepard — Gepard autoregressive TTS (~556 M, Qwen3.5 + NanoCodec FSQ)

USAGE:
  rlx-gepard --weights <DIR>  --text <TEXT> [flags]
  rlx-gepard --weights <DIR>  --text <TEXT> --ref-audio <WAV> [flags]

FLAGS:
  --weights <DIR>        checkpoint directory (gepard_config.json + model.safetensors)
  --text <TEXT>          text to synthesise
  --voice <DESC>         voice description hint (default: '')
  --ref-audio <WAV>      reference audio for zero-shot voice cloning (optional)
  --out <PATH>           output WAV path (default: /tmp/gepard_out.wav)
  --max-frames <N>       max AR frames to generate (default: 2000)
  --stop-threshold <F>   stop head probability threshold (default: 0.5)
  --temperature <F>      codebook sampling temperature (default: 0.4; 0 with --greedy)
  --seed <N>             RNG seed for sampling (default: 54 short / 4 long paragraph)
  --greedy               argmax codebook heads
  --device <NAME>        cpu|metal|mlx|cuda|rocm (default: cpu)
  --help / -h
";

/// CLI entry point.
pub fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        print!("{USAGE}");
        return Ok(());
    }

    let get = |flag: &str| -> Option<&str> {
        args.windows(2)
            .find(|w| w[0] == flag)
            .map(|w| w[1].as_str())
    };

    let weights_dir = get("--weights")
        .map(PathBuf::from)
        .context("--weights <DIR> is required")?;

    let text = get("--text").unwrap_or("Hello from Gepard.");
    let voice = get("--voice").unwrap_or("");
    let ref_audio_path = get("--ref-audio").map(PathBuf::from);
    let out_path = get("--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/gepard_out.wav"));

    let device = get("--device").unwrap_or("cpu");
    let max_fr = get("--max-frames")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2000);
    let stop_th = get("--stop-threshold")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.5);
    let greedy = args.iter().any(|a| a == "--greedy");
    let temperature = get("--temperature")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.4);
    let seed = get("--seed")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or_else(|| crate::synthesis::default_seed_for_text(text));

    eprintln!("[rlx-gepard] weights:  {}", weights_dir.display());
    eprintln!("[rlx-gepard] text:     {text}");
    eprintln!("[rlx-gepard] voice:    {voice}");
    eprintln!("[rlx-gepard] device:   {device}");
    if let Some(p) = &ref_audio_path {
        eprintln!("[rlx-gepard] ref audio: {}", p.display());
    }
    eprintln!("[rlx-gepard] output:   {}", out_path.display());

    let opts = crate::synthesis::InferOpts {
        max_frames: max_fr,
        stop_threshold: stop_th,
        temperature,
        greedy,
        seed,
        ..Default::default()
    };
    let synth = GepardSynthesizer::with_device(&weights_dir, device)
        .with_context(|| format!("open Gepard at {}", weights_dir.display()))?
        .with_opts(opts);

    let ref_codes = if let Some(ref_path) = ref_audio_path {
        match load_ref_audio_codes(&ref_path) {
            Ok(codes) => {
                eprintln!(
                    "[rlx-gepard] loaded reference audio with {} frames",
                    codes.len() / 32
                );
                Some(codes)
            }
            Err(e) => {
                eprintln!("[rlx-gepard] warning: failed to load reference audio: {e}");
                None
            }
        }
    } else {
        None
    };

    let audio = synth
        .synthesize_with_reference(text, voice, ref_codes.as_deref())
        .context("synthesis failed")?;

    eprintln!(
        "[rlx-gepard] generated {} samples ({:.2} s) device={:?}",
        audio.len(),
        audio.len() as f32 / 22050.0,
        synth.device()
    );

    synth.write_wav(&audio, &out_path)?;
    eprintln!("[rlx-gepard] saved {}", out_path.display());
    Ok(())
}

fn load_ref_audio_codes(path: &Path) -> Result<Vec<u32>> {
    use std::fs;

    let codes_path = path.with_extension("codes");
    if codes_path.exists() {
        let bytes =
            fs::read(&codes_path).with_context(|| format!("read {}", codes_path.display()))?;
        let codes: Vec<u32> = bytes
            .chunks(4)
            .map(|chunk| {
                u32::from_le_bytes([
                    chunk.first().copied().unwrap_or(0),
                    chunk.get(1).copied().unwrap_or(0),
                    chunk.get(2).copied().unwrap_or(0),
                    chunk.get(3).copied().unwrap_or(0),
                ])
            })
            .collect();
        return Ok(codes);
    }

    if path.extension().and_then(|s| s.to_str()) == Some("wav") {
        anyhow::bail!(
            "WAV reference audio requires external encoder (e.g., Python script with NanoCodec encoder). \
            Please pre-encode to .codes file: {}",
            codes_path.display()
        );
    }

    anyhow::bail!("unsupported reference audio format: {}", path.display())
}
