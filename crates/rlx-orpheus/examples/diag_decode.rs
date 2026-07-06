// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
//! Diagnostic: generate the same utterance under several decode configs from a
//! single loaded backbone, write a WAV per config for external Whisper checks.

use anyhow::Result;
use rlx_orpheus::{BackboneLoadOptions, GenerationConfig, OrpheusTts};
use std::path::PathBuf;

fn write_wav(path: &str, samples: &[f32], sr: u32) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).expect("wav");
    for &s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0).round() as i16)
            .unwrap();
    }
    w.finalize().unwrap();
}

fn main() -> Result<()> {
    let gguf = PathBuf::from(
        std::env::var("ORPHEUS_GGUF_PATH")
            .unwrap_or_else(|_| "/tmp/rlx-weights/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf".into()),
    );
    let snac = PathBuf::from(
        std::env::var("ORPHEUS_SNAC_PATH")
            .unwrap_or_else(|_| "/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors".into()),
    );
    let dev = std::env::var("ORPHEUS_DIAG_DEVICE").unwrap_or_else(|_| "metal".into());
    let device = rlx_cli::parse_device(&dev)?;
    let text =
        std::env::var("ORPHEUS_DIAG_TEXT").unwrap_or_else(|_| "The weather is nice today.".into());
    let voice = std::env::var("ORPHEUS_DIAG_VOICE").unwrap_or_else(|_| "tara".into());
    let max_tokens: u32 = std::env::var("ORPHEUS_DIAG_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(196);

    eprintln!("[diag] device={device:?} text={text:?} voice={voice}");
    let tts = OrpheusTts::load_on_with_device(
        &gguf,
        &snac,
        device,
        BackboneLoadOptions::for_tts(device),
    )?;

    let base = GenerationConfig {
        max_new_tokens: max_tokens,
        seed: 42,
        ..GenerationConfig::default()
    };
    let configs: Vec<(&str, GenerationConfig)> = vec![
        (
            "greedy_rep13",
            GenerationConfig {
                greedy: true,
                repetition_penalty: 1.3,
                ..base.clone()
            },
        ),
        (
            "greedy_rep10",
            GenerationConfig {
                greedy: true,
                repetition_penalty: 1.0,
                ..base.clone()
            },
        ),
        (
            "sampling",
            GenerationConfig {
                greedy: false,
                ..base.clone()
            },
        ),
    ];

    let mut tts = tts;
    for (name, cfg) in configs {
        tts.config = cfg.clone();
        let out = tts.synthesize(&text, Some(&voice))?;
        let path = format!("/tmp/diag_{name}.wav");
        write_wav(&path, &out.samples, out.sample_rate);
        let peak = out.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        eprintln!(
            "[diag] {name}: codes={} samples={} ({:.2}s) peak={peak:.3} greedy={} rep={} -> {path}",
            out.code_count,
            out.samples.len(),
            out.samples.len() as f64 / out.sample_rate as f64,
            cfg.greedy,
            cfg.repetition_penalty,
        );
        // First 21 codes for eyeballing structure.
        eprintln!(
            "[diag] {name} codes[..21]={:?}",
            &out.codes[..out.codes.len().min(21)]
        );
    }
    let _ = device;
    Ok(())
}
