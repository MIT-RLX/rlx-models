// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//! `rlx-moonshine` CLI.

use crate::config::{MoonshineConfig, SAMPLE_RATE};
use crate::runner::MoonshineRunner;
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::req;
use std::fs;
use std::path::PathBuf;

pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut wav: Option<PathBuf> = None;
    let mut pcm: Option<PathBuf> = None;
    let mut config: Option<PathBuf> = None;
    let mut device = "cpu".to_string();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" | "--model-dir" => {
                weights = Some(req(args, &mut i)?.into());
            }
            "--wav" => {
                wav = Some(req(args, &mut i)?.into());
            }
            "--pcm" => {
                pcm = Some(req(args, &mut i)?.into());
            }
            "--config" => {
                config = Some(req(args, &mut i)?.into());
            }
            "--device" => {
                device = req(args, &mut i)?;
            }
            "-h" | "--help" => {
                println!(
                    "rlx-moonshine --weights DIR|FILE (--wav audio.wav | --pcm raw_f32.pcm) [--device cpu|metal|mlx|cuda] [--config config.json]"
                );
                return Ok(());
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    let weights = weights.ok_or_else(|| anyhow!("--weights required"))?;
    let device = rlx_cli::parse_device(&device)?;

    let cfg = if let Some(c) = config {
        MoonshineConfig::from_file(&c)?
    } else if weights.is_dir() {
        MoonshineConfig::from_dir(&weights).unwrap_or_else(|_| MoonshineConfig::tiny())
    } else {
        weights
            .parent()
            .and_then(|d| MoonshineConfig::from_dir(d).ok())
            .unwrap_or_else(MoonshineConfig::tiny)
    };

    let samples = if let Some(w) = wav {
        load_wav_mono_f32(&w)?
    } else if let Some(p) = pcm {
        load_pcm_f32(&p)?
    } else {
        bail!("provide --wav or --pcm");
    };

    let mut runner = MoonshineRunner::builder()
        .weights(weights)
        .config(cfg)
        .device(device)
        .build()?;
    let text = runner.transcribe(&samples)?;
    println!("{text}");
    Ok(())
}

fn load_pcm_f32(path: &PathBuf) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if !bytes.len().is_multiple_of(4) {
        bail!("PCM file length not a multiple of 4");
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn load_wav_mono_f32(path: &PathBuf) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    parse_wav_mono_f32(&bytes)
}

fn parse_wav_mono_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() < 44 {
        bail!("wav too small");
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("not a RIFF/WAVE file");
    }
    let mut off = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None;
    let mut data_chunk: Option<&[u8]> = None;
    while off + 8 <= bytes.len() {
        let tag = &bytes[off..off + 4];
        let len = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()) as usize;
        off += 8;
        if off + len > bytes.len() {
            break;
        }
        match tag {
            b"fmt " => {
                if len < 16 {
                    bail!("wav fmt chunk too small");
                }
                let audio_format = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap());
                let channels = u16::from_le_bytes(bytes[off + 2..off + 4].try_into().unwrap());
                let sample_rate = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
                let bits_per_sample =
                    u16::from_le_bytes(bytes[off + 14..off + 16].try_into().unwrap());
                fmt = Some((audio_format, channels, sample_rate, bits_per_sample));
            }
            b"data" => data_chunk = Some(&bytes[off..off + len]),
            _ => {}
        }
        off += len + (len % 2);
    }
    let (audio_format, channels, sample_rate, bits) =
        fmt.ok_or_else(|| anyhow!("wav missing fmt"))?;
    let data = data_chunk.ok_or_else(|| anyhow!("wav missing data"))?;
    if audio_format != 1 && audio_format != 3 {
        bail!("unsupported wav format {audio_format}");
    }
    if channels != 1 {
        bail!("moonshine expects mono wav (got {channels} channels)");
    }
    if sample_rate != SAMPLE_RATE {
        bail!("moonshine expects {SAMPLE_RATE} Hz (got {sample_rate})");
    }
    match (audio_format, bits) {
        (1, 16) => {
            let mut out = Vec::with_capacity(data.len() / 2);
            for c in data.chunks_exact(2) {
                let s = i16::from_le_bytes([c[0], c[1]]);
                out.push(s as f32 / 32768.0);
            }
            Ok(out)
        }
        (3, 32) => {
            let mut out = Vec::with_capacity(data.len() / 4);
            for c in data.chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            Ok(out)
        }
        _ => bail!("unsupported wav pcm bits={bits} format={audio_format}"),
    }
}
