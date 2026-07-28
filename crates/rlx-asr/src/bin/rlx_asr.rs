// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! `rlx-asr` CLI — transcribe WAV (`weights/asr/model.gguf`).
//!
//! ```text
//! rlx-asr transcribe [--dir DIR] --wav audio.wav
//! ```
//!
//! Encoder is a shaped stub until the folded Conformer path is wired in Rust;
//! AED / units / Hammer load from the GGUF. For folded CTC, use
//! `just asr-e2e-native`.

use anyhow::{Context, Result, bail};
use rlx_asr::pipeline::AsrSession;
use std::path::{Path, PathBuf};

fn usage() -> ! {
    eprintln!("usage:");
    eprintln!("  rlx-asr transcribe [--dir DIR] --wav audio.wav");
    eprintln!();
    eprintln!("env: RLX_ASR_DIR  RLX_ASR_TIMING=1  RLX_ASR_GGUF=path");
    eprintln!(
        "weights: just fetch-rlx-asr  →  weights/asr/model.gguf ({})",
        rlx_asr::HF_REPO
    );
    std::process::exit(2);
}

fn parse_args() -> Result<(PathBuf, PathBuf)> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 || a[1] != "transcribe" {
        usage();
    }
    let mut dir = std::env::var_os("RLX_ASR_DIR").map(PathBuf::from);
    let mut wav = None;
    let mut i = 2;
    while i < a.len() {
        match a[i].as_str() {
            "--dir" => {
                dir = Some(PathBuf::from(a.get(i + 1).context("--dir value")?));
                i += 2;
            }
            "--wav" => {
                wav = Some(PathBuf::from(a.get(i + 1).context("--wav value")?));
                i += 2;
            }
            "--device" => {
                i += 2;
            }
            _ => i += 1,
        }
    }
    let dir = dir.unwrap_or_else(rlx_asr::asr_dir);
    let wav = wav.context("--wav required")?;
    Ok((dir, wav))
}

fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("not a RIFF/WAVE file");
    }
    let mut off = 12usize;
    let mut sr = 16_000u32;
    let mut ch = 1u16;
    let mut bits = 16u16;
    let mut data = None;
    while off + 8 <= bytes.len() {
        let id = &bytes[off..off + 4];
        let sz = u32::from_le_bytes(bytes[off + 4..off + 8].try_into()?) as usize;
        off += 8;
        if id == b"fmt " && sz >= 16 {
            ch = u16::from_le_bytes(bytes[off + 2..off + 4].try_into()?);
            sr = u32::from_le_bytes(bytes[off + 4..off + 8].try_into()?);
            bits = u16::from_le_bytes(bytes[off + 14..off + 16].try_into()?);
        } else if id == b"data" {
            data = Some(&bytes[off..off + sz]);
            break;
        }
        off += sz;
    }
    let data = data.context("no data chunk")?;
    if bits != 16 {
        bail!("only 16-bit PCM wav supported");
    }
    let samples: Vec<i16> = data
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    let mono: Vec<f32> = if ch == 1 {
        samples.iter().map(|&s| s as f32 / 32768.0).collect()
    } else {
        samples
            .chunks_exact(ch as usize)
            .map(|f| f.iter().map(|&s| s as f32).sum::<f32>() / (ch as f32 * 32768.0))
            .collect()
    };
    Ok((mono, sr))
}

fn main() -> Result<()> {
    let (dir, wav) = parse_args()?;
    let (pcm, sr) = read_wav_mono(&wav)?;
    let mut asr = AsrSession::load(&dir)?;
    let t0 = std::time::Instant::now();
    let tr = asr.transcribe(&pcm, sr)?;
    if rlx_asr::env::timing() {
        eprintln!("transcribe {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
    }
    println!("{}", tr.text);
    Ok(())
}
