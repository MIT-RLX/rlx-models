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

//! Streaming-vs-batch accuracy/latency bench for Qwen3-ASR, mirroring the
//! Nemotron harness and arXiv:2604.14493: per-config WER, RTFx (audio ÷ wall),
//! and the batch-to-stream factor (BSF = streaming WER ÷ batch WER).
//!
//! Qwen3-ASR is an LLM-based, batch-oriented model — the paper finds such
//! models degrade substantially when chunked. The "streaming" rows here feed
//! the audio as independent, non-overlapping chunks (no cross-chunk state), so
//! the BSF is a conservative measure of that degradation.
//!
//!   RLX_QWEN3_MODEL_DIR    dir with weights/config/tokenizer       (required)
//!   RLX_QWEN3_WAV          path to a 16 kHz mono WAV clip           (required)
//!   RLX_QWEN3_REFERENCE    path to a reference transcript (.txt)    (optional)
//!   RLX_QWEN3_SYSTEM       system/context prompt text   (default empty)
//!   RLX_QWEN3_DEVICE       cpu | metal | mlx | cuda  (default cpu)
//!   RLX_QWEN3_CHUNKS       comma list of chunk seconds (default "4,8")
//!
//! Run, e.g.:
//!   RLX_QWEN3_MODEL_DIR=/models/qwen3-asr-0.6b RLX_QWEN3_WAV=clip.wav \
//!   RLX_QWEN3_REFERENCE=clip.txt \
//!   cargo run -p rlx-qwen3-asr --example streaming_wer --release

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, bail};
use rlx_cli::parse_standard_device;
use rlx_core::asr_metrics::{batch_to_stream_factor, rtfx, word_error_rate};
use rlx_qwen3_asr::AsrRunner;

/// Qwen3-ASR expects 16 kHz mono PCM.
const TARGET_SR: u32 = 16_000;

struct Row {
    label: String,
    wer: Option<f64>,
    rtfx: f64,
    bsf: Option<f64>,
    wall_s: f64,
}

fn main() -> Result<()> {
    let Some(dir) = env_path("RLX_QWEN3_MODEL_DIR") else {
        return skip("RLX_QWEN3_MODEL_DIR is unset (model dir)");
    };
    let Some(wav_path) = env_path("RLX_QWEN3_WAV") else {
        return skip("RLX_QWEN3_WAV is unset (16 kHz mono WAV)");
    };
    if !dir.is_dir() {
        return skip(&format!("RLX_QWEN3_MODEL_DIR not a dir: {}", dir.display()));
    }
    if !wav_path.is_file() {
        return skip(&format!("RLX_QWEN3_WAV not found: {}", wav_path.display()));
    }

    let device_s = std::env::var("RLX_QWEN3_DEVICE").unwrap_or_else(|_| "cpu".into());
    let device = parse_standard_device("qwen3-asr", &device_s)?;
    let system = std::env::var("RLX_QWEN3_SYSTEM").unwrap_or_default();
    let chunks = parse_chunks(&std::env::var("RLX_QWEN3_CHUNKS").unwrap_or_default());
    let reference = std::env::var("RLX_QWEN3_REFERENCE")
        .ok()
        .map(std::fs::read_to_string)
        .transpose()?
        .map(|s| s.trim().to_string());

    // The builder derives config.json + tokenizer from the weights' parent dir.
    let runner = AsrRunner::builder()
        .weights(dir.join("model.safetensors"))
        .device(device)
        .build()?;

    let (samples, src_sr) = read_wav_mono(&wav_path)?;
    let pcm = resample_linear(&samples, src_sr, TARGET_SR);
    let audio_s = pcm.len() as f64 / TARGET_SR as f64;

    eprintln!(
        "[streaming_wer] device={device:?} audio={audio_s:.2}s ({src_sr}->{TARGET_SR}Hz) \
         reference={} chunks={chunks:?}",
        if reference.is_some() { "yes" } else { "none" }
    );

    // ── batch baseline ────────────────────────────────────────────────────
    let t0 = Instant::now();
    let batch_text = runner.transcribe_pcm(&pcm, &system)?;
    let batch_wall = t0.elapsed().as_secs_f64();
    let batch_wer = reference
        .as_deref()
        .map(|r| word_error_rate(r, &batch_text));

    let mut rows = vec![Row {
        label: "batch".into(),
        wer: batch_wer,
        rtfx: rtfx(audio_s, batch_wall),
        bsf: batch_wer.map(|_| 1.0),
        wall_s: batch_wall,
    }];

    // ── chunked streaming sweep ──────────────────────────────────────────
    for chunk_s in &chunks {
        let chunk_samples = (chunk_s * TARGET_SR as f64).round() as usize;
        if chunk_samples == 0 {
            continue;
        }
        let t0 = Instant::now();
        let mut pieces: Vec<String> = Vec::new();
        let mut start = 0usize;
        while start < pcm.len() {
            let end = (start + chunk_samples).min(pcm.len());
            let window = &pcm[start..end];
            start = end;
            if window.len() < (0.1 * TARGET_SR as f64) as usize {
                continue;
            }
            match runner.transcribe_pcm(window, &system) {
                Ok(t) if !t.trim().is_empty() => pieces.push(t.trim().to_string()),
                Ok(_) => {}
                Err(e) => eprintln!(
                    "  [warn] chunk @ {:.1}s failed: {e}",
                    start as f64 / TARGET_SR as f64
                ),
            }
        }
        let wall = t0.elapsed().as_secs_f64();
        let stream_text = pieces.join(" ");
        let wer = reference
            .as_deref()
            .map(|r| word_error_rate(r, &stream_text));
        let bsf = match (wer, batch_wer) {
            (Some(s), Some(b)) => Some(batch_to_stream_factor(s, b)),
            _ => None,
        };
        rows.push(Row {
            label: format!("stream chunk={chunk_s:.1}s"),
            wer,
            rtfx: rtfx(audio_s, wall),
            bsf,
            wall_s: wall,
        });
    }

    print_report(&rows, reference.is_some());
    Ok(())
}

fn print_report(rows: &[Row], have_reference: bool) {
    println!();
    println!(
        "{:<18}  {:>7}  {:>7}  {:>6}  {:>8}",
        "config", "WER%", "RTFx", "BSF", "wall_s"
    );
    println!("{}", "-".repeat(54));
    for r in rows {
        let wer = match r.wer {
            Some(w) => format!("{:.2}", w * 100.0),
            None => "  n/a".into(),
        };
        let bsf = match r.bsf {
            Some(b) => format!("{b:.2}"),
            None => " n/a".into(),
        };
        println!(
            "{:<18}  {:>7}  {:>7.2}  {:>6}  {:>8.3}",
            r.label, wer, r.rtfx, bsf, r.wall_s
        );
    }
    if !have_reference {
        println!("\n(no RLX_QWEN3_REFERENCE set — WER/BSF omitted, timing only)");
    }
    println!(
        "\nstreaming rows = independent non-overlapping chunks (no cross-chunk \
         state); BSF measures chunking degradation of this batch model."
    );
}

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var(key).ok().map(PathBuf::from)
}

fn skip(reason: &str) -> Result<()> {
    eprintln!("[streaming_wer] skipped: {reason}");
    eprintln!(
        "  set RLX_QWEN3_MODEL_DIR and RLX_QWEN3_WAV (and optionally RLX_QWEN3_REFERENCE) to run."
    );
    Ok(())
}

fn parse_chunks(s: &str) -> Vec<f64> {
    let parsed: Vec<f64> = s
        .split(',')
        .filter_map(|t| t.trim().parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .collect();
    if parsed.is_empty() {
        vec![4.0, 8.0]
    } else {
        parsed
    }
}

// ── minimal WAV reader (PCM16 / float32, mono or downmixed) ───────────────
fn read_wav_mono(path: &PathBuf) -> Result<(Vec<f32>, u32)> {
    let b = std::fs::read(path)?;
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        bail!("not a RIFF/WAVE file: {}", path.display());
    }
    let u16le = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
    let u32le = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let mut pos = 12;
    let mut fmt: Option<(u16, u16, u32, u16)> = None;
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= b.len() {
        let size = u32le(pos + 4) as usize;
        let body = pos + 8;
        let end = (body + size).min(b.len());
        match &b[pos..pos + 4] {
            b"fmt " if end - body >= 16 => {
                fmt = Some((
                    u16le(body),
                    u16le(body + 2),
                    u32le(body + 4),
                    u16le(body + 14),
                ));
            }
            b"data" => data = Some(&b[body..end]),
            _ => {}
        }
        pos = body + size + (size & 1);
    }
    let (format, channels, rate, bits) = fmt.ok_or_else(|| anyhow::anyhow!("missing fmt chunk"))?;
    let data = data.ok_or_else(|| anyhow::anyhow!("missing data chunk"))?;
    let ch = channels.max(1) as usize;
    let mono: Vec<f32> = match (format, bits) {
        (1, 16) => data
            .chunks_exact(2 * ch)
            .map(|fr| {
                let s: i32 = (0..ch)
                    .map(|c| i16::from_le_bytes([fr[2 * c], fr[2 * c + 1]]) as i32)
                    .sum();
                (s as f32 / ch as f32) / 32768.0
            })
            .collect(),
        (3, 32) => data
            .chunks_exact(4 * ch)
            .map(|fr| {
                let s: f32 = (0..ch)
                    .map(|c| {
                        f32::from_le_bytes([fr[4 * c], fr[4 * c + 1], fr[4 * c + 2], fr[4 * c + 3]])
                    })
                    .sum();
                s / ch as f32
            })
            .collect(),
        _ => bail!("unsupported WAV format {format}/{bits}-bit (need PCM16 or float32)"),
    };
    Ok((mono, rate))
}

/// Linear-interpolation resample to `to` Hz.
fn resample_linear(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = to as f64 / from as f64;
    let out_len = ((samples.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 / ratio;
        let i0 = src.floor() as usize;
        let frac = (src - i0 as f64) as f32;
        let a = samples.get(i0).copied().unwrap_or(0.0);
        let bb = samples.get(i0 + 1).copied().unwrap_or(a);
        out.push(a + (bb - a) * frac);
    }
    out
}
