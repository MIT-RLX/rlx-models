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

//! Unified ASR benchmark driver shared by every rlx ASR crate.
//!
//! Every ASR example (`bench_asr.rs`) wires its model + device into
//! [`run_asr_bench`], which drives a batch baseline plus a chunked-streaming
//! sweep and reports the **same** metrics for every model so results are
//! directly comparable:
//!
//! - **WER** / **CER** — word / character error rate vs a reference transcript
//!   (via [`crate::asr_metrics`]).
//! - **RTFx** — real-time factor (audio seconds ÷ wall seconds); `>1` is
//!   faster than real time.
//! - **BSF** — batch-to-stream factor (`streaming WER ÷ batch WER`); `1.0` =
//!   no accuracy loss from chunking, `>1` quantifies degradation.
//! - **peak RSS** — process peak resident memory (`getrusage`), so the same
//!   run also captures the memory footprint.
//!
//! Each row prints a machine-readable line:
//! `ASRBENCH crate=<c> device=<d> config=<label> wer=<%|na> cer=<%|na>
//!  rtfx=<x> bsf=<x|na> wall_s=<x> rss_mb=<n>`
//! plus a human-readable table.

use std::path::Path;
use std::time::Instant;

use anyhow::{Result, bail};

use crate::asr_metrics::{batch_to_stream_factor, character_error_rate, rtfx, word_error_rate};

/// Canonical sample rate for the shared bench clip (all rlx ASR models accept
/// 16 kHz mono PCM).
pub const TARGET_SR: u32 = 16_000;

/// Bench configuration shared across crates.
#[derive(Clone, Debug)]
pub struct AsrBenchConfig {
    /// Streaming chunk lengths in seconds (e.g. `[4.0, 8.0]`). Empty = batch only.
    pub chunks: Vec<f64>,
    /// Reference transcript; when `None`, WER/CER/BSF are omitted (timing only).
    pub reference: Option<String>,
    /// Minimum chunk fraction (of `TARGET_SR`) to bother transcribing.
    pub min_chunk_frac: f64,
}

impl Default for AsrBenchConfig {
    fn default() -> Self {
        Self {
            chunks: vec![4.0, 8.0],
            reference: None,
            min_chunk_frac: 0.1,
        }
    }
}

/// One bench row (batch or a streaming chunk size).
#[derive(Clone, Debug)]
pub struct AsrBenchRow {
    pub label: String,
    pub wer: Option<f64>,
    pub cer: Option<f64>,
    pub rtfx: f64,
    pub bsf: Option<f64>,
    pub wall_s: f64,
    pub rss_mb: u64,
    pub text: String,
}

/// Process peak resident set size in MB via `getrusage(RUSAGE_SELF)`.
/// Returns 0 if unavailable.
pub fn peak_rss_mb() -> u64 {
    peak_rss_bytes() / (1024 * 1024)
}

#[cfg(unix)]
fn peak_rss_bytes() -> u64 {
    // SAFETY: getrusage only writes into the zeroed rusage we hand it.
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) != 0 {
            return 0;
        }
        let max = ru.ru_maxrss as u64;
        // macOS reports bytes; Linux/BSD report kilobytes.
        if cfg!(target_os = "macos") {
            max
        } else {
            max.saturating_mul(1024)
        }
    }
}

#[cfg(not(unix))]
fn peak_rss_bytes() -> u64 {
    0
}

/// Drive a batch baseline + chunked-streaming sweep over `pcm` (mono,
/// [`TARGET_SR`] Hz) using `transcribe`, emit uniform `ASRBENCH` lines and a
/// table, and return the rows. `transcribe(window) -> hypothesis text`.
pub fn run_asr_bench<F>(
    crate_name: &str,
    device: &str,
    pcm: &[f32],
    audio_seconds: f64,
    cfg: &AsrBenchConfig,
    mut transcribe: F,
) -> Result<Vec<AsrBenchRow>>
where
    F: FnMut(&[f32]) -> Result<String>,
{
    let reference = cfg.reference.as_deref().map(str::trim);
    let mut rows = Vec::new();

    // ── batch baseline ───────────────────────────────────────────────────
    let t0 = Instant::now();
    let batch_text = transcribe(pcm)?;
    let batch_wall = t0.elapsed().as_secs_f64();
    let batch_wer = reference.map(|r| word_error_rate(r, &batch_text));
    rows.push(AsrBenchRow {
        label: "batch".into(),
        wer: batch_wer,
        cer: reference.map(|r| character_error_rate(r, &batch_text)),
        rtfx: rtfx(audio_seconds, batch_wall),
        bsf: batch_wer.map(|_| 1.0),
        wall_s: batch_wall,
        rss_mb: peak_rss_mb(),
        text: batch_text,
    });

    // ── chunked streaming sweep ──────────────────────────────────────────
    let min_samples = (cfg.min_chunk_frac * TARGET_SR as f64) as usize;
    for chunk_s in &cfg.chunks {
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
            if window.len() < min_samples {
                continue;
            }
            match transcribe(window) {
                Ok(t) if !t.trim().is_empty() => pieces.push(t.trim().to_string()),
                Ok(_) => {}
                Err(e) => eprintln!(
                    "  [warn] {crate_name}/{device} chunk @ {:.1}s failed: {e}",
                    start as f64 / TARGET_SR as f64
                ),
            }
        }
        let wall = t0.elapsed().as_secs_f64();
        let stream_text = pieces.join(" ");
        let wer = reference.map(|r| word_error_rate(r, &stream_text));
        let bsf = match (wer, batch_wer) {
            (Some(s), Some(b)) => Some(batch_to_stream_factor(s, b)),
            _ => None,
        };
        rows.push(AsrBenchRow {
            label: format!("stream={chunk_s:.1}s"),
            wer,
            cer: reference.map(|r| character_error_rate(r, &stream_text)),
            rtfx: rtfx(audio_seconds, wall),
            bsf,
            wall_s: wall,
            rss_mb: peak_rss_mb(),
            text: stream_text,
        });
    }

    emit(crate_name, device, audio_seconds, &rows);
    Ok(rows)
}

fn emit(crate_name: &str, device: &str, audio_seconds: f64, rows: &[AsrBenchRow]) {
    let pct = |o: Option<f64>| o.map_or("na".into(), |v| format!("{:.2}", v * 100.0));
    let num = |o: Option<f64>| o.map_or("na".into(), |v| format!("{v:.3}"));
    // Hypothesis text dump for debugging backend divergence (set RLX_ASR_DEBUG_TEXT).
    if std::env::var_os("RLX_ASR_DEBUG_TEXT").is_some() {
        for r in rows {
            eprintln!("[hyp {crate_name}/{device}/{}] {:?}", r.label, r.text);
        }
    }
    // Machine-readable lines (grep `ASRBENCH`).
    for r in rows {
        println!(
            "ASRBENCH crate={crate_name} device={device} config={} wer={} cer={} rtfx={:.3} bsf={} wall_s={:.3} rss_mb={}",
            r.label,
            pct(r.wer),
            pct(r.cer),
            r.rtfx,
            num(r.bsf),
            r.wall_s,
            r.rss_mb,
        );
    }
    // Human-readable table.
    eprintln!(
        "\n[{crate_name}] device={device} audio={audio_seconds:.2}s  peak_rss={}MB",
        rows.iter().map(|r| r.rss_mb).max().unwrap_or(0)
    );
    eprintln!(
        "{:<14}  {:>7}  {:>7}  {:>7}  {:>6}  {:>8}",
        "config", "WER%", "CER%", "RTFx", "BSF", "wall_s"
    );
    eprintln!("{}", "-".repeat(60));
    for r in rows {
        eprintln!(
            "{:<14}  {:>7}  {:>7}  {:>7.2}  {:>6}  {:>8.3}",
            r.label,
            pct(r.wer),
            pct(r.cer),
            r.rtfx,
            num(r.bsf),
            r.wall_s,
        );
    }
}

// ── shared audio helpers ─────────────────────────────────────────────────

/// Parse an ASR bench device string. Accepts the standard rlx aliases
/// (`cpu|metal|mps|mlx|gpu|wgpu|cuda|rocm|vulkan`) **plus** `coreml`/`ane`
/// → [`rlx_runtime::Device::Ane`] (the CoreML/ANE backend), which the
/// workspace's `parse_standard_device` rejects as non-standard.
pub fn parse_device(s: &str) -> Result<rlx_runtime::Device> {
    use std::str::FromStr;
    match s.trim().to_ascii_lowercase().as_str() {
        "coreml" | "ane" | "neural-engine" => Ok(rlx_runtime::Device::Ane),
        other => rlx_runtime::Device::from_str(other)
            .map_err(|e| anyhow::anyhow!("unknown ASR device {other:?}: {e}")),
    }
}

/// Parse a comma list of chunk seconds; falls back to `[4.0, 8.0]` when empty.
pub fn parse_chunks(s: &str) -> Vec<f64> {
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

/// Minimal WAV reader (PCM16 / float32, mono or downmixed). Returns
/// `(samples, sample_rate)`.
pub fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
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
pub fn resample_linear(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
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

/// Load + resample a WAV to [`TARGET_SR`] mono, returning `(pcm, audio_seconds)`.
pub fn load_clip_16k(path: &Path) -> Result<(Vec<f32>, f64)> {
    let (samples, sr) = read_wav_mono(path)?;
    let pcm = resample_linear(&samples, sr, TARGET_SR);
    let audio_s = pcm.len() as f64 / TARGET_SR as f64;
    Ok((pcm, audio_s))
}
