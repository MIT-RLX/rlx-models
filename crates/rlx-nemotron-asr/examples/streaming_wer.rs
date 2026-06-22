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

//! Streaming-vs-batch accuracy/latency bench for Nemotron ASR, in the spirit
//! of arXiv:2604.14493: report per-configuration WER, RTFx (audio ÷ wall),
//! and the batch-to-stream factor (BSF = streaming WER ÷ batch WER).
//!
//! The model + audio are supplied via environment variables so the example
//! compiles and skips cleanly when assets are absent:
//!
//!   RLX_NEMOTRON_NEMO      path to the `.nemo` checkpoint        (required)
//!   RLX_NEMOTRON_WAV       path to a mono WAV clip               (required)
//!   RLX_NEMOTRON_REFERENCE path to a reference transcript (.txt) (optional)
//!   RLX_NEMOTRON_DEVICE    cpu | metal | mlx | cuda  (default cpu)
//!   RLX_NEMOTRON_CHUNKS    comma list of chunk seconds (default "1,2,4")
//!
//! Run, e.g.:
//!   RLX_NEMOTRON_NEMO=model.nemo RLX_NEMOTRON_WAV=clip.wav \
//!   RLX_NEMOTRON_REFERENCE=clip.txt \
//!   cargo run -p rlx-nemotron-asr --example streaming_wer --release
//!
//! NOTE on faithfulness: the encoder currently runs full (unmasked) attention
//! — cache-aware left-context streaming is not yet wired in the runner. So the
//! "streaming" rows here feed the audio as independent, non-overlapping chunks
//! (no cross-chunk cache). The resulting BSF is therefore a *conservative
//! upper bound* on degradation; once cache-aware streaming lands, the same
//! harness measures the improved number.

use std::time::Instant;

use anyhow::Result;
use rlx_cli::parse_standard_device;
use rlx_core::asr_metrics::{batch_to_stream_factor, rtfx, word_error_rate};
use rlx_nemotron_asr::{NemotronAsr, wav};

/// One row of the report (batch or a chunked-streaming configuration).
struct Row {
    label: String,
    wer: Option<f64>,
    rtfx: f64,
    bsf: Option<f64>,
    wall_s: f64,
}

fn main() -> Result<()> {
    let Some(nemo) = env_path("RLX_NEMOTRON_NEMO") else {
        return skip("RLX_NEMOTRON_NEMO is unset (path to a .nemo checkpoint)");
    };
    let Some(wav_path) = env_path("RLX_NEMOTRON_WAV") else {
        return skip("RLX_NEMOTRON_WAV is unset (path to a mono WAV clip)");
    };
    if !nemo.is_file() {
        return skip(&format!("RLX_NEMOTRON_NEMO not found: {}", nemo.display()));
    }
    if !wav_path.is_file() {
        return skip(&format!(
            "RLX_NEMOTRON_WAV not found: {}",
            wav_path.display()
        ));
    }

    let device_s = std::env::var("RLX_NEMOTRON_DEVICE").unwrap_or_else(|_| "cpu".into());
    let device = parse_standard_device("nemotron-asr", &device_s)?;
    let chunks = parse_chunks(&std::env::var("RLX_NEMOTRON_CHUNKS").unwrap_or_default());
    let reference = std::env::var("RLX_NEMOTRON_REFERENCE")
        .ok()
        .map(std::fs::read_to_string)
        .transpose()?
        .map(|s| s.trim().to_string());

    // ── load model + audio ───────────────────────────────────────────────
    let asr = NemotronAsr::open(&nemo, device)?;
    let sr = asr.config().sample_rate as u32;
    let bytes = std::fs::read(&wav_path)?;
    let parsed = wav::parse(&bytes)?;
    let pcm = wav::resample(&parsed.samples, parsed.sample_rate, sr);
    let audio_s = pcm.len() as f64 / sr as f64;

    eprintln!(
        "[streaming_wer] device={device:?} sr={sr}Hz audio={audio_s:.2}s \
         reference={} chunks={chunks:?}",
        if reference.is_some() { "yes" } else { "none" }
    );

    // ── batch (full-utterance) baseline ──────────────────────────────────
    let t0 = Instant::now();
    let batch_text = asr.transcribe(&pcm)?;
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
        let chunk_samples = (chunk_s * sr as f64).round() as usize;
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
            // Skip windows too short to survive 8× subsampling.
            if window.len() < (0.1 * sr as f64) as usize {
                continue;
            }
            match asr.transcribe(window) {
                Ok(t) if !t.trim().is_empty() => pieces.push(t.trim().to_string()),
                Ok(_) => {}
                Err(e) => eprintln!(
                    "  [warn] chunk @ {:.1}s failed: {e}",
                    start as f64 / sr as f64
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
        println!("\n(no RLX_NEMOTRON_REFERENCE set — WER/BSF omitted, timing only)");
    }
    println!(
        "\nstreaming rows = independent non-overlapping chunks (no cache-aware \
         left context yet); BSF is a conservative upper bound."
    );
}

fn env_path(key: &str) -> Option<std::path::PathBuf> {
    std::env::var(key).ok().map(std::path::PathBuf::from)
}

/// Parse a comma list of chunk seconds; defaults to `[1, 2, 4]` when empty.
fn parse_chunks(s: &str) -> Vec<f64> {
    let parsed: Vec<f64> = s
        .split(',')
        .filter_map(|t| t.trim().parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .collect();
    if parsed.is_empty() {
        vec![1.0, 2.0, 4.0]
    } else {
        parsed
    }
}

fn skip(reason: &str) -> Result<()> {
    eprintln!("[streaming_wer] skipped: {reason}");
    eprintln!(
        "  set RLX_NEMOTRON_NEMO, RLX_NEMOTRON_WAV (and optionally \
         RLX_NEMOTRON_REFERENCE) to run."
    );
    Ok(())
}
