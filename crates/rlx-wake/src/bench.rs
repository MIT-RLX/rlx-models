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

//! Latency / RTF helpers for wake-word benches.

use std::time::Instant;

use crate::{SAMPLE_RATE_16K, WakeEngine, WakeStep, score_wav};

#[derive(Debug, Clone)]
pub struct BenchStats {
    pub device: String,
    pub engine: String,
    pub audio_s: f64,
    pub wall_s: f64,
    pub rtf: f64,
    pub mean_chunk_us: f64,
    pub p50_chunk_us: f64,
    pub p99_chunk_us: f64,
    pub steps: usize,
    pub peak_score: f32,
    pub mean_score: f32,
    pub fires: usize,
}

fn percentile_us(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Warm up + time `score_wav` on `pcm`, including per-chunk latency percentiles.
pub fn bench_engine<E: WakeEngine>(
    engine_name: &str,
    device_label: &str,
    eng: &mut E,
    pcm: &[f32],
    warmup: usize,
    iters: usize,
) -> anyhow::Result<BenchStats> {
    let hop = eng.config().chunk_samples.max(1);
    for _ in 0..warmup {
        eng.reset();
        let _ = score_wav(eng, pcm)?;
    }

    let mut chunk_us: Vec<f64> = Vec::new();
    let mut last: Vec<WakeStep> = Vec::new();
    let t0 = Instant::now();
    for _ in 0..iters {
        eng.reset();
        let mut steps = Vec::new();
        let mut i = 0usize;
        while i < pcm.len() {
            let end = (i + hop).min(pcm.len());
            let mut chunk = pcm[i..end].to_vec();
            if chunk.len() < hop {
                chunk.resize(hop, 0.0);
            }
            let c0 = Instant::now();
            let s = eng.push_pcm(&chunk)?;
            chunk_us.push(c0.elapsed().as_secs_f64() * 1e6);
            steps.extend(s);
            i += hop;
        }
        last = steps;
    }
    let wall = t0.elapsed().as_secs_f64() / iters.max(1) as f64;
    let audio_s = pcm.len() as f64 / SAMPLE_RATE_16K as f64;
    let rtf = wall / audio_s.max(1e-12);

    chunk_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean_chunk_us = if chunk_us.is_empty() {
        0.0
    } else {
        chunk_us.iter().sum::<f64>() / chunk_us.len() as f64
    };
    let p50_chunk_us = percentile_us(&chunk_us, 50.0);
    let p99_chunk_us = percentile_us(&chunk_us, 99.0);

    let peak = last.iter().map(|s| s.score).fold(0.0_f32, f32::max);
    let mean_score = if last.is_empty() {
        0.0
    } else {
        last.iter().map(|s| s.score).sum::<f32>() / last.len() as f32
    };
    let fires = last.iter().filter(|s| s.fired).count();
    Ok(BenchStats {
        device: device_label.into(),
        engine: engine_name.into(),
        audio_s,
        wall_s: wall,
        rtf,
        mean_chunk_us,
        p50_chunk_us,
        p99_chunk_us,
        steps: last.len(),
        peak_score: peak,
        mean_score,
        fires,
    })
}

pub fn print_bench_table(rows: &[BenchStats]) {
    println!(
        "{:<18} {:<8} {:>7} {:>9} {:>8} {:>9} {:>9} {:>9} {:>8} {:>8}",
        "engine",
        "device",
        "audio_s",
        "wall_ms",
        "RTF",
        "mean_us",
        "p50_us",
        "p99_us",
        "peak",
        "fires"
    );
    for r in rows {
        println!(
            "{:<18} {:<8} {:>7.3} {:>9.3} {:>8.4} {:>9.1} {:>9.1} {:>9.1} {:>8.4} {:>8}",
            r.engine,
            r.device,
            r.audio_s,
            r.wall_s * 1e3,
            r.rtf,
            r.mean_chunk_us,
            r.p50_chunk_us,
            r.p99_chunk_us,
            r.peak_score,
            r.fires
        );
    }
}
