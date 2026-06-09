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

//! Voice-activity segmentation before ASR (energy gate; optional Silero-style smoothing).

use crate::audio::{SAMPLE_RATE, SpeechSegment};

const CHUNK_SAMPLES: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadKind {
    /// RMS energy gate with hysteresis (fast, no extra weights).
    Energy,
    /// Silero-like: short-window probability from smoothed energy (no ONNX weights).
    SileroLite,
}

#[derive(Debug, Clone)]
pub struct VadConfig {
    pub kind: VadKind,
    /// Speech probability threshold in `[0, 1]`.
    pub threshold: f32,
    /// Minimum speech region length (samples).
    pub min_speech_samples: usize,
    /// Padding around each region (samples).
    pub pad_samples: usize,
    /// Merge regions separated by less than this (samples).
    pub min_silence_samples: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            kind: VadKind::SileroLite,
            threshold: 0.5,
            min_speech_samples: SAMPLE_RATE / 5,
            pad_samples: SAMPLE_RATE / 10,
            min_silence_samples: SAMPLE_RATE / 5,
        }
    }
}

/// Segment `pcm` into speech regions (16 kHz mono).
pub fn segments_by_vad(cfg: &VadConfig, pcm: &[f32]) -> Vec<SpeechSegment> {
    if pcm.is_empty() {
        return Vec::new();
    }
    let probs = match cfg.kind {
        VadKind::Energy => energy_probs(pcm),
        VadKind::SileroLite => silero_lite_probs(pcm),
    };
    probs_to_segments(cfg, pcm.len(), &probs)
}

fn energy_probs(pcm: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    for chunk in pcm.chunks(CHUNK_SAMPLES) {
        let rms = (chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32).sqrt();
        out.push((rms * 10.0).min(1.0));
    }
    out
}

fn silero_lite_probs(pcm: &[f32]) -> Vec<f32> {
    let raw = energy_probs(pcm);
    let mut smooth = raw.clone();
    let alpha = 0.15f32;
    for i in 1..smooth.len() {
        smooth[i] = alpha * raw[i] + (1.0 - alpha) * smooth[i - 1];
    }
    for i in (0..smooth.len().saturating_sub(1)).rev() {
        smooth[i] = alpha * smooth[i] + (1.0 - alpha) * smooth[i + 1];
    }
    smooth
}

fn probs_to_segments(cfg: &VadConfig, n_samples: usize, probs: &[f32]) -> Vec<SpeechSegment> {
    let mut segments = Vec::new();
    let mut in_speech = false;
    let mut start_chunk = 0usize;
    for (ci, &p) in probs.iter().enumerate() {
        let speech = p >= cfg.threshold;
        if speech && !in_speech {
            in_speech = true;
            start_chunk = ci;
        } else if !speech && in_speech {
            push_segment(
                cfg,
                &mut segments,
                start_chunk * CHUNK_SAMPLES,
                ci * CHUNK_SAMPLES,
                n_samples,
            );
            in_speech = false;
        }
    }
    if in_speech {
        push_segment(
            cfg,
            &mut segments,
            start_chunk * CHUNK_SAMPLES,
            n_samples,
            n_samples,
        );
    }
    merge_close(cfg, segments)
}

fn push_segment(
    cfg: &VadConfig,
    out: &mut Vec<SpeechSegment>,
    start: usize,
    end: usize,
    n_samples: usize,
) {
    let s = start.saturating_sub(cfg.pad_samples);
    let e = (end + cfg.pad_samples).min(n_samples);
    if e.saturating_sub(s) < cfg.min_speech_samples {
        return;
    }
    out.push(SpeechSegment { start: s, end: e });
}

fn merge_close(cfg: &VadConfig, segs: Vec<SpeechSegment>) -> Vec<SpeechSegment> {
    if segs.len() < 2 {
        return segs;
    }
    let mut merged = vec![segs[0]];
    for seg in segs.into_iter().skip(1) {
        let last = merged.last_mut().unwrap();
        if seg.start.saturating_sub(last.end) <= cfg.min_silence_samples {
            last.end = seg.end;
        } else {
            merged.push(seg);
        }
    }
    merged
}
