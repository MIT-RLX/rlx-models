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

//! Speech region extraction from per-frame VAD probabilities.

use crate::SpeechSegment;

#[derive(Debug, Clone)]
pub struct SegmentParams {
    pub threshold: f32,
    pub neg_threshold: Option<f32>,
    pub min_speech_samples: usize,
    pub min_silence_samples: usize,
    pub speech_pad_samples: usize,
    pub max_speech_samples: usize,
}

impl Default for SegmentParams {
    fn default() -> Self {
        Self::silero()
    }
}

impl SegmentParams {
    /// Tuned for Earshot frame scores (256-sample hop @ 16 kHz).
    pub fn earshot() -> Self {
        let sr = 16_000;
        Self {
            threshold: 0.35,
            neg_threshold: Some(0.20),
            min_speech_samples: sr / 10,
            min_silence_samples: sr / 20,
            speech_pad_samples: sr * 30 / 1000,
            max_speech_samples: usize::MAX / 2,
        }
    }

    /// Matches Silero `get_speech_timestamps` defaults @ 16 kHz.
    pub fn silero() -> Self {
        let sr = 16_000;
        Self {
            threshold: 0.5,
            neg_threshold: None,
            min_speech_samples: sr * 250 / 1000,
            min_silence_samples: sr * 100 / 1000,
            speech_pad_samples: sr * 30 / 1000,
            max_speech_samples: usize::MAX / 2,
        }
    }

    /// Tuned for NeMo MarbleNet frame scores (10 ms hop @ 16 kHz).
    pub fn marblenet() -> Self {
        let sr = 16_000;
        Self {
            threshold: 0.5,
            neg_threshold: Some(0.35),
            min_speech_samples: sr * 200 / 1000,
            min_silence_samples: sr * 100 / 1000,
            speech_pad_samples: sr * 30 / 1000,
            max_speech_samples: usize::MAX / 2,
        }
    }

    /// Segment params for a compiled-in VAD algorithm name.
    pub fn for_algorithm(algo: &str) -> Self {
        match algo {
            "earshot" => Self::earshot(),
            "silero" => Self::silero(),
            "marblenet" => Self::marblenet(),
            other => {
                let _ = other;
                Self::default()
            }
        }
    }
}

impl SegmentParams {
    pub fn neg_threshold(&self) -> f32 {
        self.neg_threshold.unwrap_or(self.threshold - 0.15)
    }
}

#[cfg(feature = "earshot")]
pub fn speech_segments_earshot(pcm: &[f32], params: &SegmentParams) -> Vec<SpeechSegment> {
    use crate::earshot::{Detector, FRAME_SAMPLES};

    let mut det = Detector::default();
    let mut probs = Vec::new();
    for chunk in pcm.chunks(FRAME_SAMPLES) {
        if chunk.len() < FRAME_SAMPLES {
            let mut pad = vec![0.0f32; FRAME_SAMPLES];
            pad[..chunk.len()].copy_from_slice(chunk);
            probs.push(det.predict_f32(&pad));
        } else {
            probs.push(det.predict_f32(chunk));
        }
    }
    segments_from_probs(pcm.len(), FRAME_SAMPLES, params, &probs)
}

#[cfg(feature = "silero")]
pub fn speech_segments_silero(
    session: &mut crate::silero::SileroSession,
    pcm: &[f32],
    params: &SegmentParams,
) -> anyhow::Result<Vec<SpeechSegment>> {
    let frame = session.frame_samples();
    session.reset();
    let mut probs = Vec::new();
    for chunk in pcm.chunks(frame) {
        probs.push(session.predict_frame_padded(chunk)?);
    }
    Ok(segments_from_probs(pcm.len(), frame, params, &probs))
}

/// Convert a sequence of per-frame speech probabilities into speech segments,
/// using the shared hysteresis logic (threshold / neg-threshold / min-speech /
/// min-silence / padding). Model-agnostic: any frame-level VAD (MarbleNet, Silero,
/// …) that produces per-frame probs at a fixed `hop` can reuse it.
pub fn speech_segments_from_probs(
    n_samples: usize,
    hop: usize,
    params: &SegmentParams,
    probs: &[f32],
) -> Vec<SpeechSegment> {
    segments_from_probs(n_samples, hop, params, probs)
}

fn segments_from_probs(
    n_samples: usize,
    hop: usize,
    params: &SegmentParams,
    probs: &[f32],
) -> Vec<SpeechSegment> {
    let neg = params.neg_threshold();
    let mut out = Vec::new();
    let mut triggered = false;
    let mut temp_end = 0usize;
    let mut current_start = 0usize;

    for (ci, &p) in probs.iter().enumerate() {
        let current_sample = (ci + 1) * hop;
        if p >= params.threshold {
            if temp_end != 0 {
                temp_end = 0;
            }
            if !triggered {
                triggered = true;
                current_start = current_sample.saturating_sub(hop);
            }
            continue;
        }
        if triggered && current_sample.saturating_sub(current_start) > params.max_speech_samples {
            push_segment(
                params,
                &mut out,
                current_start,
                current_sample.min(n_samples),
                n_samples,
            );
            triggered = false;
            temp_end = 0;
            continue;
        }
        if triggered && p < neg {
            if temp_end == 0 {
                temp_end = current_sample;
            }
            if current_sample.saturating_sub(temp_end) >= params.min_silence_samples {
                let end = temp_end;
                if end.saturating_sub(current_start) >= params.min_speech_samples {
                    push_segment(params, &mut out, current_start, end, n_samples);
                }
                triggered = false;
                temp_end = 0;
            }
        }
    }
    if triggered {
        push_segment(params, &mut out, current_start, n_samples, n_samples);
    }
    out
}

fn push_segment(
    params: &SegmentParams,
    out: &mut Vec<SpeechSegment>,
    start: usize,
    end: usize,
    n_samples: usize,
) {
    let s = start.saturating_sub(params.speech_pad_samples);
    let e = (end + params.speech_pad_samples).min(n_samples);
    if e.saturating_sub(s) >= params.min_speech_samples {
        out.push(SpeechSegment { start: s, end: e });
    }
}

#[cfg(all(test, feature = "earshot"))]
mod tests {
    use super::*;

    #[test]
    fn empty_pcm_no_segments() {
        let segs = speech_segments_earshot(&[], &SegmentParams::default());
        assert!(segs.is_empty());
    }
}
