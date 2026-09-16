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

//! Speech regions from per-frame probabilities.
//!
//! TEN-VAD emits one score per 16 ms hop, so everything here works on the frame
//! grid and converts to samples only at the end. Hysteresis (`threshold` to
//! open, `neg_threshold` to close) keeps a single dipped frame from splitting an
//! utterance.

use crate::{HOP_SIZE, SAMPLE_RATE};

/// A speech region in samples, `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechSegment {
    pub start: usize,
    pub end: usize,
}

impl SpeechSegment {
    pub fn start_seconds(&self) -> f64 {
        self.start as f64 / SAMPLE_RATE as f64
    }

    pub fn end_seconds(&self) -> f64 {
        self.end as f64 / SAMPLE_RATE as f64
    }

    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

/// Segmentation thresholds, in samples at 16 kHz.
#[derive(Debug, Clone)]
pub struct SegmentParams {
    /// Score at or above which a region opens.
    pub threshold: f32,
    /// Score below which it closes; defaults to `threshold - 0.15`.
    pub neg_threshold: Option<f32>,
    /// Regions shorter than this are dropped.
    pub min_speech_samples: usize,
    /// Gaps shorter than this do not close a region.
    pub min_silence_samples: usize,
    /// Symmetric padding added to every kept region.
    pub speech_pad_samples: usize,
}

impl Default for SegmentParams {
    fn default() -> Self {
        Self {
            threshold: crate::DEFAULT_THRESHOLD,
            neg_threshold: None,
            min_speech_samples: SAMPLE_RATE * 250 / 1000,
            min_silence_samples: SAMPLE_RATE * 100 / 1000,
            speech_pad_samples: SAMPLE_RATE * 30 / 1000,
        }
    }
}

impl SegmentParams {
    pub fn neg_threshold(&self) -> f32 {
        self.neg_threshold.unwrap_or(self.threshold - 0.15)
    }
}

/// Collapse per-frame probabilities into speech regions.
///
/// `total_samples` clamps the last region — pass the clip length.
pub fn speech_segments(
    probs: &[f32],
    total_samples: usize,
    params: &SegmentParams,
) -> Vec<SpeechSegment> {
    let neg = params.neg_threshold();
    let min_silence_frames = params.min_silence_samples.div_ceil(HOP_SIZE);
    let mut out: Vec<SpeechSegment> = Vec::new();
    let mut start: Option<usize> = None;
    let mut quiet = 0usize;

    for (i, &p) in probs.iter().enumerate() {
        match start {
            None => {
                if p >= params.threshold {
                    start = Some(i);
                    quiet = 0;
                }
            }
            Some(s) => {
                if p < neg {
                    quiet += 1;
                    if quiet > min_silence_frames {
                        push(&mut out, s, i - quiet + 1, total_samples, params);
                        start = None;
                    }
                } else {
                    quiet = 0;
                }
            }
        }
    }
    if let Some(s) = start {
        push(&mut out, s, probs.len(), total_samples, params);
    }
    out
}

fn push(
    out: &mut Vec<SpeechSegment>,
    first_frame: usize,
    end_frame: usize,
    total_samples: usize,
    params: &SegmentParams,
) {
    let start = first_frame * HOP_SIZE;
    let end = (end_frame * HOP_SIZE).min(total_samples);
    if end.saturating_sub(start) < params.min_speech_samples {
        return;
    }
    let pad = params.speech_pad_samples;
    let seg = SpeechSegment {
        start: start.saturating_sub(pad),
        end: (end + pad).min(total_samples),
    };
    // Padding can make neighbouring regions touch; merge instead of overlapping.
    match out.last_mut() {
        Some(prev) if prev.end >= seg.start => prev.end = seg.end,
        _ => out.push(seg),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SegmentParams {
        SegmentParams {
            min_speech_samples: 0,
            min_silence_samples: 0,
            speech_pad_samples: 0,
            ..Default::default()
        }
    }

    #[test]
    fn finds_one_region() {
        let probs = [0.1, 0.9, 0.9, 0.1, 0.1];
        let segs = speech_segments(&probs, 5 * HOP_SIZE, &params());
        assert_eq!(
            segs,
            vec![SpeechSegment {
                start: HOP_SIZE,
                end: 3 * HOP_SIZE
            }]
        );
    }

    #[test]
    fn hysteresis_bridges_a_single_dip() {
        // 0.4 sits between neg (0.35) and threshold (0.5), so it must not close.
        let probs = [0.9, 0.4, 0.9];
        let segs = speech_segments(&probs, 3 * HOP_SIZE, &params());
        assert_eq!(
            segs,
            vec![SpeechSegment {
                start: 0,
                end: 3 * HOP_SIZE
            }]
        );
    }

    #[test]
    fn min_speech_drops_short_blips() {
        let probs = [0.1, 0.9, 0.1, 0.1];
        let p = SegmentParams {
            min_speech_samples: 10 * HOP_SIZE,
            ..params()
        };
        assert!(speech_segments(&probs, 4 * HOP_SIZE, &p).is_empty());
    }

    #[test]
    fn trailing_speech_is_closed_at_the_end() {
        let probs = [0.1, 0.9, 0.9];
        let segs = speech_segments(&probs, 3 * HOP_SIZE, &params());
        assert_eq!(
            segs,
            vec![SpeechSegment {
                start: HOP_SIZE,
                end: 3 * HOP_SIZE
            }]
        );
    }

    #[test]
    fn padding_merges_adjacent_regions() {
        let probs = [0.9, 0.1, 0.1, 0.9];
        let p = SegmentParams {
            speech_pad_samples: HOP_SIZE,
            ..params()
        };
        let segs = speech_segments(&probs, 4 * HOP_SIZE, &p);
        assert_eq!(segs.len(), 1, "padded neighbours should merge: {segs:?}");
    }
}
