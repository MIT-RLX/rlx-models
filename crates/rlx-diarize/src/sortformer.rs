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

//! **Sortformer** end-to-end neural speaker diarization (NVIDIA NeMo).
//!
//! Unlike this crate's default embed-then-cluster pipeline, Sortformer is a single
//! network — a FastConformer encoder + transformer — that emits, for every frame,
//! a per-speaker **activity probability** (multi-label sigmoid over `max_speakers`).
//! Its "Sort Loss" trains speakers to appear in **arrival-time order**, so speaker
//! slots are canonical. Post-processing thresholds the activity and merges
//! contiguous active frames into [`SpeakerTurn`]s.
//!
//! This module provides the checkpoint-free config plus that post-processing
//! ([`activity_to_turns`], [`sort_speakers_by_arrival`]); the FastConformer +
//! transformer graph + NeMo weights are the next step.

use crate::session::SpeakerTurn;

/// Sortformer architecture / decode config.
#[derive(Debug, Clone, PartialEq)]
pub struct SortformerConfig {
    // FastConformer encoder.
    pub encoder_dim: usize,
    pub encoder_layers: usize,
    pub encoder_heads: usize,
    /// Maximum simultaneously-tracked speakers (slots).
    pub max_speakers: usize,
    /// Activity frames per second (FastConformer ≈ 12.5 fps at 80 ms).
    pub frame_rate: f32,
    /// Per-frame activity threshold for turning probs into turns.
    pub threshold: f32,
}

impl Default for SortformerConfig {
    fn default() -> Self {
        Self {
            encoder_dim: 512,
            encoder_layers: 18,
            encoder_heads: 8,
            max_speakers: 4,
            frame_rate: 12.5,
            threshold: 0.5,
        }
    }
}

impl SortformerConfig {
    /// Seconds per activity frame.
    pub fn frame_sec(&self) -> f32 {
        1.0 / self.frame_rate
    }
}

/// The permutation of speaker slots ordered by **first active frame** (Sortformer's
/// canonical arrival-time order). Speakers that never cross `threshold` sort last.
pub fn sort_speakers_by_arrival(activity: &[Vec<f32>], threshold: f32) -> Vec<usize> {
    let Some(first) = activity.first() else {
        return Vec::new();
    };
    let n_spk = first.len();
    let mut arrival = vec![usize::MAX; n_spk];
    for (f, frame) in activity.iter().enumerate() {
        for (s, a) in arrival.iter_mut().enumerate() {
            if *a == usize::MAX && frame.get(s).copied().unwrap_or(0.0) >= threshold {
                *a = f;
            }
        }
    }
    let mut order: Vec<usize> = (0..n_spk).collect();
    order.sort_by_key(|&s| arrival[s]);
    order
}

/// Threshold a `[frames][max_speakers]` activity matrix and merge each speaker's
/// contiguous active frames into [`SpeakerTurn`]s (times in seconds via `frame_sec`),
/// sorted by start time.
pub fn activity_to_turns(
    activity: &[Vec<f32>],
    threshold: f32,
    frame_sec: f32,
) -> Vec<SpeakerTurn> {
    let mut turns = Vec::new();
    let Some(first) = activity.first() else {
        return turns;
    };
    let n_spk = first.len();
    let n_frames = activity.len();

    for s in 0..n_spk {
        let mut run_start: Option<usize> = None;
        for (f, frame) in activity.iter().enumerate() {
            let active = frame.get(s).copied().unwrap_or(0.0) >= threshold;
            if active {
                if run_start.is_none() {
                    run_start = Some(f);
                }
            } else if let Some(st) = run_start.take() {
                turns.push(SpeakerTurn {
                    speaker_id: s,
                    start: st as f32 * frame_sec,
                    end: f as f32 * frame_sec,
                });
            }
        }
        if let Some(st) = run_start.take() {
            turns.push(SpeakerTurn {
                speaker_id: s,
                start: st as f32 * frame_sec,
                end: n_frames as f32 * frame_sec,
            });
        }
    }

    turns.sort_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.speaker_id.cmp(&b.speaker_id))
    });
    turns
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene() -> Vec<Vec<f32>> {
        // frame0: spk1; frame1: spk1; frame2: spk0. (spk, prob per frame)
        vec![vec![0.1, 0.9], vec![0.2, 0.8], vec![0.7, 0.1]]
    }

    #[test]
    fn config_frame_sec() {
        let c = SortformerConfig::default();
        assert_eq!(c.max_speakers, 4);
        assert!((c.frame_sec() - 0.08).abs() < 1e-6); // 1/12.5
    }

    #[test]
    fn speakers_sorted_by_arrival() {
        // spk1 arrives at frame 0, spk0 at frame 2 → order [1, 0].
        assert_eq!(sort_speakers_by_arrival(&scene(), 0.5), vec![1, 0]);
    }

    #[test]
    fn activity_becomes_turns() {
        let turns = activity_to_turns(&scene(), 0.5, 0.1);
        assert_eq!(turns.len(), 2);
        // spk1 speaks first: frames [0,2) → [0.0, 0.2)
        assert_eq!(turns[0].speaker_id, 1);
        assert!((turns[0].start - 0.0).abs() < 1e-6);
        assert!((turns[0].end - 0.2).abs() < 1e-6);
        // spk0: frame 2 → [0.2, 0.3)
        assert_eq!(turns[1].speaker_id, 0);
        assert!((turns[1].start - 0.2).abs() < 1e-6);
        assert!((turns[1].end - 0.3).abs() < 1e-6);
    }

    #[test]
    fn empty_activity_yields_nothing() {
        assert!(activity_to_turns(&[], 0.5, 0.1).is_empty());
        assert!(sort_speakers_by_arrival(&[], 0.5).is_empty());
    }
}
