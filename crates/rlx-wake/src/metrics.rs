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

//! Detection precision metrics and float precision vs a CPU reference.

use anyhow::Result;

use crate::{WakeEngine, WakeStep, score_wav};

#[derive(Debug, Clone, Copy)]
pub struct DetectionStats {
    pub threshold: f32,
    pub tp: usize,
    pub fp: usize,
    pub tn: usize,
    pub fn_: usize,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub accuracy: f32,
    pub peak_pos: f32,
    pub peak_neg: f32,
}

/// Clip-level detection: a clip is positive if any step fires (or peak ≥ threshold).
pub fn detection_stats(
    positive_peaks: &[f32],
    negative_peaks: &[f32],
    threshold: f32,
) -> DetectionStats {
    let mut tp = 0usize;
    let mut fn_ = 0usize;
    let mut fp = 0usize;
    let mut tn = 0usize;
    let mut peak_pos = 0.0f32;
    let mut peak_neg = 0.0f32;
    for &p in positive_peaks {
        peak_pos = peak_pos.max(p);
        if p >= threshold {
            tp += 1;
        } else {
            fn_ += 1;
        }
    }
    for &p in negative_peaks {
        peak_neg = peak_neg.max(p);
        if p >= threshold {
            fp += 1;
        } else {
            tn += 1;
        }
    }
    let precision = if tp + fp == 0 {
        1.0
    } else {
        tp as f32 / (tp + fp) as f32
    };
    let recall = if tp + fn_ == 0 {
        1.0
    } else {
        tp as f32 / (tp + fn_) as f32
    };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    let total = (tp + fp + tn + fn_).max(1) as f32;
    let accuracy = (tp + tn) as f32 / total;
    DetectionStats {
        threshold,
        tp,
        fp,
        tn,
        fn_,
        precision,
        recall,
        f1,
        accuracy,
        peak_pos,
        peak_neg,
    }
}

/// Peak score for one clip.
pub fn peak_of<E: WakeEngine>(eng: &mut E, pcm: &[f32]) -> Result<f32> {
    eng.reset();
    let steps = score_wav(eng, pcm)?;
    Ok(steps.iter().map(|s| s.score).fold(0.0_f32, f32::max))
}

/// Sweep thresholds and pick the one with best F1 on the given peaks.
pub fn best_f1_threshold(positive_peaks: &[f32], negative_peaks: &[f32]) -> (f32, DetectionStats) {
    let mut best_t = 0.5f32;
    let mut best = detection_stats(positive_peaks, negative_peaks, best_t);
    for i in 1..100 {
        let t = i as f32 / 100.0;
        let s = detection_stats(positive_peaks, negative_peaks, t);
        if s.f1 > best.f1 || (s.f1 == best.f1 && s.precision > best.precision) {
            best = s;
            best_t = t;
        }
    }
    (best_t, best)
}

/// Float precision of `candidate` vs `reference` score trajectories.
#[derive(Debug, Clone, Copy)]
pub struct FloatPrecision {
    pub max_abs: f32,
    pub mean_abs: f32,
    pub matched_frac: f32,
    pub n: usize,
}

pub fn float_precision(reference: &[WakeStep], candidate: &[WakeStep]) -> FloatPrecision {
    if reference.is_empty() || reference.len() != candidate.len() {
        return FloatPrecision {
            max_abs: f32::INFINITY,
            mean_abs: f32::INFINITY,
            matched_frac: 0.0,
            n: 0,
        };
    }
    let mut max_abs = 0.0f32;
    let mut sum = 0.0f32;
    let mut matched = 0usize;
    for (a, b) in reference.iter().zip(candidate.iter()) {
        let d = (a.score - b.score).abs();
        max_abs = max_abs.max(d);
        sum += d;
        if a.score.to_bits() == b.score.to_bits() {
            matched += 1;
        }
    }
    FloatPrecision {
        max_abs,
        mean_abs: sum / reference.len() as f32,
        matched_frac: matched as f32 / reference.len() as f32,
        n: reference.len(),
    }
}

pub fn print_detection_stats(engine: &str, device: &str, s: &DetectionStats) {
    println!(
        "{:<18} {:<8} thr={:.2}  P={:.3}  R={:.3}  F1={:.3}  Acc={:.3}  \
         tp={} fp={} tn={} fn={}  peak+/−={:.4}/{:.4}",
        engine,
        device,
        s.threshold,
        s.precision,
        s.recall,
        s.f1,
        s.accuracy,
        s.tp,
        s.fp,
        s.tn,
        s.fn_,
        s.peak_pos,
        s.peak_neg
    );
}
