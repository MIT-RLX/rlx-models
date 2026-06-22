// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Waveform helpers shared by tests and export tooling.

use crate::peak_amplitude;

/// Scale samples so peak amplitude matches `target_peak` (clamped to 1.0).
pub fn scale_to_peak(audio: &[f32], target_peak: f32) -> Vec<f32> {
    let peak = peak_amplitude(audio);
    if peak < 1e-8 {
        return audio.to_vec();
    }
    let scale = (target_peak / peak).min(1.0 / peak);
    audio.iter().map(|s| (s * scale).clamp(-1.0, 1.0)).collect()
}

/// Peak-normalized copy for fair ONNX/native comparison (avoids quiet ORT cherry-picking).
pub fn normalize_for_compare(audio: &[f32]) -> Vec<f32> {
    scale_to_peak(audio, 0.5)
}

/// Min peak error over sample lag (handles small vocoder phase offsets).
pub fn max_abs_best_lag(reference: &[f32], candidate: &[f32], max_lag: usize) -> (usize, f32) {
    let n = reference.len().min(candidate.len());
    if n == 0 {
        return (0, 0.0);
    }
    let max_lag = max_lag.min(n.saturating_sub(1));
    let mut best_lag = 0usize;
    let mut best = f32::MAX;
    for lag in 0..=max_lag {
        let m = n - lag;
        let mut peak = 0.0f32;
        for i in 0..m {
            peak = peak.max((reference[i] - candidate[i + lag]).abs());
        }
        if peak < best {
            best = peak;
            best_lag = lag;
        }
    }
    (best_lag, best)
}

/// Slice `candidate` at the lag that best aligns with `reference`.
pub fn align_to_reference(reference: &[f32], candidate: &[f32], max_lag: usize) -> Vec<f32> {
    let (lag, _) = max_abs_best_lag(reference, candidate, max_lag);
    let n = reference.len().min(candidate.len().saturating_sub(lag));
    if n == 0 {
        return candidate.to_vec();
    }
    candidate[lag..lag + n].to_vec()
}

/// Effective lag search window for short clips.
pub fn effective_max_lag(requested: usize, reference_len: usize, candidate_len: usize) -> usize {
    let n = reference_len.min(candidate_len);
    if n == 0 {
        return 0;
    }
    requested.min(n / 4).max(32).min(requested)
}
