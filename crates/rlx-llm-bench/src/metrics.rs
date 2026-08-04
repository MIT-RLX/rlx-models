// RLX models — LLM benchmark harness.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Small, dependency-light numeric helpers shared across the bench dimensions:
//! stable `log_softmax` (for teacher-forced log-probs), cosine similarity (for
//! parity), argmax, and process peak-RSS (for the memory column). Kept local so
//! the harness core does not pull a heavy crate just for these.

/// Index of the maximum element. Returns 0 for an empty slice.
pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

/// `log(softmax(logits))[target]` computed in a numerically stable way without
/// materializing the full softmax vector — the log-sum-exp trick, reading each
/// logit once. Returns `NEG_INFINITY` if `target` is out of range.
pub fn log_softmax_at(logits: &[f32], target: usize) -> f32 {
    if target >= logits.len() || logits.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mut max = f32::NEG_INFINITY;
    for &l in logits {
        if l > max {
            max = l;
        }
    }
    let mut sum = 0.0f64;
    for &l in logits {
        sum += ((l - max) as f64).exp();
    }
    (logits[target] - max) - (sum.ln() as f32)
}

/// Cosine similarity of two equal-length vectors. Returns 0.0 for a length
/// mismatch or a zero-norm operand (so a degenerate comparison reads as "no
/// agreement" rather than NaN).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// Process peak resident set size in MB via `getrusage(RUSAGE_SELF)`.
/// Returns 0 where unavailable. Mirrors `rlx_core::asr_bench::peak_rss_mb` so
/// the LLM and ASR leaderboards report the same memory column.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_softmax_matches_manual() {
        // Two equal logits → each prob 0.5 → log = -ln 2.
        let l = [1.0f32, 1.0];
        let expected = -(2.0f32.ln());
        assert!((log_softmax_at(&l, 0) - expected).abs() < 1e-5);
        assert!((log_softmax_at(&l, 1) - expected).abs() < 1e-5);
    }

    #[test]
    fn log_softmax_out_of_range_is_neg_inf() {
        assert_eq!(log_softmax_at(&[0.0, 1.0], 5), f32::NEG_INFINITY);
        assert_eq!(log_softmax_at(&[], 0), f32::NEG_INFINITY);
    }

    #[test]
    fn cosine_identical_is_one() {
        let a = [1.0f32, 2.0, 3.0];
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        // Length mismatch / zero-norm degrade to 0.0, not NaN.
        assert_eq!(cosine(&[1.0], &[1.0, 2.0]), 0.0);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn argmax_picks_largest() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(argmax(&[]), 0);
    }
}
