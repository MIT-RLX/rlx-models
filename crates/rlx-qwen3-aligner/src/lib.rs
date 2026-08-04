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

//! # rlx-qwen3-aligner
//!
//! **Qwen3 forced aligner** on RLX — given audio and a *known* transcript, produce
//! per-token timestamps. The acoustic encoder (reused from `rlx-qwen3-asr`) yields
//! per-frame token log-probabilities; this crate contributes the alignment
//! algorithm: a monotonic **Viterbi forced alignment** ([`forced_align`]) that maps
//! the target token sequence onto frames, plus frame↔time conversion.
//!
//! Native Rust. The DP is checkpoint-free and unit-tested; the Qwen3 encoder graph
//! + weight loading is the next step (reuse `rlx-qwen3-asr`, already multi-backend).

use anyhow::{Result, ensure};

/// Aligner config.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3AlignerConfig {
    pub encoder_dim: usize,
    pub encoder_layers: usize,
    pub vocab_size: usize,
    /// Encoder output frames per second.
    pub frame_rate: f32,
}

impl Default for Qwen3AlignerConfig {
    fn default() -> Self {
        Self {
            encoder_dim: 1024,
            encoder_layers: 24,
            vocab_size: 151_936,
            frame_rate: 25.0,
        }
    }
}

impl Qwen3AlignerConfig {
    /// Convert a frame index to seconds.
    pub fn frame_to_seconds(&self, frame: usize) -> f32 {
        frame as f32 / self.frame_rate
    }
}

/// The aligned frame span of one target token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenAlignment {
    pub token: i32,
    /// First frame assigned to this token (inclusive).
    pub start_frame: usize,
    /// One past the last frame assigned (exclusive).
    pub end_frame: usize,
}

const NEG_INF: f32 = f32::NEG_INFINITY;

/// Monotonic Viterbi forced alignment of `targets` onto `log_probs`
/// (`[frames][vocab]`). Each token consumes one or more consecutive frames, the
/// path advances through the tokens in order (no skips, no reordering), and every
/// frame is assigned to exactly one token. Returns each token's frame span.
///
/// Requires `frames >= targets.len() >= 1` (one frame minimum per token).
pub fn forced_align(log_probs: &[Vec<f32>], targets: &[i32]) -> Result<Vec<TokenAlignment>> {
    let t_frames = log_probs.len();
    let n = targets.len();
    ensure!(n >= 1, "need at least one target token");
    ensure!(
        t_frames >= n,
        "need at least as many frames ({t_frames}) as tokens ({n})"
    );
    let vocab = log_probs[0].len();
    ensure!(
        log_probs.iter().all(|row| row.len() == vocab),
        "ragged log_probs rows"
    );
    for &tok in targets {
        ensure!(
            tok >= 0 && (tok as usize) < vocab,
            "target token {tok} out of vocab range {vocab}"
        );
    }

    let emit = |t: usize, l: usize| -> f32 { log_probs[t][targets[l] as usize] };

    // dp[t][l] = best score aligning frames 0..=t with the path ending on token l.
    // back[t][l] = true if we advanced from l-1 (else stayed on l).
    let mut dp = vec![vec![NEG_INF; n]; t_frames];
    let mut back = vec![vec![false; n]; t_frames];

    dp[0][0] = emit(0, 0);
    for t in 1..t_frames {
        for l in 0..n {
            // A token l can only be reached by frame t if t >= l (need l prior frames).
            if l > t {
                continue;
            }
            let stay = dp[t - 1][l];
            let advance = if l > 0 { dp[t - 1][l - 1] } else { NEG_INF };
            let (prev, adv) = if advance > stay {
                (advance, true)
            } else {
                (stay, false)
            };
            if prev == NEG_INF {
                continue;
            }
            dp[t][l] = prev + emit(t, l);
            back[t][l] = adv;
        }
    }
    ensure!(
        dp[t_frames - 1][n - 1] > NEG_INF,
        "no valid alignment path (check frame/token counts)"
    );

    // Backtrack: assign a token index to every frame.
    let mut frame_token = vec![0usize; t_frames];
    let mut l = n - 1;
    for t in (0..t_frames).rev() {
        frame_token[t] = l;
        if back[t][l] {
            l -= 1; // we advanced into l at frame t → previous frame was l-1
        }
    }
    debug_assert_eq!(l, 0);

    // Collapse frame assignments into per-token spans.
    let mut spans = vec![(usize::MAX, 0usize); n];
    for (t, &tok_idx) in frame_token.iter().enumerate() {
        let (s, e) = &mut spans[tok_idx];
        *s = (*s).min(t);
        *e = (*e).max(t + 1);
    }
    Ok((0..n)
        .map(|l| TokenAlignment {
            token: targets[l],
            start_frame: spans[l].0,
            end_frame: spans[l].1,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_frame_to_seconds() {
        let c = Qwen3AlignerConfig::default();
        assert!((c.frame_to_seconds(50) - 2.0).abs() < 1e-6); // 50 / 25 fps
    }

    #[test]
    fn aligns_monotonically() {
        // 3 frames, tokens [0, 1]. Frames 0,1 favour token 0; frame 2 favours token 1.
        let lp = vec![vec![0.0, -5.0], vec![0.0, -5.0], vec![-5.0, 0.0]];
        let al = forced_align(&lp, &[0, 1]).unwrap();
        assert_eq!(al.len(), 2);
        // token 0 → frames [0, 2); token 1 → frame [2, 3)
        assert_eq!(
            al[0],
            TokenAlignment {
                token: 0,
                start_frame: 0,
                end_frame: 2
            }
        );
        assert_eq!(
            al[1],
            TokenAlignment {
                token: 1,
                start_frame: 2,
                end_frame: 3
            }
        );
    }

    #[test]
    fn spans_are_contiguous_and_cover_all_frames() {
        let lp = vec![
            vec![0.0, -3.0, -3.0],
            vec![-3.0, 0.0, -3.0],
            vec![-3.0, 0.0, -3.0],
            vec![-3.0, -3.0, 0.0],
        ];
        let al = forced_align(&lp, &[0, 1, 2]).unwrap();
        // contiguous, monotonically increasing, covering [0, 4)
        assert_eq!(al[0].start_frame, 0);
        assert_eq!(al.last().unwrap().end_frame, 4);
        for w in al.windows(2) {
            assert_eq!(w[0].end_frame, w[1].start_frame);
        }
    }

    #[test]
    fn rejects_more_tokens_than_frames() {
        let lp = vec![vec![0.0, 0.0]];
        assert!(forced_align(&lp, &[0, 1]).is_err());
        assert!(forced_align(&lp, &[]).is_err());
        // out-of-vocab token
        assert!(forced_align(&lp, &[9]).is_err());
    }
}
