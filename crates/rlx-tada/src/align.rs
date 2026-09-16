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

//! Monotonic text↔frame alignment (`tada.modules.aligner._align_text_tokens`).
//!
//! Given per-frame CTC logits over the Llama vocabulary and the text token
//! ids, pick one strictly increasing frame per token maximizing the summed
//! logit of the assigned cells. This is what makes TADA "dual aligned": every
//! text token owns exactly one frame, so the LM can carry text and acoustics
//! on a single 1:1 stream.
//!
//! Not a CTC forward-backward — upstream's recursion has no blank state and no
//! log-sum-exp, it is a plain max-score monotonic assignment. Ported exactly,
//! including the diagonal seeding, because the chosen frames feed straight into
//! the duration field the model was trained against.

use anyhow::{Result, bail};

/// Result of aligning one utterance.
#[derive(Debug, Clone)]
pub struct Alignment {
    /// Frame index assigned to each text token, **1-based** — upstream stores
    /// `1 + position` so that `0` can double as padding.
    pub token_positions: Vec<u32>,
    /// Per-frame `0/1` flag: 1 where some token landed. Length `num_frames`.
    pub token_mask: Vec<u8>,
    /// Mean log-softmax probability the CTC head assigns to each token at the
    /// frame it was aligned to.
    ///
    /// The aligner is a forced aligner: it places every token somewhere no
    /// matter what, so a transcript that does not match the audio produces an
    /// alignment that looks structurally fine and is meaningless. This is the
    /// signal that separates them — when the words are really there the head is
    /// confident at the chosen frames, and when they are not it is not. See
    /// [`Alignment::looks_mismatched`].
    pub mean_token_logprob: f32,
}

impl Alignment {
    /// Whether the transcript plausibly describes the audio.
    ///
    /// The threshold is empirical, from the JFK reference clip: its true
    /// transcript scores -0.34, the same text with five words of preamble that
    /// are not in the recording scores -4.66, and an unrelated sentence -16.12.
    /// `-1.5` sits an order of magnitude clear of both sides; it is a smoke
    /// alarm, not a metric.
    pub fn looks_mismatched(&self) -> bool {
        self.mean_token_logprob < -1.5
    }
}

/// Align `text_tokens` to `logits` (`[num_frames, vocab]`, row-major, raw CTC
/// logits — not log-softmaxed, matching upstream).
///
/// `num_frames` is the trusted audio length in 50 Hz frames; `token_mask` is
/// sized from it, while the dynamic program runs over every row of `logits`.
pub fn align_tokens(
    logits: &[f32],
    vocab: usize,
    text_tokens: &[u32],
    num_frames: usize,
) -> Result<Alignment> {
    let positions = solve(logits, vocab, text_tokens)?;
    let mean_token_logprob = mean_aligned_logprob(logits, vocab, text_tokens, &positions);
    let mut token_mask = vec![0u8; num_frames];
    for &p in &positions {
        let p = p as usize;
        if p >= num_frames {
            bail!("alignment picked frame {p} beyond the {num_frames}-frame audio");
        }
        token_mask[p] = 1;
    }
    Ok(Alignment {
        token_positions: positions.iter().map(|&p| p + 1).collect(),
        mean_token_logprob,
        token_mask,
    })
}

/// Mean `log_softmax(logits[frame])[token]` over every aligned token.
fn mean_aligned_logprob(
    logits: &[f32],
    vocab: usize,
    text_tokens: &[u32],
    positions: &[u32],
) -> f32 {
    if positions.is_empty() || vocab == 0 {
        return f32::NEG_INFINITY;
    }
    let mut total = 0f64;
    let mut n = 0usize;
    for (&tok, &frame) in text_tokens.iter().zip(positions) {
        let start = frame as usize * vocab;
        let Some(row) = logits.get(start..start + vocab) else {
            continue;
        };
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            continue;
        }
        let sum: f64 = row.iter().map(|v| ((v - max) as f64).exp()).sum();
        total += (row[tok as usize] - max) as f64 - sum.ln();
        n += 1;
    }
    if n == 0 {
        f32::NEG_INFINITY
    } else {
        (total / n as f64) as f32
    }
}

/// The dynamic program. Returns the 0-based frame chosen for each token.
fn solve(logits: &[f32], vocab: usize, text_tokens: &[u32]) -> Result<Vec<u32>> {
    let t = text_tokens.len();
    if t == 0 {
        return Ok(Vec::new());
    }
    if vocab == 0 || !logits.len().is_multiple_of(vocab) {
        bail!(
            "logits length {} is not a multiple of vocab {vocab}",
            logits.len()
        );
    }
    let l = logits.len() / vocab;
    if l == 0 {
        bail!("no frames to align {t} tokens against");
    }
    for &tok in text_tokens {
        if tok as usize >= vocab {
            bail!("text token {tok} is outside the {vocab}-entry CTC vocabulary");
        }
    }

    // token_probs[i][j] = logits[frame i, text_tokens[j]]
    let mut token_probs = vec![0f32; l * t];
    for i in 0..l {
        let row = &logits[i * vocab..(i + 1) * vocab];
        for (j, &tok) in text_tokens.iter().enumerate() {
            token_probs[i * t + j] = row[tok as usize];
        }
    }

    let mut score = vec![f32::NEG_INFINITY; l * t];
    // -1 marks "skipped this frame"; any other value is the frame index used.
    let mut back = vec![0i64; l * t];

    // Column 0: running max over frames of the first token's score.
    {
        let mut best = f32::NEG_INFINITY;
        let mut best_i = 0i64;
        for i in 0..l {
            let v = token_probs[i * t];
            if v > best {
                best = v;
                best_i = i as i64;
            }
            score[i * t] = best;
            back[i * t] = best_i;
        }
    }

    // Diagonal seed: the forced one-token-per-frame assignment. Never
    // overwritten below, because row `i` only relaxes columns `j < min(i, T)`.
    if t <= l {
        let mut acc = 0f32;
        for j in 0..t {
            acc += token_probs[j * t + j];
            score[j * t + j] = acc;
            back[j * t + j] = j as i64;
        }
    }

    for i in 1..l {
        let max_j = i.min(t);
        if max_j <= 1 {
            continue;
        }
        for j in 1..max_j {
            let skip = score[(i - 1) * t + j];
            let use_ = score[(i - 1) * t + (j - 1)] + token_probs[i * t + j];
            // `>=` matches upstream: ties prefer consuming the frame.
            if use_ >= skip {
                score[i * t + j] = use_;
                back[i * t + j] = i as i64;
            } else {
                score[i * t + j] = skip;
                back[i * t + j] = -1;
            }
        }
    }

    let mut positions = vec![0u32; t];
    let mut i = l - 1;
    let mut j = t - 1;
    let mut out_idx = t - 1;
    loop {
        if j == 0 {
            let b = back[i * t];
            if b < 0 {
                bail!("alignment traceback hit an unset cell at the first token");
            }
            positions[out_idx] = b as u32;
            break;
        }
        let b = back[i * t + j];
        if b < 0 {
            if i == 0 {
                bail!(
                    "alignment traceback ran out of frames with {j} tokens left — \
                     the reference audio is too short for its transcript"
                );
            }
            i -= 1;
        } else {
            positions[out_idx] = b as u32;
            if out_idx == 0 {
                break;
            }
            out_idx -= 1;
            if i == 0 {
                bail!(
                    "alignment traceback ran out of frames with {j} tokens left — \
                     the reference audio is too short for its transcript"
                );
            }
            i -= 1;
            j -= 1;
        }
    }
    Ok(positions)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `[frames, vocab]` logits that are 1.0 at `peaks[frame]` and 0 elsewhere.
    fn peaky(vocab: usize, peaks: &[Option<u32>]) -> Vec<f32> {
        let mut out = vec![0f32; peaks.len() * vocab];
        for (i, p) in peaks.iter().enumerate() {
            if let Some(tok) = p {
                out[i * vocab + *tok as usize] = 1.0;
            }
        }
        out
    }

    /// Logits that are confident about `peaks` and flat elsewhere.
    fn confident(vocab: usize, peaks: &[Option<u32>]) -> Vec<f32> {
        let mut out = vec![0f32; peaks.len() * vocab];
        for (i, p) in peaks.iter().enumerate() {
            if let Some(tok) = p {
                out[i * vocab + *tok as usize] = 20.0;
            }
        }
        out
    }

    #[test]
    fn a_matching_transcript_scores_far_above_a_mismatched_one() {
        let vocab = 8;
        // Frames that clearly contain tokens 3, 5, 7 in order.
        let logits = confident(vocab, &[Some(3), None, Some(5), None, Some(7)]);

        let good = align_tokens(&logits, vocab, &[3, 5, 7], 5).expect("align");
        let bad = align_tokens(&logits, vocab, &[1, 2, 4], 5).expect("align");

        assert!(
            good.mean_token_logprob > -0.5,
            "a transcript that is really there should score high, got {}",
            good.mean_token_logprob
        );
        assert!(
            bad.mean_token_logprob < -5.0,
            "a transcript that is not there should score low, got {}",
            bad.mean_token_logprob
        );
        // The forced aligner places every token either way — the structure of
        // the result is identical and only the score tells them apart.
        assert_eq!(good.token_positions.len(), bad.token_positions.len());
        assert!(!good.looks_mismatched());
        assert!(bad.looks_mismatched());
    }

    #[test]
    fn picks_the_peak_frame_for_each_token() {
        let vocab = 6;
        //            f0      f1      f2      f3      f4
        let peaks = [None, Some(3), None, Some(5), None];
        let logits = peaky(vocab, &peaks);
        let a = align_tokens(&logits, vocab, &[3, 5], 5).unwrap();
        // 1-based positions.
        assert_eq!(a.token_positions, vec![2, 4]);
        assert_eq!(a.token_mask, vec![0, 1, 0, 1, 0]);
    }

    #[test]
    fn assignment_is_strictly_increasing() {
        let vocab = 8;
        let peaks = [Some(1), Some(2), Some(2), Some(3), Some(4), Some(4)];
        let logits = peaky(vocab, &peaks);
        let a = align_tokens(&logits, vocab, &[1, 2, 3, 4], 6).unwrap();
        let p = &a.token_positions;
        assert_eq!(p.len(), 4);
        for w in p.windows(2) {
            assert!(w[0] < w[1], "positions not increasing: {p:?}");
        }
    }

    #[test]
    fn one_token_per_frame_when_counts_match() {
        let vocab = 5;
        let logits = peaky(vocab, &[Some(1), Some(2), Some(3)]);
        let a = align_tokens(&logits, vocab, &[1, 2, 3], 3).unwrap();
        assert_eq!(a.token_positions, vec![1, 2, 3]);
        assert_eq!(a.token_mask, vec![1, 1, 1]);
    }

    #[test]
    fn empty_text_aligns_to_nothing() {
        let a = align_tokens(&[0.0; 12], 4, &[], 3).unwrap();
        assert!(a.token_positions.is_empty());
        assert_eq!(a.token_mask, vec![0, 0, 0]);
    }

    #[test]
    fn more_tokens_than_frames_is_an_error_not_a_silent_truncation() {
        let vocab = 4;
        let logits = peaky(vocab, &[Some(1), Some(2)]);
        assert!(align_tokens(&logits, vocab, &[1, 2, 3, 1, 2], 2).is_err());
    }

    #[test]
    fn token_outside_the_vocabulary_is_rejected() {
        assert!(align_tokens(&[0.0; 8], 4, &[9], 2).is_err());
    }

    #[test]
    fn ragged_logits_are_rejected() {
        assert!(align_tokens(&[0.0; 7], 4, &[1], 2).is_err());
    }
}
