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

//! Greedy (best-path) CTC decoding over a `[frames, classes]` score matrix.
//!
//! Text-line recognizers (PyLaia, Kraken, CRNN OCR) end in a per-frame
//! distribution over `classes` labels with one blank. Best-path decoding
//! takes the arg-max per frame, merges runs of the same label, and drops
//! blanks. The scores can be logits, probabilities or log-probabilities:
//! the arg-max is the same for all three.

/// One decoded label with the frame span it was read from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CtcLabel {
    /// Class index (never the blank).
    pub label: usize,
    /// First frame of the run.
    pub start: usize,
    /// One past the last frame of the run.
    pub end: usize,
    /// Highest score of `label` over the run, as given (logit / prob).
    pub score: f32,
}

/// Arg-max per frame of a row-major `[frames, classes]` matrix. Ties go
/// to the lower index, as `torch.argmax` does.
pub fn argmax_frames(scores: &[f32], classes: usize) -> Vec<usize> {
    assert!(classes > 0, "argmax_frames: zero classes");
    assert_eq!(
        scores.len() % classes,
        0,
        "argmax_frames: {} scores is not a multiple of {classes} classes",
        scores.len()
    );
    scores
        .chunks_exact(classes)
        .map(|row| {
            let mut best = 0;
            for (i, &v) in row.iter().enumerate() {
                if v > row[best] {
                    best = i;
                }
            }
            best
        })
        .collect()
}

/// Best-path CTC: arg-max per frame, merge repeats, drop `blank`.
pub fn greedy_decode(scores: &[f32], classes: usize, blank: usize) -> Vec<CtcLabel> {
    let path = argmax_frames(scores, classes);
    let mut out: Vec<CtcLabel> = Vec::new();
    let mut prev = None;
    for (t, &k) in path.iter().enumerate() {
        let s = scores[t * classes + k];
        if prev == Some(k) {
            if let Some(last) = out.last_mut().filter(|l| l.label == k && l.end == t) {
                last.end = t + 1;
                last.score = last.score.max(s);
            }
        } else if k != blank {
            out.push(CtcLabel {
                label: k,
                start: t,
                end: t + 1,
                score: s,
            });
        }
        prev = Some(k);
    }
    out
}

/// Labels only — [`greedy_decode`] without the spans.
pub fn greedy_labels(scores: &[f32], classes: usize, blank: usize) -> Vec<usize> {
    greedy_decode(scores, classes, blank)
        .into_iter()
        .map(|l| l.label)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn onehot(path: &[usize], classes: usize) -> Vec<f32> {
        let mut v = vec![0f32; path.len() * classes];
        for (t, &k) in path.iter().enumerate() {
            v[t * classes + k] = 1.0;
        }
        v
    }

    #[test]
    fn merges_repeats_and_drops_blanks() {
        // a a _ a b b _ _ c  →  a a b c
        let s = onehot(&[1, 1, 0, 1, 2, 2, 0, 0, 3], 4);
        assert_eq!(greedy_labels(&s, 4, 0), vec![1, 1, 2, 3]);
    }

    #[test]
    fn spans_cover_the_run() {
        let s = onehot(&[0, 2, 2, 2, 0, 1], 3);
        let d = greedy_decode(&s, 3, 0);
        assert_eq!(d.len(), 2);
        assert_eq!((d[0].label, d[0].start, d[0].end), (2, 1, 4));
        assert_eq!((d[1].label, d[1].start, d[1].end), (1, 5, 6));
    }

    #[test]
    fn nonzero_blank_index() {
        // Kraken-style codecs put the blank at 0 too, but CRNNs that append
        // it last exist; the blank is whatever the caller says.
        let s = onehot(&[2, 0, 0, 2, 1], 3);
        assert_eq!(greedy_labels(&s, 3, 2), vec![0, 1]);
    }

    #[test]
    fn ties_pick_the_lower_index() {
        assert_eq!(argmax_frames(&[0.5, 0.5, 0.1], 3), vec![0]);
    }
}
