// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! CTC prefix beam search (adapted from `rlx-ocr2::beam`) + segmentation helpers.

use std::collections::HashMap;

#[inline]
fn logsumexp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        b
    } else if b == f32::NEG_INFINITY {
        a
    } else {
        let m = a.max(b);
        m + ((a - m).exp() + (b - m).exp()).ln()
    }
}

/// Streaming CTC prefix beam — push one frame at a time, read partial / final.
#[derive(Clone, Debug)]
pub struct StreamingCtcBeam {
    blank: usize,
    beam: usize,
    prune_margin: f32,
    beams: HashMap<Vec<usize>, (f32, f32)>,
    n_frames: usize,
}

impl StreamingCtcBeam {
    pub fn new(blank: usize, beam: usize) -> Self {
        let mut beams = HashMap::new();
        beams.insert(Vec::new(), (0.0, f32::NEG_INFINITY));
        Self {
            blank,
            beam: beam.max(1),
            prune_margin: 9.0,
            beams,
            n_frames: 0,
        }
    }

    pub fn reset(&mut self) {
        self.beams.clear();
        self.beams.insert(Vec::new(), (0.0, f32::NEG_INFINITY));
        self.n_frames = 0;
    }

    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Ingest one log-prob row of length `classes`.
    pub fn push(&mut self, row: &[f32]) {
        let classes = row.len();
        let maxlp = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let thresh = maxlp - self.prune_margin;
        let cands: Vec<usize> = (0..classes)
            .filter(|&c| c != self.blank && row[c] > thresh)
            .collect();
        let mut next: HashMap<Vec<usize>, (f32, f32)> = HashMap::new();
        let bump =
            |map: &mut HashMap<Vec<usize>, (f32, f32)>, key: Vec<usize>, db: f32, dnb: f32| {
                let e = map
                    .entry(key)
                    .or_insert((f32::NEG_INFINITY, f32::NEG_INFINITY));
                e.0 = logsumexp(e.0, db);
                e.1 = logsumexp(e.1, dnb);
            };
        for (prefix, &(pb, pnb)) in &self.beams {
            bump(
                &mut next,
                prefix.clone(),
                logsumexp(pb, pnb) + row[self.blank],
                f32::NEG_INFINITY,
            );
            let last = prefix.last().copied();
            for &c in &cands {
                let lp = row[c];
                if last == Some(c) {
                    bump(&mut next, prefix.clone(), f32::NEG_INFINITY, pnb + lp);
                    let mut np = prefix.clone();
                    np.push(c);
                    bump(&mut next, np, f32::NEG_INFINITY, pb + lp);
                } else {
                    let mut np = prefix.clone();
                    np.push(c);
                    bump(&mut next, np, f32::NEG_INFINITY, logsumexp(pb, pnb) + lp);
                }
            }
        }
        let mut v: Vec<(Vec<usize>, (f32, f32))> = next.into_iter().collect();
        v.sort_by(|a, b| {
            logsumexp(b.1.0, b.1.1)
                .partial_cmp(&logsumexp(a.1.0, a.1.1))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        v.truncate(self.beam);
        self.beams = v.into_iter().collect();
        self.n_frames += 1;
    }

    pub fn push_many(&mut self, logp: &[f32], seq: usize, classes: usize) {
        for t in 0..seq {
            self.push(&logp[t * classes..(t + 1) * classes]);
        }
    }

    pub fn best(&self) -> (Vec<usize>, f32) {
        self.beams
            .iter()
            .map(|(p, (pb, pnb))| (p.clone(), logsumexp(*pb, *pnb)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or_else(|| (Vec::new(), f32::NEG_INFINITY))
    }

    pub fn partial_ids(&self) -> Vec<usize> {
        self.best().0
    }
}

/// Prefix beam search over `[seq, classes]` log-probs. Returns collapsed, blank-free
/// label sequences with their total log-probability, best first.
pub fn ctc_beam_nbest(
    logp: &[f32],
    seq: usize,
    classes: usize,
    blank: usize,
    beam: usize,
) -> Vec<(Vec<usize>, f32)> {
    let mut beams: HashMap<Vec<usize>, (f32, f32)> = HashMap::new();
    beams.insert(Vec::new(), (0.0, f32::NEG_INFINITY));
    const PRUNE_MARGIN: f32 = 9.0;
    for t in 0..seq {
        let row = &logp[t * classes..(t + 1) * classes];
        let maxlp = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let thresh = maxlp - PRUNE_MARGIN;
        let cands: Vec<usize> = (0..classes)
            .filter(|&c| c != blank && row[c] > thresh)
            .collect();
        let mut next: HashMap<Vec<usize>, (f32, f32)> = HashMap::new();
        let bump =
            |map: &mut HashMap<Vec<usize>, (f32, f32)>, key: Vec<usize>, db: f32, dnb: f32| {
                let e = map
                    .entry(key)
                    .or_insert((f32::NEG_INFINITY, f32::NEG_INFINITY));
                e.0 = logsumexp(e.0, db);
                e.1 = logsumexp(e.1, dnb);
            };
        for (prefix, &(pb, pnb)) in &beams {
            bump(
                &mut next,
                prefix.clone(),
                logsumexp(pb, pnb) + row[blank],
                f32::NEG_INFINITY,
            );
            let last = prefix.last().copied();
            for &c in &cands {
                let lp = row[c];
                if last == Some(c) {
                    bump(&mut next, prefix.clone(), f32::NEG_INFINITY, pnb + lp);
                    let mut np = prefix.clone();
                    np.push(c);
                    bump(&mut next, np, f32::NEG_INFINITY, pb + lp);
                } else {
                    let mut np = prefix.clone();
                    np.push(c);
                    bump(&mut next, np, f32::NEG_INFINITY, logsumexp(pb, pnb) + lp);
                }
            }
        }
        let mut v: Vec<(Vec<usize>, (f32, f32))> = next.into_iter().collect();
        v.sort_by(|a, b| {
            logsumexp(b.1.0, b.1.1)
                .partial_cmp(&logsumexp(a.1.0, a.1.1))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        v.truncate(beam.max(1));
        beams = v.into_iter().collect();
    }
    let mut out: Vec<(Vec<usize>, f32)> = beams
        .into_iter()
        .map(|(p, (pb, pnb))| (p, logsumexp(pb, pnb)))
        .collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Split a CTC hypothesis on a segmentation token id (e.g. `▁<segE>`).
pub fn split_segments(ids: &[usize], seg_id: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut cur = Vec::new();
    for &id in ids {
        if id == seg_id {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        } else {
            cur.push(id);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn beam_picks_constant_path() {
        // 2 frames, 3 classes; blank=0; class 1 dominates.
        let logp = [
            -10.0, 0.0, -10.0, //
            -10.0, 0.0, -10.0,
        ];
        let nbest = ctc_beam_nbest(&logp, 2, 3, 0, 3);
        assert_eq!(nbest[0].0, vec![1]);
    }

    #[test]
    fn streaming_matches_batch() {
        let logp = [
            0.0f32, -1.0, -2.0, // t0
            0.0, -0.5, -2.0, // t1
        ];
        let batch = ctc_beam_nbest(&logp, 2, 3, 0, 4);
        let mut s = StreamingCtcBeam::new(0, 4);
        s.push_many(&logp, 2, 3);
        let (ids, _) = s.best();
        assert_eq!(ids, batch[0].0);
    }
}
