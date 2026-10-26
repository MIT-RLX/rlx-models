// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! CTC prefix beam search ([`rlx_search::PrefixBeam`]) + segmentation helpers.

use rlx_search::PrefixBeam;

/// Labels within this many nats of a frame's best score are the only ones
/// extended; the rest contribute about `e^-9` each.
const PRUNE_MARGIN: f32 = 9.0;

/// Streaming CTC prefix beam — push one frame at a time, read partial / final.
#[derive(Clone, Debug)]
pub struct StreamingCtcBeam {
    inner: PrefixBeam,
}

impl StreamingCtcBeam {
    pub fn new(blank: usize, beam: usize) -> Self {
        Self {
            inner: PrefixBeam::new(blank as u32, beam).prune_margin(PRUNE_MARGIN),
        }
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    pub fn n_frames(&self) -> usize {
        self.inner.frames()
    }

    /// Ingest one log-prob row of length `classes`.
    pub fn push(&mut self, row: &[f32]) {
        self.inner.push(row);
    }

    pub fn push_many(&mut self, logp: &[f32], seq: usize, classes: usize) {
        self.inner.push_frames(&logp[..seq * classes], classes);
    }

    pub fn best(&self) -> (Vec<usize>, f32) {
        let p = self.inner.best();
        (p.tokens.iter().map(|t| t.id as usize).collect(), p.score)
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
    let mut s = StreamingCtcBeam::new(blank, beam);
    s.push_many(logp, seq, classes);
    s.inner
        .nbest()
        .into_iter()
        .map(|p| (p.tokens.iter().map(|t| t.id as usize).collect(), p.score))
        .collect()
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
