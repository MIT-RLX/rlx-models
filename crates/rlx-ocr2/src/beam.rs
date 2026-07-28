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

//! CTC prefix beam search producing N-best label sequences (for rescoring).

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

/// Prefix beam search over `[seq, classes]` log-probs. Returns collapsed, blank-free
/// label sequences with their total log-probability, best first.
pub fn ctc_beam_nbest(
    logp: &[f32],
    seq: usize,
    classes: usize,
    blank: usize,
    beam: usize,
) -> Vec<(Vec<usize>, f32)> {
    // prefix -> (log p_blank_end, log p_nonblank_end)
    let mut beams: HashMap<Vec<usize>, (f32, f32)> = HashMap::new();
    beams.insert(Vec::new(), (0.0, f32::NEG_INFINITY));

    // Only labels within `PRUNE_MARGIN` nats of the per-timestep max can matter; the rest
    // contribute e^-margin ≈ 0. This shrinks the inner loop from ~439 to a handful.
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
            // extend with blank -> same prefix, blank-ended
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
                    // same symbol extends the current run (from non-blank end)
                    bump(&mut next, prefix.clone(), f32::NEG_INFINITY, pnb + lp);
                    // a *new* occurrence of the same symbol requires a preceding blank
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
