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

//! CTC decoding (greedy + beam search).

use crate::config::DecodeMethod;
use rten_tensor::NdTensorView;
use rten_tensor::prelude::*;
use std::collections::HashMap;
use std::num::NonZeroU32;

/// Item in a decoded sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CtcStep {
    pub label: u32,
    pub pos: u32,
}

/// Decoded label sequence with log score.
#[derive(Clone, Debug)]
pub struct CtcHypothesis {
    steps: Vec<CtcStep>,
    score: f32,
}

impl CtcHypothesis {
    pub fn steps(&self) -> &[CtcStep] {
        &self.steps
    }

    pub fn score(&self) -> f32 {
        self.score
    }
}

/// Decode a `[seq, classes]` log-probability matrix.
pub fn decode(input_seq: NdTensorView<f32, 2>, method: DecodeMethod) -> CtcHypothesis {
    // Debug knob: `OCR_DECODE=greedy` or `OCR_DECODE=beam:<width>` overrides the
    // configured method (lets the benches A/B greedy vs beam without rewiring).
    let method = std::env::var("OCR_DECODE")
        .ok()
        .and_then(|s| {
            let s = s.trim();
            if s.eq_ignore_ascii_case("greedy") {
                Some(DecodeMethod::Greedy)
            } else if let Some(w) = s.strip_prefix("beam:") {
                Some(DecodeMethod::BeamSearch {
                    width: w.trim().parse().ok()?,
                })
            } else {
                None
            }
        })
        .unwrap_or(method);
    match method {
        DecodeMethod::Greedy => decode_greedy(input_seq),
        DecodeMethod::BeamSearch { width } => {
            decode_beam(input_seq, width.max(1)).unwrap_or_else(|| decode_greedy(input_seq))
        }
    }
}

fn decode_greedy(prob_seq: NdTensorView<f32, 2>) -> CtcHypothesis {
    let mut last_label = 0;
    let mut steps = Vec::new();
    let mut score = 0.;

    for pos in 0..prob_seq.size(0) {
        let mut best_label = 0usize;
        let mut best_lp = prob_seq[[pos, 0]];
        for label in 1..prob_seq.size(1) {
            let lp = prob_seq[[pos, label]];
            if lp > best_lp {
                best_lp = lp;
                best_label = label;
            }
        }
        let label = best_label;
        score += best_lp;
        if label == last_label {
            continue;
        }
        last_label = label;
        if label > 0 {
            steps.push(CtcStep {
                label: label as u32,
                pos: pos as u32,
            });
        }
    }

    CtcHypothesis { steps, score }
}

#[derive(Debug)]
struct BeamProbs {
    prob_blank: f32,
    prob_no_blank: f32,
}

fn log_sum_exp<const N: usize>(log_probs: [f32; N]) -> f32 {
    if log_probs.iter().all(|&x| x == f32::NEG_INFINITY) {
        f32::NEG_INFINITY
    } else {
        let lp_max = log_probs
            .into_iter()
            .reduce(f32::max)
            .unwrap_or(f32::NEG_INFINITY);
        lp_max
            + log_probs
                .iter()
                .map(|x| (x - lp_max).exp())
                .sum::<f32>()
                .ln()
    }
}

fn decode_beam(prob_seq: NdTensorView<f32, 2>, beam_size: u32) -> Option<CtcHypothesis> {
    let beam_size = NonZeroU32::new(beam_size)?;
    let mut states: HashMap<Vec<CtcStep>, BeamProbs> = HashMap::new();
    states.insert(
        Vec::new(),
        BeamProbs {
            prob_blank: 0.,
            prob_no_blank: f32::NEG_INFINITY,
        },
    );

    for t in 0..prob_seq.size(0) {
        let mut next: HashMap<Vec<CtcStep>, BeamProbs> = HashMap::new();
        let blank_lp = prob_seq[[t, 0]];
        for (prefix, state) in &states {
            let p_b = state.prob_blank;
            let p_nb = state.prob_no_blank;
            merge_beam(
                &mut next,
                prefix.clone(),
                log_sum_exp([p_b + blank_lp, p_nb + blank_lp]),
                f32::NEG_INFINITY,
            );
            for label in 1..prob_seq.size(1) {
                let lp = prob_seq[[t, label]];
                let mut new_prefix = prefix.clone();
                let step = CtcStep {
                    label: label as u32,
                    pos: t as u32,
                };
                let last = new_prefix.last().map(|s| s.label);
                if last != Some(step.label) {
                    new_prefix.push(step);
                }
                let (nb_lp, b_lp) = if last == Some(step.label) {
                    (p_nb + lp, p_b + lp)
                } else {
                    (log_sum_exp([p_b + lp, p_nb + lp]), f32::NEG_INFINITY)
                };
                merge_beam(&mut next, new_prefix, b_lp, nb_lp);
            }
        }
        let mut ranked: Vec<_> = next.into_iter().collect();
        ranked.sort_by(|(_, a), (_, b)| {
            let sa = log_sum_exp([a.prob_blank, a.prob_no_blank]);
            let sb = log_sum_exp([b.prob_blank, b.prob_no_blank]);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked.truncate(beam_size.get() as usize);
        states = ranked.into_iter().collect();
    }

    let (prefix, probs) = states.into_iter().max_by(|(_, a), (_, b)| {
        let sa = log_sum_exp([a.prob_blank, a.prob_no_blank]);
        let sb = log_sum_exp([b.prob_blank, b.prob_no_blank]);
        sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
    })?;
    Some(CtcHypothesis {
        steps: prefix,
        score: log_sum_exp([probs.prob_blank, probs.prob_no_blank]),
    })
}

fn merge_beam(map: &mut HashMap<Vec<CtcStep>, BeamProbs>, prefix: Vec<CtcStep>, pb: f32, pnb: f32) {
    use std::collections::hash_map::Entry;
    match map.entry(prefix) {
        Entry::Vacant(e) => {
            e.insert(BeamProbs {
                prob_blank: pb,
                prob_no_blank: pnb,
            });
        }
        Entry::Occupied(mut e) => {
            let s = e.get_mut();
            s.prob_blank = log_sum_exp([s.prob_blank, pb]);
            s.prob_no_blank = log_sum_exp([s.prob_no_blank, pnb]);
        }
    }
}
