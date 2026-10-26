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

/// Class 0 is the CTC blank.
const BLANK: u32 = 0;

fn hypothesis(path: rlx_search::CtcPath) -> CtcHypothesis {
    CtcHypothesis {
        steps: path
            .tokens
            .into_iter()
            .map(|t| CtcStep {
                label: t.id,
                pos: t.frame,
            })
            .collect(),
        score: path.score,
    }
}

fn decode_greedy(prob_seq: NdTensorView<f32, 2>) -> CtcHypothesis {
    let classes = prob_seq.size(1);
    hypothesis(rlx_search::ctc::greedy(&prob_seq.to_vec(), classes, BLANK))
}

/// Prefix beam search; `None` for a zero width. Prefixes are label sequences
/// (each step's `pos` is the frame that first emitted it), as in rten's
/// `CtcDecoder::decode_beam`.
fn decode_beam(prob_seq: NdTensorView<f32, 2>, beam_size: u32) -> Option<CtcHypothesis> {
    if beam_size == 0 {
        return None;
    }
    let classes = prob_seq.size(1);
    let mut beam = rlx_search::PrefixBeam::new(BLANK, beam_size as usize);
    beam.push_frames(&prob_seq.to_vec(), classes);
    Some(hypothesis(beam.best()))
}
