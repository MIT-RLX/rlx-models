// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! Joint CTC / AED beam search + rescoring scales.

use crate::beam::ctc_beam_nbest;
use crate::spec::{
    AED_SCALE, BEAM, BLANK, CTC_SCALE, EOS, RESCORE_AED_SCALE, RESCORE_CTC_SCALE, SOS, VOCAB,
};

/// First-pass CTC hypotheses.
pub fn ctc_first_pass(wp_logprob: &[f32], n_frames: usize) -> Vec<(Vec<usize>, f32)> {
    ctc_beam_nbest(wp_logprob, n_frames, VOCAB, BLANK as usize, BEAM)
}

/// Combine CTC and AED sequence scores (first-pass joint weights).
pub fn joint_score(ctc: f32, aed: f32) -> f32 {
    CTC_SCALE * ctc + AED_SCALE * aed
}

/// Second-pass rescoring weights.
pub fn rescore_score(ctc: f32, aed: f32) -> f32 {
    RESCORE_CTC_SCALE * ctc + RESCORE_AED_SCALE * aed
}

/// Seed AED beams with SOS.
pub fn aed_start_tokens() -> [u32; BEAM] {
    [SOS; BEAM]
}

/// True if any beam has emitted EOS.
pub fn any_eos(tokens: &[u32]) -> bool {
    tokens.contains(&EOS)
}

/// Argmax of one beam's logprob row.
pub fn argmax_token(logprob: &[f32], beam: usize) -> u32 {
    let row = &logprob[beam * VOCAB..(beam + 1) * VOCAB];
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}
