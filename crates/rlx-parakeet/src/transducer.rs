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

//! The TDT transducer: binds the reused LSTM prediction network to the
//! [`TdtJoint`] and drives the shared TDT greedy decode loop.

use anyhow::{Result, ensure};
use rlx_audio_blocks::decoders::tdt::{
    TdtDecodeResult, TdtDecoderCore, TdtJointStep, run_tdt_greedy_duration_loop,
};
use rlx_nemotron_asr::decoder::{PredState, PredictionNet};

use crate::joint::TdtJoint;

/// A [`TdtDecoderCore`] over a Parakeet prediction net + [`TdtJoint`].
///
/// The prediction network is the token-serial recurrence; the joint is applied
/// once per candidate frame. State (the LSTM `(h, c)` per layer and the current
/// prediction hidden) is carried across the whole utterance, primed by the SOS
/// step, exactly as NeMo's transducer greedy search does.
pub struct TdtCore<'a> {
    pred: &'a PredictionNet,
    joint: &'a TdtJoint,
    state: PredState,
    pred_out: Vec<f32>,
}

impl<'a> TdtCore<'a> {
    pub fn new(pred: &'a PredictionNet, joint: &'a TdtJoint) -> Self {
        Self {
            state: pred.zero_state(),
            pred_out: Vec::new(),
            pred,
            joint,
        }
    }
}

impl TdtDecoderCore for TdtCore<'_> {
    fn reset_state(&mut self) {
        self.state = self.pred.zero_state();
        self.pred_out.clear();
    }

    fn predict_start(&mut self, _blank_id: i32) {
        // SOS: step the recurrence from the zero state with a zero input, and
        // *carry* the resulting state forward (NeMo semantics).
        let (po, st) = self.pred.step(None, &self.state);
        self.pred_out = po;
        self.state = st;
    }

    fn predict_token(&mut self, token_id: i32) {
        let label = usize::try_from(token_id).ok();
        let (po, st) = self.pred.step(label, &self.state);
        self.pred_out = po;
        self.state = st;
    }

    fn joint_step_argmax(&mut self, encoder_frame: &[f32]) -> TdtJointStep {
        let enc_proj = self.joint.enc_proj(encoder_frame);
        self.joint.argmax_step(&enc_proj, &self.pred_out)
    }
}

/// Greedy TDT decode over an acoustic encoder output `enc` laid out
/// `[frames, d_model]` (row-major). `durations` is the duration table the joint's
/// duration head indexes into (e.g. `[0, 1, 2, 3, 4]`), and `max_symbols_per_step`
/// caps tokens emitted at a single frame. Returns emitted tokens with their frame
/// timestamps and per-token durations.
pub fn tdt_greedy_decode(
    pred: &PredictionNet,
    joint: &TdtJoint,
    enc: &[f32],
    durations: &[i32],
    max_symbols_per_step: usize,
) -> Result<TdtDecodeResult> {
    let d_model = joint.d_model();
    ensure!(d_model > 0, "joint d_model must be positive");
    ensure!(
        enc.len().is_multiple_of(d_model),
        "encoder output length {} not a multiple of d_model {d_model}",
        enc.len()
    );
    ensure!(
        durations.len() == joint.num_durations,
        "durations table len {} != joint num_durations {}",
        durations.len(),
        joint.num_durations
    );
    let frames = enc.len() / d_model;
    let blank = joint.blank_id() as i32;

    let mut core = TdtCore::new(pred, joint);
    run_tdt_greedy_duration_loop(
        &mut core,
        enc,
        frames,
        d_model,
        blank,
        durations,
        max_symbols_per_step,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_nemotron_asr::decoder::LstmCell;

    /// A tiny prediction net: `vocab` tokens, `dim`-wide embed == LSTM hidden,
    /// one all-zero LSTM layer (so it never blows up regardless of inputs).
    fn tiny_pred(vocab: usize, dim: usize) -> PredictionNet {
        let lstm = LstmCell::new(
            dim,
            dim,
            vec![0.0; 4 * dim * dim],
            vec![0.0; 4 * dim * dim],
            vec![0.0; 4 * dim],
            vec![0.0; 4 * dim],
        )
        .unwrap();
        PredictionNet::new(vec![0.0; vocab * dim], dim, vocab, vec![lstm]).unwrap()
    }

    /// A joint whose classifier weight is all-zero, so token/duration argmax are
    /// driven purely by the bias — giving us a deterministic, input-independent
    /// oracle to exercise the decode loop through the *real* joint + core.
    ///
    /// `token_bias` picks the winning token class; `dur_bias` picks the winning
    /// duration index.
    fn oracle_joint(
        d_model: usize,
        joint_hidden: usize,
        pred_in: usize,
        n_classes: usize,
        durations: usize,
        token_bias: Vec<f32>,
        dur_bias: Vec<f32>,
    ) -> TdtJoint {
        assert_eq!(token_bias.len(), n_classes);
        assert_eq!(dur_bias.len(), durations);
        let n_out = n_classes + durations;
        let mut net_b = token_bias;
        net_b.extend(dur_bias);
        TdtJoint::from_raw(
            vec![0.0; joint_hidden * d_model],
            vec![0.0; joint_hidden],
            d_model,
            vec![0.0; joint_hidden * pred_in],
            vec![0.0; joint_hidden],
            pred_in,
            vec![0.0; n_out * joint_hidden], // classifier weight all zero
            net_b,
            joint_hidden,
            n_classes,
            durations,
        )
        .unwrap()
    }

    #[test]
    fn emits_token_and_skips_by_duration() {
        // n_classes = 3 (tokens 0,1 + blank=2), durations table [0,1,2,3,4].
        // token bias favors class 0; duration bias favors index 2 -> duration 2.
        let d_model = 2;
        let pred = tiny_pred(2, 2);
        let joint = oracle_joint(
            d_model,
            2,
            pred.hidden(),
            3,
            5,
            vec![1.0, 0.0, -1.0],          // token 0 wins (not blank)
            vec![0.0, 0.0, 1.0, 0.0, 0.0], // duration index 2 -> durations[2]=2
        );
        let durations = [0, 1, 2, 3, 4];
        let frames = 6;
        let enc = vec![0.5f32; frames * d_model];

        let out = tdt_greedy_decode(&pred, &joint, &enc, &durations, 5).unwrap();
        // Emit at frames 0,2,4 then step to 6 == frames and stop.
        assert_eq!(out.token_ids, vec![0, 0, 0]);
        assert_eq!(out.token_timestamps, vec![0, 2, 4]);
        assert_eq!(out.token_durations, vec![2, 2, 2]);
    }

    #[test]
    fn all_blank_consumes_frames_without_tokens() {
        let d_model = 2;
        let pred = tiny_pred(2, 2);
        // token bias favors the blank (class 2); duration index 1 -> duration 1.
        let joint = oracle_joint(
            d_model,
            2,
            pred.hidden(),
            3,
            5,
            vec![-1.0, -1.0, 1.0],         // blank wins
            vec![0.0, 1.0, 0.0, 0.0, 0.0], // duration 1
        );
        let durations = [0, 1, 2, 3, 4];
        let frames = 4;
        let enc = vec![0.0f32; frames * d_model];
        let out = tdt_greedy_decode(&pred, &joint, &enc, &durations, 5).unwrap();
        assert!(out.token_ids.is_empty());
    }

    #[test]
    fn zero_duration_tokens_are_capped_then_advance() {
        // token 0 wins with duration index 0 (== 0 frames). Without the symbol
        // cap this would spin forever at frame 0; the loop must cap and advance.
        let d_model = 2;
        let pred = tiny_pred(2, 2);
        let joint = oracle_joint(
            d_model,
            2,
            pred.hidden(),
            3,
            5,
            vec![1.0, 0.0, -1.0],          // token 0 wins
            vec![1.0, 0.0, 0.0, 0.0, 0.0], // duration index 0 -> 0 frames
        );
        let durations = [0, 1, 2, 3, 4];
        let frames = 3;
        let enc = vec![0.0f32; frames * d_model];
        let cap = 2;
        let out = tdt_greedy_decode(&pred, &joint, &enc, &durations, cap).unwrap();
        // At most `cap` tokens at frame 0, and the decode terminates.
        let at_t0 = out.token_timestamps.iter().filter(|&&t| t == 0).count();
        assert!(at_t0 <= cap, "emitted {at_t0} at t=0 (cap {cap})");
        assert!(!out.token_ids.is_empty());
    }

    #[test]
    fn rejects_durations_table_mismatch() {
        let pred = tiny_pred(2, 2);
        let joint = oracle_joint(2, 2, pred.hidden(), 3, 5, vec![0.0; 3], vec![0.0; 5]);
        let enc = vec![0.0f32; 4];
        // durations table length (3) != joint.num_durations (5)
        let err = tdt_greedy_decode(&pred, &joint, &enc, &[0, 1, 2], 5);
        assert!(err.is_err());
    }
}
