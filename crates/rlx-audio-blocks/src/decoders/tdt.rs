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

//! Token-and-Duration Transducer (TDT) greedy decoder.
//!
//! TDT (Xu et al., "Efficient Sequence Transduction by Jointly Predicting Tokens
//! and Durations", 2023) augments a standard RNN-T joint network with a second
//! head that predicts, for the emitted token, how many encoder frames to *skip*
//! before the next joint step. Decoding therefore advances the time index by a
//! learned `duration ∈ durations[..]` instead of always by one frame, which is
//! what makes Parakeet-TDT and TDT-variant Nemotron models fast.
//!
//! This module is the model-agnostic decode loop only. The caller supplies a
//! [`TdtDecoderCore`] that owns the prediction-network state and performs the
//! joint argmax over one *projected* encoder frame — mirroring the sibling RNN-T
//! path in `rlx-nemotron-asr::decoder`. It is a direct port of audio.cpp's
//! `framework/decoders` (`run_tdt_decoder_greedy_duration_loop`).

use anyhow::{Result, bail, ensure};

/// The argmax result of a single TDT joint step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TdtJointStep {
    /// Argmax over the `n_classes` token logits (the last class is the blank).
    pub label: i32,
    /// The winning token logit (pre-softmax); carried for optional scoring.
    pub label_score: f32,
    /// Argmax over the duration head — an index into the `durations` table.
    pub duration_index: i32,
}

/// A completed TDT decode: emitted tokens plus their frame timestamps and the
/// duration (frames consumed) attached to each.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TdtDecodeResult {
    /// Emitted non-blank token ids, in order.
    pub token_ids: Vec<i32>,
    /// Encoder-frame index at which each token was emitted.
    pub token_timestamps: Vec<i32>,
    /// Frames consumed (`durations[duration_index]`) after emitting each token.
    pub token_durations: Vec<i32>,
}

/// The prediction + joint "core" that a TDT decoder drives. Implementors own the
/// prediction-network recurrence and expose an argmax over the joint given one
/// projected encoder frame. Mirrors audio.cpp's `TdtDecoderCore` (the subset the
/// greedy duration loop actually uses).
pub trait TdtDecoderCore {
    /// Reset the prediction-network recurrence to its zero state.
    fn reset_state(&mut self);

    /// Prime the recurrence with the start-of-sequence step. `blank_id` is passed
    /// for cores that model SOS as a blank embedding; cores that use a dedicated
    /// zero input may ignore it.
    fn predict_start(&mut self, blank_id: i32);

    /// Advance the recurrence by one emitted (non-blank) token.
    fn predict_token(&mut self, token_id: i32);

    /// Argmax over the joint network for one projected encoder frame
    /// (`encoder_frame.len() == hidden_size`).
    fn joint_step_argmax(&mut self, encoder_frame: &[f32]) -> TdtJointStep;
}

/// Which TDT greedy algorithm to run. Only [`TdtAlgorithm::GreedyDurationLoop`]
/// — the NeMo default and audio.cpp default — is implemented today; the other
/// variants are reserved so callers can pin behaviour explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TdtAlgorithm {
    /// Emit-and-skip greedy loop (NeMo `greedy_batch` duration loop).
    GreedyDurationLoop,
}

/// Dispatch a TDT greedy decode by algorithm. See
/// [`run_tdt_greedy_duration_loop`] for argument semantics.
pub fn run_tdt_decoder<C: TdtDecoderCore + ?Sized>(
    algorithm: TdtAlgorithm,
    core: &mut C,
    encoder_projected: &[f32],
    frames: usize,
    hidden_size: usize,
    blank_id: i32,
    durations: &[i32],
    max_symbols_per_step: usize,
) -> Result<TdtDecodeResult> {
    match algorithm {
        TdtAlgorithm::GreedyDurationLoop => run_tdt_greedy_duration_loop(
            core,
            encoder_projected,
            frames,
            hidden_size,
            blank_id,
            durations,
            max_symbols_per_step,
        ),
    }
}

/// Greedy TDT decode over a row-major `[frames, hidden_size]` encoder projection.
///
/// - `encoder_projected` — the acoustic encoder output already projected into the
///   joint's hidden space, laid out frame-major (`frames * hidden_size` values).
/// - `blank_id` — the blank class id (typically `n_classes - 1`).
/// - `durations` — the duration table the duration head indexes into (e.g.
///   `[0, 1, 2, 3, 4]`). A predicted `duration == 0` on a *blank* is clamped to 1
///   so the loop always makes progress.
/// - `max_symbols_per_step` — cap on non-blank tokens emitted at one time index
///   before the loop is forced to advance (guards against zero-duration loops).
///
/// Faithful port of audio.cpp `run_tdt_decoder_greedy_duration_loop`.
pub fn run_tdt_greedy_duration_loop<C: TdtDecoderCore + ?Sized>(
    core: &mut C,
    encoder_projected: &[f32],
    frames: usize,
    hidden_size: usize,
    blank_id: i32,
    durations: &[i32],
    max_symbols_per_step: usize,
) -> Result<TdtDecodeResult> {
    ensure!(hidden_size > 0, "TDT decoder hidden_size must be positive");
    ensure!(
        encoder_projected.len() == frames * hidden_size,
        "TDT decoder encoder_projected shape mismatch: got {} values, expected frames*hidden = {}*{} = {}",
        encoder_projected.len(),
        frames,
        hidden_size,
        frames * hidden_size,
    );
    ensure!(
        max_symbols_per_step > 0,
        "TDT decoder max_symbols_per_step must be positive"
    );
    ensure!(
        !durations.is_empty(),
        "TDT decoder durations table is empty"
    );

    core.reset_state();
    core.predict_start(blank_id);

    let mut result = TdtDecodeResult {
        token_ids: Vec::with_capacity(frames),
        token_timestamps: Vec::with_capacity(frames),
        token_durations: Vec::with_capacity(frames),
    };

    // Signed cursors: `last_label_time_idx` starts at -1 ("no label emitted
    // yet"), matching the reference so the zero-duration guard triggers correctly.
    let mut time_idx: i64 = 0;
    let mut last_label_time_idx: i64 = -1;
    let mut labels_at_current_time_idx: usize = 0;
    let frames_i = frames as i64;

    while time_idx < frames_i {
        // Inner loop emits at most `max_symbols_per_step` tokens at this frame
        // before `break`ing back out to re-check the outer time bound.
        loop {
            if time_idx >= frames_i {
                break;
            }
            let start = (time_idx as usize) * hidden_size;
            let frame = &encoder_projected[start..start + hidden_size];
            let step = core.joint_step_argmax(frame);

            let di = step.duration_index;
            ensure!(
                di >= 0 && (di as usize) < durations.len(),
                "TDT duration_index {di} out of range for durations len {}",
                durations.len()
            );
            let mut duration = durations[di as usize];

            if step.label == blank_id {
                // A zero-duration blank would stall; force one frame of progress.
                if duration == 0 {
                    duration = 1;
                }
                time_idx += duration as i64;
                continue;
            }

            // Emit a real token.
            result.token_ids.push(step.label);
            result.token_timestamps.push(time_idx as i32);
            result.token_durations.push(duration);
            core.predict_token(step.label);

            if time_idx == last_label_time_idx {
                labels_at_current_time_idx += 1;
            } else {
                last_label_time_idx = time_idx;
                labels_at_current_time_idx = 1;
            }

            time_idx += duration as i64;

            // If we've hit the per-frame symbol cap and duration didn't move us,
            // force-advance so we can never spin on a single frame.
            if labels_at_current_time_idx >= max_symbols_per_step && time_idx == last_label_time_idx
            {
                time_idx += 1;
            }

            break;
        }
    }

    // Sanity: parallel arrays stay in lockstep.
    if result.token_ids.len() != result.token_timestamps.len()
        || result.token_ids.len() != result.token_durations.len()
    {
        bail!("TDT decoder produced desynchronised result arrays");
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scripted core that replays a fixed queue of joint steps, ignoring the
    /// encoder frame. Lets us exercise the decode loop deterministically without
    /// any real prediction/joint weights.
    struct ScriptedCore {
        steps: Vec<TdtJointStep>,
        cursor: usize,
        predicted: Vec<i32>,
        started: bool,
    }

    impl ScriptedCore {
        fn new(steps: Vec<TdtJointStep>) -> Self {
            Self {
                steps,
                cursor: 0,
                predicted: Vec::new(),
                started: false,
            }
        }
    }

    impl TdtDecoderCore for ScriptedCore {
        fn reset_state(&mut self) {
            self.cursor = 0;
            self.predicted.clear();
            self.started = false;
        }
        fn predict_start(&mut self, _blank_id: i32) {
            self.started = true;
        }
        fn predict_token(&mut self, token_id: i32) {
            self.predicted.push(token_id);
        }
        fn joint_step_argmax(&mut self, _encoder_frame: &[f32]) -> TdtJointStep {
            // Once the script is exhausted, always return a blank that advances
            // one frame, so decoding terminates by consuming the remaining time.
            let s = self
                .steps
                .get(self.cursor)
                .copied()
                .unwrap_or(TdtJointStep {
                    label: BLANK,
                    label_score: 0.0,
                    duration_index: 1, // durations[1] == 1
                });
            self.cursor += 1;
            s
        }
    }

    const BLANK: i32 = 100;
    // Index into this table via `duration_index`.
    const DURATIONS: [i32; 5] = [0, 1, 2, 3, 4];

    fn step(label: i32, dur_idx: i32) -> TdtJointStep {
        TdtJointStep {
            label,
            label_score: 1.0,
            duration_index: dur_idx,
        }
    }

    #[test]
    fn emits_tokens_and_skips_by_duration() {
        // frame0: token 7 with duration 2 -> jump to frame 2
        // frame2: token 9 with duration 3 -> jump to frame 5 (== frames, stop)
        let mut core = ScriptedCore::new(vec![step(7, 2), step(9, 3)]);
        let frames = 5;
        let enc = vec![0.0f32; frames * 4];
        let out =
            run_tdt_greedy_duration_loop(&mut core, &enc, frames, 4, BLANK, &DURATIONS, 5).unwrap();
        assert_eq!(out.token_ids, vec![7, 9]);
        assert_eq!(out.token_timestamps, vec![0, 2]);
        assert_eq!(out.token_durations, vec![2, 3]);
        // The core must have been advanced by each emitted token.
        assert_eq!(core.predicted, vec![7, 9]);
        assert!(core.started);
    }

    #[test]
    fn blank_with_zero_duration_still_advances() {
        // A blank predicting duration index 0 (== 0 frames) must be clamped to a
        // single-frame advance, otherwise the loop would spin forever.
        let mut core = ScriptedCore::new(vec![step(BLANK, 0), step(BLANK, 0)]);
        let frames = 3;
        let enc = vec![0.0f32; frames * 2];
        let out =
            run_tdt_greedy_duration_loop(&mut core, &enc, frames, 2, BLANK, &DURATIONS, 5).unwrap();
        // No tokens emitted; loop terminated by consuming all frames.
        assert!(out.token_ids.is_empty());
    }

    #[test]
    fn max_symbols_per_step_breaks_zero_duration_token_loop() {
        // Repeated zero-duration *token* emissions at the same frame must be
        // capped by max_symbols_per_step and then force a time advance.
        let cap = 3;
        let mut core = ScriptedCore::new(vec![
            step(1, 0),
            step(2, 0),
            step(3, 0),
            step(4, 0), // would be a 4th at the same frame — cap forces advance
        ]);
        let frames = 4;
        let enc = vec![0.0f32; frames * 2];
        let out = run_tdt_greedy_duration_loop(&mut core, &enc, frames, 2, BLANK, &DURATIONS, cap)
            .unwrap();
        // At most `cap` tokens are emitted at time index 0 before it advances.
        let at_t0 = out.token_timestamps.iter().filter(|&&t| t == 0).count();
        assert!(at_t0 <= cap, "emitted {at_t0} tokens at t=0, cap was {cap}");
        // And the whole thing terminates (no hang) with some tokens emitted.
        assert!(!out.token_ids.is_empty());
    }

    #[test]
    fn rejects_shape_mismatch() {
        let mut core = ScriptedCore::new(vec![]);
        let enc = vec![0.0f32; 7]; // not frames*hidden
        let err = run_tdt_greedy_duration_loop(&mut core, &enc, 3, 4, BLANK, &DURATIONS, 5);
        assert!(err.is_err());
    }

    #[test]
    fn rejects_duration_index_out_of_range() {
        let mut core = ScriptedCore::new(vec![step(5, 99)]); // 99 >> durations.len()
        let enc = vec![0.0f32; 4];
        let err = run_tdt_greedy_duration_loop(&mut core, &enc, 1, 4, BLANK, &DURATIONS, 5);
        assert!(err.is_err());
    }

    #[test]
    fn dispatch_matches_direct_call() {
        let steps = vec![step(7, 2), step(9, 3)];
        let frames = 5;
        let enc = vec![0.0f32; frames * 4];
        let mut a = ScriptedCore::new(steps.clone());
        let mut b = ScriptedCore::new(steps);
        let via_dispatch = run_tdt_decoder(
            TdtAlgorithm::GreedyDurationLoop,
            &mut a,
            &enc,
            frames,
            4,
            BLANK,
            &DURATIONS,
            5,
        )
        .unwrap();
        let direct =
            run_tdt_greedy_duration_loop(&mut b, &enc, frames, 4, BLANK, &DURATIONS, 5).unwrap();
        assert_eq!(via_dispatch, direct);
    }
}
