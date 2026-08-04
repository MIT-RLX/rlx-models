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

//! Stateless-predictor RNN-T greedy decoder.
//!
//! The k2 / sherpa-onnx "stateless decoder" transducer (used by streaming
//! Zipformer2 models such as Kroko ASR) replaces the recurrent prediction network
//! with a **stateless** one: the decoder input is just the last `context_size`
//! emitted non-blank tokens (blank-padded at the start). Greedy search therefore
//! carries a tiny fixed-width context instead of an LSTM state, and — by
//! convention — the **blank id is 0**.
//!
//! This module owns that decode loop; the caller supplies a
//! [`StatelessTransducerCore`] that runs its stateless predictor over the context
//! and returns the joint argmax for one encoder frame. It is the sibling of the
//! [`super::tdt`] decoder (which handles the token-and-duration variant).

use anyhow::{Result, ensure};

/// The argmax of one joint step in a stateless transducer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransducerStep {
    /// Argmax token over the joint logits (blank included).
    pub label: i32,
    /// The winning logit (carried for optional scoring / beam search).
    pub score: f32,
}

/// A completed greedy decode.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GreedyTransducerResult {
    /// Emitted non-blank token ids, in order.
    pub token_ids: Vec<i32>,
    /// Encoder-frame index at which each token was emitted.
    pub frame_indices: Vec<i64>,
}

/// The stateless predictor + joint a greedy transducer decode drives. The loop
/// owns the decoder context (the last `context_size` non-blank tokens); the
/// implementor runs its stateless predictor over that context and joins it with
/// the encoder frame, returning the argmax.
pub trait StatelessTransducerCore {
    /// Argmax over the joint at `encoder_frame` (`len == hidden_size`) given the
    /// current decoder `context` (`len == context_size`, blank-padded).
    fn step_argmax(&mut self, encoder_frame: &[f32], context: &[i32]) -> TransducerStep;
}

/// Greedy decode a stateless-predictor transducer over a row-major
/// `[frames, hidden_size]` encoder output.
///
/// - `blank_id` — usually `0` for k2/sherpa Zipformer2 packages.
/// - `context_size` — decoder context width (Kroko uses `2`).
/// - `max_symbols_per_frame` — cap on non-blank tokens emitted at one frame.
pub fn run_stateless_transducer_greedy<C: StatelessTransducerCore + ?Sized>(
    core: &mut C,
    encoder_output: &[f32],
    frames: usize,
    hidden_size: usize,
    blank_id: i32,
    context_size: usize,
    max_symbols_per_frame: usize,
) -> Result<GreedyTransducerResult> {
    ensure!(hidden_size > 0, "transducer hidden_size must be positive");
    ensure!(context_size > 0, "transducer context_size must be positive");
    ensure!(
        max_symbols_per_frame > 0,
        "transducer max_symbols_per_frame must be positive"
    );
    ensure!(
        encoder_output.len() == frames * hidden_size,
        "transducer encoder_output shape mismatch: got {} values, expected {}*{} = {}",
        encoder_output.len(),
        frames,
        hidden_size,
        frames * hidden_size,
    );

    // Decoder context: the last `context_size` emitted non-blank tokens,
    // initialised to all-blank (start of sequence).
    let mut context = vec![blank_id; context_size];
    let mut out = GreedyTransducerResult::default();

    for t in 0..frames {
        let frame = &encoder_output[t * hidden_size..(t + 1) * hidden_size];
        let mut emitted = 0usize;
        loop {
            let step = core.step_argmax(frame, &context);
            if step.label == blank_id {
                break;
            }
            out.token_ids.push(step.label);
            out.frame_indices.push(t as i64);
            // Slide the fixed-width context: drop oldest, append the new token.
            context.remove(0);
            context.push(step.label);
            emitted += 1;
            if emitted >= max_symbols_per_frame {
                break;
            }
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLANK: i32 = 0;

    /// A scripted core that replays a fixed queue of steps and records the
    /// context it was called with at each step.
    struct ScriptedCore {
        steps: Vec<TransducerStep>,
        cursor: usize,
        seen_contexts: Vec<Vec<i32>>,
    }
    impl ScriptedCore {
        fn new(labels: &[i32]) -> Self {
            Self {
                steps: labels
                    .iter()
                    .map(|&l| TransducerStep {
                        label: l,
                        score: 1.0,
                    })
                    .collect(),
                cursor: 0,
                seen_contexts: Vec::new(),
            }
        }
    }
    impl StatelessTransducerCore for ScriptedCore {
        fn step_argmax(&mut self, _frame: &[f32], context: &[i32]) -> TransducerStep {
            self.seen_contexts.push(context.to_vec());
            let s = self
                .steps
                .get(self.cursor)
                .copied()
                .unwrap_or(TransducerStep {
                    label: BLANK,
                    score: 0.0,
                });
            self.cursor += 1;
            s
        }
    }

    #[test]
    fn emits_and_slides_context() {
        // Emit 5 then 7 at frame 0, then blank.
        let mut core = ScriptedCore::new(&[5, 7, BLANK]);
        let frames = 1;
        let enc = vec![0.0f32; frames * 3];
        let out = run_stateless_transducer_greedy(&mut core, &enc, frames, 3, BLANK, 2, 8).unwrap();
        assert_eq!(out.token_ids, vec![5, 7]);
        assert_eq!(out.frame_indices, vec![0, 0]);
        // Context started [0,0], then slid to [0,5], then [5,7].
        assert_eq!(core.seen_contexts[0], vec![0, 0]);
        assert_eq!(core.seen_contexts[1], vec![0, 5]);
        assert_eq!(core.seen_contexts[2], vec![5, 7]);
    }

    #[test]
    fn blank_advances_frames() {
        // All blanks → no tokens, one joint call per frame.
        let mut core = ScriptedCore::new(&[]);
        let frames = 4;
        let enc = vec![0.0f32; frames * 2];
        let out = run_stateless_transducer_greedy(&mut core, &enc, frames, 2, BLANK, 2, 8).unwrap();
        assert!(out.token_ids.is_empty());
        assert_eq!(core.seen_contexts.len(), frames);
    }

    #[test]
    fn max_symbols_per_frame_caps_emission() {
        // Never emits blank → the cap must stop it at each frame.
        let mut core = ScriptedCore::new(&[1, 2, 3, 4, 5, 6]);
        let frames = 2;
        let enc = vec![0.0f32; frames * 2];
        let cap = 2;
        let out =
            run_stateless_transducer_greedy(&mut core, &enc, frames, 2, BLANK, 2, cap).unwrap();
        // At most `cap` per frame × 2 frames.
        assert_eq!(out.token_ids.len(), cap * frames);
        assert_eq!(out.frame_indices, vec![0, 0, 1, 1]);
    }

    #[test]
    fn rejects_shape_mismatch() {
        let mut core = ScriptedCore::new(&[]);
        let enc = vec![0.0f32; 5];
        let err = run_stateless_transducer_greedy(&mut core, &enc, 3, 2, BLANK, 2, 8);
        assert!(err.is_err());
    }
}
