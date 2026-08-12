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

//! The block-diffusion generation loop.
//!
//! ```text
//! for each canvas (block):
//!     encoder: extend the KV cache with everything generated so far
//!     canvas  ← uniform random tokens
//!     for step = max_denoising_steps … 1:
//!         logits, soft_embeds ← denoiser(canvas, sc_signal, temperature(step))
//!         accept the low-entropy positions, re-noise the rest
//!         sc_signal ← soft_embeds
//!         stop early once the draft is stable and confident
//!     append the argmax draft to the context
//!     stop at EOS
//! ```
//!
//! The loop is written against the [`Denoiser`] trait rather than a compiled
//! model so it can be exercised without weights — see the tests at the bottom.

use anyhow::Result;

use crate::config::DiffusionGenerationConfig;
use crate::sampler::{EntropyBoundSampler, Rng, StableAndConfident, StepScores};

/// One denoiser forward pass.
#[derive(Debug, Clone)]
pub struct DenoiserOutput {
    /// Processed (soft-capped, temperature-scaled) logits, `[canvas · vocab]`.
    pub logits: Vec<f32>,
    /// `softmax(logits) @ embed_tokens · sqrt(hidden)`, `[canvas · hidden]` —
    /// the next step's self-conditioning signal.
    pub soft_embeds: Vec<f32>,
}

/// A compiled decoder graph, or a stand-in for tests.
pub trait Denoiser {
    /// Run one denoising step over `canvas` with the previous step's
    /// `sc_signal` (all zeros on the first step) at `temperature`.
    fn step(
        &mut self,
        canvas: &[u32],
        sc_signal: &[f32],
        temperature: f32,
    ) -> Result<DenoiserOutput>;

    /// Extend the read-only encoder cache with `tokens`. Called once before each
    /// block: with the prompt first, then with each finished canvas.
    fn encode(&mut self, tokens: &[u32]) -> Result<()>;
}

/// What a block produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenoiseOutcome {
    /// The block's tokens (the final argmax draft), EOS-truncated.
    pub tokens: Vec<u32>,
    /// Denoising steps actually run — lower than `max_denoising_steps` when the
    /// adaptive stopping criteria fired.
    pub steps: usize,
    /// An EOS token landed in this block, so generation is finished.
    pub hit_eos: bool,
}

/// Drives the diffusion loop for one sequence.
pub struct BlockDiffusion {
    pub cfg: DiffusionGenerationConfig,
    pub canvas_length: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    sampler: EntropyBoundSampler,
    stopping: StableAndConfident,
    rng: Rng,
}

impl BlockDiffusion {
    pub fn new(
        cfg: DiffusionGenerationConfig,
        canvas_length: usize,
        vocab_size: usize,
        hidden_size: usize,
        seed: u64,
    ) -> Self {
        let sampler =
            EntropyBoundSampler::new(cfg.sampler_config.entropy_bound, canvas_length, vocab_size);
        let stopping = StableAndConfident::new(cfg.stability_threshold, cfg.confidence_threshold);
        Self {
            cfg,
            canvas_length,
            vocab_size,
            hidden_size,
            sampler,
            stopping,
            rng: Rng::seed_from_u64(seed),
        }
    }

    /// Denoise a single block. The caller must have encoded the context first.
    pub fn denoise_block(&mut self, model: &mut impl Denoiser) -> Result<DenoiseOutcome> {
        self.stopping.reset();
        let mut canvas = self.sampler.initialize_canvas(&mut self.rng);
        let mut sc_signal = vec![0f32; self.canvas_length * self.hidden_size];
        let mut draft = canvas.clone();
        let mut steps = 0usize;

        // Denoising runs the reverse diffusion process, so the step index counts
        // *down* — that is also what the temperature schedule expects.
        for step in (1..=self.cfg.max_denoising_steps).rev() {
            steps += 1;
            let temperature = self.cfg.temperature(step);
            let out = model.step(&canvas, &sc_signal, temperature)?;
            anyhow::ensure!(
                out.logits.len() == self.canvas_length * self.vocab_size,
                "denoiser returned {} logits, expected {}",
                out.logits.len(),
                self.canvas_length * self.vocab_size
            );

            let scores = StepScores::from_logits(
                &out.logits,
                self.canvas_length,
                self.vocab_size,
                &mut self.rng,
            );
            let accepted = self
                .sampler
                .accept(&canvas, &scores.sampled, &scores.entropy);
            canvas = self.sampler.renoise(&accepted, &mut self.rng);
            draft = scores.argmax;
            sc_signal = out.soft_embeds;

            if self.stopping.should_stop(&draft, &scores.entropy) {
                break;
            }
        }

        Ok(self.finalize(draft, steps))
    }

    /// Truncate the block at the first EOS and pad the rest, matching
    /// `_finalize_canvas`: the EOS itself is kept, everything after it is pad.
    fn finalize(&self, mut tokens: Vec<u32>, steps: usize) -> DenoiseOutcome {
        let eos = tokens
            .iter()
            .position(|t| self.cfg.eos_token_id.contains(t));
        let hit_eos = eos.is_some();
        if let Some(i) = eos {
            for t in tokens.iter_mut().skip(i + 1) {
                *t = self.cfg.pad_token_id;
            }
        }
        DenoiseOutcome {
            tokens,
            steps,
            hit_eos,
        }
    }

    /// Full generation: encode `prompt`, then denoise blocks until EOS or
    /// `max_new_tokens`. Returns only the generated tokens.
    pub fn generate(&mut self, model: &mut impl Denoiser, prompt: &[u32]) -> Result<Vec<u32>> {
        let max_canvases = self.cfg.max_new_tokens.div_ceil(self.canvas_length).max(1);
        let mut out = Vec::new();
        let mut pending: Vec<u32> = prompt.to_vec();
        for _ in 0..max_canvases {
            // The encoder consumes the prompt first, then each finished canvas —
            // this is what grows the read-only cache the denoiser attends to.
            model.encode(&pending)?;
            let block = self.denoise_block(model)?;
            out.extend_from_slice(&block.tokens);
            if block.hit_eos {
                break;
            }
            pending = block.tokens;
        }
        out.truncate(self.cfg.max_new_tokens);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A denoiser that always predicts `target`, with confidence that ramps up
    /// over `warmup` steps so the stopping criteria have something to trip on.
    struct FakeDenoiser {
        target: Vec<u32>,
        vocab: usize,
        hidden: usize,
        confidence: f32,
        calls: usize,
        encoded: Vec<Vec<u32>>,
    }

    impl Denoiser for FakeDenoiser {
        fn step(
            &mut self,
            canvas: &[u32],
            sc_signal: &[f32],
            temperature: f32,
        ) -> Result<DenoiserOutput> {
            assert_eq!(canvas.len(), self.target.len());
            assert_eq!(sc_signal.len(), self.target.len() * self.hidden);
            assert!(temperature > 0.0);
            self.calls += 1;
            let mut logits = vec![0f32; self.target.len() * self.vocab];
            for (c, &t) in self.target.iter().enumerate() {
                logits[c * self.vocab + t as usize] = self.confidence;
            }
            Ok(DenoiserOutput {
                logits,
                soft_embeds: vec![0.5; self.target.len() * self.hidden],
            })
        }

        fn encode(&mut self, tokens: &[u32]) -> Result<()> {
            self.encoded.push(tokens.to_vec());
            Ok(())
        }
    }

    fn fake(target: Vec<u32>, confidence: f32) -> FakeDenoiser {
        FakeDenoiser {
            target,
            vocab: 8,
            hidden: 2,
            confidence,
            calls: 0,
            encoded: Vec::new(),
        }
    }

    fn cfg(max_steps: usize) -> DiffusionGenerationConfig {
        DiffusionGenerationConfig {
            max_denoising_steps: max_steps,
            max_new_tokens: 4,
            eos_token_id: vec![7],
            ..Default::default()
        }
    }

    #[test]
    fn a_confident_denoiser_stops_early_and_returns_its_draft() {
        let target = vec![1u32, 2, 3, 4];
        let mut m = fake(target.clone(), 60.0); // near one-hot → ~0 entropy
        let mut bd = BlockDiffusion::new(cfg(48), 4, 8, 2, 99);
        let outcome = bd.denoise_block(&mut m).unwrap();
        assert_eq!(outcome.tokens, target);
        // Needs one step to fill the history, a second to compare against it.
        assert_eq!(outcome.steps, 2, "should stop as soon as it is stable");
        assert!(!outcome.hit_eos);
    }

    #[test]
    fn an_unconfident_denoiser_runs_every_step() {
        // Flat logits → entropy ln(8), far above confidence_threshold.
        let mut m = fake(vec![1, 2, 3, 4], 0.0);
        let mut bd = BlockDiffusion::new(cfg(5), 4, 8, 2, 3);
        let outcome = bd.denoise_block(&mut m).unwrap();
        assert_eq!(outcome.steps, 5);
        assert_eq!(m.calls, 5);
    }

    #[test]
    fn eos_truncates_the_block_and_stops_generation() {
        let mut m = fake(vec![1, 7, 3, 4], 60.0);
        let mut bd = BlockDiffusion::new(cfg(48), 4, 8, 2, 1);
        let outcome = bd.denoise_block(&mut m).unwrap();
        assert!(outcome.hit_eos);
        // EOS kept, everything after it padded.
        assert_eq!(outcome.tokens, vec![1, 7, 0, 0]);
    }

    #[test]
    fn generate_encodes_the_prompt_then_each_finished_canvas() {
        let mut m = fake(vec![1, 2, 3, 4], 60.0);
        let mut c = cfg(48);
        c.max_new_tokens = 8; // two canvases of 4
        let mut bd = BlockDiffusion::new(c, 4, 8, 2, 5);
        let out = bd.generate(&mut m, &[9, 9, 9]).unwrap();
        assert_eq!(out, vec![1, 2, 3, 4, 1, 2, 3, 4]);
        assert_eq!(
            m.encoded,
            vec![vec![9, 9, 9], vec![1, 2, 3, 4]],
            "prompt first, then the previous canvas"
        );
    }

    #[test]
    fn generate_stops_at_eos_without_filling_max_new_tokens() {
        let mut m = fake(vec![1, 7, 3, 4], 60.0);
        let mut c = cfg(48);
        c.max_new_tokens = 16;
        let mut bd = BlockDiffusion::new(c, 4, 8, 2, 5);
        let out = bd.generate(&mut m, &[9]).unwrap();
        assert_eq!(out, vec![1, 7, 0, 0]);
        assert_eq!(m.encoded.len(), 1, "only the prompt was encoded");
    }

    #[test]
    fn the_first_step_gets_a_zero_self_conditioning_signal() {
        struct CheckFirst {
            seen: Vec<f32>,
            n: usize,
        }
        impl Denoiser for CheckFirst {
            fn step(&mut self, c: &[u32], sc: &[f32], _t: f32) -> Result<DenoiserOutput> {
                if self.n == 0 {
                    self.seen = sc.to_vec();
                }
                self.n += 1;
                Ok(DenoiserOutput {
                    logits: vec![0f32; c.len() * 8],
                    soft_embeds: vec![1.25; c.len() * 2],
                })
            }
            fn encode(&mut self, _t: &[u32]) -> Result<()> {
                Ok(())
            }
        }
        let mut m = CheckFirst {
            seen: Vec::new(),
            n: 0,
        };
        let mut bd = BlockDiffusion::new(cfg(2), 4, 8, 2, 11);
        bd.denoise_block(&mut m).unwrap();
        assert_eq!(m.seen, vec![0.0; 8], "first step has no prior logits");
        assert_eq!(m.n, 2);
    }

    #[test]
    fn temperature_anneals_across_the_denoising_loop() {
        struct RecordTemps(Vec<f32>);
        impl Denoiser for RecordTemps {
            fn step(&mut self, c: &[u32], _s: &[f32], t: f32) -> Result<DenoiserOutput> {
                self.0.push(t);
                Ok(DenoiserOutput {
                    logits: vec![0f32; c.len() * 8],
                    soft_embeds: vec![0.0; c.len() * 2],
                })
            }
            fn encode(&mut self, _t: &[u32]) -> Result<()> {
                Ok(())
            }
        }
        let mut m = RecordTemps(Vec::new());
        let mut bd = BlockDiffusion::new(cfg(4), 4, 8, 2, 2);
        bd.denoise_block(&mut m).unwrap();
        assert_eq!(m.0.len(), 4);
        assert!(m.0[0] > m.0[3], "temperature must decrease: {:?}", m.0);
        assert!((m.0[0] - 0.8).abs() < 1e-6, "first step is t_max");
    }
}
