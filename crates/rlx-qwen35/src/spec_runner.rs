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

//! End-to-end MTP speculative decoder for Qwen3.5 / Qwen3.6 (PLAN.md M6).
//!
//! Wires the existing [`crate::spec::Qwen35MtpDraft`] (draft head over
//! MTP layers) and [`crate::spec::Qwen35TrunkTarget`] (full-trunk
//! verifier with persistent decode cache) into a single
//! [`SpecDecoder`] driver, then exposes a streaming
//! [`Qwen35SpecRunner::generate`] entry point that matches the shape
//! of `Qwen35Runner::generate`.
//!
//! ## Why two runners
//!
//! Each [`Speculator`] owns its own [`Qwen35Runner`] because draft and
//! target keep distinct decode caches — the draft speculates with
//! checkpoint/restore on a cheap MTP forward, the target advances a
//! persistent cache across rounds and verifies via the full trunk.
//! Sharing one runner would force the draft's checkpoint/restore
//! semantics onto the target, which doesn't match what the
//! `Speculator` impls actually do today (see `src/spec.rs`).
//!
//! ## Memory cost (be aware)
//!
//! Two `Qwen35Runner` instances mean two copies of the dequantised
//! weights in `Device` memory. At Q4_K_M:
//!
//! | model size | per-runner footprint | two-runner total |
//! |-----------:|---------------------:|-----------------:|
//! | 4 B        | ~3 GB                | ~6 GB            |
//! | 9 B        | ~6 GB                | ~12 GB           |
//! | 27 B       | ~17 GB               | ~34 GB           |
//!
//! Use `.packed_weights(true)` to keep K-quants packed in the arena
//! (cuts host memory ~6×). A shared-weight refactor — wrapping the
//! runtime's weight arena in `Arc<...>` and constructing a second
//! `Qwen35Runner` that borrows it — is tracked in PLAN.md M6 follow-up.

use crate::runner::{Qwen35Runner, Qwen35RunnerBuilder};
use crate::spec::{Qwen35MtpDraft, Qwen35TrunkTarget};
use anyhow::{Context, Result, anyhow};
use rlx_cli::LmRunner;
use rlx_runtime::Device;
use rlx_runtime::spec_decode::SpecDecoder;
use std::path::PathBuf;

/// End-to-end MTP speculative-decoding runner for Qwen3.5 / Qwen3.6.
pub struct Qwen35SpecRunner {
    decoder: SpecDecoder<Qwen35MtpDraft, Qwen35TrunkTarget>,
}

impl Qwen35SpecRunner {
    pub fn builder() -> Qwen35SpecRunnerBuilder {
        Qwen35SpecRunnerBuilder::default()
    }

    /// Borrow the underlying decoder — useful for advanced callers
    /// that want to drive a single `step()` themselves or inspect
    /// the per-round accept ratio.
    pub fn decoder(&self) -> &SpecDecoder<Qwen35MtpDraft, Qwen35TrunkTarget> {
        &self.decoder
    }

    pub fn decoder_mut(&mut self) -> &mut SpecDecoder<Qwen35MtpDraft, Qwen35TrunkTarget> {
        &mut self.decoder
    }

    /// Generate up to `n_new` tokens starting from `prompt_ids`.
    /// `on_token` fires once per accepted token (multiple per
    /// speculative round when accept rate is high). Returns the
    /// generated token sequence (excluding the prompt). Stops early
    /// if the decoder produces zero accepted tokens for a round
    /// (defensive — would indicate misconfigured Speculator state).
    pub fn generate<F: FnMut(u32)>(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        mut on_token: F,
    ) -> Result<Vec<u32>> {
        let mut context: Vec<u32> = prompt_ids.to_vec();
        let mut produced: Vec<u32> = Vec::with_capacity(n_new);
        while produced.len() < n_new {
            let accepted = self.decoder.step(&context);
            if accepted.is_empty() {
                break;
            }
            for tok in &accepted {
                if produced.len() == n_new {
                    break;
                }
                produced.push(*tok);
                on_token(*tok);
                context.push(*tok);
            }
        }
        Ok(produced)
    }
}

#[derive(Default)]
pub struct Qwen35SpecRunnerBuilder {
    weights: Option<PathBuf>,
    device: Option<Device>,
    max_seq: Option<usize>,
    /// Number of tokens the draft proposes per round (`n` in
    /// [`SpecDecoder`]). Default 4 — matches the llama.cpp MTP fast
    /// path. Larger values speculate further at higher reject cost.
    draft_len: Option<usize>,
    /// Seed for the accept/reject sampler. Default 1.
    seed: Option<u64>,
    /// Forward to both per-runner `packed_weights(...)` builders.
    /// Default false — set true to cut host memory ~6× on Q4_K_M.
    packed_weights: bool,
}

impl Qwen35SpecRunnerBuilder {
    pub fn weights(mut self, path: impl Into<PathBuf>) -> Self {
        self.weights = Some(path.into());
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }
    pub fn max_seq(mut self, n: usize) -> Self {
        self.max_seq = Some(n);
        self
    }
    pub fn draft_len(mut self, n: usize) -> Self {
        self.draft_len = Some(n);
        self
    }
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }
    pub fn packed_weights(mut self, on: bool) -> Self {
        self.packed_weights = on;
        self
    }

    pub fn build(self) -> Result<Qwen35SpecRunner> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| anyhow!("weights path required (call .weights(...))"))?
            .clone();
        let device = self.device.unwrap_or(Device::Cpu);
        let max_seq = self.max_seq.unwrap_or(128);
        let draft_len = self.draft_len.unwrap_or(4);
        let seed = self.seed.unwrap_or(1);

        let mk_runner = |label: &'static str| -> Result<Qwen35Runner> {
            let mut b = Qwen35RunnerBuilder::default()
                .weights(weights.clone())
                .device(device)
                .max_seq(max_seq)
                .enable_mtp(true);
            if self.packed_weights {
                b = b.packed_weights(true);
            }
            b.build()
                .with_context(|| format!("Qwen35SpecRunner: building {label} runner"))
        };

        let draft_runner = mk_runner("draft")?;
        let target_runner = mk_runner("target")?;
        let draft = Qwen35MtpDraft::new(draft_runner);
        let target = Qwen35TrunkTarget::new(target_runner);
        let decoder = SpecDecoder::new(draft, target, draft_len, seed);

        Ok(Qwen35SpecRunner { decoder })
    }
}

impl LmRunner for Qwen35SpecRunner {
    fn family(&self) -> &'static str {
        "qwen35-spec"
    }
    fn vocab_size(&self) -> usize {
        self.decoder.target.runner().lm_vocab_size()
    }
    /// Routes to the target runner's prefill — speculation only
    /// helps during decode, not single-shot logits. Mirrors what
    /// `Qwen35Runner::predict_logits` would return at this prompt.
    fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        let out = self
            .decoder
            .target
            .runner_mut()
            .predict_logits(prompt_ids)?;
        Ok(out.logits)
    }
    fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        // Use the inherent SpecRunner generate (speculative decode).
        // The inherent signature takes `FnMut(u32)` so we drop the
        // bool — speculative-decode stop semantics need to interact
        // with the accept/reject sampler, which is its own design.
        Qwen35SpecRunner::generate(self, prompt_ids, n_new, |tok| {
            let _ = on_token(tok);
        })
    }
}
