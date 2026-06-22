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

//! End-to-end speculative-decoding loop with the `AuxStateBuffer`
//! verifier→speculator bridge.
//!
//! Demonstrates exactly the wiring a real rlx-gemma integration
//! would use, with a synthetic verifier in place of the real
//! Gemma 4 31B decode:
//!
//! 1. Verifier runs one decode step, emits aux hidden states.
//! 2. Aux states are pushed to an `AuxStateBuffer`.
//! 3. The buffer is wrapped as a `VerifierHiddenSource`.
//! 4. `Eagle3Speculator` (with HIR runner) pulls from it during
//!    `propose(n)`.
//! 5. `SpecDecoder::step` runs the Leviathan accept algorithm.
//!
//! Swap step 1 for `rlx_gemma::GemmaRunner::generate_with_aux(...)`
//! and the bridge becomes the real verifier link:
//!
//! ```ignore
//! let writer = aux_buffer.clone();
//! gemma_runner.generate_with_aux(
//!     prompt_ids,
//!     n_new,
//!     vec![2, 30, 57],           // EAGLE3 aux layer ids
//!     |tok| { /* accumulate target tokens */ },
//!     |aux| { writer.write(aux); },
//! )?;
//! ```
//!
//! Run (requires the `gemma` feature for the bridge module):
//! ```bash
//! cargo run -p rlx-eagle3 --release --features "metal mlx gemma" \
//!     --example spec_decoder_end_to_end -- \
//!     /Users/Shared/rlx-models/.eagle3-bench/weights/draft
//! ```

#![cfg(feature = "gemma")]

use anyhow::{Context, Result};
use rlx_eagle3::config::Eagle3Config;
use rlx_eagle3::draft::DraftGeom;
use rlx_eagle3::gemma_bridge::AuxStateBuffer;
use rlx_eagle3::speculator::Eagle3Speculator;
use rlx_eagle3::weights::Eagle3DraftWeights;
use rlx_runtime::Device;
use rlx_runtime::is_available;
use rlx_runtime::spec_decode::{DraftProposal, SpecDecoder, Speculator, VerifyResult};
use std::path::PathBuf;
use std::time::Instant;

/// Stand-in for `rlx_gemma::GemmaRunner::decode_with_aux`. Returns
/// deterministic aux states so each call produces the same EAGLE3
/// proposal — easier to reason about output.
fn synthetic_verifier_decode(target_hidden: usize, n_layers: usize, step: usize) -> Vec<Vec<f32>> {
    (0..n_layers)
        .map(|l| {
            (0..target_hidden)
                .map(|d| {
                    let phase = (step as f32) * 0.01 + (l as f32) * 0.0007;
                    ((d as f32) * 0.001 + phase).sin()
                })
                .collect()
        })
        .collect()
}

/// Identity verifier — accepts every draft proposal. In a real
/// pipeline this would be `rlx-gemma`'s decode against the proposed
/// tokens, returning per-position target probabilities. We
/// short-circuit to "always accept" so the loop runs end-to-end and
/// we can show the per-step accept rate is 100% (trivial case).
struct IdentityTarget {
    target_vocab: usize,
}
impl Speculator for IdentityTarget {
    fn propose(&mut self, _ctx: &[u32], _n: usize) -> DraftProposal {
        unimplemented!("target only verifies")
    }
    fn verify(&mut self, _ctx: &[u32], proposed: &[u32]) -> VerifyResult {
        let probs = proposed
            .iter()
            .map(|&t| {
                let mut r = vec![0.0; self.target_vocab];
                r[t as usize] = 1.0;
                r
            })
            .collect();
        VerifyResult { probs }
    }
}

fn pick_device() -> Device {
    if is_available(Device::Metal) {
        Device::Metal
    } else if is_available(Device::Mlx) {
        Device::Mlx
    } else {
        Device::Cpu
    }
}

fn main() -> Result<()> {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: spec_decoder_end_to_end <draft-dir>")?;

    println!("→ Loading config + draft weights from {:?}", dir);
    let cfg = Eagle3Config::from_file(dir.join("config.json"))?;
    let geom = DraftGeom::from_cfg(&cfg);
    let weights = Eagle3DraftWeights::open(dir.join("model.safetensors"))?;
    let n = cfg.speculative_tokens;
    let n_aux = cfg
        .eagle_aux_hidden_state_layer_ids
        .as_ref()
        .map(|v| v.len())
        .unwrap_or(3);

    // 1. Set up the shared aux buffer + bridge.
    let aux_buffer = AuxStateBuffer::new();
    let writer = aux_buffer.clone();
    let hidden_source = aux_buffer.into_hidden_source(geom.h_target, n_aux);

    // 2. Construct the speculator with the bridge as its hidden source.
    let device = pick_device();
    println!("→ Compiling HIR draft graphs on {device:?} (n_max = {n})");
    let mut draft_speculator =
        Eagle3Speculator::new(cfg.clone(), weights, hidden_source)?.with_hir_runner(device, n)?;
    assert!(draft_speculator.uses_hir());

    // Pre-warm by running propose once (compiles + warms graphs).
    writer.write(synthetic_verifier_decode(geom.h_target, n_aux, 0));
    let _ = draft_speculator.propose(&[1, 2, 3, 4], n);

    // 3. Run the full speculative-decode round-trip.
    let target_vocab = geom.target_vocab;
    let target = IdentityTarget { target_vocab };

    println!("\n→ Running 5 SpecDecoder rounds with the bridge in place");
    let mut decoder = SpecDecoder::new(draft_speculator, target, n, 42);
    let mut context: Vec<u32> = vec![1, 2, 3, 4, 5];
    let mut total_tokens = 0usize;
    let t0 = Instant::now();
    for round in 0..5 {
        // Verifier runs first and writes aux states.
        let aux = synthetic_verifier_decode(geom.h_target, n_aux, round);
        writer.write(aux);

        let emitted = decoder.step(&context);
        println!(
            "   round {round}: emitted {} target tokens (first few: {:?})",
            emitted.len(),
            &emitted[..emitted.len().min(3)],
        );
        total_tokens += emitted.len();
        context.extend(emitted);
    }
    let secs = t0.elapsed().as_secs_f32();
    println!(
        "\n   {total_tokens} accepted tokens across 5 rounds in {secs:.2}s ({:.2} tok/s)",
        total_tokens as f32 / secs.max(1e-6),
    );

    println!(
        "\n✓ DONE — bridge works end-to-end. Swap `synthetic_verifier_decode`\n  \
         for `rlx_gemma::GemmaRunner::decode_with_aux_hidden(...)` once that\n  \
         entry point lands (PLAN.md task #8) and the loop is real Gemma 4 + EAGLE3."
    );
    Ok(())
}
