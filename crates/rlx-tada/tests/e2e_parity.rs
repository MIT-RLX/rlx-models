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

//! End-to-end parity vs `TadaForCausalLM.generate` on a miniature model.
//!
//! The component tests pin each sub-model; this one pins the part that is pure
//! bookkeeping and therefore the easiest to get wrong without any symptom: how
//! many positions get prefilled, which token's latent is fed back at which
//! step, when the prompt hands over to prediction, how the prompt's text is
//! masked, and where the generated span starts.
//!
//! Made exactly comparable by setting `noise_temperature = 0`. The
//! flow-matching solve then starts from the zero vector in both
//! implementations and the whole ODE is deterministic, so the outputs can be
//! compared value by value instead of distributionally. The audio is
//! meaningless — the wiring is the point.
//!
//! Requires a Llama-3.2 `tokenizer.json`; set `RLX_TADA_TOKENIZER` to a
//! directory holding one, otherwise the test is skipped.

use ndarray::Array2;
use rlx_llama32::Llama32Config;
mod common;
use common::device;
use rlx_tada::backbone::{Backbone, InputEmbedder};
use rlx_tada::config::TadaConfig;
use rlx_tada::head::{CfgSchedule, DiffusionHead, SolveOptions, TimeSchedule};
use rlx_tada::prompt::VoicePrompt;
use rlx_tada::synth::{SynthOptions, Synthesizer};
use rlx_tada::tokenizer::TadaTokenizer;
use rlx_tada::weights::TensorStore;
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Fixture {
    config: Value,
    prompt_text: String,
    target_text: String,
    prompt_token_ids: Vec<u32>,
    prompt_positions: Vec<u32>,
    prompt_values: Vec<f32>,
    audio_samples: usize,
    latents: Vec<f32>,
    latents_shape: Vec<usize>,
    gaps: Vec<u32>,
}

#[test]
fn matches_the_upstream_generation_loop() {
    let Some(tok_dir) = std::env::var_os("RLX_TADA_TOKENIZER") else {
        eprintln!("RLX_TADA_TOKENIZER not set — skipping end-to-end parity");
        return;
    };
    let f: Fixture = serde_json::from_str(include_str!("fixtures/e2e_reference.json"))
        .expect("parse e2e_reference.json");
    let weights = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/e2e_reference.safetensors");
    if !weights.is_file() {
        // Deliberately not committed (8 MB); regenerate with the crate's
        // fixture script rather than shipping it in the published crate.
        eprintln!("{} missing — skipping end-to-end parity", weights.display());
        return;
    }
    let store = std::sync::Arc::new(TensorStore::open(&weights).expect("open fixture weights"));

    let cfg_text = serde_json::to_string(&f.config).unwrap();
    let cfg: TadaConfig = serde_json::from_str(&cfg_text).expect("TadaConfig");
    let llama_cfg: Llama32Config = serde_json::from_str(&cfg_text).expect("Llama32Config");

    let solve = SolveOptions {
        num_steps: 3,
        acoustic_cfg_scale: 1.6,
        duration_cfg_scale: 1.0,
        cfg_schedule: CfgSchedule::Cosine,
        time_schedule: TimeSchedule::LogSnr,
        // Zero noise makes the solve deterministic and therefore comparable.
        noise_temperature: 0.0,
    };

    let embed = InputEmbedder::load(store.clone(), &cfg).expect("embedder");
    let backbone =
        Backbone::new(store.clone(), llama_cfg, device(), 2, "e2e_fixture").expect("backbone");
    let head = DiffusionHead::load(store, &cfg, "prediction_head.").expect("head");
    let tokenizer = TadaTokenizer::load(std::path::Path::new(&tok_dir)).expect("tokenizer");
    let mut synth = Synthesizer::new(
        cfg.clone(),
        device(),
        embed,
        backbone,
        head,
        None,
        tokenizer,
    );

    let width = cfg.acoustic_dim;
    let prompt = VoicePrompt {
        text: f.prompt_text.clone(),
        token_ids: f.prompt_token_ids.clone(),
        token_positions: f.prompt_positions.clone(),
        token_values: Array2::from_shape_vec(
            (f.prompt_token_ids.len(), width),
            f.prompt_values.clone(),
        )
        .expect("prompt latents"),
        // Not part of the upstream comparison; the fixture carries latents directly.
        alignment_score: f32::NAN,
        audio_samples: f.audio_samples,
        sample_rate: rlx_tada::config::SAMPLE_RATE,
    };

    let opts = SynthOptions {
        solve,
        seed: 1,
        trim_leading_silence: false,
        // Parity compares raw model output; post-processing is a separate concern.
        runaway_trim: None,
    };
    let (latents, gaps) = synth
        .synthesize_latents(&prompt, &f.target_text, &opts)
        .expect("synthesize");

    assert_eq!(
        (latents.nrows(), latents.ncols()),
        (f.latents_shape[0], f.latents_shape[1]),
        "generated span has a different shape than torch — a prefill or skip \
         offset is wrong, not a numeric issue"
    );
    assert_eq!(gaps, f.gaps, "predicted frame gaps disagree");

    let got: Vec<f32> = latents.iter().copied().collect();
    let mut worst = 0f32;
    let mut at = 0usize;
    for (i, (a, b)) in got.iter().zip(&f.latents).enumerate() {
        let d = (a - b).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    // Latents here run to ~4 in magnitude after the std rescale; 5e-3 is
    // accumulation slack across a 2-layer LM plus a 3-step ODE, and orders of
    // magnitude below what any mis-wired offset would produce.
    if worst >= 5e-3 {
        // Which row first diverges says what broke: row 0 means setup, a later
        // row means the acoustic feedback or the KV carry.
        let w = latents.ncols();
        for r in 0..latents.nrows() {
            let d = (0..w)
                .map(|c| (got[r * w + c] - f.latents[r * w + c]).abs())
                .fold(0f32, f32::max);
            eprintln!("row {r}: worst {d:.6}");
        }
        panic!(
            "worst deviation {worst} at {at} (got {}, want {})",
            got[at], f.latents[at]
        );
    }
}
