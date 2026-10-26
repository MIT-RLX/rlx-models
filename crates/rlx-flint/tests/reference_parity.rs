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

//! Parity against `scripts/flint_reference.py` — an independent torch
//! transcription of Recursive's SGLang model — on the synthetic checkpoint.
//!
//! The fixture holds the reference's logits at every position of a 13-token
//! prompt that crosses the tiny model's 4-token window. The checkpoint is
//! regenerated here from the same seed (the writer is deterministic), so the
//! test needs neither Python nor files outside the repo.
//!
//! The negative controls make sure the comparison can fail: the same model
//! with the sinks zeroed, or with the window disabled, must *not* match.

use rlx_flint::synth::{SynthDtype, tiny_config_json, write_checkpoint};
use rlx_flint::{Checkpoint, FlintConfig, FlintRunner, RunnerOptions};
use rlx_runtime::Device;
use std::sync::Arc;

const FIXTURE: &str = include_str!("fixtures/synthetic_reference.json");

struct Reference {
    ids: Vec<u32>,
    logits: Vec<f32>,
    seed: u64,
}

fn reference() -> Reference {
    let v: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    Reference {
        ids: v["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect(),
        logits: v["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect(),
        seed: v["seed"].as_u64().unwrap(),
    }
}

fn run(config: serde_json::Value, seed: u64, ids: &[u32], chunk: usize) -> Vec<f32> {
    let tmp = tempfile::tempdir().unwrap();
    // The fixture was produced over the canonical config; a control may
    // change how the config is *read* but keeps the same weights.
    write_checkpoint(tmp.path(), &tiny_config_json(), SynthDtype::F32, 3, seed).unwrap();
    let cfg = FlintConfig::from_value(&config).unwrap();
    let ckpt = Arc::new(Checkpoint::open(tmp.path()).unwrap());
    let opts = RunnerOptions::new(Device::Cpu)
        .max_context(32)
        .prefill_chunk(chunk);
    let mut r = FlintRunner::from_checkpoint(cfg, ckpt, opts).unwrap();
    r.all_logits(ids).unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn matches_the_torch_reference_at_every_position() {
    let r = reference();
    for chunk in [1, 5] {
        let got = run(tiny_config_json(), r.seed, &r.ids, chunk);
        let d = max_abs_diff(&got, &r.logits);
        assert!(d < 2e-4, "chunk {chunk}: max |rlx - torch| = {d}");
    }
}

#[test]
fn a_model_without_the_window_does_not_match() {
    let r = reference();
    let mut cfg = tiny_config_json();
    cfg["use_sliding_window"] = false.into();
    let got = run(cfg, r.seed, &r.ids, 1);
    let d = max_abs_diff(&got, &r.logits);
    assert!(d > 1e-2, "dropping the window changed logits by only {d}");
}

#[test]
fn a_model_with_rope_everywhere_does_not_match() {
    let r = reference();
    let mut cfg = tiny_config_json();
    cfg["no_rope_layer_indices"] = serde_json::json!([]);
    let got = run(cfg, r.seed, &r.ids, 1);
    let d = max_abs_diff(&got, &r.logits);
    assert!(
        d > 1e-2,
        "rotating the NoPE layers changed logits by only {d}"
    );
}
