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

//! Parity vs `tada.nn.vibevoice.VibeVoiceDiffusionHead`, through the solver.
//!
//! Driven with one Euler step on a uniform schedule and guidance off, so the
//! graph reduces to `noise + head(noise, t = 0, cond)` — which is exactly what
//! the reference computes. Routing it through
//! [`DiffusionHead::compile_solver`] rather than a bare forward means the test
//! also covers the parts that only exist in this port: the timestep embedding
//! folded to a host constant, the conditioning sum, and the Euler update.
//!
//! The upstream module's `initialize_weights` zeroes every adaLN projection
//! and the final linear, which would make any output identically the input —
//! so the fixture randomizes all of them first.

mod common;
use common::device;
use rlx_tada::config::TadaConfig;
use rlx_tada::head::{CfgSchedule, DiffusionHead, SolveOptions, TimeSchedule};
use rlx_tada::weights::TensorStore;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    hidden: usize,
    latent: usize,
    layers: usize,
    acoustic_dim: usize,
    num_time_classes: usize,
    noise: Vec<f32>,
    cond: Vec<f32>,
    out: Vec<f32>,
}

#[test]
fn matches_the_upstream_diffusion_head() {
    let f: Fixture = serde_json::from_str(include_str!("fixtures/head_reference.json"))
        .expect("parse head_reference.json");
    let weights = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/head_reference.safetensors");
    let store = std::sync::Arc::new(TensorStore::open(&weights).expect("open fixture weights"));

    let cfg_json = format!(
        r#"{{"hidden_size":{},"intermediate_size":32,"num_hidden_layers":1,
            "num_attention_heads":2,"num_key_value_heads":1,"vocab_size":16,
            "acoustic_dim":{},"num_time_classes":{},"head_layers":{},
            "head_ffn_ratio":2.0,"max_position_embeddings":64}}"#,
        f.hidden, f.acoustic_dim, f.num_time_classes, f.layers
    );
    let cfg: TadaConfig = serde_json::from_str(&cfg_json).expect("config");
    assert_eq!(cfg.latent_dim(), f.latent, "fixture/config latent mismatch");

    let head = DiffusionHead::load(store, &cfg, "prediction_head.").expect("load head");
    let opts = SolveOptions {
        num_steps: 1,
        // Guidance off → single branch, and the Euler step is the whole solve.
        acoustic_cfg_scale: 1.0,
        duration_cfg_scale: 1.0,
        cfg_schedule: CfgSchedule::Constant,
        time_schedule: TimeSchedule::Uniform,
        noise_temperature: 1.0,
    };
    let mut solver = head
        .compile_solver(device(), &opts)
        .expect("compile solver");

    let got = solver
        .run(&[("noise", &f.noise), ("cond", &f.cond)])
        .into_iter()
        .next()
        .expect("no output");
    assert_eq!(got.len(), f.out.len());

    let mut worst = 0f32;
    let mut at = 0usize;
    for (i, (a, b)) in got.iter().zip(&f.out).enumerate() {
        let d = (a - b).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    assert!(
        worst < 1e-4,
        "worst deviation {worst} at index {at} (got {}, want {})",
        got[at],
        f.out[at]
    );
}
