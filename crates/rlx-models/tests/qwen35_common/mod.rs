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

//! Shared synthetic Qwen3.5 weights and backend test helpers.

#![allow(dead_code)]

use rlx_models::Qwen35RunnerBuilder;
use rlx_models::qwen3::SampleOpts;
use rlx_models::qwen35::synth;
use rlx_runtime::Device;

const SEQ: usize = 4;

pub fn tiny_cfg() -> rlx_models::Qwen35Config {
    synth::tiny_cfg()
}

fn build_runner(device: Device, bucketed_decode: bool) -> rlx_models::Qwen35Runner {
    let cfg = tiny_cfg();
    let weights = synth::synth_weights(&cfg);
    Qwen35RunnerBuilder::default()
        .inline_weights(cfg, weights)
        .device(device)
        .max_seq(SEQ + 8)
        .last_logits_only(true)
        .bucketed_decode(bucketed_decode)
        .build()
        .expect("runner")
}

pub fn run_prefill_last_logits(device: Device) {
    let cfg = tiny_cfg();
    let mut runner = build_runner(device, false);
    let logits = runner
        .prefill_get_last_logits(&[1, 2, 3, 4])
        .expect("prefill");
    assert_eq!(logits.len(), cfg.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

pub fn run_prefill_last_logits_if_available(device: Device) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip qwen35 {device:?}: backend not available");
        return;
    }
    run_prefill_last_logits(device);
}

/// Greedy decode via [`Qwen35Runner::generate_with_opts`] (KV-cache path).
pub fn run_runner_greedy(device: Device) {
    let cfg = tiny_cfg();
    let mut runner = build_runner(device, false);
    let out = runner
        .generate_with_opts(&[1, 2, 3], 2, SampleOpts::greedy(), |_| true)
        .expect("generate");
    assert_eq!(out.len(), 2);
    assert!(out.iter().all(|&t| t < cfg.vocab_size as u32));
}

pub fn run_runner_greedy_if_available(device: Device) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip qwen35 runner {device:?}: backend not available");
        return;
    }
    run_runner_greedy(device);
}
