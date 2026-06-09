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

//! Block diffusion driver parity with TIDE `generate` control flow.

use rlx_models::llada2::{GenerateConfig, LLaDA2Runner, synth};
use rlx_models::tide::num_transfer_tokens_schedule;
use rlx_runtime::Device;

#[test]
fn transfer_schedule_matches_pytorch() {
    assert_eq!(num_transfer_tokens_schedule(32, 32), vec![1; 32]);
    assert_eq!(num_transfer_tokens_schedule(10, 3), vec![4, 3, 3]);
}

#[test]
fn generate_runs_on_synthetic_runner() {
    let cfg = synth::tiny_cfg();
    let weights = synth::tiny_weights(&cfg);
    let mut runner = LLaDA2Runner::builder()
        .inline_weights(cfg.clone(), weights)
        .device(Device::Cpu)
        .batch_seq(1, 8)
        .build()
        .expect("runner");
    let gen_cfg = GenerateConfig {
        block_length: 4,
        steps: 2,
        gen_length: 4,
        ..GenerateConfig::from_model(&cfg)
    };
    let (out, stats) = runner.generate(&gen_cfg, &[1, 2]).expect("generate");
    assert!(!out.is_empty());
    assert!(stats.len() <= gen_cfg.steps);
}
