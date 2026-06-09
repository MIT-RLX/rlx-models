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

//! MoE prefill logits unchanged when expert pool mask is partial (CPU).

mod compile_support;

use rlx_models::build_qwen35_graph_sized;
use rlx_models::qwen35::Qwen35Runner;
use rlx_models::qwen35::synth;
use rlx_runtime::Device;

#[test]
fn moe_prefill_logits_invariant_under_residency_mask() {
    let cfg = synth::moe_cfg();
    let weights = synth::moe_synth_weights(&cfg);
    let input_ids: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    let (g0, p0, _) =
        build_qwen35_graph_sized(&cfg, weights.clone(), 1, 4, true, true, false).unwrap();
    let mut exe_all = compile_support::compile_qwen35_prefill(Device::Cpu, g0, p0);
    let logits_all = exe_all.run(&[("input_ids", &input_ids)])[0].clone();

    let (g1, p1, _) = build_qwen35_graph_sized(&cfg, weights, 1, 4, true, true, false).unwrap();
    let mut exe_masked = compile_support::compile_qwen35_prefill(Device::Cpu, g1, p1);
    let mask = vec![true, false, false, true];
    exe_masked.set_moe_resident_experts(&mask);
    let logits_masked = exe_masked.run(&[("input_ids", &input_ids)])[0].clone();

    assert_eq!(logits_all.len(), logits_masked.len());
    for (i, (a, b)) in logits_all.iter().zip(logits_masked.iter()).enumerate() {
        assert!((a - b).abs() < 1e-4, "logit[{i}] diverged: {a} vs {b}");
    }
}

#[test]
fn runner_syncs_pool_before_decode_compile_path() {
    let cfg = synth::moe_cfg();
    let weights = synth::moe_synth_weights(&cfg);
    let runner = Qwen35Runner::builder()
        .inline_weights(cfg, weights)
        .device(Device::Cpu)
        .max_seq(8)
        .max_gpu_experts_per_layer(2)
        .build()
        .expect("runner");
    let mo = runner.moe_offload().expect("offload");
    assert_eq!(mo.pools[0].gpu_budget(), 2);
    let mask = mo.merged_resident_mask();
    assert_eq!(mask.iter().filter(|&&r| r).count(), 2);
}
