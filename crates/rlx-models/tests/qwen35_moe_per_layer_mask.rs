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

//! Per-layer TIDE masks are lossless vs all-resident (CPU).

mod compile_support;

use rlx_models::build_qwen35_graph_sized;
use rlx_models::qwen35::build_moe_offload;
use rlx_models::qwen35::synth;
use rlx_runtime::Device;

#[test]
fn per_layer_masks_match_merged_lossless_logits() {
    let cfg = synth::moe_cfg();
    let weights = synth::moe_synth_weights(&cfg);
    let per_layer = build_moe_offload(&cfg, &weights, Some(2), None, None, 1.5, false)
        .expect("offload")
        .per_layer_resident_masks();
    assert_eq!(per_layer.len(), 3);
    let per: Vec<&[bool]> = per_layer.iter().map(|m| m.as_slice()).collect();

    let input_ids: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let (g0, p0, _) =
        build_qwen35_graph_sized(&cfg, weights.clone(), 1, 4, true, true, false).unwrap();
    let mut exe_all = compile_support::compile_qwen35_prefill(Device::Cpu, g0, p0);
    let logits_all = exe_all.run(&[("input_ids", &input_ids)])[0].clone();

    let (g1, p1, _) = build_qwen35_graph_sized(&cfg, weights, 1, 4, true, true, false).unwrap();
    let mut exe = compile_support::compile_qwen35_prefill(Device::Cpu, g1, p1);
    exe.set_moe_resident_experts_per_layer(&per);
    let logits_per = exe.run(&[("input_ids", &input_ids)])[0].clone();

    assert_eq!(logits_all.len(), logits_per.len());
    for (i, (a, b)) in logits_all.iter().zip(logits_per.iter()).enumerate() {
        assert!((a - b).abs() < 1e-4, "logit[{i}] diverged: {a} vs {b}");
    }
}
