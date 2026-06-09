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

//! MoE router TopK capture + auto-refresh on synthetic prefill.

mod compile_support;

use rlx_models::build_qwen35_graph_sized;
use rlx_models::qwen35::synth;

#[test]
fn prefill_capture_matches_moe_layer_count() {
    let cfg = synth::moe_cfg();
    let weights = synth::moe_synth_weights(&cfg);
    let (graph, params, _) =
        build_qwen35_graph_sized(&cfg, weights, 1, 4, true, true, false).expect("graph");
    let mut exe = compile_support::compile_qwen35_prefill(rlx_runtime::Device::Cpu, graph, params);
    assert!(exe.enable_moe_topk_capture(cfg.num_experts));
    let _ = exe.run(&[("input_ids", &[1.0, 2.0, 3.0, 4.0])]);
    let layers = exe.take_moe_topk_capture().expect("captured topk");
    assert_eq!(
        layers.len(),
        3,
        "synth has 3 MoE trunk layers (2 linear + 1 full-attn)"
    );
    for layer in &layers {
        assert!(!layer.is_empty());
    }
}
