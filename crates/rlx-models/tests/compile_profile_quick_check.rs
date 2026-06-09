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

//! Quick check: tier-1 profile compile helpers build and run a tiny Qwen3.5 graph.

mod compile_support;

use rlx_models::build_qwen35_graph_sized;
use rlx_models::qwen35::synth;
use rlx_runtime::Device;

#[test]
fn compile_support_qwen35_prefill_profile_runs() {
    let cfg = synth::tiny_cfg();
    let weights = synth::synth_weights(&cfg);
    let (graph, params, _) =
        build_qwen35_graph_sized(&cfg, weights, 1, 4, true, true, false).expect("graph");
    let mut compiled = compile_support::compile_qwen35_prefill(Device::Cpu, graph, params);
    let outs = compiled.run(&[("input_ids", &[1.0, 2.0, 3.0, 4.0])]);
    assert!(!outs.is_empty());
    assert!(outs[0].iter().all(|v| v.is_finite()));
}

#[test]
fn compile_support_qwen35_decode_profile_compiles() {
    use rlx_models::build_qwen35_decode_graph;

    let cfg = synth::tiny_cfg();
    let weights = synth::synth_weights(&cfg);
    let past_seq = 4usize;
    let (graph, params, _) =
        build_qwen35_decode_graph(&cfg, weights, 1, past_seq).expect("decode graph");
    let _compiled = compile_support::compile_qwen35_decode(Device::Cpu, graph, params);
}
