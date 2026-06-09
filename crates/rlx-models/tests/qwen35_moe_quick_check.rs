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

//! Synthetic qwen35moe forward quick check — exercises TopK + GroupedMatMul MoE FFN on CPU.

mod compile_support;

use rlx_models::build_qwen35_graph_sized;
use rlx_models::qwen35::synth;
use rlx_runtime::Device;

#[test]
fn qwen35_moe_trunk_prefill_quick_check() {
    let cfg = synth::moe_cfg();
    let weights = synth::moe_synth_weights(&cfg);
    let (graph, params, _packed) =
        build_qwen35_graph_sized(&cfg, weights, 1, 4, true, true, false).expect("build moe graph");
    let mut exe = compile_support::compile_qwen35_prefill(Device::Cpu, graph, params);
    let input_ids: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let outs = exe.run(&[("input_ids", &input_ids)]);
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].len(), cfg.vocab_size, "last-token logits");
    assert!(
        outs[0].iter().all(|x| x.is_finite()),
        "logits must be finite"
    );
    eprintln!(
        "qwen35 MoE quick check ok: experts={} top_k={} logits={}",
        cfg.num_experts,
        cfg.num_experts_used,
        outs[0].len()
    );
}
