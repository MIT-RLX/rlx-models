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

//! MoE expert store + host bind + residency stats (CPU).

mod compile_support;

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_models::build_qwen35_graph_sized;
use rlx_models::qwen35::synth;
use rlx_models::qwen35::{build_moe_expert_store, moe_host_bind_from_store};
use rlx_runtime::{Device, Session};

#[test]
fn grouped_matmul_host_bind_lossless() {
    let cfg = synth::moe_cfg();
    let weights = synth::moe_synth_weights(&cfg);
    let store = build_moe_expert_store(&cfg, &weights).expect("store");
    let gate = &store.layers[0].gate;
    let m = 4usize;
    let k = gate.k;
    let n = gate.n;
    let num_experts = gate.num_experts;

    let mut g = Graph::new("gmm_host");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[num_experts, k, n], DType::F32));
    let idx_in = g.input("expert_idx", Shape::new(&[m], DType::F32));
    let out = g.add_node(
        Op::GroupedMatMul,
        vec![x_in, w, idx_in],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![out]);

    let x: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.05).collect();
    let expert_idx = vec![0.0, 1.0, 1.0, 2.0];

    let mut compiled = Session::new(Device::Cpu).compile(g);
    compiled.set_param("w", gate.as_slice());

    rlx_cpu::moe_residency::bind_host_weights(Some(moe_host_bind_from_store(&store)));
    let partial = vec![false, true, true, true];
    compiled.set_moe_resident_experts(&partial);
    let out_part = compiled.run(&[("x", &x), ("expert_idx", &expert_idx)])[0].clone();
    let stats = compiled.take_moe_residency_stats().expect("stats");
    assert!(stats.cpu_tokens > 0, "{stats:?}");

    compiled.set_moe_resident_experts(&vec![true; num_experts]);
    let out_all = compiled.run(&[("x", &x), ("expert_idx", &expert_idx)])[0].clone();
    for (i, (a, b)) in out_all.iter().zip(out_part.iter()).enumerate() {
        assert!((a - b).abs() < 1e-4, "mismatch at {i}: {a} vs {b}");
    }
    rlx_cpu::moe_residency::bind_host_weights(None);
}

#[test]
fn moe_expert_store_apply_lossless_under_mask() {
    let cfg = synth::moe_cfg();
    let weights = synth::moe_synth_weights(&cfg);
    let store = build_moe_expert_store(&cfg, &weights).expect("store");
    assert_eq!(store.num_layers(), 3);

    let (g0, p0, _) =
        build_qwen35_graph_sized(&cfg, weights.clone(), 1, 4, true, true, false).unwrap();
    let mut exe_all = compile_support::compile_qwen35_prefill(Device::Cpu, g0, p0);
    let input_ids: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let logits_all = exe_all.run(&[("input_ids", &input_ids)])[0].clone();

    let (g1, p1, _) = build_qwen35_graph_sized(&cfg, weights, 1, 4, true, true, false).unwrap();
    let mut exe = compile_support::compile_qwen35_prefill(Device::Cpu, g1, p1);
    store.apply_to_compiled(&mut exe);
    rlx_cpu::moe_residency::bind_host_weights(Some(moe_host_bind_from_store(&store)));

    let mask = vec![false, true, true, true];
    exe.set_moe_resident_experts(&mask);
    let logits_masked = exe.run(&[("input_ids", &input_ids)])[0].clone();

    for (i, (a, b)) in logits_all.iter().zip(logits_masked.iter()).enumerate() {
        assert!((a - b).abs() < 1e-4, "logit[{i}] diverged: {a} vs {b}");
    }
    rlx_cpu::moe_residency::bind_host_weights(None);
}
