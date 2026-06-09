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

//! Group-limited gate: custom op vs CPU reference.

use rlx_ir::{DType, Graph, Shape};
use rlx_models::llada2::gate::gate_forward_host;
use rlx_models::llada2::gate_op::ensure_group_limited_gate_registered;
use rlx_models::llada2::synth;

#[test]
fn custom_gate_matches_host_reference() {
    ensure_group_limited_gate_registered();
    let cfg = synth::tiny_cfg();
    let weights = synth::tiny_weights(&cfg);
    let il = 1usize;
    let moe = match &weights.layers[il].ffn {
        rlx_models::llada2::weights::LayerFfn::Moe(m) => m,
        _ => panic!("layer 1 should be moe"),
    };

    let rows = 4usize;
    let h = cfg.hidden_size;
    let e = cfg.num_experts;
    let hidden: Vec<f32> = (0..rows * h).map(|i| 0.01 * (i as f32)).collect();

    let (host_idx, host_probs) = gate_forward_host(&cfg, &hidden, &moe.router, &moe.expert_bias);

    let mut g = Graph::new("gate_check");
    let x = g.input("x", Shape::new(&[rows, h], DType::F32));
    let w = g.param("w", Shape::new(&[h, e], DType::F32));
    let b = g.param("b", Shape::new(&[e], DType::F32));
    let (top_idx, top_probs) =
        rlx_models::llada2::gate::emit_group_limited_gate(&mut g, x, w, b, &cfg, rows);
    g.set_outputs(vec![top_idx, top_probs]);

    let mut params = std::collections::HashMap::new();
    params.insert("w".into(), moe.router.clone());
    params.insert("b".into(), moe.expert_bias.clone());

    let built = rlx_flow::BuiltModel::from_graph(g, params).expect("built");
    let mut compiled = rlx_models::flow_util::compile_built_cpu(built).expect("compile");
    let outs = compiled.run(&[("x", &hidden)]);
    let idx = &outs[0];
    let probs = &outs[1];
    let k = cfg.num_experts_per_tok;
    for t in 0..rows {
        for ki in 0..k {
            let got_i = idx[t * k + ki] as u32;
            let got_p = probs[t * k + ki];
            assert_eq!(got_i, host_idx[t * k + ki], "token {t} expert {ki}");
            assert!(
                (got_p - host_probs[t * k + ki]).abs() < 1e-5,
                "token {t} prob {ki}: got {got_p} host {}",
                host_probs[t * k + ki]
            );
        }
    }
}
