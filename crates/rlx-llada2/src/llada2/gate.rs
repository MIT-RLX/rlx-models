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

// RLX — LLaDA2 MoE router (`LLaDA2MoeGate` in TIDE modeling_llada2_moe.py).

use crate::config::LLaDA2MoeConfig;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

/// Group-limited TopK on CPU (TIDE `group_limited_topk`).
pub use rlx_cpu::llada2_gate::group_limited_topk;

/// Sigmoid router + bias routing + group-limited topk (PyTorch `LLaDA2MoeGate.forward`).
pub fn emit_group_limited_gate(
    g: &mut Graph,
    hidden_2d: NodeId,
    router_w: NodeId,
    expert_bias: NodeId,
    cfg: &LLaDA2MoeConfig,
    rows: usize,
) -> (NodeId, NodeId) {
    use crate::gate_op::{self, OP_NAME};

    gate_op::ensure_group_limited_gate_registered();

    let n_expert = cfg.num_experts;
    let top_k = cfg.num_experts_per_tok;
    let logits = g.mm(hidden_2d, router_w);
    let log_shape = g.shape(logits).clone();
    let scores_sigmoid = g.add_node(
        Op::Activation(Activation::Sigmoid),
        vec![logits],
        log_shape.clone(),
    );
    let bias = g.reshape_(expert_bias, vec![1, n_expert as i64]);
    let scores_route = g.add(scores_sigmoid, bias);
    let attrs = gate_op::gate_attrs_bytes(
        cfg.n_group,
        cfg.topk_group,
        top_k,
        cfg.routed_scaling_factor,
        n_expert,
    );
    let packed = g.custom_op_packed(
        OP_NAME,
        attrs,
        vec![scores_sigmoid, scores_route],
        Shape::new(&[rows, top_k * 2], DType::F32),
    );
    let packed = g.reshape_(packed, vec![rows as i64, (top_k * 2) as i64]);
    let top_idx = g.narrow_(packed, 1, 0, top_k);
    let top_probs = g.narrow_(packed, 1, top_k, top_k);
    (top_idx, top_probs)
}

/// Full gate forward on host (tests / reference).
pub fn gate_forward_host(
    cfg: &LLaDA2MoeConfig,
    hidden: &[f32],
    router: &[f32],
    expert_bias: &[f32],
) -> (Vec<u32>, Vec<f32>) {
    let h = cfg.hidden_size;
    let e = cfg.num_experts;
    let top_k = cfg.num_experts_per_tok;
    let rows = hidden.len() / h;
    let mut scores_sigmoid = vec![0f32; rows * e];
    let mut scores_route = vec![0f32; rows * e];
    for t in 0..rows {
        let x = &hidden[t * h..(t + 1) * h];
        for ei in 0..e {
            let mut dot = 0f32;
            for i in 0..h {
                dot += x[i] * router[i * e + ei];
            }
            let s = 1.0 / (1.0 + (-dot).exp());
            scores_sigmoid[t * e + ei] = s;
            scores_route[t * e + ei] = s + expert_bias[ei];
        }
    }
    let (_, idx) = group_limited_topk(&scores_route, rows, e, cfg.n_group, cfg.topk_group, top_k);
    let mut weights = Vec::with_capacity(rows * top_k);
    for t in 0..rows {
        let row_sig = &scores_sigmoid[t * e..(t + 1) * e];
        let mut picked = Vec::with_capacity(top_k);
        for ki in 0..top_k {
            picked.push(row_sig[idx[t * top_k + ki] as usize]);
        }
        let sum: f32 = picked.iter().sum::<f32>() + 1e-20;
        let norm = if top_k > 1 { 1.0 / sum } else { 1.0 };
        for &p in &picked {
            weights.push(p * norm * cfg.routed_scaling_factor);
        }
    }
    (idx, weights)
}
