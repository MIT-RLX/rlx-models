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

//! DeepSeek-V3 fine-grained MoE (`DeepseekV3MoE`).
//!
//! Group-limited `noaux_tc` router (sigmoid + per-expert correction bias +
//! n_group/topk_group group selection + top-k, weights `·routed_scaling`) reusing
//! rlx-llada2's `group_limited_gate` custom op, top-`k` routed experts via
//! `Op::GroupedMatMul`, plus one always-on shared expert. Routing weights are
//! applied to the expert **output** (matching HF). Expert weights are stored
//! `[E, N, K]`, so they're transposed in-graph to the `[E, K, N]` GroupedMatMul
//! layout.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::{Activation, Op};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use rlx_llada2::llada2::gate_op::{
    OP_NAME, ensure_group_limited_gate_registered, gate_attrs_bytes,
};

#[derive(Debug, Clone, Copy)]
pub struct DeepseekMoeDims {
    pub hidden: usize,
    pub moe_inter: usize,
    pub n_routed: usize,
    pub top_k: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub routed_scaling: f32,
    pub shared_inter: usize,
    pub seq: usize,
}

/// Emit the MoE FFN for `model.layers.{i}.mlp` (`prefix`) on `[1,seq,hidden]`.
pub fn emit_deepseek_moe(
    emit: &mut Emit<'_>,
    prefix: &str,
    hidden: HirNodeId,
    d: DeepseekMoeDims,
) -> Result<HirNodeId> {
    ensure_group_limited_gate_registered();
    let f = DType::F32;
    let rows = d.seq;
    let inter = d.moe_inter;

    let router_w = emit.load_param(&format!("{prefix}.gate.weight"), true)?;
    let ebias = emit.load_param(&format!("{prefix}.gate.e_score_correction_bias"), false)?;
    let gate_up_w = emit.load_param(&format!("{prefix}.experts.gate_up_proj"), false)?; // [E,2inter,hidden]
    let down_w = emit.load_param(&format!("{prefix}.experts.down_proj"), false)?; // [E,hidden,inter]
    let s_gate = emit.load_param(&format!("{prefix}.shared_experts.gate_proj.weight"), true)?;
    let s_up = emit.load_param(&format!("{prefix}.shared_experts.up_proj.weight"), true)?;
    let s_down = emit.load_param(&format!("{prefix}.shared_experts.down_proj.weight"), true)?;

    let attrs = gate_attrs_bytes(
        d.n_group,
        d.topk_group,
        d.top_k,
        d.routed_scaling,
        d.n_routed,
    );

    let mut gb = HirMut::new(emit.hir());
    let h2d = gb.reshape_(hidden, vec![rows as i64, d.hidden as i64]);

    // --- Group-limited router → (top_idx, top_probs) ---
    let logits = gb.mm(h2d, router_w); // [rows, n_routed]
    let sig = gb.add_node(
        Op::Activation(Activation::Sigmoid),
        vec![logits],
        Shape::new(&[rows, d.n_routed], f),
    );
    let bias = gb.reshape_(ebias, vec![1, d.n_routed as i64]);
    let route = gb.add(sig, bias);
    let packed = gb.add_node(
        Op::Custom {
            name: OP_NAME.to_string(),
            num_inputs: 2,
            attrs,
        },
        vec![sig, route],
        Shape::new(&[rows, d.top_k * 2], f),
    );
    let top_idx = gb.narrow_(packed, 1, 0, d.top_k);
    let top_probs = gb.narrow_(packed, 1, d.top_k, d.top_k);

    // Experts stored [E,N,K] → transpose to [E,K,N] for GroupedMatMul.
    let gate_up_t = gb.transpose_(gate_up_w, vec![0, 2, 1]); // [E, hidden, 2inter]
    let down_t = gb.transpose_(down_w, vec![0, 2, 1]); // [E, inter, hidden]

    let mut acc: Option<HirNodeId> = None;
    for ki in 0..d.top_k {
        let idx_col = gb.narrow_(top_idx, 1, ki, 1);
        let eidx = gb.reshape_(idx_col, vec![rows as i64]);
        let prob_col = gb.narrow_(top_probs, 1, ki, 1);
        let prob = gb.reshape_(prob_col, vec![rows as i64, 1]);

        let gate_up = gb.add_node(
            Op::GroupedMatMul,
            vec![h2d, gate_up_t, eidx],
            Shape::new(&[rows, 2 * inter], f),
        );
        let g = gb.narrow_(gate_up, 1, 0, inter);
        let u = gb.narrow_(gate_up, 1, inter, inter);
        let act = gb.silu(g);
        let hx = gb.mul(act, u);
        let down = gb.add_node(
            Op::GroupedMatMul,
            vec![hx, down_t, eidx],
            Shape::new(&[rows, d.hidden], f),
        );
        let weighted = gb.mul(down, prob);
        acc = Some(match acc {
            Some(a) => gb.add(a, weighted),
            None => weighted,
        });
    }
    let routed = acc.expect("top_k >= 1");

    // Shared expert (SwiGLU), added to the routed sum.
    let sg = gb.mm(h2d, s_gate);
    let su = gb.mm(h2d, s_up);
    let sact = gb.silu(sg);
    let sh = gb.mul(sact, su);
    let shared = gb.mm(sh, s_down);

    let out2d = gb.add(shared, routed);
    let _ = d.shared_inter; // documented invariant; shared weights define their dims
    Ok(gb.reshape_(out2d, vec![1, d.seq as i64, d.hidden as i64]))
}
