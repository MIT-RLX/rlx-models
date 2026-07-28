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

//! MiniMax-M3 fine-grained MoE (`block_sparse_moe`).
//!
//! Plain top-k **sigmoid** router with a per-expert correction bias (no
//! group-limiting): `sig = sigmoid(h·gateᵀ)`, `route = sig + e_score_bias`,
//! `idx = topk(route, k)`, `w = normalize(sig[idx])`. The routing weight is
//! applied to each expert's **output**, the routed sum is scaled by
//! `routed_scaling_factor`, then the always-on shared expert is added. Routed
//! experts use `Op::GroupedMatMul` over stacked `[E, 2·inter, hidden]` /
//! `[E, hidden, inter]` weights (transposed in-graph to the `[E,K,N]` layout).

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::Op;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use super::mlp::oai_consts;
use super::ops::swiglu_oai;

/// Dimensions for one MiniMax-M3 MoE block.
#[derive(Debug, Clone, Copy)]
pub struct M3MoeDims {
    /// Residual-stream width.
    pub hidden: usize,
    /// Inner width of each routed expert.
    pub moe_inter: usize,
    /// Inner width of the shared expert.
    pub shared_inter: usize,
    /// Number of routed experts.
    pub n_routed: usize,
    /// Experts selected per token.
    pub top_k: usize,
    /// Scale applied to the routed-expert sum.
    pub routed_scaling: f32,
    /// SwiGLU-OAI sigmoid gain.
    pub alpha: f32,
    /// SwiGLU-OAI clamp bound.
    pub limit: f32,
    /// Sequence length (rows).
    pub seq: usize,
}

/// Emit the MoE FFN for `{prefix}` (`block_sparse_moe`) over `[1, seq, hidden]`.
pub fn emit_m3_moe(
    emit: &mut Emit<'_>,
    prefix: &str,
    hidden: HirNodeId,
    d: M3MoeDims,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let rows = d.seq;
    let inter = d.moe_inter;

    let (alpha_node, one_node) = oai_consts(emit, prefix, d.alpha);
    let scale_node = emit.synth_param(
        &format!("{prefix}.m3_routed_scale"),
        vec![d.routed_scaling],
        Shape::new(&[1], f),
    );
    let router_w = emit.load_param(&format!("{prefix}.gate.weight"), true)?; // [hidden, E]
    let ebias = emit.load_param(&format!("{prefix}.e_score_correction_bias"), false)?; // [E]
    let gate_up_w = emit.load_param(&format!("{prefix}.experts.gate_up_proj"), false)?; // [E,2inter,hidden]
    let down_w = emit.load_param(&format!("{prefix}.experts.down_proj"), false)?; // [E,hidden,inter]
    let s_gate = emit.load_param(&format!("{prefix}.shared_experts.gate_proj.weight"), true)?;
    let s_up = emit.load_param(&format!("{prefix}.shared_experts.up_proj.weight"), true)?;
    let s_down = emit.load_param(&format!("{prefix}.shared_experts.down_proj.weight"), true)?;

    let mut gb = HirMut::new(emit.hir());
    let h2d = gb.reshape_(hidden, vec![rows as i64, d.hidden as i64]);

    // --- Router: sigmoid weights, select on (sig + bias), normalize sig ---
    let logits = gb.mm(h2d, router_w); // [rows, E]
    let sig = gb.activation(
        rlx_ir::op::Activation::Sigmoid,
        logits,
        Shape::new(&[rows, d.n_routed], f),
    );
    let bias = gb.reshape_(ebias, vec![1, d.n_routed as i64]);
    let route = gb.add(sig, bias); // [rows, E]
    let top_idx = gb.add_node(
        Op::TopK { k: d.top_k },
        vec![route],
        Shape::new(&[rows, d.top_k], f),
    );
    // take-along-axis: gather the sigmoid weights at the selected experts.
    let top_sig = gb.add_node(
        Op::GatherElements { axis: 1 },
        vec![sig, top_idx],
        Shape::new(&[rows, d.top_k], f),
    ); // [rows, top_k]
    // Normalize the selected sigmoid weights so they sum to 1 per row.
    let denom = gb.add_node(
        Op::Reduce {
            op: rlx_ir::op::ReduceOp::Sum,
            axes: vec![1],
            keep_dim: true,
        },
        vec![top_sig],
        Shape::new(&[rows, 1], f),
    );
    let top_w = gb.div(top_sig, denom); // [rows, top_k] (broadcast)

    // Experts stored [E, N, K] → transpose to [E, K, N] for GroupedMatMul.
    let gate_up_t = gb.transpose_(gate_up_w, vec![0, 2, 1]); // [E, hidden, 2inter]
    let down_t = gb.transpose_(down_w, vec![0, 2, 1]); // [E, inter, hidden]

    let mut acc: Option<HirNodeId> = None;
    for ki in 0..d.top_k {
        let idx_col = gb.narrow_(top_idx, 1, ki, 1);
        let eidx = gb.reshape_(idx_col, vec![rows as i64]);
        let w_col = gb.narrow_(top_w, 1, ki, 1); // [rows, 1]

        let gate_up = gb.add_node(
            Op::GroupedMatMul,
            vec![h2d, gate_up_t, eidx],
            Shape::new(&[rows, 2 * inter], f),
        );
        let g = gb.narrow_(gate_up, 1, 0, inter);
        let u = gb.narrow_(gate_up, 1, inter, inter);
        let hx = swiglu_oai(&mut gb, g, u, rows, inter, d.limit, alpha_node, one_node);
        let down = gb.add_node(
            Op::GroupedMatMul,
            vec![hx, down_t, eidx],
            Shape::new(&[rows, d.hidden], f),
        );
        let weighted = gb.mul(down, w_col);
        acc = Some(match acc {
            Some(a) => gb.add(a, weighted),
            None => weighted,
        });
    }
    let routed = acc.expect("top_k >= 1");

    // Scale routed sum by routed_scaling_factor.
    let routed = gb.mul(routed, scale_node);

    // Shared expert (SwiGLU-OAI over shared_inter).
    let sg = gb.mm(h2d, s_gate);
    let su = gb.mm(h2d, s_up);
    let sh = swiglu_oai(
        &mut gb,
        sg,
        su,
        rows,
        d.shared_inter,
        d.limit,
        alpha_node,
        one_node,
    );
    let shared = gb.mm(sh, s_down);

    let out2d = gb.add(shared, routed);
    Ok(gb.reshape_(out2d, vec![1, d.seq as i64, d.hidden as i64]))
}
