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

//! Motif's MoE FFN (`MoE` / `TokenChoiceTopKRouter` / `MotifExperts`).
//!
//! ```text
//!   scores = σ(x·W_gateᵀ)                                   fp32, [rows, E]
//!   idx    = topk(scores + expert_bias, k)                  selection only
//!   w      = scores[idx] / Σ scores[idx] · route_scale       (route_norm)
//!   y      = Σ_k w_k · Expert_{idx_k}(x)  +  SharedExpert(x)
//! ```
//!
//! Selection-by-biased-score / weighting-by-raw-sigmoid, normalize, scale is
//! exactly what [`rlx_llada2`]'s `group_limited_gate` kernel does, so this reuses
//! it with `n_group = topk_group = 1` (Motif has no expert groups).
//!
//! The experts themselves are the interesting part: each one owns its own
//! PolyNorm coefficients (`GroupedPolyNorm`, `[E, 3]` weights + `[E, 1]` bias),
//! which is why upstream forces the eager per-expert Python loop. In a graph
//! that constraint disappears — the coefficients are a table, so one `Gather`
//! by routed expert id gives every token its own `[w₀, w₁, w₂, b]` row and the
//! whole top-k slot still runs as a single [`Op::GroupedMatMul`].

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::{Activation, Op};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};
use rlx_llada2::llada2::gate_op::{
    OP_NAME, ensure_group_limited_gate_registered, gate_attrs_bytes,
};

use crate::polynorm::{PolyNormSpec, emit_poly_norm_mul};

/// Everything one MoE layer needs that is not a weight.
#[derive(Debug, Clone, Copy)]
pub struct MotifMoeDims {
    /// Model width — the block's input and output.
    pub hidden: usize,
    /// FFN width of one expert (`moe_intermediate_size`).
    pub moe_inter: usize,
    /// Routed experts in the bank.
    pub num_experts: usize,
    /// Experts each token is routed to.
    pub top_k: usize,
    /// `config.route_scale`.
    pub route_scale: f32,
    /// `MoE.expert_bias` exists only when `load_balance_coeff` is set.
    pub has_expert_bias: bool,
    /// `num_shared_experts > 0`.
    pub has_shared_expert: bool,
    /// PolyNorm flavour for the *routed* experts; the shared expert derives its
    /// own from this (no clamp on the product).
    pub poly: PolyNormSpec,
    /// Prompt length this graph is built for.
    pub seq: usize,
}

/// Emit the MoE FFN for `model.layers.{i}.moe` (`prefix`) on `[1, seq, hidden]`.
///
/// Expert banks must already be in [`Op::GroupedMatMul`]'s `[E, K, N]` layout and
/// the PolyNorm coefficients folded into `experts.act_fn.coeff` — see
/// [`crate::weights::prepare_checkpoint`].
pub fn emit_motif_moe(
    emit: &mut Emit<'_>,
    prefix: &str,
    hidden: HirNodeId,
    d: MotifMoeDims,
) -> Result<HirNodeId> {
    ensure_group_limited_gate_registered();
    let f = DType::F32;
    let rows = d.seq;
    let inter = d.moe_inter;

    let router_w = emit.load_param(&format!("{prefix}.router.gate.weight"), true)?;
    let expert_bias = d
        .has_expert_bias
        .then(|| emit.load_param(&format!("{prefix}.expert_bias"), false))
        .transpose()?;
    let gate_up_w = emit.load_param(&format!("{prefix}.experts.gate_up_proj"), false)?; // [E,H,2I]
    let down_w = emit.load_param(&format!("{prefix}.experts.down_proj"), false)?; // [E,I,H]
    let coeff_bank = emit.load_param(&format!("{prefix}.experts.act_fn.coeff"), false)?; // [E,4]

    let attrs = gate_attrs_bytes(1, 1, d.top_k, d.route_scale, d.num_experts);

    let (top_idx, top_probs, h2d) = {
        let mut gb = HirMut::new(emit.hir());
        let h2d = gb.reshape_(hidden, vec![rows as i64, d.hidden as i64]);
        let logits = gb.mm(h2d, router_w); // [rows, E]
        let scores = gb.add_node(
            Op::Activation(Activation::Sigmoid),
            vec![logits],
            Shape::new(&[rows, d.num_experts], f),
        );
        // Selection scores: sigmoid + the load-balancing correction bias.
        let route = match expert_bias {
            Some(b) => {
                let b = gb.reshape_(b, vec![1, d.num_experts as i64]);
                gb.add(scores, b)
            }
            None => scores,
        };
        let packed = gb.add_node(
            Op::Custom {
                name: OP_NAME.to_string(),
                num_inputs: 2,
                attrs,
            },
            vec![scores, route],
            Shape::new(&[rows, d.top_k * 2], f),
        );
        let idx = gb.narrow_(packed, 1, 0, d.top_k);
        let probs = gb.narrow_(packed, 1, d.top_k, d.top_k);
        (idx, probs, h2d)
    };

    let mut acc: Option<HirNodeId> = None;
    for ki in 0..d.top_k {
        let (eidx, prob, gate, up) = {
            let mut gb = HirMut::new(emit.hir());
            let col = gb.narrow_(top_idx, 1, ki, 1);
            let eidx = gb.reshape_(col, vec![rows as i64]);
            let pcol = gb.narrow_(top_probs, 1, ki, 1);
            let prob = gb.reshape_(pcol, vec![rows as i64, 1]);
            // `grouped_matmul` derives `[rows, 2·inter]` from the operands, so a
            // bank still in the checkpoint's `[E, N, K]` order is rejected here
            // instead of silently short-writing the output row.
            let gate_up = gb.grouped_matmul(h2d, gate_up_w, eidx);
            let gate = gb.narrow_(gate_up, 1, 0, inter);
            let up = gb.narrow_(gate_up, 1, inter, inter);
            (eidx, prob, gate, up)
        };
        // Per-expert PolyNorm coefficients, one row per token.
        let coeff = {
            let mut gb = HirMut::new(emit.hir());
            gb.gather_(coeff_bank, eidx, 0) // [rows, 4]
        };
        let act = emit_poly_norm_mul(
            emit,
            &format!("{prefix}.experts.k{ki}"),
            gate,
            up,
            coeff,
            inter,
            d.poly,
        )?;
        let mut gb = HirMut::new(emit.hir());
        let down = gb.grouped_matmul(act, down_w, eidx);
        let weighted = gb.mul(down, prob);
        acc = Some(match acc {
            Some(a) => gb.add(a, weighted),
            None => weighted,
        });
    }
    let routed = acc.expect("top_k >= 1");

    let out = if d.has_shared_expert {
        let shared = emit_motif_mlp_2d(
            emit,
            &format!("{prefix}.shared_experts"),
            h2d,
            inter,
            d.poly_shared(),
        )?;
        let mut gb = HirMut::new(emit.hir());
        gb.add(routed, shared)
    } else {
        routed
    };

    let mut gb = HirMut::new(emit.hir());
    Ok(gb.reshape_(out, vec![1, d.seq as i64, d.hidden as i64]))
}

impl MotifMoeDims {
    /// The shared expert is a plain `MotifMLP`, so it uses `PolyNormTorch`
    /// semantics: no bias clamp (already folded) and no clamp on the product.
    fn poly_shared(&self) -> PolyNormSpec {
        PolyNormSpec {
            clamp_result: false,
            ..self.poly
        }
    }
}

/// `MotifMLP` over a `[rows, hidden]` matrix — the dense FFN and the MoE shared
/// expert share this body, differing only in `intermediate_size`.
pub fn emit_motif_mlp_2d(
    emit: &mut Emit<'_>,
    prefix: &str,
    x: HirNodeId,
    inter: usize,
    poly: PolyNormSpec,
) -> Result<HirNodeId> {
    let w_gate = emit.load_param(&format!("{prefix}.gate_proj.weight"), true)?;
    let w_up = emit.load_param(&format!("{prefix}.up_proj.weight"), true)?;
    let w_down = emit.load_param(&format!("{prefix}.down_proj.weight"), true)?;
    let coeff = emit.load_param(&format!("{prefix}.act_fn.coeff"), false)?; // [1, 4]
    let (gate, up) = {
        let mut gb = HirMut::new(emit.hir());
        (gb.mm(x, w_gate), gb.mm(x, w_up))
    };
    let act = emit_poly_norm_mul(emit, prefix, gate, up, coeff, inter, poly)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.mm(act, w_down))
}
