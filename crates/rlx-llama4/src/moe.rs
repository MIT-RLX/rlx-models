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

//! Llama-4 mixture-of-experts FFN (`Llama4TextMoe`).
//!
//! Top-`k` routing (Scout uses top-1) with the routing weight applied to the
//! expert **input** (`hidden · sigmoid(top_logit)` — nonlinear-correct, matching
//! HF), a grouped-GEMM over the selected expert, plus an always-on shared
//! expert:
//! ```text
//!   logits = h @ routerᵀ;  idx = topk(logits, k);  score = sigmoid(logits[idx])
//!   scaled = h · score
//!   expert = down_e( silu(gate) · up )  where [gate|up] = scaled @ gate_up_proj[idx]
//!   out    = shared_expert(h) + Σ_k expert_k
//! ```
//! Expert weights are stored `[E, K, N]` (`gate_up_proj [E,H,2·inter]`,
//! `down_proj [E,inter,H]`) — exactly the `Op::GroupedMatMul` layout.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::{Activation, Op};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

/// Emit the MoE FFN for `model.layers.{i}.feed_forward` (`prefix`), consuming a
/// `[1, seq, hidden]` input and producing `[1, seq, hidden]`.
pub fn emit_moe_ffn(
    emit: &mut Emit<'_>,
    prefix: &str,
    hidden: HirNodeId,
    seq: usize,
    hidden_size: usize,
    inter: usize,
    top_k: usize,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let rows = seq;

    let router_w = emit.load_param(&format!("{prefix}.router.weight"), true)?;
    let gate_up_w = emit.load_param(&format!("{prefix}.experts.gate_up_proj"), false)?;
    let down_w = emit.load_param(&format!("{prefix}.experts.down_proj"), false)?;
    let s_gate = emit.load_param(&format!("{prefix}.shared_expert.gate_proj.weight"), true)?;
    let s_up = emit.load_param(&format!("{prefix}.shared_expert.up_proj.weight"), true)?;
    let s_down = emit.load_param(&format!("{prefix}.shared_expert.down_proj.weight"), true)?;

    let mut gb = HirMut::new(emit.hir());
    let h2d = gb.reshape_(hidden, vec![rows as i64, hidden_size as i64]);

    // Router: raw logits → top-k indices → sigmoid of the selected logit.
    let logits = gb.mm(h2d, router_w); // [rows, E]
    let top_idx = gb.add_node(
        Op::TopK { k: top_k },
        vec![logits],
        Shape::new(&[rows, top_k], f),
    );
    let top_val = gb.gather_(logits, top_idx, 1); // [rows, top_k]

    // Sum over the k selected experts (input-scaled), then add the shared expert.
    let mut routed: Option<HirNodeId> = None;
    for ki in 0..top_k {
        let idx_col = gb.narrow_(top_idx, 1, ki, 1);
        let expert_idx = gb.reshape_(idx_col, vec![rows as i64]);
        let val_col = gb.narrow_(top_val, 1, ki, 1);
        let val_2d = gb.reshape_(val_col, vec![rows as i64, 1]); // normalize rank
        let score = gb.activation(Activation::Sigmoid, val_2d, Shape::new(&[rows, 1], f));
        let scaled = gb.mul(h2d, score); // [rows, hidden]

        let gate_up = gb.add_node(
            Op::GroupedMatMul,
            vec![scaled, gate_up_w, expert_idx],
            Shape::new(&[rows, 2 * inter], f),
        );
        let gate = gb.narrow_(gate_up, 1, 0, inter);
        let up = gb.narrow_(gate_up, 1, inter, inter);
        let act = gb.silu(gate);
        let hx = gb.mul(act, up); // [rows, inter]
        let out = gb.add_node(
            Op::GroupedMatMul,
            vec![hx, down_w, expert_idx],
            Shape::new(&[rows, hidden_size], f),
        );
        routed = Some(match routed {
            Some(acc) => gb.add(acc, out),
            None => out,
        });
    }

    // Shared expert (SwiGLU, always on).
    let sg = gb.mm(h2d, s_gate);
    let su = gb.mm(h2d, s_up);
    let sact = gb.silu(sg);
    let sh = gb.mul(sact, su);
    let shared = gb.mm(sh, s_down);

    let out2d = match routed {
        Some(r) => gb.add(shared, r),
        None => shared,
    };
    Ok(gb.reshape_(out2d, vec![1, seq as i64, hidden_size as i64]))
}
