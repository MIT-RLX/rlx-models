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

//! Gemma 4 sparse MoE — `Gemma4TextRouter` + `Gemma4TextExperts`.
//!
//! The router is *not* the DeepSeek `noaux_tc` gate: there is no sigmoid, no
//! correction bias and no group limiting. It scale-free RMS-norms the token,
//! rescales by a learned per-channel `scale` times `hidden^-0.5`, projects to
//! `num_experts`, softmaxes, takes top-`k`, renormalizes the k weights to sum to
//! 1, and finally multiplies each by its expert's `per_expert_scale`.
//!
//! Experts are the usual stacked-bank SwiGLU-shaped FFN, except the activation
//! is `gelu_pytorch_tanh` rather than SiLU, and the two halves come from one
//! packed `gate_up_proj` (gate = first `moe_intermediate_size` columns).

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::{Op, ReduceOp};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

/// Geometry of one MoE block.
#[derive(Debug, Clone, Copy)]
pub struct MoeDims {
    pub hidden: usize,
    /// Per-expert FFN width (`moe_intermediate_size`, 704) — *not*
    /// `intermediate_size`, which sizes the always-on shared `mlp`.
    pub moe_inter: usize,
    pub num_experts: usize,
    pub top_k: usize,
    /// Flattened token count (`batch · seq`).
    pub rows: usize,
    pub eps: f32,
    /// `hidden^-0.5` (`Gemma4TextRouter.scalar_root_size`).
    pub root_scale: f32,
    /// Expert banks already stored in `GroupedMatMul`'s `[E, K, N]` layout
    /// (`gate_up_proj` as `[E, hidden, 2·inter]`, `down_proj` as
    /// `[E, inter, hidden]`), so the in-graph transpose can be skipped.
    ///
    /// Transposing in-graph is not free: constant folding materializes a second
    /// copy of the whole bank in the arena. For DiffusionGemma-26B that is
    /// 20.3 GB → 40.6 GB of f32 experts, so loaders should pre-transpose.
    pub experts_pretransposed: bool,
}

/// Scale-free RMS norm (`Gemma4RMSNorm(..., with_scale=False)`).
fn rms_no_scale(emit: &mut Emit<'_>, tag: &str, x: HirNodeId, dim: usize, eps: f32) -> HirNodeId {
    let ones = emit.synth_param(
        &format!("{tag}.ones"),
        vec![1.0; dim],
        Shape::new(&[dim], DType::F32),
    );
    let zeros = emit.synth_param(
        &format!("{tag}.zeros"),
        vec![0.0; dim],
        Shape::new(&[dim], DType::F32),
    );
    let mut gb = HirMut::new(emit.hir());
    gb.rms_norm(x, ones, zeros, eps)
}

/// `Gemma4TextRouter` on flattened tokens `[rows, hidden]`.
///
/// Returns `(top_idx, top_weights)`, both `[rows, top_k]`; indices are
/// f32-encoded (rlx's I/O convention).
pub fn emit_router(
    emit: &mut Emit<'_>,
    prefix: &str,
    h2d: HirNodeId,
    d: MoeDims,
) -> Result<(HirNodeId, HirNodeId)> {
    let f = DType::F32;
    let (rows, k) = (d.rows, d.top_k);

    let normed = rms_no_scale(emit, &format!("{prefix}.norm"), h2d, d.hidden, d.eps);
    let scale = emit.load_param(&format!("{prefix}.scale"), false)?; // [hidden]
    let proj = emit.load_param(&format!("{prefix}.proj.weight"), true)?; // [hidden, E]
    let pes = emit.load_param(&format!("{prefix}.per_expert_scale"), false)?; // [E]
    let root = emit.synth_param(
        &format!("{prefix}.root_scale"),
        vec![d.root_scale],
        Shape::new(&[1], f),
    );

    let mut gb = HirMut::new(emit.hir());
    let scale_2d = gb.reshape_(scale, vec![1, d.hidden as i64]);
    let x = gb.mul(normed, scale_2d);
    let x = gb.mul(x, root);

    let logits = gb.mm(x, proj); // [rows, E]
    let probs = gb.sm(logits, -1);

    let top_idx = gb.add_node(Op::TopK { k }, vec![probs], Shape::new(&[rows, k], f));
    let top_w = gb.add_node(
        Op::GatherElements { axis: 1 },
        vec![probs, top_idx],
        Shape::new(&[rows, k], f),
    );

    // Renormalize the k weights to sum to 1, then apply the per-expert scale.
    let denom = gb.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![1],
            keep_dim: true,
        },
        vec![top_w],
        Shape::new(&[rows, 1], f),
    );
    let top_w = gb.div(top_w, denom);

    // `per_expert_scale[top_idx]`: 1-D table gathered by a 2-D index block.
    let pes_gathered = gb.gather_(pes, top_idx, 0); // [rows, k]
    let top_w = gb.mul(top_w, pes_gathered);
    Ok((top_idx, top_w))
}

/// `Gemma4TextExperts` on flattened tokens `[rows, hidden]`, dispatching the
/// `top_k` selected banks and accumulating the weighted outputs.
pub fn emit_experts(
    emit: &mut Emit<'_>,
    prefix: &str,
    h2d: HirNodeId,
    top_idx: HirNodeId,
    top_w: HirNodeId,
    d: MoeDims,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let (rows, inter) = (d.rows, d.moe_inter);

    let gate_up_w = emit.load_param(&format!("{prefix}.gate_up_proj"), false)?;
    let down_w = emit.load_param(&format!("{prefix}.down_proj"), false)?;

    // Guard the bank layout. `experts_pretransposed` is easy to get wrong, and
    // a silently mis-oriented bank produces plausible-looking garbage rather
    // than an error — this is what catches it at build time.
    let want_gate_up = if d.experts_pretransposed {
        [d.num_experts, d.hidden, 2 * inter]
    } else {
        [d.num_experts, 2 * inter, d.hidden]
    };
    let got_dims = {
        let gb = HirMut::new(emit.hir());
        let got = gb.shape(gate_up_w);
        [
            got.dim(0).unwrap_static(),
            got.dim(1).unwrap_static(),
            got.dim(2).unwrap_static(),
        ]
    };
    anyhow::ensure!(
        got_dims == want_gate_up,
        "{prefix}.gate_up_proj is {got_dims:?}, expected {want_gate_up:?} \
         (experts_pretransposed = {}) — did `prepare_checkpoint` run?",
        d.experts_pretransposed
    );

    let mut gb = HirMut::new(emit.hir());
    // Checkpoint layout is [E, 2·inter, hidden] / [E, hidden, inter];
    // GroupedMatMul wants [E, K, N].
    let (gate_up_t, down_t) = if d.experts_pretransposed {
        (gate_up_w, down_w)
    } else {
        (
            gb.transpose_(gate_up_w, vec![0, 2, 1]), // [E, hidden, 2·inter]
            gb.transpose_(down_w, vec![0, 2, 1]),    // [E, inter, hidden]
        )
    };

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
        let gate = gb.narrow_(gate_up, 1, 0, inter);
        let up = gb.narrow_(gate_up, 1, inter, inter);
        // `gelu_pytorch_tanh`, not SiLU.
        let act = gb.gelu_approx(gate);
        let hx = gb.mul(act, up);
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
    Ok(acc.expect("top_k >= 1"))
}

/// Router + experts for `…layers.{i}` — `prefix` is the layer prefix, since the
/// two live side by side as `.router` and `.experts`.
pub fn emit_moe(
    emit: &mut Emit<'_>,
    layer_prefix: &str,
    routing_input: HirNodeId,
    expert_input: HirNodeId,
    d: MoeDims,
) -> Result<HirNodeId> {
    let (idx, w) = emit_router(emit, &format!("{layer_prefix}.router"), routing_input, d)?;
    emit_experts(
        emit,
        &format!("{layer_prefix}.experts"),
        expert_input,
        idx,
        w,
        d,
    )
}
