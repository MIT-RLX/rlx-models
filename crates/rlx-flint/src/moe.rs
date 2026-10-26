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

//! `FlintSparseMoeBlock` — 256 routed experts, 6 active, plus an ungated
//! shared expert.
//!
//! ```text
//!   scores = σ(x @ gateᵀ)                         [rows, 256], f32
//!   ids    = topk(scores + expert_bias, 6)        bias picks, never weighs
//!   w      = scores[ids] / (Σ scores[ids] + 1e-20)
//!   out    = Σ_k w_k · down_k(silu(gate_k(x)) · up_k(x))  +  shared(x)
//! ```
//!
//! The selection is `rlx-llada2`'s `group_limited_gate` kernel with one
//! group: its first input weighs, its second selects, and it divides by
//! `Σ + 1e-20` — the reference's routing function to the epsilon. The resident
//! flow runs it in the graph ([`emit_router`]). The paged runner takes the
//! sigmoid scores out of the layer graph ([`emit_router_scores`]) and runs the
//! same kernel's CPU body on the host ([`select_experts`]), so the sparse
//! layers lower on every backend, not only those with the custom op. Either
//! way the scores come from one dot product, and the pick from one top-k
//! implementation.
//!
//! The shared expert has **no gate** (unlike Qwen2-MoE's `σ(shared_gate(x))`);
//! the checkpoint has no `shared_expert_gate` tensor and the reference adds it
//! straight in.
//!
//! Routed experts are expressed over a **bank** `[slots, in, out]` and a slot
//! index per (row, k). Resident, the bank is all 256 experts and the index is
//! the router's expert id. Paged ([`crate::runner`]), the bank holds only the
//! experts this step fired and the index points into it — the same emitter, so
//! the two paths cannot disagree about the math.

use anyhow::{Result, bail};
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::{Activation, Op};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};
use rlx_llada2::llada2::gate_op::{
    OP_NAME, ensure_group_limited_gate_registered, gate_attrs_bytes,
};

use crate::checkpoint::Checkpoint;
use crate::common::linear;
use crate::config::FlintConfig;

/// The three routed projections, in the order banks are built and paged.
pub const EXPERT_PROJS: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

/// HF name of one routed expert's projection.
pub fn expert_key(layer: usize, expert: usize, proj: &str) -> String {
    format!("model.layers.{layer}.mlp.experts.{expert}.{proj}.weight")
}

/// Router output: `[rows, top_k]` each. `idx` is f32-encoded.
#[derive(Debug, Clone, Copy)]
pub struct Routing {
    pub idx: HirNodeId,
    pub weight: HirNodeId,
}

/// HF name of a sparse layer's selection bias. Absent from the published
/// checkpoint (the reference zero-fills it), but part of the architecture: it
/// shifts selection only.
pub fn expert_bias_key(layer: usize) -> String {
    format!("model.layers.{layer}.mlp.gate.expert_bias")
}

/// `σ(x @ gateᵀ)` for `model.layers.{layer}.mlp` over `x2d` `[rows, hidden]`:
/// `[rows, num_experts]`.
pub fn emit_router_scores(emit: &mut Emit<'_>, layer: usize, x2d: HirNodeId) -> Result<HirNodeId> {
    let logits = linear(emit, &format!("model.layers.{layer}.mlp.gate.weight"), x2d)?;
    let mut gb = HirMut::new(emit.hir());
    let s = gb.shape(logits).clone();
    Ok(gb.add_node(Op::Activation(Activation::Sigmoid), vec![logits], s))
}

/// Emit the router for `model.layers.{layer}.mlp` over `x2d` `[rows, hidden]`,
/// selection included.
pub fn emit_router(
    emit: &mut Emit<'_>,
    cfg: &FlintConfig,
    layer: usize,
    x2d: HirNodeId,
) -> Result<Routing> {
    ensure_group_limited_gate_registered();
    let scores = emit_router_scores(emit, layer, x2d)?;
    let bias_key = expert_bias_key(layer);
    let bias = if emit.weights.has(&bias_key) {
        Some(emit.load_param(&bias_key, false)?)
    } else {
        None
    };
    let (e, k) = (cfg.num_experts, cfg.num_experts_per_tok);
    let mut gb = HirMut::new(emit.hir());
    let rows = gb.shape(x2d).dim(0).unwrap_static();
    let choice = match bias {
        Some(b) => gb.add(scores, b),
        None => scores,
    };
    let packed = gb.add_node(
        Op::Custom {
            name: OP_NAME.to_string(),
            num_inputs: 2,
            attrs: gate_attrs_bytes(1, 1, k, 1.0, e),
        },
        vec![scores, choice],
        Shape::new(&[rows, 2 * k], DType::F32),
    );
    Ok(Routing {
        idx: gb.narrow_(packed, 1, 0, k),
        weight: gb.narrow_(packed, 1, k, k),
    })
}

/// A sparse layer's selection bias, if the checkpoint has one.
pub fn load_expert_bias(ckpt: &Checkpoint, layer: usize) -> Result<Option<Vec<f32>>> {
    let key = expert_bias_key(layer);
    if !ckpt.contains(&key) {
        return Ok(None);
    }
    Ok(Some(ckpt.load_f32(&key)?.0))
}

/// The router's selection on the host, over sigmoid `scores`
/// `[rows, num_experts]`: `(idx, w)`, each `[rows, top_k]`, `idx` f32-encoded
/// like [`Routing::idx`]. Runs the CPU body of the kernel [`emit_router`]
/// puts in the graph, so both paths pick the same experts, ties included.
pub fn select_experts(
    cfg: &FlintConfig,
    scores: &[f32],
    bias: Option<&[f32]>,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let (e, k) = (cfg.num_experts, cfg.num_experts_per_tok);
    if !scores.len().is_multiple_of(e) {
        bail!("flint: {} scores are not rows of {e}", scores.len());
    }
    let rows = scores.len() / e;
    let choice: Vec<f32>;
    let route = match bias {
        Some(b) => {
            if b.len() != e {
                bail!("flint: expert_bias has {} entries, not {e}", b.len());
            }
            choice = scores
                .chunks_exact(e)
                .flat_map(|row| row.iter().zip(b).map(|(s, b)| s + b))
                .collect();
            &choice[..]
        }
        None => scores,
    };
    let mut packed = vec![0.0f32; rows * 2 * k];
    rlx_cpu::llada2_gate::execute_gate_f32(
        scores,
        route,
        &mut packed,
        &gate_attrs_bytes(1, 1, k, 1.0, e),
    )
    .map_err(|err| anyhow::anyhow!("flint: expert selection: {err}"))?;
    let (mut idx, mut w) = (Vec::with_capacity(rows * k), Vec::with_capacity(rows * k));
    for row in packed.chunks_exact(2 * k) {
        idx.extend_from_slice(&row[..k]);
        w.extend_from_slice(&row[k..]);
    }
    Ok((idx, w))
}

/// Routed expert banks in the grouped-GEMM orientation `[slots, in, out]`.
#[derive(Debug, Clone, Copy)]
pub struct ExpertBanks {
    /// `[slots, hidden, moe_ffn]`.
    pub gate: HirNodeId,
    /// `[slots, hidden, moe_ffn]`.
    pub up: HirNodeId,
    /// `[slots, moe_ffn, hidden]`.
    pub down: HirNodeId,
}

/// `Σ_k w[:, k] · expert_{idx[:, k]}(x)` over `x2d` `[rows, hidden]`.
///
/// `slot_idx` is `[rows, top_k]` (f32-encoded) and indexes the banks' first
/// axis; `weight` is `[rows, top_k]`.
pub fn emit_routed(
    gb: &mut HirMut<'_>,
    x2d: HirNodeId,
    banks: ExpertBanks,
    slot_idx: HirNodeId,
    weight: HirNodeId,
    top_k: usize,
) -> HirNodeId {
    let rows = gb.shape(x2d).dim(0).unwrap_static() as i64;
    let mut acc: Option<HirNodeId> = None;
    for ki in 0..top_k {
        let col = gb.narrow_(slot_idx, 1, ki, 1);
        let eidx = gb.reshape_(col, vec![rows]);
        let w = gb.narrow_(weight, 1, ki, 1); // [rows, 1]
        let g = gb.grouped_matmul(x2d, banks.gate, eidx);
        let u = gb.grouped_matmul(x2d, banks.up, eidx);
        let a = gb.silu(g);
        let h = gb.mul(a, u);
        let y = gb.grouped_matmul(h, banks.down, eidx);
        let y = gb.mul(y, w);
        acc = Some(match acc {
            Some(prev) => gb.add(prev, y),
            None => y,
        });
    }
    acc.expect("top_k >= 1")
}

/// Transpose a row-major `[rows, cols]` matrix into `out` as `[cols, rows]`.
pub fn transpose_into(src: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    debug_assert_eq!(src.len(), rows * cols);
    debug_assert_eq!(out.len(), rows * cols);
    const B: usize = 32;
    for r0 in (0..rows).step_by(B) {
        for c0 in (0..cols).step_by(B) {
            for r in r0..(r0 + B).min(rows) {
                for c in c0..(c0 + B).min(cols) {
                    out[c * rows + r] = src[r * cols + c];
                }
            }
        }
    }
}

/// Load all of a layer's experts as resident banks.
///
/// The checkpoint stores one `[out, in]` tensor per expert per projection;
/// this stacks them and transposes each to `[in, out]`. For the real model that
/// is 1.6 GB of f32 *per layer* — the resident path is for synthetic weights
/// and small subsets. The full checkpoint pages experts instead.
pub fn load_resident_banks(
    emit: &mut Emit<'_>,
    cfg: &FlintConfig,
    layer: usize,
) -> Result<ExpertBanks> {
    let (e, h, i) = (cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size);
    let mut nodes = Vec::with_capacity(3);
    for proj in EXPERT_PROJS {
        // gate/up: [i, h] → [h, i];  down: [h, i] → [i, h].
        let (out_dim, in_dim) = if proj == "down_proj" { (h, i) } else { (i, h) };
        let mut bank = vec![0.0f32; e * out_dim * in_dim];
        for x in 0..e {
            let key = expert_key(layer, x, proj);
            let (data, shape) = emit.weights.take(&key, false)?;
            if shape != [out_dim, in_dim] {
                bail!("flint: {key} is {shape:?}, expected [{out_dim}, {in_dim}]");
            }
            let n = out_dim * in_dim;
            transpose_into(&data, out_dim, in_dim, &mut bank[x * n..(x + 1) * n]);
        }
        let name = format!("model.layers.{layer}.mlp.experts.{proj}.bank");
        nodes.push(emit.synth_param(&name, bank, Shape::new(&[e, in_dim, out_dim], DType::F32)));
    }
    Ok(ExpertBanks {
        gate: nodes[0],
        up: nodes[1],
        down: nodes[2],
    })
}

/// The whole sparse block, resident: router, all experts, shared expert.
/// `x2d` is the post-attention-normed `[rows, hidden]`.
pub fn emit_moe_resident(
    emit: &mut Emit<'_>,
    cfg: &FlintConfig,
    layer: usize,
    x2d: HirNodeId,
) -> Result<HirNodeId> {
    let route = emit_router(emit, cfg, layer, x2d)?;
    let banks = load_resident_banks(emit, cfg, layer)?;
    let shared = emit_shared(emit, cfg, layer, x2d)?;
    let mut gb = HirMut::new(emit.hir());
    let routed = emit_routed(
        &mut gb,
        x2d,
        banks,
        route.idx,
        route.weight,
        cfg.num_experts_per_tok,
    );
    Ok(match shared {
        Some(s) => gb.add(routed, s),
        None => routed,
    })
}

/// The ungated shared expert, or `None` when the config has none.
pub fn emit_shared(
    emit: &mut Emit<'_>,
    cfg: &FlintConfig,
    layer: usize,
    x2d: HirNodeId,
) -> Result<Option<HirNodeId>> {
    if cfg.shared_expert_intermediate_size == 0 {
        return Ok(None);
    }
    crate::common::swiglu(
        emit,
        &format!("model.layers.{layer}.mlp.shared_expert"),
        x2d,
    )
    .map(Some)
}
