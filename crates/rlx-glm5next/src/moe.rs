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

//! `Glm5NextTextMoE` — 288 routed experts, 8 active, one always-on shared
//! expert, `noaux_tc` sigmoid routing, clamped SwiGLU.
//!
//! The router is the DeepSeek-V3 `noaux_tc` gate and is shared verbatim with
//! `rlx_deepseek::moe` through rlx-llada2's `group_limited_gate` custom op:
//! sigmoid the logits, add `e_score_correction_bias` for *selection only*,
//! group-limit, take the top `k`, then renormalize the *unbiased* sigmoid
//! probabilities and scale by `routed_scaling_factor`.
//!
//! Two things differ from `rlx_deepseek::moe` and are why this is a separate
//! emitter rather than a call into it:
//!
//! * **The SwiGLU is clamped.** `gate.clamp(max=10)` and `up.clamp(-10, 10)`
//!   before `silu(gate)·up`, in the routed experts *and* the shared expert.
//!   Unclamped, a single outlier activation changes the layer's output.
//! * **GGUF keeps `ffn_gate_exps` and `ffn_up_exps` as separate banks**, while
//!   `rlx_deepseek::moe` wants HF's fused `gate_up_proj`. Fusing them host-side
//!   would materialize a second copy of every expert bank; at 288 experts ×
//!   42 layers that is not a copy worth making, so both banks are read directly
//!   and the two `GroupedMatMul`s are issued separately.
//!
//! `n_group = topk_group = 1` in GLM-5.3-Flash, so the group limiting is a
//! no-op there — it is still wired because the metadata carries the fields and
//! nothing else in the gate changes.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::{Activation, Op};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};
use rlx_llada2::llada2::gate_op::{
    OP_NAME, ensure_group_limited_gate_registered, gate_attrs_bytes,
};

use crate::common::{clamp_both, clamp_max, linear, scalar};

#[derive(Debug, Clone, Copy)]
pub struct MoeDims {
    pub hidden: usize,
    pub moe_inter: usize,
    pub n_routed: usize,
    pub top_k: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub routed_scaling: f32,
    /// `swiglu_limit`; `None` (or non-finite) disables the clamp.
    pub swiglu_limit: Option<f32>,
    pub seq: usize,
    /// Read the expert banks as **pre-gathered slots** rather than as the whole
    /// `n_routed`-expert banks.
    ///
    /// `None` is the resident path: the banks are the model's, indexed by the
    /// router's global expert ids. `Some(())` is the paged path — the banks
    /// hold exactly `seq * top_k` experts, slot `r * top_k + k` being the
    /// expert row `r` fired at position `k`, gathered by
    /// [`rlx_distributed::ExpertPager`] before the graph ran.
    ///
    /// The point of the slot layout is that it makes the expert index a
    /// CONSTANT (`arange(seq) * top_k + k`) instead of router output. A
    /// data-dependent index would force the gather to happen inside the graph,
    /// which is the thing paging exists to avoid: a layer would have to hold
    /// all 288 experts to index into them.
    ///
    /// Costs `seq * top_k` slices per bank instead of `n_routed` — for
    /// GLM-5.3-Flash decode, 8 experts instead of 288, 0.05 GB instead of
    /// 1.88 GB.
    pub paged: bool,
}

/// The routing half of the MoE: `sigmoid(x @ W_g) + bias` through the
/// group-limited top-k gate, giving `(top_idx, top_probs)`.
///
/// Extracted so that **anything** needing this model's routing runs this exact
/// code rather than a second implementation of it. Paged inference has to know
/// which experts a token fires before it can page them, so the routing has to
/// be available outside the layer graph.
///
/// Why that matters, stated carefully. Top-k selection is a comparison, so it
/// is only as stable as the gap between the k-th and (k+1)-th score. At
/// GLM-5.3-Flash's shape (`hidden = 4096`, 288 experts) that gap was measured
/// as tight as **4.6e-5**, while two f32 orderings of the same dot product — a
/// sequential loop against a BLAS GEMM — differ by up to **2.7e-4**. The
/// perturbation is an order of magnitude larger than the decision margin, and
/// when it flips a pick the token is weighted by a different expert's matrix
/// with no other symptom.
///
/// A host reimplementation is therefore not wrong so much as *conditionally*
/// right: it agrees only while it happens to accumulate the way the graph's
/// lowering does. (Measured: a naive host loop matched this router on 400/400
/// inputs at that shape — so the hazard is latent, not active.) A BLAS
/// threshold, a fusion decision, or a different device moves the lowering and
/// nothing announces it. One source of truth removes the coincidence.
fn emit_router_from_logits(
    gb: &mut HirMut<'_>,
    logits: HirNodeId,
    ebias: HirNodeId,
    d: MoeDims,
    attrs: Vec<u8>,
) -> (HirNodeId, HirNodeId) {
    let f = DType::F32;
    let rows = d.seq;
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
    (top_idx, top_probs)
}

/// Emit ONLY the router for block `prefix` over `[1, seq, hidden]`, returning
/// the fired expert ids as `[seq, top_k]`.
///
/// The graph a paged runner evaluates before it pages. Same emitters, same
/// weights, same kernels as [`emit_glm5next_moe`]'s internal routing, so the
/// two agree bit for bit — see the private `emit_router_from_logits` for why that is not
/// something to leave to chance.
pub fn emit_glm5next_router(
    emit: &mut Emit<'_>,
    prefix: &str,
    hidden: HirNodeId,
    d: MoeDims,
) -> Result<HirNodeId> {
    ensure_group_limited_gate_registered();
    let ebias = emit.load_param(&format!("{prefix}.exp_probs_b.bias"), false)?;
    let attrs = gate_attrs_bytes(
        d.n_group,
        d.topk_group,
        d.top_k,
        d.routed_scaling,
        d.n_routed,
    );
    let h2d = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(hidden, vec![d.seq as i64, d.hidden as i64])
    };
    let logits = crate::common::linear(emit, &format!("{prefix}.ffn_gate_inp.weight"), h2d)?;
    let mut gb = HirMut::new(emit.hir());
    let (top_idx, _) = emit_router_from_logits(&mut gb, logits, ebias, d, attrs);
    Ok(top_idx)
}

/// Emit the MoE FFN for GGUF block `prefix` (`blk.{i}`) over `[1, seq, hidden]`.
pub fn emit_glm5next_moe(
    emit: &mut Emit<'_>,
    prefix: &str,
    hidden: HirNodeId,
    d: MoeDims,
) -> Result<HirNodeId> {
    ensure_group_limited_gate_registered();
    let f = DType::F32;
    let rows = d.seq;
    let inter = d.moe_inter;

    let ebias = emit.load_param(&format!("{prefix}.exp_probs_b.bias"), false)?;

    // The routed banks are the whole model's weight budget — 288 experts × 3
    // banks × 42 layers. Take them packed when the source has them: the F32
    // alternative is tens of GB for a single layer.
    let banks = ExpertBanks::load(emit, prefix)?;

    let bounds = d.swiglu_limit.filter(|l| l.is_finite()).map(|l| {
        (
            scalar(emit, &format!("{prefix}.moe.swiglu_lo"), -l),
            scalar(emit, &format!("{prefix}.moe.swiglu_hi"), l),
        )
    });

    // The paged path's expert index, materialized before the graph borrow.
    let slot_idx: Option<Vec<HirNodeId>> = d.paged.then(|| {
        (0..d.top_k)
            .map(|ki| {
                let slots: Vec<f32> = (0..rows).map(|r| (r * d.top_k + ki) as f32).collect();
                emit.synth_param(
                    &format!("{prefix}.moe.slot_idx.{ki}"),
                    slots,
                    Shape::new(&[rows], f),
                )
            })
            .collect()
    });

    let attrs = gate_attrs_bytes(
        d.n_group,
        d.topk_group,
        d.top_k,
        d.routed_scaling,
        d.n_routed,
    );

    let h2d = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(hidden, vec![rows as i64, d.hidden as i64])
    };

    // ── group-limited sigmoid router → (top_idx, top_probs) ──
    // The router is F32 even in a quantized GGUF, so this takes the dense path;
    // it goes through `linear` anyway so nothing here has to know that.
    let logits = crate::common::linear(emit, &format!("{prefix}.ffn_gate_inp.weight"), h2d)?;

    // ── shared expert projections (packed when offered) ──
    let sg_lin = crate::common::linear(emit, &format!("{prefix}.ffn_gate_shexp.weight"), h2d)?;
    let su_lin = crate::common::linear(emit, &format!("{prefix}.ffn_up_shexp.weight"), h2d)?;

    let mut gb = HirMut::new(emit.hir());
    let (top_idx, top_probs) = emit_router_from_logits(&mut gb, logits, ebias, d, attrs);

    let prepared = banks.prepare(&mut gb);

    let mut acc: Option<HirNodeId> = None;
    for ki in 0..d.top_k {
        // Resident: index the model's banks by the router's global expert id.
        // Paged: the bank IS this token's gather, so slot `r * top_k + ki` is
        // by construction the expert row `r` fired at position `ki` — a
        // constant, and `top_idx` is only used for the gather that already
        // happened on the host.
        let eidx = match slot_idx {
            Some(ref slots) => slots[ki],
            None => {
                let idx_col = gb.narrow_(top_idx, 1, ki, 1);
                gb.reshape_(idx_col, vec![rows as i64])
            }
        };
        let prob_col = gb.narrow_(top_probs, 1, ki, 1);
        let prob = gb.reshape_(prob_col, vec![rows as i64, 1]);

        let g = prepared.gate(&mut gb, h2d, eidx, rows, inter);
        let u = prepared.up(&mut gb, h2d, eidx, rows, inter);
        let (g, u) = match bounds {
            Some((lo, hi)) => (clamp_max(&mut gb, g, hi), clamp_both(&mut gb, u, lo, hi)),
            None => (g, u),
        };
        let a = gb.silu(g);
        let hx = gb.mul(a, u);
        let down = prepared.down(&mut gb, hx, eidx, rows, d.hidden);
        let weighted = gb.mul(down, prob);
        acc = Some(match acc {
            Some(prev) => gb.add(prev, weighted),
            None => weighted,
        });
    }
    let routed = acc.expect("top_k >= 1");

    // ── shared expert, same clamped SwiGLU, added to the routed sum ──
    let (sg, su) = (sg_lin, su_lin);
    let (sg, su) = match bounds {
        Some((lo, hi)) => (clamp_max(&mut gb, sg, hi), clamp_both(&mut gb, su, lo, hi)),
        None => (sg, su),
    };
    let sact = gb.silu(sg);
    let sh = gb.mul(sact, su);
    let routed_out = routed;
    // `gb` borrows the graph; the down-projection needs `emit` back to consult
    // the packed weight source.
    let _ = gb;
    let shared = crate::common::linear(emit, &format!("{prefix}.ffn_down_shexp.weight"), sh)?;
    let mut gb = HirMut::new(emit.hir());
    let out2d = gb.add(shared, routed_out);
    Ok(gb.reshape_(out2d, vec![1, d.seq as i64, d.hidden as i64]))
}

/// The dense MLP used by the first `first_k_dense_replace` layers — same
/// clamped SwiGLU at `intermediate_size` width.
pub fn emit_dense_mlp(
    emit: &mut Emit<'_>,
    prefix: &str,
    hidden: HirNodeId,
    seq: usize,
    hidden_size: usize,
    swiglu_limit: Option<f32>,
) -> Result<HirNodeId> {
    let x2d = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(hidden, vec![seq as i64, hidden_size as i64])
    };
    let gate = linear(emit, &format!("{prefix}.ffn_gate.weight"), x2d)?;
    let up = linear(emit, &format!("{prefix}.ffn_up.weight"), x2d)?;
    let act = crate::common::clamped_swiglu(emit, &format!("{prefix}.mlp"), gate, up, swiglu_limit);
    let down = linear(emit, &format!("{prefix}.ffn_down.weight"), act)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.reshape_(down, vec![1, seq as i64, hidden_size as i64]))
}

/// The three routed expert banks, packed or dense.
///
/// Packed and dense differ in more than precision, which is why this is a type
/// rather than an `if`: `Op::DequantGroupedMatMul` reads GGUF's native
/// `[experts, out, in]` slabs directly, while `Op::GroupedMatMul` wants
/// `[experts, in, out]` and so needs an in-graph transpose — which
/// constant-folding then materializes as a *second* copy of the bank.
enum ExpertBanks {
    Dense {
        gate: HirNodeId,
        up: HirNodeId,
        down: HirNodeId,
    },
    Packed {
        gate: (HirNodeId, QuantScheme),
        up: (HirNodeId, QuantScheme),
        down: (HirNodeId, QuantScheme),
    },
}

/// [`ExpertBanks`] after any in-graph reshaping, ready to issue per-expert GEMMs.
enum PreparedBanks {
    Dense {
        gate: HirNodeId,
        up: HirNodeId,
        down: HirNodeId,
    },
    Packed {
        gate: (HirNodeId, QuantScheme),
        up: (HirNodeId, QuantScheme),
        down: (HirNodeId, QuantScheme),
    },
}

impl ExpertBanks {
    fn load(emit: &mut Emit<'_>, prefix: &str) -> Result<Self> {
        let keys = [
            format!("{prefix}.ffn_gate_exps.weight"),
            format!("{prefix}.ffn_up_exps.weight"),
            format!("{prefix}.ffn_down_exps.weight"),
        ];
        // All three must agree: a half-packed layer would silently mix layouts.
        let mut packed = Vec::with_capacity(3);
        for k in &keys {
            match crate::common::packed_bank(emit, k)? {
                Some((node, scheme, _)) => packed.push((node, scheme)),
                None => {
                    packed.clear();
                    break;
                }
            }
        }
        if packed.len() == 3 {
            return Ok(Self::Packed {
                gate: packed[0],
                up: packed[1],
                down: packed[2],
            });
        }
        Ok(Self::Dense {
            gate: emit.load_param(&keys[0], false)?,
            up: emit.load_param(&keys[1], false)?,
            down: emit.load_param(&keys[2], false)?,
        })
    }

    fn prepare(self, gb: &mut HirMut<'_>) -> PreparedBanks {
        match self {
            // Banks arrive as [E, out, in]; `GroupedMatMul` contracts [E, in, out].
            Self::Dense { gate, up, down } => PreparedBanks::Dense {
                gate: gb.transpose_(gate, vec![0, 2, 1]),
                up: gb.transpose_(up, vec![0, 2, 1]),
                down: gb.transpose_(down, vec![0, 2, 1]),
            },
            // Packed slabs are already [E, out, in] — the op transposes internally.
            Self::Packed { gate, up, down } => PreparedBanks::Packed { gate, up, down },
        }
    }
}

impl PreparedBanks {
    fn issue(
        gb: &mut HirMut<'_>,
        bank: Result<HirNodeId, (HirNodeId, QuantScheme)>,
        x: HirNodeId,
        eidx: HirNodeId,
        rows: usize,
        out: usize,
    ) -> HirNodeId {
        match bank {
            Ok(dense) => gb.grouped_matmul(x, dense, eidx),
            Err((wq, scheme)) => gb.0.dequant_grouped_matmul_packed(
                x,
                wq,
                eidx,
                scheme,
                Shape::new(&[rows, out], DType::F32),
            ),
        }
    }
    fn gate(
        &self,
        gb: &mut HirMut<'_>,
        x: HirNodeId,
        eidx: HirNodeId,
        rows: usize,
        out: usize,
    ) -> HirNodeId {
        let b = match self {
            Self::Dense { gate, .. } => Ok(*gate),
            Self::Packed { gate, .. } => Err(*gate),
        };
        Self::issue(gb, b, x, eidx, rows, out)
    }
    fn up(
        &self,
        gb: &mut HirMut<'_>,
        x: HirNodeId,
        eidx: HirNodeId,
        rows: usize,
        out: usize,
    ) -> HirNodeId {
        let b = match self {
            Self::Dense { up, .. } => Ok(*up),
            Self::Packed { up, .. } => Err(*up),
        };
        Self::issue(gb, b, x, eidx, rows, out)
    }
    fn down(
        &self,
        gb: &mut HirMut<'_>,
        x: HirNodeId,
        eidx: HirNodeId,
        rows: usize,
        out: usize,
    ) -> HirNodeId {
        let b = match self {
            Self::Dense { down, .. } => Ok(*down),
            Self::Packed { down, .. } => Err(*down),
        };
        Self::issue(gb, b, x, eidx, rows, out)
    }
}
