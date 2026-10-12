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

//! `Qwen4ExpTextSparseMoeBlock` — 512 routed experts, 10 active, one always-on
//! shared expert behind its own scalar gate.
//!
//! ```text
//!   probs  = softmax(x @ ffn_gate_inpᵀ)                     f32, [s, 512]
//!   idx, w = topk(probs, 10);  w /= Σ w                     norm_topk_prob
//!   routed = Σ_k w_k · down_{idx_k}(silu(gate_{idx_k}(x)) · up_{idx_k}(x))
//!   shared = σ(x @ ffn_gate_inp_shexpᵀ) · down_s(silu(gate_s(x)) · up_s(x))
//!   out    = routed + shared
//! ```
//!
//! The router is a plain **softmax** top-k with renormalization — no sigmoid,
//! no `e_score_correction_bias`, no expert groups, no `routed_scaling_factor`.
//! That is the degenerate case of [`rlx_llada2`]'s `group_limited_gate` kernel
//! (`n_group = topk_group = 1`, `routed_scaling = 1`), and this reuses it
//! rather than open-coding a second top-k: both inputs are the same softmax
//! probabilities, so selection and weighting read one tensor.
//!
//! Using the shared kernel is not just code economy. Top-k is a comparison, so
//! it is only as stable as the gap between the 10th and 11th score; at 512
//! experts over a 2560-wide dot product that gap can be smaller than the
//! difference between two valid f32 summation orders. A second implementation
//! would agree only while it happened to accumulate the way the graph's
//! lowering does — and nothing announces a BLAS threshold or fusion change.
//!
//! One deliberate deviation: the kernel divides by `Σ w + 1e-20` where the
//! reference divides by `Σ w`. With 10 softmax probabilities the sum is
//! comfortably positive, so the floor never engages; it is there to keep an
//! all-zero row finite.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};
use rlx_llada2::llada2::gate_op::{
    OP_NAME, ensure_group_limited_gate_registered, gate_attrs_bytes,
};

use crate::common::{linear, packed_bank, sigmoid};
use crate::config::Qwen4ExpConfig;

#[derive(Debug, Clone, Copy)]
pub struct MoeDims {
    pub hidden: usize,
    /// `moe_intermediate_size` — one routed expert's FFN width.
    pub expert_ffn: usize,
    /// `shared_expert_intermediate_size`.
    pub shared_ffn: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub seq: usize,
    /// Read the expert banks as **pre-gathered slots** rather than as all
    /// `num_experts` of them.
    ///
    /// `false` is the resident path: the banks are the model's, indexed by the
    /// router's global expert ids. `true` is the paged path — the banks hold
    /// exactly `seq * top_k` experts, slot `r * top_k + k` being the expert row
    /// `r` fired at position `k`, gathered by [`rlx_distributed::ExpertPager`]
    /// before the graph ran.
    ///
    /// The point of the slot layout is that it makes the expert index a
    /// **constant** (`arange(seq) * top_k + k`) instead of router output. A
    /// data-dependent index would force the gather inside the graph, which is
    /// the thing paging exists to avoid: the layer would have to hold all 512
    /// experts to index into them.
    ///
    /// Costs `seq * top_k` slices per bank instead of `num_experts` — for this
    /// model's decode, 10 experts instead of 512.
    pub paged: bool,
}

impl MoeDims {
    /// Derive from the model config, for a step of `seq` positions.
    ///
    /// There is exactly one statement of how a config becomes these dimensions, so
    /// prefill, chunked prefill, the decode step, the MTP block and the tests
    /// cannot disagree about what model they are building. They used to hold four
    /// copies of it.
    /// `paged` defaults off — the banks are the model's own.
    pub fn from_config(cfg: &Qwen4ExpConfig, seq: usize) -> Self {
        MoeDims {
            hidden: cfg.hidden_size,
            expert_ffn: cfg.expert_ffn_size,
            shared_ffn: cfg.shared_expert_ffn_size,
            num_experts: cfg.num_experts,
            top_k: cfg.num_experts_per_tok,
            seq,
            paged: false,
        }
    }
}

/// Emit the MoE FFN for GGUF block `prefix` over `[1, seq, hidden]`.
pub fn emit_moe(
    emit: &mut Emit<'_>,
    prefix: &str,
    hidden: HirNodeId,
    d: MoeDims,
) -> Result<HirNodeId> {
    ensure_group_limited_gate_registered();
    let f = DType::F32;
    let rows = d.seq;

    // The routed banks are most of the model's weight budget — 512 experts ×
    // 3 banks × 48 layers. Take them packed when the source has them; the F32
    // alternative is tens of gigabytes for a single layer.
    let banks = ExpertBanks::load(emit, prefix)?;

    let h2d = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(hidden, vec![rows as i64, d.hidden as i64])
    };

    // ── router: softmax over all experts, then the shared top-k gate ──
    let logits = linear(emit, &format!("{prefix}.ffn_gate_inp.weight"), h2d)?;
    // `Linear(hidden, 1)` — a 1-D GGUF tensor, so it cannot go through
    // `linear`'s `mm` (a rank-1 operand is not a matrix).
    let shared_gate_w = emit.load_param(&format!("{prefix}.ffn_gate_inp_shexp.weight"), false)?;

    // ── shared expert projections (packed when offered) ──
    let sg_lin = linear(emit, &format!("{prefix}.ffn_gate_shexp.weight"), h2d)?;
    let su_lin = linear(emit, &format!("{prefix}.ffn_up_shexp.weight"), h2d)?;

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

    let mut gb = HirMut::new(emit.hir());
    let (top_idx, top_probs) = emit_router_from_logits(&mut gb, logits, d);

    let prepared = banks.prepare(&mut gb);

    let mut acc: Option<HirNodeId> = None;
    for ki in 0..d.top_k {
        let eidx = match &slot_idx {
            Some(slots) => slots[ki],
            None => {
                let idx_col = gb.narrow_(top_idx, 1, ki, 1);
                gb.reshape_(idx_col, vec![rows as i64])
            }
        };
        let prob_col = gb.narrow_(top_probs, 1, ki, 1);
        let prob = gb.reshape_(prob_col, vec![rows as i64, 1]);

        let g = prepared.gate(&mut gb, h2d, eidx, rows, d.expert_ffn);
        let u = prepared.up(&mut gb, h2d, eidx, rows, d.expert_ffn);
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

    // ── shared expert, gated by its own scalar ──
    // The shared projections' width comes from the weights, so check it against
    // the config: a transposed `ffn_gate_shexp` still multiplies cleanly and
    // would only show up as a wrong answer.
    let got = gb.shape(sg_lin).dim(1).unwrap_static();
    if got != d.shared_ffn {
        anyhow::bail!(
            "qwen4exp: {prefix}.ffn_gate_shexp projects to {got}, but \
             shared_expert_intermediate_size is {}",
            d.shared_ffn
        );
    }
    let sact = gb.silu(sg_lin);
    let sh = gb.mul(sact, su_lin);
    let gw = gb.reshape_(shared_gate_w, vec![d.hidden as i64, 1]);
    let sgate = gb.mm(h2d, gw); // [rows, 1]
    let sgate = sigmoid(&mut gb, sgate);
    // `gb` borrows the graph; the down projection needs `emit` back to consult
    // the packed weight source.
    let _ = gb;
    let shared = linear(emit, &format!("{prefix}.ffn_down_shexp.weight"), sh)?;

    let mut gb = HirMut::new(emit.hir());
    let shared = gb.mul(shared, sgate);
    let out2d = gb.add(routed, shared);
    Ok(gb.reshape_(out2d, vec![1, d.seq as i64, d.hidden as i64]))
}

/// `softmax(x @ W_g)` through the group-limited top-k gate, giving
/// `(top_idx, top_probs)` — both `[rows, top_k]`, ids f32-encoded.
///
/// Extracted so that **anything** needing this model's routing runs this exact
/// code rather than a second implementation of it. Paged inference has to know
/// which experts a token fires before it can page them, so the routing has to
/// be reachable outside the layer graph.
///
/// Why it is shared rather than re-derived on the host: top-k is a comparison,
/// so it is only as stable as the gap between the k-th and (k+1)-th score. With
/// 512 experts and `top_k = 10` those gaps are small, while two f32 orderings
/// of the same dot product — a host loop against a BLAS GEMM — differ by more
/// than that. A flipped pick weights the token by another expert's matrix and
/// looks entirely normal.
fn emit_router_from_logits(
    gb: &mut HirMut<'_>,
    logits: HirNodeId,
    d: MoeDims,
) -> (HirNodeId, HirNodeId) {
    let f = DType::F32;
    let attrs = gate_attrs_bytes(1, 1, d.top_k, 1.0, d.num_experts);
    let probs = gb.sm(logits, -1);
    let packed = gb.add_node(
        rlx_ir::op::Op::Custom {
            name: OP_NAME.to_string(),
            num_inputs: 2,
            attrs,
        },
        // Selection scores and weighting scores are the same tensor: there is
        // no correction bias in this router.
        vec![probs, probs],
        Shape::new(&[d.seq, d.top_k * 2], f),
    );
    let top_idx = gb.narrow_(packed, 1, 0, d.top_k);
    let top_probs = gb.narrow_(packed, 1, d.top_k, d.top_k);
    (top_idx, top_probs)
}

/// Experts per routed bank as the **tensors** report it.
///
/// The authoritative source for this number is the bank's own shape, not
/// `qwen4exp.expert_count` in the metadata: a checkpoint can be sliced (the
/// repo's own `scripts/qwen4exp_subset.py` keeps `--experts N` of 512 and
/// leaves the metadata describing all of them), and configuring the metadata's
/// count against sliced banks indexes weights that are not there — or, worse,
/// the wrong ones, which still runs.
///
/// GGUF stores dimensions innermost-first, so the expert count is the **last**
/// entry: `[2560, 640, 8]` is 8 experts of a 2560x640 projection, not 2560 of
/// anything. Reads the header only.
///
/// `None` when the file carries no routed expert bank at all.
pub fn gguf_expert_count(path: impl AsRef<std::path::Path>) -> Result<Option<usize>> {
    let path = path.as_ref();
    let g = rlx_gguf::GgufFile::header_from_path(path)
        .map_err(|e| anyhow::anyhow!("qwen4exp: reading {path:?} header: {e}"))?;
    let mut seen: Option<(String, usize)> = None;
    for t in g.tensors.values() {
        if !t.name.ends_with(".ffn_gate_exps.weight") {
            continue;
        }
        let n = *t
            .shape
            .last()
            .ok_or_else(|| anyhow::anyhow!("qwen4exp: {} has an empty shape", t.name))?;
        match &seen {
            // Every block must agree, or "the expert count" is not a number.
            Some((first, m)) if *m != n => {
                anyhow::bail!("qwen4exp: {first} has {m} experts but {} has {n}", t.name)
            }
            _ => seen = Some((t.name.clone(), n)),
        }
    }
    Ok(seen.map(|(_, n)| n))
}

/// The routing half alone: `[rows, top_k]` global expert ids, f32-encoded.
///
/// The graph a paged runner evaluates before it pages. Same emitter as the
/// layer's, so the ids are the layer graph's by construction rather than by
/// agreement.
pub fn emit_qwen4exp_router(
    emit: &mut Emit<'_>,
    prefix: &str,
    hidden: HirNodeId,
    d: MoeDims,
) -> Result<HirNodeId> {
    ensure_group_limited_gate_registered();
    let h2d = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(hidden, vec![d.seq as i64, d.hidden as i64])
    };
    let logits = linear(emit, &format!("{prefix}.ffn_gate_inp.weight"), h2d)?;
    let mut gb = HirMut::new(emit.hir());
    let (top_idx, _) = emit_router_from_logits(&mut gb, logits, d);
    Ok(top_idx)
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

/// [`ExpertBanks`] after any in-graph reshaping, ready to issue per-slot GEMMs.
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
            match packed_bank(emit, k)? {
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
