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
use rlx_ir::quant::QuantScheme;
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
    /// Expert banks are already stored in `GroupedMatMul`'s `[E, K, N]` layout,
    /// so skip the in-graph transpose.
    ///
    /// The transpose is not free: constant-folding materializes a *second* copy
    /// of the whole expert bank in the arena, doubling resident MoE weights. For
    /// Ling-3.0-tiny that is 27.8 GB → 55.6 GB, which overruns Metal's max buffer.
    /// Loaders that can transpose host-side while stacking should set this and
    /// store `experts.gate_up_proj` as `[E, hidden, 2*inter]` and
    /// `experts.down_proj` as `[E, inter, hidden]`.
    pub experts_pretransposed: bool,
    /// Run the routed experts as **MXFP4** (`Op::DequantGroupedMatMulMlx`)
    /// instead of f32 `Op::GroupedMatMul`, at this group size (32).
    ///
    /// The banks are NOT read from the weight map here: the caller declares
    /// `{prefix}.experts.{gate_up,down}_{codes,scales,biases}` by name and
    /// attaches the bytes after compile via `CompiledGraph::set_param_typed`
    /// (see `rlx_ling::streaming`), so the f32 bank never has to exist — which
    /// is the point, at 25.9 GiB for Ling-3.0-tiny.
    ///
    /// Layout is the **untransposed** `[E, N, K]`, i.e. the stock HF
    /// orientation: `gate_up` is `[E, 2*inter, hidden]` and `down` is
    /// `[E, hidden, inter]`. That is the opposite of
    /// [`Self::experts_pretransposed`], which this overrides — the packed op
    /// contracts along the last dim, so it wants exactly what the checkpoint
    /// already has and needs no transpose anywhere.
    pub mxfp4_group: Option<u32>,
}

impl Default for DeepseekMoeDims {
    fn default() -> Self {
        Self {
            hidden: 0,
            moe_inter: 0,
            n_routed: 0,
            top_k: 0,
            n_group: 1,
            topk_group: 1,
            routed_scaling: 1.0,
            shared_inter: 0,
            seq: 0,
            experts_pretransposed: false,
            mxfp4_group: None,
        }
    }
}

/// Declare one layer's packed expert bank params and emit the routed
/// `Op::DequantGroupedMatMulMlx` calls. Bytes are attached post-compile.
struct PackedBank {
    codes: HirNodeId,
    scales: HirNodeId,
    biases: HirNodeId,
}

/// Param names of a packed bank — the contract between the emitter and whoever
/// uploads the bytes. `stem` is `"{prefix}.experts.gate_up"` or `"…down"`.
pub fn packed_bank_keys(stem: &str) -> (String, String, String) {
    (
        format!("{stem}_codes"),
        format!("{stem}_scales"),
        format!("{stem}_biases"),
    )
}

fn declare_packed_bank(
    gb: &mut HirMut,
    stem: &str,
    experts: usize,
    n: usize,
    k: usize,
    gs: usize,
) -> PackedBank {
    let (c_key, s_key, b_key) = packed_bank_keys(stem);
    let n_groups = k / gs;
    PackedBank {
        codes: gb.param(&c_key, Shape::new(&[experts * n * (k / 2)], DType::U8)),
        // Grouped convention: the DECODED float scale, as bf16 (the dense
        // `Op::DequantMatMul` wants raw E8M0 bytes instead — see
        // `rlx_models_core::mxfp4_pack`).
        scales: gb.param(&s_key, Shape::new(&[experts, n, n_groups], DType::BF16)),
        biases: gb.param(&b_key, Shape::new(&[experts, n, n_groups], DType::BF16)),
    }
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
    // MXFP4 declares its banks by name below (no f32 bank in the weight map).
    let dense_banks = if d.mxfp4_group.is_none() {
        Some((
            emit.load_param(&format!("{prefix}.experts.gate_up_proj"), false)?, // [E,2inter,hidden]
            emit.load_param(&format!("{prefix}.experts.down_proj"), false)?,    // [E,hidden,inter]
        ))
    } else {
        None
    };
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

    // Experts stored [E,N,K] → transpose to [E,K,N] for GroupedMatMul. (MXFP4
    // needs no transpose at all — its op contracts along the last dim, so the
    // stock `[E,N,K]` checkpoint order is already right.)
    let dense_t = dense_banks.map(|(gate_up_w, down_w)| {
        if d.experts_pretransposed {
            (gate_up_w, down_w)
        } else {
            (
                gb.transpose_(gate_up_w, vec![0, 2, 1]), // [E, hidden, 2inter]
                gb.transpose_(down_w, vec![0, 2, 1]),    // [E, inter, hidden]
            )
        }
    });
    let packed_banks = d.mxfp4_group.map(|gs| {
        let gs = gs as usize;
        (
            declare_packed_bank(
                &mut gb,
                &format!("{prefix}.experts.gate_up"),
                d.n_routed,
                2 * inter,
                d.hidden,
                gs,
            ),
            declare_packed_bank(
                &mut gb,
                &format!("{prefix}.experts.down"),
                d.n_routed,
                d.hidden,
                inter,
                gs,
            ),
            QuantScheme::MlxMxfp4 {
                group_size: gs as u32,
            },
        )
    });

    let mut acc: Option<HirNodeId> = None;
    for ki in 0..d.top_k {
        let idx_col = gb.narrow_(top_idx, 1, ki, 1);
        let eidx = gb.reshape_(idx_col, vec![rows as i64]);
        let prob_col = gb.narrow_(top_probs, 1, ki, 1);
        let prob = gb.reshape_(prob_col, vec![rows as i64, 1]);

        // `grouped_matmul` derives `[rows, 2·inter]` from the operands, which is
        // also what rejects an expert bank still in `[E, N, K]` order — the one
        // mistake `experts_pretransposed` makes easy to get wrong.
        let gate_up = match (&dense_t, &packed_banks) {
            (Some((gu, _)), _) => gb.grouped_matmul(h2d, *gu, eidx),
            (None, Some((gu, _, scheme))) => gb.add_node(
                Op::DequantGroupedMatMulMlx { scheme: *scheme },
                vec![h2d, gu.codes, gu.scales, gu.biases, eidx],
                Shape::new(&[rows, 2 * inter], f),
            ),
            (None, None) => unreachable!("one of dense/packed is always set"),
        };
        let g = gb.narrow_(gate_up, 1, 0, inter);
        let u = gb.narrow_(gate_up, 1, inter, inter);
        let act = gb.silu(g);
        let hx = gb.mul(act, u);
        let down = match (&dense_t, &packed_banks) {
            (Some((_, dn)), _) => gb.grouped_matmul(hx, *dn, eidx),
            (None, Some((_, dn, scheme))) => gb.add_node(
                Op::DequantGroupedMatMulMlx { scheme: *scheme },
                vec![hx, dn.codes, dn.scales, dn.biases, eidx],
                Shape::new(&[rows, d.hidden], f),
            ),
            (None, None) => unreachable!("one of dense/packed is always set"),
        };
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
