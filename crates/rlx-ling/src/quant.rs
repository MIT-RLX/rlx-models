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

//! Weight precision for the Ling graphs — one knob, threaded through every
//! projection.
//!
//! Ling-3.0-tiny is 7.9 B parameters in an f32 arena: 29.5 GiB, of which the
//! 128-expert MoE banks are 25.9 GiB. That does not fit a 16 GB consumer GPU,
//! and on a machine that *can* hold it, decode is bandwidth-bound at ~2.9 GB of
//! weight traffic per token. [`Quant::Mxfp4`] cuts both by ~8×:
//!
//! ```text
//!                       arena      expert banks   rest
//!   f32                 29.5 GiB   25.9           3.5
//!   MXFP4 experts        6.8 GiB    3.4           3.5
//!   MXFP4 everything     4.0 GiB    3.4           0.6
//! ```
//!
//! The weights are quantized **host-side at build time** by
//! [`rlx_core::mxfp4_pack`] and handed to the graph as typed params, so
//! this works from a stock bf16/f32 HF checkpoint — no pre-quantized MXFP4
//! artifact needed. On CUDA the packed operands then run
//! `Step::DequantGroupedMatmulMlxNative` (register nibble-decode, no host
//! round-trip); elsewhere they take each backend's MXFP4 path.
//!
//! ## What stays f32
//!
//! The **token embedding**. It is a gather, not a matmul, and rlx has no
//! MXFP4 gather — `Op::DequantMatMul` cannot express a row lookup. At
//! 157184×1536 it is 0.97 GiB, the bulk of the 0.6 GiB "rest" figure above not
//! being reached. Norm gammas, biases, `A_log`/`dt_bias` and the router stay
//! f32 too: they are kilobytes, and quantizing a norm gamma is how you lose
//! accuracy for nothing.
//!
//! ## Two scale conventions
//!
//! `Op::DequantMatMul` reads scales as **raw E8M0 bytes** (`U8`);
//! `Op::DequantGroupedMatMulMlx` reads them as the **decoded float** (`BF16`).
//! See the [`rlx_core::mxfp4_pack`] docs — both are handled here so no
//! caller has to know.

use anyhow::Result;
use rlx_core::mxfp4_pack::{GROUP_SIZE, quantize_rows};
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::Op;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

/// Stored precision for the projection weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Quant {
    /// Dense f32 `Op::MatMul` — the reference path.
    #[default]
    F32,
    /// MXFP4 (E2M1 nibbles + per-group E8M0 scale) via `Op::DequantMatMul` /
    /// `Op::DequantGroupedMatMulMlx`.
    Mxfp4 { group_size: u32 },
}

impl Quant {
    /// MXFP4 at the standard group size (32 — the only one with E8M0 scales;
    /// MLX switches to FP8 E4M3 scales at 16).
    pub const MXFP4: Self = Self::Mxfp4 {
        group_size: GROUP_SIZE as u32,
    };

    /// The MXFP4 group size along the contraction dim, or `None` when f32.
    pub fn group_size(self) -> Option<usize> {
        match self {
            Self::F32 => None,
            Self::Mxfp4 { group_size } => Some(group_size as usize),
        }
    }

    /// The IR quantization scheme the packed ops take, or `None` when f32.
    pub fn scheme(self) -> Option<QuantScheme> {
        match self {
            Self::F32 => None,
            Self::Mxfp4 { group_size } => Some(QuantScheme::MlxMxfp4 { group_size }),
        }
    }

    /// A contraction dim MXFP4 can group evenly. Ling's are all multiples of 32
    /// (hidden 1536, moe_inter 512, kv_lora 512, q_lora 256), but a projection
    /// with an odd `in_features` silently falls back rather than panicking deep
    /// inside a kernel.
    pub fn fits(self, k: usize) -> bool {
        match self.group_size() {
            None => false,
            Some(gs) => k.is_multiple_of(gs) && k.is_multiple_of(2),
        }
    }
}

/// Per-tensor-class precision plan.
///
/// The LM head is split out from the rest because its error does **not** get
/// diluted. Every other projection feeds a residual stream, so a 4-bit weight
/// perturbs a hidden state that is then re-normalized and added to a much larger
/// carry. The LM head's output *is* the logits. Measured on the 4-layer
/// synthetic Ling (`tests/mxfp4_model.rs`), max relative deviation from f32:
///
/// ```text
///   body MXFP4, f32 head    1.9e-3
///   body + head MXFP4       3.1e-2      ← 16x, all of it from the head
/// ```
///
/// That matters because of what [`crate::flow_decode`] already recorded for the
/// f16 head: at 5e-4 relative error, 157k near-tied logits flipped argmax often
/// enough to derail greedy generation ("The capital of Germany is Berlin…" →
/// "Is the capital of France the same capital as…"). MXFP4 is ~3.3 mantissa bits
/// against f16's 10, so [`Self::mxfp4_all`] is the aggressive setting and
/// [`Self::mxfp4_body`] the one to reach for when output quality matters more
/// than the head's 0.85 GB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QuantPlan {
    /// Attention, MLP and MoE expert projections.
    pub proj: Quant,
    /// `lm_head` only — see the accuracy note above before setting this to MXFP4.
    pub lm_head: Quant,
}

impl QuantPlan {
    /// Everything f32 — the reference.
    pub const F32: Self = Self {
        proj: Quant::F32,
        lm_head: Quant::F32,
    };

    /// MXFP4 everywhere, including the LM head. Smallest arena (~3.2 GiB of
    /// weights); see the accuracy note above.
    pub const fn mxfp4_all() -> Self {
        Self {
            proj: Quant::MXFP4,
            lm_head: Quant::MXFP4,
        }
    }

    /// MXFP4 for the body, f32 LM head (+0.85 GiB, 16x less logit error).
    pub const fn mxfp4_body() -> Self {
        Self {
            proj: Quant::MXFP4,
            lm_head: Quant::F32,
        }
    }
}

impl From<Quant> for QuantPlan {
    fn from(q: Quant) -> Self {
        Self {
            proj: q,
            lm_head: q,
        }
    }
}

/// `y = x @ Wᵀ` for a stock HF `[out, in]` weight, at the requested precision.
///
/// Shape-agnostic in `x`: everything but the last dim is folded into the matmul
/// row count and restored afterwards, so `[1, seq, in]` and `[rows, in]` both
/// work.
pub fn linear(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId, q: Quant) -> Result<HirNodeId> {
    let key = format!("{prefix}.weight");
    let Some(scheme) = q.scheme() else {
        let w = emit.load_param(&key, true)?;
        let mut gb = HirMut::new(emit.hir());
        return Ok(gb.mm(x, w));
    };

    // `[out, in]` — the packer's row-major orientation, so do NOT let the
    // loader transpose (the f32 path wants `[in, out]` for `mm`; MXFP4 wants
    // the original and contracts along the last dim).
    let (data, shape) = emit.weights.take(&key, false)?;
    anyhow::ensure!(
        shape.len() == 2,
        "{key}: expected a 2-D weight, got {shape:?}"
    );
    let (n, k) = (shape[0], shape[1]);
    if !q.fits(k) {
        // Fall back to dense rather than failing the build — transpose here,
        // not in-graph, so constant folding cannot leave two copies resident.
        let mut t = vec![0f32; n * k];
        for r in 0..n {
            for c in 0..k {
                t[c * n + r] = data[r * k + c];
            }
        }
        drop(data);
        let id = emit.synth_param(&key, t, Shape::new(&[k, n], DType::F32));
        let mut gb = HirMut::new(emit.hir());
        return Ok(gb.mm(x, id));
    }

    let gs = q.group_size().expect("mxfp4 has a group size");
    let packed = quantize_rows(&data, n, k, gs);
    drop(data);
    let n_groups = k / gs;

    let c_key = format!("{key}.codes");
    let s_key = format!("{key}.scales");
    let b_key = format!("{key}.biases");
    let (c_id, s_id, b_id) = {
        let h = emit.hir();
        (
            h.param(&c_key, Shape::new(&[packed.codes.len()], DType::U8)),
            // Dense convention: RAW E8M0 bytes.
            h.param(&s_key, Shape::new(&[n, n_groups], DType::U8)),
            h.param(&b_key, Shape::new(&[n, n_groups], DType::U8)),
        )
    };
    emit.state
        .typed_params
        .push((s_key, packed.scales_e8m0().to_vec(), DType::U8));
    emit.state
        .typed_params
        .push((b_key, packed.zero_biases_u8(), DType::U8));
    emit.state
        .typed_params
        .push((c_key, packed.codes, DType::U8));

    let dims: Vec<usize> = emit
        .hir()
        .node(x)
        .shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let mut gb = HirMut::new(emit.hir());
    let x2 = gb.reshape_(x, vec![rows as i64, k as i64]);
    let y = gb.add_node(
        Op::DequantMatMul { scheme },
        vec![x2, c_id, s_id, b_id],
        Shape::new(&[rows, n], DType::F32),
    );
    let mut out_dims: Vec<i64> = dims[..dims.len() - 1].iter().map(|&d| d as i64).collect();
    out_dims.push(n as i64);
    Ok(gb.reshape_(y, out_dims))
}
