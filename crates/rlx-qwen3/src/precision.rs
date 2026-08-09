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

//! Precision-parametric packed linear emission for the packed decode path.
//!
//! A K-quant / low-precision weight linear lowers to `Op::DequantMatMul`, but
//! the operand layout differs by precision family:
//!   - **GGUF** (`Q4_K`, `Q6_K`, `Q8_0`, MXFP4, IQ*, …): one self-describing
//!     packed blob → 2-input `(x, packed_w)`.
//!   - **affine / MLX / int-block** (`MlxAffine`, `MlxMxfp4`, `Int8Block`, …):
//!     separate code + scale + zero-point/bias → 4-input `(x, w_q, scale, zp)`.
//!   - **NVFP4**: 4-input with FP8 block scales + optional f32 global scale.
//!
//! Rather than hand-write that dispatch per projection × per layer, the
//! `dequant_forms!` macro generates one **exhaustive** `QuantScheme →
//! DequantForm` classifier: adding a new [`QuantScheme`] variant without
//! assigning it a form is a COMPILE error, so no precision is silently left
//! un-lowered on the packed decode path. [`emit_packed_linear`] then emits the
//! right graph op for whatever form a weight carries.

use rlx_ir::quant::QuantScheme;
use rlx_ir::{Graph, NodeId, Shape};

/// How a [`QuantScheme`] maps to a `Op::DequantMatMul` operand layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DequantForm {
    /// GGUF self-describing blob: `(x, packed_w)`.
    Packed2,
    /// Affine / MLX / int-block: `(x, w_q, scale, zp)`.
    Affine4,
    /// NVFP4 block: `(x, w_q, block_scales, global_scale)`.
    Nvfp4,
}

/// Generate the exhaustive `QuantScheme → DequantForm` classifier. Listing a
/// scheme in the wrong bucket is a logic bug, but OMITTING one is a COMPILE
/// error (non-exhaustive match) — which is the point: a new precision can't
/// reach the packed decode path without an explicit form decision.
macro_rules! dequant_forms {
    (
        packed2: [ $($p2:pat_param),* $(,)? ],
        affine4: [ $($a4:pat_param),* $(,)? ],
        nvfp4:   [ $($nv:pat_param),* $(,)? ] $(,)?
    ) => {
        /// Classify a scheme's operand layout. Exhaustive over [`QuantScheme`].
        pub fn dequant_form(scheme: QuantScheme) -> DequantForm {
            match scheme {
                $( $p2 => DequantForm::Packed2, )*
                $( $nv => DequantForm::Nvfp4, )*
                $( $a4 => DequantForm::Affine4, )*
            }
        }
    };
}

dequant_forms! {
    // GGUF family — one packed, self-describing blob (scales embedded).
    packed2: [
        QuantScheme::GgufQ4K, QuantScheme::GgufQ5K, QuantScheme::GgufQ6K,
        QuantScheme::GgufQ8K, QuantScheme::GgufQ2K, QuantScheme::GgufQ3K,
        QuantScheme::GgufQ4_0, QuantScheme::GgufQ4_1, QuantScheme::GgufQ5_0,
        QuantScheme::GgufQ5_1, QuantScheme::GgufQ8_0,
        QuantScheme::GgufIQ4NL, QuantScheme::GgufIQ4XS, QuantScheme::GgufIQ2XXS,
        QuantScheme::GgufIQ2XS, QuantScheme::GgufIQ2S, QuantScheme::GgufIQ3XXS,
        QuantScheme::GgufIQ3S, QuantScheme::GgufIQ1S, QuantScheme::GgufIQ1M,
        QuantScheme::GgufTQ1_0, QuantScheme::GgufTQ2_0, QuantScheme::GgufMXFP4,
        QuantScheme::GgufNVFP4, QuantScheme::GgufQ1_0, QuantScheme::GgufQ2_0,
        QuantScheme::GgufFV5, QuantScheme::GgufFV5B,
    ],
    // Affine / MLX / int-block / FP8 — separate code + scale + zp/bias operands.
    affine4: [
        QuantScheme::Int8Block { .. }, QuantScheme::Int8BlockAsym { .. },
        QuantScheme::Int4Block { .. }, QuantScheme::Fp8E4m3, QuantScheme::Fp8E5m2,
        QuantScheme::MlxAffine { .. }, QuantScheme::MlxMxfp4 { .. },
        QuantScheme::MlxMxfp8 { .. }, QuantScheme::MxFp4x2Block { .. },
    ],
    // NVFP4 block (E2M1 + FP8 block scales + optional f32 global scale).
    nvfp4: [ QuantScheme::Nvfp4Block ],
}

/// Emit a single packed linear `y = x · dequant(w)ᵀ` as the precision-correct
/// `Op::DequantMatMul`. `w_q` is the U8 packed-code param; `scale`/`zp` are
/// the auxiliary params (ignored for [`DequantForm::Packed2`], where the GGUF
/// blob is self-describing). `out_shape` is the result shape `[.., n]`.
pub fn emit_packed_linear(
    g: &mut Graph,
    x: NodeId,
    w_q: NodeId,
    scale: NodeId,
    zp: NodeId,
    scheme: QuantScheme,
    out_shape: Shape,
) -> NodeId {
    match dequant_form(scheme) {
        DequantForm::Packed2 => g.dequant_matmul_packed(x, w_q, scheme, out_shape),
        DequantForm::Nvfp4 => g.dequant_matmul_nvfp4(x, w_q, scale, zp, out_shape),
        DequantForm::Affine4 => g.dequant_matmul(x, w_q, scale, zp, scheme, out_shape),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::{DType, Op};

    // The classifier's operand-count contract must match the graph helpers.
    #[test]
    fn forms_match_operand_counts() {
        assert_eq!(dequant_form(QuantScheme::GgufQ4K), DequantForm::Packed2);
        assert_eq!(dequant_form(QuantScheme::GgufQ8_0), DequantForm::Packed2);
        assert_eq!(dequant_form(QuantScheme::GgufMXFP4), DequantForm::Packed2);
        assert_eq!(
            dequant_form(QuantScheme::MlxAffine {
                bits: 4,
                group_size: 64
            }),
            DequantForm::Affine4
        );
        assert_eq!(dequant_form(QuantScheme::Fp8E4m3), DequantForm::Affine4);
        assert_eq!(dequant_form(QuantScheme::Nvfp4Block), DequantForm::Nvfp4);
    }

    // Each precision emits a DequantMatMul carrying its own scheme, with the
    // operand count its form dictates (2 for GGUF, 4 for affine/nvfp4).
    fn emit_and_check(scheme: QuantScheme, want_inputs: usize) {
        let mut g = Graph::new("packed_linear_test");
        let x = g.input("x", Shape::new(&[1, 1024], DType::F32));
        let w = g.param("w", Shape::new(&[1024], DType::U8));
        let s = g.param("s", Shape::new(&[1, 1], DType::F32));
        let z = g.param("z", Shape::new(&[1, 1], DType::F32));
        let out = emit_packed_linear(
            &mut g,
            x,
            w,
            s,
            z,
            scheme,
            Shape::new(&[1, 512], DType::F32),
        );
        let node = g.node(out);
        assert!(
            matches!(node.op, Op::DequantMatMul { scheme: sc } if sc == scheme),
            "emitted op is not DequantMatMul with the right scheme for {scheme:?}"
        );
        assert_eq!(
            node.inputs.len(),
            want_inputs,
            "operand count for {scheme:?}"
        );
    }

    #[test]
    fn gguf_is_two_input() {
        emit_and_check(QuantScheme::GgufQ4K, 2);
        emit_and_check(QuantScheme::GgufQ6K, 2);
        emit_and_check(QuantScheme::GgufQ8_0, 2);
    }

    #[test]
    fn affine_and_nvfp4_are_four_input() {
        emit_and_check(
            QuantScheme::MlxAffine {
                bits: 4,
                group_size: 64,
            },
            4,
        );
        emit_and_check(QuantScheme::MlxMxfp4 { group_size: 32 }, 4);
        emit_and_check(QuantScheme::Nvfp4Block, 4);
    }
}
