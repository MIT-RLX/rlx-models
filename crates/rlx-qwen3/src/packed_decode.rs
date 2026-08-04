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

//! Packed-weights decode via **graph rewrite**.
//!
//! Rather than hand-write a second decode graph, we build the validated F32
//! decode graph through the existing flow, then rewrite its weight-`MatMul`s
//! into packed [`Op::DequantMatMul`]s — reusing every RoPE / QK-norm / GQA /
//! KV-cache / SwiGLU detail the flow already gets right, and touching only the
//! linears. Precision dispatch reuses [`crate::precision::dequant_form`], so any
//! [`QuantScheme`] the classifier covers is handled here (GGUF → 2-input,
//! affine/MLX/FP8 → 4-input with synthesized scale/zp params).
//!
//! The rewrite re-declares each matched weight param as U8 packed bytes and
//! returns the rewritten weight keys; the caller drops those from the f32 param
//! map and binds the packed bytes via `set_param_typed` (as the packed prefill
//! path already does).

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use crate::precision::{DequantForm, dequant_form};

/// Everything the rewrite needs to re-declare a weight as packed.
#[derive(Clone, Copy, Debug)]
pub struct PackedWeightInfo {
    pub scheme: QuantScheme,
    /// Packed byte length (the U8 weight param's new shape).
    pub nbytes: usize,
    /// Output features (rows) — for affine/nvfp4 scale/zp param shapes.
    pub n: usize,
    /// Groups per row (affine scale/zp columns); ignored for GGUF.
    pub n_groups: usize,
}

/// Rewrite F32 weight-`MatMul`s into packed `DequantMatMul`s **in place**. For
/// each `MatMul(x, Param{name})` where `lookup(name)` returns packed info, swap
/// the op to `DequantMatMul` (2-input for GGUF, 4-input for affine/nvfp4, adding
/// `{name}.scales`/`{name}.biases` params) and re-declare the weight param as U8.
/// Returns the rewritten weight keys (drop these from the f32 params + bind the
/// packed bytes as U8).
pub fn rewrite_matmuls_to_packed(
    g: &mut Graph,
    lookup: &dyn Fn(&str) -> Option<PackedWeightInfo>,
) -> Vec<String> {
    // Phase 1 (immutable scan): collect (matmul, x, weight_param, name, info).
    let mut todo: Vec<(NodeId, NodeId, NodeId, String, PackedWeightInfo)> = Vec::new();
    for n in g.nodes() {
        if !matches!(n.op, Op::MatMul) || n.inputs.len() != 2 {
            continue;
        }
        let (x_id, w_id) = (n.inputs[0], n.inputs[1]);
        if let Op::Param { name } = &g.node(w_id).op {
            if let Some(info) = lookup(name) {
                todo.push((n.id, x_id, w_id, name.clone(), info));
            }
        }
    }
    // Phase 2 (mutate): re-declare the weight U8 + retarget the matmul op/inputs.
    let mut done = Vec::with_capacity(todo.len());
    for (mm_id, x_id, w_id, name, info) in todo {
        g.node_mut(w_id).shape = Shape::new(&[info.nbytes], DType::U8);
        match dequant_form(info.scheme) {
            DequantForm::Packed2 => {
                g.node_mut(mm_id).op = Op::DequantMatMul {
                    scheme: info.scheme,
                };
                g.set_inputs(mm_id, vec![x_id, w_id]);
            }
            DequantForm::Affine4 | DequantForm::Nvfp4 => {
                let cols = info.n_groups.max(1);
                let scale = g.param(
                    format!("{name}.scales"),
                    Shape::new(&[info.n, cols], DType::F32),
                );
                let zp = g.param(
                    format!("{name}.biases"),
                    Shape::new(&[info.n, cols], DType::F32),
                );
                g.node_mut(mm_id).op = Op::DequantMatMul {
                    scheme: info.scheme,
                };
                g.set_inputs(mm_id, vec![x_id, w_id, scale, zp]);
            }
        }
        done.push(name);
    }
    done
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_linear(name: &str) -> (Graph, NodeId, NodeId, NodeId) {
        let mut g = Graph::new("linear");
        let x = g.input("x", Shape::new(&[1, 1024], DType::F32));
        let w = g.param(name, Shape::new(&[1024, 512], DType::F32));
        let mm = g.matmul(x, w, Shape::new(&[1, 512], DType::F32));
        (g, x, w, mm)
    }

    #[test]
    fn rewrites_gguf_matmul_to_2input_packed() {
        let (mut g, x, w, mm) = build_linear("model.layers.0.self_attn.q_proj.weight");
        let done = rewrite_matmuls_to_packed(&mut g, &|name| {
            name.ends_with("q_proj.weight").then_some(PackedWeightInfo {
                scheme: QuantScheme::GgufQ4K,
                nbytes: 1024 * 512 * 144 / 256,
                n: 512,
                n_groups: 0,
            })
        });
        assert_eq!(
            done,
            vec!["model.layers.0.self_attn.q_proj.weight".to_string()]
        );
        let node = g.node(mm);
        assert!(matches!(
            node.op,
            Op::DequantMatMul {
                scheme: QuantScheme::GgufQ4K
            }
        ));
        assert_eq!(node.inputs, vec![x, w]);
        assert_eq!(g.node(w).shape.dtype(), DType::U8);
    }

    #[test]
    fn rewrites_affine_to_4input_with_scale_zp() {
        let (mut g, x, w, mm) = build_linear("ffn.gate.weight");
        let done = rewrite_matmuls_to_packed(&mut g, &|_| {
            Some(PackedWeightInfo {
                scheme: QuantScheme::MlxAffine {
                    bits: 4,
                    group_size: 64,
                },
                nbytes: 1024 * 512 / 2,
                n: 512,
                n_groups: 16,
            })
        });
        assert_eq!(done.len(), 1);
        let node = g.node(mm);
        assert!(matches!(
            node.op,
            Op::DequantMatMul {
                scheme: QuantScheme::MlxAffine { .. }
            }
        ));
        // 4-input: x, w, scale, zp — and the two aux params were added.
        assert_eq!(node.inputs.len(), 4);
        assert_eq!(node.inputs[0], x);
        assert_eq!(node.inputs[1], w);
    }

    #[test]
    fn leaves_unlisted_weights_alone() {
        let (mut g, _x, _w, mm) = build_linear("embed_tokens.weight");
        let done = rewrite_matmuls_to_packed(&mut g, &|_| None);
        assert!(done.is_empty());
        assert!(matches!(g.node(mm).op, Op::MatMul));
    }
}
