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

//! Model-specific bundle graph patches before `rlx-onnx-import` lowering.

use std::cell::Cell;

use rlx_ir::Shape;
use rlx_onnx_import::BundleNode;

thread_local! {
    static IMPORT_SEQUENCE_LENGTH: Cell<usize> = const { Cell::new(128) };
}

pub fn set_import_sequence_length(seq: usize) {
    IMPORT_SEQUENCE_LENGTH.with(|c| c.set(seq));
}

pub fn import_output_shape_fix(name: &str, shape: &Shape) -> Option<Shape> {
    let seq = IMPORT_SEQUENCE_LENGTH.with(|c| c.get());
    output_shape_fix(name, shape, seq)
}

/// Apply Kitten TTS patches (duration carry, decoder sine-gen shapes, BERT mask).
pub fn patch_bundle_nodes(nodes: &mut [BundleNode], sequence_length: usize) {
    crate::bundle_compile::rewrite_duration_carry(nodes);
    patch_l_sin_gen_shapes(nodes, sequence_length);
    patch_bert_attention_mask_shapes(nodes, sequence_length);
}

fn patch_l_sin_gen_shapes(nodes: &mut [BundleNode], sequence_length: usize) {
    // Decoder sine-gen tensors: ONNX shape inference leaves `[1,2,seq]`; force `[1,300,seq]`.
    let last = sequence_length;
    let meta = serde_json::json!({
        "shape": [1, 300, last],
        "dtype": "f32",
    });
    for node in nodes.iter_mut() {
        if !node.name.contains("l_sin_gen") && !node.name.contains("/decoder/generator/m_source/") {
            continue;
        }
        if node.output_meta.is_empty() {
            node.output_meta.push(meta.clone());
        } else {
            for slot in &mut node.output_meta {
                *slot = meta.clone();
            }
        }
    }
}

/// Lower-time shape fix for decoder sine-generator tensors (bad ONNX metadata).
pub fn output_shape_fix(node_name: &str, shape: &Shape, sequence_length: usize) -> Option<Shape> {
    if !node_name.contains("l_sin_gen") && !node_name.contains("/decoder/generator/m_source/") {
        return None;
    }
    let rank = shape.rank();
    if rank == 2
        || (rank == 3
            && shape.dim(0).unwrap_static() == 1
            && shape.dim(1).unwrap_static() == 2
            && shape.dim(2).unwrap_static() == 128)
        || (rank == 3 && shape.dim(2).unwrap_static() == 9)
    {
        Some(Shape::new(&[1, 300, sequence_length], shape.dtype()))
    } else {
        None
    }
}

fn patch_bert_attention_mask_shapes(nodes: &mut [BundleNode], sequence_length: usize) {
    let seq = sequence_length as u64;
    for node in nodes.iter_mut() {
        if node.name != "/bert/Expand_1" && node.name != "/bert/Where_2" {
            continue;
        }
        let meta = serde_json::json!({
            "shape": [1, 1, seq, seq],
            "dtype": "f32",
        });
        if node.output_meta.is_empty() {
            node.output_meta.push(meta);
        } else {
            node.output_meta[0] = meta;
        }
    }
}
