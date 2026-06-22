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
use std::collections::HashMap;

use rlx_ir::hir::{HirGraphExt, HirModule, HirMut, HirNodeId, HirOp};
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Op, Shape};
use rlx_onnx_import::BundleNode;

use crate::kernels::{
    ALIGNMENT_SCATTER_INDICES, CONCAT_FROM_SEQUENCE, CONCAT_FROM_SEQUENCE_ONNX, F0_IF_SELECT,
};
use crate::mel_align;
use crate::opts::{ALIGNMENT_FRAME_COUNT, CONCAT_SEQUENCE_STUB, RANGE_2_STUB};

const HARMONICS: usize = 9;
const MEL_DIV: usize = 300;
const SHAPE8_STUB: &str = "__stub__//Shape_8_output_0";
const IF_OUTPUT_STUB: &str = "__stub__//If_output_0";
const IF_1_OUTPUT_STUB: &str = "__stub__//If_1_output_0";
const SLICE_3_NODE: &str = "/decoder/generator/Slice_3";
const VOCODER_WAVEFORM_SLICE: &str = "onnx.VocoderWaveformSlice";
const EXPAND_I64_ALIGN: &str = "onnx.ExpandI64Align";

/// Vocoder hop: waveform samples per alignment frame (matches ONNX mini 0.8).
pub const SAMPLES_PER_ALIGNMENT_FRAME: usize = 600;

fn max_alignment_frames(sequence_length: usize, max_waveform_samples: usize) -> usize {
    mel_align::compile_mel_cap(
        sequence_length,
        max_waveform_samples,
        crate::bundle_compile::MAX_FRAMES_PER_TOKEN,
    )
}

fn repurpose_as_align_param(hir: &mut HirModule, node_id: HirNodeId) {
    let node = hir.node_mut(node_id);
    node.op = HirOp::Mir(Op::Param {
        name: ALIGNMENT_FRAME_COUNT.to_string(),
    });
    node.inputs.clear();
    node.shape = Shape::new(&[1], DType::I64);
}

fn repurpose_as_alignment_range(
    hir: &mut HirModule,
    node_id: HirNodeId,
    align_id: HirNodeId,
    max_frames: usize,
) {
    let node = hir.node_mut(node_id);
    node.op = HirOp::Mir(Op::Custom {
        name: "onnx.AlignmentRange".to_string(),
        num_inputs: 1,
        attrs: vec![],
    });
    node.inputs = vec![align_id];
    node.shape = Shape::new(&[max_frames], DType::I64);
}

fn find_concat_output(hir: &HirModule) -> Option<HirNodeId> {
    for idx in 0..hir.len() {
        let id = HirNodeId(idx as u32);
        let node = hir.node(id);
        if let HirOp::Mir(Op::Param { name }) = &node.op {
            if name == CONCAT_SEQUENCE_STUB {
                return Some(id);
            }
        }
        if let HirOp::Mir(Op::Custom { name, .. }) = &node.op {
            if name == CONCAT_FROM_SEQUENCE || name == CONCAT_FROM_SEQUENCE_ONNX {
                return Some(id);
            }
        }
    }
    None
}

fn param_stub_name(op: &HirOp) -> Option<&str> {
    match op {
        HirOp::Param { name } => Some(name.as_str()),
        HirOp::Mir(Op::Param { name }) => Some(name.as_str()),
        _ => None,
    }
}

fn find_if_output_stub(hir: &HirModule, stub: &str) -> Option<HirNodeId> {
    for idx in 0..hir.len() {
        let id = HirNodeId(idx as u32);
        if param_stub_name(&hir.node(id).op) == Some(stub) {
            return Some(id);
        }
    }
    None
}

fn find_node_by_name(hir: &HirModule, name: &str) -> Option<HirNodeId> {
    for idx in 0..hir.len() {
        let id = HirNodeId(idx as u32);
        if hir.node(id).name.as_deref() == Some(name) {
            return Some(id);
        }
    }
    None
}

fn rewire_hir_inputs(hir: &mut HirModule, from: HirNodeId, to: HirNodeId) {
    for idx in 0..hir.len() {
        let id = HirNodeId(idx as u32);
        let node = hir.node_mut(id);
        for inp in &mut node.inputs {
            if *inp == from {
                *inp = to;
            }
        }
    }
}

fn find_shape_of_tensor(hir: &HirModule, tensor: HirNodeId) -> Option<HirNodeId> {
    if let Some(id) = find_node_by_name(hir, "/Shape_8") {
        let node = hir.node(id);
        if node.inputs.is_empty() || node.inputs.first() == Some(&tensor) {
            return Some(id);
        }
    }
    for idx in 0..hir.len() {
        let id = HirNodeId(idx as u32);
        if let HirOp::Mir(Op::Param { name }) = &hir.node(id).op {
            if name == SHAPE8_STUB {
                return Some(id);
            }
        }
    }
    None
}

fn find_gather_5(hir: &HirModule) -> Option<HirNodeId> {
    for idx in 0..hir.len() {
        let id = HirNodeId(idx as u32);
        let node = hir.node(id);
        if let HirOp::Mir(Op::Gather { axis: 0 }) = &node.op {
            if node.shape.dtype() != DType::I64 {
                continue;
            }
            let Some(indices) = node.inputs.get(1) else {
                continue;
            };
            let idx_node = hir.node(*indices);
            if idx_node.name.as_deref() == Some("onnx::Range_4162") {
                return Some(id);
            }
            if let HirOp::Mir(Op::Param { name }) = &idx_node.op {
                if name == "onnx::Range_4162" {
                    return Some(id);
                }
            }
        }
    }
    None
}

fn find_gather_axis0_from(hir: &HirModule, source: HirNodeId) -> Option<HirNodeId> {
    for idx in 0..hir.len() {
        let id = HirNodeId(idx as u32);
        let node = hir.node(id);
        if let HirOp::Mir(Op::Gather { axis: 0 }) = &node.op {
            if node.inputs.first() == Some(&source) {
                return Some(id);
            }
        }
    }
    None
}

fn find_alignment_range_input(hir: &HirModule, concat_id: HirNodeId) -> Option<HirNodeId> {
    if let Some(id) = find_node_by_name(hir, "/Range_2") {
        return Some(id);
    }
    for idx in 0..hir.len() {
        let id = HirNodeId(idx as u32);
        if let HirOp::Mir(Op::Param { name }) = &hir.node(id).op {
            if name == RANGE_2_STUB {
                return Some(id);
            }
        }
    }
    for idx in 0..hir.len() {
        let id = HirNodeId(idx as u32);
        let node = hir.node(id);
        if let HirOp::Mir(Op::Binary(BinaryOp::Add)) = &node.op {
            if node.inputs.first() == Some(&concat_id) {
                return node.inputs.get(1).copied();
            }
        }
    }
    None
}

fn inject_mel_shape9_dynamic(hir: &mut HirModule, align_id: HirNodeId) -> bool {
    let Some(shape9_id) = find_node_by_name(hir, "/Shape_9") else {
        return false;
    };
    rewire_hir_inputs(hir, shape9_id, align_id);
    true
}

fn inject_alignment_scatter_indices(
    hir: &mut HirModule,
    concat_seq_id: HirNodeId,
    align_id: HirNodeId,
    max_frames: usize,
) -> bool {
    let Some(concat4_id) = find_node_by_name(hir, "/Concat_4") else {
        return false;
    };
    let node = hir.node_mut(concat4_id);
    node.op = HirOp::Mir(Op::Custom {
        name: ALIGNMENT_SCATTER_INDICES.to_string(),
        num_inputs: 2,
        attrs: vec![],
    });
    node.inputs = vec![concat_seq_id, align_id];
    node.shape = Shape::new(&[max_frames, 2], DType::I64);
    true
}

fn inject_expand3_alignment_shape(
    hir: &mut HirModule,
    concat_id: HirNodeId,
    align_id: HirNodeId,
) -> bool {
    let Some(expand_id) = find_node_by_name(hir, "/Expand_3") else {
        return false;
    };
    let out_shape = hir.node(concat_id).shape.clone();
    let node = hir.node_mut(expand_id);
    node.op = HirOp::Mir(Op::Custom {
        name: EXPAND_I64_ALIGN.to_string(),
        num_inputs: 2,
        attrs: vec![],
    });
    node.inputs = vec![concat_id, align_id];
    node.shape = out_shape;
    true
}

fn patch_alignment_expand_shapes(hir: &mut HirModule, concat_id: HirNodeId, max_frames: usize) {
    let frame_shape = Shape::new(&[max_frames], DType::I64);
    for idx in 0..hir.len() {
        let id = HirNodeId(idx as u32);
        let node = hir.node(id);
        let touches_alignment = node.inputs.iter().any(|inp| {
            *inp == concat_id
                || matches!(
                    hir.node(*inp).op,
                    HirOp::Mir(Op::Param { ref name }) if name == RANGE_2_STUB
                )
        });
        if !touches_alignment {
            continue;
        }
        match &node.op {
            HirOp::Mir(Op::Expand { .. }) | HirOp::Mir(Op::Binary(BinaryOp::Add))
                if node.shape.dtype() == DType::I64 =>
            {
                hir.node_mut(id).shape = frame_shape.clone();
            }
            _ => {}
        }
    }
}

fn find_f0_proj_cast(hir: &HirModule) -> Option<HirNodeId> {
    find_node_by_name(hir, "/F0_proj/Conv_output_0_Cast_to_float32_0")
        .or_else(|| find_node_by_name(hir, "/F0_proj/Conv_output_0_Cast_to_float32_output_0"))
        .or_else(|| find_node_by_name(hir, "/F0_proj/Conv_output_0"))
        .or_else(|| find_f0_decoder_input(hir))
}

fn find_n_proj_cast(hir: &HirModule) -> Option<HirNodeId> {
    find_node_by_name(hir, "/N_proj/Conv_output_0_Cast_to_float32_0")
        .or_else(|| find_node_by_name(hir, "/N_proj/Conv_output_0_Cast_to_float32_output_0"))
        .or_else(|| find_node_by_name(hir, "/N_proj/Conv_output_0"))
}

fn find_f0_decoder_input(hir: &HirModule) -> Option<HirNodeId> {
    let mut best: Option<(u32, HirNodeId)> = None;
    for idx in 0..hir.len() {
        let id = HirNodeId(idx as u32);
        let node = hir.node(id);
        let HirOp::Mir(Op::Cast { to }) = &node.op else {
            continue;
        };
        if *to != DType::F32 || node.shape.rank() != 3 {
            continue;
        }
        if node.shape.dim(0).unwrap_static() != 1 {
            continue;
        }
        let channels = node.shape.dim(1).unwrap_static();
        if channels == 0 || channels > 32 {
            continue;
        }
        best = Some((idx as u32, id));
    }
    best.map(|(_, id)| id)
}

pub(crate) fn inject_if_proj_bypass(hir: &mut HirModule, stub: &str, proj_id: HirNodeId) -> bool {
    let Some(if_id) = find_if_output_stub(hir, stub) else {
        return false;
    };
    let align_id = ensure_alignment_frame_param(hir);
    let stub_shape = hir.node(if_id).shape.clone();
    let node = hir.node_mut(if_id);
    node.op = HirOp::Mir(Op::Custom {
        name: F0_IF_SELECT.to_string(),
        num_inputs: 2,
        attrs: vec![],
    });
    node.inputs = vec![proj_id, align_id];
    // Keep `lower_if_stub` output shape — runtime `F0IfSelect` trims/pads inside the buffer.
    node.shape = stub_shape;
    true
}

pub(crate) fn inject_if_f0_bypass(hir: &mut HirModule) -> bool {
    let Some(f0_id) = find_f0_proj_cast(hir) else {
        return false;
    };
    inject_if_proj_bypass(hir, IF_OUTPUT_STUB, f0_id)
}

pub(crate) fn inject_if_n_bypass(hir: &mut HirModule) -> bool {
    let Some(n_id) = find_n_proj_cast(hir) else {
        return false;
    };
    inject_if_proj_bypass(hir, IF_1_OUTPUT_STUB, n_id)
}

fn ensure_alignment_frame_param(hir: &mut HirModule) -> HirNodeId {
    for idx in 0..hir.len() {
        let id = HirNodeId(idx as u32);
        if param_stub_name(&hir.node(id).op) == Some(ALIGNMENT_FRAME_COUNT) {
            return id;
        }
    }
    if let Some(id) = find_node_by_name(hir, "/Shape_8") {
        repurpose_as_align_param(hir, id);
        return id;
    }
    for idx in 0..hir.len() {
        let id = HirNodeId(idx as u32);
        if param_stub_name(&hir.node(id).op) == Some(SHAPE8_STUB) {
            repurpose_as_align_param(hir, id);
            return id;
        }
    }
    let mut m = HirMut::new(hir);
    m.param(ALIGNMENT_FRAME_COUNT, Shape::new(&[1], DType::I64))
}

/// Narrow-seq bundle import: replace static `/decoder/generator/Slice_3` with alignment-driven trim.
fn inject_vocoder_waveform_slice(hir: &mut HirModule, max_waveform_samples: usize) -> bool {
    let Some(slice_id) = find_node_by_name(hir, SLICE_3_NODE) else {
        return false;
    };
    if let HirOp::Mir(Op::Custom { name, .. }) = &hir.node(slice_id).op {
        if name == VOCODER_WAVEFORM_SLICE {
            return true;
        }
    }
    let Some(input) = hir.node(slice_id).inputs.first().copied() else {
        return false;
    };
    let wave_input = {
        let mut m = HirMut::new(hir);
        let mut x = input;
        let s = m.shape(x).clone();
        if s.rank() == 4 && s.dim(1).unwrap_static() == 1 {
            x = m.reshape_(
                x,
                vec![
                    s.dim(0).unwrap_static() as i64,
                    s.dim(2).unwrap_static() as i64,
                    s.dim(3).unwrap_static() as i64,
                ],
            );
        }
        let s = m.shape(x).clone();
        if s.rank() == 3 && s.dim(1).unwrap_static() > s.dim(2).unwrap_static() {
            x = m.transpose_(x, vec![0, 2, 1]);
        }
        x
    };
    let align_id = ensure_alignment_frame_param(hir);
    let x_shape = hir.node(wave_input).shape.clone();
    let max_time = max_waveform_samples.max(1);
    let out_shape = if x_shape.rank() == 3 {
        Shape::new(
            &[
                x_shape.dim(0).unwrap_static(),
                x_shape.dim(1).unwrap_static(),
                max_time,
            ],
            x_shape.dtype(),
        )
    } else {
        Shape::new(&[max_time], x_shape.dtype())
    };
    let node = hir.node_mut(slice_id);
    node.op = HirOp::Mir(Op::Custom {
        name: VOCODER_WAVEFORM_SLICE.to_string(),
        num_inputs: 2,
        attrs: vec![],
    });
    node.inputs = vec![wave_input, align_id];
    node.shape = out_shape;
    true
}

/// Patch vocoder alignment for bundle import and weights-only stubs.
pub fn inject_vocoder_dynamic_alignment(
    hir: &mut HirModule,
    sequence_length: usize,
    max_waveform_samples: usize,
) -> bool {
    let max_frames = max_alignment_frames(sequence_length, max_waveform_samples);

    // Weights-only `graph.rs` path (import stubs).
    let mut shape8_stub = None;
    let mut range2_stub = None;
    for idx in 0..hir.len() {
        let id = HirNodeId(idx as u32);
        if let HirOp::Mir(Op::Param { name }) = &hir.node(id).op {
            match name.as_str() {
                SHAPE8_STUB => shape8_stub = Some(id),
                RANGE_2_STUB => range2_stub = Some(id),
                CONCAT_SEQUENCE_STUB => {
                    hir.node_mut(id).shape = Shape::new(&[max_frames], DType::I64);
                }
                _ => {}
            }
        }
    }
    let mut patched = false;
    if let (Some(shape8_id), Some(range2_stub_id)) = (shape8_stub, range2_stub) {
        // Weights-only graph: repurpose stubs in-place (never append nodes — breaks HIR lower order).
        if find_concat_output(hir).is_none() {
            repurpose_as_align_param(hir, shape8_id);
            repurpose_as_alignment_range(hir, range2_stub_id, shape8_id, max_frames);
            patched = true;
        }
    }

    // Bundle import: mel length must follow runtime `ALIGNMENT_FRAME_COUNT`, not constant-folded Shape ops.
    if let Some(concat_id) = find_concat_output(hir) {
        let align_id = ensure_alignment_frame_param(hir);
        if let Some(range2_id) = find_alignment_range_input(hir, concat_id) {
            repurpose_as_alignment_range(hir, range2_id, align_id, max_frames);
            patched = true;
        }
        if inject_alignment_scatter_indices(hir, concat_id, align_id, max_frames) {
            patched = true;
        }
        if let Some(shape8_id) = find_shape_of_tensor(hir, concat_id) {
            if let Some(gather_id) =
                find_gather_axis0_from(hir, shape8_id).or_else(|| find_gather_5(hir))
            {
                hir.node_mut(gather_id).inputs[0] = align_id;
                patched = true;
            }
        }
        if inject_mel_shape9_dynamic(hir, align_id) {
            patched = true;
        }
        patch_alignment_expand_shapes(hir, concat_id, max_frames);
        if inject_expand3_alignment_shape(hir, concat_id, align_id) {
            patched = true;
        }
        if crate::compile_profile::env_flag("KITTEN_RLX_DEBUG_DURATION") {
            eprintln!("[kitten] bundle mel-align inject patched={patched}");
        }
    }

    if inject_if_f0_bypass(hir) {
        patched = true;
    }
    if inject_if_n_bypass(hir) {
        patched = true;
    }
    if sequence_length < 32
        && crate::compile_profile::env_flag("KITTEN_RLX_ENABLE_NARROW_WAVEFORM_SLICE")
        && inject_vocoder_waveform_slice(hir, max_waveform_samples)
    {
        patched = true;
    }
    patched
}

thread_local! {
    static IMPORT_SEQUENCE_LENGTH: Cell<usize> = const { Cell::new(128) };
    static IMPORT_MAX_WAVEFORM: Cell<usize> = const { Cell::new(48_000) };
}

pub fn set_import_sequence_length(seq: usize) {
    IMPORT_SEQUENCE_LENGTH.with(|c| c.set(seq));
}

pub fn set_import_max_waveform_samples(samples: usize) {
    IMPORT_MAX_WAVEFORM.with(|c| c.set(samples));
}

pub(crate) fn import_sequence_length() -> usize {
    IMPORT_SEQUENCE_LENGTH.with(|c| c.get())
}

pub(crate) fn import_max_waveform_samples() -> usize {
    IMPORT_MAX_WAVEFORM.with(|c| c.get())
}

fn frame_cap(max_wave: usize) -> usize {
    max_wave.div_ceil(MEL_DIV).max(1)
}

fn vocoder_import_shape(
    node_name: &str,
    max_wave: usize,
    sequence_length: usize,
) -> Option<Vec<usize>> {
    if sequence_length < 32 {
        narrow_wave_vocoder_shape(node_name, max_wave)
            .or_else(|| explicit_vocoder_shape(node_name, max_wave))
    } else {
        explicit_vocoder_shape(node_name, max_wave)
    }
}

pub fn import_output_shape_fix(name: &str, shape: &Shape) -> Option<Shape> {
    let max_wave = import_max_waveform_samples();
    let seq = import_sequence_length();
    if let Some(target) = vocoder_import_shape(name, max_wave, seq) {
        let cur: Vec<usize> = shape.dims().iter().map(|d| d.unwrap_static()).collect();
        if cur == target {
            return None;
        }
        return Some(Shape::new(&target, shape.dtype()));
    }
    output_shape_fix(name, shape, seq)
}

/// Apply Kitten TTS patches (duration carry, decoder vocoder shapes, BERT mask).
pub fn patch_bundle_nodes(
    nodes: &mut [BundleNode],
    sequence_length: usize,
    max_waveform_samples: usize,
) {
    set_import_max_waveform_samples(max_waveform_samples);
    set_import_sequence_length(sequence_length);
    crate::bundle_compile::rewrite_duration_carry(nodes);
    patch_duration_where_input(nodes);
    patch_split_duration_mask_use_carry(nodes, sequence_length);
    patch_l_sin_gen_shapes(nodes, sequence_length);
    patch_explicit_vocoder_shapes(nodes, max_waveform_samples, sequence_length);
    patch_bert_attention_mask_shapes(nodes, sequence_length);
    mel_align::patch_alignment_mask(
        nodes,
        sequence_length,
        max_waveform_samples,
        crate::bundle_compile::MAX_FRAMES_PER_TOKEN,
        sequence_length < 32,
    );
}

/// Apply the same wide-sequence shape fixes to weights-only HIR (`graph.rs` path).
pub fn apply_hir_patches(hir: &mut HirModule, sequence_length: usize, max_waveform_samples: usize) {
    set_import_max_waveform_samples(max_waveform_samples);
    set_import_sequence_length(sequence_length);
    if sequence_length < 32 {
        return;
    }
    let seq = sequence_length;
    let l_sin_patched = [1usize, 300, seq];
    let matmul1: HashMap<&str, Vec<usize>> = HashMap::from([
        ("/ConstantOfShape_4", vec![seq, seq]),
        ("/ScatterND", vec![seq, seq]),
        ("/Unsqueeze_11", vec![1, seq, seq]),
    ]);
    for idx in 0..hir.len() {
        let id = HirNodeId(idx as u32);
        let node = hir.node(id);
        let Some(name) = node.name.as_deref() else {
            continue;
        };
        if let Some(shape) = explicit_vocoder_shape(name, max_waveform_samples) {
            hir.node_mut(id).shape = Shape::new(&shape, node.shape.dtype());
            continue;
        }
        if let Some(shape) = matmul1.get(name) {
            hir.node_mut(id).shape = Shape::new(shape, node.shape.dtype());
            continue;
        }
        if name == "/bert/Expand_1" || name == "/bert/Where_2" {
            hir.node_mut(id).shape = Shape::new(&[1, 1, seq, seq], node.shape.dtype());
            continue;
        }
        if !name.contains("l_sin_gen") && !name.contains("/decoder/generator/m_source/") {
            continue;
        }
        if name.contains("/Shape")
            || name.contains("/Constant")
            || name.contains("/Gather")
            || name.contains("/Where")
            || name.contains("/Reshape")
            || name.contains("/Slice")
            || name.contains("/Range")
            || name.contains("/Expand")
            || name.contains("/Equal")
            || name.contains("/Cast")
            || name.contains("/Unsqueeze")
        {
            continue;
        }
        if node.shape.dtype() != DType::F32 {
            continue;
        }
        let rank = node.shape.rank();
        let needs = rank == 2
            || (rank == 3
                && node.shape.dim(0).unwrap_static() == 1
                && node.shape.dim(1).unwrap_static() == 2)
            || node.shape.dims().is_empty();
        if needs {
            hir.node_mut(id).shape = Shape::new(&l_sin_patched, DType::F32);
        }
    }
}

fn explicit_vocoder_shape(node_name: &str, max_wave: usize) -> Option<Vec<usize>> {
    let frames = frame_cap(max_wave);
    let table: HashMap<&str, Vec<usize>> = HashMap::from([
        ("/decoder/generator/f0_upsamp/Resize", vec![1, 1, max_wave]),
        (
            "/decoder/generator/m_source/l_sin_gen/Resize",
            vec![1, HARMONICS, frames],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Resize_1",
            vec![1, HARMONICS, max_wave],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Transpose",
            vec![1, HARMONICS, max_wave],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Transpose_1",
            vec![1, frames, HARMONICS],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Transpose_2",
            vec![1, HARMONICS, frames],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Transpose_3",
            vec![1, max_wave, HARMONICS],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/ScatterND_1",
            vec![1, max_wave, HARMONICS],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/ScatterND",
            vec![1, HARMONICS],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Add_1",
            vec![1, HARMONICS],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/RandomUniformLike",
            vec![1, HARMONICS],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/CumSum",
            vec![1, frames, HARMONICS],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Mul_7",
            vec![1, frames, HARMONICS],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Mul_8",
            vec![1, frames, HARMONICS],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Mul_9",
            vec![1, HARMONICS, frames],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Sub",
            vec![1, max_wave, HARMONICS],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Mul",
            vec![1, max_wave, HARMONICS],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Add_5",
            vec![1, max_wave, HARMONICS],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Div",
            vec![1, max_wave, HARMONICS],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Div_1",
            vec![1, max_wave, HARMONICS],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Floor",
            vec![1, max_wave, HARMONICS],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Sin",
            vec![1, max_wave, HARMONICS],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Mul_10",
            vec![1, max_wave, HARMONICS],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Mul_13",
            vec![1, max_wave, HARMONICS],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Mul_14",
            vec![1, max_wave, HARMONICS],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/RandomNormalLike",
            vec![1, max_wave, HARMONICS],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Mul_11",
            vec![1, max_wave, 1],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Mul_12",
            vec![1, max_wave, 1],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Div_2",
            vec![1, max_wave, 1],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Add_4",
            vec![1, max_wave, 1],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Sub_1",
            vec![1, max_wave, 1],
        ),
    ]);
    table.get(node_name).cloned()
}

fn narrow_wave_vocoder_shape(node_name: &str, max_wave: usize) -> Option<Vec<usize>> {
    let frames = frame_cap(max_wave);
    let seq = import_sequence_length();
    let h = HARMONICS;
    let table: &[(&str, &[usize])] = &[
        ("/decoder/generator/f0_upsamp/Resize", &[1, 1, max_wave]),
        (
            "/decoder/generator/m_source/l_sin_gen/Resize_1",
            &[1, h, max_wave],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Gather_4",
            &[1, seq, h],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Expand_3",
            &[1, seq, h],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Transpose_3",
            &[1, frames, h],
        ),
        ("/decoder/generator/m_source/l_sin_gen/Sin", &[1, frames, h]),
        (
            "/decoder/generator/m_source/l_sin_gen/Mul_10",
            &[1, frames, h],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Mul_13",
            &[1, frames, h],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Mul_14",
            &[1, max_wave, h],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/RandomNormalLike",
            &[1, frames, h],
        ),
        // F0 phase / scatter chain (mel-frame axis until Resize_1 upsample).
        ("/decoder/generator/m_source/l_sin_gen/Sub", &[1, frames, h]),
        ("/decoder/generator/m_source/l_sin_gen/Mul", &[1, frames, h]),
        ("/decoder/generator/m_source/l_sin_gen/Div", &[1, frames, h]),
        (
            "/decoder/generator/m_source/l_sin_gen/Floor",
            &[1, frames, h],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/ScatterND_1",
            &[1, frames, h],
        ),
        // Mel-frame → wave upsample inside sine gen.
        (
            "/decoder/generator/m_source/l_sin_gen/Mul_9",
            &[1, h, frames],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Resize",
            &[1, h, frames],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Mul_11",
            &[1, frames, 1],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Mul_12",
            &[1, frames, 1],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Div_2",
            &[1, frames, 1],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Add_4",
            &[1, frames, 1],
        ),
        (
            "/decoder/generator/m_source/l_sin_gen/Sub_1",
            &[1, frames, 1],
        ),
    ];
    table
        .iter()
        .find(|(name, _)| *name == node_name)
        .map(|(_, shape)| shape.to_vec())
}

fn patch_narrow_wave_vocoder_shapes(nodes: &mut [BundleNode], max_wave: usize) {
    const SKIP_OPS: &[&str] = &["Shape", "Concat", "Where", "Reshape", "Constant"];
    for node in nodes.iter_mut() {
        let Some(shape) = narrow_wave_vocoder_shape(&node.name, max_wave) else {
            continue;
        };
        if SKIP_OPS.contains(&node.op.as_str()) {
            continue;
        }
        let meta = serde_json::json!({ "shape": shape, "dtype": "f32" });
        if node.output_meta.is_empty() {
            node.output_meta.push(meta);
        } else {
            node.output_meta[0] = meta;
        }
    }
}

fn patch_explicit_vocoder_shapes(
    nodes: &mut [BundleNode],
    max_wave: usize,
    sequence_length: usize,
) {
    if sequence_length < 32 {
        patch_narrow_wave_vocoder_shapes(nodes, max_wave);
        return;
    }
    // Wave-axis vocoder nodes need `max_wave`; mel rows stay `[1,300,seq]` below 32 slots.
    const SKIP_OPS: &[&str] = &["Shape", "Concat", "Where", "Reshape", "Constant", "Gather"];
    for node in nodes.iter_mut() {
        let Some(shape) = explicit_vocoder_shape(&node.name, max_wave) else {
            continue;
        };
        if SKIP_OPS.contains(&node.op.as_str()) {
            continue;
        }
        let meta = serde_json::json!({ "shape": shape, "dtype": "f32" });
        if node.output_meta.is_empty() {
            node.output_meta.push(meta);
        } else {
            node.output_meta[0] = meta;
        }
    }
}

fn patch_l_sin_gen_shapes(nodes: &mut [BundleNode], sequence_length: usize) {
    const SKIP_OPS: &[&str] = &[
        "Shape",
        "Concat",
        "Where",
        "Reshape",
        "Constant",
        "ConstantOfShape",
        "Gather",
        "Cast",
        "Unsqueeze",
        "Slice",
        "Range",
        "Expand",
        "Equal",
        "Greater",
        "Less",
        "Not",
    ];
    // Mel rows use `[1, 300, mel_frames]` where `mel_frames` tracks the waveform
    // axis via `frame_cap` (not token `sequence_length`, and not alignment upper bound).
    let mel_frames = frame_cap(import_max_waveform_samples()).max(sequence_length);
    let patched = serde_json::json!({ "shape": [1, 300, mel_frames], "dtype": "f32" });
    for node in nodes.iter_mut() {
        if explicit_vocoder_shape(&node.name, import_max_waveform_samples()).is_some() {
            continue;
        }
        if !node.name.contains("l_sin_gen") && !node.name.contains("/decoder/generator/m_source/") {
            continue;
        }
        if SKIP_OPS.contains(&node.op.as_str()) {
            continue;
        }
        if node.output_meta.is_empty() {
            continue;
        }
        for slot in &mut node.output_meta {
            let Some(dtype) = slot.get("dtype").and_then(|v| v.as_str()) else {
                continue;
            };
            if dtype != "f32" {
                continue;
            }
            let shape = slot.get("shape").and_then(|v| v.as_array());
            let needs = match shape {
                None => true,
                Some(arr) if arr.is_empty() => true,
                Some(arr) if arr.len() == 3 => {
                    arr.first().and_then(|v| v.as_u64()) == Some(1)
                        && (arr.get(1).and_then(|v| v.as_u64()) == Some(2)
                            || (arr.get(1).and_then(|v| v.as_u64()) == Some(300)
                                && arr.get(2).and_then(|v| v.as_u64())
                                    == Some(sequence_length as u64)))
                }
                _ => false,
            };
            if needs {
                *slot = patched.clone();
            }
        }
    }
}

pub use crate::mel_align::{
    compile_mel_cap, explicit_mel_hir_shape, post_shape_propagate_hook,
    post_shape_propagate_mel_time, pre_shape_propagate_hook, pre_shape_propagate_mel_time,
};

pub fn output_shape_fix(node_name: &str, shape: &Shape, sequence_length: usize) -> Option<Shape> {
    let max_wave = import_max_waveform_samples();
    if sequence_length < 32 {
        let mel_cap = mel_align::compile_mel_cap(
            sequence_length,
            max_wave,
            crate::bundle_compile::MAX_FRAMES_PER_TOKEN,
        );
        if let Some(fixed) = mel_align::explicit_mel_hir_shape(node_name, mel_cap, shape.dtype()) {
            if fixed.dims() != shape.dims() {
                return Some(fixed);
            }
            return None;
        }
    }
    if vocoder_import_shape(node_name, max_wave, sequence_length).is_some() {
        return None;
    }
    let rank = shape.rank();
    if rank == 3 {
        let d0 = shape.dim(0).unwrap_static();
        let d1 = shape.dim(1).unwrap_static();
        let d2 = shape.dim(2).unwrap_static();
        if d0 == 1 && d1 == 2 && d2 == sequence_length {
            return Some(Shape::new(
                &[1, sequence_length, sequence_length],
                shape.dtype(),
            ));
        }
    }
    if !node_name.contains("l_sin_gen") && !node_name.contains("/decoder/generator/m_source/") {
        return None;
    }
    let mel_frames = frame_cap(max_wave).max(sequence_length);
    // m_source / l_sin_gen reshapes: several rank-3 source shapes all retarget to
    // `[1, 300, mel_frames]`, so the matching conditions share one body.
    let matches_source_shape = rank == 3
        && ((shape.dim(0).unwrap_static() == 1
            && shape.dim(1).unwrap_static() == 2
            && shape.dim(2).unwrap_static() == 128)
            || shape.dim(2).unwrap_static() == 9
            || (shape.dim(0).unwrap_static() == 1
                && shape.dim(1).unwrap_static() == 300
                && shape.dim(2).unwrap_static() == sequence_length));
    if matches_source_shape {
        Some(Shape::new(&[1, 300, mel_frames], shape.dtype()))
    } else {
        None
    }
}

/// `/SplitToSequence` on the duration mask: wide-seq alignment reads carry directly
/// (ORT oracle / refined carry). `/Where_1` mixes Expand(carry) with carry for the
/// internal loop; that diverges from ORT when carry is pre-seeded.
fn patch_split_duration_mask_use_carry(nodes: &mut [BundleNode], sequence_length: usize) {
    if sequence_length < 32 {
        return;
    }
    for node in nodes.iter_mut() {
        if node.name != "/SplitToSequence" {
            continue;
        }
        if node
            .inputs
            .first()
            .is_some_and(|s| s == "/Where_1_output_0")
        {
            node.inputs[0] = crate::opts::DURATION_CARRY.to_string();
        }
    }
}

/// `/Where_1` third input: duration carry (matches ORT fixed-point + ORT carry seed).
fn patch_duration_where_input(nodes: &mut [BundleNode]) {
    for node in nodes.iter_mut() {
        if node.name != "/Where_1" || node.inputs.len() < 3 {
            continue;
        }
        if node.inputs[2] == "duration" {
            node.inputs[2] = crate::opts::DURATION_CARRY.to_string();
        }
    }
}

fn patch_bert_attention_mask_shapes(nodes: &mut [BundleNode], sequence_length: usize) {
    let seq = sequence_length as u64;
    for node in nodes.iter_mut() {
        if node.name != "/bert/Expand_1" && node.name != "/bert/Where_2" {
            continue;
        }
        let meta = serde_json::json!({ "shape": [1, 1, seq, seq], "dtype": "f32" });
        if node.output_meta.is_empty() {
            node.output_meta.push(meta);
        } else {
            node.output_meta[0] = meta;
        }
    }
}

#[cfg(test)]
mod f0_bypass_tests {
    use super::*;
    use crate::bundle_compile::{ensure_kernels_registered, import_from_bundle_cached};
    use crate::opts::GraphOptions;
    use rlx_ir::hir::HirOp;

    #[test]
    fn f0_feed_hir_shapes_seq8() {
        ensure_kernels_registered();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
        let opts = GraphOptions {
            sequence_length: 8,
            max_waveform_samples: 8 * 600 + 12_000,
        };
        let import = import_from_bundle_cached(&dir, &opts).expect("import");
        let mel = mel_align::compile_mel_cap(
            opts.sequence_length,
            opts.max_waveform_samples,
            crate::bundle_compile::MAX_FRAMES_PER_TOKEN,
        );
        let (hir, _) = crate::bundle_compile::prepare_hir_for_compile(
            import.hir.clone(),
            &import.params,
            &import.typed,
        );
        for name in [
            "/Transpose_1",
            "/shared/Reshape",
            "/F0.0/Div",
            "/F0.0/Add",
            "/F0.2/Div",
            "/F0_proj/Conv_output_0_Cast_to_float32_0",
        ] {
            let node = hir
                .nodes()
                .iter()
                .find(|n| n.name.as_deref() == Some(name))
                .unwrap_or_else(|| panic!("missing {name}"));
            let dims: Vec<usize> = node
                .shape
                .dims()
                .iter()
                .map(|d| d.unwrap_static())
                .collect();
            eprintln!(
                "{name} {dims:?} elems={}",
                node.shape.num_elements().unwrap_or(0)
            );
            match name {
                "/Transpose_1" | "/F0.0/Div" | "/F0.0/Add" => {
                    assert_eq!(dims, vec![1, 512, mel], "{name}");
                }
                "/shared/Reshape" => assert_eq!(dims, vec![mel, 1, 512], "{name}"),
                "/F0.2/Div" => assert_eq!(dims, vec![1, 256, mel], "{name}"),
                "/F0_proj/Conv_output_0_Cast_to_float32_0" => {
                    assert_eq!(dims, vec![1, 1, mel], "{name}");
                }
                _ => {}
            }
        }
    }

    #[test]
    fn bundle_hir_installs_f0_bypass() {
        ensure_kernels_registered();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
        let opts = GraphOptions {
            sequence_length: 16,
            max_waveform_samples: 50_400,
        };
        let import = import_from_bundle_cached(&dir, &opts).expect("import");
        let mut hir = import.hir;
        let if_id = (0..hir.len()).find_map(|idx| {
            let id = rlx_ir::hir::HirNodeId(idx as u32);
            if let HirOp::Param { name } = &hir.node(id).op {
                if name == "__stub__//If_output_0" {
                    return Some(id);
                }
            }
            None
        });
        let f0_id = (0..hir.len()).find_map(|idx| {
            let id = rlx_ir::hir::HirNodeId(idx as u32);
            if hir.node(id).name.as_deref() == Some("/F0_proj/Conv_output_0_Cast_to_float32_0") {
                return Some(id);
            }
            None
        });
        assert!(if_id.is_some(), "If stub id not found");
        assert!(f0_id.is_some(), "F0 cast id not found");
        assert!(
            inject_if_f0_bypass(&mut hir),
            "inject_if_f0_bypass failed despite stub/f0 ids"
        );
        assert!(
            inject_if_n_bypass(&mut hir),
            "inject_if_n_bypass failed on imported bundle HIR"
        );
    }

    #[test]
    #[cfg(feature = "native")]
    fn prepare_hir_installs_f0_bypass() {
        ensure_kernels_registered();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
        let opts = GraphOptions {
            sequence_length: 8,
            max_waveform_samples: 24_000,
        };
        let import = import_from_bundle_cached(&dir, &opts).expect("import");
        let mut hir = import.hir;
        let mut params = import.params.clone();
        crate::bundle_patches::set_import_sequence_length(opts.sequence_length);
        crate::bundle_patches::set_import_max_waveform_samples(opts.max_waveform_samples);
        crate::native::flow::finish_bundle_hir_for_compile(
            &mut hir,
            &mut params,
            opts.sequence_length,
            opts.max_waveform_samples,
        );
        assert_f0_if_bypass(&hir);
        assert_n_if_bypass(&hir);
        if crate::compile_profile::env_flag("KITTEN_RLX_ENABLE_NARROW_WAVEFORM_SLICE") {
            assert!(
                (0..hir.len()).any(|idx| {
                    let id = rlx_ir::hir::HirNodeId(idx as u32);
                    matches!(
                        &hir.node(id).op,
                        HirOp::Mir(Op::Custom { name, .. }) if name == VOCODER_WAVEFORM_SLICE
                    )
                }),
                "VocoderWaveformSlice missing when KITTEN_RLX_ENABLE_NARROW_WAVEFORM_SLICE=1"
            );
        }
    }

    #[cfg(feature = "native")]
    fn assert_f0_if_bypass(hir: &HirModule) {
        let bypass = (0..hir.len()).find_map(|idx| {
            let id = rlx_ir::hir::HirNodeId(idx as u32);
            let node = hir.node(id);
            if let HirOp::Mir(Op::Custom { name, .. }) = &node.op {
                if name == F0_IF_SELECT {
                    return Some(id);
                }
            }
            None
        });
        assert!(
            bypass.is_some(),
            "F0 If select custom op missing after prepare"
        );
    }

    #[cfg(feature = "native")]
    fn assert_n_if_bypass(hir: &HirModule) {
        let bypass = (0..hir.len()).find_map(|idx| {
            let id = rlx_ir::hir::HirNodeId(idx as u32);
            let node = hir.node(id);
            if let HirOp::Mir(Op::Custom { name, .. }) = &node.op {
                if name == F0_IF_SELECT {
                    let feeds_n = node.inputs.iter().any(|&inp| {
                        hir.node(inp)
                            .name
                            .as_deref()
                            .is_some_and(|n| n.contains("N_proj"))
                    });
                    if feeds_n {
                        return Some(id);
                    }
                }
            }
            None
        });
        assert!(
            bypass.is_some(),
            "N If select custom op missing after prepare"
        );
    }

    #[test]
    fn pre_shape_propagate_binds_f0_mel_axis() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
        let bundle = rlx_onnx_import::load_bundle(&dir).expect("bundle");
        let mut nodes = bundle.nodes.clone();
        crate::bundle_patches::patch_bundle_nodes(&mut nodes, 8, 50_400);
        let opts = crate::bundle_compile::import_opts(&GraphOptions {
            sequence_length: 8,
            max_waveform_samples: 50_400,
        });
        crate::bundle_patches::pre_shape_propagate_mel_time(&mut nodes, &opts);
        crate::bundle_patches::post_shape_propagate_mel_time(&mut nodes, &opts);
        let mel_cap = crate::mel_align::compile_mel_cap(
            8,
            50_400,
            crate::bundle_compile::MAX_FRAMES_PER_TOKEN,
        );
        let f0 = nodes
            .iter()
            .find(|n| n.name == "/F0.0/Div")
            .expect("/F0.0/Div");
        let sh = f0.output_meta[0]["shape"].as_array().expect("shape");
        assert_eq!(sh[2].as_u64(), Some(mel_cap as u64));
    }

    #[test]
    fn propagate_shapes_keeps_f0_mel_axis() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
        let bundle = rlx_onnx_import::load_bundle(&dir).expect("bundle");
        let mut nodes = bundle.nodes.clone();
        crate::bundle_patches::patch_bundle_nodes(&mut nodes, 8, 50_400);
        let opts = crate::bundle_compile::import_opts(&GraphOptions {
            sequence_length: 8,
            max_waveform_samples: 50_400,
        });
        crate::bundle_patches::pre_shape_propagate_mel_time(&mut nodes, &opts);
        let init_shapes = std::collections::HashMap::new();
        rlx_onnx_import::shape_propagate::propagate_shapes(
            &mut nodes,
            &bundle.manifest,
            &init_shapes,
            &opts,
        );
        crate::bundle_patches::post_shape_propagate_mel_time(&mut nodes, &opts);
        let mel_cap = crate::mel_align::compile_mel_cap(
            8,
            50_400,
            crate::bundle_compile::MAX_FRAMES_PER_TOKEN,
        );
        let f0 = nodes
            .iter()
            .find(|n| n.name == "/F0.0/Div")
            .expect("/F0.0/Div");
        let sh = f0.output_meta[0]["shape"].as_array().expect("shape");
        assert_eq!(
            sh[2].as_u64(),
            Some(mel_cap as u64),
            "propagate_shapes regressed mel axis: {sh:?}"
        );
        let shared = nodes
            .iter()
            .find(|n| n.name == "/shared/Transpose")
            .expect("/shared/Transpose");
        let sh = shared.output_meta[0]["shape"].as_array().expect("shape");
        assert_eq!(
            sh[0].as_u64(),
            Some(mel_cap as u64),
            "/shared/Transpose mel axis: {sh:?}"
        );
        assert_eq!(sh[2].as_u64(), Some(640));
    }

    #[test]
    fn f0_mel_time_import_shapes() {
        ensure_kernels_registered();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
        let opts = GraphOptions {
            sequence_length: 8,
            max_waveform_samples: 50_400,
        };
        let import = import_from_bundle_cached(&dir, &opts).expect("import");
        let mel_cap = crate::mel_align::compile_mel_cap(
            opts.sequence_length,
            opts.max_waveform_samples,
            crate::bundle_compile::MAX_FRAMES_PER_TOKEN,
        );
        let mut nodes = rlx_onnx_import::load_bundle(&dir).expect("bundle").nodes;
        let import_opts = crate::bundle_compile::import_opts(&opts);
        crate::bundle_patches::pre_shape_propagate_mel_time(&mut nodes, &import_opts);
        let init_shapes = std::collections::HashMap::new();
        rlx_onnx_import::shape_propagate::propagate_shapes(
            &mut nodes,
            &rlx_onnx_import::load_bundle(&dir).expect("bundle").manifest,
            &init_shapes,
            &import_opts,
        );
        crate::bundle_patches::post_shape_propagate_mel_time(&mut nodes, &import_opts);
        for name in ["/F0.0/Div", "/MatMul", "/shared/Transpose"] {
            let node = nodes
                .iter()
                .find(|n| n.name == name)
                .unwrap_or_else(|| panic!("missing bundle {name}"));
            let sh = node.output_meta[0]["shape"].as_array().expect("shape");
            let t = if name == "/shared/Transpose" {
                sh[0].as_u64().unwrap() as usize
            } else {
                sh[2].as_u64().unwrap() as usize
            };
            assert_eq!(
                t, mel_cap,
                "{name} bundle mel axis {t} != {mel_cap}: {sh:?}"
            );
        }
        let matmul = import
            .hir
            .nodes()
            .iter()
            .find(|n| n.name.as_deref() == Some("/MatMul"))
            .expect("/MatMul hir");
        let t34 = import
            .hir
            .nodes()
            .iter()
            .find(|n| n.name.as_deref() == Some("Transpose_token_34"))
            .expect("Transpose_token_34");
        let t34_dims: Vec<_> = t34.shape.dims().iter().map(|d| d.unwrap_static()).collect();
        assert_eq!(
            t34_dims,
            vec![1, 640, opts.sequence_length],
            "Where_4→MatMul feed must be NCL [1,640,seq]"
        );
        assert_eq!(
            matmul.shape.dim(2).unwrap_static(),
            mel_cap,
            "/MatMul HIR mel axis should use compile mel cap: {:?}",
            matmul.shape.dims()
        );
        let mm_in = import.hir.node(matmul.id).inputs[0];
        let feed = &import.hir.node(mm_in);
        assert!(
            feed.name.as_deref() == Some("Transpose_token_34")
                || feed.name.as_deref() == Some("/text_encoder_1/Where_4_output_0"),
            "MatMul feed {:?}",
            feed.name
        );
        assert_eq!(
            matmul.shape.dim(1).unwrap_static(),
            640,
            "/MatMul HIR channel axis: {:?}",
            matmul.shape.dims()
        );
        let st = import
            .hir
            .nodes()
            .iter()
            .find(|n| n.name.as_deref() == Some("/shared/Transpose"))
            .expect("/shared/Transpose");
        assert_eq!(
            st.shape.dim(2).unwrap_static(),
            640,
            "/shared/Transpose LSTM feature dim: {:?}",
            st.shape.dims()
        );
        let f0_div = import
            .hir
            .nodes()
            .iter()
            .find(|n| n.name.as_deref() == Some("/F0.0/Div"))
            .expect("/F0.0/Div hir");
        assert!(
            f0_div.shape.dim(2).unwrap_static() <= mel_cap,
            "/F0.0/Div HIR mel axis should not exceed cap: {:?}",
            f0_div.shape.dims()
        );
    }

    #[test]
    fn bundle_hir_installs_mel_alignment_inject() {
        ensure_kernels_registered();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
        let opts = GraphOptions {
            sequence_length: 8,
            max_waveform_samples: 50_400,
        };
        let import = import_from_bundle_cached(&dir, &opts).expect("import");
        let mut hir = import.hir;
        crate::bundle_patches::set_import_sequence_length(opts.sequence_length);
        crate::bundle_patches::set_import_max_waveform_samples(opts.max_waveform_samples);
        assert!(
            inject_vocoder_dynamic_alignment(
                &mut hir,
                opts.sequence_length,
                opts.max_waveform_samples,
            ),
            "mel alignment inject failed on bundle HIR"
        );
        let expand = (0..hir.len()).find_map(|idx| {
            let id = rlx_ir::hir::HirNodeId(idx as u32);
            if hir.node(id).name.as_deref() == Some("/Expand_3") {
                return Some(id);
            }
            None
        });
        let expand = expand.expect("/Expand_3 node");
        assert!(
            matches!(
                &hir.node(expand).op,
                HirOp::Mir(Op::Custom { name, .. }) if name == EXPAND_I64_ALIGN
            ),
            "Expand_3 should lower to ExpandI64Align after inject"
        );
    }
}
