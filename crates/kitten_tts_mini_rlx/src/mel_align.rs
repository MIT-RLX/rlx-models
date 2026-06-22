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

//! Mel-time alignment for the F0 / shared-LSTM vocoder feed.
//!
//! # ONNX chain (narrow seq)
//!
//! ```text
//! Where_4 [1,seq,640] → Transpose_token_34 → [1,640,seq]
//!   → MatMul × alignment mask [1,seq,mel_cap] → [1,640,mel_cap]
//!   → /shared/Transpose → [mel_cap,1,640] → LSTM → F0 stack
//! ```
//!
//! # Compile vs runtime width
//!
//! - **Compile cap** (`compile_mel_cap`): upper bound on mel-time axis in bundle meta / HIR
//!   allocation (from `alignment_buffer_upper_bound`, typically ~512 for seq=8).
//! - **Runtime width** (`runtime_mel_frames` = alignment frame count): active mel steps;
//!   LSTM, F0IfSelect, and DynamicQuantizeLinearExport trim to this at execute time.
//!
//! # Import pipeline
//!
//! 1. [`patch_alignment_mask`] — alignment scatter mask `[seq, mel_cap]` (in `patch_bundle_nodes`)
//! 2. Optional [`pre_shape_propagate`] / [`post_shape_propagate`] — rebind ONNX `unk__*` mel axes
//!    (gated by [`import_mel_propagate_enabled`]; off by default until split-graph arena is fixed)
//! 3. HIR lower uses bundle meta; kernels honor `runtime_mel_frames()`

use rlx_onnx_import::alignment_buffer_upper_bound;
use rlx_onnx_import::{BundleNode, ImportOptions};

use crate::compile_profile;
use crate::opts::GraphOptions;

/// Alignment mask + scatter nodes (`/MatMul` left operand is `Unsqueeze_11`).
pub const ALIGNMENT_MASK_NODES: &[&str] = &["/ConstantOfShape_4", "/ScatterND", "/Unsqueeze_11"];

/// Probes for the text-encoder → F0 feed (compare_intermediates / parity).
pub const F0_FEED_PROBE_NODES: &[&str] =
    &["/MatMul", "/shared/Transpose", "/Transpose_1", "/F0.0/Div"];

const MEL_TIME_SYMBOLS: &[&str] = &[
    "unk__318", "unk__319", "unk__325", "unk__337", "unk__342", "unk__360", "unk__386", "unk__387",
    "unk__388", "unk__390", "unk__391", "unk__392", "unk__393", "unk__394", "unk__396", "unk__397",
    "unk__398", "unk__399", "unk__400", "unk__401", "unk__402", "unk__403", "unk__405", "unk__406",
    "unk__407", "unk__408", "unk__409", "unk__411", "unk__412", "unk__413", "unk__414", "unk__415",
    "unk__825", "unk__826", "unk__832", "unk__833", "unk__840", "unk__841", "unk__844", "unk__845",
    "unk__846", "unk__847", "unk__852", "unk__853", "unk__854", "unk__855", "unk__858", "unk__859",
];

/// Mel-time compile cap: matches `lower_if_stub` / `alignment_slots` (runtime uses alignment frames).
pub fn compile_mel_cap(
    sequence_length: usize,
    max_waveform_samples: usize,
    max_frames_per_token: usize,
) -> usize {
    alignment_buffer_upper_bound(sequence_length, max_waveform_samples, max_frames_per_token)
}

pub fn compile_mel_cap_from_opts(opts: &ImportOptions) -> usize {
    compile_mel_cap(
        opts.sequence_length,
        opts.max_waveform_samples,
        opts.max_frames_per_token,
    )
}

pub fn compile_mel_cap_from_graph(opts: &GraphOptions, max_frames_per_token: usize) -> usize {
    compile_mel_cap(
        opts.sequence_length,
        opts.max_waveform_samples,
        max_frames_per_token,
    )
}

/// Whether import runs minimal F0-feed meta fix after `propagate_shapes` (default on for seq&lt;32).
pub fn import_f0_feed_meta_enabled(sequence_length: usize) -> bool {
    if sequence_length >= 32 {
        return false;
    }
    !compile_profile::env_flag("KITTEN_RLX_FULL_GRAPH")
}

/// Whether bundle import runs full mel pre/post shape hooks (`KITTEN_RLX_MEL_SHAPE_PROPAGATE=1`).
/// Default import only patches alignment mask + F0 feed meta (see [`patch_f0_feed_meta`]).
pub fn import_mel_propagate_enabled(sequence_length: usize) -> bool {
    if sequence_length >= 32 {
        return false;
    }
    if compile_profile::env_flag("KITTEN_RLX_FULL_GRAPH") {
        return false;
    }
    compile_profile::env_flag("KITTEN_RLX_MEL_SHAPE_PROPAGATE")
}

/// Set alignment mask tensor shapes before ONNX shape propagation.
pub fn patch_alignment_mask(
    nodes: &mut [BundleNode],
    sequence_length: usize,
    max_waveform_samples: usize,
    max_frames_per_token: usize,
    use_mel_cap: bool,
) {
    let seq = sequence_length as u64;
    let mel = if use_mel_cap {
        compile_mel_cap(sequence_length, max_waveform_samples, max_frames_per_token) as u64
    } else {
        seq
    };
    let table = [
        ("/ConstantOfShape_4", vec![seq, mel]),
        ("/ScatterND", vec![seq, mel]),
        ("/Unsqueeze_11", vec![1, seq, mel]),
    ];
    for node in nodes.iter_mut() {
        let Some((_, shape)) = table.iter().find(|(name, _)| *name == node.name.as_str()) else {
            continue;
        };
        let meta = serde_json::json!({
            "shape": shape.iter().map(|d| serde_json::json!(d)).collect::<Vec<_>>(),
            "dtype": "f32",
        });
        if node.output_meta.is_empty() {
            node.output_meta.push(meta);
        } else {
            node.output_meta[0] = meta;
        }
    }
}

/// Patch F0-feed bundle meta (`/MatMul` → `/shared/Transpose` chain) without widening the
/// whole vocoder graph. Safe for import + execute; full [`pre_shape_propagate`] is opt-in.
pub fn patch_f0_feed_meta(
    nodes: &mut [BundleNode],
    sequence_length: usize,
    max_waveform_samples: usize,
    max_frames_per_token: usize,
) {
    if sequence_length >= 32 {
        return;
    }
    patch_f0_bundle_shapes(
        nodes,
        sequence_length,
        max_waveform_samples,
        max_frames_per_token,
    );
}

/// Hook for `ImportOptions::pre_shape_propagate` (also used by unit tests).
pub fn pre_shape_propagate(nodes: &mut [BundleNode], opts: &ImportOptions) {
    let mel_cap = compile_mel_cap_from_opts(opts);
    for node in nodes.iter_mut() {
        if !mel_bundle_path(node.name.as_str()) {
            continue;
        }
        for meta in node.output_meta.iter_mut() {
            if let Some(shape) = meta.get_mut("shape").and_then(|s| s.as_array_mut()) {
                replace_mel_symbols(shape, mel_cap);
            }
        }
    }
}

/// Hook for `ImportOptions::post_shape_propagate` (rebind after `propagate_shapes`).
pub fn post_shape_propagate(nodes: &mut [BundleNode], opts: &ImportOptions) {
    let mel_cap = compile_mel_cap_from_opts(opts);
    let seq = opts.sequence_length;
    for node in nodes.iter_mut() {
        if !mel_bundle_path(node.name.as_str()) {
            continue;
        }
        for meta in node.output_meta.iter_mut() {
            let Some(shape) = meta.get_mut("shape").and_then(|s| s.as_array_mut()) else {
                continue;
            };
            replace_mel_symbols(shape, mel_cap);
            apply_mel_cap_rank3(node.name.as_str(), shape, mel_cap, seq);
        }
    }
    patch_f0_bundle_shapes(
        nodes,
        seq,
        opts.max_waveform_samples,
        opts.max_frames_per_token,
    );
}

/// After ONNX `propagate_shapes`, widen F0/N mel-time axes to compile cap (no `unk__*` pre-pass).
pub fn post_shape_propagate_minimal(nodes: &mut [BundleNode], opts: &ImportOptions) {
    patch_f0_bundle_shapes(
        nodes,
        opts.sequence_length,
        opts.max_waveform_samples,
        opts.max_frames_per_token,
    );
}

#[inline]
pub fn pre_shape_propagate_mel_time(nodes: &mut [BundleNode], opts: &ImportOptions) {
    pre_shape_propagate(nodes, opts);
}

#[inline]
pub fn post_shape_propagate_mel_time(nodes: &mut [BundleNode], opts: &ImportOptions) {
    post_shape_propagate(nodes, opts);
}

/// Dev-tool aliases (hir_shape_dump, manual propagate tests).
#[inline]
pub fn pre_shape_propagate_hook(nodes: &mut [BundleNode], opts: &ImportOptions) {
    pre_shape_propagate(nodes, opts);
}

#[inline]
pub fn post_shape_propagate_hook(nodes: &mut [BundleNode], opts: &ImportOptions) {
    post_shape_propagate(nodes, opts);
}

const TEXT_ENCODER_FEATURES: usize = 640;
const F0_BLOCK0_CHANNELS: usize = 512;
const F0_BLOCK1_CHANNELS: usize = 256;
const SHARED_LSTM_HIDDEN: usize = 256;

/// Canonical HIR shape for F0-feed nodes (used by import `output_shape_fix`).
pub fn explicit_mel_hir_shape(
    node_name: &str,
    mel_cap: usize,
    dtype: rlx_ir::DType,
) -> Option<rlx_ir::Shape> {
    explicit_mel_bundle_shape(node_name, mel_cap).map(|dims| rlx_ir::Shape::new(&dims, dtype))
}

fn explicit_mel_bundle_shape(node_name: &str, mel_cap: usize) -> Option<Vec<usize>> {
    match node_name {
        "/MatMul" => Some(vec![1, TEXT_ENCODER_FEATURES, mel_cap]),
        "/shared/Transpose" => Some(vec![mel_cap, 1, TEXT_ENCODER_FEATURES]),
        "/shared/LSTM_quant" => Some(vec![mel_cap, 2, 1, SHARED_LSTM_HIDDEN]),
        "/shared/Transpose_1" => Some(vec![mel_cap, 1, 2, SHARED_LSTM_HIDDEN]),
        "/shared/Reshape" => Some(vec![mel_cap, 1, F0_BLOCK0_CHANNELS]),
        "/Transpose_1" => Some(vec![1, F0_BLOCK0_CHANNELS, mel_cap]),
        "/F0.0/Div" | "/N.0/Div" => Some(vec![1, F0_BLOCK0_CHANNELS, mel_cap]),
        "/F0.1/Div" | "/F0.2/Div" | "/N.1/Div" | "/N.2/Div" => {
            Some(vec![1, F0_BLOCK1_CHANNELS, mel_cap])
        }
        "/F0_proj/Conv_output_0"
        | "/F0_proj/Conv_output_0_Cast_to_float32_0"
        | "/F0_proj/Conv_output_0_Cast_to_float32_output_0" => Some(vec![1, 1, mel_cap]),
        _ => None,
    }
}

fn predictor_block_channels(node_name: &str) -> Option<usize> {
    if node_name.starts_with("/F0.0/") || node_name.starts_with("/N.0/") {
        return Some(F0_BLOCK0_CHANNELS);
    }
    if node_name.starts_with("/F0.1/")
        || node_name.starts_with("/F0.2/")
        || node_name.starts_with("/N.1/")
        || node_name.starts_with("/N.2/")
    {
        return Some(F0_BLOCK1_CHANNELS);
    }
    None
}

fn set_bundle_output_shape(node: &mut BundleNode, dims: &[usize]) {
    let shape: Vec<serde_json::Value> = dims.iter().map(|&d| serde_json::json!(d)).collect();
    let meta = serde_json::json!({ "shape": shape, "dtype": "f32" });
    if node.output_meta.is_empty() {
        node.output_meta.push(meta);
    } else {
        node.output_meta[0] = meta;
    }
}

fn f0_graph_node(name: &str) -> bool {
    name.starts_with("/F0")
        || name.starts_with("/N.")
        || name.starts_with("/N_proj")
        || name == "/Transpose_1"
        || name == "/MatMul"
        || name.starts_with("/shared/")
}

fn mel_bundle_path(name: &str) -> bool {
    f0_graph_node(name)
}

fn replace_mel_symbols(shape: &mut [serde_json::Value], mel_cap: usize) {
    for d in shape.iter_mut() {
        if d.as_str().is_some_and(|s| MEL_TIME_SYMBOLS.contains(&s)) {
            *d = serde_json::json!(mel_cap);
        }
    }
}

fn is_mel_time_dim(v: &serde_json::Value, sequence_length: usize, mel_cap: usize) -> bool {
    v.as_u64() == Some(mel_cap as u64)
        || v.as_str()
            .is_some_and(|s| MEL_TIME_SYMBOLS.contains(&s) || s == "sequence_length")
        || v.as_u64() == Some(sequence_length as u64)
}

fn is_channel_dim(v: &serde_json::Value, mel_cap: usize) -> bool {
    v.as_u64()
        .is_some_and(|n| (64..=1024).contains(&(n as usize)) && n as usize != mel_cap)
}

fn apply_mel_cap_rank3(
    node_name: &str,
    arr: &mut [serde_json::Value],
    mel_cap: usize,
    sequence_length: usize,
) {
    if arr.len() != 3 {
        return;
    }
    let d0 = arr[0].clone();
    let d1 = arr[1].clone();
    let d2 = arr[2].clone();

    if d0.as_u64() == Some(1) && is_channel_dim(&d1, mel_cap) && d2.as_u64() == Some(1) {
        return;
    }

    if node_name == "/shared/Transpose" && is_channel_dim(&d2, mel_cap) {
        arr[0] = serde_json::json!(mel_cap);
        arr[1] = serde_json::json!(1);
        return;
    }
    if node_name == "/shared/Reshape" {
        arr[0] = serde_json::json!(mel_cap);
        arr[1] = serde_json::json!(1);
        arr[2] = serde_json::json!(F0_BLOCK0_CHANNELS);
        return;
    }
    if node_name == "/MatMul" {
        arr[0] = serde_json::json!(1);
        arr[1] = serde_json::json!(TEXT_ENCODER_FEATURES);
        arr[2] = serde_json::json!(mel_cap);
        return;
    }
    if node_name == "/Transpose_1" {
        arr[0] = serde_json::json!(1);
        arr[1] = serde_json::json!(F0_BLOCK0_CHANNELS);
        arr[2] = serde_json::json!(mel_cap);
        return;
    }
    if let Some(ch) = predictor_block_channels(node_name) {
        // AdaIN heads are `[*, block_ch, mel]`; skip internal conv/pool tensors whose
        // channel axis is not the block width (e.g. `/N.1/pool/ConvTranspose` uses 512).
        if d1.as_u64() == Some(ch as u64) {
            arr[0] = serde_json::json!(1);
            arr[1] = serde_json::json!(ch);
            arr[2] = serde_json::json!(mel_cap);
        }
        return;
    }
    if node_name.contains("ConvTranspose") || node_name.contains("/pool/") {
        return;
    }
    if node_name.starts_with("/N_proj") {
        arr[0] = serde_json::json!(1);
        if !is_channel_dim(&d1, mel_cap) && d1.as_u64().is_some_and(|n| n <= 1) {
            arr[1] = serde_json::json!(1);
        }
        arr[2] = serde_json::json!(mel_cap);
        return;
    }

    if is_channel_dim(&d2, mel_cap)
        && (is_mel_time_dim(&d0, sequence_length, mel_cap) || d0.as_str().is_some())
    {
        arr[0] = serde_json::json!(mel_cap);
        if d1.as_str().is_some() || d1.as_u64().is_some_and(|n| n <= 2) {
            arr[1] = serde_json::json!(1);
        }
        return;
    }

    if is_mel_time_dim(&d2, sequence_length, mel_cap)
        || (is_channel_dim(&d1, mel_cap) && !is_channel_dim(&d2, mel_cap))
    {
        if d0.as_str().is_some() || d0.as_u64().unwrap_or(2) <= 2 {
            arr[0] = serde_json::json!(1);
        }
        arr[2] = serde_json::json!(mel_cap);
        return;
    }

    if d0.as_u64() == Some(1)
        && d1.as_u64().is_some_and(|n| n <= 1)
        && (is_mel_time_dim(&d2, sequence_length, mel_cap) || d2.as_str().is_some())
    {
        arr[2] = serde_json::json!(mel_cap);
    }
}

fn patch_f0_bundle_shapes(
    nodes: &mut [BundleNode],
    sequence_length: usize,
    max_waveform_samples: usize,
    max_frames_per_token: usize,
) {
    let mel_cap = compile_mel_cap(sequence_length, max_waveform_samples, max_frames_per_token);
    for node in nodes.iter_mut() {
        let name = node.name.as_str();
        if !mel_bundle_path(name) {
            continue;
        }
        if node.output_meta.is_empty() {
            continue;
        }
        if let Some(dims) = explicit_mel_bundle_shape(name, mel_cap) {
            set_bundle_output_shape(node, &dims);
            continue;
        }
        let Some(shape) = node.output_meta[0].get_mut("shape") else {
            continue;
        };
        let Some(arr) = shape.as_array_mut() else {
            continue;
        };
        if arr.len() == 3 {
            apply_mel_cap_rank3(name, arr, mel_cap, sequence_length);
        }
    }
}
