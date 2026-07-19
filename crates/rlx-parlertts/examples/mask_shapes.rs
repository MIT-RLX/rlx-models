// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Print the HIR node shapes along the T5 encoder attention-mask chain, so we can
//! see exactly where the rank collapses (`[1,1,1,128]` → `[128]`). No ort.
//!
//! ```text
//! RLX_PARLER_DIR=weights/tts/parlertts cargo run -p rlx-parlertts --example mask_shapes --features native
//! ```

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use rlx_onnx_import::{ImportOptions, build_hir_from_onnx_file};

fn main() -> Result<()> {
    let dir = std::env::var("RLX_PARLER_DIR").unwrap_or_else(|_| "weights/tts/parlertts".into());
    let path = Path::new(&dir).join("onnx/text_encoder.onnx");
    // Match native_parity's exact config (sequence_length + t + batch_size; NO `b`).
    let mut named: HashMap<String, usize> = HashMap::new();
    for (k, v) in [("sequence_length", 128usize), ("t", 128), ("batch_size", 1)] {
        named.insert(k.to_string(), v);
    }
    let opts = ImportOptions {
        sequence_length: 128,
        named_lengths: named,
        strict: false,
        ..Default::default()
    };
    let (hir, _params, _r, _m) = build_hir_from_onnx_file(&path, opts)?;

    // The importer renames nodes; match on substrings of the ONNX output names that
    // the env inserts. We print every node whose recorded name touches the mask path.
    let keys = [
        "attention_mask",
        "Unsqueeze_1",
        "Unsqueeze_2",
        "Cast_1",
        "Sub",
        "Mul_output",
        "block.0/layer.0/SelfAttention/Slice_1",
        "block.0/layer.0/SelfAttention/Slice_output",
        "block.0/layer.0/SelfAttention/Add_3",
        "block.0/layer.0/SelfAttention/Add_4",
        "block.0/layer.0/SelfAttention/MatMul_output",
        "relative_attention_bias",
        "Transpose_4",
        "Unsqueeze_5",
        "final_layer_norm/Cast",
        "final_layer_norm/Pow",
        "final_layer_norm/ReduceMean",
        "final_layer_norm/Add",
        "final_layer_norm/Sqrt",
        "final_layer_norm/Div",
        "final_layer_norm/Mul",
    ];
    if std::env::var_os("RLX_TRACE_ATTN").is_some() {
        // Full block.0/layer.0/SelfAttention node list in topo order.
        for n in hir.nodes() {
            let Some(nm) = n.name.as_deref() else {
                continue;
            };
            if nm.contains("block.0/layer.0/SelfAttention")
                && !nm.contains("block.10")
                && !nm.contains("block.1/")
            {
                let short = nm.rsplit('/').next().unwrap_or(nm);
                println!("{short:24}  {:?}  shape={:?}", n.op, n.shape.dims());
            }
        }
        return Ok(());
    }
    for n in hir.nodes() {
        let Some(nm) = n.name.as_deref() else {
            continue;
        };
        if keys.iter().any(|k| nm.contains(k)) {
            println!("{:60}  {:?}  shape={:?}", nm, n.op, n.shape.dims());
        }
    }
    Ok(())
}
