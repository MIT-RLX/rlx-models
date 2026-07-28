// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Inspect QMatMul / QMatMulBaked node structure in the compiled Kitten HIR.
//!
//! For each quantized matmul, prints num_inputs and the producer op-kind / name /
//! shape of the activation input (0) and weight input (3). Used to design the
//! native-f32 matmul rewrite (eliminate host QDQ round-trips on GPU backends).

use std::collections::HashMap;

use kitten_tts_mini_rlx::GraphOptions;
use kitten_tts_mini_rlx::bundle_compile::import_from_bundle_cached;
use rlx_ir::Op;
use rlx_ir::hir::{HirModule, HirNodeId, HirOp};

fn producer_desc(hir: &HirModule, id: HirNodeId) -> String {
    let n = hir.node(id);
    let kind = match &n.op {
        HirOp::Param { name } => format!("Param({name})"),
        HirOp::Mir(Op::Custom {
            name, num_inputs, ..
        }) => {
            format!("Custom({name},ni={num_inputs})")
        }
        HirOp::Mir(op) => format!("Mir({op:?})"),
        other => format!("{other:?}"),
    };
    let kind = if kind.len() > 60 {
        kind[..60].to_string()
    } else {
        kind
    };
    format!(
        "{kind} shape={:?} dtype={:?}",
        n.shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect::<Vec<_>>(),
        n.shape.dtype()
    )
}

fn main() -> anyhow::Result<()> {
    let bundle_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    let ids: Vec<i64> = vec![0, 50, 83, 156, 54, 57, 135, 10, 0];
    let seq = kitten_tts_mini_rlx::compile_profile::compile_slot_length(ids.len());
    let graph_opts = GraphOptions {
        sequence_length: seq,
        max_waveform_samples: 55_200,
    };
    kitten_tts_mini_rlx::opts::set_compile_sequence_length(seq);
    let import = import_from_bundle_cached(&bundle_dir, &graph_opts)?;

    // Mirror compile: bake + fuse.
    let mut hir = import.hir.clone();
    let baked =
        kitten_tts_mini_rlx::qmatmul_bake::bake_qmatmul_weights(&import.typed, &import.params);
    let fused = kitten_tts_mini_rlx::hir_qdq_fuse::fuse_qmatmul_baked_weights(&mut hir, &baked);
    eprintln!("baked {} weights, fused {fused} QMatMul→Baked", baked.len());

    let mut counts: HashMap<(String, u32), usize> = HashMap::new();
    let mut printed = 0usize;
    let ids_all: Vec<HirNodeId> = hir.nodes().iter().map(|n| n.id).collect();
    for id in ids_all {
        let node = hir.node(id);
        let HirOp::Mir(Op::Custom {
            name, num_inputs, ..
        }) = &node.op
        else {
            continue;
        };
        if name != "onnx.QMatMul" && name != "onnx.QMatMulBaked" {
            continue;
        }
        *counts.entry((name.clone(), *num_inputs)).or_default() += 1;
        if printed < 24 {
            printed += 1;
            let inputs = node.inputs.clone();
            let act = inputs
                .first()
                .map(|i| producer_desc(&hir, *i))
                .unwrap_or_default();
            let w = inputs
                .get(3)
                .map(|i| producer_desc(&hir, *i))
                .unwrap_or_default();
            eprintln!(
                "--- {name} ni={num_inputs} out_shape={:?}",
                node.shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static())
                    .collect::<Vec<_>>()
            );
            eprintln!("    act[0]: {act}");
            eprintln!("    w  [3]: {w}");
            // If act producer is DQL, show its input producer (the real f32 activation).
            let act_id = inputs[0];
            if let HirOp::Mir(Op::Custom { name: an, .. }) = &hir.node(act_id).op {
                if an.contains("DynamicQuantize") {
                    let src = hir.node(act_id).inputs.first().copied();
                    if let Some(s) = src {
                        eprintln!("    act src (DQL in): {}", producer_desc(&hir, s));
                    }
                }
            }
        }
    }
    eprintln!("\n== QMatMul node counts (static, one graph) ==");
    let mut v: Vec<_> = counts.into_iter().collect();
    v.sort();
    for ((name, ni), c) in v {
        eprintln!("  {name} ni={ni}: {c}");
    }
    Ok(())
}
