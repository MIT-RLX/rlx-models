// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Compile-time QDQ fusion: fold baked f32 weights into `onnx.QMatMulBaked` (4 inputs).

use std::collections::HashMap;

use rlx_ir::hir::{HirModule, HirNodeId, HirOp};
use rlx_ir::{DType, Op, Shape};

use crate::kernels::Q_MATMUL_BAKED;
use crate::qmatmul_bake::baked_param_name;

fn resolve_param_name(hir: &HirModule, id: HirNodeId) -> Option<String> {
    match &hir.node(id).op {
        HirOp::Param { name } => Some(name.clone()),
        _ => None,
    }
}

/// Replace 6-input `onnx.QMatMul` with 4-input baked-weight variant when companions exist.
/// Reuses the weight `Param` node id (now f32 baked) so HIR lower order stays valid.
pub fn fuse_qmatmul_baked_weights(hir: &mut HirModule, baked: &HashMap<String, Vec<f32>>) -> usize {
    if baked.is_empty() {
        return 0;
    }
    let ids: Vec<HirNodeId> = hir.nodes().iter().map(|n| n.id).collect();
    let mut fused = 0usize;
    for id in ids {
        let node = hir.node(id);
        let HirOp::Mir(Op::Custom {
            name, num_inputs, ..
        }) = &node.op
        else {
            continue;
        };
        if name != crate::kernels::Q_MATMUL || *num_inputs != 6 {
            continue;
        }
        let inputs = node.inputs.clone();
        if inputs.len() < 6 {
            continue;
        }
        let Some(w_name) = resolve_param_name(hir, inputs[3]) else {
            continue;
        };
        let baked_key = baked_param_name(&w_name);
        let Some(w_f32) = baked.get(&baked_key) else {
            continue;
        };
        let k = hir.node(inputs[3]).shape.dim(0).unwrap_static().max(1);
        let n = w_f32
            .len()
            .checked_div(k)
            .unwrap_or_else(|| hir.node(inputs[3]).shape.dim(1).unwrap_static().max(1));
        let w_node = hir.node_mut(inputs[3]);
        w_node.op = HirOp::Param {
            name: baked_key.clone(),
        };
        w_node.shape = Shape::new(&[k.max(1), n.max(1)], DType::F32);

        let node_mut = hir.node_mut(id);
        node_mut.op = HirOp::Mir(Op::Custom {
            name: Q_MATMUL_BAKED.to_string(),
            num_inputs: 4,
            attrs: vec![],
        });
        node_mut.inputs = vec![inputs[0], inputs[1], inputs[2], inputs[3]];
        fused += 1;
    }
    fused
}
