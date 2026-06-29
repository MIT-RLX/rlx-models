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

//! Dump post-fusion MLX graph for qwen35 tiny config.

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn dump_mlx_fused_graph() {
    mod compile_support;

    use rlx_ir::Op;
    use rlx_models::build_qwen35_graph_sized;
    use rlx_models::qwen35::synth;
    use rlx_runtime::{CompileOptions, Device};

    let cfg = synth::tiny_cfg();
    let weights = synth::synth_weights(&cfg);
    let (graph, params, _) =
        build_qwen35_graph_sized(&cfg, weights, 1, 4, true, true, true).expect("build");

    eprintln!("=== Rope nodes (pre-compile) ===");
    for node in graph.nodes() {
        if let Op::Rope {
            head_dim, n_rot, ..
        } = &node.op
        {
            let x = graph.node(node.inputs[0]).shape.dims();
            let c = graph.node(node.inputs[1]).shape.dims();
            let s = graph.node(node.inputs[2]).shape.dims();
            eprintln!(
                "Rope id={} hd={head_dim} n_rot={n_rot} x={x:?} cos={c:?} sin={s:?} x_in={} cos_in={} sin_in={}",
                node.id.0, node.inputs[0].0, node.inputs[1].0, node.inputs[2].0,
            );
        }
    }

    use rlx_ir::OpKind;
    let supported: &[OpKind] = &[
        OpKind::Input,
        OpKind::Param,
        OpKind::Constant,
        OpKind::Activation,
        OpKind::Cast,
        OpKind::Binary,
        OpKind::Compare,
        OpKind::Where,
        OpKind::ElementwiseRegion,
        OpKind::MatMul,
        OpKind::DotGeneral,
        OpKind::LayerNorm,
        OpKind::RmsNorm,
        OpKind::Attention,
        OpKind::Rope,
        OpKind::Reshape,
        OpKind::Transpose,
        OpKind::Narrow,
        OpKind::Concat,
        OpKind::Expand,
        OpKind::Gather,
        OpKind::Reduce,
        OpKind::Softmax,
        OpKind::Cumsum,
        OpKind::TopK,
        OpKind::Sample,
        OpKind::Conv,
        OpKind::Pool,
        OpKind::GroupedMatMul,
        OpKind::ScatterAdd,
        OpKind::LoraMatMul,
        OpKind::DequantMatMul,
        OpKind::SelectiveScan,
        OpKind::GatedDeltaNet,
        OpKind::FusedSwiGLU,
        OpKind::FusedMatMulBiasAct,
        OpKind::FusedResidualLN,
        OpKind::FusedResidualRmsNorm,
        OpKind::FusedAttentionBlock,
        OpKind::FusedTransformerLayer,
    ];
    let rewritten = rlx_opt::rewrite_for_backend(graph.clone(), supported);
    let result = rlx_runtime::stages::compile_graph_stages_for_backend(
        Device::Mlx,
        rewritten,
        &CompileOptions::new(),
        supported,
    );
    let fused = result.lir.into_graph();
    for node in fused.nodes() {
        if node.id.0 >= 1 && node.id.0 <= 20 {
            let dims: Vec<_> = node
                .shape
                .dims()
                .iter()
                .map(|d| d.unwrap_static())
                .collect();
            let ins: Vec<_> = node.inputs.iter().map(|i| i.0).collect();
            eprintln!(
                "id={} op={:?} dims={dims:?} ins={ins:?}",
                node.id.0, node.op
            );
        }
    }

    eprintln!("=== post-fusion ops ===");
    for node in fused.nodes() {
        let dims: Vec<_> = node
            .shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        if dims.iter().any(|&d| d == 34)
            || matches!(
                node.op,
                Op::Rope { .. }
                    | Op::FusedAttentionBlock { .. }
                    | Op::FusedSwiGLU { .. }
                    | Op::GatedDeltaNet { .. }
            )
        {
            let ins: Vec<_> = node
                .inputs
                .iter()
                .map(|i| {
                    let n = &fused.node(*i);
                    let d: Vec<_> = n.shape.dims().iter().map(|x| x.unwrap_static()).collect();
                    format!("{}:{:?}", i.0, d)
                })
                .collect();
            eprintln!(
                "id={} op={:?} dims={dims:?} ins=[{}]",
                node.id.0,
                node.op,
                ins.join(", ")
            );
        }
    }

    let _compiled = compile_support::compile_qwen35_prefill(Device::Mlx, graph, params);
    eprintln!("compile finished");
}
