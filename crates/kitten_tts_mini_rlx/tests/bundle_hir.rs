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

// Ported from rlx-onnx-import/tests/kitten_bundle.rs (model-specific HIR checks).

mod common;

use common::{build_hir, bundle_dir, load_bundle, opts_seq8};
use rlx_onnx_import::coverage::op_is_supported;
use rlx_onnx_import::{ImportOptions, load_bundle as load_rlx_bundle};

#[test]
fn kitten_mini_bundle_op_coverage_100() {
    let dir = bundle_dir();
    if !dir.join("manifest.json").exists() {
        eprintln!("skip: {}", dir.display());
        return;
    }
    let bundle = load_bundle().expect("load bundle");
    let mut unsupported = Vec::new();
    let mut total = 0usize;
    for (op, count) in &bundle.manifest.op_histogram {
        total += count;
        if !op_is_supported(op) {
            unsupported.push((op.clone(), *count));
        }
    }
    if !unsupported.is_empty() {
        eprintln!("unsupported ops: {unsupported:?}");
    }
    assert!(
        unsupported.is_empty(),
        "expected 100% op coverage, missing: {unsupported:?}"
    );
    assert_eq!(total, bundle.manifest.node_count);
}

#[test]
fn kitten_mini_bundle_builds_hir() {
    let dir = bundle_dir();
    if !dir.join("manifest.json").exists() {
        eprintln!("skip: {}", dir.display());
        return;
    }
    let bundle = load_bundle().expect("load bundle");
    let (hir, _params, _typed, report) =
        build_hir(&bundle, ImportOptions::quant_bundle()).expect("lower");
    let total = report.lowered + report.skipped;
    let pct = 100.0 * report.lowered as f64 / total as f64;
    eprintln!(
        "kitten HIR: lowered={} skipped={} ({pct:.1}%) unsupported={:?}",
        report.lowered, report.skipped, report.unsupported
    );
    assert!(pct >= 99.0, "expected >=99% nodes lowered, got {pct:.1}%");
    assert!(report.unsupported.is_empty(), "{:?}", report.unsupported);
    assert!(!hir.outputs.is_empty());
}

#[test]
fn kitten_hir_binary_broadcast_check() {
    use rlx_ir::HirOp;
    let dir = bundle_dir();
    if !dir.join("manifest.json").exists() {
        return;
    }
    let bundle = load_bundle().expect("load");
    let (hir, _p, _typed, _) = build_hir(&bundle, ImportOptions::quant_bundle()).expect("hir");
    let mut bad = 0usize;
    for node in hir.nodes() {
        if let HirOp::Mir(rlx_ir::Op::Binary(_)) = &node.op {
            if node
                .name
                .as_deref()
                .is_some_and(|n| n.starts_with("/decoder/"))
            {
                continue;
            }
            if node.inputs.len() < 2 {
                continue;
            }
            let sa = &hir.node(node.inputs[0]).shape;
            let sb = &hir.node(node.inputs[1]).shape;
            let da: Vec<_> = sa.dims().iter().map(|d| d.unwrap_static()).collect();
            let db: Vec<_> = sb.dims().iter().map(|d| d.unwrap_static()).collect();
            let dout: Vec<_> = node
                .shape
                .dims()
                .iter()
                .map(|d| d.unwrap_static())
                .collect();
            let clash = |a: &[usize], b: &[usize]| -> Option<usize> {
                let rank = a.len().max(b.len());
                for ax in 0..rank {
                    let ai = if ax + a.len() > rank {
                        a[ax + a.len() - rank]
                    } else {
                        1
                    };
                    let bi = if ax + b.len() > rank {
                        b[ax + b.len() - rank]
                    } else {
                        1
                    };
                    if ai != 1 && bi != 1 && ai != bi {
                        return Some(ax);
                    }
                }
                None
            };
            if let Some(ax) = clash(&da, &db) {
                bad += 1;
                eprintln!(
                    "BAD {:?} axis={ax} a={da:?} b={db:?} out={dout:?}",
                    node.name.as_deref().unwrap_or("?")
                );
            }
        }
    }
    assert_eq!(bad, 0, "found {bad} binary ops with 512 vs 128 on axis 1");
}

#[test]
fn kitten_duration_hir_shapes_seq8() {
    let dir = bundle_dir();
    if !dir.join("manifest.json").exists() {
        return;
    }
    let opts = opts_seq8();
    let (hir, _, _, _) = build_hir(&load_bundle().unwrap(), opts).unwrap();
    let mir = hir.clone().lower_to_mir().unwrap();
    let hg = mir.as_graph();
    for node in hg.nodes() {
        if node.name.as_deref().is_some_and(|n| {
            n.contains("ReduceSum")
                || n == "/Div"
                || n == "/Squeeze"
                || n == "/Round"
                || n == "/Clip"
                || n == "/Cast_3"
        }) {
            eprintln!(
                "dur seq8 HIR {:?} {:?} elems={:?}",
                node.name,
                node.shape.dims(),
                node.shape.num_elements()
            );
        }
    }
    for (i, &oid) in hg.outputs.iter().enumerate() {
        let n = hg.node(oid);
        eprintln!("HIR.out[{i}] {:?} {:?}", n.name, n.shape.dims());
    }
    assert!(
        !hir.outputs.is_empty(),
        "duration path HIR should have outputs"
    );
}

#[test]
fn kitten_text_encoder_concat1_seq_first_seq8() {
    let dir = bundle_dir();
    if !dir.join("manifest.json").exists() {
        return;
    }
    let opts = opts_seq8();
    let (hir, _, _, _) = build_hir(&load_bundle().unwrap(), opts).unwrap();
    for want in [
        "/text_encoder_1/Transpose",
        "/text_encoder_1/Expand",
        "/text_encoder_1/Concat_1",
    ] {
        let node = hir
            .nodes()
            .iter()
            .find(|n| n.name.as_deref() == Some(want))
            .unwrap_or_else(|| panic!("missing {want}"));
        let dims: Vec<_> = node
            .shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        eprintln!("{want} HIR shape={dims:?}");
        if want == "/text_encoder_1/Concat_1" {
            assert_eq!(dims, [8, 1, 640], "{want} should be seq-first");
        }
    }
}

#[test]
fn kitten_albert_dense_matmul_uses_act_scale_epilogue() {
    let dir = bundle_dir();
    if !dir.join("manifest.json").exists() {
        return;
    }
    let opts = opts_seq8();
    let bundle = load_bundle().unwrap();
    let (hir, _, _, _) = build_hir(&bundle, opts).unwrap();
    let add = hir
        .nodes()
        .iter()
        .find(|n| {
            n.name.as_deref()
                == Some("/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/dense_1/Add")
        })
        .expect("dense_1 Add");
    let qmm = hir
        .nodes()
        .iter()
        .find(|n| {
            n.name.as_deref() == Some(
                "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/dense_1/MatMul_quant_f32",
            ) && matches!(&n.op, rlx_ir::HirOp::Mir(rlx_ir::Op::Custom { name, .. }) if name == "onnx.QMatMul")
        })
        .expect("dense_1 QMatMul");
    let add_in = hir.node(add.id).inputs[1];
    assert_eq!(
        add_in, qmm.id,
        "dense_1 Add should consume onnx.QMatMul output"
    );
}

#[test]
fn kitten_albert_query_matmul_bypasses_output_scale() {
    let dir = bundle_dir();
    if !dir.join("manifest.json").exists() {
        return;
    }
    let opts = opts_seq8();
    let bundle = load_bundle().unwrap();
    let (hir, _, _, _) = build_hir(&bundle, opts).unwrap();
    let add = hir
        .nodes()
        .iter()
        .find(|n| {
            n.name.as_deref()
                == Some("/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/query/Add")
        })
        .expect("query Add");
    let matmul = hir
        .nodes()
        .iter()
        .find(|n| {
            n.name.as_deref() == Some(
                "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/query/MatMul_quant_f32",
            )
        })
        .expect("query MatMul");
    let add_in = hir.node(add.id).inputs[1];
    assert_eq!(
        add_in, matmul.id,
        "embedding-fed query Add should consume f32 MatMul directly"
    );
}

#[test]
fn kitten_ffn_matmul_lowers_to_qmatmul() {
    let dir = bundle_dir();
    if !dir.join("manifest.json").exists() {
        return;
    }
    let opts = opts_seq8();
    let bundle = load_bundle().unwrap();
    let (hir, _, _, _) = build_hir(&bundle, opts).unwrap();
    let ffn = hir
        .nodes()
        .iter()
        .find(|n| {
            n.name.as_deref()
                == Some("/bert/encoder/albert_layer_groups.0/albert_layers.0/ffn/MatMul_quant_f32")
        })
        .expect("ffn MatMul");
    assert!(
        matches!(&ffn.op, rlx_ir::HirOp::Mir(rlx_ir::Op::Custom { name, .. }) if name == "onnx.QMatMul"),
        "ffn matmul should lower to onnx.QMatMul, got {:?}",
        ffn.op
    );
    let q1 = hir
        .nodes()
        .iter()
        .find(|n| {
            n.name.as_deref() == Some(
                "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/query_1/MatMul_quant_f32",
            )
        })
        .expect("query_1 MatMul");
    assert!(
        matches!(&q1.op, rlx_ir::HirOp::Mir(rlx_ir::Op::Custom { name, .. }) if name == "onnx.QMatMul"),
        "query_1 matmul should lower to onnx.QMatMul, got {:?}",
        q1.op
    );
    assert_eq!(
        hir.node(ffn.id).inputs.len(),
        6,
        "fused QMatMul should take DQL act q/scale/zp + weight q/scale/zp"
    );
    let ffn_matmuls: Vec<_> = hir
        .nodes()
        .iter()
        .filter(|n| {
            n.name
                .as_deref()
                .is_some_and(|s| s.contains("/ffn/MatMul") && s.contains("quant_f32"))
        })
        .collect();
    assert_eq!(ffn_matmuls.len(), 1, "expected one ffn quant matmul node");
}

#[test]
fn kitten_qkv_each_use_distinct_dql_quantized() {
    let dir = bundle_dir();
    if !dir.join("manifest.json").exists() {
        return;
    }
    let opts = opts_seq8();
    let bundle = load_bundle().unwrap();
    let (hir, _, _, _) = build_hir(&bundle, opts).unwrap();
    for qkv in ["query_1", "key_1", "value_1"] {
        let mm = hir
            .nodes()
            .iter()
            .find(|n| {
                n.name.as_deref() == Some(&format!(
                    "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/{qkv}/MatMul_quant_f32"
                ))
            })
            .expect(qkv);
        assert!(
            matches!(&mm.op, rlx_ir::HirOp::Mir(rlx_ir::Op::Custom { name, .. }) if name == "onnx.QMatMul"),
            "{qkv} should lower to onnx.QMatMul, got {:?}",
            mm.op
        );
        let act_q = hir.node(mm.id).inputs[0];
        let dql = hir.node(act_q);
        assert!(
            matches!(
                &dql.op,
                rlx_ir::HirOp::Mir(rlx_ir::Op::Custom { name, .. }) if name == "onnx.DynamicQuantizeLinearExport"
            ),
            "{qkv} act input should be DynamicQuantizeLinearExport, got {:?}",
            dql.op
        );
    }
}

#[test]
fn kitten_bert_encoder_matmul_uses_act_scale_epilogue() {
    let dir = bundle_dir();
    if !dir.join("manifest.json").exists() {
        return;
    }
    let opts = opts_seq8();
    let bundle = load_bundle().unwrap();
    let (hir, _, _, _) = build_hir(&bundle, opts).unwrap();
    let add = hir
        .nodes()
        .iter()
        .find(|n| n.name.as_deref() == Some("/bert_encoder/Add"))
        .expect("bert_encoder Add");
    let qmm = hir
        .nodes()
        .iter()
        .find(|n| {
            n.name.as_deref() == Some("/bert_encoder/MatMul_quant_f32")
                && matches!(&n.op, rlx_ir::HirOp::Mir(rlx_ir::Op::Custom { name, .. }) if name == "onnx.QMatMul")
        })
        .expect("bert_encoder QMatMul");
    let add_in = hir.node(add.id).inputs[1];
    assert_eq!(add_in, qmm.id);
}

#[test]
fn kitten_text_encoder_lstms5_fc_uses_qmatmul() {
    let dir = bundle_dir();
    if !dir.join("manifest.json").exists() {
        return;
    }
    let opts = opts_seq8();
    let bundle = load_bundle().unwrap();
    let (hir, _, _, _) = build_hir(&bundle, opts).unwrap();
    let add = hir
        .nodes()
        .iter()
        .find(|n| n.name.as_deref() == Some("/text_encoder/lstms.5/fc/Gemm_Add"))
        .expect("lstms.5 Gemm_Add");
    let qmm = hir
        .nodes()
        .iter()
        .find(|n| {
            n.name.as_deref() == Some("/text_encoder/lstms.5/fc/Gemm_MatMul_quant_f32")
                && matches!(
                    &n.op,
                    rlx_ir::HirOp::Mir(rlx_ir::Op::Custom { name, .. }) if name == "onnx.QMatMul"
                )
        })
        .expect("lstms.5 Gemm QMatMul");
    let add_in = hir.node(add.id).inputs[0];
    assert_eq!(
        add_in, qmm.id,
        "lstms.5 fc Add should consume onnx.QMatMul output"
    );
}

#[test]
fn kitten_text_encoder_where4_broadcasts_cond_seq8() {
    let dir = bundle_dir();
    if !dir.join("manifest.json").exists() {
        return;
    }
    let opts = opts_seq8();
    let (hir, _, _, _) = build_hir(&load_bundle().unwrap(), opts).unwrap();
    let where4 = hir
        .nodes()
        .iter()
        .find(|n| n.name.as_deref() == Some("/text_encoder_1/Where_4"))
        .expect("Where_4");
    let dims: Vec<_> = where4
        .shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    assert_eq!(dims, [1, 8, 640], "Where_4 output shape");
    assert!(matches!(&where4.op, rlx_ir::HirOp::Mir(rlx_ir::Op::Where)));
    for &inp in &hir.node(where4.id).inputs {
        let in_dims: Vec<_> = hir
            .node(inp)
            .shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        assert_eq!(
            in_dims,
            [1, 8, 640],
            "Where_4 inputs should be broadcast-expanded"
        );
    }
}

#[test]
fn kitten_bert_attention_mask_shapes_seq8() {
    let dir = bundle_dir();
    if !dir.join("manifest.json").exists() {
        return;
    }
    let opts = opts_seq8();
    let bundle = load_bundle().unwrap();
    let (hir, _, _, _) = build_hir(&bundle, opts).unwrap();
    let mask_shape = |name: &str| -> Vec<usize> {
        hir.nodes()
            .iter()
            .find(|n| n.name.as_deref() == Some(name))
            .map(|n| n.shape.dims().iter().map(|d| d.unwrap_static()).collect())
            .unwrap_or_default()
    };
    assert_eq!(
        mask_shape("/bert/Expand_1"),
        vec![1, 1, 8, 8],
        "Expand_1 should broadcast attention mask to [1,1,seq,seq]"
    );
    assert_eq!(
        mask_shape("/bert/Where_2"),
        vec![1, 1, 8, 8],
        "Where_2 attention mask should be [1,1,seq,seq]"
    );
}

#[test]
fn kitten_albert_attention_head_dim_slice() {
    let dir = bundle_dir();
    if !dir.join("manifest.json").exists() {
        return;
    }
    let opts = opts_seq8();
    let bundle = load_bundle().unwrap();
    let (hir, _, _, _) = build_hir(&bundle, opts).unwrap();
    let cast = hir
        .nodes()
        .iter()
        .find(|n| {
            n.name.as_deref()
                == Some("/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/Cast")
        })
        .expect("attention Cast");
    let slice = hir
        .nodes()
        .iter()
        .find(|n| {
            n.name.as_deref()
                == Some("/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/Slice")
        })
        .expect("attention Slice");
    assert_eq!(
        hir.node(cast.id).inputs[0],
        slice.id,
        "Cast should consume Slice output for head-dim scale"
    );
}

#[test]
fn kitten_style_slice_not_zero_stub() {
    let dir = bundle_dir();
    if !dir.join("manifest.json").exists() {
        return;
    }
    let opts = opts_seq8();
    let (_, params, _, _) = build_hir(&load_bundle().unwrap(), opts).unwrap();
    assert!(
        !params.contains_key("__stub__//Slice_1_output_0"),
        "/Slice_1 must narrow style, not zero stub"
    );
}

#[test]
fn kitten_text_encoder_lstm_reshape_seq_first_seq8() {
    let dir = bundle_dir();
    if !dir.join("manifest.json").exists() {
        return;
    }
    let opts = opts_seq8();
    let (hir, _, _, _) = build_hir(&load_bundle().unwrap(), opts).unwrap();
    for want in [
        "/text_encoder/lstms.2/Reshape",
        "/text_encoder/lstm/Reshape",
        "/lstm/Reshape",
    ] {
        let node = hir
            .nodes()
            .iter()
            .find(|n| n.name.as_deref() == Some(want))
            .unwrap_or_else(|| panic!("missing {want}"));
        let dims: Vec<_> = node
            .shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        assert_eq!(dims, [8, 1, 512], "{want} shape");
    }
    let lstms1 = hir
        .nodes()
        .iter()
        .find(|n| n.name.as_deref() == Some("/text_encoder/lstms.1/Reshape"))
        .expect("lstms.1 Reshape");
    let dims: Vec<_> = lstms1
        .shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    assert_eq!(dims, [1, 1024, 1], "lstms.1 fc reshape");
}

#[test]
fn kitten_lstm_duration_path_shapes_seq8() {
    let dir = bundle_dir();
    if !dir.join("manifest.json").exists() {
        return;
    }
    let opts = opts_seq8();
    let (hir, _, _, _) = build_hir(&load_bundle().unwrap(), opts).unwrap();
    for node in hir.nodes() {
        let name = node.name.as_deref().unwrap_or("");
        if name.contains("lstm/Reshape")
            || (name.contains("text_encoder/lstms.") && name.ends_with("/Reshape"))
            || name == "/text_encoder/lstm/Reshape"
            || name.contains("lstm/Transpose")
            || name.contains("text_encoder_1/Where_4")
            || name.contains("Transpose_2_output_0_Cast")
            || name.contains("duration_proj")
            || name == "/Sigmoid"
            || name == "/ReduceSum"
        {
            eprintln!(
                "HIR {:?} shape={:?} dtype={:?}",
                name,
                node.shape.dims(),
                node.shape.dtype()
            );
        }
    }
}

#[test]
fn kitten_bundle_compiles_cpu_with_fusion() {
    use kitten_tts_mini_rlx::GraphOptions;
    use rlx_runtime::Device;
    let dir = bundle_dir();
    if !dir.join("manifest.json").exists() {
        eprintln!("skip: {}", dir.display());
        return;
    }
    let graph_opts = GraphOptions {
        sequence_length: 8,
        max_waveform_samples: 24_000,
    };
    kitten_tts_mini_rlx::bundle_compile::compile_from_bundle_fresh(Device::Cpu, &dir, &graph_opts)
        .expect("compile bundle to CPU");
}

#[test]
fn kitten_lstms5_slice_lowers_to_narrow_not_stub() {
    let dir = bundle_dir();
    if !dir.join("manifest.json").exists() {
        return;
    }
    let opts = opts_seq8();
    let (hir, _, _, _) = build_hir(&load_bundle().unwrap(), opts).unwrap();
    for name in [
        "/text_encoder/lstms.5/Slice",
        "/text_encoder/lstms.5/Slice_1",
    ] {
        let node = hir
            .nodes()
            .iter()
            .find(|n| n.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("missing {name}"));
        assert!(
            matches!(
                &node.op,
                rlx_ir::HirOp::Mir(rlx_ir::Op::Narrow { .. })
                    | rlx_ir::HirOp::Mir(rlx_ir::Op::Reshape { .. })
            ),
            "{name} should narrow/reshape, got {:?}",
            node.op
        );
    }
}

#[test]
fn kitten_lstm_reshape_meta_after_propagate() {
    use rlx_onnx_import::shape_propagate::propagate_shapes;
    let dir = bundle_dir();
    if !dir.join("manifest.json").exists() {
        return;
    }
    let mut bundle = load_rlx_bundle(&bundle_dir()).unwrap();
    let opts = rlx_onnx_import::ImportOptions {
        sequence_length: 8,
        max_waveform_samples: 24000,
        ..Default::default()
    };
    let init_shapes = std::collections::HashMap::new();
    propagate_shapes(&mut bundle.nodes, &bundle.manifest, &init_shapes, &opts);
    for n in &bundle.nodes {
        if n.name == "/lstm/Reshape" || n.name == "/lstm/Transpose_2" {
            eprintln!("propagated {:?} meta={:?}", n.name, n.output_meta);
        }
    }
}
