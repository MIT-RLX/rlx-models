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

//! Verify word embedding Gather sees the runtime `input_ids` values.

use kitten_tts_mini_rlx::GraphOptions;
use kitten_tts_mini_rlx::bundle_compile::{
    compile_probe_graph, import_from_bundle_cached, set_runtime_input_ids_shape,
};
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

#[test]
fn minimal_gather_i64_session_smoke() {
    let mut g = Graph::new("gather_i64");
    let table = g.param("emb", Shape::new(&[178, 128], DType::F32));
    let ids = g.input("input_ids", Shape::new(&[1, 8], DType::I64));
    let out = g.gather_(table, ids, 0);
    g.set_outputs(vec![out]);
    let mut table_data = vec![0.0f32; 178 * 128];
    for i in 0..178 {
        table_data[i * 128] = i as f32;
    }
    let mut exec = Session::new(Device::Cpu).compile(g);
    exec.set_param("emb", &table_data);
    let indices: [i64; 8] = [0, 50, 83, 156, 54, 57, 135, 0];
    let ids_bytes: Vec<u8> = indices.iter().flat_map(|v| v.to_le_bytes()).collect();
    let outs = exec.run_typed(&[("input_ids", &ids_bytes, DType::I64)]);
    let emb: Vec<f32> = outs[0]
        .0
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert!((emb[0] - 0.0).abs() < 1e-5);
    assert!((emb[128] - 50.0).abs() < 1e-5);
}

#[test]
fn word_embedding_gather_matches_weight_rows() {
    let bundle_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    if !bundle_dir.join("graph.json").is_file() {
        return;
    }

    let seq = 8usize;
    let ids: [i64; 8] = [0, 50, 83, 156, 54, 57, 135, 0];
    let opts = GraphOptions {
        sequence_length: seq,
        max_waveform_samples: 12_000,
    };
    let import = import_from_bundle_cached(&bundle_dir, &opts).expect("import");

    let gather = import
        .hir
        .nodes()
        .iter()
        .find(|n| n.name.as_deref() == Some("/bert/embeddings/word_embeddings/Gather"))
        .expect("gather node");
    eprintln!(
        "gather inputs in HIR: {:?}",
        import
            .hir
            .node(gather.id)
            .inputs
            .iter()
            .map(|&id| {
                let n = import.hir.node(id);
                (n.name.clone(), n.op.clone(), n.shape.dims().to_vec())
            })
            .collect::<Vec<_>>()
    );
    eprintln!(
        "input_ids input shape (from manifest): {:?}",
        import
            .hir
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, rlx_ir::hir::HirOp::Input { name } if name == "input_ids"))
            .map(|n| n.shape.dims().to_vec())
    );
    eprintln!("gather out shape {:?}", gather.shape.dims());

    let mut graph = compile_probe_graph(
        Device::Cpu,
        &bundle_dir,
        &opts,
        &import,
        gather.id,
        "/bert/embeddings/word_embeddings/Gather",
    )
    .expect("probe");
    set_runtime_input_ids_shape(&mut graph, seq).expect("shape");

    let ids_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let style = vec![0.0f32; 256];
    let style_bytes: Vec<u8> = style.iter().flat_map(|v| v.to_le_bytes()).collect();
    let speed_bytes: Vec<u8> = 1.0f32.to_le_bytes().to_vec();

    kitten_tts_mini_rlx::opts::set_compile_sequence_length(seq);

    let outs = graph.run_typed(&[
        ("input_ids", ids_bytes.as_slice(), rlx_ir::DType::I64),
        ("style", style_bytes.as_slice(), rlx_ir::DType::F32),
        ("speed", speed_bytes.as_slice(), rlx_ir::DType::F32),
    ]);
    let (emb_bytes, _) = outs.first().expect("probe output");
    let emb: Vec<f32> = emb_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(emb.len(), seq * 128);

    let weight_bytes = std::fs::read(bundle_dir.join("weights.safetensors")).expect("weights");
    let weights = safetensors::SafeTensors::deserialize(&weight_bytes).expect("safetensors");
    let w = weights
        .tensor("kmodel.bert.embeddings.word_embeddings.weight")
        .expect("emb weight");
    let w: Vec<f32> = w
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let cols = 128usize;

    for t in 0..seq {
        let expected_id = ids[t] as usize;
        let got = &emb[t * cols..(t + 1) * cols];
        let row = &w[expected_id * cols..(expected_id + 1) * cols];
        let l2: f32 = got
            .iter()
            .zip(row)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        eprintln!(
            "token {t} id={} L2={l2:.6} got0={} row0={}",
            ids[t], got[0], row[0]
        );
        assert!(l2 < 1e-4, "token {t} id={} gather mismatch L2={l2}", ids[t]);
    }
}
