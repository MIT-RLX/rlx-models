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

//! Second-half query_1: DQL uint8 activations and QMatMul output in one run.

use kitten_tts_mini_rlx::GraphOptions;
use kitten_tts_mini_rlx::bundle_compile::{
    compile_multi_probe_graph, import_from_bundle_cached, probe_output_f32_at, run_parity_inputs,
};
use rlx_ir::HirOp;
use rlx_runtime::Device;

const QMM: &str =
    "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/query_1/MatMul_quant_f32";

#[test]
fn query1_qmatmul_uses_dql_uint8_path_in_one_run() {
    let bundle_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    if !bundle_dir.join("manifest.json").exists() {
        return;
    }
    let opts = GraphOptions {
        sequence_length: 8,
        max_waveform_samples: 24_000,
    };
    let import = import_from_bundle_cached(&bundle_dir, &opts).expect("import");
    let find = |name: &str| {
        import
            .hir
            .nodes()
            .iter()
            .find(|n| n.name.as_deref() == Some(name))
            .map(|n| n.id)
            .unwrap_or_else(|| panic!("missing {name}"))
    };
    let qmm = find(QMM);
    assert!(
        matches!(
            &import.hir.node(qmm).op,
            HirOp::Mir(rlx_ir::Op::Custom { name, .. }) if name == "onnx.QMatMul"
        ),
        "expected onnx.QMatMul"
    );
    assert_eq!(import.hir.node(qmm).inputs.len(), 6);
    let act_q = import.hir.node(qmm).inputs[0];
    let probes = [(act_q, "act_q"), (qmm, QMM)];
    let mut graph = compile_multi_probe_graph(Device::Cpu, &bundle_dir, &opts, &import, &probes)
        .expect("compile");
    let ids: Vec<i64> = vec![0, 50, 83, 156, 54, 57, 135, 0];
    let style = vec![0.0f32; 256];
    kitten_tts_mini_rlx::set_env_var("KITTEN_RLX_SKIP_FUSION", "1");
    kitten_tts_mini_rlx::opts::set_compile_sequence_length(8);
    let outs = run_parity_inputs(&mut graph, 8, &ids, &style);
    let _act_q_bytes = outs.first().expect("act_q").0.clone();
    let q_vals = probe_output_f32_at(&outs, 1).expect("qmm");
    let idx = 3576usize;
    eprintln!("query1 dql-path idx3576 q={}", q_vals[idx]);
    assert!(q_vals[idx].is_finite());
}
