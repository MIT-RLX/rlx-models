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

//! ffn_output/Add should equal QMatMul output + bias in one run.

use kitten_tts_mini_rlx::GraphOptions;
use kitten_tts_mini_rlx::bundle_compile::{
    compile_multi_probe_graph, import_from_bundle_cached, probe_output_f32_at, run_parity_inputs,
};
use rlx_runtime::Device;

const FFN_OUT_MM: &str =
    "/bert/encoder/albert_layer_groups.0/albert_layers.0/ffn_output/MatMul_quant_f32";
const FFN_OUT_ADD: &str = "/bert/encoder/albert_layer_groups.0/albert_layers.0/ffn_output/Add";

#[test]
fn ffn_output_add_includes_bias_in_one_run() {
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
    let mm = find(FFN_OUT_MM);
    let add = find(FFN_OUT_ADD);
    let add_node = import.hir.node(add);
    let bias_id = add_node.inputs[0];
    let bias_name = import.hir.node(bias_id).name.clone().unwrap_or_default();
    eprintln!("ffn_output Add bias node: {bias_name}");
    let probes = [(mm, FFN_OUT_MM), (add, FFN_OUT_ADD)];
    let mut graph = compile_multi_probe_graph(Device::Cpu, &bundle_dir, &opts, &import, &probes)
        .expect("compile");
    let ids: Vec<i64> = vec![0, 50, 83, 156, 54, 57, 135, 0];
    kitten_tts_mini_rlx::set_env_var("KITTEN_RLX_SKIP_FUSION", "1");
    kitten_tts_mini_rlx::opts::set_compile_sequence_length(8);
    let outs = run_parity_inputs(&mut graph, 8, &ids, &vec![0.0; 256]);
    let mm_v = probe_output_f32_at(&outs, 0).expect("mm");
    let add_v = probe_output_f32_at(&outs, 1).expect("add");
    let idx = 1272usize;
    let diff = (add_v[idx] - mm_v[idx]).abs();
    eprintln!(
        "idx {idx}: mm={} add={} delta={}",
        mm_v[idx], add_v[idx], diff
    );
    assert!(
        diff > 1e-3,
        "ffn_output Add should differ from QMatMul by bias at idx {idx}"
    );
}
