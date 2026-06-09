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

//! First-half attention residual: embedding + dense should equal Add_1 in one run.

use kitten_tts_mini_rlx::GraphOptions;
use kitten_tts_mini_rlx::bundle_compile::{
    compile_multi_probe_graph, import_from_bundle_cached, probe_output_f32_at, run_parity_inputs,
};
use rlx_runtime::Device;

const ADD1: &str = "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/Add_1";
const EMB: &str = "/bert/encoder/embedding_hidden_mapping_in/Add";
const DENSE: &str = "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/dense/Add";

#[test]
fn attention_add1_matches_sum_of_inputs_in_one_run() {
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
    let add1 = find(ADD1);
    let emb = find(EMB);
    let dense = find(DENSE);
    let probes = [(emb, EMB), (dense, DENSE), (add1, ADD1)];
    let mut graph = compile_multi_probe_graph(Device::Cpu, &bundle_dir, &opts, &import, &probes)
        .expect("compile");
    let ids: Vec<i64> = vec![0, 50, 83, 156, 54, 57, 135, 0];
    let style = vec![0.0f32; 256];
    kitten_tts_mini_rlx::set_env_var("KITTEN_RLX_SKIP_FUSION", "1");
    kitten_tts_mini_rlx::opts::set_compile_sequence_length(8);
    let outs = run_parity_inputs(&mut graph, 8, &ids, &style);
    let a = probe_output_f32_at(&outs, 0).expect("emb");
    let b = probe_output_f32_at(&outs, 1).expect("dense");
    let sum = probe_output_f32_at(&outs, 2).expect("add1");
    let mut max = 0.0f32;
    let mut idx = 0usize;
    for (j, (&x, (&u, &v))) in sum.iter().zip(a.iter().zip(b.iter())).enumerate() {
        let d = (x - (u + v)).abs();
        if d > max {
            max = d;
            idx = j;
        }
    }
    eprintln!(
        "Add_1 coexec max |sum-(a+b)|={max} at idx={idx}: add1={} a+b={} a={} b={}",
        sum[idx],
        a[idx] + b[idx],
        a[idx],
        b[idx]
    );
    assert!(
        max < 1e-4,
        "Add_1 should equal embedding + dense in one run, max diff {max} at {idx}"
    );
}
