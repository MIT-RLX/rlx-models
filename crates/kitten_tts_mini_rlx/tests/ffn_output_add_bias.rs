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

//! Bias param must be non-zero and Add must include it.

use kitten_tts_mini_rlx::GraphOptions;
use kitten_tts_mini_rlx::bundle_compile::{
    compile_probe_graph, import_from_bundle_cached, probe_output_f32_at, run_parity_inputs,
};
use rlx_runtime::Device;

#[test]
fn ffn_output_bias_param_loaded() {
    let bundle_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    if !bundle_dir.join("manifest.json").exists() {
        return;
    }
    let import = import_from_bundle_cached(
        &bundle_dir,
        &GraphOptions {
            sequence_length: 8,
            max_waveform_samples: 24_000,
        },
    )
    .expect("import");
    let key = "kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.bias";
    let bias = import.params.get(key).expect("bias in params");
    assert_eq!(bias.len(), 768);
    eprintln!("bias[504]={}", bias[504]);
    assert!(
        bias[504].abs() > 1.0,
        "bias[504] should be ~-2.59, got {}",
        bias[504]
    );
}

#[test]
fn ffn_output_add_applies_bias_after_warmup() {
    let bundle_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    if !bundle_dir.join("manifest.json").exists() {
        return;
    }
    let opts = GraphOptions {
        sequence_length: 8,
        max_waveform_samples: 24_000,
    };
    let import = import_from_bundle_cached(&bundle_dir, &opts).expect("import");
    kitten_tts_mini_rlx::set_env_var("KITTEN_RLX_SKIP_FUSION", "1");
    kitten_tts_mini_rlx::opts::set_compile_sequence_length(8);
    let ids: Vec<i64> = vec![0, 50, 83, 156, 54, 57, 135, 0];
    let style = vec![0.0f32; 256];
    // Warmup chain for reliable ffn_output values.
    for name in [
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/LayerNorm/LayerNormalization",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/ffn/MatMul_quant_f32",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/activation/Mul_3",
    ] {
        let id = import
            .hir
            .nodes()
            .iter()
            .find(|n| n.name.as_deref() == Some(name))
            .map(|n| n.id)
            .unwrap_or_else(|| panic!("missing {name}"));
        let mut g = compile_probe_graph(Device::Cpu, &bundle_dir, &opts, &import, id, name)
            .expect("warmup");
        let _ = run_parity_inputs(&mut g, 8, &ids, &style);
    }
    let mm_name = "/bert/encoder/albert_layer_groups.0/albert_layers.0/ffn_output/MatMul_quant_f32";
    let add_name = "/bert/encoder/albert_layer_groups.0/albert_layers.0/ffn_output/Add";
    let find = |name: &str| {
        import
            .hir
            .nodes()
            .iter()
            .find(|n| n.name.as_deref() == Some(name))
            .map(|n| n.id)
            .unwrap()
    };
    let mut mm_g = compile_probe_graph(
        Device::Cpu,
        &bundle_dir,
        &opts,
        &import,
        find(mm_name),
        mm_name,
    )
    .expect("mm");
    let mut add_g = compile_probe_graph(
        Device::Cpu,
        &bundle_dir,
        &opts,
        &import,
        find(add_name),
        add_name,
    )
    .expect("add");
    let mm = probe_output_f32_at(&run_parity_inputs(&mut mm_g, 8, &ids, &style), 0).unwrap();
    let add = probe_output_f32_at(&run_parity_inputs(&mut add_g, 8, &ids, &style), 0).unwrap();
    let bias = import
        .params
        .get("kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.bias")
        .unwrap();
    let idx = 1272usize;
    let dim = idx % 768;
    eprintln!(
        "mm={} bias={} add={} expected={}",
        mm[idx],
        bias[dim],
        add[idx],
        mm[idx] + bias[dim]
    );
    assert!(
        (add[idx] - (mm[idx] + bias[dim])).abs() < 1e-3,
        "Add should equal QMatMul + bias[dim]; got {} expected {}",
        add[idx],
        mm[idx] + bias[dim]
    );
}
