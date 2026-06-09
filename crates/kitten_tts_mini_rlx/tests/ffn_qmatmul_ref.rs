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

//! ffn QMatMul: native LN → reference qmatmul vs native probe.

use kitten_tts_mini_rlx::GraphOptions;
use kitten_tts_mini_rlx::bundle_compile::{
    compile_probe_graph, import_from_bundle_cached, probe_output_f32_at, run_parity_inputs,
};
use kitten_tts_mini_rlx::qmatmul::qmatmul_f32_act_i8_weight;
use rlx_runtime::Device;
use std::process::Command;

#[test]
fn ffn_qmatmul_matches_reference_on_native_ln() {
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
    let style: Vec<f32> = {
        let voices =
            std::path::Path::new("/Users/Shared/rlx-models/.cache/kittentts-mini-0.8/voices.npz");
        if !voices.is_file() {
            return;
        }
        let out = Command::new("python3")
            .args([
                "-c",
                "import numpy as np,sys; z=np.load(sys.argv[1]); sys.stdout.buffer.write(z['expr-voice-2-m'][6].astype('float32').tobytes())",
                voices.to_str().unwrap(),
            ])
            .output()
            .expect("style");
        out.stdout
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let ln_name = "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/LayerNorm/LayerNormalization";
    let ffn_name = "/bert/encoder/albert_layer_groups.0/albert_layers.0/ffn/MatMul_quant_f32";
    let ln_id = import
        .hir
        .nodes()
        .iter()
        .find(|n| n.name.as_deref() == Some(ln_name))
        .unwrap()
        .id;
    let ffn_id = import
        .hir
        .nodes()
        .iter()
        .find(|n| n.name.as_deref() == Some(ffn_name))
        .unwrap()
        .id;
    let mut ln_g =
        compile_probe_graph(Device::Cpu, &bundle_dir, &opts, &import, ln_id, "ln").expect("ln");
    let mut ffn_g =
        compile_probe_graph(Device::Cpu, &bundle_dir, &opts, &import, ffn_id, "ffn").expect("ffn");
    let ln = probe_output_f32_at(&run_parity_inputs(&mut ln_g, 8, &ids, &style), 0).unwrap();
    let ffn = probe_output_f32_at(&run_parity_inputs(&mut ffn_g, 8, &ids, &style), 0).unwrap();
    let script = r#"
import json, numpy as np, onnx, onnx.numpy_helper as nh
model = onnx.load('/Users/Shared/rlx-models/.cache/kittentts-mini-0.8/kitten_tts_mini_v0_8.onnx')
init = {i.name: nh.to_array(i) for i in model.graph.initializer}
w = init['onnx::MatMul_5895_quantized'].astype(np.int8)
ws = float(init['onnx::MatMul_5895_scale'].reshape(-1)[0])
wzp = int(init['onnx::MatMul_5895_zero_point'].reshape(-1)[0])
import sys
sys.stdout.buffer.write(json.dumps({'ws': ws, 'wzp': wzp}).encode())
sys.stdout.buffer.write(w.tobytes())
"#;
    let out = Command::new("python3")
        .arg("-c")
        .arg(script)
        .output()
        .expect("py");
    let split = out.stdout.iter().position(|&b| b == b'}').unwrap() + 1;
    let meta: serde_json::Value = serde_json::from_slice(&out.stdout[..split]).expect("meta");
    let w_q: Vec<i8> = out.stdout[split..].iter().map(|&b| b as i8).collect();
    let ws = meta["ws"].as_f64().unwrap() as f32;
    let wzp = meta["wzp"].as_i64().unwrap() as i32;
    let ref_out = qmatmul_f32_act_i8_weight(&ln, &[1, 8, 768], &w_q, &[768, 2048], ws, wzp);
    let max = ffn
        .iter()
        .zip(ref_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let idx = 1272usize;
    eprintln!(
        "ffn QMatMul vs ref on native LN: max={max:.4} @1272 native={} ref={}",
        ffn[idx], ref_out[idx]
    );
    assert!(
        max < 0.01,
        "ffn QMatMul kernel should match reference on native LN, max={max}"
    );
}
