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

//! Compare native vs ORT DQL scale on GELU for ffn_output QMatMul.

use kitten_tts_mini_rlx::GraphOptions;
use kitten_tts_mini_rlx::bundle_compile::{
    compile_probe_graph, import_from_bundle_cached, probe_output_f32_at, run_parity_inputs,
};
use kitten_tts_mini_rlx::qmatmul::dynamic_quantize_uint8;
use rlx_runtime::Device;
use std::process::Command;

fn load_style_row() -> Vec<f32> {
    let voices =
        std::path::Path::new("/Users/Shared/rlx-models/.cache/kittentts-mini-0.8/voices.npz");
    if !voices.is_file() {
        return vec![0.0; 256];
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
}

#[test]
fn ffn_output_qmatmul_dql_from_native_gelu() {
    let bundle_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    if !bundle_dir.join("manifest.json").exists() {
        return;
    }
    let opts = GraphOptions {
        sequence_length: 8,
        max_waveform_samples: 24_000,
    };
    let import = import_from_bundle_cached(&bundle_dir, &opts).expect("import");
    let gelu_name = "/bert/encoder/albert_layer_groups.0/albert_layers.0/activation/Mul_3";
    let gelu_id = import
        .hir
        .nodes()
        .iter()
        .find(|n| n.name.as_deref() == Some(gelu_name))
        .map(|n| n.id)
        .expect("gelu");
    kitten_tts_mini_rlx::set_env_var("KITTEN_RLX_SKIP_FUSION", "1");
    kitten_tts_mini_rlx::opts::set_compile_sequence_length(8);
    let mut graph = compile_probe_graph(
        Device::Cpu,
        &bundle_dir,
        &opts,
        &import,
        gelu_id,
        "gelu_probe",
    )
    .expect("compile");
    let ids: Vec<i64> = vec![0, 50, 83, 156, 54, 57, 135, 0];
    let style = load_style_row();
    let outs = run_parity_inputs(&mut graph, 8, &ids, &style);
    let nat = probe_output_f32_at(&outs, 0).expect("gelu");
    let (q, s, zp) = dynamic_quantize_uint8(&nat);
    let script = r#"
import onnx, numpy as np, onnxruntime as ort, json
from onnx import helper, TensorProto
model = onnx.load('/Users/Shared/rlx-models/.cache/kittentts-mini-0.8/kitten_tts_mini_v0_8.onnx')
for name, dt in [
    ('/bert/encoder/albert_layer_groups.0/albert_layers.0/activation/Mul_3_output_0', TensorProto.FLOAT),
    ('/bert/encoder/albert_layer_groups.0/albert_layers.0/activation/Mul_3_output_0_scale', TensorProto.FLOAT),
    ('/bert/encoder/albert_layer_groups.0/albert_layers.0/activation/Mul_3_output_0_zero_point', TensorProto.UINT8),
]:
    model.graph.output.append(helper.make_tensor_value_info(name, dt, None))
onnx.save(model, '/tmp/k_gelu.onnx')
voices = np.load('/Users/Shared/rlx-models/.cache/kittentts-mini-0.8/voices.npz')
style = voices['expr-voice-2-m'][6:7].astype(np.float32)
ids = np.array([[0,50,83,156,54,57,135,0]], dtype=np.int64)
sess = ort.InferenceSession('/tmp/k_gelu.onnx', providers=['CPUExecutionProvider'])
o = dict(zip([x.name for x in sess.get_outputs()], sess.run(None, {'input_ids': ids, 'style': style, 'speed': np.array([1.0], np.float32)})))
g = o['/bert/encoder/albert_layer_groups.0/albert_layers.0/activation/Mul_3_output_0'].astype(np.float32).reshape(-1)
print(json.dumps({
    'scale': float(o['/bert/encoder/albert_layer_groups.0/albert_layers.0/activation/Mul_3_output_0_scale'].reshape(-1)[0]),
    'zp': int(o['/bert/encoder/albert_layer_groups.0/albert_layers.0/activation/Mul_3_output_0_zero_point'].reshape(-1)[0]),
    'gelu': g.tolist(),
}))
"#;
    let out = Command::new("python3")
        .arg("-c")
        .arg(script)
        .output()
        .expect("py");
    let ort: serde_json::Value = serde_json::from_slice(&out.stdout).expect("json");
    let ort_gelu: Vec<f32> = ort["gelu"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    let gelu_max = nat
        .iter()
        .zip(ort_gelu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let idx = 1272usize;
    eprintln!(
        "gelu max diff={gelu_max:.4} @1272 nat={} ort={}",
        nat[idx], ort_gelu[idx]
    );
    eprintln!(
        "DQL native scale={s:.8} zp={zp} | ORT scale={:.8} zp={}",
        ort["scale"].as_f64().unwrap(),
        ort["zp"].as_u64().unwrap()
    );
    eprintln!("q len={} q@1272={}", q.len(), q[idx]);
    std::fs::write(
        "/tmp/nat_gelu.bin",
        nat.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    )
    .ok();
    let (_, s_ort, _zp_ort) = dynamic_quantize_uint8(&ort_gelu);
    let (q_ort, _, _) = dynamic_quantize_uint8(&ort_gelu);
    eprintln!("DQL on ORT gelu scale={s_ort:.8} q@1272={}", q_ort[idx]);
}
