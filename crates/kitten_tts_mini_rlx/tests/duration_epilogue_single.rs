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

//! Duration epilogue: single-probe parity (multi-probe batch is unreliable).

use kitten_tts_mini_rlx::GraphOptions;
use kitten_tts_mini_rlx::bundle_compile::{
    compile_from_bundle, compile_probe_graph, import_from_bundle_cached, probe_output_f32_at,
    run_parity_inputs, run_with_duration_fixed_point,
};
use rlx_ir::DType;
use rlx_runtime::Device;
use std::process::Command;

const PROBES: &[(&str, &str, &str)] = &[
    ("/bert_encoder/Add", "/bert_encoder/Add_output_0", "f32"),
    ("/text_encoder_1/Where_4", "/text_encoder_1/Where_4", "f32"),
    ("/lstm/Transpose", "/lstm/Transpose_output_0", "f32"),
    ("/lstm/LSTM_quant", "/lstm/LSTM_output_0", "f32"),
    (
        "/duration_proj/linear_layer/Add",
        "/duration_proj/linear_layer/Add_output_0",
        "f32",
    ),
    ("/Sigmoid", "/Sigmoid_output_0", "f32"),
    ("/ReduceSum", "/ReduceSum_output_0", "f32"),
];

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

fn ort_tensor(name: &str, dtype: &str) -> Vec<f32> {
    let script = format!(
        r#"
import onnx, numpy as np, onnxruntime as ort, json
from onnx import helper, TensorProto
DT = {{'f32': TensorProto.FLOAT, 'f16': TensorProto.FLOAT16}}[{dtype:?}]
model = onnx.load('/Users/Shared/rlx-models/.cache/kittentts-mini-0.8/kitten_tts_mini_v0_8.onnx')
model.graph.ClearField('output')
model.graph.output.append(helper.make_tensor_value_info({name:?}, DT, None))
onnx.save(model, '/tmp/k_dur_ep.onnx')
voices = np.load('/Users/Shared/rlx-models/.cache/kittentts-mini-0.8/voices.npz')
style = voices['expr-voice-2-m'][6:7].astype(np.float32)
ids = np.array([[0,50,83,156,54,57,135,0]], dtype=np.int64)
sess = ort.InferenceSession('/tmp/k_dur_ep.onnx', providers=['CPUExecutionProvider'])
a = sess.run(None, {{'input_ids': ids, 'style': style, 'speed': np.array([1.0], np.float32)}})[0]
a = np.asarray(a).reshape(-1)
if a.dtype == np.float16:
    a = a.astype(np.float32)
else:
    a = a.astype(np.float32)
print(json.dumps(a.tolist()))
"#,
        name = name,
        dtype = dtype
    );
    let out = Command::new("python3")
        .arg("-c")
        .arg(script)
        .output()
        .expect("ort");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Vec<f64> = serde_json::from_slice(&out.stdout).expect("json");
    v.into_iter().map(|x| x as f32).collect()
}

#[test]
fn duration_epilogue_single_probe_parity() {
    let bundle_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    if !bundle_dir.join("manifest.json").exists() {
        return;
    }
    let opts = GraphOptions {
        sequence_length: 8,
        max_waveform_samples: 8 * 600 + 12_000,
    };
    let import = import_from_bundle_cached(&bundle_dir, &opts).expect("import");
    let ids: Vec<i64> = vec![0, 50, 83, 156, 54, 57, 135, 0];
    let style = load_style_row();
    kitten_tts_mini_rlx::set_env_var("KITTEN_RLX_SKIP_FUSION", "1");
    kitten_tts_mini_rlx::opts::set_compile_sequence_length(8);
    eprintln!("=== duration epilogue single-probe ===");
    let mut worst = (0.0f32, String::new(), 0usize);
    for (hir, ort_name, dtype) in PROBES {
        let node_id = import
            .hir
            .nodes()
            .iter()
            .find(|n| n.name.as_deref() == Some(*hir))
            .map(|n| n.id)
            .unwrap_or_else(|| panic!("missing {hir}"));
        let mut graph = compile_probe_graph(Device::Cpu, &bundle_dir, &opts, &import, node_id, hir)
            .expect("compile");
        let nat =
            probe_output_f32_at(&run_parity_inputs(&mut graph, 8, &ids, &style), 0).expect("out");
        let o = ort_tensor(ort_name, dtype);
        let (max, idx) = nat
            .iter()
            .zip(o.iter())
            .enumerate()
            .map(|(j, (a, b))| ((a - b).abs(), j))
            .fold(
                (0.0f32, 0usize),
                |(m, i), (d, j)| {
                    if d > m { (d, j) } else { (m, i) }
                },
            );
        let short = hir
            .rsplit('/')
            .take(2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("/");
        eprintln!(
            "{short:40} max={max:.4} idx={idx} nat0={} ort0={}",
            nat[0], o[0]
        );
        if max > worst.0 {
            worst = (max, short, idx);
        }
    }
    eprintln!("worst single-probe: {} max={:.4}", worst.1, worst.0);

    // Full graph duration with fixed-point carry.
    let mut graph = compile_from_bundle(Device::Cpu, &bundle_dir, &opts).expect("full");
    let ids_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let style_bytes: Vec<u8> = style.iter().flat_map(|v| v.to_le_bytes()).collect();
    let speed_bytes: Vec<u8> = 1.0f32.to_le_bytes().to_vec();
    let outs = run_with_duration_fixed_point(
        &mut graph,
        &[
            ("input_ids", ids_bytes.as_slice(), DType::I64),
            ("style", style_bytes.as_slice(), DType::F32),
            ("speed", speed_bytes.as_slice(), DType::F32),
        ],
    );
    if let Some((dur_bytes, DType::I64)) = outs.get(1) {
        let dur: Vec<i64> = dur_bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        eprintln!("native duration={dur:?} sum={}", dur.iter().sum::<i64>());
    }
}
