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

//! Albert cross-attention (attention_1) and dense_1 single-probe parity vs ORT.

use kitten_tts_mini_rlx::GraphOptions;
use kitten_tts_mini_rlx::bundle_compile::{
    compile_probe_graph, import_from_bundle_cached, probe_output_f32_at, run_parity_inputs,
};
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

fn ort_tensor(name: &str) -> Vec<f32> {
    let script = format!(
        r#"
import onnx, numpy as np, onnxruntime as ort, json
from onnx import helper, TensorProto
model = onnx.load('/Users/Shared/rlx-models/.cache/kittentts-mini-0.8/kitten_tts_mini_v0_8.onnx')
model.graph.ClearField('output')
model.graph.output.append(helper.make_tensor_value_info({name:?}, TensorProto.FLOAT, None))
onnx.save(model, '/tmp/k_attn1.onnx')
voices = np.load('/Users/Shared/rlx-models/.cache/kittentts-mini-0.8/voices.npz')
style = voices['expr-voice-2-m'][6:7].astype(np.float32)
ids = np.array([[0,50,83,156,54,57,135,0]], dtype=np.int64)
a = ort.InferenceSession('/tmp/k_attn1.onnx', providers=['CPUExecutionProvider']).run(None, {{'input_ids': ids, 'style': style, 'speed': np.array([1.0], np.float32)}})[0]
print(json.dumps(a.astype(np.float32).reshape(-1).tolist()))
"#,
        name = name
    );
    let out = Command::new("python3")
        .arg("-c")
        .arg(script)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice::<Vec<f64>>(&out.stdout)
        .unwrap()
        .into_iter()
        .map(|x| x as f32)
        .collect()
}

fn probe_one(
    hir: &str,
    ort: &str,
    import: &kitten_tts_mini_rlx::bundle_compile::BundleImport,
    bundle_dir: &std::path::Path,
    opts: &GraphOptions,
    ids: &[i64],
    style: &[f32],
) -> (f32, usize) {
    let id = import
        .hir
        .nodes()
        .iter()
        .find(|n| n.name.as_deref() == Some(hir))
        .map(|n| n.id)
        .unwrap_or_else(|| panic!("missing {hir}"));
    let mut graph =
        compile_probe_graph(Device::Cpu, bundle_dir, opts, import, id, hir).expect("compile");
    let nat = probe_output_f32_at(&run_parity_inputs(&mut graph, 8, ids, style), 0).unwrap();
    let o = ort_tensor(ort);
    nat.iter()
        .zip(o.iter())
        .enumerate()
        .map(|(j, (a, b))| ((a - b).abs(), j))
        .fold(
            (0.0f32, 0usize),
            |(m, i), (d, j)| {
                if d > m { (d, j) } else { (m, i) }
            },
        )
}

const WARMUP: &[(&str, &str)] = &[
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/LayerNorm/LayerNormalization",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/LayerNorm/LayerNormalization_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/ffn/MatMul_quant_f32",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/ffn/MatMul_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/activation/Mul_3",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/activation/Mul_3_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/ffn_output/Add",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/ffn_output/Add_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/full_layer_layer_norm/LayerNormalization",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/full_layer_layer_norm/LayerNormalization_output_0",
    ),
];

const PROBES: &[(&str, &str)] = &[
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/query_1/Add",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/query_1/Add_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/key_1/Add",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/key_1/Add_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/value_1/Add",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/value_1/Add_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/MatMul",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/MatMul_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/Softmax",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/Softmax_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/Reshape_3",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/Reshape_3_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/dense_1/Add",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/dense_1/Add_output_0",
    ),
];

#[test]
fn attention1_chain_vs_ort() {
    let bundle_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    if !bundle_dir.join("manifest.json").exists() {
        return;
    }
    let opts = GraphOptions {
        sequence_length: 8,
        max_waveform_samples: 24_000,
    };
    let import = import_from_bundle_cached(&bundle_dir, &opts).expect("import");
    let ids: Vec<i64> = vec![0, 50, 83, 156, 54, 57, 135, 0];
    let style = load_style_row();
    kitten_tts_mini_rlx::set_env_var("KITTEN_RLX_SKIP_FUSION", "1");
    kitten_tts_mini_rlx::opts::set_compile_sequence_length(8);
    eprintln!("=== attention_1 chain single-probe (Albert warmup) ===");
    for (hir, ort) in WARMUP {
        let (max, idx) = probe_one(hir, ort, &import, &bundle_dir, &opts, &ids, &style);
        let short = hir
            .rsplit('/')
            .take(2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("/");
        eprintln!("warm {short:38} max={max:.4} idx={idx}");
    }
    for (hir, ort) in PROBES {
        let (max, idx) = probe_one(hir, ort, &import, &bundle_dir, &opts, &ids, &style);
        let short = hir
            .rsplit('/')
            .take(2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("/");
        eprintln!("{short:40} max={max:.4} idx={idx}");
    }
}
