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

//! lstms.5 fc Gemm_Add vs ORT after QMatMul import fix.

use kitten_tts_mini_rlx::GraphOptions;
use kitten_tts_mini_rlx::bundle_compile::{
    compile_probe_graph, import_from_bundle_cached, probe_output_f32_at, run_parity_inputs,
};
use rlx_runtime::Device;
use std::process::Command;

#[test]
fn lstms5_fc_gemm_add_vs_ort() {
    let bundle_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    if !bundle_dir.join("manifest.json").exists() {
        return;
    }
    let opts = GraphOptions {
        sequence_length: 8,
        max_waveform_samples: 24_000,
    };
    let import = import_from_bundle_cached(&bundle_dir, &opts).expect("import");
    let hir = "/text_encoder/lstms.5/fc/Gemm_Add";
    let ort = "/text_encoder/lstms.5/fc/Gemm_output_0";
    let id = import
        .hir
        .nodes()
        .iter()
        .find(|n| n.name.as_deref() == Some(hir))
        .map(|n| n.id)
        .expect("node");
    kitten_tts_mini_rlx::set_env_var("KITTEN_RLX_SKIP_FUSION", "1");
    kitten_tts_mini_rlx::opts::set_compile_sequence_length(8);
    let mut graph =
        compile_probe_graph(Device::Cpu, &bundle_dir, &opts, &import, id, hir).expect("compile");
    let ids: Vec<i64> = vec![0, 50, 83, 156, 54, 57, 135, 0];
    let voices =
        std::path::Path::new("/Users/Shared/rlx-models/.cache/kittentts-mini-0.8/voices.npz");
    let style: Vec<f32> = if voices.is_file() {
        Command::new("python3")
            .args([
                "-c",
                "import numpy as np,sys; z=np.load(sys.argv[1]); sys.stdout.buffer.write(z['expr-voice-2-m'][6].astype('float32').tobytes())",
                voices.to_str().unwrap(),
            ])
            .output()
            .unwrap()
            .stdout
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    } else {
        vec![0.0; 256]
    };
    let nat = probe_output_f32_at(&run_parity_inputs(&mut graph, 8, &ids, &style), 0).unwrap();
    let script = format!(
        r#"
import onnx, numpy as np, onnxruntime as ort, json
from onnx import helper, TensorProto
model = onnx.load('/Users/Shared/rlx-models/.cache/kittentts-mini-0.8/kitten_tts_mini_v0_8.onnx')
model.graph.ClearField('output')
model.graph.output.append(helper.make_tensor_value_info({ort:?}, TensorProto.FLOAT, None))
onnx.save(model, '/tmp/k_lstms5b.onnx')
voices = np.load('/Users/Shared/rlx-models/.cache/kittentts-mini-0.8/voices.npz')
style = voices['expr-voice-2-m'][6:7].astype(np.float32)
ids = np.array([[0,50,83,156,54,57,135,0]], dtype=np.int64)
a = ort.InferenceSession('/tmp/k_lstms5b.onnx', providers=['CPUExecutionProvider']).run(None, {{'input_ids': ids, 'style': style, 'speed': np.array([1.0], np.float32)}})[0]
print(json.dumps(a.astype(np.float32).reshape(-1).tolist()))
"#,
        ort = ort
    );
    let out = Command::new("python3")
        .arg("-c")
        .arg(script)
        .output()
        .unwrap();
    let o: Vec<f32> = serde_json::from_slice::<Vec<f64>>(&out.stdout)
        .unwrap()
        .into_iter()
        .map(|x: f64| x as f32)
        .collect();
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
    eprintln!(
        "lstms5 Gemm_Add max={max:.4} idx={idx} nat0={} ort0={} nat1430={} ort1430={}",
        nat[0],
        o[0],
        nat[1430.min(nat.len() - 1)],
        o[1430.min(o.len() - 1)]
    );
    assert!(max < 0.5, "lstms5 fc Gemm_Add max diff {max} at {idx}");
}

#[test]
fn lstms5_add2_and_concat4_vs_ort() {
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
    let style = vec![0.0f32; 256]; // will load below
    let voices =
        std::path::Path::new("/Users/Shared/rlx-models/.cache/kittentts-mini-0.8/voices.npz");
    let style: Vec<f32> = if voices.is_file() {
        std::process::Command::new("python3")
            .args([
                "-c",
                "import numpy as np,sys; z=np.load(sys.argv[1]); sys.stdout.buffer.write(z['expr-voice-2-m'][6].astype('float32').tobytes())",
                voices.to_str().unwrap(),
            ])
            .output()
            .unwrap()
            .stdout
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    } else {
        style
    };
    for (hir, ort) in [
        (
            "/text_encoder/lstms.5/Mul_2",
            "/text_encoder/lstms.5/Mul_2_output_0",
        ),
        (
            "/text_encoder/lstms.5/Add_2",
            "/text_encoder/lstms.5/Add_2_output_0",
        ),
        ("/text_encoder_1/Concat_4", "/text_encoder_1/Concat_4"),
    ] {
        let id = import
            .hir
            .nodes()
            .iter()
            .find(|n| n.name.as_deref() == Some(hir))
            .map(|n| n.id)
            .unwrap();
        let mut graph = compile_probe_graph(Device::Cpu, &bundle_dir, &opts, &import, id, hir)
            .expect("compile");
        let nat = probe_output_f32_at(&run_parity_inputs(&mut graph, 8, &ids, &style), 0).unwrap();
        let script = format!(
            r#"
import onnx, numpy as np, onnxruntime as ort, json
from onnx import helper, TensorProto
model = onnx.load('/Users/Shared/rlx-models/.cache/kittentts-mini-0.8/kitten_tts_mini_v0_8.onnx')
model.graph.ClearField('output')
model.graph.output.append(helper.make_tensor_value_info({ort:?}, TensorProto.FLOAT, None))
onnx.save(model, '/tmp/k_probe.onnx')
voices = np.load('/Users/Shared/rlx-models/.cache/kittentts-mini-0.8/voices.npz')
style = voices['expr-voice-2-m'][6:7].astype(np.float32)
ids = np.array([[0,50,83,156,54,57,135,0]], dtype=np.int64)
a = ort.InferenceSession('/tmp/k_probe.onnx', providers=['CPUExecutionProvider']).run(None, {{'input_ids': ids, 'style': style, 'speed': np.array([1.0], np.float32)}})[0]
print(json.dumps(a.astype(np.float32).reshape(-1).tolist()))
"#
        );
        let out = std::process::Command::new("python3")
            .arg("-c")
            .arg(script)
            .output()
            .unwrap();
        let o: Vec<f32> = serde_json::from_slice::<Vec<f64>>(&out.stdout)
            .unwrap()
            .into_iter()
            .map(|x| x as f32)
            .collect();
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
        eprintln!(
            "{hir} max={max:.4} idx={idx} @0 nat={} ort={}",
            nat[0], o[0]
        );
    }
}

#[test]
fn text_encoder1_concat1_vs_ort() {
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
    let hir = "/text_encoder_1/Concat_1";
    let ort = "/text_encoder_1/Concat_1_output_0";
    let id = import
        .hir
        .nodes()
        .iter()
        .find(|n| n.name.as_deref() == Some(hir))
        .map(|n| n.id)
        .unwrap();
    let mut graph =
        compile_probe_graph(Device::Cpu, &bundle_dir, &opts, &import, id, hir).expect("compile");
    let ids: Vec<i64> = vec![0, 50, 83, 156, 54, 57, 135, 0];
    let voices =
        std::path::Path::new("/Users/Shared/rlx-models/.cache/kittentts-mini-0.8/voices.npz");
    let style: Vec<f32> = if voices.is_file() {
        std::process::Command::new("python3")
            .args([
                "-c",
                "import numpy as np,sys; z=np.load(sys.argv[1]); sys.stdout.buffer.write(z['expr-voice-2-m'][6].astype('float32').tobytes())",
                voices.to_str().unwrap(),
            ])
            .output()
            .unwrap()
            .stdout
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    } else {
        vec![0.0; 256]
    };
    let nat = probe_output_f32_at(&run_parity_inputs(&mut graph, 8, &ids, &style), 0).unwrap();
    let script = format!(
        r#"
import onnx, numpy as np, onnxruntime as ort, json
from onnx import helper, TensorProto
model = onnx.load('/Users/Shared/rlx-models/.cache/kittentts-mini-0.8/kitten_tts_mini_v0_8.onnx')
model.graph.ClearField('output')
model.graph.output.append(helper.make_tensor_value_info({ort:?}, TensorProto.FLOAT, None))
onnx.save(model, '/tmp/k_c1.onnx')
voices = np.load('/Users/Shared/rlx-models/.cache/kittentts-mini-0.8/voices.npz')
style = voices['expr-voice-2-m'][6:7].astype(np.float32)
ids = np.array([[0,50,83,156,54,57,135,0]], dtype=np.int64)
a = ort.InferenceSession('/tmp/k_c1.onnx', providers=['CPUExecutionProvider']).run(None, {{'input_ids': ids, 'style': style, 'speed': np.array([1.0], np.float32)}})[0]
print(json.dumps(a.astype(np.float32).reshape(-1).tolist()))
"#
    );
    let out = std::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .output()
        .unwrap();
    let o: Vec<f32> = serde_json::from_slice::<Vec<f64>>(&out.stdout)
        .unwrap()
        .into_iter()
        .map(|x| x as f32)
        .collect();
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
    eprintln!(
        "Concat_1 max={max:.4} idx={idx} @0 nat={} ort={}",
        nat[0], o[0]
    );
}

#[test]
fn attention_dense1_add_vs_ort() {
    probe_one(
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/dense_1/Add",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/dense_1/Add_output_0",
    );
}

fn probe_one(hir: &str, ort: &str) {
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
    let id = import
        .hir
        .nodes()
        .iter()
        .find(|n| n.name.as_deref() == Some(hir))
        .map(|n| n.id)
        .unwrap();
    let mut graph =
        compile_probe_graph(Device::Cpu, &bundle_dir, &opts, &import, id, hir).expect("compile");
    let ids: Vec<i64> = vec![0, 50, 83, 156, 54, 57, 135, 0];
    let style = vec![0.0f32; 256];
    let nat = probe_output_f32_at(&run_parity_inputs(&mut graph, 8, &ids, &style), 0).unwrap();
    let script = format!(
        r#"
import onnx, numpy as np, onnxruntime as ort, json
from onnx import helper, TensorProto
model = onnx.load('/Users/Shared/rlx-models/.cache/kittentts-mini-0.8/kitten_tts_mini_v0_8.onnx')
model.graph.ClearField('output')
model.graph.output.append(helper.make_tensor_value_info({ort:?}, TensorProto.FLOAT, None))
onnx.save(model, '/tmp/k_probe2.onnx')
voices = np.load('/Users/Shared/rlx-models/.cache/kittentts-mini-0.8/voices.npz')
style = voices['expr-voice-2-m'][6:7].astype(np.float32)
ids = np.array([[0,50,83,156,54,57,135,0]], dtype=np.int64)
a = ort.InferenceSession('/tmp/k_probe2.onnx', providers=['CPUExecutionProvider']).run(None, {{'input_ids': ids, 'style': style, 'speed': np.array([1.0], np.float32)}})[0]
print(json.dumps(a.astype(np.float32).reshape(-1).tolist()))
"#
    );
    let out = std::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .output()
        .unwrap();
    let o: Vec<f32> = serde_json::from_slice::<Vec<f64>>(&out.stdout)
        .unwrap()
        .into_iter()
        .map(|x| x as f32)
        .collect();
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
    eprintln!("{hir} max={max:.4} idx={idx}");
}
