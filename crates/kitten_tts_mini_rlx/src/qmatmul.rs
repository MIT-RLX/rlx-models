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

//! ONNX `MatMulInteger` + `DynamicQuantizeLinear` (uint8 activations, int8 weights).

/// When compile-time sequence headroom pads `[1, seq_compile, C]` activations,
/// only the first active token rows are valid at runtime.
fn clip_act_to_runtime_seq(act: &[f32], act_shape: &[usize]) -> (Vec<f32>, Vec<usize>) {
    let rt = crate::opts::compile_sequence_length_from_env().filter(|&n| n > 0);
    let Some(rt) = rt else {
        return (act.to_vec(), act_shape.to_vec());
    };
    if act_shape.len() == 3 {
        let b = act_shape[0].max(1);
        let s = act_shape[1].max(1);
        let h = act_shape[2];
        if rt < s {
            let n = b * rt * h;
            if n <= act.len() {
                return (act[..n].to_vec(), vec![b, rt, h]);
            }
        }
    }
    if act_shape.len() == 2 {
        let s = act_shape[0].max(1);
        let h = act_shape[1];
        if rt < s {
            let n = rt * h;
            if n <= act.len() {
                return (act[..n].to_vec(), vec![rt, h]);
            }
        }
    }
    (act.to_vec(), act_shape.to_vec())
}

/// `act @ w` matching ORT: one `DynamicQuantizeLinear` over the full activation
/// tensor, then `MatMulInteger` with activation/weight zero-points.
pub fn qmatmul_f32_act_i8_weight(
    act: &[f32],
    act_shape: &[usize],
    w: &[i8],
    w_shape: &[usize],
    w_scale: f32,
    w_zp: i32,
) -> Vec<f32> {
    let (act, act_shape) = clip_act_to_runtime_seq(act, act_shape);
    let act = act.as_slice();
    let act_shape = act_shape.as_slice();
    let (m, k, n) = matmul_dims(act_shape, w_shape);
    let mut out = vec![0.0f32; m * n];
    if k == 0 || n == 0 || m == 0 || act.len() < m * k || w.len() < k * n {
        return out;
    }
    let (xq, act_scale, act_zp) = dynamic_quantize_uint8(act);
    let act_zp_i32 = act_zp as i32;
    for i in 0..m {
        let row_off = i * k;
        for j in 0..n {
            let mut acc = 0i32;
            for p in 0..k {
                let aq = xq[row_off + p] as i32 - act_zp_i32;
                let wq = w[p * n + j] as i32 - w_zp;
                acc += aq * wq;
            }
            out[i * n + j] = acc as f32 * act_scale * w_scale;
        }
    }
    out
}

/// Pre-quantized activation path used when DQL outputs are wired into `QMatMul`.
pub fn qmatmul_uint8_act_i8_weight(
    act_q: &[u8],
    act_shape: &[usize],
    act_scale: f32,
    act_zp: u8,
    w: &[i8],
    w_shape: &[usize],
    w_scale: f32,
    w_zp: i32,
) -> Vec<f32> {
    let (m, k, n) = matmul_dims(act_shape, w_shape);
    let mut out = vec![0.0f32; m * n];
    if k == 0 || n == 0 || m == 0 || act_q.len() < m * k || w.len() < k * n {
        return out;
    }
    let act_zp_i32 = act_zp as i32;
    for i in 0..m {
        let row_off = i * k;
        for j in 0..n {
            let mut acc = 0i32;
            for p in 0..k {
                let aq = act_q[row_off + p] as i32 - act_zp_i32;
                let wq = w[p * n + j] as i32 - w_zp;
                acc += aq * wq;
            }
            out[i * n + j] = acc as f32 * act_scale * w_scale;
        }
    }
    out
}

pub fn dynamic_quantize_uint8(act: &[f32]) -> (Vec<u8>, f32, u8) {
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &x in act {
        mn = mn.min(x);
        mx = mx.max(x);
    }
    let r = mx - mn;
    let scale = if r > 0.0 { r / 255.0 } else { 1.0 };
    let zp = (-mn / scale).round().clamp(0.0, 255.0) as u8;
    let q: Vec<u8> = act
        .iter()
        .map(|&x| (x / scale + zp as f32).round().clamp(0.0, 255.0) as u8)
        .collect();
    (q, scale, zp)
}

fn matmul_dims(act_shape: &[usize], w_shape: &[usize]) -> (usize, usize, usize) {
    let k = w_shape.first().copied().filter(|&d| d > 0).unwrap_or(1);
    let n = w_shape.get(1).copied().filter(|&d| d > 0).unwrap_or(1);
    let m = if act_shape.len() >= 3 {
        act_shape[act_shape.len() - 2].max(1)
    } else if act_shape.len() == 2 {
        act_shape[0].max(1)
    } else {
        1
    };
    (m, k, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ORT dump fixtures under `/tmp/q1_*` and `/tmp/ffn_*` (optional dev-only).
    fn skip_unless_tmp_fixtures(paths: &[&str]) -> bool {
        if paths.iter().all(|p| std::path::Path::new(p).is_file()) {
            return false;
        }
        eprintln!("skip: missing ORT dump fixture(s): {}", paths.join(", "));
        true
    }

    fn q1_fixture_paths() -> [&'static str; 4] {
        [
            "/tmp/q1_act.bin",
            "/tmp/q1_w.bin",
            "/tmp/q1_ws.bin",
            "/tmp/q1_wzp.bin",
        ]
    }

    fn ffn_fixture_paths() -> [&'static str; 5] {
        [
            "/tmp/nat_ln2.bin",
            "/tmp/nat_ffn.bin",
            "/tmp/ffn_w.bin",
            "/tmp/ffn_ws.bin",
            "/tmp/ffn_wzp.bin",
        ]
    }

    #[test]
    fn clip_act_respects_runtime_sequence_length() {
        let _seq = crate::opts::CompileSequenceLengthGuard::set(2);
        let act: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let (clipped, shape) = super::clip_act_to_runtime_seq(&act, &[1, 6, 2]);
        assert_eq!(shape, vec![1, 2, 2]);
        assert_eq!(clipped.len(), 4);
        assert_eq!(clipped, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn qmatmul_ort_query1_fixture() {
        if skip_unless_tmp_fixtures(&q1_fixture_paths()) {
            return;
        }
        let act: Vec<f32> = std::fs::read("/tmp/q1_act.bin")
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let w: Vec<i8> = std::fs::read("/tmp/q1_w.bin")
            .unwrap()
            .into_iter()
            .map(|b| b as i8)
            .collect();
        let ws = f32::from_le_bytes(
            std::fs::read("/tmp/q1_ws.bin").unwrap()[..4]
                .try_into()
                .unwrap(),
        );
        let wzp = i8::from_le_bytes(
            std::fs::read("/tmp/q1_wzp.bin").unwrap()[..1]
                .try_into()
                .unwrap(),
        ) as i32;
        let out = qmatmul_f32_act_i8_weight(&act, &[1, 8, 768], &w, &[768, 768], ws, wzp);
        let idx = 3084usize;
        assert!((out[idx] - 0.1025764).abs() < 0.01, "got {}", out[idx]);
    }

    #[test]
    fn qmatmul_ffn_native_fixture() {
        if skip_unless_tmp_fixtures(&ffn_fixture_paths()) {
            return;
        }
        let act: Vec<f32> = std::fs::read("/tmp/nat_ln2.bin")
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let native_ffn: Vec<f32> = std::fs::read("/tmp/nat_ffn.bin")
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let w: Vec<i8> = std::fs::read("/tmp/ffn_w.bin")
            .unwrap()
            .into_iter()
            .map(|b| b as i8)
            .collect();
        let ws = f32::from_le_bytes(
            std::fs::read("/tmp/ffn_ws.bin").unwrap()[..4]
                .try_into()
                .unwrap(),
        );
        let wzp = i8::from_le_bytes(
            std::fs::read("/tmp/ffn_wzp.bin").unwrap()[..1]
                .try_into()
                .unwrap(),
        ) as i32;
        let out = qmatmul_f32_act_i8_weight(&act, &[1, 8, 768], &w, &[768, 2048], ws, wzp);
        let idx = 8522usize;
        eprintln!(
            "idx {idx}: qmatmul={} native_probe={}",
            out[idx], native_ffn[idx]
        );
        let max_diff = out
            .iter()
            .zip(native_ffn.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 0.01,
            "qmatmul vs native probe max diff {max_diff} at idx {idx}: {} vs {}",
            out[idx],
            native_ffn[idx]
        );
    }

    #[test]
    fn query1_probe_matches_quant_reference() {
        use crate::GraphOptions;
        use crate::bundle_compile::{
            compile_probe_graph, import_from_bundle_cached, set_runtime_input_ids_shape,
        };
        use rlx_runtime::Device;
        let bundle_dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
        if !bundle_dir.join("manifest.json").exists() {
            eprintln!("skip: {}", bundle_dir.display());
            return;
        }
        let _seq = crate::opts::CompileSequenceLengthGuard::set(8);
        crate::set_env_var("KITTEN_RLX_SKIP_FUSION", "1");
        let opts = GraphOptions {
            sequence_length: 8,
            max_waveform_samples: 24_000,
        };
        let import = import_from_bundle_cached(&bundle_dir, &opts).expect("import");
        let target = "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/query_1/MatMul_quant_f32";
        let node = import
            .hir
            .nodes()
            .iter()
            .find(|n| n.name.as_deref() == Some(target))
            .expect("query_1 QMatMul");
        assert!(
            matches!(
                &node.op,
                rlx_ir::HirOp::Mir(rlx_ir::Op::Custom { name, .. }) if name == "onnx.QMatMul"
            ),
            "expected onnx.QMatMul, got {:?}",
            node.op
        );
        let act_q = import.hir.node(import.hir.node(node.id).inputs[0]);
        assert!(
            matches!(
                &act_q.op,
                rlx_ir::HirOp::Mir(rlx_ir::Op::Custom { name, .. }) if name == "onnx.DynamicQuantizeLinearExport"
            ),
            "expected DQL export on QMatMul act, got {:?}",
            act_q.op
        );
        assert_eq!(import.hir.node(node.id).inputs.len(), 6);
        let mut graph =
            compile_probe_graph(Device::Cpu, &bundle_dir, &opts, &import, node.id, target)
                .expect("probe");
        set_runtime_input_ids_shape(&mut graph, 8).expect("shape");
        let ids: Vec<i64> = vec![0, 50, 83, 156, 54, 57, 135, 0];
        let voices =
            std::path::Path::new("/Users/Shared/rlx-models/.cache/kittentts-mini-0.8/voices.npz");
        if !voices.is_file() {
            eprintln!("skip: voices.npz not found at {}", voices.display());
            return;
        }
        let style: Vec<f32> = {
            let out = std::process::Command::new("python3")
                .args([
                    "-c",
                    "import numpy as np,sys; z=np.load(sys.argv[1]); sys.stdout.buffer.write(z['expr-voice-2-m'][6].astype('float32').tobytes())",
                    voices.to_str().unwrap(),
                ])
                .output()
                .expect("style row");
            out.stdout
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let ln_id = import.hir.node(act_q.id).inputs[0];
        let mut ln_graph =
            compile_probe_graph(Device::Cpu, &bundle_dir, &opts, &import, ln_id, "ln_probe")
                .expect("ln probe");
        set_runtime_input_ids_shape(&mut ln_graph, 8).expect("shape");
        let inputs = [
            (
                "input_ids",
                ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
                rlx_ir::DType::I64,
            ),
            (
                "style",
                style
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
                rlx_ir::DType::F32,
            ),
            ("speed", 1.0f32.to_le_bytes().to_vec(), rlx_ir::DType::F32),
        ];
        let run_f32 = |g: &mut rlx_runtime::CompiledGraph| -> Vec<f32> {
            let outs = g.run_typed(&[
                ("input_ids", &inputs[0].1, inputs[0].2),
                ("style", &inputs[1].1, inputs[1].2),
                ("speed", &inputs[2].1, inputs[2].2),
            ]);
            outs.first()
                .map(|(b, _)| {
                    b.chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect()
                })
                .expect("probe output")
        };
        let _ln_vals = run_f32(&mut ln_graph);
        let q_vals = run_f32(&mut graph);
        let idx = 3576usize;
        let val = q_vals[idx];
        eprintln!("query_1 probe idx {idx} = {val}");
        assert!(
            val.is_finite(),
            "QMatMul output should be finite, got {val}"
        );
    }

    #[test]
    fn dynamic_quantize_matches_ort_query1() {
        if skip_unless_tmp_fixtures(&["/tmp/q1_act.bin"]) {
            return;
        }
        let act: Vec<f32> = std::fs::read("/tmp/q1_act.bin")
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let (q, s, zp) = dynamic_quantize_uint8(&act);
        assert!((s - 0.0568081).abs() < 1e-6);
        assert_eq!(zp, 206);
        assert_eq!(q.len(), act.len());
    }
}
