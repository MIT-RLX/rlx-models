//! Focused duration-path bisection: native vs ONNX Runtime, only the
//! duration-predictor probes (avoids the vocoder probes that segfault the
//! full 58-probe harness).

use std::process::Command;

use kitten_tts_mini_rlx::GraphOptions;
use kitten_tts_mini_rlx::bundle_compile::{
    compile_multi_probe_graph, import_from_bundle_cached, probe_output_f32_at,
    set_runtime_active_sequence, set_runtime_input_ids_shape,
};
use rlx_ir::DType;
use rlx_runtime::Device;

const VOC: &[(&str, &str)] = &[
    ("/decoder/Unsqueeze", "/decoder/Unsqueeze_output_0"),
    (
        "/decoder/generator/f0_upsamp/Resize",
        "/decoder/generator/f0_upsamp/Resize_output_0",
    ),
];

const EXTRA: &[(&str, &str)] = &[
    ("/bert_encoder/Add", "/bert_encoder/Add_output_0"),
    (
        "/text_encoder_1/Concat_1",
        "/text_encoder_1/Concat_1_output_0",
    ),
    (
        "/text_encoder_1/Transpose_3",
        "/text_encoder_1/Transpose_3_output_0",
    ),
    (
        "/text_encoder/lstms.0/Reshape",
        "/text_encoder/lstms.0/Reshape_output_0",
    ),
    (
        "/text_encoder/lstms.2/Transpose",
        "/text_encoder/lstms.2/Transpose_output_0",
    ),
    (
        "/text_encoder/lstms.4/Transpose",
        "/text_encoder/lstms.4/Transpose_output_0",
    ),
    (
        "/text_encoder_1/Transpose_14",
        "/text_encoder_1/Transpose_14_output_0",
    ),
    (
        "/text_encoder_1/Gather_10",
        "/text_encoder_1/Gather_10_output_0",
    ),
    (
        "/text_encoder/lstms.5/LayerNormalization",
        "/text_encoder/lstms.5/LayerNormalization_output_0",
    ),
];

fn dur_watch() -> Vec<(&'static str, &'static str)> {
    if std::env::args().any(|a| a == "--voc") {
        return VOC.to_vec();
    }
    // Full WATCH minus vocoder/F0/N probes (those segfault the run).
    let mut v: Vec<(&'static str, &'static str)> = kitten_tts_mini_rlx::probe_watch::WATCH
        .iter()
        .copied()
        .filter(|(hir, _)| {
            !hir.contains("/decoder")
                && !hir.contains("/F0")
                && !hir.starts_with("/N.")
                && !hir.contains("/N_proj")
        })
        .collect();
    v.extend_from_slice(EXTRA);
    v
}

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap_or_else(|_| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

fn load_style_row() -> Vec<f32> {
    let script = r#"
import numpy as np, sys
z = np.load(sys.argv[1])
v = z['expr-voice-2-m'][int(sys.argv[2])]
sys.stdout.buffer.write(v.astype(np.float32).tobytes())
"#;
    let voices = repo_root().join(".cache/kittentts-mini-0.8/voices.npz");
    let out = Command::new("python3")
        .args(["-c", script, voices.to_str().unwrap(), "6"])
        .output()
        .expect("python");
    out.stdout
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn ort_tensors(
    ort_names: &[&str],
    ids: &[i64],
) -> anyhow::Result<std::collections::HashMap<String, Vec<f32>>> {
    let root = repo_root();
    let model = root.join(".cache/kittentts-mini-0.8/kitten_tts_mini_v0_8.onnx");
    let voices = root.join(".cache/kittentts-mini-0.8/voices.npz");
    let ids_list: Vec<i64> = ids.to_vec();
    let script = format!(
        r#"
import onnx, numpy as np, onnxruntime as ort, json
from onnx import helper, TensorProto
model = onnx.load({model:?})
existing = {{o.name for o in model.graph.output}}
want = {ort_names:?}
for name in want:
    if name not in existing:
        model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
onnx.save(model, '/tmp/kitten_durcmp.onnx')
voices = np.load({voices:?})
style = voices['expr-voice-2-m'][6:7].astype(np.float32)
ids = np.array([{ids_list:?}], dtype=np.int64)
speed = np.array([1.0], dtype=np.float32)
sess = ort.InferenceSession('/tmp/kitten_durcmp.onnx', providers=['CPUExecutionProvider'])
outs = sess.run(None, {{'input_ids': ids, 'style': style, 'speed': speed}})
out = {{}}
for n, a in zip([o.name for o in sess.get_outputs()], outs):
    a = np.asarray(a)
    if a.dtype == np.float32:
        out[n] = a.astype(np.float32).reshape(-1).tolist()
print(json.dumps(out))
"#
    );
    let out = Command::new("python3").arg("-c").arg(&script).output()?;
    if !out.status.success() {
        anyhow::bail!(
            "ort script failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let map: std::collections::HashMap<String, Vec<f64>> = serde_json::from_slice(&out.stdout)?;
    Ok(map
        .into_iter()
        .map(|(k, v)| (k, v.into_iter().map(|x| x as f32).collect()))
        .collect())
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    let n = a.len().min(b.len());
    let mut max = 0.0f32;
    let mut idx = 0usize;
    for j in 0..n {
        let d = (a[j] - b[j]).abs();
        if d > max {
            max = d;
            idx = j;
        }
    }
    (max, idx)
}

fn main() -> anyhow::Result<()> {
    let bundle_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    let big = std::env::args().any(|a| a == "--big");
    let huge = std::env::args().any(|a| a == "--huge");
    let ids: Vec<i64> = if huge {
        // 40 tokens → compile_seq >= 32 → the FULL vocoder-shape override path.
        let mut v = vec![83i64; 40];
        v[0] = 0;
        v[39] = 0;
        v
    } else if big {
        vec![
            0, 81, 83, 16, 53, 65, 156, 102, 53, 16, 44, 123, 156, 43, 135, 56, 0,
        ]
    } else {
        vec![0, 50, 83, 156, 54, 57, 135, 10, 0]
    };
    let token_len = ids.len();
    let seq = if big || huge {
        kitten_tts_mini_rlx::compile_profile::compile_slot_length(token_len)
    } else {
        8usize.max(token_len.next_power_of_two())
    };
    eprintln!("token_len={token_len} seq={seq}");
    let max_wave = token_len
        .saturating_mul(600)
        .saturating_mul(8)
        .saturating_add(12_000)
        .max(24_000);
    let style = load_style_row();

    let graph_opts = GraphOptions {
        sequence_length: seq,
        max_waveform_samples: max_wave,
    };

    let import = import_from_bundle_cached(&bundle_dir, &graph_opts)?;

    if std::env::args().any(|a| a == "--fullwave") {
        // Compile the FULL waveform graph via the same path the CLI synth uses.
        let mut graph = kitten_tts_mini_rlx::bundle_compile::compile_from_bundle(
            Device::Cpu,
            &bundle_dir,
            &graph_opts,
        )?;
        kitten_tts_mini_rlx::opts::set_compile_sequence_length(seq);
        set_runtime_input_ids_shape(&mut graph, seq)?;
        set_runtime_active_sequence(&mut graph, token_len, seq);
        let mut ids_padded = ids.clone();
        ids_padded.resize(seq, 0);
        let ids_bytes: Vec<u8> = ids_padded.iter().flat_map(|v| v.to_le_bytes()).collect();
        let style_bytes: Vec<u8> = style.iter().flat_map(|v| v.to_le_bytes()).collect();
        let speed = 1.0f32.to_le_bytes();
        let inputs: [(&str, &[u8], DType); 3] = [
            ("input_ids", &ids_bytes, DType::I64),
            ("style", &style_bytes, DType::F32),
            ("speed", &speed, DType::F32),
        ];
        let outs =
            kitten_tts_mini_rlx::bundle_compile::run_with_duration_fixed_point(&mut graph, &inputs);
        if let Some((wave, _)) = outs.first() {
            let s: Vec<f32> = wave
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let peak = s.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
            // active audio = runtime_mel * 600 samples; split active vs padded tail.
            let active = kitten_tts_mini_rlx::opts::runtime_mel_frames().unwrap_or(0) * 600;
            let act = active.min(s.len());
            let ap = s[..act].iter().fold(0.0f32, |a, &x| a.max(x.abs()));
            let pp = s[act..].iter().fold(0.0f32, |a, &x| a.max(x.abs()));
            eprintln!(
                "fullwave peak={peak:.4e} samples={} | ACTIVE({act}) peak={ap:.4e} PAD peak={pp:.4e}",
                s.len()
            );
        }
        return Ok(());
    }

    if std::env::args().any(|a| a == "--f0path") {
        let nodes = import.hir.nodes();
        for n in nodes.iter() {
            if let Some(nm) = n.name.as_deref() {
                if nm.contains("F0_proj")
                    || nm == "/decoder/Unsqueeze"
                    || nm.contains("f0_upsamp")
                    || nm.contains("/If")
                    || nm.contains("F0IfSelect")
                {
                    let dims: Vec<usize> =
                        n.shape.dims().iter().map(|d| d.unwrap_static()).collect();
                    eprintln!(
                        "{:>62} {:>16} {dims:?}",
                        nm,
                        format!("{:?}", n.op).chars().take(16).collect::<String>()
                    );
                }
            }
        }
        return Ok(());
    }

    if std::env::args().any(|a| a == "--sine") {
        let nodes = import.hir.nodes();
        for n in nodes.iter() {
            if let Some(nm) = n.name.as_deref() {
                if nm.contains("l_sin_gen/") {
                    let dims: Vec<usize> =
                        n.shape.dims().iter().map(|d| d.unwrap_static()).collect();
                    eprintln!(
                        "{:>60} {:>18} {dims:?}",
                        nm,
                        format!("{:?}", n.op).chars().take(18).collect::<String>()
                    );
                }
            }
        }
        return Ok(());
    }

    if std::env::args().any(|a| a == "--gen") {
        // Dump generator trunk node widths (time axis) from import.hir.
        let filt = std::env::args()
            .find_map(|a| a.strip_prefix("--gen=").map(|s| s.to_string()))
            .unwrap_or_default();
        let nodes = import.hir.nodes();
        for n in nodes.iter() {
            if let Some(nm) = n.name.as_deref() {
                let is_gen =
                    nm.starts_with("/decoder/generator/") || nm.starts_with("/decoder/decode.");
                if !is_gen || nm.contains("l_sin_gen/") {
                    continue;
                }
                if !filt.is_empty() && !nm.contains(&filt) {
                    continue;
                }
                let dims: Vec<usize> = n.shape.dims().iter().map(|d| d.unwrap_static()).collect();
                // only convs/transposes/leakyrelu/adain norms (structural)
                let opn = format!("{:?}", n.op);
                if opn.contains("Conv")
                    || opn.contains("Custom")
                    || nm.contains("LeakyRelu")
                    || nm.contains("ConvTranspose")
                {
                    eprintln!(
                        "{:>62} {:>16} {dims:?}",
                        nm,
                        opn.chars().take(16).collect::<String>()
                    );
                }
            }
        }
        return Ok(());
    }

    if std::env::args().any(|a| a == "--radmul") {
        let nodes = import.hir.nodes();
        let by_id = |id: rlx_ir::HirNodeId| nodes.iter().find(|n| n.id == id);
        for n in nodes.iter() {
            if n.name.as_deref() == Some("/decoder/generator/m_source/l_sin_gen/Mul")
                && matches!(n.op, rlx_ir::hir::HirOp::Mir(rlx_ir::Op::Binary(_)))
            {
                eprintln!(
                    "RAD Binary(Mul) shape={:?}",
                    n.shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect::<Vec<_>>()
                );
                for (k, inp) in n.inputs.iter().enumerate() {
                    if let Some(pn) = by_id(*inp) {
                        let dims: Vec<usize> =
                            pn.shape.dims().iter().map(|d| d.unwrap_static()).collect();
                        let nel: usize = dims.iter().product();
                        eprintln!(
                            "   in[{k}] op={:?} name={:?} shape={dims:?} nelem={nel}",
                            pn.op, pn.name
                        );
                        for (kk, i2) in pn.inputs.iter().enumerate() {
                            if let Some(p2) = by_id(*i2) {
                                let d2: Vec<usize> =
                                    p2.shape.dims().iter().map(|d| d.unwrap_static()).collect();
                                let n2: usize = d2.iter().product();
                                eprintln!(
                                    "      in[{k}].in[{kk}] op={:?} name={:?} shape={d2:?} nelem={n2}",
                                    p2.op, p2.name
                                );
                            }
                        }
                    }
                }
            }
        }
        return Ok(());
    }

    if std::env::args().any(|a| a == "--hir") {
        let nodes = import.hir.nodes();
        let by_id = |id: rlx_ir::HirNodeId| nodes.iter().find(|n| n.id == id);
        for target in [
            "/decoder/generator/m_source/l_sin_gen/Mul",
            "/decoder/generator/m_source/l_sin_gen/Resize",
            "/decoder/generator/m_source/l_sin_gen/Transpose_1",
            "/decoder/generator/m_source/l_sin_gen/CumSum",
            "/decoder/generator/m_source/l_sin_gen/Mul_7",
            "/decoder/generator/m_source/l_sin_gen/Transpose_2",
        ] {
            if let Some(n) = nodes.iter().find(|n| n.name.as_deref() == Some(target)) {
                eprintln!("NODE {target}: op={:?} shape={:?}", n.op, n.shape);
                for (k, inp) in n.inputs.iter().enumerate() {
                    if let Some(pn) = by_id(*inp) {
                        eprintln!(
                            "   in[{k}] id={:?} op={:?} name={:?} shape={:?}",
                            inp, pn.op, pn.name, pn.shape
                        );
                    } else {
                        eprintln!("   in[{k}] id={:?} <missing>", inp);
                    }
                }
            } else {
                eprintln!("NODE {target}: NOT FOUND");
            }
        }
        return Ok(());
    }

    let watch = dur_watch();
    let mut probes = Vec::new();
    for (hir, _ort) in &watch {
        // Names can collide (gamma-reshape + LayerNorm share the ONNX node name);
        // prefer the node whose op "matches" the tensor semantics.
        let matches: Vec<_> = import
            .hir
            .nodes()
            .iter()
            .filter(|n| n.name.as_deref() == Some(*hir))
            .collect();
        let picked = if hir.ends_with("LayerNormalization") {
            matches
                .iter()
                .find(|n| format!("{:?}", n.op).contains("LayerNorm"))
                .or_else(|| matches.first())
                .copied()
        } else {
            matches.first().copied()
        };
        if let Some(n) = picked {
            probes.push((n.id, *hir));
        } else {
            eprintln!("skip {hir}: missing HIR node");
        }
    }
    if probes.is_empty() {
        anyhow::bail!("no duration probes resolved");
    }

    let mut graph =
        compile_multi_probe_graph(Device::Cpu, &bundle_dir, &graph_opts, &import, &probes)?;
    eprintln!("compiled {} duration probes", probes.len());

    // Seed inputs (ORT-oracle style: seed carry then single run_typed).
    kitten_tts_mini_rlx::opts::set_compile_sequence_length(seq);
    set_runtime_input_ids_shape(&mut graph, seq)?;
    set_runtime_active_sequence(&mut graph, token_len, seq);

    let mut ids_padded = ids.clone();
    ids_padded.resize(seq, 0);
    let ids_bytes: Vec<u8> = ids_padded.iter().flat_map(|v| v.to_le_bytes()).collect();
    let style_bytes: Vec<u8> = style.iter().flat_map(|v| v.to_le_bytes()).collect();
    let speed = 1.0f32.to_le_bytes();
    let inputs: [(&str, &[u8], DType); 3] = [
        ("input_ids", &ids_bytes, DType::I64),
        ("style", &style_bytes, DType::F32),
        ("speed", &speed, DType::F32),
    ];
    if let Some(m) = std::env::args().find_map(|a| a.strip_prefix("--mel=").map(|s| s.to_string()))
    {
        let mel: usize = m.parse().unwrap();
        kitten_tts_mini_rlx::opts::set_runtime_mel_frames(mel);
        eprintln!("set runtime_mel_frames={mel}");
    }
    let voc = std::env::args().any(|a| a == "--voc");
    let outs = if voc {
        // Seed ORT durations so the alignment/mel matches ORT; isolates the vocoder path.
        let ort_dur: Vec<i64> = if big {
            vec![19, 2, 2, 2, 2, 1, 2, 2, 3, 2, 2, 2, 2, 2, 2, 3, 2]
        } else {
            vec![19, 2, 1, 2, 3, 2, 3, 2]
        };
        kitten_tts_mini_rlx::bundle_compile::run_parity_inputs_with_duration(
            &mut graph,
            seq,
            token_len,
            &ids,
            &style,
            Some(&ort_dur),
        )
    } else {
        graph.run_typed(&inputs)
    };

    let ort_names: Vec<&str> = probes
        .iter()
        .map(|(_, hir)| watch.iter().find(|(h, _)| h == hir).unwrap().1)
        .collect();
    let ort = ort_tensors(&ort_names, &ids)?;

    eprintln!("=== duration path: native vs ORT ===");
    for (i, (_, hir)) in probes.iter().enumerate() {
        let Some(nat) = probe_output_f32_at(&outs, i) else {
            eprintln!(
                "{hir}: missing native probe (dt={:?})",
                outs.get(i).map(|(_, d)| d)
            );
            continue;
        };
        let ort_name = watch.iter().find(|(h, _)| h == hir).unwrap().1;
        let Some(ort_v) = ort.get(ort_name) else {
            let m = nat.iter().copied().map(f32::abs).fold(0.0, f32::max);
            eprintln!(
                "{hir}: ORT missing; native len={} max={m:.4} n0={:?}",
                nat.len(),
                &nat[..nat.len().min(6)]
            );
            continue;
        };
        let (max_abs, idx) = max_abs_diff(&nat, ort_v);
        let mut fin = 0usize;
        let mut nan = 0usize;
        let mut inf = 0usize;
        let mut fmin = f32::INFINITY;
        let mut fmax = f32::NEG_INFINITY;
        for &x in &nat {
            if x.is_nan() {
                nan += 1;
            } else if x.is_infinite() {
                inf += 1;
            } else {
                fin += 1;
                fmin = fmin.min(x);
                fmax = fmax.max(x);
            }
        }
        eprintln!(
            "{hir}: len nat={} ort={} | finite={fin} nan={nan} inf={inf} fmin={fmin:.3} fmax={fmax:.3} | max_abs={max_abs:.4}@{idx} nat0={:?} ort0={:?}",
            nat.len(),
            ort_v.len(),
            &nat[..nat.len().min(4)],
            &ort_v[..ort_v.len().min(4)],
        );
    }
    Ok(())
}
