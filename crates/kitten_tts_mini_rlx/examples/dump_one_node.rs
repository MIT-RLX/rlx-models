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

//! Dump one intermediate from the shared batch-probe graph (accurate ActCopy outputs).
//!
//! Compiles all [`probe_watch::WATCH`] nodes once (~15s cold), then ~1s per lookup.
//! Use `compare_intermediates` for a full table; add `--ort` here for a single ORT t0 check.

use std::process::Command;

use kitten_tts_mini_rlx::GraphOptions;
use kitten_tts_mini_rlx::bundle_compile::{
    compile_multi_probe_graph, compile_probe_graph, import_from_bundle_cached, probe_output_f32_at,
    run_parity_inputs,
};
use kitten_tts_mini_rlx::probe_watch::WATCH;
use rlx_ir::DType;
use rlx_runtime::Device;

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
    if !voices.is_file() {
        return vec![0.0; 256];
    }
    let out = Command::new("python3")
        .args(["-c", script, voices.to_str().unwrap(), "6"])
        .output()
        .expect("python");
    out.stdout
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn ort_tensor(ort_name: &str) -> Option<Vec<f32>> {
    let root = repo_root();
    let script = format!(
        r#"
import onnx, numpy as np, onnxruntime as ort, json
from onnx import helper, TensorProto
model = onnx.load({model:?})
name = {ort_name:?}
existing = {{o.name for o in model.graph.output}}
if name not in existing:
    model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
onnx.save(model, '/tmp/kitten_one.onnx')
voices = np.load({voices:?})
style = voices['expr-voice-2-m'][6:7].astype(np.float32)
ids = np.array([[0,50,83,156,54,57,135,10,0]], dtype=np.int64)
sess = ort.InferenceSession('/tmp/kitten_one.onnx', providers=['CPUExecutionProvider'])
outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {{'input_ids': ids, 'style': style, 'speed': np.array([1.0], np.float32)}})))
a = np.asarray(outs[name]).astype(np.float32).reshape(-1)
print(json.dumps(a[:8].tolist()))
"#,
        model = root.join(".cache/kittentts-mini-0.8/kitten_tts_mini_v0_8.onnx"),
        voices = root.join(".cache/kittentts-mini-0.8/voices.npz"),
    );
    let out = Command::new("python3")
        .arg("-c")
        .arg(script)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let v: Vec<f64> = serde_json::from_slice(&out.stdout).ok()?;
    Some(v.into_iter().map(|x| x as f32).collect())
}

fn resolve_watch_index(target: &str) -> Option<usize> {
    let target = target.trim();
    if let Some(i) = WATCH.iter().position(|(hir, _)| *hir == target) {
        return Some(i);
    }
    if let Some(i) = WATCH
        .iter()
        .position(|(hir, ort)| *hir == target || *ort == target)
    {
        return Some(i);
    }
    let needle = target.trim_start_matches('/');
    let mut best: Option<(usize, usize)> = None;
    for (i, (hir, ort)) in WATCH.iter().enumerate() {
        for cand in [*hir, *ort] {
            if cand.ends_with(needle) || cand.ends_with(&format!("/{needle}")) {
                let score = cand.len();
                if best.is_none_or(|(_, s)| score > s) {
                    best = Some((i, score));
                }
            }
        }
    }
    best.map(|(i, _)| i)
}

fn main() -> anyhow::Result<()> {
    let mut with_ort = false;
    let mut single_probe = false;
    let mut target = String::new();
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--ort" => with_ort = true,
            "--single" => single_probe = true,
            _ if target.is_empty() => target = arg,
            _ => {}
        }
    }
    if target.is_empty() {
        anyhow::bail!("usage: dump_one_node [--ort] NODE | dump_one_node --list [PATTERN]");
    }

    let bundle_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    let ids: Vec<i64> = vec![0, 50, 83, 156, 54, 57, 135, 10, 0];
    // Match production slot length (not next_power_of_two) so attention masks align with ORT.
    let seq = kitten_tts_mini_rlx::compile_profile::compile_slot_length(ids.len());
    let max_wave = 55_200usize;
    let graph_opts = GraphOptions {
        sequence_length: seq,
        max_waveform_samples: max_wave,
    };
    let import = import_from_bundle_cached(&bundle_dir, &graph_opts)?;

    if target == "--list" {
        let pat = std::env::args().nth(2).unwrap_or_default();
        for (hir, ort) in WATCH {
            if pat.is_empty() || hir.contains(&pat) || ort.contains(&pat) {
                println!("{hir}  ->  {ort}");
            }
        }
        return Ok(());
    }

    let idx = resolve_watch_index(&target)
        .ok_or_else(|| anyhow::anyhow!("{target} not in probe_watch::WATCH; use --list"))?;
    let (hir_name, ort_name) = WATCH[idx];
    // Prefer the Custom KittenInstanceNormActive (or other non-Reshape) when names collide.
    let node = import
        .hir
        .nodes()
        .iter()
        .filter(|n| n.name.as_deref() == Some(hir_name))
        .max_by_key(|n| n.shape.dims().iter().map(|d| d.unwrap_static()).product::<usize>())
        .ok_or_else(|| anyhow::anyhow!("HIR node missing: {hir_name}"))?;

    let probes: Vec<_> = WATCH
        .iter()
        .filter_map(|(hir, _)| {
            import
                .hir
                .nodes()
                .iter()
                .filter(|n| n.name.as_deref() == Some(*hir))
                .max_by_key(|n| n.shape.dims().iter().map(|d| d.unwrap_static()).product::<usize>())
                .map(|n| (n.id, *hir))
        })
        .collect();

    let style = load_style_row();
    kitten_tts_mini_rlx::opts::set_compile_sequence_length(seq);

    let vals = if single_probe {
        let mut graph = compile_probe_graph(
            Device::Cpu,
            &bundle_dir,
            &graph_opts,
            &import,
            node.id,
            hir_name,
        )?;
        let outs = run_parity_inputs(&mut graph, seq, &ids, &style);
        outs.first()
            .and_then(|(b, _)| probe_output_f32_at(&[(b.clone(), DType::F32)], 0))
            .expect("single probe output")
    } else {
        let mut graph =
            compile_multi_probe_graph(Device::Cpu, &bundle_dir, &graph_opts, &import, &probes)?;
        let outs = run_parity_inputs(&mut graph, seq, &ids, &style);
        let probe_idx = probes
            .iter()
            .position(|(_, h)| *h == hir_name)
            .expect("probe index");
        probe_output_f32_at(&outs, probe_idx).expect("probe output")
    };

    println!(
        "native {hir_name} shape={:?} t0={:?} len={}",
        node.shape.dims(),
        &vals[..4.min(vals.len())],
        vals.len()
    );
    if with_ort {
        if let Some(ort) = ort_tensor(ort_name) {
            println!("ort    t0={:?}", &ort[..4.min(ort.len())]);
        } else {
            eprintln!("ort dump failed for {ort_name}");
        }
    }
    if let Some(i) = std::env::var("IDX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        if i < vals.len() {
            println!("native idx={i} val={}", vals[i]);
        }
    }
    Ok(())
}
