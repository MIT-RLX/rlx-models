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

//! Compare native intermediates vs ONNX Runtime in one compile + one forward pass.
//!
//! Optional filter: `PROBE_FILTER=bert` or `--filter=bert`.

use std::process::Command;
use std::time::Instant;

use kitten_tts_mini_rlx::GraphOptions;
use kitten_tts_mini_rlx::bundle_compile::{
    compile_multi_probe_graph, import_from_bundle_cached, probe_output_f32_at, run_parity_inputs,
};
use kitten_tts_mini_rlx::probe_watch::{self, WATCH};
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

fn probe_filter() -> Option<String> {
    std::env::args()
        .skip(1)
        .find_map(|a| a.strip_prefix("--filter=").map(str::to_string))
        .or_else(|| std::env::var("PROBE_FILTER").ok())
}

struct DiffRow {
    label: String,
    max_abs: f32,
    spot: f32,
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

fn ort_tensors(ort_names: &[&str]) -> anyhow::Result<std::collections::HashMap<String, Vec<f32>>> {
    let root = repo_root();
    let model = root.join(".cache/kittentts-mini-0.8/kitten_tts_mini_v0_8.onnx");
    let voices = root.join(".cache/kittentts-mini-0.8/voices.npz");
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
onnx.save(model, '/tmp/kitten_cmp.onnx')
voices = np.load({voices:?})
style = voices['expr-voice-2-m'][6:7].astype(np.float32)
ids = np.array([[0,50,83,156,54,57,135,0]], dtype=np.int64)
speed = np.array([1.0], dtype=np.float32)
sess = ort.InferenceSession('/tmp/kitten_cmp.onnx', providers=['CPUExecutionProvider'])
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

fn main() -> anyhow::Result<()> {
    let filter = probe_filter();
    let bundle_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    let ids: Vec<i64> = vec![0, 50, 83, 156, 54, 57, 135, 0];
    let token_len = ids.len();
    let seq = 8usize.max(token_len.next_power_of_two());
    let max_wave = token_len
        .saturating_mul(600)
        .saturating_mul(8)
        .saturating_add(12_000)
        .max(seq.saturating_mul(600).saturating_mul(2))
        .max(24_000);
    let style = load_style_row();

    let graph_opts = GraphOptions {
        sequence_length: seq,
        max_waveform_samples: max_wave,
    };

    let t0 = Instant::now();
    let import = import_from_bundle_cached(&bundle_dir, &graph_opts)?;
    eprintln!("import: {:.1}s", t0.elapsed().as_secs_f64());

    {
        let cache = kitten_tts_mini_rlx::bundle_compile::SeqCompileCache::new(
            Device::Cpu,
            bundle_dir.clone(),
            seq,
            graph_opts.max_waveform_samples,
            2,
        );
        let graphs = cache.cached_graphs_for_seq(seq)?;
        let mut g = graphs.full.lock().expect("seq cache graph");
        let ort_dur: Vec<i64> = vec![19, 2, 1, 2, 3, 2, 3, 2];
        let outs = kitten_tts_mini_rlx::bundle_compile::run_parity_inputs_with_duration(
            &mut g,
            seq,
            ids.len(),
            &ids,
            &style,
            Some(&ort_dur),
        );
        if let Some((wave, _)) = outs.first() {
            let peak = wave
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "seq_cache waveform peak={peak:.4e} samples={}",
                wave.len() / 4
            );
        }
    }

    let mut resolved = Vec::new();
    for (hir_name, ort_name) in WATCH {
        let Some(node) = import
            .hir
            .nodes()
            .iter()
            .find(|n| n.name.as_deref() == Some(*hir_name))
        else {
            eprintln!("skip {hir_name}: missing HIR node");
            continue;
        };
        resolved.push((node.id, *hir_name, *ort_name));
    }
    if resolved.is_empty() {
        anyhow::bail!("probe_watch::WATCH is empty");
    }

    let t1 = Instant::now();
    let all_probes: Vec<_> = WATCH
        .iter()
        .filter_map(|(hir, _)| {
            import
                .hir
                .nodes()
                .iter()
                .find(|n| n.name.as_deref() == Some(*hir))
                .map(|n| (n.id, *hir))
        })
        .collect();
    let mut graph =
        compile_multi_probe_graph(Device::Cpu, &bundle_dir, &graph_opts, &import, &all_probes)?;
    eprintln!(
        "compile {} probes: {:.1}s",
        all_probes.len(),
        t1.elapsed().as_secs_f64()
    );

    let t2 = Instant::now();
    kitten_tts_mini_rlx::opts::set_compile_sequence_length(seq);
    let outs = run_parity_inputs(&mut graph, seq, &ids, &style);
    if let Some((wave, _)) = outs.first() {
        let peak = wave
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "graph output waveform peak={peak:.4e} samples={}",
            wave.len() / 4
        );
    }
    eprintln!("run: {:.1}s", t2.elapsed().as_secs_f64());

    let ort_names: Vec<&str> = resolved
        .iter()
        .filter(|(_, hir, _)| probe_watch::matches_filter(hir, filter.as_deref()))
        .map(|(_, _, ort)| *ort)
        .collect();
    let t3 = Instant::now();
    let ort = ort_tensors(&ort_names).unwrap_or_else(|e| {
        eprintln!("ort dump failed: {e}");
        std::collections::HashMap::new()
    });
    eprintln!("ort: {:.1}s", t3.elapsed().as_secs_f64());

    let display: Vec<_> = resolved
        .iter()
        .filter(|(_, hir, _)| probe_watch::matches_filter(hir, filter.as_deref()))
        .collect();
    eprintln!("=== native vs ORT ({} probes) ===", display.len());
    let mut rows = Vec::new();
    for (_, hir_name, ort_name) in display {
        let probe_idx = all_probes
            .iter()
            .position(|(_, h)| *h == *hir_name)
            .expect("probe in all_probes");
        let Some(nat) = probe_output_f32_at(&outs, probe_idx) else {
            eprintln!("{hir_name}: missing native probe output");
            continue;
        };
        let Some(ort_v) = ort.get(*ort_name) else {
            let nat_max = nat.iter().copied().map(f32::abs).fold(0.0, f32::max);
            let native_t0: Vec<f32> = nat.iter().take(4).copied().collect();
            eprintln!("{hir_name}: ort missing; native max={nat_max:.4} nat0={native_t0:?}");
            continue;
        };
        let (max_abs, max_idx) = max_abs_diff(&nat, ort_v);
        const SPOT: usize = 3576;
        let spot = if nat.len() > SPOT && ort_v.len() > SPOT {
            (nat[SPOT] - ort_v[SPOT]).abs()
        } else {
            0.0
        };
        let short = hir_name
            .rsplit('/')
            .take(2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("/");
        let native_t0: Vec<f32> = nat.iter().take(4).copied().collect();
        let ort_t0: Vec<f32> = ort_v.iter().take(4).copied().collect();
        eprintln!(
            "{short}: max={max_abs:.4} idx={max_idx} spot3576={spot:.4} nat0={native_t0:?} ort0={ort_t0:?}"
        );
        rows.push(DiffRow {
            label: short,
            max_abs,
            spot,
        });
    }

    rows.sort_by(|a, b| b.max_abs.partial_cmp(&a.max_abs).unwrap());
    eprintln!("\n=== worst first ===");
    for r in rows.iter().take(8) {
        eprintln!(
            "{:>40} max={:.4} spot3576={:.4}",
            r.label, r.max_abs, r.spot
        );
    }
    Ok(())
}
