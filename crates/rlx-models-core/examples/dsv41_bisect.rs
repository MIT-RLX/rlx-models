// RLX — versatile ML compiler + runtime. GPLv3.
//! Walk the DeepSeek-V4.1 port against the reference dump stage by stage.
//!
//! Point `RLX_DSV41_REF` at a full (untrimmed) dump from the reference harness
//! and run `cargo run -p rlx-models-core --example dsv41_bisect`. Each tap the
//! builder exposes is compiled on its own and compared to the matching
//! `inter.<stage>.<layer>` entry, so the first stage that diverges is the one
//! that is wrong — not the twentieth one downstream of it.

use rlx_models_core::dsv41::DeepseekV41Spec;
use rlx_models_core::dsv41_graph::{V41Inputs, build_deepseek_v41_prefill};
use rlx_models_core::weight_loader::WeightLoader;
use rlx_runtime::{Device, Session};
use serde_json::Value;
use std::collections::BTreeMap;

fn fnv1a(name: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn param_values(name: &str, shape: &[usize]) -> Vec<f32> {
    let n: usize = shape.iter().product::<usize>().max(1);
    let scale = if shape.len() >= 2 {
        (1.0f64 / *shape.last().unwrap() as f64).sqrt()
    } else {
        0.2
    };
    let mut s = fnv1a(name);
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let f = (z >> 11) as f64 / (1u64 << 53) as f64;
            ((f - 0.5) * 2.0 * scale) as f32
        })
        .collect()
}

struct RefLoader {
    shapes: BTreeMap<String, Vec<usize>>,
}

impl WeightLoader for RefLoader {
    fn take(&mut self, key: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        let shape = self
            .shapes
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("no tensor `{key}`"))?
            .clone();
        Ok((param_values(key, &shape), shape))
    }
    fn take_transposed(&mut self, key: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        let (d, s) = self.take(key)?;
        if s.len() != 2 {
            return Ok((d, s));
        }
        let (r, c) = (s[0], s[1]);
        let mut o = vec![0f32; d.len()];
        for i in 0..r {
            for j in 0..c {
                o[j * r + i] = d[i * c + j];
            }
        }
        Ok((o, vec![c, r]))
    }
    fn len(&self) -> usize {
        self.shapes.len()
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.shapes.keys().cloned().collect()
    }
}

fn main() -> anyhow::Result<()> {
    let path = std::env::var("RLX_DSV41_REF").unwrap_or_else(|_| {
        format!(
            "{}/tests/fixtures/dsv41_toy_ref.json",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let spec = DeepseekV41Spec::from_config(&fx["config"])?;
    let shapes: BTreeMap<String, Vec<usize>> = fx["shapes"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_u64().unwrap() as usize)
                    .collect(),
            )
        })
        .collect();
    let ids: Vec<f32> = fx["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as f32)
        .collect();
    let engram_rows: Vec<i64> = fx
        .get("engram_rows")
        .and_then(Value::as_array)
        .map(|a| a.iter().map(|v| v.as_i64().unwrap()).collect())
        .unwrap_or_default();
    let inputs = V41Inputs {
        engram_rows,
        image_positions: Vec::new(),
        emit_main_hidden: false,
    };

    let run = || -> anyhow::Result<Vec<f32>> {
        let mut loader = RefLoader {
            shapes: shapes.clone(),
        };
        let mut packed = std::collections::HashMap::new();
        let (g, params) =
            build_deepseek_v41_prefill(&spec, &mut loader, ids.len(), &inputs, &mut packed)?;
        let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
            &rlx_flow::CompileProfile::qwen3_prefill(),
            Device::Cpu,
        );
        let mut c = Session::new(Device::Cpu).compile_with(g, &opts);
        for (n, d) in &params {
            c.set_param(n, d);
        }
        Ok(c.run(&[("input_ids", ids.as_slice())])[0].clone())
    };

    if let Ok(k) = std::env::var("RLX_DSV41_DUMPPARAM") {
        let sh = shapes.get(&k).unwrap_or_else(|| panic!("no shape for {k}"));
        let v = param_values(&k, sh);
        let mut st = fnv1a(&k);
        let mut zs = Vec::new();
        for _ in 0..4 {
            st = st.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = st;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            zs.push(z);
        }
        println!("seed {} z {:?}", fnv1a(&k), zs);
        println!("{k} shape {sh:?} first8 {:?}", &v[..8.min(v.len())]);
        return Ok(());
    }

    let empty = serde_json::Map::new();
    let inter = fx["inter"].as_object().unwrap_or(&empty);
    let mut plan: Vec<(String, usize)> = Vec::new();
    for il in 0..spec.n_layers {
        for stage in ["engram", "comp", "compkv", "topk", "attn", "ffn", "block"] {
            if inter.contains_key(&format!("{}.{il}", stage_key(stage))) {
                plan.push((stage.to_string(), il));
            }
        }
    }

    for (stage, il) in plan {
        // SAFETY: single-threaded example; the builder reads these at graph time.
        unsafe {
            std::env::set_var("RLX_DSV41_DBG", &stage);
            std::env::set_var("RLX_DSV41_DBGLAYER", il.to_string());
        }
        let key = format!("{}.{il}", stage_key(&stage));
        let raw: Vec<f64> = inter[&key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        match run() {
            Ok(got) if stage == "topk" => {
                // the reference publishes selected indices; compare the SETS
                let ncomp = got.len() / ids.len();
                let topk = raw.len() / ids.len();
                let mut bad = 0;
                let mut first = String::new();
                for q in 0..ids.len() {
                    let mut want_set: Vec<usize> = raw[q * topk..(q + 1) * topk]
                        .iter()
                        .filter(|&&v| v >= 0.0)
                        .map(|&v| v as usize - ids.len())
                        .collect();
                    want_set.sort_unstable();
                    // a mask entry only matters if it does not annihilate the
                    // softmax term; -1 is already e^-1 of weight, -30 is nothing
                    let mut got_set: Vec<usize> = (0..ncomp)
                        .filter(|&c| got[q * ncomp + c] > -30.0)
                        .collect();
                    got_set.sort_unstable();
                    if want_set != got_set {
                        bad += 1;
                        if first.is_empty() {
                            first = format!(
                                "q{q}: got {got_set:?} want {want_set:?} mask {:?}",
                                got[q * ncomp..(q + 1) * ncomp]
                                    .iter()
                                    .map(|v| if *v < -1e6 { -1e6 } else { *v })
                                    .collect::<Vec<_>>()
                            );
                        }
                    }
                }
                println!(
                    "{} {key:<16} {bad}/{} rows differ  {first}",
                    if bad == 0 { "ok  " } else { "FAIL" },
                    ids.len()
                );
            }
            Ok(got) => {
                let want: Vec<f32> = raw.iter().map(|&v| v as f32).collect();
                report(&key, &got, &want)
            }
            Err(e) => println!("{key:<16} BUILD FAILED: {e}"),
        }
    }

    unsafe {
        std::env::remove_var("RLX_DSV41_DBG");
        std::env::remove_var("RLX_DSV41_DBGLAYER");
    }
    let got = run()?;
    let want: Vec<f32> = fx["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    report("logits", &got, &want);
    Ok(())
}

/// Tap name → the key the reference dumper used.
fn stage_key(stage: &str) -> &str {
    match stage {
        "engram" => "engram_out",
        "attn" => "attn_out",
        "ffn" => "ffn_out",
        "block" => "block_out",
        other => other,
    }
}

fn report(label: &str, got: &[f32], want: &[f32]) {
    if got.len() != want.len() {
        println!("{label:<16} LEN {} vs {}", got.len(), want.len());
        return;
    }
    let scale = want.iter().fold(0f32, |a, b| a.max(b.abs())).max(1e-9);
    let (mut max_abs, mut at) = (0f32, 0usize);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > max_abs {
            max_abs = d;
            at = i;
        }
    }
    let rel = max_abs / scale;
    let verdict = if rel < 1e-4 { "ok  " } else { "FAIL" };
    println!(
        "{verdict} {label:<16} rel {rel:.3e}  max|Δ| {max_abs:.3e} at {at}  (got {:+.6}, want {:+.6})",
        got[at], want[at]
    );
}
