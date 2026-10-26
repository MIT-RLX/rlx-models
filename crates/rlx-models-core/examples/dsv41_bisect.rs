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
use rlx_models_core::parity::Deviation;
use rlx_models_core::weight_loader::SyntheticLoader;
use rlx_runtime::{Device, Session};

/// `RLX_DSV41_DEVICE=metal|mlx|gpu|cpu` — which backend to bisect.
fn device() -> Device {
    match rlx_ir::env::var_result("RLX_DSV41_DEVICE")
        .unwrap_or_default()
        .as_str()
    {
        #[cfg(feature = "metal")]
        "metal" => Device::Metal,
        #[cfg(feature = "mlx")]
        "mlx" => Device::Mlx,
        #[cfg(feature = "gpu")]
        "gpu" => Device::Gpu,
        "" | "cpu" => Device::Cpu,
        other => panic!("device `{other}` is not enabled in this build"),
    }
}
use serde_json::Value;
use std::collections::BTreeMap;

fn main() -> anyhow::Result<()> {
    let path = rlx_ir::env::var_result("RLX_DSV41_REF").unwrap_or_else(|_| {
        format!(
            "{}/tests/fixtures/dsv41_toy_ref.json",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    eprintln!("bisecting on {:?}", device());
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
        ..Default::default()
    };

    let run_on = |dev: Device| -> anyhow::Result<Vec<f32>> {
        let mut loader = SyntheticLoader::new(shapes.clone());
        let mut packed = std::collections::HashMap::new();
        let (g, params, _) =
            build_deepseek_v41_prefill(&spec, &mut loader, ids.len(), &inputs, &mut packed)?;
        let opts =
            rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
                &rlx_flow::CompileProfile::qwen3_prefill(),
                dev,
            );
        let mut c = Session::new(dev).compile_with(g, &opts);
        for (n, d) in &params {
            c.set_param(n, d);
        }
        Ok(c.run(&[("input_ids", ids.as_slice())])[0].clone())
    };

    let run = || -> anyhow::Result<Vec<f32>> {
        let mut loader = SyntheticLoader::new(shapes.clone());
        let mut packed = std::collections::HashMap::new();
        let (g, params, _) =
            build_deepseek_v41_prefill(&spec, &mut loader, ids.len(), &inputs, &mut packed)?;
        let dev = device();
        let opts =
            rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
                &rlx_flow::CompileProfile::qwen3_prefill(),
                dev,
            );
        let mut c = Session::new(dev).compile_with(g, &opts);
        for (n, d) in &params {
            c.set_param(n, d);
        }
        Ok(c.run(&[("input_ids", ids.as_slice())])[0].clone())
    };

    let empty = serde_json::Map::new();
    let inter = fx["inter"].as_object().unwrap_or(&empty);
    let all_stages = [
        "engram", "xa", "comp", "compkv", "topk", "sa", "oinv", "attn", "ffn", "block",
    ];
    let mut plan: Vec<(String, usize)> = Vec::new();
    for il in 0..spec.n_layers {
        for stage in all_stages {
            if inter.contains_key(&format!("{}.{il}", stage_key(stage))) {
                plan.push((stage.to_string(), il));
            }
        }
    }

    // Device-vs-device: CPU is the known-good side, so a backend bug can be
    // bisected without any reference dump — and at every tap, not just the ones
    // the Python harness happened to hook.
    if device() != Device::Cpu {
        println!("── {:?} vs Cpu ──", device());
        for il in 0..spec.n_layers {
            for stage in all_stages {
                rlx_ir::env::set("RLX_DSV41_DBG", stage);
                rlx_ir::env::set("RLX_DSV41_DBGLAYER", il.to_string());
                let (Ok(cpu), Ok(dev)) = (run_on(Device::Cpu), run_on(device())) else {
                    continue; // this tap does not exist on this layer
                };
                report(&format!("{stage}.{il}"), &dev, &cpu);
            }
        }
        rlx_ir::env::remove("RLX_DSV41_DBG");
        rlx_ir::env::remove("RLX_DSV41_DBGLAYER");
        let (cpu, dev) = (run_on(Device::Cpu)?, run_on(device())?);
        report("logits", &dev, &cpu);
        return Ok(());
    }

    for (stage, il) in plan {
        rlx_ir::env::set("RLX_DSV41_DBG", &stage);
        rlx_ir::env::set("RLX_DSV41_DBGLAYER", il.to_string());
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
                    let mut got_set: Vec<usize> =
                        (0..ncomp).filter(|&c| got[q * ncomp + c] > -30.0).collect();
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

    rlx_ir::env::remove("RLX_DSV41_DBG");
    rlx_ir::env::remove("RLX_DSV41_DBGLAYER");
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
    let d = Deviation::between(got, want);
    println!(
        "{} {label:<16} {d}",
        if d.is_within(1e-4) { "ok  " } else { "FAIL" }
    );
}
