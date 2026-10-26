// RLX — versatile ML compiler + runtime. GPLv3.
//! Is one compiled session self-consistent?
//!
//! A graph that returns a different answer for the same inputs on the same
//! device is the worst kind of bug: every parity number becomes a coin flip, and
//! a test that passes proves nothing. This runs one graph repeatedly and reports
//! *which* elements move, which is usually enough to point at the op.
//!
//! ```sh
//! RLX_DSV41_WEIGHTS=/tmp/dsv41w cargo run --release -p rlx-models-core \
//!     --example dsv41_determinism_probe
//! ```

use rlx_models_core::dsv41::DeepseekV41Spec;
use rlx_models_core::dsv41_graph::{StageSpan, V41Inputs, build_deepseek_v41_stage};
use rlx_models_core::dsv41_quant::{DEFAULT_BLOCK, DsV41Loader};
use rlx_models_core::weight_loader::WeightLoader;
use rlx_runtime::{Device, Session};
use serde_json::Value;

fn main() -> anyhow::Result<()> {
    let dir = rlx_ir::env::var_result("RLX_DSV41_WEIGHTS")
        .map(std::path::PathBuf::from)
        .expect("set RLX_DSV41_WEIGHTS");
    let layer: usize = rlx_ir::env::var_result("RLX_DSV41_LAYER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let runs: usize = rlx_ir::env::var_result("RLX_DSV41_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let seq: usize = rlx_ir::env::var_result("RLX_DSV41_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(160);
    let stage = rlx_ir::env::var_result("RLX_DSV41_DBG").unwrap_or_else(|_| "attn".into());

    let root = env!("CARGO_MANIFEST_DIR");
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(format!(
        "{root}/tests/fixtures/dsv41_config.json"
    ))?)?;
    let spec = DeepseekV41Spec::from_config(&cfg)?;

    let mut loader = DsV41Loader::open(&dir, DEFAULT_BLOCK)?;
    let (rows, rshape) = loader.take("embed.rows")?;
    assert!(rshape[0] >= seq);
    let mut hidden = vec![0f32; seq * spec.hc_mult * spec.dim];
    for t in 0..seq {
        for c in 0..spec.hc_mult {
            let dst = (t * spec.hc_mult + c) * spec.dim;
            hidden[dst..dst + spec.dim].copy_from_slice(&rows[t * spec.dim..(t + 1) * spec.dim]);
        }
    }
    let mut pre_mix = vec![0f32; seq * spec.hc_mult];
    for t in 0..seq {
        pre_mix[t * spec.hc_mult] = 1.0;
    }

    // `RLX_DSV41_DBG=none` builds the stage with no tap at all, so the graph
    // ends at its natural output instead of an intermediate — which is the
    // difference that matters if output buffers are being reused too early.
    let tapped = stage != "none";
    if tapped {
        rlx_ir::env::set("RLX_DSV41_DBG", &stage);
        rlx_ir::env::set("RLX_DSV41_DBGLAYER", layer.to_string());
    }
    let mut loader = DsV41Loader::open(&dir, DEFAULT_BLOCK)?;
    let mut packed = std::collections::HashMap::new();
    let span = StageSpan::middle(layer..layer + 1);
    let built = build_deepseek_v41_stage(
        &spec,
        &mut loader,
        seq,
        &span,
        &V41Inputs::default(),
        &mut packed,
    );
    rlx_ir::env::remove("RLX_DSV41_DBG");
    rlx_ir::env::remove("RLX_DSV41_DBGLAYER");
    let (g, params, _) = built?;
    let n_nodes = g.nodes().len();

    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    );
    let mut sess = Session::new(Device::Cpu).compile_with(g, &opts);
    for (n, d) in &params {
        sess.set_param(n, d);
    }

    println!(
        "layer {layer}, {}, seq {seq}, {n_nodes} nodes, {runs} runs",
        if tapped {
            format!("tap `{stage}`")
        } else {
            "no tap".into()
        }
    );
    // `RLX_DSV41_FRESH=1` recompiles a new session for every run, which tells
    // session reuse apart from anything inherent to the graph.
    let fresh = rlx_ir::env::var_result("RLX_DSV41_FRESH").is_ok();
    let mut first: Option<Vec<f32>> = None;
    let mut ever_differed = 0usize;
    for r in 0..runs {
        if fresh {
            let mut l = DsV41Loader::open(&dir, DEFAULT_BLOCK)?;
            let mut pk = std::collections::HashMap::new();
            if tapped {
                rlx_ir::env::set("RLX_DSV41_DBG", &stage);
                rlx_ir::env::set("RLX_DSV41_DBGLAYER", layer.to_string());
            }
            let b =
                build_deepseek_v41_stage(&spec, &mut l, seq, &span, &V41Inputs::default(), &mut pk);
            rlx_ir::env::remove("RLX_DSV41_DBG");
            rlx_ir::env::remove("RLX_DSV41_DBGLAYER");
            let (g2, p2, _) = b?;
            sess = Session::new(Device::Cpu).compile_with(g2, &opts);
            for (n, d) in &p2 {
                sess.set_param(n, d);
            }
        }
        let out = sess.run(&[
            ("hidden_in", hidden.as_slice()),
            ("pre_mix_in", pre_mix.as_slice()),
        ])[0]
            .clone();
        match &first {
            None => {
                println!(
                    "  run 0: {} elements, absmean {:.9}",
                    out.len(),
                    absmean(&out)
                );
                first = Some(out);
            }
            Some(f) => {
                let diffs: Vec<usize> = (0..out.len()).filter(|&i| out[i] != f[i]).collect();
                if diffs.is_empty() {
                    println!("  run {r}: identical");
                } else {
                    ever_differed += 1;
                    let maxd = diffs
                        .iter()
                        .map(|&i| (out[i] - f[i]).abs())
                        .fold(0f32, f32::max);
                    // where in the [rows, width] output do they sit?
                    let width = out.len() / seq;
                    let rows_hit: std::collections::BTreeSet<usize> =
                        diffs.iter().map(|i| i / width).collect();
                    let cols_hit: std::collections::BTreeSet<usize> =
                        diffs.iter().map(|i| i % width).collect();
                    println!(
                        "  run {r}: {} of {} elements differ, max |Δ| {maxd:.3e}, absmean {:.9}",
                        diffs.len(),
                        out.len(),
                        absmean(&out)
                    );
                    println!(
                        "           rows {}..={} ({} of {seq}), cols {}..={} ({} of {width})",
                        rows_hit.first().unwrap(),
                        rows_hit.last().unwrap(),
                        rows_hit.len(),
                        cols_hit.first().unwrap(),
                        cols_hit.last().unwrap(),
                        cols_hit.len(),
                    );
                }
            }
        }
    }
    if ever_differed > 0 {
        eprintln!(
            "\n{ever_differed}/{} repeat runs disagreed with the first",
            runs - 1
        );
        std::process::exit(1);
    }
    println!("all {runs} runs identical");
    Ok(())
}

fn absmean(v: &[f32]) -> f64 {
    v.iter().map(|x| x.abs() as f64).sum::<f64>() / v.len() as f64
}
