// RLX — versatile ML compiler + runtime. GPLv3.
//! Minimal repro for a non-deterministic matmul.
//!
//! DeepSeek-V4/V4.1 attention is MQA: one `kv` latent serves as both key and
//! value, so the graph uses the *same* tensor twice —
//!
//! ```text
//! scores = q @ transpose(kv)     // kv as keys
//! out    = attn @ kv             // kv as values, untransposed
//! ```
//!
//! which is precisely the shape a `transpose → matmul` fold has to be careful
//! with: the transposed operand is not dead after the fold. This builds that
//! pattern alone and runs it repeatedly, so a corrupting fold shows up without
//! any model around it.
//!
//! `cargo run --release -p rlx-models-core --example dsv41_matmul_probe`

use rlx_ir::GraphExt;
use rlx_ir::graph::Graph;
use rlx_ir::{DType, Shape};
use rlx_models_core::weight_loader::SyntheticLoader;
use rlx_runtime::{Device, Session};
use std::collections::HashMap;

/// `out = (q @ kvᵀ) @ kv`, with `kv` deliberately used both ways.
fn build(m: usize, k: usize, n: usize, reuse: bool) -> (Graph, HashMap<String, Vec<f32>>) {
    let mut g = Graph::new("matmul_probe");
    let mut p: HashMap<String, Vec<f32>> = HashMap::new();
    let q = g.param("q", Shape::new(&[m, k], DType::F32));
    p.insert("q".into(), SyntheticLoader::values("probe.a1", &[m * k]));
    let kv = g.param("kv", Shape::new(&[n, k], DType::F32));
    p.insert("kv".into(), SyntheticLoader::values("probe.b2", &[n * k]));
    // a second, identical tensor — so the "value" matmul reads its own copy
    let kv2 = if reuse {
        kv
    } else {
        let id = g.param("kv2", Shape::new(&[n, k], DType::F32));
        p.insert("kv2".into(), SyntheticLoader::values("probe.b2", &[n * k]));
        id
    };
    let kt = g.transpose_(kv, vec![1, 0]); // [k, n]
    let sc = g.mm(q, kt); // [m, n]
    let sc = g.sm(sc, -1);
    let out = g.mm(sc, kv2); // [m, k]
    g.set_outputs(vec![out]);
    (g, p)
}

fn run_many(label: &str, m: usize, k: usize, n: usize, reuse: bool, runs: usize) -> bool {
    let (g, params) = build(m, k, n, reuse);
    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    );
    let mut s = Session::new(Device::Cpu).compile_with(g, &opts);
    for (nm, d) in &params {
        s.set_param(nm, d);
    }
    let first = s.run(&[])[0].clone();
    let mut bad = 0;
    for _ in 1..runs {
        let out = s.run(&[])[0].clone();
        if out != first {
            bad += 1;
        }
    }
    println!(
        "{} {label:<44} {}/{} repeats differ",
        if bad == 0 { "ok  " } else { "FAIL" },
        bad,
        runs - 1
    );
    bad == 0
}

/// The real thing: `build_v4_sink_attention` as the model builds it.
fn run_sink(rows: usize, nh: usize, hd: usize, nk: usize, runs: usize) -> bool {
    use rlx_models_core::standard_decoder::build_v4_sink_attention;
    let mut g = Graph::new("sink_probe");
    let mut p: HashMap<String, Vec<f32>> = HashMap::new();
    let q = g.param("q", Shape::new(&[rows, nh, hd], DType::F32));
    p.insert(
        "q".into(),
        SyntheticLoader::values("probe.11", &[rows * nh * hd]),
    );
    let kv = g.param("kv", Shape::new(&[nk, hd], DType::F32));
    p.insert("kv".into(), SyntheticLoader::values("probe.22", &[nk * hd]));
    let sink = g.param("sink", Shape::new(&[nh], DType::F32));
    p.insert("sink".into(), SyntheticLoader::values("probe.33", &[nh]));
    // a sliding-window causal mask, as the model builds it
    let mut m = vec![0f32; rows * nk];
    for qi in 0..rows {
        for ki in 0..nk {
            if ki > qi || qi - ki >= 128 {
                m[qi * nk + ki] = -1e30;
            }
        }
    }
    let mask = g.param("mask", Shape::new(&[rows, nk], DType::F32));
    p.insert("mask".into(), m);
    let o = build_v4_sink_attention(
        &mut g,
        &mut p,
        q,
        kv,
        mask,
        sink,
        (hd as f32).powf(-0.5),
        rows,
        nh,
        hd,
        nk,
        "probe",
    );
    g.set_outputs(vec![o]);

    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    );
    let mut s = Session::new(Device::Cpu).compile_with(g, &opts);
    for (nm, d) in &p {
        s.set_param(nm, d);
    }
    let first = s.run(&[])[0].clone();
    let mut bad = 0;
    for _ in 1..runs {
        if s.run(&[])[0] != first {
            bad += 1;
        }
    }
    let label = format!("sink_attention rows={rows} heads={nh} hd={hd} keys={nk}");
    println!(
        "{} {label:<44} {}/{} repeats differ",
        if bad == 0 { "ok  " } else { "FAIL" },
        bad,
        runs - 1
    );
    bad == 0
}

fn main() {
    let runs: usize = rlx_ir::env::var_result("RLX_DSV41_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    // the released V4.1 attention shape: 160 queries x 64 heads, head_dim 512,
    // 160 keys
    let (m, k, n) = (160 * 64, 512, 160);
    let mut ok = true;
    ok &= run_many(
        "kv reused as key (transposed) and value",
        m,
        k,
        n,
        true,
        runs,
    );
    ok &= run_many("two separate tensors (control)", m, k, n, false, runs);
    // smaller, to see whether it is size-dependent
    ok &= run_many("small: kv reused", 12 * 2, 32, 12, true, runs);
    // and the op the model actually calls, at the released shape
    ok &= run_sink(160, 64, 512, 160, runs);
    ok &= run_sink(160, 64, 512, 160 + 80, runs); // with compressed keys appended
    ok &= run_sink(12, 2, 32, 12, runs); // toy scale, for contrast
    if !ok {
        std::process::exit(1);
    }
}
