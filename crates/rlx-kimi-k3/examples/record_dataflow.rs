//! `record_dataflow` — use the **rlx-opscope** recording harness to RECORD the
//! data flowing through **every real KDA (69) or MLA (24) layer** of Kimi-K3.
//! For each layer it loads the real weights (`CheckpointLoader`), injects matmul
//! stat-taps (density / per-channel outliers / histogram / temporal drift) on
//! every matmul's lhs/rhs/out, runs the layer over a few activation steps, and
//! records the sketches (per-layer `dist = L{i}`) to one CSV that `opscope-mine`
//! reads — revealing how quantizability (per-channel int8 headroom) evolves with
//! depth across the attention stack.
//!
//!   cargo run -p rlx-kimi-k3 --features cluster --example record_dataflow -- \
//!       out.csv [steps] [seq] [model_dir] [kda|mla]
//!   (from ../rlx) cargo run -p rlx-opscope --bin opscope-mine -- out.csv

use rlx_core::flow_util::graph_from_hir;
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Op, Philox4x32, Shape};
use rlx_kimi_k3::config::KimiK3Config;
use rlx_kimi_k3::kda::{KdaDims, build_kda_layer};
use rlx_kimi_k3::loader::CheckpointLoader;
use rlx_kimi_k3::mla::{MlaDims, build_mla_layer};
use rlx_opscope::{Recorder, StatConfig, inject_matmul_stats};
use rlx_runtime::{Device, Session};
use std::collections::HashMap;
use std::path::Path;

type Built = (HirModule, HashMap<String, Vec<f32>>);

fn build_kda(ck: &mut CheckpointLoader, i: usize, d: KdaDims) -> Result<Built, String> {
    let w = ck
        .load_kda(&format!("language_model.model.layers.{i}"), d)
        .map_err(|e| format!("kda {i}: {e}"))?;
    let mut hir = HirModule::new("kda");
    let mut g = HirMut::new(&mut hir);
    let hin = g.input("h", Shape::new(&[1, d.seq, d.hidden], DType::F32));
    let mut p = HashMap::new();
    let out = build_kda_layer(&mut g, &mut p, "kda", hin, &w, d).map_err(|e| e.to_string())?;
    g.set_outputs(vec![out]);
    Ok((hir, p))
}

fn build_mla(ck: &mut CheckpointLoader, i: usize, d: MlaDims) -> Result<Built, String> {
    let w = ck
        .load_mla(&format!("language_model.model.layers.{i}"), d)
        .map_err(|e| format!("mla {i}: {e}"))?;
    let mut hir = HirModule::new("mla");
    let mut g = HirMut::new(&mut hir);
    let hin = g.input("h", Shape::new(&[1, d.seq, d.hidden], DType::F32));
    let mut p = HashMap::new();
    let out = build_mla_layer(&mut g, &mut p, "mla", hin, &w, d).map_err(|e| e.to_string())?;
    g.set_outputs(vec![out]);
    Ok((hir, p))
}

fn main() -> Result<(), String> {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "kimi_flow.csv".into());
    let steps: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let seq: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let model_dir = std::env::args()
        .nth(4)
        .unwrap_or_else(|| "/Volumes/FOUR/kimi".into());
    let unit = std::env::args().nth(5).unwrap_or_else(|| "kda".into());

    if !Path::new(&model_dir).join("config.json").exists() {
        eprintln!("skip: {model_dir}/config.json not found");
        return Ok(());
    }
    let kc =
        KimiK3Config::load(Path::new(&model_dir).join("config.json")).map_err(|e| e.to_string())?;
    let tc = &kc.text_config;
    let hidden = tc.hidden_size;
    let is_mla = unit == "mla";
    let layers: Vec<usize> = (0..tc.num_hidden_layers)
        .filter(|&i| tc.is_kda_layer(i) != is_mla)
        .collect();
    eprintln!(
        "[opscope] recording {} REAL {} layers of {} total, seq={seq} x {steps} steps -> {out}",
        layers.len(),
        unit.to_uppercase(),
        tc.num_hidden_layers
    );

    let kda = KdaDims {
        hidden,
        num_heads: 96,
        head_dim: 128,
        conv_kernel: 4,
        gate_lower_bound: Some(-5.0),
        eps: 1e-5,
        batch: 1,
        seq,
    };
    let mla = MlaDims {
        hidden,
        num_heads: 96,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 128,
        qk_rope_head_dim: 64,
        v_head_dim: 128,
        eps: 1e-5,
        batch: 1,
        seq,
    };

    let mut ck = CheckpointLoader::open(&model_dir).map_err(|e| e.to_string())?;
    let mut rng = Philox4x32::new(0xC0FFEE);
    let mut rec = Recorder::create(&out).map_err(|e| e.to_string())?;
    let numel = seq * hidden;
    let mut n_sketches = 0usize;

    for &i in &layers {
        let t0 = std::time::Instant::now();
        let (hir, params_hir) = if is_mla {
            build_mla(&mut ck, i, mla)?
        } else {
            build_kda(&mut ck, i, kda)?
        };
        let (graph, params) = graph_from_hir(hir, params_hir).map_err(|e| e.to_string())?;

        let (ginj, specs) = inject_matmul_stats(&graph, &StatConfig::default());
        n_sketches = specs.len();
        let input_name = graph
            .nodes()
            .iter()
            .find_map(|n| match &n.op {
                Op::Input { name } => Some(name.clone()),
                _ => None,
            })
            .expect("no input");
        let mut compiled = Session::new(Device::Cpu).compile(ginj);
        for (name, data) in &params {
            compiled.set_param(name, data);
        }

        let dist = format!("L{i:03}");
        for step in 0..steps {
            let mut x = vec![0f32; numel];
            rng.fill_normal(&mut x);
            let outs = compiled.run(&[(input_name.as_str(), x.as_slice())]);
            rec.record(i as u64, step, "cpu", &dist, 1, hidden, 0, &specs, &outs)
                .map_err(|e| e.to_string())?;
        }
        eprintln!(
            "  {} layer {i:>3}: recorded ({:.1}s)",
            unit.to_uppercase(),
            t0.elapsed().as_secs_f64()
        );
    }
    rec.flush().map_err(|e| e.to_string())?;
    eprintln!(
        "[opscope] done: {} {} layers x {} matmul-sketches/step x {steps} -> {out}",
        layers.len(),
        unit.to_uppercase(),
        n_sketches
    );
    Ok(())
}
