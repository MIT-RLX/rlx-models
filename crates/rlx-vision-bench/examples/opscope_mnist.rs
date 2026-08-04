// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Run the **rlx-opscope** harness on the **real** rlx-vision-bench MNIST model
//! (`build_eval_graph`) fed **real** MNIST images (`datasets::load`), inside the
//! rlx-models workspace. Requires `just link-local` so rlx-* resolve to ../rlx.
//!
//!   cargo run -p rlx-vision-bench --example opscope_mnist --release -- /tmp/m.csv
//!   (from ../rlx) cargo run -p rlx-opscope --bin opscope-mine -- /tmp/m.csv

use rlx_ir::{Op, Philox4x32};
use rlx_opscope::{Recorder, StatConfig, inject_matmul_stats};
use rlx_runtime::{Device, Session};
use rlx_vision_bench::datasets::{self, DatasetKind};
use rlx_vision_bench::{Config, build_eval_graph};

fn main() -> Result<(), String> {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "opscope_mnist.csv".into());
    let steps: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);

    let cfg = Config::default(); // Mlp, 784 → hidden → 10, ReLU
    let batch = cfg.batch;
    let g = build_eval_graph(&cfg);

    let input_name = g
        .nodes()
        .iter()
        .find_map(|n| match &n.op {
            Op::Input { name } => Some(name.clone()),
            _ => None,
        })
        .expect("eval graph has no input");
    let params: Vec<(String, Vec<usize>)> = g
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Param { name } => Some((
                name.clone(),
                (0..n.shape.rank())
                    .map(|i| n.shape.dim(i).unwrap_static())
                    .collect(),
            )),
            _ => None,
        })
        .collect();

    let (ginj, specs) = inject_matmul_stats(&g, &StatConfig::default());
    let mut compiled = Session::new(Device::Cpu).compile(ginj);

    // Random He-scaled weights, zero biases; fixed across steps (untrained is
    // fine — MNIST input + post-ReLU sparsity are training-independent).
    let mut rng = Philox4x32::new(0xC0FFEE);
    for (name, dims) in &params {
        let numel: usize = dims.iter().product();
        let mut data = vec![0f32; numel];
        if dims.len() >= 2 {
            rng.fill_normal(&mut data);
            let scale = (2.0 / dims[0] as f32).sqrt();
            for v in &mut data {
                *v *= scale;
            }
        }
        compiled.set_param(name, &data);
    }

    let data = datasets::load(DatasetKind::Mnist, Some(batch * steps as usize), None)?;
    let px = data.train.pixels();
    let steps = (data.train.len() / batch).min(steps as usize) as u64;

    let mut rec = Recorder::create(&out).map_err(|e| e.to_string())?;
    for step in 0..steps {
        let lo = step as usize * batch * px;
        let x = &data.train.images[lo..lo + batch * px];
        let outs = compiled.run(&[(input_name.as_str(), x)]);
        rec.record(0, step, "cpu", "mnist", batch, px, 0, &specs, &outs)
            .map_err(|e| e.to_string())?;
    }
    rec.flush().map_err(|e| e.to_string())?;
    eprintln!(
        "[opscope] REAL rlx-vision-bench MLP + REAL MNIST ({} params, {} sketches/step) × {steps} → {out}",
        params.len(),
        specs.len()
    );
    Ok(())
}
