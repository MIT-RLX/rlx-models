// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Quick Qwen3 forward benchmark for comparing RLX backends against
// external reference timings. Measures run() only after compile and
// parameter upload.

use anyhow::Result;
use rlx_models::qwen3::{
    Qwen3Config, build_qwen3_graph_sized, build_qwen3_graph_sized_last_logits,
};
use rlx_models::weight_map::WeightMap;
use rlx_runtime::Device;
use std::env;
use std::time::Instant;

const TOKEN_POOL: &[u32] = &[
    1, 17, 42, 314, 2718, 9001, 27182, 8128, 65535, 12345, 256, 1024, 4096, 16384, 32768, 100, 200,
    300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
    11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000,
    24000, 25000, 26000, 27000, 28000, 29000, 30000, 31000, 32000, 33000, 34000, 35000, 36000,
    37000, 38000, 39000, 40000, 41000, 42000, 43000, 44000, 45000, 46000, 47000, 48000, 49000,
    50000, 51000, 52000, 53000, 54000, 55000, 56000, 57000, 58000, 59000, 60000, 61000, 62000,
    63000, 64000, 65000, 66000, 67000, 68000, 69000, 70000, 71000, 72000, 73000, 74000, 75000,
    76000, 77000, 78000, 79000, 80000, 81000, 82000, 83000, 84000, 85000, 86000, 87000, 88000,
    89000, 90000, 91000, 92000, 93000, 94000, 95000, 96000, 97000, 98000, 99000, 100000, 101000,
    102000, 103000, 104000, 105000, 106000, 107000, 108000, 109000, 110000,
];

fn main() -> Result<()> {
    let cfg_path = rlx_ir::env::var("RLX_QWEN3_CONFIG")
        .ok_or_else(|| anyhow::anyhow!("set RLX_QWEN3_CONFIG"))?;
    let weights_path = rlx_ir::env::var("RLX_QWEN3_WEIGHTS")
        .ok_or_else(|| anyhow::anyhow!("set RLX_QWEN3_WEIGHTS"))?;
    let cfg = Qwen3Config::from_file(std::path::Path::new(&cfg_path))?;

    let devices = env::args()
        .skip(1)
        .map(|s| parse_device(&s))
        .collect::<Result<Vec<_>>>()?;
    let devices = if devices.is_empty() {
        vec![Device::Cpu, Device::Metal, Device::Mlx, Device::Gpu]
    } else {
        devices
    };

    let last_logits_only = rlx_ir::env::flag("RLX_QWEN3_LAST_ONLY");
    println!("device,B,L,mode,min_ms,median_ms,all_ms");
    for device in devices {
        for batch in [1usize, 2, 4] {
            for seq in [8usize, 32, 64, 128] {
                let ids = make_batched_ids(batch, seq);
                let times = bench_case(
                    &cfg,
                    &weights_path,
                    device,
                    batch,
                    seq,
                    &ids,
                    last_logits_only,
                )?;
                let min = times.iter().copied().fold(f64::INFINITY, f64::min);
                let mut sorted = times.clone();
                sorted.sort_by(|a, b| a.total_cmp(b));
                let median = sorted[sorted.len() / 2];
                println!(
                    "{},{batch},{seq},{},{min:.1},{median:.1},{:?}",
                    device.name(),
                    if last_logits_only { "last" } else { "full" },
                    rounded(&times)
                );
            }
        }
    }
    Ok(())
}

fn bench_case(
    cfg: &Qwen3Config,
    weights_path: &str,
    device: Device,
    batch: usize,
    seq: usize,
    ids: &[u32],
    last_logits_only: bool,
) -> Result<Vec<f64>> {
    let mut wm = WeightMap::from_file(weights_path)?;
    let (graph, params) = if last_logits_only {
        build_qwen3_graph_sized_last_logits(
            cfg, &mut wm, batch, seq, /*with_kv_outputs*/ false,
        )?
    } else {
        build_qwen3_graph_sized(
            cfg, &mut wm, batch, seq, /*with_lm_head*/ true, /*with_kv_outputs*/ false,
        )?
    };
    let mut compiled =
        rlx_models::flow_util::compile_graph_qwen3_prefill_with_params(device, graph, params)?;

    let ids_f32: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    for _ in 0..2 {
        let _ = compiled.run(&[("input_ids", ids_f32.as_slice())]);
    }

    let mut times = Vec::new();
    for _ in 0..5 {
        let t0 = Instant::now();
        let out = compiled.run(&[("input_ids", ids_f32.as_slice())]);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        anyhow::ensure!(
            out.first().is_some_and(|v| {
                let expected_seq = if last_logits_only { 1 } else { seq };
                v.len() == batch * expected_seq * cfg.vocab_size
            }),
            "unexpected logits shape for {:?} B={batch} L={seq}",
            device
        );
        times.push(ms);
    }
    Ok(times)
}

fn make_batched_ids(batch: usize, seq: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(batch * seq);
    for b in 0..batch {
        let offset = (b * 7) % TOKEN_POOL.len();
        for i in 0..seq {
            out.push(TOKEN_POOL[(offset + i) % TOKEN_POOL.len()]);
        }
    }
    out
}

fn parse_device(s: &str) -> Result<Device> {
    match s {
        "cpu" => Ok(Device::Cpu),
        "metal" | "mps" => Ok(Device::Metal),
        "mlx" => Ok(Device::Mlx),
        "gpu" | "wgpu" => Ok(Device::Gpu),
        other => anyhow::bail!("unknown device {other}; use cpu|metal|mps|mlx|gpu|wgpu"),
    }
}

fn rounded(xs: &[f64]) -> Vec<f64> {
    xs.iter().map(|x| (x * 10.0).round() / 10.0).collect()
}
