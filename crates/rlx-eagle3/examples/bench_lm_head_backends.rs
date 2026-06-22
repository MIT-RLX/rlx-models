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

//! Backend smoke-test for the EAGLE3 draft's dominant op.
//!
//! Loads the **real** RedHatAI/gemma-4-31B-it-speculator.eagle3
//! `lm_head.weight` (`[V_draft=32000, H_draft=5376]` = 172M params,
//! ~660 MB f32) and times `lm_head @ x` on every backend feature
//! enabled at build time.
//!
//! This is the single op that dominates per-step cost in the draft
//! forward (the matmul that produces draft-vocab logits). If MLX
//! gives a clear speedup here, the full HIR port of the forward is
//! justified. If it doesn't, the bottleneck is elsewhere (memory
//! bandwidth on unified memory, per-token launch overhead, etc.).
//!
//! Run with:
//! ```bash
//! # CPU only
//! cargo run -p rlx-eagle3 --release --example bench_lm_head_backends -- \
//!     /Users/Shared/rlx-models/.eagle3-bench/weights/draft
//!
//! # Add MLX (Apple Silicon)
//! cargo run -p rlx-eagle3 --release --features mlx --example bench_lm_head_backends -- ...
//!
//! # Add Metal
//! cargo run -p rlx-eagle3 --release --features metal --example bench_lm_head_backends -- ...
//! ```

use anyhow::Result;
use rlx_eagle3::weights::Eagle3DraftWeights;
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Graph, HirGraphExt, Shape, hir_to_graph};
use rlx_runtime::{Device, Session, is_available};
use std::path::PathBuf;
use std::time::Instant;

const WARMUP: usize = 5;
const ITERS: usize = 100;
const BATCHES: &[usize] = &[1, 16];
/// Dtypes to sweep. F16 + Bf16 may pick a different MLX matmul kernel.
const DTYPES: &[DType] = &[DType::F32, DType::F16, DType::BF16];

/// Build `x[1, B, H] @ W[H, V] = logits[1, B, V]` — batched matvec.
/// HIR's `mm` expects `[..., K] @ [K, N]`. We bench multiple
/// batch sizes to see whether MLX amortizes per-call overhead.
fn build_lm_head_graph(v: usize, h: usize, batch: usize, dtype: DType) -> Graph {
    let mut hir = HirModule::new("lm_head_bench");
    let mut gb = HirMut::new(&mut hir);
    let x = gb.input("x", Shape::new(&[1, batch, h], dtype));
    let w = gb.param("lm_head", Shape::new(&[h, v], dtype));
    let logits = gb.mm(x, w);
    gb.set_outputs(vec![logits]);
    hir_to_graph(hir).expect("hir → graph lowers cleanly for a single mm")
}

/// Transpose `[V, H]` row-major → `[H, V]` row-major. 656 MB allocation
/// happens once at startup, not in the hot loop.
fn transpose_to_hv(weight_vh: &[f32], v: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; v * h];
    for row in 0..v {
        for col in 0..h {
            // out[col, row] = in[row, col]
            out[col * v + row] = weight_vh[row * h + col];
        }
    }
    out
}

fn bench_device(
    device: Device,
    label: &str,
    weight: &[f32],
    v: usize,
    h: usize,
    batch: usize,
    dtype: DType,
    x: &[f32],
) -> Result<(f64, f64)> {
    if !is_available(device) {
        return Ok((f64::NAN, f64::NAN));
    }

    let graph = build_lm_head_graph(v, h, batch, dtype);
    let session = Session::new(device);
    let mut compiled = session.compile(graph);
    compiled.set_param("lm_head", weight);

    // Warmup
    for _ in 0..WARMUP {
        let _ = compiled.run(&[("x", x)]);
    }

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = compiled.run(&[("x", x)]);
    }
    let total = t0.elapsed().as_secs_f64();
    let per_call_ms = total * 1000.0 / ITERS as f64;
    let per_row_us = per_call_ms * 1000.0 / batch as f64;

    let outs = compiled.run(&[("x", x)]);
    let logits = outs.first().ok_or_else(|| anyhow::anyhow!("no output"))?;
    let expected = v * batch;
    if logits.len() != expected {
        anyhow::bail!(
            "[{label} b={batch} {dtype:?}] expected {expected} logits, got {}",
            logits.len()
        );
    }
    if logits.iter().any(|v| !v.is_finite()) {
        anyhow::bail!("[{label} b={batch} {dtype:?}] non-finite logits");
    }

    let dtype_label = match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        _ => "??",
    };
    println!(
        "   [{label:6} {dtype_label} b={batch:>3}] {per_call_ms:7.3} ms/call · {per_row_us:7.1} µs/row",
    );
    Ok((per_call_ms, per_row_us))
}

fn main() -> Result<()> {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: bench_lm_head_backends <draft-dir>"))?;
    let model_path = dir.join("model.safetensors");

    println!("→ Loading lm_head.weight from {:?}", model_path);
    let t0 = Instant::now();
    let weights = Eagle3DraftWeights::open(&model_path)?;
    println!(
        "   total checkpoint loaded in {:.2}s",
        t0.elapsed().as_secs_f64()
    );
    let lm_head = weights
        .get("lm_head.weight")
        .ok_or_else(|| anyhow::anyhow!("lm_head.weight missing"))?;
    let v = lm_head.shape[0];
    let h = lm_head.shape[1];
    println!(
        "   lm_head shape = [{v}, {h}]  ({:.1} MB f32)",
        (v * h * 4) as f64 / 1024.0 / 1024.0
    );

    // Largest batch's input; smaller batches read a prefix.
    let max_batch = *BATCHES.iter().max().unwrap();
    let x_big: Vec<f32> = (0..max_batch * h)
        .map(|i| ((i as f32) * 0.001).sin())
        .collect();

    // Transpose [V, H] → [H, V] once so HIR's `mm(x, w)` consumes
    // the standard `[..., K] @ [K, N]` shape.
    println!("\n→ Transposing lm_head to [H, V] for HIR `mm` convention...");
    let t0 = Instant::now();
    let lm_head_hv = transpose_to_hv(&lm_head.data, v, h);
    println!("   {:.2}s", t0.elapsed().as_secs_f64());

    println!("\n→ Benching `lm_head @ x` — {ITERS} iterations + {WARMUP} warmup\n");

    let backends = [
        (Device::Cpu, "CPU"),
        (Device::Metal, "Metal"),
        (Device::Mlx, "MLX"),
    ];
    let mut grid: Vec<(String, &'static str, usize, f64, f64)> = Vec::new();
    for (device, label) in backends {
        if !is_available(device) {
            println!("   [{label}] not available — skipped");
            continue;
        }
        for &dtype in DTYPES {
            let dtype_label: &'static str = match dtype {
                DType::F32 => "f32",
                DType::F16 => "f16",
                DType::BF16 => "bf16",
                _ => "??",
            };
            for &batch in BATCHES {
                let x = &x_big[..batch * h];
                let (per_call_ms, per_row_us) =
                    bench_device(device, label, &lm_head_hv, v, h, batch, dtype, x)?;
                if !per_call_ms.is_nan() {
                    grid.push((
                        label.to_string(),
                        dtype_label,
                        batch,
                        per_call_ms,
                        per_row_us,
                    ));
                }
            }
        }
        println!();
    }

    // Summary table: dtype × batch grid of per-row µs.
    for label in &["CPU", "Metal", "MLX"] {
        println!("\n→ {label} per-row µs (smaller is faster):");
        print!("   {:<5}", "Batch");
        for &dt in &["f32", "f16", "bf16"] {
            print!("{dt:>10}");
        }
        println!();
        for &b in BATCHES {
            print!("   b={b:<3}");
            for &dt in &["f32", "f16", "bf16"] {
                let cell = grid
                    .iter()
                    .find(|(l, d, bb, _, _)| l == label && *d == dt && *bb == b)
                    .map(|(_, _, _, _, per_row)| *per_row);
                match cell {
                    Some(v) => print!("{v:>10.1}"),
                    None => print!("{:>10}", "-"),
                }
            }
            println!();
        }
    }

    // MLX f16/bf16 vs MLX f32 — does dtype unlock kernel speedup?
    println!("\n→ MLX dtype speedup vs MLX f32 at each batch size:");
    for &b in BATCHES {
        let f32_us = grid
            .iter()
            .find(|(l, d, bb, _, _)| l == "MLX" && *d == "f32" && *bb == b)
            .map(|(_, _, _, _, p)| *p);
        if let Some(f32) = f32_us {
            print!("   b={b:>3} →");
            for &dt in &["f16", "bf16"] {
                let dev = grid
                    .iter()
                    .find(|(l, d, bb, _, _)| l == "MLX" && *d == dt && *bb == b)
                    .map(|(_, _, _, _, p)| *p);
                if let Some(d) = dev {
                    print!("  {dt} {:.2}×", f32 / d);
                }
            }
            println!();
        }
    }

    println!("\n✓ DONE.");
    Ok(())
}
