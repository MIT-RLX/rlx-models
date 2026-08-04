//! `bench_vision_fusion` — wall-clock A/B of the MoonViT vision tower at real
//! head dims (hidden=1024, qkv_hidden=nh·dh=1536, 27 blocks) with the CPU
//! attention fusion ON (default) vs OFF (`RLX_FUSE_ATTN_THRESHOLD=0`).
//!
//! The auto-fusion only fires when `batch·seq ≤ RLX_FUSE_ATTN_THRESHOLD` (64),
//! so this benches at `grid 8×8 = 64` patches by default (fusion active). It
//! compiles once per mode (compile cost reported separately), warms up, then
//! times execution-only iterations.
//!
//!   cargo run -p rlx-kimi-k3 --example bench_vision_fusion --release -- [grid] [iters]

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Shape};
use rlx_kimi_k3::vision::{VisionBlockWeights, VisionDims, VisionWeights, build_vision};
use rlx_runtime::Device;
use std::collections::HashMap;
use std::time::Instant;

fn fill(n: usize, s: u64) -> Vec<f32> {
    let mut x = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((x >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.1
        })
        .collect()
}

fn dims(grid: usize) -> VisionDims {
    VisionDims {
        hidden: 1024,
        qkv_hidden: 1536,
        num_heads: 12,
        head_dim: 128,
        inter: 4096,
        merge: 2,
        text_hidden: 7168,
        proj_mid: 4096,
        eps: 1e-5,
        grid_h: grid,
        grid_w: grid,
    }
}

fn weights(d: &VisionDims) -> VisionWeights {
    let (hid, qh) = (d.hidden, d.qkv_hidden);
    let blocks: Vec<VisionBlockWeights> = (0..27)
        .map(|i| {
            let sd = 100 + i as u64 * 50;
            VisionBlockWeights {
                norm0: vec![1.0; hid],
                wqkv: fill(hid * 3 * qh, sd + 1),
                wo: fill(qh * hid, sd + 2),
                norm1: vec![1.0; hid],
                fc0: fill(hid * d.inter, sd + 3),
                fc1: fill(d.inter * hid, sd + 4),
            }
        })
        .collect();
    VisionWeights {
        blocks,
        final_norm: vec![1.0; hid],
        proj0: fill(d.merge_in() * d.proj_mid, 700),
        proj2: fill(d.proj_mid * d.text_hidden, 701),
        post_norm: vec![1.0; d.text_hidden],
    }
}

/// (mean_run_ms, compile_ms, checksum) for one fusion setting.
fn bench(label: &str, threshold: Option<&str>, grid: usize, iters: usize) -> (f64, f64, f64) {
    match threshold {
        Some(t) => unsafe { std::env::set_var("RLX_FUSE_ATTN_THRESHOLD", t) },
        None => unsafe { std::env::remove_var("RLX_FUSE_ATTN_THRESHOLD") },
    }
    let d = dims(grid);
    let w = weights(&d);
    let (l, hid, hd) = (d.seq_len(), d.hidden, d.head_dim);

    let t_c = Instant::now();
    let mut hir = HirModule::new("vision");
    let mut g = HirMut::new(&mut hir);
    let hh = g.input("hidden", Shape::new(&[1, l, hid], DType::F32));
    let cos = g.input("cos", Shape::new(&[l, hd / 2], DType::F32));
    let sin = g.input("sin", Shape::new(&[l, hd / 2], DType::F32));
    let mut p = HashMap::new();
    let out = build_vision(&mut g, &mut p, hh, cos, sin, &w, d).unwrap();
    g.set_outputs(vec![out]);
    let mut c = compile_built(built_from_hir(hir, p).unwrap(), Device::Cpu).unwrap();
    let compile_ms = t_c.elapsed().as_secs_f64() * 1e3;

    let hin = fill(l * hid, 1);
    let cosd = fill(l * (hd / 2), 2);
    let sind = fill(l * (hd / 2), 3);
    let feed = &[
        ("hidden", hin.as_slice()),
        ("cos", cosd.as_slice()),
        ("sin", sind.as_slice()),
    ];
    // warmup
    let mut y = c.run(feed).remove(0);
    let _ = c.run(feed).remove(0);
    // timed
    let t = Instant::now();
    for _ in 0..iters {
        y = c.run(feed).remove(0);
    }
    let mean_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    let cs: f64 = y.iter().map(|v| *v as f64).sum();
    println!(
        "  {label:<8} seq={l:<4} compile={compile_ms:8.1}ms   run={mean_ms:8.3}ms/iter   checksum={cs:.4}"
    );
    (mean_ms, compile_ms, cs)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let grid: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(30);
    let seq = grid * grid;
    println!(
        "MoonViT vision tower (27 blocks, hidden=1024, 12×128 heads, inter=4096), \
         grid {grid}×{grid} = {seq} patches, {iters} iters, CPU\n"
    );
    let (unf, _, cs_u) = bench("unfused", Some("0"), grid, iters);
    let (fus, _, cs_f) = bench("fused", None, grid, iters);
    let speedup = unf / fus;
    println!(
        "\n  fused is {speedup:.3}× {} than unfused  (Δchecksum={:.2e})",
        if speedup >= 1.0 { "FASTER" } else { "SLOWER" },
        (cs_u - cs_f).abs()
    );
    if seq > 64 {
        println!(
            "  NOTE: seq={seq} > 64 default threshold — fusion did NOT fire (raise RLX_FUSE_ATTN_THRESHOLD to force)."
        );
    }
}
