//! `cargo run --release -p rlx-tinystories --example inspect_ops`
//!
//! Op-level inspection of the forward + backward graph — **timing, shapes, and
//! data flow** — using the `rlx-opscope` cost model (per-op FLOPs, DRAM bytes,
//! arithmetic intensity → roofline memory-vs-compute classification).
//!
//! Answers "where does the backward spend its time and *why*": low arithmetic
//! intensity (FLOP/byte) = memory-bandwidth-bound (parallelizing won't help,
//! only reading less will); high = compute-bound.

use std::collections::BTreeMap;

use rlx_opscope::shapes::{DEFAULT_RIDGE, OpCost, gemm_shape_histogram, op_costs, roofline_class};
use rlx_tensor::{DType, Device, is_available};
use rlx_tinystories::config::GptConfig;
use rlx_tinystories::model;

/// (count, Σflops, Σbytes) per op kind.
fn agg(costs: &[OpCost]) -> BTreeMap<String, (usize, u64, u64)> {
    let mut m: BTreeMap<String, (usize, u64, u64)> = BTreeMap::new();
    for c in costs {
        let e = m.entry(c.op.clone()).or_insert((0, 0, 0));
        e.0 += 1;
        e.1 += c.flops;
        e.2 += c.bytes;
    }
    m
}

fn main() {
    let cfg = {
        let mut c = GptConfig::default_metal();
        c.batch = 32;
        c.block_size = 256;
        c.n_layer = 6;
        c.n_embd = 256;
        c.n_head = 8;
        c
    };
    let bt = cfg.batch * cfg.block_size;
    println!(
        "=== inspect_ops: {}L·{}d·{}h, batch {}, seq {} (B·T={bt}) ===\n",
        cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.batch, cfg.block_size
    );

    let m = model::build(&cfg, cfg.batch, true, DType::F32);
    let vg = m.value_and_grad_all();
    let fwd_costs = op_costs(m.graph());
    let all_costs = op_costs(vg.graph());

    // ── Roofline table (fwd+bwd), sorted by DRAM traffic ────────────────────
    let a = agg(&all_costs);
    let mut rows: Vec<(&String, &(usize, u64, u64))> = a.iter().collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1.2)); // by bytes desc
    let (tot_flops, tot_bytes): (u64, u64) = all_costs
        .iter()
        .fold((0, 0), |(f, b), c| (f + c.flops, b + c.bytes));
    println!(
        "── ROOFLINE (fwd+bwd graph, {} nodes; ridge {:.0} FLOP/byte) ──",
        all_costs.len(),
        DEFAULT_RIDGE
    );
    println!(
        "{:<26} {:>5} {:>9} {:>9} {:>10} {:>14}",
        "op", "count", "GFLOP", "DRAM MB", "FLOP/byte", "roofline"
    );
    for (op, (cnt, fl, by)) in &rows {
        let intensity = if *by == 0 {
            0.0
        } else {
            *fl as f64 / *by as f64
        };
        // Synthesize a representative cost for the class label.
        let repr = OpCost {
            id: 0,
            op: (*op).clone(),
            m: 0,
            k: 0,
            n: 0,
            flops: *fl,
            bytes: *by,
            internal_bytes: 0,
            fused: false,
        };
        println!(
            "{:<26} {:>5} {:>9.2} {:>9.1} {:>10.2} {:>14}",
            op,
            cnt,
            *fl as f64 / 1e9,
            *by as f64 / 1e6,
            intensity,
            roofline_class(&repr, DEFAULT_RIDGE),
        );
    }
    println!(
        "{:<26} {:>5} {:>9.2} {:>9.1} {:>10.2}",
        "TOTAL",
        all_costs.len(),
        tot_flops as f64 / 1e9,
        tot_bytes as f64 / 1e6,
        if tot_bytes == 0 {
            0.0
        } else {
            tot_flops as f64 / tot_bytes as f64
        },
    );

    // ── Forward vs backward-added ───────────────────────────────────────────
    let (ff, fb): (u64, u64) = fwd_costs
        .iter()
        .fold((0, 0), |(f, b), c| (f + c.flops, b + c.bytes));
    println!("\n── FORWARD vs BACKWARD (Δ = backward-added) ──");
    println!(
        "forward   : {:>4} nodes  {:>7.2} GFLOP  {:>7.1} MB",
        fwd_costs.len(),
        ff as f64 / 1e9,
        fb as f64 / 1e6
    );
    println!(
        "backward Δ: {:>4} nodes  {:>7.2} GFLOP  {:>7.1} MB",
        all_costs.len() - fwd_costs.len(),
        (tot_flops - ff) as f64 / 1e9,
        (tot_bytes - fb) as f64 / 1e6
    );

    // ── Hot GEMM shapes (M,K,N → count, total FLOP) ─────────────────────────
    println!("\n── HOT GEMM SHAPES (fwd+bwd) ──");
    println!(
        "{:>7} {:>7} {:>7} {:>6} {:>9}",
        "M", "K", "N", "count", "GFLOP"
    );
    for ((gm, gk, gn), (cnt, fl)) in gemm_shape_histogram(&all_costs).iter().take(10) {
        println!("{gm:>7} {gk:>7} {gn:>7} {cnt:>6} {:>9.2}", *fl as f64 / 1e9);
    }

    // ── Data flow: repeated per-layer motif (shapes) ────────────────────────
    println!("\n── DATA FLOW (forward, first transformer layer's op→shape chain) ──");
    let g = m.graph();
    let mut shown = 0;
    let mut started = false;
    for node in g.nodes() {
        let name = format!("{:?}", node.op);
        let name = name.split(['{', '(', ' ']).next().unwrap_or("").to_string();
        if matches!(name.as_str(), "Input" | "Param" | "Constant") {
            continue;
        }
        // Start at the first LayerNorm (top of a block), stop after one block.
        if !started && name.contains("LayerNorm") {
            started = true;
        }
        if started {
            let ins: Vec<String> = node.inputs.iter().map(|i| format!("{}", i.0)).collect();
            println!(
                "  #{:<4} {:<22} {:?}  ← [{}]",
                node.id.0,
                name,
                node.shape.dims(),
                ins.join(",")
            );
            shown += 1;
            if name.contains("LayerNorm") && shown > 1 {
                break; // reached the next block's first norm
            }
        }
    }

    // ── Timing: forward-only vs fwd+bwd (warm) ──────────────────────────────
    if is_available(Device::Metal) {
        let dev = Device::Metal;
        let mut rng: u64 = 0x1234_5678;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng % 256) as f32
        };
        let tok: Vec<f32> = (0..bt).map(|_| next()).collect();
        let tgt: Vec<f32> = (0..bt).map(|_| next()).collect();
        let feed: &[(&str, &[f32])] = &[("tok_ids", &tok), ("tgt_ids", &tgt)];
        for _ in 0..3 {
            let _ = m.run_on(dev, feed);
            let _ = vg.run_on(dev, feed);
        }
        let n = 10;
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            let _ = m.run_on(dev, feed);
        }
        let fwd = t0.elapsed().as_secs_f64() / n as f64 * 1e3;
        let t1 = std::time::Instant::now();
        for _ in 0..n {
            let _ = vg.run_on(dev, feed);
        }
        let fwdbwd = t1.elapsed().as_secs_f64() / n as f64 * 1e3;
        println!(
            "\n── MEASURED TIMING (Metal, warm) ──\nforward-only {fwd:.1}ms   fwd+bwd {fwdbwd:.1}ms   → backward ≈ {:.1}ms ({:.1}× fwd)",
            fwdbwd - fwd,
            (fwdbwd - fwd) / fwd
        );
    } else {
        println!("\n(Metal unavailable — skipping measured timing)");
    }
}
