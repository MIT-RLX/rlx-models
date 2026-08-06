// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Parity test for the FlashKDA chunked-parallel forward
//! ([`rlx_kimi_k3::kda_chunk::build_kda_chunked_scan`]) against the ground-truth
//! **sequential** per-channel gated-delta-net recurrence — both the native
//! `Op::GatedDeltaNet` (`gated_delta_net_pc`) and a plain-Rust reference of the
//! same recurrence. The chunked form is an algebraic identity of the sequential
//! scan, so the two must agree to f32 accumulation noise. Runs on CPU (or the
//! backend named by `RLX_TEST_DEVICE`).

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_kimi_k3::kda::{build_kda_layer, KdaDims, KdaWeights};
use rlx_kimi_k3::kda_chunk::{build_kda_chunked_scan, ChunkDims};
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;

fn dev() -> Device {
    match std::env::var("RLX_TEST_DEVICE").ok().as_deref() {
        Some("metal") | Some("mtl") => Device::Metal,
        Some("mlx") => Device::Mlx,
        Some("gpu") | Some("wgpu") => Device::Gpu,
        Some("cuda") => Device::Cuda,
        Some("vulkan") | Some("vk") => Device::Vulkan,
        _ => Device::Cpu,
    }
}

/// Small deterministic pseudo-random fill in a bounded range.
fn fill(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 2.0 * amp
        })
        .collect()
}

/// Reference per-channel gated delta-net (no carry; state reset per batch) —
/// the exact recurrence `Op::GatedDeltaNet { gate_per_channel: true }` runs.
#[allow(clippy::too_many_arguments)]
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    b: usize,
    s: usize,
    h: usize,
    n: usize,
) -> Vec<f32> {
    let scale = 1.0f32 / (n as f32).sqrt();
    let mut out = vec![0f32; b * s * h * n];
    let hn = h * n;
    for bi in 0..b {
        for hi in 0..h {
            let mut smat = vec![0f32; n * n]; // S[i,j] at i*n+j
            for ti in 0..s {
                let base = bi * s * hn + ti * hn + hi * n;
                let betabase = bi * s * h + ti * h + hi;
                let (qr, kr, vr) = (&q[base..base + n], &k[base..base + n], &v[base..base + n]);
                let gr = &g[base..base + n];
                let bt = beta[betabase];
                for i in 0..n {
                    let a = gr[i].exp();
                    for j in 0..n {
                        smat[i * n + j] *= a;
                    }
                }
                let mut sk = vec![0f32; n];
                for i in 0..n {
                    for j in 0..n {
                        sk[j] += smat[i * n + j] * kr[i];
                    }
                }
                for j in 0..n {
                    sk[j] = (vr[j] - sk[j]) * bt;
                }
                for i in 0..n {
                    for j in 0..n {
                        smat[i * n + j] += kr[i] * sk[j];
                    }
                }
                for j in 0..n {
                    let mut acc = 0f32;
                    for i in 0..n {
                        acc += smat[i * n + j] * qr[i];
                    }
                    out[base + j] = acc * scale;
                }
            }
        }
    }
    out
}

/// Run the chunked builder on a `[b,s,h,n]` case and return the flattened output.
/// `use_scan` picks the `Op::Scan` K2 vs the unrolled loop.
#[allow(clippy::too_many_arguments)]
fn run_chunked(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    b: usize,
    s: usize,
    h: usize,
    n: usize,
    chunk: usize,
    use_scan: bool,
) -> Vec<f32> {
    let mut hir = HirModule::new("kda_chunk_test");
    let mut gb = HirMut::new(&mut hir);
    let f = DType::F32;
    let q_in = gb.input("q", Shape::new(&[b, s, h, n], f));
    let k_in = gb.input("k", Shape::new(&[b, s, h, n], f));
    let v_in = gb.input("v", Shape::new(&[b, s, h, n], f));
    let g_in = gb.input("g", Shape::new(&[b, s, h, n], f));
    let beta_in = gb.input("beta", Shape::new(&[b, s, h], f));
    let (out, _final) = build_kda_chunked_scan(
        &mut gb,
        q_in,
        k_in,
        v_in,
        g_in,
        beta_in,
        ChunkDims {
            batch: b,
            seq: s,
            heads: h,
            head_dim: n,
            chunk,
            use_scan,
        },
        None,
    );
    gb.set_outputs(vec![out]);

    let built = built_from_hir(hir, HashMap::new()).expect("build chunked graph");
    let mut compiled = compile_built(built, dev()).expect("compile chunked graph");
    compiled
        .run(&[
            ("q", q),
            ("k", k),
            ("v", v),
            ("g", g),
            ("beta", beta),
        ])
        .into_iter()
        .next()
        .expect("chunked output")
}

/// Run the **native** sequential `Op::GatedDeltaNet` (per-channel) on the same
/// case — the exact op the KDA layer uses today.
#[allow(clippy::too_many_arguments)]
fn run_native_pc(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    b: usize,
    s: usize,
    h: usize,
    n: usize,
) -> Vec<f32> {
    let mut hir = HirModule::new("kda_native_pc");
    let mut gb = HirMut::new(&mut hir);
    let f = DType::F32;
    let q_in = gb.input("q", Shape::new(&[b, s, h, n], f));
    let k_in = gb.input("k", Shape::new(&[b, s, h, n], f));
    let v_in = gb.input("v", Shape::new(&[b, s, h, n], f));
    let g_in = gb.input("g", Shape::new(&[b, s, h, n], f));
    let beta_in = gb.input("beta", Shape::new(&[b, s, h], f));
    let out = gb.gated_delta_net_pc(q_in, k_in, v_in, g_in, beta_in, n, Shape::new(&[b, s, h, n], f));
    gb.set_outputs(vec![out]);
    let built = built_from_hir(hir, HashMap::new()).expect("build native pc graph");
    let mut compiled = compile_built(built, dev()).expect("compile native pc graph");
    compiled
        .run(&[("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)])
        .into_iter()
        .next()
        .expect("native pc output")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// L2-normalize each length-`n` row in place (as the KDA layer feeds q/k). Keeps
/// the delta-rule state bounded — un-normalized keys overflow f32 over long T.
fn l2norm_rows(x: &mut [f32], rows: usize, n: usize) {
    for r in 0..rows {
        let sl = &mut x[r * n..r * n + n];
        let nrm = (sl.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
        sl.iter_mut().for_each(|v| *v /= nrm);
    }
}

/// Chunked forward matches the sequential reference across several
/// (seq, heads, head_dim, chunk) shapes — including a seq that is NOT a multiple
/// of the chunk size (exercises the zero-padding path).
#[test]
fn chunked_matches_sequential_reference() {
    // (b, s, h, n, chunk)
    let cases = [
        (1usize, 16usize, 2usize, 8usize, 4usize),
        (1, 32, 3, 8, 8),
        (2, 20, 2, 8, 4),   // s=20 not a multiple of chunk=4 -> exact chunking
        (1, 33, 2, 16, 16), // s=33 not a multiple of chunk=16 -> pad path
        (2, 12, 4, 4, 4),
        (1, 48, 2, 128, 16), // real FlashKDA config: head_dim=128, CHUNK=16
    ];
    for (ci, &(b, s, h, n, chunk)) in cases.iter().enumerate() {
        let bshn = b * s * h * n;
        // q,k are L2-normed per (b,s,h) row (as the KDA layer feeds them).
        let mut q = fill(bshn, 10 + ci as u64, 1.0);
        let mut k = fill(bshn, 20 + ci as u64, 1.0);
        for row in 0..(b * s * h) {
            for src in [&mut q, &mut k] {
                let sl = &mut src[row * n..row * n + n];
                let nrm = (sl.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
                sl.iter_mut().for_each(|x| *x /= nrm);
            }
        }
        let v = fill(bshn, 30 + ci as u64, 1.0);
        // g_log ≤ 0 (the real KDA gate is negative); keep it modest so exp(±cumsum)
        // stays well within f32 range for this chunk size.
        let g: Vec<f32> = fill(bshn, 40 + ci as u64, 0.25).iter().map(|x| -(x.abs())).collect();
        let beta: Vec<f32> = fill(b * s * h, 50 + ci as u64, 4.0)
            .iter()
            .map(|x| 1.0 / (1.0 + (-x).exp())) // sigmoid, as the layer applies
            .collect();

        let want = reference(&q, &k, &v, &g, &beta, b, s, h, n);
        // Also cross-check against the native sequential op the KDA layer runs.
        let native = run_native_pc(&q, &k, &v, &g, &beta, b, s, h, n);
        assert_eq!(native.len(), want.len());

        let ref_amp = want.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
        let tol = 2e-3 * ref_amp.max(1.0);

        // Both K2 implementations (unrolled loop + Op::Scan) vs reference/native.
        let unroll = run_chunked(&q, &k, &v, &g, &beta, b, s, h, n, chunk, false);
        let scan = run_chunked(&q, &k, &v, &g, &beta, b, s, h, n, chunk, true);
        assert_eq!(unroll.len(), want.len());
        assert_eq!(scan.len(), want.len());

        for (label, got) in [("unroll", &unroll), ("scan", &scan)] {
            let worst = max_abs_diff(got, &want);
            let worst_native = max_abs_diff(got, &native);
            assert!(
                worst < tol,
                "case {ci} [b{b} s{s} h{h} n{n} c{chunk}] {label}-vs-ref diff {worst} (ref_amp {ref_amp})"
            );
            assert!(
                worst_native < tol,
                "case {ci} [b{b} s{s} h{h} n{n} c{chunk}] {label}-vs-native diff {worst_native}"
            );
        }
        // The two K2 paths must be bit-identical (same math, different schedule).
        let scan_vs_unroll = max_abs_diff(&scan, &unroll);
        assert!(
            scan_vs_unroll < tol,
            "case {ci} [b{b} s{s} h{h} n{n} c{chunk}] scan-vs-unroll diff {scan_vs_unroll}"
        );
        println!(
            "case {ci} [b{b} s{s} h{h} n{n} c{chunk}]: unroll-vs-ref {:.2e}, scan-vs-ref {:.2e}, scan-vs-unroll {scan_vs_unroll:.2e} (tol {tol:.2e})",
            max_abs_diff(&unroll, &want),
            max_abs_diff(&scan, &want)
        );
    }
}

/// Build + run one full KDA layer (fused input projection, causal conv, gate
/// activation, recurrence, gated-RMSNorm, o_proj) on CPU, returning the output.
fn build_and_run_layer(d: KdaDims, w: &KdaWeights, hin: &[f32]) -> Vec<f32> {
    let mut hir = HirModule::new("kda_layer");
    let mut g = HirMut::new(&mut hir);
    let h_in = g.input("h", Shape::new(&[d.batch, d.seq, d.hidden], DType::F32));
    let mut params = HashMap::new();
    let out = build_kda_layer(&mut g, &mut params, "kda", h_in, w, d).expect("build kda layer");
    g.set_outputs(vec![out]);
    let built = built_from_hir(hir, params).expect("build kda model");
    let mut compiled = compile_built(built, dev()).expect("compile kda layer");
    compiled
        .run(&[("h", hin)])
        .into_iter()
        .next()
        .expect("kda layer output")
}

/// End-to-end: a full KDA layer run with the FlashKDA chunked-parallel path
/// (`RLX_KDA_CHUNK=16`) matches the same layer run on the native sequential
/// `Op::GatedDeltaNet`. Exercises the real wiring — the chunked scan fed by the
/// layer's own L2-normed q/k, per-channel gate, and sigmoid beta — over a
/// sequence (`seq=20`) that spans multiple chunks and needs zero-padding.
#[test]
fn full_kda_layer_chunked_matches_native() {
    let d = KdaDims {
        hidden: 16,
        num_heads: 2,
        head_dim: 8,
        conv_kernel: 4,
        gate_lower_bound: Some(-5.0),
        eps: 1e-5,
        batch: 1,
        seq: 20,
    };
    let (hidden, h, hd, proj, kk) = (d.hidden, d.num_heads, d.head_dim, d.proj(), d.conv_kernel);
    let w = KdaWeights {
        q_proj: fill(hidden * proj, 1, 0.2),
        k_proj: fill(hidden * proj, 2, 0.2),
        v_proj: fill(hidden * proj, 3, 0.2),
        q_conv: fill(proj * kk, 4, 0.2),
        k_conv: fill(proj * kk, 5, 0.2),
        v_conv: fill(proj * kk, 6, 0.2),
        f_a: fill(hidden * hd, 7, 0.2),
        f_b: fill(hd * proj, 8, 0.2),
        dt_bias: fill(proj, 9, 0.2),
        a_log: fill(hd, 10, 0.2),
        b_proj: fill(hidden * h, 11, 0.2),
        g_proj: fill(hidden * proj, 12, 0.2),
        o_norm: vec![1.0; hd],
        o_proj: fill(proj * hidden, 13, 0.2),
    };
    let hin = fill(d.batch * d.seq * hidden, 100, 0.2);

    // Native sequential path (flag unset).
    unsafe { std::env::remove_var("RLX_KDA_CHUNK") };
    let y_native = build_and_run_layer(d, &w, &hin);

    // FlashKDA chunked-parallel path.
    unsafe { std::env::set_var("RLX_KDA_CHUNK", "16") };
    let y_chunk = build_and_run_layer(d, &w, &hin);
    unsafe { std::env::remove_var("RLX_KDA_CHUNK") };

    assert_eq!(y_native.len(), y_chunk.len());
    assert!(y_chunk.iter().all(|v| v.is_finite()), "chunked KDA output must be finite");
    let amp = y_native.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
    let worst = max_abs_diff(&y_native, &y_chunk);
    assert!(
        worst < 2e-3 * amp.max(1.0),
        "full KDA layer chunked-vs-native diff {worst} (amp {amp}) too large"
    );
    println!("full KDA layer: chunked-vs-native {worst:.2e} (amp {amp:.2e})");
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark: Op::Scan K2 vs unrolled K2 as sequence length grows.
// Run with:  cargo test -p rlx-kimi-k3 --test kda_chunk_pc bench_scan_vs_unroll \
//              --release -- --ignored --nocapture
// ─────────────────────────────────────────────────────────────────────────────

use std::time::Instant;

/// Returns (output, hir_node_count, build+compile ms, single-run ms).
#[allow(clippy::too_many_arguments)]
fn build_compile_run(
    q: &[f32], k: &[f32], v: &[f32], g: &[f32], beta: &[f32],
    b: usize, s: usize, h: usize, n: usize, chunk: usize, use_scan: bool,
) -> (Vec<f32>, usize, f64, f64) {
    // Graph build + compile (this is where unroll's O(T) node count bites).
    let t0 = Instant::now();
    let mut hir = HirModule::new("kda_bench");
    let mut gb = HirMut::new(&mut hir);
    let f = DType::F32;
    let q_in = gb.input("q", Shape::new(&[b, s, h, n], f));
    let k_in = gb.input("k", Shape::new(&[b, s, h, n], f));
    let v_in = gb.input("v", Shape::new(&[b, s, h, n], f));
    let g_in = gb.input("g", Shape::new(&[b, s, h, n], f));
    let beta_in = gb.input("beta", Shape::new(&[b, s, h], f));
    let (out, _final) = build_kda_chunked_scan(
        &mut gb, q_in, k_in, v_in, g_in, beta_in,
        ChunkDims { batch: b, seq: s, heads: h, head_dim: n, chunk, use_scan },
        None,
    );
    gb.set_outputs(vec![out]);
    let n_nodes = hir.len(); // HIR node count — deterministic graph-size metric
    let built = built_from_hir(hir, HashMap::new()).expect("build");
    let mut compiled = compile_built(built, dev()).expect("compile");
    let compile_ms = t0.elapsed().as_secs_f64() * 1e3;

    let inputs: [(&str, &[f32]); 5] = [("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)];
    let t = Instant::now();
    let out = compiled.run(&inputs).into_iter().next().unwrap();
    let run_ms = t.elapsed().as_secs_f64() * 1e3;
    (out, n_nodes, compile_ms, run_ms)
}

#[test]
#[ignore = "benchmark; run explicitly with --ignored --nocapture"]
fn bench_scan_vs_unroll() {
    let (b, h, n, chunk) = (1usize, 2usize, 128usize, 16usize);
    println!("\nFlashKDA K2: unrolled loop vs Op::Scan   (b={b} h={h} n={n} C={chunk}, device={:?})", dev());
    println!(
        "{:>6} | {:>6} | {:>12} {:>10} | {:>12} {:>10} | {:>9}",
        "T", "chunks", "nodes(unrl)", "nodes(scn)", "b+cmp unrl", "b+cmp scn", "max|Δ|"
    );
    for &s in &[256usize, 1024, 4096] {
        let bshn = b * s * h * n;
        let q = fill(bshn, 1, 1.0);
        let k = fill(bshn, 2, 1.0);
        let v = fill(bshn, 3, 1.0);
        let g: Vec<f32> = fill(bshn, 4, 0.25).iter().map(|x| -(x.abs())).collect();
        let beta: Vec<f32> = fill(b * s * h, 5, 4.0).iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();

        let (o_u, nn_u, c_u, _r_u) = build_compile_run(&q, &k, &v, &g, &beta, b, s, h, n, chunk, false);
        let (o_s, nn_s, c_s, _r_s) = build_compile_run(&q, &k, &v, &g, &beta, b, s, h, n, chunk, true);
        let d = max_abs_diff(&o_u, &o_s);
        println!(
            "{:>6} | {:>6} | {:>12} {:>10} | {:>10.1}ms {:>8.1}ms | {:>9.2e}",
            s, s / chunk, nn_u, nn_s, c_u, c_s, d
        );
        assert!(d < 1e-3, "scan/unroll diverge at T={s}: {d}");
    }
    println!();
}

// ─────────────────────────────────────────────────────────────────────────────
// Runtime benchmark: native sequential Op::GatedDeltaNet vs chunked unroll vs
// chunked Op::Scan. Run with:
//   cargo test -p rlx-kimi-k3 --test kda_chunk_pc bench_runtime \
//     --release -- --ignored --nocapture
// (RLX_TEST_DEVICE=metal|mlx|cuda to pick a backend.)
// ─────────────────────────────────────────────────────────────────────────────

fn compile_chunked_graph(b: usize, s: usize, h: usize, n: usize, chunk: usize, use_scan: bool) -> CompiledGraph {
    let mut hir = HirModule::new("kda_bench_rt");
    let mut gb = HirMut::new(&mut hir);
    let f = DType::F32;
    let q = gb.input("q", Shape::new(&[b, s, h, n], f));
    let k = gb.input("k", Shape::new(&[b, s, h, n], f));
    let v = gb.input("v", Shape::new(&[b, s, h, n], f));
    let g = gb.input("g", Shape::new(&[b, s, h, n], f));
    let beta = gb.input("beta", Shape::new(&[b, s, h], f));
    let (out, _f) = build_kda_chunked_scan(
        &mut gb, q, k, v, g, beta,
        ChunkDims { batch: b, seq: s, heads: h, head_dim: n, chunk, use_scan },
        None,
    );
    gb.set_outputs(vec![out]);
    let built = built_from_hir(hir, HashMap::new()).expect("build");
    compile_built(built, dev()).expect("compile")
}

fn compile_native_graph(b: usize, s: usize, h: usize, n: usize) -> CompiledGraph {
    let mut hir = HirModule::new("kda_bench_native");
    let mut gb = HirMut::new(&mut hir);
    let f = DType::F32;
    let q = gb.input("q", Shape::new(&[b, s, h, n], f));
    let k = gb.input("k", Shape::new(&[b, s, h, n], f));
    let v = gb.input("v", Shape::new(&[b, s, h, n], f));
    let g = gb.input("g", Shape::new(&[b, s, h, n], f));
    let beta = gb.input("beta", Shape::new(&[b, s, h], f));
    let out = gb.gated_delta_net_pc(q, k, v, g, beta, n, Shape::new(&[b, s, h, n], f));
    gb.set_outputs(vec![out]);
    let built = built_from_hir(hir, HashMap::new()).expect("build");
    compile_built(built, dev()).expect("compile")
}

fn time_min(compiled: &mut CompiledGraph, inputs: &[(&str, &[f32])], reps: usize) -> (f64, Vec<f32>) {
    let mut out = compiled.run(inputs).into_iter().next().unwrap(); // warm
    let _ = compiled.run(inputs);
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        out = compiled.run(inputs).into_iter().next().unwrap();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    (best, out)
}

#[test]
#[ignore = "runtime benchmark; run with --release --ignored --nocapture"]
fn bench_runtime() {
    let (b, h, n, chunk) = (1usize, 16usize, 128usize, 16usize);
    println!("\nKDA forward runtime   (b={b} h={h} n={n} C={chunk}, device={:?})", dev());
    println!("{:>6} | {:>12} {:>12} {:>12} | {:>10} {:>10}", "T", "native(ms)", "unroll(ms)", "scan(ms)", "spd(u/n)", "spd(s/n)");
    for &s in &[512usize, 2048] {
        let bshn = b * s * h * n;
        let mut q = fill(bshn, 1, 1.0);
        let mut k = fill(bshn, 2, 1.0);
        l2norm_rows(&mut q, b * s * h, n);
        l2norm_rows(&mut k, b * s * h, n);
        let v = fill(bshn, 3, 1.0);
        let g: Vec<f32> = fill(bshn, 4, 0.25).iter().map(|x| -(x.abs())).collect();
        let beta: Vec<f32> = fill(b * s * h, 5, 4.0).iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        let inputs: [(&str, &[f32]); 5] = [("q", &q), ("k", &k), ("v", &v), ("g", &g), ("beta", &beta)];

        let (t_nat, o_nat) = time_min(&mut compile_native_graph(b, s, h, n), &inputs, 8);
        let (t_unr, o_unr) = time_min(&mut compile_chunked_graph(b, s, h, n, chunk, false), &inputs, 8);
        let (t_scn, o_scn) = time_min(&mut compile_chunked_graph(b, s, h, n, chunk, true), &inputs, 8);

        // sanity: all three agree
        let d1 = max_abs_diff(&o_nat, &o_unr);
        let d2 = max_abs_diff(&o_unr, &o_scn);
        assert!(d1 < 2e-3 && d2 < 1e-3, "T={s} divergence native/unroll {d1}, unroll/scan {d2}");

        println!(
            "{:>6} | {:>12.3} {:>12.3} {:>12.3} | {:>9.2}x {:>9.2}x",
            s, t_nat, t_unr, t_scn, t_nat / t_unr, t_nat / t_scn
        );
    }
    println!();
}

// ─────────────────────────────────────────────────────────────────────────────
// Profile the NATIVE fused Op::GatedDeltaNet kernel (the SG kernel at n=128) to
// decide whether a chunk-parallel rewrite could win: scaling vs T (serial-depth /
// latency bound?) and vs heads (occupancy saturated?).
//   cargo test -p rlx-kimi-k3 --release --features metal --test kda_chunk_pc \
//     bench_native_scaling -- --ignored --nocapture   (RLX_TEST_DEVICE=metal)
// ─────────────────────────────────────────────────────────────────────────────
#[test]
#[ignore = "profiling; run with --release --ignored --nocapture"]
fn bench_native_scaling() {
    let n = 128usize;
    println!("\nNative Op::GatedDeltaNet scaling  (n={n}, device={:?})", dev());

    println!("-- vs sequence length T  (b=1, h=16) --");
    println!("{:>6} | {:>10} | {:>12}", "T", "time(ms)", "us/token");
    let (b, h) = (1usize, 16usize);
    for &s in &[256usize, 512, 1024, 2048, 4096] {
        let bshn = b * s * h * n;
        let mut q = fill(bshn, 1, 1.0);
        let mut k = fill(bshn, 2, 1.0);
        l2norm_rows(&mut q, b * s * h, n);
        l2norm_rows(&mut k, b * s * h, n);
        let v = fill(bshn, 3, 1.0);
        let g: Vec<f32> = fill(bshn, 4, 0.25).iter().map(|x| -(x.abs())).collect();
        let beta: Vec<f32> = fill(b * s * h, 5, 4.0).iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        let inputs: [(&str, &[f32]); 5] = [("q", &q), ("k", &k), ("v", &v), ("g", &g), ("beta", &beta)];
        let (t, _o) = time_min(&mut compile_native_graph(b, s, h, n), &inputs, 8);
        println!("{:>6} | {:>10.3} | {:>12.4}", s, t, t * 1e3 / s as f64);
    }

    println!("-- vs head count h  (b=1, T=1024) --");
    println!("{:>6} | {:>10} | {:>12}", "h", "time(ms)", "ms/head");
    let (b, s) = (1usize, 1024usize);
    for &h in &[1usize, 2, 4, 8, 16, 32, 64] {
        let bshn = b * s * h * n;
        let mut q = fill(bshn, 1, 1.0);
        let mut k = fill(bshn, 2, 1.0);
        l2norm_rows(&mut q, b * s * h, n);
        l2norm_rows(&mut k, b * s * h, n);
        let v = fill(bshn, 3, 1.0);
        let g: Vec<f32> = fill(bshn, 4, 0.25).iter().map(|x| -(x.abs())).collect();
        let beta: Vec<f32> = fill(b * s * h, 5, 4.0).iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        let inputs: [(&str, &[f32]); 5] = [("q", &q), ("k", &k), ("v", &v), ("g", &g), ("beta", &beta)];
        let (t, _o) = time_min(&mut compile_native_graph(b, s, h, n), &inputs, 8);
        println!("{:>6} | {:>10.3} | {:>12.4}", h, t, t / h as f64);
    }
    println!();
}
