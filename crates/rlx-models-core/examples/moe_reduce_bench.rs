// RLX — versatile ML compiler + runtime. GPLv3.
//! **MoE reduce-strategy microbenchmark** (portable, CPU, no checkpoint/GPU deps —
//! runs identically on mac/msi/amd over ssh). Compares how you fold the per-expert
//! contributions of a MoE layer for a batch of `B` tokens (256 experts, top-k):
//!
//!   S1 token-major   — per token, per chosen expert: GEMV, accumulate. The weight
//!                      of a shared expert is re-read once per token (memory-heavy).
//!   S2 expert-major  — per active expert, loop its tokens as GEMVs, accumulate.
//!                      Weight read once/expert, but still many small GEMVs.
//!   S3 grouped       — group tokens by expert, ONE GEMM over each expert's token
//!     scatter-reduce   group, scatter-add to the output. Weight read once/expert,
//!                      one large BLAS-friendly op per expert (the batched reduce).
//!
//! Reports ms/token + GFLOP/s per strategy — the winner shifts with core count and
//! memory bandwidth, so run it on each box. Synthetic f32 weights (representative
//! shapes); the compute PATTERN is what differs, independent of quant/IO.
//!
//! `--gemm blas` swaps the naive triple-loop matmul for the production
//! `rlx_cpu::blas::sgemm` (real BLAS — Accelerate on macOS, OpenBLAS/MKL on x86) to
//! show the **real hardware ceiling**: S3's big per-expert GEMM hits the BLAS
//! microkernel while S1's per-token GEMV (m=1) can't, so S3's lead WIDENS.
//!
//! `--device metal` (build `--features metal`) / `cpu` / `mlx` reports the **raw
//! per-device GEMM ceiling** — it runs the S3-dominant MoE GEMM shapes
//! (`[batch,h]@[h,inter]` gate/up + `[batch,inter]@[inter,h]` down) through the real
//! rlx compiled graph on that device, so you see the GPU (Metal) vs CPU-BLAS GEMM
//! throughput S3 can actually reach.
//!
//!   moe_reduce_bench --experts 64 --hidden 2048 --inter 1024 --topk 8 --batch 32
//!   moe_reduce_bench --gemm blas --batch 128            # real BLAS ceiling
//!   moe_reduce_bench --device metal --batch 128         # GPU GEMM ceiling (--features metal)

use rayon::prelude::*;
use std::time::Instant;

fn flag(a: &[String], k: &str) -> Option<String> {
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1))
        .cloned()
}

/// Raw GEMM throughput of `y[m,n] = x[m,k] @ w[k,n]` on `dev`, via the real rlx
/// compiled graph (CPU→BLAS, Metal/MLX→GPU). Returns GFLOP/s. This is the ceiling
/// the batched S3 reduce can hit per device.
fn device_gemm_gflops(dev: rlx_runtime::Device, m: usize, k: usize, n: usize, iters: usize) -> f64 {
    use rlx_ir::infer::GraphExt;
    use rlx_ir::{DType, Graph, Shape};
    use rlx_runtime::Session;
    let f = DType::F32;
    let mut g = Graph::new("gemm_ceiling");
    let x = g.input("x", Shape::new(&[m, k], f));
    let w = g.param("w", Shape::new(&[k, n], f));
    let y = g.mm(x, w);
    g.set_outputs(vec![y]);
    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill(dev);
    let mut c = Session::new(dev).compile_with(g, &opts);
    let wv: Vec<f32> = (0..k * n).map(|i| rnd(i) * 0.03).collect();
    c.set_param("w", &wv);
    let xv: Vec<f32> = (0..m * k).map(|i| rnd(i + 7) * 0.1).collect();
    let _ = c.run(&[("x", xv.as_slice())]); // warm (compile caches, GPU upload)
    let t = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(c.run(&[("x", xv.as_slice())]));
    }
    let dt = t.elapsed().as_secs_f64() / iters as f64;
    2.0 * (m * k * n) as f64 / dt / 1e9
}

fn now() -> Instant {
    Instant::now()
}

// Deterministic pseudo-random f32 in [-1,1].
fn rnd(seed: usize) -> f32 {
    ((seed.wrapping_mul(2654435761) % 2003) as f32) / 1001.0 - 1.0
}

fn silu(z: f32) -> f32 {
    z / (1.0 + (-z).exp())
}

/// y[m,n] = x[m,k] @ w[k,n] (row-major). `blas` → production `rlx_cpu::blas::sgemm`
/// (real BLAS: Accelerate/OpenBLAS/MKL); else the naive rayon-parallel triple loop.
fn matmul(x: &[f32], w: &[f32], m: usize, k: usize, n: usize, y: &mut [f32], blas: bool) {
    if blas {
        // C = A[m,k] @ B[k,n] (no accumulate) — exactly y = x@w.
        rlx_cpu::blas::sgemm(x, w, y, m, k, n);
        return;
    }
    y.par_chunks_mut(n).enumerate().for_each(|(r, yr)| {
        let xr = &x[r * k..r * k + k];
        for (c, yc) in yr.iter_mut().enumerate() {
            let mut s = 0f32;
            for i in 0..k {
                s += xr[i] * w[i * n + c];
            }
            *yc = s;
        }
    });
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n_exp: usize = flag(&a, "--experts")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let h: usize = flag(&a, "--hidden")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);
    let inter: usize = flag(&a, "--inter")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let top_k: usize = flag(&a, "--topk").and_then(|s| s.parse().ok()).unwrap_or(8);
    let batch: usize = flag(&a, "--batch")
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let iters: usize = flag(&a, "--iters")
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let mode = flag(&a, "--mode").unwrap_or_else(|| "all".into());
    let blas = flag(&a, "--gemm").as_deref() == Some("blas");

    // Per-expert weights: Wg,Wu [h,inter] (x@W), Wd [inter,h].
    let gate: Vec<Vec<f32>> = (0..n_exp)
        .map(|e| (0..h * inter).map(|i| rnd(e * 7919 + i) * 0.03).collect())
        .collect();
    let up: Vec<Vec<f32>> = (0..n_exp)
        .map(|e| {
            (0..h * inter)
                .map(|i| rnd(e * 5147 + i + 11) * 0.03)
                .collect()
        })
        .collect();
    let down: Vec<Vec<f32>> = (0..n_exp)
        .map(|e| {
            (0..inter * h)
                .map(|i| rnd(e * 3301 + i + 23) * 0.03)
                .collect()
        })
        .collect();
    let x: Vec<f32> = (0..batch * h).map(|i| rnd(i + 1) * 0.1).collect();
    // Routing: each token → top_k distinct experts + weights.
    let route: Vec<Vec<(usize, f32)>> = (0..batch)
        .map(|b| {
            let mut es: Vec<usize> = (0..top_k).map(|j| (b * 131 + j * 977) % n_exp).collect();
            es.sort_unstable();
            es.dedup();
            es.into_iter()
                .map(|e| (e, 0.1 + 0.01 * (e % 7) as f32))
                .collect()
        })
        .collect();

    // FLOPs/token = top_k experts × (2 up-projs [h·inter] + 1 down [inter·h]) × 2.
    let flop_per_tok = top_k as f64 * 3.0 * (h * inter) as f64 * 2.0;
    let total_flop = flop_per_tok * batch as f64;

    // ── --device: raw per-device GEMM ceiling (the throughput S3 can reach) ──
    if let Some(dstr) = flag(&a, "--device") {
        use rlx_runtime::Device;
        let dev = match dstr.as_str() {
            "metal" => Device::Metal,
            "mlx" => Device::Mlx,
            "cuda" => Device::Cuda,
            "vulkan" => Device::Vulkan,
            "gpu" => Device::Gpu,
            _ => Device::Cpu,
        };
        let it = iters * 4;
        // MoE GEMM shapes at this batch: gate/up [batch,h]@[h,inter], down [batch,inter]@[inter,h].
        let up = device_gemm_gflops(dev, batch, h, inter, it);
        let down = device_gemm_gflops(dev, batch, inter, h, it);
        // FLOPs/token across an expert's 3 GEMMs; ms/token at this ceiling (÷ top_k experts served).
        let flop_tok = top_k as f64 * 3.0 * (h * inter) as f64 * 2.0;
        let ceil_gflops = (up + up + down) / 3.0; // avg of gate+up+down
        eprintln!(
            "[moe-reduce] device={dstr} batch={batch} h={h} inter={inter} \
             → GEMM ceiling: gate/up {up:.0} GFLOP/s, down {down:.0} GFLOP/s, \
             avg {ceil_gflops:.0} GFLOP/s ⇒ ~{:.3} ms/token (top_k={top_k})",
            flop_tok / (ceil_gflops * 1e9) * 1e3
        );
        return;
    }

    let gb = (n_exp * 3 * h * inter * 4) as f64 / 1e9;
    eprintln!(
        "[moe-reduce] experts={n_exp} h={h} inter={inter} top_k={top_k} batch={batch} \
         gemm={} (weights {gb:.1} GB f32, {} cores)",
        if blas { "blas" } else { "naive" },
        rayon::current_num_threads()
    );

    // ── S1: token-major (GEMV per token per chosen expert) ──
    let s1 = |x: &[f32]| -> Vec<f32> {
        let mut out = vec![0f32; batch * h];
        for b in 0..batch {
            let xb = &x[b * h..b * h + h];
            for &(e, w) in &route[b] {
                let mut g = vec![0f32; inter];
                let mut u = vec![0f32; inter];
                matmul(xb, &gate[e], 1, h, inter, &mut g, blas);
                matmul(xb, &up[e], 1, h, inter, &mut u, blas);
                let glu: Vec<f32> = (0..inter).map(|i| silu(g[i]) * u[i]).collect();
                let mut d = vec![0f32; h];
                matmul(&glu, &down[e], 1, inter, h, &mut d, blas);
                for i in 0..h {
                    out[b * h + i] += w * d[i];
                }
            }
        }
        out
    };

    // ── S3: grouped scatter-reduce (one GEMM per expert over its token group) ──
    let s3 = |x: &[f32]| -> Vec<f32> {
        // Build per-expert token groups.
        let mut groups: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_exp]; // expert → [(token, weight)]
        for (b, r) in route.iter().enumerate() {
            for &(e, w) in r {
                groups[e].push((b, w));
            }
        }
        let mut out = vec![0f32; batch * h];
        for e in 0..n_exp {
            let toks = &groups[e];
            if toks.is_empty() {
                continue;
            }
            let m = toks.len();
            // Gather this expert's tokens into a contiguous [m, h] tile.
            let mut xg = vec![0f32; m * h];
            for (r, &(b, _)) in toks.iter().enumerate() {
                xg[r * h..r * h + h].copy_from_slice(&x[b * h..b * h + h]);
            }
            let mut g = vec![0f32; m * inter];
            let mut u = vec![0f32; m * inter];
            matmul(&xg, &gate[e], m, h, inter, &mut g, blas); // ONE GEMM for all m tokens
            matmul(&xg, &up[e], m, h, inter, &mut u, blas);
            let glu: Vec<f32> = (0..m * inter).map(|i| silu(g[i]) * u[i]).collect();
            let mut d = vec![0f32; m * h];
            matmul(&glu, &down[e], m, inter, h, &mut d, blas);
            for (r, &(b, w)) in toks.iter().enumerate() {
                for i in 0..h {
                    out[b * h + i] += w * d[r * h + i]; // scatter-add (reduce) to the token
                }
            }
        }
        out
    };

    let bench = |name: &str, f: &dyn Fn(&[f32]) -> Vec<f32>| -> Vec<f32> {
        let warm = f(&x);
        let t = now();
        for _ in 0..iters {
            std::hint::black_box(f(&x));
        }
        let dt = t.elapsed().as_secs_f64() / iters as f64;
        eprintln!(
            "  [{name}] {:.2} ms/pass, {:.3} ms/token, {:.1} GFLOP/s",
            dt * 1e3,
            dt * 1e3 / batch as f64,
            total_flop / dt / 1e9
        );
        warm
    };

    let r1 = if mode == "s1" || mode == "all" {
        Some(bench("S1 token-major   ", &s1))
    } else {
        None
    };
    let r3 = if mode == "s3" || mode == "all" {
        Some(bench("S3 grouped-reduce", &s3))
    } else {
        None
    };
    // Parity: S1 and S3 must agree (same math, different reduction order).
    if let (Some(a), Some(b)) = (&r1, &r3) {
        let err = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "  parity S1 vs S3: max_err {err:e} ({})",
            if err < 1e-2 { "OK" } else { "MISMATCH" }
        );
    }
}
