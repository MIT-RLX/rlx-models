// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Microbenchmark for `Op::DequantGroupedMatMulMlx{MlxMxfp4}` at Ling-3.0-tiny's
//! real MoE shapes, so the kernel can be iterated on in seconds instead of
//! through a 60-second whole-model prefill.
//!
//! Reports the two numbers that tell you what is wrong: achieved weight-read
//! bandwidth (the kernel is memory-bound — every output element streams a full
//! `k/2`-byte packed weight row) and GFLOP/s.
//!
//! ```text
//! cargo run --release -p rlx-models-core --features cuda \
//!   --example mxfp4_grouped_bench -- --device cuda
//! ```

use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::op::Op;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_models_core::flow_util::{built_from_hir, compile_built};
use rlx_models_core::mxfp4_pack::{GROUP_SIZE, quantize_rows};
use rlx_runtime::Device;
use std::collections::HashMap;
use std::time::Instant;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((s >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0) * 0.05
        })
        .collect()
}

/// One shape to time. `experts` is `None` for a dense `Op::DequantMatMul` and
/// `Some(E)` for a grouped `Op::DequantGroupedMatMulMlx` over an `E`-expert bank.
struct Case {
    name: &'static str,
    experts: Option<usize>,
    m: usize,
    k: usize,
    n: usize,
}

/// Dense `Op::DequantMatMul{MlxMxfp4}` at one projection's shape.
fn run_dense(dev: Device, c: &Case, reps: usize) {
    let gs = GROUP_SIZE;
    let ng = c.k / gs;
    let w = fill(c.n * c.k, 7);
    let q = quantize_rows(&w, c.n, c.k, gs);
    drop(w);
    let x = fill(c.m * c.k, 11);

    let scheme = QuantScheme::MlxMxfp4 {
        group_size: gs as u32,
    };
    let mut hir = HirModule::new("bench_dense");
    let mut g = HirMut::new(&mut hir);
    let x_id = g.input("x", Shape::new(&[c.m, c.k], DType::F32));
    let c_id = g.param("codes", Shape::new(&[q.codes.len()], DType::U8));
    let s_id = g.param("scales", Shape::new(&[c.n, ng], DType::U8));
    let b_id = g.param("biases", Shape::new(&[c.n, ng], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_id, c_id, s_id, b_id],
        Shape::new(&[c.m, c.n], DType::F32),
    );
    g.set_outputs(vec![y]);
    let built = built_from_hir(hir, HashMap::new()).expect("built");
    let mut compiled = compile_built(built, dev).expect("compile");
    compiled.set_param_typed("codes", &q.codes, DType::U8);
    compiled.set_param_typed("scales", q.scales_e8m0(), DType::U8);
    compiled.set_param_typed("biases", &q.zero_biases_u8(), DType::U8);

    let inputs: Vec<(&str, &[f32])> = vec![("x", x.as_slice())];
    let out = compiled.run(&inputs);
    let checksum: f32 = out[0].iter().take(8).sum();
    let t = Instant::now();
    for _ in 0..reps {
        let _ = compiled.run(&inputs);
    }
    let secs = t.elapsed().as_secs_f64() / reps as f64;
    // A dense weight is read ONCE regardless of m (it is shared by all rows).
    let wbytes = (c.n * c.k / 2) as f64;
    let flops = 2.0 * (c.m * c.n * c.k) as f64;
    println!(
        "{:<10} DENSE      m={:<3} k={:<5} n={:<6} {:>8.2} ms  {:>7.1} GB/s weights                        \
{:>7.1} GFLOP/s  [chk {checksum:+.4}]",
        c.name,
        c.m,
        c.k,
        c.n,
        secs * 1e3,
        wbytes / secs / 1e9,
        flops / secs / 1e9,
    );
}

fn run_case(dev: Device, c: &Case, reps: usize) {
    let e = c.experts.expect("run_case is the grouped path");
    let gs = GROUP_SIZE;
    let ng = c.k / gs;
    // Only the routed experts are actually read, but the bank is full size so the
    // working set (and therefore the cache behaviour) matches the real model.
    let w = fill(e * c.n * c.k, 7);
    let q = quantize_rows(&w, e * c.n, c.k, gs);
    drop(w);
    let x = fill(c.m * c.k, 11);
    // Distinct experts per row — the worst case for weight reuse, and what a real
    // router produces at prefill.
    let eidx: Vec<f32> = (0..c.m).map(|i| ((i * 7) % e) as f32).collect();

    let scheme = QuantScheme::MlxMxfp4 {
        group_size: gs as u32,
    };
    let mut hir = HirModule::new("bench");
    let mut g = HirMut::new(&mut hir);
    let x_id = g.input("x", Shape::new(&[c.m, c.k], DType::F32));
    let idx_id = g.input("eidx", Shape::new(&[c.m], DType::F32));
    let c_id = g.param("codes", Shape::new(&[q.codes.len()], DType::U8));
    let s_id = g.param("scales", Shape::new(&[e, c.n, ng], DType::BF16));
    let b_id = g.param("biases", Shape::new(&[e, c.n, ng], DType::BF16));
    let y = g.add_node(
        Op::DequantGroupedMatMulMlx { scheme },
        vec![x_id, c_id, s_id, b_id, idx_id],
        Shape::new(&[c.m, c.n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let built = built_from_hir(hir, HashMap::new()).expect("built");
    let mut compiled = compile_built(built, dev).expect("compile");
    compiled.set_param_typed("codes", &q.codes, DType::U8);
    compiled.set_param_typed("scales", &q.scales_bf16(), DType::BF16);
    compiled.set_param_typed("biases", &q.zero_biases_bf16(), DType::BF16);

    let inputs: Vec<(&str, &[f32])> = vec![("x", x.as_slice()), ("eidx", eidx.as_slice())];
    let out = compiled.run(&inputs); // warm-up / JIT
    let checksum: f32 = out[0].iter().take(8).sum();

    let t = Instant::now();
    for _ in 0..reps {
        let _ = compiled.run(&inputs);
    }
    let secs = t.elapsed().as_secs_f64() / reps as f64;

    // Each output element streams one packed weight row (k/2 bytes). That is the
    // kernel's inherent traffic with no cross-row reuse; achieved bandwidth
    // against it is the honest "how close to memory-bound" number.
    let wbytes = (c.m * c.n * c.k / 2) as f64;
    let flops = 2.0 * (c.m * c.n * c.k) as f64;
    // With perfect reuse a warp/block would read each fired expert's rows once.
    let uniq = c.m.min(e);
    let ideal = (uniq * c.n * c.k / 2) as f64;
    println!(
        "{:<10} E={:<4} m={:<3} k={:<5} n={:<5}  {:>8.2} ms  {:>7.1} GB/s streamed \
         ({:>6.2} GB/s if fully reused)  {:>7.1} GFLOP/s  [chk {checksum:+.4}]",
        c.name,
        e,
        c.m,
        c.k,
        c.n,
        secs * 1e3,
        wbytes / secs / 1e9,
        ideal / secs / 1e9,
        flops / secs / 1e9,
    );
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut dev = Device::Cpu;
    let mut reps = 5usize;
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--device" => {
                dev = match argv[i + 1].as_str() {
                    "cuda" => Device::Cuda,
                    "metal" => Device::Metal,
                    "mlx" => Device::Mlx,
                    "rocm" => Device::Rocm,
                    "gpu" | "wgpu" => Device::Gpu,
                    "cpu" => Device::Cpu,
                    o => panic!("unknown device {o}"),
                };
                i += 2;
            }
            "--reps" => {
                reps = argv[i + 1].parse().expect("reps");
                i += 2;
            }
            other => panic!("unknown flag {other}"),
        }
    }
    println!("MXFP4 grouped matmul on {dev:?} ({reps} reps)\n");
    // Ling-3.0-tiny: hidden 1536, moe_inter 512, 128 experts. gate_up is
    // [E, 2*inter, hidden]; down is [E, hidden, inter].
    let cases = [
        Case {
            name: "gate_up",
            experts: Some(128),
            m: 1,
            k: 1536,
            n: 1024,
        },
        Case {
            name: "gate_up",
            experts: Some(128),
            m: 8,
            k: 1536,
            n: 1024,
        },
        Case {
            name: "gate_up",
            experts: Some(128),
            m: 64,
            k: 1536,
            n: 1024,
        },
        Case {
            name: "down",
            experts: Some(128),
            m: 1,
            k: 512,
            n: 1536,
        },
        Case {
            name: "down",
            experts: Some(128),
            m: 64,
            k: 512,
            n: 1536,
        },
    ];
    for c in &cases {
        run_case(dev, c, reps);
    }
    println!();
    // The dense projections Ling quantizes: per layer q/kv/o plus the shared
    // expert, and one lm_head at the end.
    let dense = [
        Case {
            name: "attn_proj",
            experts: None,
            m: 64,
            k: 1536,
            n: 1536,
        },
        Case {
            name: "shared_up",
            experts: None,
            m: 64,
            k: 1536,
            n: 512,
        },
        Case {
            name: "lm_head",
            experts: None,
            m: 64,
            k: 1536,
            n: 157184,
        },
        Case {
            name: "lm_head",
            experts: None,
            m: 1,
            k: 1536,
            n: 157184,
        },
    ];
    for c in &dense {
        run_dense(dev, c, reps);
    }
}
