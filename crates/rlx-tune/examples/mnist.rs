// RLX models — fine-tuning.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! **Distributed MNIST trainer** — a real CNN on real data, through the crate's
//! data-parallel machinery ([`Trainer`] / [`DpConfig`] / the `--nnodes`
//! launcher). Each rank trains on its own shard of the 60k training set (index
//! `≡ rank mod world`); gradients are averaged, so it's data-parallel SGD over
//! the whole dataset. Reports **train loss**, **test accuracy**, and
//! **throughput** (samples/s).
//!
//! MNIST is auto-downloaded (via `curl` + `gunzip`) to `~/.cache/rlx-mnist` on
//! first run (override with `--data <dir>` or `RLX_MNIST_DIR`).
//!
//! ```bash
//! cargo run --release -p rlx-tune --example mnist                                 # single process
//! cargo run --release -p rlx-tune --example mnist -- --nnodes 4 --overlap         # 4-way DP
//! cargo run --release -p rlx-tune --example mnist -- --nnodes 4 --shard --overlap --bf16 --accum 2
//! cargo run --release -p rlx-tune --example mnist -- --prefetch --steps 800       # background data prefetch
//! # GPU (needs the backend feature); `--big` widens channels so the GPU wins:
//! cargo run --release -p rlx-tune --features cuda --example mnist -- --device cuda --big
//! cargo run --release -p rlx-tune --features cuda --example mnist -- --device cuda --big --resident --batch 1024
//! ```
//!
//! On CUDA the fast path is `--resident --big` with a large `--batch` (the step
//! is conv-compute-bound, so bigger batches scale throughput *up*). It needs a
//! **loadable `libcudnn.so`** — rlx-cuda warns once and falls back to a ~10×
//! slower conv if it can't find one (point `LD_LIBRARY_PATH` at e.g. a PyTorch
//! env's `torch/lib`). Convs default to strict FP32; `RLX_CUDA_CONV_TF32=1` opts
//! into TF32 tensor cores (~1.4×, but it can destabilize very-large-batch
//! training — leave it off for training, use it for inference).
//!
//! Model: `conv3×3/s2(1→C1) → relu → conv3×3/s2(C1→C2) → relu → flatten →
//! linear → softmax-cross-entropy` (C1/C2 = 16/32, or 64/128 with `--big`;
//! all-convolutional so no pooling op is needed).

use rlx_ir::infer::GraphExt;
use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_runtime::{CompileOptions, CompiledGraph, Device, Session};
use rlx_tune::cluster::{Role, launch_or_join};
use rlx_tune::{DpConfig, ParamSlot, ResidentTrainer, StepMetrics, Trainer};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

const PX: usize = 28 * 28; // 784
const CLASSES: usize = 10;
const MEAN: f32 = 0.1307; // standard MNIST normalization
const STD: f32 = 0.3081;

// --- MNIST loading ---------------------------------------------------------

const FILES: [&str; 4] = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
];

/// Download + decompress MNIST into `dir` if missing. Each process/machine
/// fetches independently (needed for a real multi-node run — separate
/// filesystems), downloading to a per-pid temp then atomically renaming, so
/// co-located ranks on a shared fs never corrupt each other's file.
fn ensure_mnist(dir: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let pid = std::process::id();
    for f in FILES {
        let out = dir.join(f);
        if out.exists() {
            continue;
        }
        let gz = dir.join(format!("{f}.{pid}.gz"));
        let tmp = dir.join(format!("{f}.{pid}"));
        let url = format!("https://storage.googleapis.com/cvdf-datasets/mnist/{f}.gz");
        eprintln!("downloading {f} …");
        let ok = std::process::Command::new("curl")
            .args(["-fsSL", "--max-time", "180", "-o"])
            .arg(&gz)
            .arg(&url)
            .status()?
            .success();
        anyhow::ensure!(ok, "curl failed for {url}");
        let ok = std::process::Command::new("gunzip")
            .arg("-f")
            .arg(&gz)
            .status()?
            .success();
        anyhow::ensure!(ok, "gunzip failed for {gz:?}"); // → dir/{f}.{pid}
        std::fs::rename(&tmp, &out)?; // atomic publish
    }
    anyhow::ensure!(
        FILES.iter().all(|f| dir.join(f).exists()),
        "MNIST files missing in {dir:?}"
    );
    Ok(())
}

/// Read an IDX3 image file → raw `u8` bytes (`count · 784`).
fn read_images(path: &Path) -> anyhow::Result<Vec<u8>> {
    let b = std::fs::read(path)?;
    anyhow::ensure!(
        b.len() > 16 && b[2] == 0x08 && b[3] == 0x03,
        "bad idx3 {path:?}"
    );
    Ok(b[16..].to_vec())
}

/// Read an IDX1 label file → `u8` labels.
fn read_labels(path: &Path) -> anyhow::Result<Vec<u8>> {
    let b = std::fs::read(path)?;
    anyhow::ensure!(
        b.len() > 8 && b[2] == 0x08 && b[3] == 0x01,
        "bad idx1 {path:?}"
    );
    Ok(b[8..].to_vec())
}

#[inline]
fn norm(byte: u8) -> f32 {
    (byte as f32 / 255.0 - MEAN) / STD
}

// --- model -----------------------------------------------------------------

fn pseudo(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / 16_777_216.0 - 0.5) * 0.2
        })
        .collect()
}

fn winit(n: usize, fan_in: usize, seed: u32) -> Vec<f32> {
    let s = 1.0 / (fan_in as f32).sqrt() / 0.1;
    pseudo(n, seed).into_iter().map(|v| v * s).collect()
}

/// Flattened feature size after two stride-2 convs: `c2 · 7 · 7`.
fn flat_dim(c2: usize) -> usize {
    c2 * 7 * 7
}

fn init_params(c1: usize, c2: usize) -> HashMap<String, Vec<f32>> {
    let flat = flat_dim(c2);
    let mut p = HashMap::new();
    p.insert("c1".to_string(), winit(c1 * 9, 9, 1));
    p.insert("c2".to_string(), winit(c2 * c1 * 9, c1 * 9, 2));
    p.insert("fc".to_string(), winit(flat * CLASSES, flat, 3));
    p.insert("fb".to_string(), vec![0.0; CLASSES]);
    p
}

/// Build the CNN (channels `c1`, `c2`) up to logits for batch `nb`; returns the
/// graph, the `logits` node, and the trainable param slots.
fn build_body(nb: usize, c1c: usize, c2c: usize) -> (Graph, NodeId, [ParamSlot; 4]) {
    let f = DType::F32;
    let flat = flat_dim(c2c);
    let mut g = Graph::new("mnist");
    let x = g.input("x", Shape::new(&[nb, 1, 28, 28], f));
    let c1 = g.param("c1", Shape::new(&[c1c, 1, 3, 3], f));
    let c2 = g.param("c2", Shape::new(&[c2c, c1c, 3, 3], f));
    let fc = g.param("fc", Shape::new(&[flat, CLASSES], f));
    let fb = g.param("fb", Shape::new(&[CLASSES], f));

    let h = g.conv2d(x, c1, [3, 3], [2, 2], [1, 1], [1, 1], 1);
    let h = g.activation(Activation::Relu, h, Shape::new(&[nb, c1c, 14, 14], f));
    let h = g.conv2d(h, c2, [3, 3], [2, 2], [1, 1], [1, 1], 1);
    let h = g.activation(Activation::Relu, h, Shape::new(&[nb, c2c, 7, 7], f));
    let flat_h = g.reshape_(h, vec![nb as i64, flat as i64]);
    let fc_out = g.matmul(flat_h, fc, Shape::new(&[nb, CLASSES], f));
    let logits = g.add(fc_out, fb);

    let wrt = [
        ParamSlot {
            name: "c1".into(),
            node: c1,
        },
        ParamSlot {
            name: "c2".into(),
            node: c2,
        },
        ParamSlot {
            name: "fc".into(),
            node: fc,
        },
        ParamSlot {
            name: "fb".into(),
            node: fb,
        },
    ];
    (g, logits, wrt)
}

/// One batch of `batch` normalized images drawn (with replacement) from this
/// rank's `shard` of the training set.
fn gen_batch(
    images: &[u8],
    labels: &[u8],
    shard: &[usize],
    batch: usize,
    seed: u32,
) -> (Vec<f32>, Vec<f32>) {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    let mut x = Vec::with_capacity(batch * PX);
    let mut lab = Vec::with_capacity(batch);
    for _ in 0..batch {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let idx = shard[(s >> 8) as usize % shard.len()];
        x.extend(images[idx * PX..(idx + 1) * PX].iter().map(|&b| norm(b)));
        lab.push(labels[idx] as f32);
    }
    (x, lab)
}

/// Top-1 accuracy of `params` on the first `eb·(n/eb)` test samples.
fn evaluate(
    sess: &mut CompiledGraph,
    params: &HashMap<String, Vec<f32>>,
    images: &[u8],
    labels: &[u8],
    eb: usize,
) -> f32 {
    for (k, v) in params {
        sess.set_param(k, v);
    }
    let chunks = labels.len() / eb;
    let mut x = vec![0.0f32; eb * PX];
    let mut correct = 0usize;
    for c in 0..chunks {
        for j in 0..eb {
            let idx = c * eb + j;
            for (t, &b) in x[j * PX..(j + 1) * PX]
                .iter_mut()
                .zip(&images[idx * PX..(idx + 1) * PX])
            {
                *t = norm(b);
            }
        }
        let outs = sess.run(&[("x", &x)]);
        let logits = &outs[0];
        for j in 0..eb {
            let row = &logits[j * CLASSES..(j + 1) * CLASSES];
            let pred = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            correct += (pred == labels[c * eb + j] as usize) as usize;
        }
    }
    correct as f32 / (chunks * eb) as f32
}

fn main() -> anyhow::Result<()> {
    // Fast CPU conv (im2col + BLAS) by default — ~10× over the naive reference
    // kernel for this CNN, same result. Override with `RLX_FAST_CONV=0`. Set
    // per process (before any conv compiles), so it also covers spawned /
    // remote ranks, which each run `main`.
    if std::env::var_os("RLX_FAST_CONV").is_none() {
        rlx_ir::env::set("RLX_FAST_CONV", "1");
    }
    let args: Vec<String> = std::env::args().collect();
    let flag = |n: &str| args.iter().any(|a| a == n);
    let uval = |n: &str| {
        args.iter()
            .position(|a| a == n)
            .and_then(|i| args.get(i + 1)?.parse::<usize>().ok())
    };
    let sval = |n: &str| {
        args.iter()
            .position(|a| a == n)
            .and_then(|i| args.get(i + 1).cloned())
    };

    let (rank, world, comm) = match launch_or_join()? {
        Role::Launcher => return Ok(()),
        Role::Worker { rank, world, comm } => (rank as usize, world as usize, comm),
    };

    let dir: PathBuf = sval("--data")
        .or_else(|| std::env::var("RLX_MNIST_DIR").ok())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/rlx-mnist")
        });
    ensure_mnist(&dir)?;

    let train_x = read_images(&dir.join("train-images-idx3-ubyte"))?;
    let train_y = read_labels(&dir.join("train-labels-idx1-ubyte"))?;
    let n_train = train_y.len();

    let batch = uval("--batch").unwrap_or(64);
    let steps = uval("--steps").unwrap_or(400);
    let eval_every = uval("--eval-every").unwrap_or(100);

    // This rank's data shard: indices ≡ rank (mod world).
    let shard: Vec<usize> = (rank..n_train).step_by(world).collect();

    // --device metal|cuda|gpu|cpu runs forward+backward on that device
    // (needs the matching build feature: `--features metal|cuda|gpu`).
    let dev = match sval("--device").as_deref() {
        Some("metal") => Device::Metal,
        Some("cuda") => Device::Cuda,
        Some("gpu") => Device::Gpu,
        _ => Device::Cpu,
    };
    let mut cfg = DpConfig::new(1e-3).log_every(50).device(dev);
    if flag("--shard") {
        cfg = cfg.shard();
    }
    if flag("--overlap") {
        cfg = cfg.overlap();
    }
    if flag("--bf16") {
        cfg = cfg.bf16();
    }
    if let Some(a) = uval("--accum") {
        cfg = cfg.grad_accum(a);
    }
    let ga = cfg.grad_accum.max(1);
    // `--big` widens the conv channels (16/32 → 64/128, ~16× the conv compute) —
    // enough to make a GPU (`--device cuda`) clearly beat the CPU.
    let (c1, c2) = if flag("--big") { (64, 128) } else { (16, 32) };

    // Training graph (scalar loss) + trainer.
    let (mut g, logits, wrt) = build_body(batch, c1, c2);
    let f = DType::F32;
    let labels_in = g.input("labels", Shape::new(&[batch], f));
    let ce = g.softmax_cross_entropy_with_logits(logits, labels_in);
    let loss = g.mean(ce, vec![0], false);
    g.set_outputs(vec![loss]);

    // --resident: GPU-resident *fused* optimizer — the Adam update runs inside
    // the graph and params + moments live in device handles, so nothing
    // round-trips to host per step. Single process (device residency is a
    // single-GPU optimization); ignores `comm`.
    if flag("--resident") {
        let wd = sval("--wd")
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.0);
        let mut rt = ResidentTrainer::new(&g, &wrt, &init_params(c1, c2), &cfg.adam, wd, dev)?;
        let eb = 1000usize;
        let test_x = read_images(&dir.join("t10k-images-idx3-ubyte"))?;
        let test_y = read_labels(&dir.join("t10k-labels-idx1-ubyte"))?;
        let (mut ge, elogits, _) = build_body(eb, c1, c2);
        ge.set_outputs(vec![elogits]);
        let mut eval_sess = Session::new(dev).compile_with(ge, &CompileOptions::new());
        eprintln!(
            "resident optimizer: {} | {}",
            if rt.is_resident() {
                "device-resident"
            } else {
                "host-fallback"
            },
            cfg.describe(),
        );
        eprintln!(
            "  init test acc {:.2}%",
            evaluate(&mut eval_sess, &rt.params(), &test_x, &test_y, eb) * 100.0
        );
        let t0 = Instant::now();
        for step in 0..steps {
            let sd = (step * 131 + 7) as u32;
            let (x, l) = gen_batch(&train_x, &train_y, &shard, batch, sd);
            let loss = rt.step(&[("x", &x), ("labels", &l)]);
            if step.is_multiple_of(cfg.log_every) {
                eprintln!("step {step:>4} | loss {loss:.6}");
            }
            if (step + 1).is_multiple_of(eval_every) {
                let acc = evaluate(&mut eval_sess, &rt.params(), &test_x, &test_y, eb);
                eprintln!("  step {} test acc {:.2}%", step + 1, acc * 100.0);
            }
        }
        let wall = t0.elapsed().as_secs_f64();
        let acc = evaluate(&mut eval_sess, &rt.params(), &test_x, &test_y, eb);
        println!(
            "done: test acc {:.2}% | {:.0} samples/s (resident {}, {steps} steps in {wall:.2}s)",
            acc * 100.0,
            (batch * steps) as f64 / wall,
            if rt.is_resident() { "device" } else { "host" },
        );
        return Ok(());
    }

    let mut trainer = Trainer::new(g, &wrt, &init_params(c1, c2), steps, comm.as_deref(), &cfg)?;

    // Eval session (rank 0 only): forward → logits, eval batch 1000.
    let eb = 1000usize;
    let (test_x, test_y);
    let mut eval_sess = if rank == 0 {
        test_x = read_images(&dir.join("t10k-images-idx3-ubyte"))?;
        test_y = read_labels(&dir.join("t10k-labels-idx1-ubyte"))?;
        let (mut ge, elogits, _) = build_body(eb, c1, c2);
        ge.set_outputs(vec![elogits]);
        Some(Session::new(dev).compile_with(ge, &CompileOptions::new()))
    } else {
        test_x = Vec::new();
        test_y = Vec::new();
        None
    };

    if rank == 0 {
        eprintln!(
            "MNIST: {n_train} train / {} test | shard {} imgs/rank | batch {batch} × accum {ga} × {world} ranks | {}",
            test_y.len(),
            shard.len(),
            cfg.describe(),
        );
        if let Some(s) = eval_sess.as_mut() {
            let acc = evaluate(s, &trainer.params(), &test_x, &test_y, eb);
            eprintln!("  init test acc {:.2}%", acc * 100.0);
        }
    }

    // Per-rank data generator: fresh batch per (step, micro) from this shard.
    let (imgs, labs, shard_v) = (train_x, train_y, shard);
    let seed = |step: usize, micro: usize| (rank * 1_000_003 + step * 131 + micro * 977 + 7) as u32;
    let mut next_batch = |step: usize, micro: usize| {
        let (x, l) = gen_batch(&imgs, &labs, &shard_v, batch, seed(step, micro));
        vec![("x".to_string(), x), ("labels".to_string(), l)]
    };

    let sps = |m: &StepMetrics| (batch * ga * world) as f64 / (m.step_ms / 1e3);
    let mut train_secs = 0.0f64; // compute only — excludes periodic eval

    if flag("--prefetch") {
        // Background data prefetch (final eval only).
        let on_step = |m: &StepMetrics| {
            if rank == 0 {
                eprintln!("{m} | {:>7.0} samples/s", sps(m));
            }
        };
        let t0 = Instant::now();
        trainer.run_prefetched(next_batch, on_step)?;
        train_secs = t0.elapsed().as_secs_f64();
    } else {
        // Manual loop so rank 0 can eval test accuracy periodically.
        while trainer.total_steps_remaining() > 0 {
            let t0 = Instant::now();
            let m = trainer.step(&mut next_batch)?;
            train_secs += t0.elapsed().as_secs_f64();
            if rank == 0 && m.step.is_multiple_of(cfg.log_every) {
                eprintln!("{m} | {:>7.0} samples/s", sps(&m));
            }
            if rank == 0 && (m.step + 1).is_multiple_of(eval_every) {
                if let Some(s) = eval_sess.as_mut() {
                    let acc = evaluate(s, &trainer.params(), &test_x, &test_y, eb);
                    eprintln!("  step {} test acc {:.2}%", m.step + 1, acc * 100.0);
                }
            }
        }
    }

    if rank == 0 {
        let acc = eval_sess
            .as_mut()
            .map(|s| evaluate(s, &trainer.params(), &test_x, &test_y, eb))
            .unwrap_or(f32::NAN);
        let total = batch * ga * world * steps;
        println!(
            "done: test acc {:.2}% | {:.0} samples/s ({total} samples over {world} rank(s), {train_secs:.2}s compute)",
            acc * 100.0,
            total as f64 / train_secs,
        );
    }
    Ok(())
}
