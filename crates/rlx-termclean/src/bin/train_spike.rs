//! Phase-0 Metal training spike.
//!
//! Proves the load-bearing unknown for the whole plan: does rlx's native
//! training (autodiff + optimizer step, through a transformer-shaped forward
//! incl. fused attention) actually run **on Metal**, and does it agree with
//! CPU?
//!
//! It's a thin vertical slice of the real model: a tiny bidirectional
//! (non-causal) fused-attention encoder + a per-token content/chrome head,
//! trained with MSE to overfit a handful of REAL `rlx-termclean` samples. We
//! run the identical model, from the identical initialization, on CPU and on
//! Metal, and compare the loss trajectories + final per-token accuracy.
//!
//! Run: `cargo run -p rlx-termclean --bin rlx-termclean-train-spike --features train`

use rlx_tensor::{
    Adam, Device, Func, GraphScope, MaskKind, Optimizer, Tensor, is_available, shape,
};

use rlx_termclean::{Rng, generate};

/// Fixed problem dimensions (kept tiny so it overfits in a few hundred steps).
#[derive(Clone, Copy)]
struct Dims {
    b: usize,  // batch = number of screens
    l: usize,  // sequence length (chars per screen)
    v: usize,  // vocab size (distinct chars across the batch)
    nh: usize, // attention heads
    dh: usize, // head dim
}
impl Dims {
    fn d(&self) -> usize {
        self.nh * self.dh
    }
}

const PARAMS: &[&str] = &["we", "wq", "wk", "wv", "wo", "bo"];

/// Deterministic small spread to break weight symmetry (no RNG dependency).
fn spread(n: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((((i * 31 + seed * 17) % 23) as f32) - 11.0) * scale)
        .collect()
}

fn init_params(dm: Dims) -> Vec<(String, Vec<f32>)> {
    let d = dm.d();
    vec![
        ("we".into(), spread(dm.v * d, 1, 0.05)),
        ("wq".into(), spread(d * d, 2, 0.05)),
        ("wk".into(), spread(d * d, 3, 0.05)),
        ("wv".into(), spread(d * d, 4, 0.05)),
        ("wo".into(), spread(d, 5, 0.10)),
        ("bo".into(), vec![0.0]),
    ]
}

/// Forward pass. When `with_loss`, output[0] is the scalar MSE loss (needs the
/// `y` input); otherwise output[0] is the per-token logit `[b*l, 1]`.
fn forward(s: &mut GraphScope, dm: Dims, with_loss: bool) -> Tensor {
    let Dims { b, l, v, nh, dh } = dm;
    let d = dm.d();
    let bl = (b * l) as i64;

    let x = s.input("x", shape![b, l, v]); // one-hot chars
    let we = s.param("we", shape![v, d]);
    let wq = s.param("wq", shape![d, d]);
    let wk = s.param("wk", shape![d, d]);
    let wv = s.param("wv", shape![d, d]);
    let wo = s.param("wo", shape![d, 1]);
    let bo = s.param("bo", shape![1]);

    // Embed: [b*l, v] @ [v, d] -> [b*l, d]
    let h = x.reshape(vec![bl, v as i64]).matmul(&we);

    // Project to q/k/v and shape as [b, l, nh, dh] for fused attention.
    let to_heads = |t: Tensor| t.reshape(vec![b as i64, l as i64, nh as i64, dh as i64]);
    // tanh-bounded Q/K caps scores at ±sqrt(dh) so the softmax can't
    // overflow to NaN in this unnormalized net (smooth, not a clamp).
    let q = to_heads(h.matmul(&wq).tanh());
    let k = to_heads(h.matmul(&wk).tanh());
    let vv = to_heads(h.matmul(&wv));

    // Bidirectional (non-causal) self-attention — the encoder's core op.
    let attn = q.attention(&k, &vv, nh, dh, MaskKind::None); // [b, l, nh, dh]
    let ctx = attn.reshape(vec![bl, d as i64]); // [b*l, d]
    let ctx = &ctx + &h; // residual

    let logit = &ctx.matmul(&wo) + &bo; // [b*l, 1]
    if with_loss {
        let y = s.input("y", shape![b * l, 1]);
        let diff = &logit - &y;
        (&diff * &diff).mean([0, 1], false)
    } else {
        logit
    }
}

fn build(dm: Dims, params: &[(String, Vec<f32>)], with_loss: bool) -> Func {
    let mut f = Func::new("enc", move |s| forward(s, dm, with_loss));
    for (n, data) in params {
        f = f.with_param(n.clone(), data.clone());
    }
    f
}

/// One optimizer step, pinned to an explicit device. Mirrors
/// `Func::train_step` but uses `run_on(device, …)` so we control CPU vs Metal.
/// (Adam is elementwise and ignores the shape arg, so `&[len]` is fine.)
fn train_step_on(
    model: &Func,
    device: Device,
    opt: &mut dyn Optimizer,
    feed: &[(&str, &[f32])],
) -> (Func, f32) {
    let outputs = model.value_and_grad(PARAMS).run_on(device, feed);
    let loss = outputs[0][0];
    let mut updated = model.clone();
    for (i, name) in PARAMS.iter().enumerate() {
        let grad = &outputs[i + 1];
        let mut data = model.param_binding(name).unwrap().to_vec();
        opt.step(name, &[data.len()], &mut data, grad);
        updated = updated.with_param(*name, data);
    }
    opt.end_iteration();
    (updated, loss)
}

struct Batch {
    onehot: Vec<f32>,
    labels: Vec<f32>,
    dims: Dims,
}

/// Sample `b` real screens of at least `l` chars, char-tokenize to a per-batch
/// vocab, and build one-hot inputs + per-token content(1)/chrome(0) labels.
fn make_batch(b: usize, l: usize) -> Batch {
    let mut rng = Rng::new(7);
    let mut chosen: Vec<(Vec<char>, Vec<char>)> = Vec::new();
    let mut id = 0u64;
    while chosen.len() < b {
        let s = generate(&mut rng, id);
        id += 1;
        let inp: Vec<char> = s.input.chars().collect();
        let tags: Vec<char> = s.tags.chars().collect();
        if inp.len() >= l {
            chosen.push((inp[..l].to_vec(), tags[..l].to_vec()));
        }
    }

    // Build vocab from the observed chars.
    let mut vocab: Vec<char> = Vec::new();
    for (inp, _) in &chosen {
        for &c in inp {
            if !vocab.contains(&c) {
                vocab.push(c);
            }
        }
    }
    let idx = |c: char| vocab.iter().position(|&x| x == c).unwrap();
    let v = vocab.len();

    let mut onehot = vec![0.0f32; b * l * v];
    let mut labels = vec![0.0f32; b * l];
    for (bi, (inp, tags)) in chosen.iter().enumerate() {
        for (li, (&c, &t)) in inp.iter().zip(tags).enumerate() {
            onehot[(bi * l + li) * v + idx(c)] = 1.0;
            labels[bi * l + li] = if t == 'C' { 1.0 } else { 0.0 };
        }
    }

    Batch {
        onehot,
        labels,
        dims: Dims {
            b,
            l,
            v,
            nh: 2,
            dh: 16,
        },
    }
}

/// Train `steps` iterations on `device` from a fresh copy of `init`; return the
/// loss trajectory (one entry per step) and the trained model.
fn train(
    device: Device,
    dm: Dims,
    init: &[(String, Vec<f32>)],
    feed: &[(&str, &[f32])],
    steps: usize,
) -> (Vec<f32>, Func) {
    let mut model = build(dm, init, true);
    let mut opt = Adam::new(0.01);
    let mut losses = Vec::with_capacity(steps);
    for _ in 0..steps {
        let (next, loss) = train_step_on(&model, device, &mut opt, feed);
        model = next;
        losses.push(loss);
    }
    (losses, model)
}

/// Per-token accuracy of the trained model on CPU (logit > 0.5 vs label).
fn accuracy(dm: Dims, model: &Func, onehot: &[f32], labels: &[f32]) -> f32 {
    // Re-bind the trained params into a logit-emitting graph.
    let mut pred = build(dm, &[], false);
    for name in PARAMS {
        pred = pred.with_param(*name, model.param_binding(name).unwrap().to_vec());
    }
    let logits = pred.run_on(Device::Cpu, &[("x", onehot)]);
    let logit = &logits[0];
    let correct = logit
        .iter()
        .zip(labels)
        .filter(|(p, t)| (**p > 0.5) == (**t > 0.5))
        .count();
    correct as f32 / labels.len() as f32
}

fn main() {
    const L: usize = 64;
    const B: usize = 8;
    const STEPS: usize = 400;

    let batch = make_batch(B, L);
    let dm = batch.dims;
    let feed: &[(&str, &[f32])] = &[("x", &batch.onehot), ("y", &batch.labels)];
    let init = init_params(dm);

    println!(
        "spike: b={} l={} vocab={} d={} (nh={} dh={}), params={}, steps={}",
        dm.b,
        dm.l,
        dm.v,
        dm.d(),
        dm.nh,
        dm.dh,
        init.iter().map(|(_, d)| d.len()).sum::<usize>(),
        STEPS,
    );

    // --- CPU ---
    println!("\n[CPU] training…");
    let (cpu_losses, cpu_model) = train(Device::Cpu, dm, &init, feed, STEPS);
    let cpu_acc = accuracy(dm, &cpu_model, &batch.onehot, &batch.labels);

    // --- Metal ---
    let metal_ok = is_available(Device::Metal);
    let (metal_losses, metal_acc) = if metal_ok {
        println!("[Metal] training…");
        let (ml, mm) = train(Device::Metal, dm, &init, feed, STEPS);
        let acc = accuracy(dm, &mm, &batch.onehot, &batch.labels);
        (Some(ml), Some(acc))
    } else {
        println!("[Metal] UNAVAILABLE on this host — CPU-only run.");
        (None, None)
    };

    // --- Report ---
    println!("\n{:>6}  {:>12}  {:>12}", "step", "cpu_loss", "metal_loss");
    for &i in &[0usize, 25, 50, 100, 200, 300, STEPS - 1] {
        let m = metal_losses.as_ref().map(|v| v[i]).unwrap_or(f32::NAN);
        println!("{i:>6}  {:>12.6}  {:>12.6}", cpu_losses[i], m);
    }

    let c0 = cpu_losses[0];
    let cf = *cpu_losses.last().unwrap();
    println!(
        "\nCPU   : loss {c0:.5} -> {cf:.5}  ({:.1}x drop), per-token acc {:.1}%",
        c0 / cf.max(1e-9),
        100.0 * cpu_acc
    );
    if let (Some(ml), Some(ma)) = (&metal_losses, metal_acc) {
        let m0 = ml[0];
        let mf = *ml.last().unwrap();
        let rel = ((mf - cf) / cf.max(1e-9)).abs();
        println!(
            "Metal : loss {m0:.5} -> {mf:.5}  ({:.1}x drop), per-token acc {:.1}%",
            m0 / mf.max(1e-9),
            100.0 * ma
        );
        println!("step-0 CPU vs Metal loss: {c0:.6} vs {m0:.6} (forward parity)");
        println!(
            "final  CPU vs Metal loss: {cf:.6} vs {mf:.6} (rel diff {:.2}%)",
            100.0 * rel
        );

        let verdict = cf < 0.5 * c0 && mf < 0.5 * m0 && (rel < 0.15 || (mf - cf).abs() < 0.02);
        println!(
            "\nVERDICT: {}",
            if verdict {
                "PASS — Metal training runs, learns, and tracks CPU."
            } else {
                "INVESTIGATE — see trajectory above."
            }
        );
    }

    rlx_tensor::clear_cache();
}
