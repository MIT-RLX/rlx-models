// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! LSTM vs SRU as the recurrent core, trained the same way and compared on the
//! two things that matter here: error against the teacher, and cost.
//!
//! The LSTM's recurrent term is a matmul (`h @ W_hh`, 4H x H per layer —
//! 32,768 MACs, 41% of the whole model). An SRU's is elementwise, so all its
//! matmuls depend only on `x`. Two consequences beyond the MAC count:
//!
//! * `c_t = f*c_{t-1} + (1-f)*g` is a convex combination, so |c| <= 1 by
//!   induction. The LSTM's `c_t = f*c_{t-1} + i*g` has independent gates and
//!   grows — measured max |cell| 326.9, which is what forced a 36-bit
//!   accumulator and the `narrow-acc` accuracy trade.
//! * Two sigmoids per unit instead of three sigmoids and two tanh.
//!
//! Both students are trained **from scratch**: an SRU cannot inherit LSTM
//! weights, so fine-tuning one and not the other would settle nothing.
//!
//! ## Cost, which is settled before any training runs
//!
//! MACs per 16 ms frame, and what they imply on the two deployment targets
//! (`rv32imc` at 8.16 instructions/MAC measured; FPGA at 2 cycles/MAC serial):
//!
//! | arch | MACs | rel | net insn | % of C3 core | FPGA cycles | min clock |
//! |---|---|---|---|---|---|---|
//! | LSTM H=64 | 79,295 | 100% | 924,128 | 114% | 173,070 | 10.8 MHz |
//! | GRU H=64 | 61,887 | 78% | 721,249 | 106% | 135,075 | 8.4 MHz |
//! | SRU H=64 | 37,311 | 47% | 434,833 | 95% | 81,435 | 5.0 MHz |
//! | SRU H=48 | 27,071 | 34% | 315,493 | 90% | 59,085 | 3.6 MHz |
//! | LSTM H=32 | 30,175 | 38% | 351,668 | 92% | 70,260 | 4.3 MHz |
//!
//! So the question this example answers is not whether SRU is cheaper — it is,
//! by 2.1x at equal width — but whether it is *as accurate* at that price.
//!
//! Note the training cost runs the other way: SRU takes ~3.7x longer per step
//! here, because the elementwise recurrence traces to many small ops where the
//! LSTM gets one fused matmul. That is a property of the traced graph, not of
//! the deployed kernel, and it does not appear in any row above.
//!
//! Run: cargo run --release -p rlx-ten-vad --example arch_compare -- \
//!          --arch sru --hidden 64 --steps 4000

use rlx_optim::AdamW;
use rlx_runtime::Device;
use rlx_ten_vad_core::net::Net;
use rlx_ten_vad_core::weights::embedded_net;
use rlx_ten_vad_core::{CONTEXT_FRAMES, FEATURE_LEN};
use rlx_tensor::{DType, Func, GraphScope, LrSchedule, Tensor, shape};
use std::path::Path;

const CH: usize = 16;
const FLAT: usize = 80;
const DENSE: usize = 32;
const STACK: usize = CONTEXT_FRAMES * FEATURE_LEN;
const CONV_MACS: usize = 5535;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arch {
    Lstm,
    Sru,
}

impl Arch {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "lstm" => Some(Self::Lstm),
            "sru" => Some(Self::Sru),
            _ => None,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Lstm => "LSTM",
            Self::Sru => "SRU",
        }
    }
    /// Gate projections per unit: LSTM i/f/g/o, SRU x-tilde/f/r.
    fn gates(self) -> usize {
        match self {
            Self::Lstm => 4,
            Self::Sru => 3,
        }
    }
    /// MACs per frame for the whole model at hidden size `h`.
    fn macs(self, h: usize) -> usize {
        let g = self.gates();
        let rec = match self {
            // W_ih [IN, gH] + W_hh [H, gH], both layers.
            Self::Lstm => g * h * (FLAT + h) + g * h * (h + h),
            // x-only projections; the recurrence is elementwise (O(h), ignored).
            Self::Sru => g * h * FLAT + g * h * h,
        };
        CONV_MACS + rec + 2 * h * DENSE + DENSE
    }
}

/// Shared conv stack: `[B,1,3,41]` -> `[B,80]`.
fn conv_stack(x: &Tensor, p: &Params, b: i64) -> Tensor {
    let y = x.conv2d(&p.c0dw, [3, 3], [1, 1], [0, 0], [1, 1], 1);
    let y = y.conv2d(&p.c0pw, [1, 1], [1, 1], [0, 0], [1, 1], 1);
    let y = (&y + &p.c0b.reshape([1, CH as i64, 1, 1])).relu();
    let y = y.transpose([0, 1, 3, 2]).max_pool2d([3, 1], [2, 1], [0, 0]);
    let y = y.pad(2, 1, 1, 0.0);
    let y = y.conv2d(&p.s1dw, [3, 1], [2, 1], [0, 0], [1, 1], CH);
    let y = y.conv2d(&p.s1pw, [1, 1], [1, 1], [0, 0], [1, 1], 1);
    let y = (&y + &p.s1b.reshape([1, CH as i64, 1, 1])).relu();
    let y = y.pad(2, 0, 1, 0.0);
    let y = y.conv2d(&p.s2dw, [3, 1], [2, 1], [0, 0], [1, 1], CH);
    let y = y.conv2d(&p.s2pw, [1, 1], [1, 1], [0, 0], [1, 1], 1);
    let y = (&y + &p.s2b.reshape([1, CH as i64, 1, 1])).relu();
    y.reshape([b, CH as i64, 5])
        .transpose([0, 2, 1])
        .reshape([b, FLAT as i64])
}

struct Params {
    c0dw: Tensor,
    c0pw: Tensor,
    c0b: Tensor,
    s1dw: Tensor,
    s1pw: Tensor,
    s1b: Tensor,
    s2dw: Tensor,
    s2pw: Tensor,
    s2b: Tensor,
    l1w: Tensor,
    l1b: Tensor,
    l2w: Tensor,
    l2b: Tensor,
    d1w: Tensor,
    d1b: Tensor,
    d2w: Tensor,
    d2b: Tensor,
    /// SRU only: elementwise recurrent coefficients, and the highway
    /// projection layer 1 needs because its input width (80) differs from H.
    v1: Option<Tensor>,
    v2: Option<Tensor>,
    hw1: Option<Tensor>,
}

fn params(s: &mut GraphScope, arch: Arch, h: usize) -> Params {
    let g = arch.gates();
    let in1 = match arch {
        Arch::Lstm => FLAT + h,
        Arch::Sru => FLAT,
    };
    let in2 = match arch {
        Arch::Lstm => 2 * h,
        Arch::Sru => h,
    };
    let gh = g * h;
    Params {
        c0dw: s.param("c0dw", shape![1, 1, 3, 3]),
        c0pw: s.param("c0pw", shape![CH, 1, 1, 1]),
        c0b: s.param("c0b", shape![CH]),
        s1dw: s.param("s1dw", shape![CH, 1, 3, 1]),
        s1pw: s.param("s1pw", shape![CH, CH, 1, 1]),
        s1b: s.param("s1b", shape![CH]),
        s2dw: s.param("s2dw", shape![CH, 1, 3, 1]),
        s2pw: s.param("s2pw", shape![CH, CH, 1, 1]),
        s2b: s.param("s2b", shape![CH]),
        l1w: s.param("l1w", shape![in1, gh]),
        l1b: s.param("l1b", shape![gh]),
        l2w: s.param("l2w", shape![in2, gh]),
        l2b: s.param("l2b", shape![gh]),
        d1w: s.param("d1w", shape![2 * h, DENSE]),
        d1b: s.param("d1b", shape![DENSE]),
        d2w: s.param("d2w", shape![DENSE, 1]),
        d2b: s.param("d2b", shape![1]),
        v1: (arch == Arch::Sru).then(|| s.param("v1", shape![2 * h])),
        v2: (arch == Arch::Sru).then(|| s.param("v2", shape![2 * h])),
        hw1: (arch == Arch::Sru).then(|| s.param("hw1", shape![FLAT, h])),
    }
}

fn lstm_step(
    s: &mut GraphScope,
    x: &Tensor,
    st: (&Tensor, &Tensor),
    w: &Tensor,
    b: &Tensor,
    h: usize,
) -> (Tensor, Tensor) {
    let (hp, cp) = st;
    let z = &s.cat(&[x, hp], 1).matmul(w) + b;
    let i = z.narrow(1, 0, h).sigmoid();
    let f = z.narrow(1, h, h).sigmoid();
    let g = z.narrow(1, 2 * h, h).tanh();
    let o = z.narrow(1, 3 * h, h).sigmoid();
    let c = &(&f * cp) + &(&i * &g);
    (&o * &c.tanh(), c)
}

/// One SRU step. Note what is *not* here: any matmul against `c`.
///
/// `v` supplies the recurrent coupling elementwise, so the only sequential work
/// is `h` multiply-adds. `c` is a convex combination of its previous value and
/// a bounded input, so it cannot grow.
fn sru_step(
    x: &Tensor,
    c_prev: &Tensor,
    skip: &Tensor,
    w: &Tensor,
    b: &Tensor,
    v: &Tensor,
    h: usize,
) -> (Tensor, Tensor) {
    let z = &x.matmul(w) + b;
    let g = z.narrow(1, 0, h).tanh();
    let f = (&z.narrow(1, h, h) + &(&v.narrow(0, 0, h) * c_prev)).sigmoid();
    let r = (&z.narrow(1, 2 * h, h) + &(&v.narrow(0, h, h) * c_prev)).sigmoid();
    let one = 1.0f32;
    let c = &(&f * c_prev) + &(&(&f * -one + one) * &g);
    let hn = &(&r * &c.tanh()) + &(&(&r * -one + one) * skip);
    (hn, c)
}

/// Same network, but the graph's output is the probability sequence rather
/// than the loss — so a trained model can be scored without a scalar
/// re-implementation of whichever cell it uses.
fn build_probs(arch: Arch, h: usize, frames: usize) -> Func {
    Func::new("arch-eval", move |s| {
        let feat = s.input("feat", shape![frames, STACK]);
        let p = params(s, arch, h);
        let zeros = |s: &mut GraphScope| s.constant_nd(vec![0.0; h], vec![1, h], DType::F32);
        let (mut h1, mut c1) = (zeros(s), zeros(s));
        let (mut h2, mut c2) = (zeros(s), zeros(s));
        let mut probs: Vec<Tensor> = Vec::with_capacity(frames);
        for t in 0..frames {
            let x = feat.narrow(0, t, 1).reshape([1, 1, 3, 41]);
            let flat = conv_stack(&x, &p, 1);
            let (nh1, nc1, nh2, nc2) = match arch {
                Arch::Lstm => {
                    let (a, b_) = lstm_step(s, &flat, (&h1, &c1), &p.l1w, &p.l1b, h);
                    let (c, d) = lstm_step(s, &a, (&h2, &c2), &p.l2w, &p.l2b, h);
                    (a, b_, c, d)
                }
                Arch::Sru => {
                    let skip1 = flat.matmul(p.hw1.as_ref().unwrap());
                    let (a, b_) = sru_step(
                        &flat,
                        &c1,
                        &skip1,
                        &p.l1w,
                        &p.l1b,
                        p.v1.as_ref().unwrap(),
                        h,
                    );
                    let (c, d) = sru_step(&a, &c2, &a, &p.l2w, &p.l2b, p.v2.as_ref().unwrap(), h);
                    (a, b_, c, d)
                }
            };
            h1 = nh1;
            c1 = nc1;
            h2 = nh2;
            c2 = nc2;
            let d = (&s.cat(&[&h2, &h1], 1).matmul(&p.d1w) + &p.d1b).relu();
            probs.push((&d.matmul(&p.d2w) + &p.d2b).sigmoid());
        }
        let refs: Vec<&Tensor> = probs.iter().collect();
        s.cat(&refs, 0).reshape([frames as i64])
    })
}

fn build(arch: Arch, h: usize, batch: usize, frames: usize, warmup: usize) -> Func {
    Func::new("arch-compare", move |s| {
        let b = batch as i64;
        let feat = s.input("feat", shape![frames * batch, STACK]);
        let tgt = s.input("tgt", shape![frames * batch, 1]);
        let p = params(s, arch, h);

        let zeros =
            |s: &mut GraphScope| s.constant_nd(vec![0.0; batch * h], vec![batch, h], DType::F32);
        let (mut h1, mut c1) = (zeros(s), zeros(s));
        let (mut h2, mut c2) = (zeros(s), zeros(s));

        let mut loss: Option<Tensor> = None;
        for t in 0..frames {
            let x = feat.narrow(0, t * batch, batch).reshape([b, 1, 3, 41]);
            let flat = conv_stack(&x, &p, b);

            let (nh1, nc1, nh2, nc2) = match arch {
                Arch::Lstm => {
                    let (a, b_) = lstm_step(s, &flat, (&h1, &c1), &p.l1w, &p.l1b, h);
                    let (c, d) = lstm_step(s, &a, (&h2, &c2), &p.l2w, &p.l2b, h);
                    (a, b_, c, d)
                }
                Arch::Sru => {
                    // Layer 1's input is 80 wide and its state is H, so the
                    // highway skip needs a projection; layer 2's widths match.
                    let skip1 = flat.matmul(p.hw1.as_ref().unwrap());
                    let (a, b_) = sru_step(
                        &flat,
                        &c1,
                        &skip1,
                        &p.l1w,
                        &p.l1b,
                        p.v1.as_ref().unwrap(),
                        h,
                    );
                    let (c, d) = sru_step(&a, &c2, &a, &p.l2w, &p.l2b, p.v2.as_ref().unwrap(), h);
                    (a, b_, c, d)
                }
            };
            h1 = nh1;
            c1 = nc1;
            h2 = nh2;
            c2 = nc2;

            let d = (&s.cat(&[&h2, &h1], 1).matmul(&p.d1w) + &p.d1b).relu();
            let prob = (&d.matmul(&p.d2w) + &p.d2b).sigmoid();
            if t >= warmup {
                let e = &prob - &tgt.narrow(0, t * batch, batch);
                let sq = (&e * &e).mean_all();
                loss = Some(loss.map_or_else(|| sq.clone(), |acc| &acc + &sq));
            }
        }
        &loss.expect("a scored frame") * (1.0 / (frames - warmup) as f64)
    })
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.0
    }
    fn normal(&mut self) -> f32 {
        // Irwin-Hall: 12 uniforms minus 6 is close enough to N(0,1) for init.
        let mut s = 0.0f32;
        for _ in 0..12 {
            s += (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        }
        s - 6.0
    }
}

fn read_f32(p: &Path) -> Vec<f32> {
    let raw = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    raw.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn main() -> anyhow::Result<()> {
    let (mut arch, mut hidden) = (Arch::Sru, 64usize);
    let (mut steps, mut batch, mut frames, mut lr) = (4000usize, 8usize, 24usize, 1e-3f32);
    let (mut seed, mut eval_every) = (1u64, 500usize);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut num = || args.next().and_then(|v| v.parse().ok());
        match a.as_str() {
            "--arch" => {
                let v = args.next().unwrap_or_default();
                arch = Arch::parse(&v).unwrap_or_else(|| panic!("--arch lstm|sru, got {v:?}"));
            }
            "--hidden" => hidden = num().unwrap_or(hidden),
            "--steps" => steps = num().unwrap_or(steps),
            "--batch" => batch = num().unwrap_or(batch),
            "--frames" => frames = num().unwrap_or(frames),
            "--lr" => lr = args.next().and_then(|v| v.parse().ok()).unwrap_or(lr),
            "--seed" => seed = args.next().and_then(|v| v.parse().ok()).unwrap_or(seed),
            "--eval-every" => eval_every = num().unwrap_or(eval_every),
            _ => {}
        }
    }
    let warmup = frames / 4;
    let macs = arch.macs(hidden);
    println!(
        "{} H={hidden}: {macs} MACs/frame ({:.0}% of the 79,295-MAC LSTM-64)",
        arch.label(),
        100.0 * macs as f64 / 79_295.0
    );

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let feats = read_f32(&root.join("target/distill/features.f32"));
    let n_frames = feats.len() / FEATURE_LEN;
    let train_frames = n_frames * 9 / 10;
    let val_len = 128usize;
    let val_starts: Vec<usize> = (0..24)
        .map(|i| train_frames + i * ((n_frames - train_frames - val_len - 1) / 24))
        .collect();

    // Random init, both architectures alike: fan-in scaled normal.
    let mut irng = Rng(seed ^ 0x5EED);
    let model = build(arch, hidden, batch, frames, warmup);
    let mut m = model.init_params(move |name, dims| {
        let n: usize = dims.iter().product();
        let fan_in = if dims.len() >= 2 { dims[0] } else { 1 };
        let sd = (1.0f32 / fan_in.max(1) as f32).sqrt();
        // Biases at zero, except SRU's `v` which starts small and nonzero.
        if dims.len() == 1 && !name.starts_with('v') {
            vec![0.0; n]
        } else {
            (0..n).map(|_| irng.normal() * sd).collect()
        }
    });

    let mut opt = AdamW::new(lr);
    opt.weight_decay = 0.0;
    let sched = LrSchedule::Cosine {
        base: lr,
        min: lr * 0.05,
        total: steps,
    };
    let mut rng = Rng(seed);
    let mut teacher = Net::new(embedded_net());

    // Evaluation runs the student's own graph, one window at a time, against
    // the teacher started from the same zero state.
    // `init_params` consumes the Func, so the eval graph is rebuilt per call.
    // Evals are every few hundred steps; construction is not the cost.
    let feats_ref = &feats;
    let vs_ref = &val_starts;
    let score = |m: &Func| -> (f32, usize, usize) {
        let vals: std::collections::HashMap<String, Vec<f32>> = m
            .param_names()
            .iter()
            .map(|n| (n.clone(), m.param_binding(n).expect("bound").to_vec()))
            .collect();
        let em = build_probs(arch, hidden, val_len).init_params(move |name, _| vals[name].clone());
        let (mut worst, mut flips, mut n) = (0.0f32, 0usize, 0usize);
        let mut tnet = Net::new(embedded_net());
        for &st in vs_ref {
            tnet.reset();
            let mut fbuf = vec![0.0f32; val_len * STACK];
            let mut tbuf = vec![0.0f32; val_len];
            let mut stack = vec![0.0f32; STACK];
            for t in 0..val_len {
                let row = &feats_ref[(st + t) * FEATURE_LEN..(st + t + 1) * FEATURE_LEN];
                stack.copy_within(FEATURE_LEN.., 0);
                stack[(CONTEXT_FRAMES - 1) * FEATURE_LEN..].copy_from_slice(row);
                fbuf[t * STACK..(t + 1) * STACK].copy_from_slice(&stack);
                tbuf[t] = tnet.forward(&stack);
            }
            let got = em.run_on(Device::Cpu, &[("feat", &fbuf[..])]).remove(0);
            for t in val_len / 4..val_len {
                let (a, b) = (got[t], tbuf[t]);
                worst = worst.max((a - b).abs());
                if (a >= 0.5) != (b >= 0.5) {
                    flips += 1;
                }
                n += 1;
            }
        }
        (worst, flips, n)
    };

    println!("training from scratch: {steps} steps, batch {batch} x {frames} frames, lr {lr:.0e}");
    let t0 = std::time::Instant::now();
    for step in 0..steps {
        let mut fbuf = vec![0.0f32; frames * batch * STACK];
        let mut tbuf = vec![0.0f32; frames * batch];
        for bi in 0..batch {
            let start = (rng.next() as usize) % (train_frames - frames - CONTEXT_FRAMES);
            teacher.reset();
            let mut stack = vec![0.0f32; STACK];
            for t in 0..frames {
                let row = &feats[(start + t) * FEATURE_LEN..(start + t + 1) * FEATURE_LEN];
                stack.copy_within(FEATURE_LEN.., 0);
                stack[(CONTEXT_FRAMES - 1) * FEATURE_LEN..].copy_from_slice(row);
                let at = (t * batch + bi) * STACK;
                fbuf[at..at + STACK].copy_from_slice(&stack);
                tbuf[t * batch + bi] = teacher.forward(&stack);
            }
        }
        let feed: &[(&str, &[f32])] = &[("feat", &fbuf), ("tgt", &tbuf)];
        let (next, loss) =
            m.train_step_all_at_on_clipped(Device::Cpu, &mut opt, &sched, step, 1.0, feed);
        m = next;
        let loss = loss.first().copied().unwrap_or(f32::NAN);
        if step % eval_every == 0 || step == steps - 1 {
            let (w, f, n) = score(&m);
            println!(
                "  step {step:5}  loss {loss:.6}  max|Δ| {w:.3e}  flips {f}/{n}  ({:.0}s)",
                t0.elapsed().as_secs_f64()
            );
        }
    }
    let (worst, flips, n) = score(&m);
    println!(
        "\n{} H={hidden}: max|Δ| {worst:.3e}  flips {flips}/{n}  \
         {macs} MACs ({:.0}% of LSTM-64)  trained in {:.0}s",
        arch.label(),
        100.0 * macs as f64 / 79_295.0,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
