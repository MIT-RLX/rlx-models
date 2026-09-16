// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Quantisation-aware distillation: train TEN-VAD so it survives low-bit weights.
//!
//! Post-training quantisation of this model hits a wall at int8 — MXFP4 lands
//! 28 decision flips out of 250 on the reference clip, which is unusable. The
//! question this answers is whether the wall is the *format* or the *rounding*:
//! if the network is trained with FP4 rounding in the forward pass, does it
//! learn weights that tolerate the grid?
//!
//! Setup:
//!
//! `--format` selects what the forward emulates: `fp4` (E2M1), `int4`, `int6`,
//! `int8`, or `none` — the control that shows the training loop is not itself
//! what moves the model.
//!
//! * **Student** — the same architecture, traced as an rlx `Func`, with every
//!   2-D and 4-D weight fake-quantised (block-32, power-of-two scale) on the
//!   way into the forward. Gradients are taken at the quantised
//!   point and applied to f32 masters: a straight-through estimator.
//! * **Teacher** — the f32 scalar net, run over the same window from the same
//!   zero state, so targets and student see identical context. Using the
//!   stored per-clip probabilities instead would mismatch: the teacher's state
//!   there evolved from the clip start, the student's starts at zero.
//! * **Loss** — MSE on probabilities, skipping a warmup so the zero-state
//!   transient is not what gets optimised.
//!
//! ```text
//! cargo run -p rlx-ten-vad --release --example dump_distill_set -- --clips 300
//! cargo run -p rlx-ten-vad --release --example qat -- --format fp4 --steps 2000
//! ```
//!
//! ## Reading the numbers
//!
//! **A single run cannot settle anything here.** Window sampling is stochastic
//! and the spread is large: MXFP4 at the default settings scored 88, 104 and 62
//! held-out flips on seeds 1, 2 and 3, and 138 on the original seed — a better
//! than 2x range from the seed alone. With ~100 flips out of 2304 the Poisson
//! standard error is already ~10. Use `--seed` and compare means over several
//! runs; anything under ~20 flips of difference on one run is noise.
//!
//! Two things were tried and measured and did **not** work, recorded so they
//! are not tried again:
//!
//! * **Weighting the loss by the teacher's margin** (`--margin <sigma>`, off by
//!   default). The reasoning was that a flip needs an error larger than
//!   `|t - 0.5|`, so near-threshold frames are the only ones at risk. It is
//!   worse on every seed tried (88->92, 104->108, 62->72). MSE on the
//!   probability already has a `p(1-p)` factor in its gradient with respect to
//!   the logit, which peaks at the boundary; the explicit weight
//!   over-concentrates and lets confident frames drift until they flip.
//! ## Range-aware distillation, and why it did not work
//!
//! The integer net accumulates `i16 x Q15` products in `i64`. On a 32-bit core
//! that is ~2x the instruction count of an `i32` accumulate, and the net is 31%
//! of an ESP32-C3 frame — so narrowing it is worth real time. `--range-lambda`
//! adds a squared hinge on gate pre-activations, cell state and the conv output,
//! aiming to shrink what the accumulator must hold.
//!
//! **It does not work, and the numbers say why.** Measured over the
//! distillation set (`--features range-probe`, and
//! `rlx-ten-vad --example int_ranges` for the integer side):
//!
//! | | value |
//! |---|---|
//! | max \|partial sum\| | 36 bits (i32 needs <= 31) |
//! | max \|pre-activation\| | 191, where sigmoid/tanh saturate by ~16 |
//! | max \|cell\| | 128 |
//! | max \|conv output\| | 18.8 |
//! | max row L1 of the LSTM matrices | 92.0 |
//!
//! At `lambda 0.02` with budgets 12/12/2 the term is plainly active — the loss
//! goes from 0.000000 to 0.002062 at step 0 — and it wrecks the model (0 -> 174
//! held-out flips) while moving the ranges essentially not at all
//! (\|pre\| 191.0 -> 190.6). Two reasons, both structural:
//!
//! * The penalty is a **mean** and the accumulator is set by a **max**. One
//!   outlier contributes 1/(batch*4H) of the mean, so the gradient it receives
//!   is negligible while the bulk of the distribution gets dragged down.
//! * Budget 12 sits far inside a distribution that reaches 191, so the hinge is
//!   active nearly everywhere and fights the distillation loss everywhere.
//!
//! The quantity that actually sets the width is `max_r ||W_r||_1 * max|x|`,
//! reported above: 92.0 * 18.8. Reaching 31 bits needs that product cut ~32x,
//! to about 54. That is not a regularisation nudge, it is a different network.
//!
//! The lever more likely to pay is a **scale** rather than a penalty: the LSTM
//! input is Q15, and storing it at Q10 would drop the partial sums by 5 bits —
//! exactly the gap — at the cost of 5 bits of activation resolution on values
//! whose dynamic range is ~14 bits. That is untested here.
//!
//! ## What pruning costs
//!
//! Measured with `--format none` (the shipped MCU build is int16, which is
//! lossless at 0 flips, so pruning is the only compression on the table), 1200
//! steps at `1e-4`, held-out flips out of 2304:
//!
//! | `--prune` | MACs removed | before QAT | after QAT |
//! |---|---|---|---|
//! | 0.1 | 8.9% | 260 | **96 / 110 / 108** (seeds 1/2/3) |
//! | 0.2 | 18.1% | 369 | 229 |
//! | 0.3 | 27.7% | 424 | 175 |
//!
//! QAT recovers roughly 60% of the damage, and the 10% row is reproducible to
//! within 14 flips across seeds — much tighter than the quantisation runs. But
//! ~105 flips is **4.5% of decisions**, for 8.9% of the MACs. On an ESP32-C3
//! that is about 82 k instructions of 926 k, ~3% of the frame budget. Pruning
//! is not a free lunch here at any fraction tried.
//!
//! * **Training longer at the default learning rate.** 400 -> 1200 steps at
//!   `3e-4` diverges: 88 -> 231, 104 -> 238, 62 -> 120. At `1e-4` the extra
//!   steps help modestly instead (mean 85 -> 75 over three seeds), which is
//!   about 2 sigma and not more.

use rlx_optim::AdamW;
use rlx_runtime::Device;
use rlx_ten_vad_core::net::Net;
use rlx_ten_vad_core::weights::{NetWeights, embedded_net};
use rlx_ten_vad_core::{CONTEXT_FRAMES, FEATURE_LEN, HIDDEN};
use rlx_tensor::{DType, Func, GraphScope, LrSchedule, Tensor, shape};
use std::path::Path;

const CH: usize = 16;
const FLAT: usize = 80;
const DENSE: usize = 32;
const GATES: usize = 4 * HIDDEN;
const STACK: usize = CONTEXT_FRAMES * FEATURE_LEN;

// ---- MXFP4 --------------------------------------------------------------

/// E2M1 magnitudes: 1 sign, 2 exponent, 1 mantissa bits.
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
const BLOCK: usize = 32;

/// What gets removed before the forward pass.
#[derive(Clone, Copy, PartialEq)]
enum Prune {
    None,
    /// Zero the smallest-magnitude fraction of each tensor. Nominally cheap,
    /// but only realisable with a sparse kernel — and an index list costs more
    /// than the multiply it skips in a row-major layout.
    Unstructured(f32),
    /// Drop whole *taps* — rows of a `[taps, outputs]` matmul weight, scored by
    /// L2 norm. This is the one that cuts real MACs: a dead tap leaves the dot
    /// product entirely, contiguously, with nothing to index around.
    Tap(f32),
}

impl Prune {
    fn parse(mode: &str, frac: f32) -> Self {
        match mode {
            _ if frac <= 0.0 => Self::None,
            "unstructured" => Self::Unstructured(frac),
            "tap" => Self::Tap(frac),
            other => panic!("unknown --prune-mode {other:?} (unstructured|tap)"),
        }
    }

    fn label(self) -> String {
        match self {
            Self::None => "dense".into(),
            Self::Unstructured(f) => format!("{:.0}% unstructured", f * 100.0),
            Self::Tap(f) => format!("{:.0}% taps", f * 100.0),
        }
    }
}

/// Per-tensor keep masks, built once from the pretrained weights and then held
/// fixed — a mask recomputed every step is dynamic sparse training, not
/// something you can ship.
fn build_masks(
    base: &[(String, Vec<f32>)],
    shapes: &[(String, Vec<usize>)],
    prune: Prune,
) -> std::collections::HashMap<String, Vec<bool>> {
    let mut masks = std::collections::HashMap::new();
    for (name, v) in base {
        let shape = shapes
            .iter()
            .find(|(m, _)| m == name)
            .map_or_else(|| vec![v.len()], |(_, s)| s.clone());
        // Biases are 0.8% of the weights and every one of them is used; there
        // is nothing to win and a lot to break.
        if shape.len() < 2 {
            continue;
        }
        let keep = match prune {
            Prune::None => continue,
            Prune::Unstructured(f) => {
                let mut mag: Vec<f32> = v.iter().map(|x| x.abs()).collect();
                mag.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let cut = mag[((mag.len() as f32 * f) as usize).min(mag.len() - 1)];
                v.iter().map(|x| x.abs() > cut).collect()
            }
            Prune::Tap(f) => {
                // Only 2-D matmul weights have taps in the sense that matters.
                if shape.len() != 2 {
                    continue;
                }
                let (k, n) = (shape[0], shape[1]);
                let mut norm: Vec<(f32, usize)> = (0..k)
                    .map(|t| {
                        let e: f32 = v[t * n..(t + 1) * n].iter().map(|x| x * x).sum();
                        (e.sqrt(), t)
                    })
                    .collect();
                norm.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let drop: std::collections::HashSet<usize> = norm
                    .iter()
                    .take((k as f32 * f) as usize)
                    .map(|(_, t)| *t)
                    .collect();
                (0..v.len()).map(|i| !drop.contains(&(i / n))).collect()
            }
        };
        masks.insert(name.clone(), keep);
    }
    masks
}

/// MACs per frame removed by a tap mask, counted exactly.
fn macs_saved(
    masks: &std::collections::HashMap<String, Vec<bool>>,
    shapes: &[(String, Vec<usize>)],
) -> usize {
    let mut saved = 0;
    for (name, keep) in masks {
        let Some((_, shape)) = shapes.iter().find(|(m, _)| m == name) else {
            continue;
        };
        if shape.len() != 2 {
            continue;
        }
        let (k, n) = (shape[0], shape[1]);
        let dead = (0..k).filter(|t| !keep[t * n]).count();
        saved += dead * n;
    }
    saved
}

/// Weight format emulated in the forward pass.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    /// E2M1 — 1 sign, 2 exponent, 1 mantissa bit.
    Fp4,
    /// Symmetric integer with `bits` of range.
    Int(u32),
    /// No quantisation — the control that shows the training loop itself is
    /// not what degrades the model.
    None,
}

impl Format {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "fp4" => Some(Self::Fp4),
            "int4" => Some(Self::Int(4)),
            "int6" => Some(Self::Int(6)),
            "int8" => Some(Self::Int(8)),
            "none" => Some(Self::None),
            _ => None,
        }
    }
    fn label(self) -> String {
        match self {
            Self::Fp4 => "MXFP4".into(),
            Self::Int(b) => format!("int{b}, block-32"),
            Self::None => "unquantised".into(),
        }
    }
    /// Largest representable magnitude before the shared scale.
    fn amax(self) -> f32 {
        match self {
            Self::Fp4 => 6.0,
            Self::Int(b) => ((1u32 << (b - 1)) - 1) as f32,
            Self::None => 1.0,
        }
    }
    fn round(self, y: f32) -> f32 {
        match self {
            Self::Fp4 => nearest_e2m1(y),
            Self::Int(b) => {
                let lim = ((1u32 << (b - 1)) - 1) as f32;
                y.round().clamp(-lim - 1.0, lim)
            }
            Self::None => y,
        }
    }
}

fn nearest_e2m1(y: f32) -> f32 {
    let a = y.abs();
    let mut best = E2M1[0];
    let mut err = (a - E2M1[0]).abs();
    for &g in &E2M1[1..] {
        let e = (a - g).abs();
        if e < err {
            err = e;
            best = g;
        }
    }
    if y < 0.0 { -best } else { best }
}

/// Quantise one block in place with a shared power-of-two scale.
///
/// The scale rounds *up* in exponent so the largest magnitude still fits;
/// rounding the other way clips the tail, which is worth orders of magnitude.
fn quant_block(fmt: Format, vals: &mut [f32], stride: usize, base: usize, len: usize) {
    let idx = |i: usize| base + i * stride;
    let amax = (0..len).fold(0.0f32, |m, i| m.max(vals[idx(i)].abs()));
    if amax == 0.0 {
        return;
    }
    let scale = (amax / fmt.amax()).log2().ceil().exp2();
    for i in 0..len {
        let j = idx(i);
        vals[j] = fmt.round(vals[j] / scale) * scale;
    }
}

/// Fake-quantise a parameter to MXFP4, blocking along the axis the matmul
/// contracts over.
///
/// Blocking the other way would give every output its own scale part-way
/// through a dot product, which no kernel can apply — the block layout is a
/// hardware constraint, not a free choice.
///
/// 1-D parameters (biases) are left alone: they are 0.8% of the weights and in
/// any real datapath stay wider than the multiplicands.
fn fake_quant(fmt: Format, shape: &[usize], w: &mut [f32]) {
    if fmt == Format::None {
        return;
    }
    match shape.len() {
        // [K, N] matmul weight: contraction over K, so a block walks K with
        // stride N.
        2 => {
            let (k, n) = (shape[0], shape[1]);
            for col in 0..n {
                for start in (0..k).step_by(BLOCK) {
                    quant_block(fmt, w, n, start * n + col, BLOCK.min(k - start));
                }
            }
        }
        // [C_out, C_in, kH, kW] conv weight: contraction over the trailing
        // dims, which are contiguous.
        4 => {
            let inner = shape[1] * shape[2] * shape[3];
            for oc in 0..shape[0] {
                for start in (0..inner).step_by(BLOCK) {
                    quant_block(fmt, w, 1, oc * inner + start, BLOCK.min(inner - start));
                }
            }
        }
        _ => {}
    }
}

// ---- the student, traced ------------------------------------------------

/// One LSTM step over `[B, IN]` input and `[B, HIDDEN]` state.
/// Squared hinge on |v| above `budget`, meaned. Zero inside the budget, so it
/// only pushes on the values that are actually too large.
fn over_budget(v: &Tensor, budget: f32) -> Tensor {
    let a = (v * v).sqrt();
    let over = (&a - budget).relu();
    (&over * &over).mean_all()
}

fn lstm_step(
    s: &mut GraphScope,
    x: &Tensor,
    h: &Tensor,
    c: &Tensor,
    w: &Tensor,
    b: &Tensor,
    pen: &mut Option<Tensor>,
    budget: Option<(f32, f32)>,
) -> (Tensor, Tensor) {
    let xh = s.cat(&[x, h], 1);
    let z = &xh.matmul(w) + b;
    let i = z.narrow(1, 0, HIDDEN).sigmoid();
    let f = z.narrow(1, HIDDEN, HIDDEN).sigmoid();
    let g = z.narrow(1, 2 * HIDDEN, HIDDEN).tanh();
    let o = z.narrow(1, 3 * HIDDEN, HIDDEN).sigmoid();
    let c_new = &(&f * c) + &(&i * &g);
    let h_new = &o * &c_new.tanh();
    if let Some((z_budget, c_budget)) = budget {
        // Past |z| ~ 16 the sigmoid/tanh LUT is saturated: the deployed net
        // returns the same value for 16 as for 190, so shrinking these costs
        // almost nothing in function value. What it buys is accumulator bits —
        // the measured partial sums need 36, and an i32 datapath needs 31.
        let add = |acc: &mut Option<Tensor>, t: Tensor| {
            *acc = Some(acc.as_ref().map_or(t.clone(), |a| a + &t));
        };
        add(pen, over_budget(&z, z_budget));
        add(pen, over_budget(&c_new, c_budget));
    }
    (h_new, c_new)
}

/// Trace the unrolled student with its distillation loss.
///
/// `warmup` frames are computed but excluded from the loss: every window starts
/// from zero state, and optimising that transient teaches the model to be good
/// at something inference never does.
fn build(batch: usize, frames: usize, warmup: usize, ranges: Option<RangeBudget>) -> Func {
    Func::new("ten-vad-qat", move |s| {
        let b = batch as i64;
        let feat = s.input("feat", shape![frames * batch, STACK]);
        let tgt = s.input("tgt", shape![frames * batch, 1]);
        // Per-frame loss weight, computed host-side from the teacher's margin.
        // Uniform weighting optimises the wrong thing: the deployed metric is
        // decision flips at 0.5, and flipping a frame takes an error larger
        // than |t - 0.5|, so a frame the teacher calls at 0.99 is nearly
        // unflippable while one at 0.52 flips on almost any perturbation.
        // Averaging over all of them spends the model's capacity where it
        // cannot buy a flip.
        let wgt = s.input("wgt", shape![frames * batch, 1]);

        let c0dw = s.param("c0dw", shape![1, 1, 3, 3]);
        let c0pw = s.param("c0pw", shape![CH, 1, 1, 1]);
        let c0b = s.param("c0b", shape![CH]);
        let s1dw = s.param("s1dw", shape![CH, 1, 3, 1]);
        let s1pw = s.param("s1pw", shape![CH, CH, 1, 1]);
        let s1b = s.param("s1b", shape![CH]);
        let s2dw = s.param("s2dw", shape![CH, 1, 3, 1]);
        let s2pw = s.param("s2pw", shape![CH, CH, 1, 1]);
        let s2b = s.param("s2b", shape![CH]);
        let l1w = s.param("l1w", shape![FLAT + HIDDEN, GATES]);
        let l1b = s.param("l1b", shape![GATES]);
        let l2w = s.param("l2w", shape![2 * HIDDEN, GATES]);
        let l2b = s.param("l2b", shape![GATES]);
        let d1w = s.param("d1w", shape![2 * HIDDEN, DENSE]);
        let d1b = s.param("d1b", shape![DENSE]);
        let d2w = s.param("d2w", shape![DENSE, 1]);
        let d2b = s.param("d2b", shape![1]);

        let zeros = |s: &mut GraphScope| {
            s.constant_nd(vec![0.0; batch * HIDDEN], vec![batch, HIDDEN], DType::F32)
        };
        let (mut h1, mut c1) = (zeros(s), zeros(s));
        let (mut h2, mut c2) = (zeros(s), zeros(s));

        let mut loss: Option<Tensor> = None;
        let mut pen: Option<Tensor> = None;
        for t in 0..frames {
            let x = feat.narrow(0, t * batch, batch).reshape([b, 1, 3, 41]);

            // conv0: 3x3 valid window to one channel, 1x1 projection to CH.
            let y = x.conv2d(&c0dw, [3, 3], [1, 1], [0, 0], [1, 1], 1);
            let y = y.conv2d(&c0pw, [1, 1], [1, 1], [0, 0], [1, 1], 1);
            let y = (&y + &c0b.reshape([1, CH as i64, 1, 1])).relu();

            // Length W -> H, then max-pool k=3 s=2.
            let y = y.transpose([0, 1, 3, 2]).max_pool2d([3, 1], [2, 1], [0, 0]);

            // Two separable stages. The padding is asymmetric on the second,
            // which is why it is an explicit pad and not a conv `padding`.
            let y = y.pad(2, 1, 1, 0.0);
            let y = y.conv2d(&s1dw, [3, 1], [2, 1], [0, 0], [1, 1], CH);
            let y = y.conv2d(&s1pw, [1, 1], [1, 1], [0, 0], [1, 1], 1);
            let y = (&y + &s1b.reshape([1, CH as i64, 1, 1])).relu();

            let y = y.pad(2, 0, 1, 0.0);
            let y = y.conv2d(&s2dw, [3, 1], [2, 1], [0, 0], [1, 1], CH);
            let y = y.conv2d(&s2pw, [1, 1], [1, 1], [0, 0], [1, 1], 1);
            let y = (&y + &s2b.reshape([1, CH as i64, 1, 1])).relu();

            // [B,CH,5,1] -> position-major [B,80], as the graph's
            // Squeeze -> Transpose -> Reshape produces.
            let flat = y
                .reshape([b, CH as i64, 5])
                .transpose([0, 2, 1])
                .reshape([b, FLAT as i64]);

            let zc = ranges.map(|r| (r.pre, r.cell));
            if let Some(r) = ranges {
                // The conv stack feeds the first LSTM, and its ReLU output is
                // what makes the partial sums wide: measured at 17.6 in Q15
                // units against activations that are otherwise <= 1.
                let add = |acc: &mut Option<Tensor>, t: Tensor| {
                    *acc = Some(acc.as_ref().map_or(t.clone(), |a| a + &t));
                };
                add(&mut pen, over_budget(&flat, r.act));
            }
            let (nh1, nc1) = lstm_step(s, &flat, &h1, &c1, &l1w, &l1b, &mut pen, zc);
            h1 = nh1;
            c1 = nc1;
            let (nh2, nc2) = lstm_step(s, &h1, &h2, &c2, &l2w, &l2b, &mut pen, zc);
            h2 = nh2;
            c2 = nc2;

            let d = (&s.cat(&[&h2, &h1], 1).matmul(&d1w) + &d1b).relu();
            let p = (&d.matmul(&d2w) + &d2b).sigmoid();

            if t >= warmup {
                let e = &p - &tgt.narrow(0, t * batch, batch);
                let sq = (&(&e * &e) * &wgt.narrow(0, t * batch, batch)).mean_all();
                loss = Some(loss.map_or_else(|| sq.clone(), |acc| &acc + &sq));
            }
        }
        let n = (frames - warmup) as f64;
        let base = &loss.expect("at least one scored frame") * (1.0 / n);
        match (ranges, pen) {
            (Some(r), Some(p)) => &base + &(&p * f64::from(r.lambda / frames as f32)),
            _ => base,
        }
    })
}

/// Budgets for the range-aware term, in the same units the f32 student works
/// in (Q15 activation value 1.0 = 32768 on device).
#[derive(Debug, Clone, Copy)]
struct RangeBudget {
    /// Conv-stack output fed to LSTM 1.
    act: f32,
    /// LSTM gate pre-activations.
    pre: f32,
    /// LSTM cell state.
    cell: f32,
    lambda: f32,
}

// ---- weight plumbing ----------------------------------------------------

/// Concatenate an LSTM's two projections into the `[IN+H, 4H]` matrix the
/// traced student contracts against.
fn concat_gates(ih: &[f32], hh: &[f32], input: usize) -> Vec<f32> {
    let mut out = vec![0.0; (input + HIDDEN) * GATES];
    for r in 0..GATES {
        for i in 0..input {
            out[i * GATES + r] = ih[r * input + i];
        }
        for i in 0..HIDDEN {
            out[(input + i) * GATES + r] = hh[r * HIDDEN + i];
        }
    }
    out
}

fn split_gates(cat: &[f32], input: usize) -> (Vec<f32>, Vec<f32>) {
    let mut ih = vec![0.0; GATES * input];
    let mut hh = vec![0.0; GATES * HIDDEN];
    for r in 0..GATES {
        for i in 0..input {
            ih[r * input + i] = cat[i * GATES + r];
        }
        for i in 0..HIDDEN {
            hh[r * HIDDEN + i] = cat[(input + i) * GATES + r];
        }
    }
    (ih, hh)
}

fn pretrained() -> Vec<(String, Vec<f32>)> {
    let w = embedded_net();
    vec![
        ("c0dw".into(), w.conv0_depthwise.to_vec()),
        ("c0pw".into(), w.conv0_pointwise.to_vec()),
        ("c0b".into(), w.conv0_bias.to_vec()),
        ("s1dw".into(), w.sep1_depthwise.to_vec()),
        ("s1pw".into(), w.sep1_pointwise.to_vec()),
        ("s1b".into(), w.sep1_bias.to_vec()),
        ("s2dw".into(), w.sep2_depthwise.to_vec()),
        ("s2pw".into(), w.sep2_pointwise.to_vec()),
        ("s2b".into(), w.sep2_bias.to_vec()),
        (
            "l1w".into(),
            concat_gates(w.lstm1_weight_ih, w.lstm1_weight_hh, FLAT),
        ),
        ("l1b".into(), w.lstm1_bias.to_vec()),
        (
            "l2w".into(),
            concat_gates(w.lstm2_weight_ih, w.lstm2_weight_hh, HIDDEN),
        ),
        ("l2b".into(), w.lstm2_bias.to_vec()),
        ("d1w".into(), w.dense1_weight.to_vec()),
        ("d1b".into(), w.dense1_bias.to_vec()),
        ("d2w".into(), w.dense2_weight.to_vec()),
        ("d2b".into(), w.dense2_bias.to_vec()),
    ]
}

/// Materialise trained parameters back into the scalar net's layout so the
/// result can be scored by the same code the parity suite uses.
struct Weights {
    v: std::collections::HashMap<String, Vec<f32>>,
    l1ih: Vec<f32>,
    l1hh: Vec<f32>,
    l2ih: Vec<f32>,
    l2hh: Vec<f32>,
}

impl Weights {
    fn new(params: Vec<(String, Vec<f32>)>) -> Self {
        let v: std::collections::HashMap<_, _> = params.into_iter().collect();
        let (l1ih, l1hh) = split_gates(&v["l1w"], FLAT);
        let (l2ih, l2hh) = split_gates(&v["l2w"], HIDDEN);
        Self {
            v,
            l1ih,
            l1hh,
            l2ih,
            l2hh,
        }
    }

    fn net(&self) -> NetWeights<'_> {
        NetWeights {
            conv0_depthwise: &self.v["c0dw"],
            conv0_pointwise: &self.v["c0pw"],
            conv0_bias: &self.v["c0b"],
            sep1_depthwise: &self.v["s1dw"],
            sep1_pointwise: &self.v["s1pw"],
            sep1_bias: &self.v["s1b"],
            sep2_depthwise: &self.v["s2dw"],
            sep2_pointwise: &self.v["s2pw"],
            sep2_bias: &self.v["s2b"],
            lstm1_weight_ih: &self.l1ih,
            lstm1_weight_hh: &self.l1hh,
            lstm1_bias: &self.v["l1b"],
            lstm2_weight_ih: &self.l2ih,
            lstm2_weight_hh: &self.l2hh,
            lstm2_bias: &self.v["l2b"],
            dense1_weight: &self.v["d1w"],
            dense1_bias: &self.v["d1b"],
            dense2_weight: &self.v["d2w"],
            dense2_bias: &self.v["d2b"],
        }
    }
}

/// Apply the keep mask, then the format. Order matters: pruning first means
/// the surviving weights set the block scales, so the format is not wasting
/// range on values that are about to be zeroed.
fn deploy(
    fmt: Format,
    masks: &std::collections::HashMap<String, Vec<bool>>,
    name: &str,
    shape: &[usize],
    w: &mut [f32],
) {
    if let Some(keep) = masks.get(name) {
        for (v, k) in w.iter_mut().zip(keep) {
            if !k {
                *v = 0.0;
            }
        }
    }
    fake_quant(fmt, shape, w);
}

fn quantised(
    fmt: Format,
    masks: &std::collections::HashMap<String, Vec<bool>>,
    params: &[(String, Vec<f32>)],
    shapes: &[(String, Vec<usize>)],
) -> Vec<(String, Vec<f32>)> {
    params
        .iter()
        .map(|(n, v)| {
            let mut v = v.clone();
            let shape = shapes
                .iter()
                .find(|(m, _)| m == n)
                .map_or_else(|| vec![v.len()], |(_, s)| s.clone());
            deploy(fmt, masks, n, &shape, &mut v);
            (n.clone(), v)
        })
        .collect()
}

// ---- evaluation ---------------------------------------------------------

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

struct Score {
    max: f32,
    cos: f64,
    flips: usize,
}

/// Score against the teacher on held-out windows of the training distribution.
///
/// The reference clip is synthetic and only 250 frames; if the student drifts
/// toward LibriSpeech and ESC-50 it would look worse there for a reason that
/// has nothing to do with the quantisation grid. This measures the thing QAT is
/// actually optimising.
fn score_val(w: &Weights, feats: &[f32], starts: &[usize], len: usize) -> Score {
    let mut student = Net::new(w.net());
    let mut teacher = Net::new(embedded_net());
    let (mut got, mut want) = (Vec::new(), Vec::new());
    for &start in starts {
        student.reset();
        teacher.reset();
        let mut stack = vec![0.0f32; STACK];
        for t in 0..len {
            let row = &feats[(start + t) * FEATURE_LEN..(start + t + 1) * FEATURE_LEN];
            stack.copy_within(FEATURE_LEN.., 0);
            stack[(CONTEXT_FRAMES - 1) * FEATURE_LEN..].copy_from_slice(row);
            let (a, b) = (student.forward(&stack), teacher.forward(&stack));
            // Skip the zero-state transient, as training does.
            if t >= len / 4 {
                got.push(a);
                want.push(b);
            }
        }
    }
    let dot: f64 = got
        .iter()
        .zip(&want)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
    let na: f64 = got
        .iter()
        .map(|a| f64::from(*a).powi(2))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = want
        .iter()
        .map(|b| f64::from(*b).powi(2))
        .sum::<f64>()
        .sqrt();
    Score {
        max: got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max),
        cos: dot / (na * nb),
        flips: got
            .iter()
            .zip(&want)
            .filter(|(a, b)| (**a >= 0.5) != (**b >= 0.5))
            .count(),
    }
}

fn score(w: &Weights, rows: &[Vec<f32>], want: &[f32]) -> Score {
    let mut net = Net::new(w.net());
    let mut stack = vec![0.0f32; STACK];
    let got: Vec<f32> = rows
        .iter()
        .map(|row| {
            stack.copy_within(FEATURE_LEN.., 0);
            stack[(CONTEXT_FRAMES - 1) * FEATURE_LEN..].copy_from_slice(row);
            net.forward(&stack)
        })
        .collect();
    let dot: f64 = got
        .iter()
        .zip(want)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
    let na: f64 = got
        .iter()
        .map(|a| f64::from(*a).powi(2))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = want
        .iter()
        .map(|b| f64::from(*b).powi(2))
        .sum::<f64>()
        .sqrt();
    Score {
        max: got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max),
        cos: dot / (na * nb),
        flips: got
            .iter()
            .zip(want)
            .filter(|(a, b)| (**a >= 0.5) != (**b >= 0.5))
            .count(),
    }
}

fn report(label: &str, s: &Score, n: usize) {
    println!(
        "  {label:34} max|Δ|={:.3e}  1-cos={:.3e}  flips={}/{n}",
        s.max,
        1.0 - s.cos,
        s.flips
    );
}

// ---- training -----------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.0 >> 33
    }
}

fn main() -> anyhow::Result<()> {
    let (mut steps, mut batch, mut frames, mut lr) = (400usize, 8usize, 24usize, 3e-4f32);
    let mut eval_every = 100usize;
    let mut warmup_override: Option<usize> = None;
    // Width of the near-threshold band the loss concentrates on. `inf` recovers
    // uniform weighting, which is what this harness did before.
    let mut margin_sigma = f32::INFINITY;
    // Range-aware distillation. Off unless --range-lambda is given.
    let (mut r_act, mut r_pre, mut r_cell, mut r_lambda) = (2.0f32, 16.0f32, 16.0f32, 0.0f32);
    // Window sampling is the only stochastic part of this harness, so varying
    // the seed is how a run's spread gets measured. With ~130 flips out of
    // 2304 the Poisson standard error is ~11, and a single run cannot tell a
    // real 10-flip gain from noise.
    let mut seed = 0xC0FFEEu64;
    let mut probe_warmup = false;
    let mut fmt = Format::Fp4;
    let (mut prune_frac, mut prune_mode) = (0.0f32, "tap".to_string());
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut num = || args.next().and_then(|v| v.parse().ok());
        match a.as_str() {
            "--steps" => steps = num().unwrap_or(steps),
            "--batch" => batch = num().unwrap_or(batch),
            "--frames" => frames = num().unwrap_or(frames),
            "--lr" => lr = args.next().and_then(|v| v.parse().ok()).unwrap_or(lr),
            "--eval-every" => eval_every = num().unwrap_or(eval_every),
            "--warmup" => warmup_override = num(),
            "--seed" => seed = args.next().and_then(|v| v.parse().ok()).unwrap_or(seed),
            "--range-lambda" => {
                r_lambda = args.next().and_then(|v| v.parse().ok()).unwrap_or(r_lambda)
            }
            "--range-act" => r_act = args.next().and_then(|v| v.parse().ok()).unwrap_or(r_act),
            "--range-pre" => r_pre = args.next().and_then(|v| v.parse().ok()).unwrap_or(r_pre),
            "--range-cell" => r_cell = args.next().and_then(|v| v.parse().ok()).unwrap_or(r_cell),
            "--margin" => {
                margin_sigma = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(margin_sigma)
            }
            // Diagnostic: how long a cold-started net needs before it agrees
            // with the continuously-run teacher. That interval is a lower bound
            // on a useful warmup, and the loss is meaningless inside it.
            "--probe-warmup" => probe_warmup = true,
            "--prune" => {
                prune_frac = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(prune_frac)
            }
            "--prune-mode" => prune_mode = args.next().unwrap_or(prune_mode.clone()),
            "--format" => {
                let v = args.next().unwrap_or_default();
                fmt = Format::parse(&v)
                    .unwrap_or_else(|| panic!("unknown --format {v:?} (fp4|int4|int6|int8|none)"));
            }
            _ => {}
        }
    }
    let warmup = warmup_override.unwrap_or(frames / 4);

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let feats = read_f32(&root.join("target/distill/features.f32"));
    let n_frames = feats.len() / FEATURE_LEN;
    anyhow::ensure!(
        n_frames > frames * 4,
        "distillation set is too small ({n_frames} frames) — run dump_distill_set first"
    );
    let train_frames = n_frames * 9 / 10;
    let val_len = 128usize;
    let all_starts: Vec<usize> = (0..48)
        .map(|i| train_frames + i * ((n_frames - train_frames - val_len - 1) / 48))
        .collect();
    // Disjoint: the probe picks the checkpoint, `val_starts` reports it. Sharing
    // windows between the two would let checkpoint selection borrow the score
    // it is later judged by.
    let (probe_starts, val_starts) = {
        let (a, b) = all_starts.split_at(all_starts.len() / 2);
        (a.to_vec(), b.to_vec())
    };
    if probe_warmup {
        // The teacher was dumped by running one `Net` continuously across each
        // clip, so `teacher[k]` carries LSTM state from the clip's first frame.
        // The student trains on windows that start from zero state. This
        // measures the gap that creates: run the *same* f32 net cold from each
        // window start and compare, frame by frame, against the stored
        // continuous output. Any disagreement here is a target the student
        // cannot reach no matter how well it learns.
        let teacher = read_f32(&root.join("target/distill/teacher.f32"));
        let w0 = Weights::new(pretrained());
        let mut agree = vec![0usize; val_len];
        let mut worst = vec![0.0f32; val_len];
        for &st in &val_starts {
            let mut net = Net::new(w0.net());
            let mut stack = vec![0.0f32; STACK];
            for t in 0..val_len {
                let row = &feats[(st + t) * FEATURE_LEN..(st + t + 1) * FEATURE_LEN];
                stack.copy_within(FEATURE_LEN.., 0);
                stack[(CONTEXT_FRAMES - 1) * FEATURE_LEN..].copy_from_slice(row);
                let cold = net.forward(&stack);
                let warm = teacher[st + t];
                if (cold >= 0.5) == (warm >= 0.5) {
                    agree[t] += 1;
                }
                worst[t] = worst[t].max((cold - warm).abs());
            }
        }
        let n = val_starts.len();
        println!(
            "\ncold-start vs continuous teacher, over {n} windows — the target the\n\
             student is asked to match from zero state:\n"
        );
        println!("  frame   decisions agreeing   worst |Δ|");
        for t in 0..val_len {
            if t < 24 || t % 16 == 0 {
                println!("  {t:5}   {:>8}/{n}          {:.3e}", agree[t], worst[t]);
            }
        }
        return Ok(());
    }
    let val_scored = val_starts.len() * (val_len - val_len / 4);
    println!(
        "distillation set: {n_frames} frames ({train_frames} train, {} held out)",
        n_frames - train_frames
    );

    // Reference clip, for scoring only — never trained on.
    let ref_rows: Vec<Vec<f32>> =
        read_f32(&root.join("crates/rlx-ten-vad/tests/fixtures/reference_features.f32"))
            .chunks_exact(FEATURE_LEN)
            .map(<[f32]>::to_vec)
            .collect();
    let published =
        read_f32(&root.join("crates/rlx-ten-vad/tests/fixtures/reference_probs_onnx.f32"));

    let base = pretrained();
    let shapes: Vec<(String, Vec<usize>)> = vec![
        ("c0dw".into(), vec![1, 1, 3, 3]),
        ("c0pw".into(), vec![CH, 1, 1, 1]),
        ("s1dw".into(), vec![CH, 1, 3, 1]),
        ("s1pw".into(), vec![CH, CH, 1, 1]),
        ("s2dw".into(), vec![CH, 1, 3, 1]),
        ("s2pw".into(), vec![CH, CH, 1, 1]),
        ("l1w".into(), vec![FLAT + HIDDEN, GATES]),
        ("l2w".into(), vec![2 * HIDDEN, GATES]),
        ("d1w".into(), vec![2 * HIDDEN, DENSE]),
        ("d2w".into(), vec![DENSE, 1]),
    ];

    let prune = Prune::parse(&prune_mode, prune_frac);
    let masks = build_masks(&base, &shapes, prune);
    if prune != Prune::None {
        let total: usize = base.iter().map(|(_, v)| v.len()).sum();
        let dead: usize = masks.values().flatten().filter(|k| !**k).count();
        let saved = macs_saved(&masks, &shapes);
        println!(
            "\npruning: {} -> {dead} of {total} weights removed ({:.1}%)",
            prune.label(),
            100.0 * dead as f64 / total as f64
        );
        if saved > 0 {
            println!(
                "  {saved} of 79295 MACs per frame removed ({:.1}%) — contiguous, no index list",
                100.0 * saved as f64 / 79_295.0
            );
        }
    }

    let base_w = Weights::new(base.clone());
    let ptq_w = Weights::new(quantised(fmt, &masks, &base, &shapes));
    println!("\nheld-out windows, vs the f32 teacher (what QAT optimises):");
    report(
        "f32 (starting point)",
        &score_val(&base_w, &feats, &val_starts, val_len),
        val_scored,
    );
    report(
        &format!("{}, post-training", fmt.label()),
        &score_val(&ptq_w, &feats, &val_starts, val_len),
        val_scored,
    );
    println!("\n250-frame reference clip, vs the published model:");
    report(
        "f32 (starting point)",
        &score(&base_w, &ref_rows, &published),
        250,
    );
    report(
        &format!("{}, post-training", fmt.label()),
        &score(&ptq_w, &ref_rows, &published),
        250,
    );

    // ---- train ----
    let ranges = (r_lambda > 0.0).then_some(RangeBudget {
        act: r_act,
        pre: r_pre,
        cell: r_cell,
        lambda: r_lambda,
    });
    if let Some(r) = ranges {
        println!(
            "range-aware: |conv|<={:.1} |pre|<={:.1} |cell|<={:.1}, lambda {:.3}",
            r.act, r.pre, r.cell, r.lambda
        );
    }
    let model = build(batch, frames, warmup, ranges);
    let init: std::collections::HashMap<String, Vec<f32>> = base.iter().cloned().collect();
    let mut m = model.init_params(move |name, _dims| {
        init.get(name)
            .unwrap_or_else(|| panic!("no pretrained weight for {name}"))
            .clone()
    });

    // Decoupled weight decay pulls every weight toward zero, which is wrong
    // for a fine-tune that starts from a converged model.
    let mut opt = AdamW::new(lr);
    opt.weight_decay = 0.0;
    let sched = LrSchedule::Cosine {
        base: lr,
        min: lr * 0.05,
        total: steps,
    };
    let mut rng = Rng(seed);
    let mut teacher = Net::new(embedded_net());
    let probe_scored = probe_starts.len() * (val_len - val_len / 4);
    let mut best: Option<Vec<(String, Vec<f32>)>> = None;
    let mut best_flips = usize::MAX;

    println!(
        "\ntraining {}: {steps} steps, batch {batch} x {frames} frames (warmup {warmup}), lr {lr:.0e}",
        fmt.label()
    );
    let t0 = std::time::Instant::now();
    for step in 0..steps {
        // Sample independent windows; targets come from the teacher run over
        // the same window from zero state, matching the student's context.
        let mut fbuf = vec![0.0f32; frames * batch * STACK];
        let mut tbuf = vec![0.0f32; frames * batch];
        let mut wbuf = vec![1.0f32; frames * batch];
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
        // Weight each frame by how close the teacher put it to the decision
        // boundary. Renormalised to mean 1 so the effective learning rate does
        // not move with the weighting.
        if margin_sigma.is_finite() {
            let k = -1.0 / (2.0 * margin_sigma * margin_sigma);
            for (w, &t) in wbuf.iter_mut().zip(&tbuf) {
                let d = t - 0.5;
                *w = (k * d * d).exp();
            }
            let mean = wbuf.iter().sum::<f32>() / wbuf.len() as f32;
            if mean > 0.0 {
                for w in &mut wbuf {
                    *w /= mean;
                }
            }
        }
        let feed: &[(&str, &[f32])] = &[("feat", &fbuf), ("tgt", &tbuf), ("wgt", &wbuf)];
        let (next, loss) = m.train_step_all_at_on_qat_with(
            Device::Cpu,
            &mut opt,
            &sched,
            step,
            1.0,
            |n, sh, w| deploy(fmt, &masks, n, sh, w),
            feed,
        );
        m = next;

        // Judge on held-out data at the quantised point, not on training loss:
        // the loss is measured on random windows and says nothing about whether
        // the FP4 grid is being accommodated.
        if step % eval_every == 0 || step + 1 == steps {
            let now: Vec<(String, Vec<f32>)> = base
                .iter()
                .map(|(n, _)| (n.clone(), m.param_binding(n).expect("bound").to_vec()))
                .collect();
            let qw = Weights::new(quantised(fmt, &masks, &now, &shapes));
            let sc = score_val(&qw, &feats, &probe_starts, val_len);
            let mark = if sc.flips < best_flips {
                "  <- best"
            } else {
                ""
            };
            if sc.flips < best_flips {
                best_flips = sc.flips;
                best = Some(now);
            }
            println!(
                "  step {step:5}  loss {:.6}  held-out flips {:4}/{probe_scored}  1-cos {:.3e}  ({:.0}s){mark}",
                loss[0],
                sc.flips,
                1.0 - sc.cos,
                t0.elapsed().as_secs_f32()
            );
        }
    }

    // ---- evaluate ----
    let final_params: Vec<(String, Vec<f32>)> = base
        .iter()
        .map(|(n, _)| (n.clone(), m.param_binding(n).expect("bound").to_vec()))
        .collect();
    let trained = best.unwrap_or(final_params);
    let trained_w = Weights::new(trained.clone());
    let qat_w = Weights::new(quantised(fmt, &masks, &trained, &shapes));
    // Two rows, and only the second is the model that would ship. The first is
    // the f32 masters with **neither** the prune mask nor the quantisation
    // applied — a diagnostic for whether training drifted, not a result. With
    // `--format none` both used to read "unquantised", which is an easy way to
    // quote the wrong number.
    let masters_label = "f32 masters (diagnostic: no mask, no quant)";
    let deployed_label = if prune_frac > 0.0 {
        format!(
            "-> DEPLOYED: {}, {:.0}% weights pruned",
            fmt.label(),
            prune_frac * 100.0
        )
    } else {
        format!("-> DEPLOYED: {}", fmt.label())
    };
    #[cfg(feature = "range-probe")]
    {
        // The point of the range term is accumulator width, which is not
        // visible in flips. Report it directly.
        let scan = |w: &Weights, label: &str| {
            let mut net = Net::new(w.net());
            let mut stack = vec![0.0f32; STACK];
            for &st in &val_starts {
                net.reset();
                for t in 0..val_len {
                    let row = &feats[(st + t) * FEATURE_LEN..(st + t + 1) * FEATURE_LEN];
                    stack.copy_within(FEATURE_LEN.., 0);
                    stack[(CONTEXT_FRAMES - 1) * FEATURE_LEN..].copy_from_slice(row);
                    std::hint::black_box(net.forward(&stack));
                }
            }
            let r = net.ranges();
            // What actually sets the accumulator width is not |pre| but
            // max_r ||W_r||_1 * max|x|: the largest a partial sum can reach
            // part-way through a gate's dot product. Report it, because it is
            // the quantity a range-aware objective has to move.
            let row_l1 = |m: &[f32], cols: usize| -> f32 {
                m.chunks_exact(cols)
                    .map(|row| row.iter().map(|v| v.abs()).sum::<f32>())
                    .fold(0.0f32, f32::max)
            };
            let l1 = row_l1(&w.v["l1w"], GATES).max(row_l1(&w.v["l2w"], GATES));
            let bound = f64::from(l1) * f64::from(r.max_flat.max(1.0)) * 32768.0 * 32768.0;
            let bits = bound.log2().ceil();
            // Worst-case accumulator bits for the widest gate row, given the
            // observed activation bound: log2(||W||_1 * max|x|) in Q15 units.
            println!(
                "  {label:34} |conv|<={:.2}  |pre|<={:.1}  |cell|<={:.1}  \
                 max row L1 {l1:.1}  -> acc {bits:.0} bits (i32 needs <=31)",
                r.max_flat, r.max_pre, r.max_cell
            );
        };
        println!("\nactivation ranges (what sets the integer accumulator width):");
        scan(&Weights::new(pretrained()), "teacher (starting point)");
        scan(&trained_w, "after QAT");
    }
    println!("\nafter QAT — held-out windows, vs the f32 teacher:");
    report(
        masters_label,
        &score_val(&trained_w, &feats, &val_starts, val_len),
        val_scored,
    );
    report(
        &deployed_label,
        &score_val(&qat_w, &feats, &val_starts, val_len),
        val_scored,
    );
    println!("\nafter QAT — 250-frame reference clip, vs the published model:");
    report(
        masters_label,
        &score(&trained_w, &ref_rows, &published),
        250,
    );
    report(&deployed_label, &score(&qat_w, &ref_rows, &published), 250);
    Ok(())
}
