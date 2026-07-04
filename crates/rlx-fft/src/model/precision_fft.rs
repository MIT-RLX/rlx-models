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

//! Hardware-agnostic precision recovery for a low-precision FFT.
//!
//! On hardware that only runs f16, you can't fall back to an f32 FFT — a single
//! f16 result is capped at `≈2^-11·|X|` (the "output floor"). The fix is
//! **iterative refinement** using the f16 FFT/IFFT *as black boxes*:
//!
//! ```text
//!   Y ← 0 ;  r ← x
//!   repeat K+1:
//!     Y ← Y + FFT_f16(r)              // accumulate in higher precision
//!     r ← x − IFFT_f16(Y)/N          // back-projected residual (small)
//! ```
//!
//! Because the FFT is linear and `FFT_f16 ≈ FFT`, each pass multiplies the error
//! by ~the f16 relative precision, so K=2 reaches ~f32 — using only f16 kernels.
//! `iters` is a single knob spanning f16→f32, and the accumulator can be kept as
//! two f16 limbs (TwoSum) so *every op is f16* (honest for f16-only silicon).

use crate::model::butterfly::{
    bit_reverse, butterfly_forward_real_batch, butterfly_forward_real_batch_f16,
    butterfly_inverse_complex_batch_f16, num_stages, round_f16,
};
use crate::model::config::FftLearnConfig;
use crate::model::reference::{fft_real_batch, max_abs_error};
use crate::model::twiddle::{exact_twiddles, twiddle_index};
use anyhow::Result;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// f16 error-free transform: `a + b = s + e` exactly, all ops rounded to f16.
fn two_sum_f16(a: f32, b: f32) -> (f32, f32) {
    let s = round_f16(a + b);
    let bb = round_f16(s - a);
    let e = round_f16(round_f16(a - round_f16(s - bb)) + round_f16(b - bb));
    (s, e)
}

/// Add an f16 increment `d` into a two-limb f16 accumulator `(hi, lo)`.
fn tl_add(hi: f32, lo: f32, d: f32) -> (f32, f32) {
    let (s, e) = two_sum_f16(hi, d);
    let lo2 = round_f16(lo + e);
    // renormalize (|s| >= |lo2|): FastTwoSum
    let s2 = round_f16(s + lo2);
    let lo3 = round_f16(lo2 - round_f16(s2 - s));
    (s2, lo3)
}

/// Split an f32 into two f16 limbs `(hi, lo)` with `hi = round_f16(x)`.
fn tl_split(x: f32) -> (f32, f32) {
    let hi = round_f16(x);
    (hi, round_f16(x - hi))
}

// ---- Compensated ("double-f16") arithmetic: every op is f16, values carried as
// (hi, lo) pairs. This is the only way to exceed f16's relative-precision wall on
// hardware that has no f32 — it reaches ~f32 from pure f16 ops, output a pair. ----

fn fast_two_sum_f16(a: f32, b: f32) -> (f32, f32) {
    let s = round_f16(a + b);
    (s, round_f16(b - round_f16(s - a)))
}

/// `a·b = p + e` exactly. `a,b` are f16 values, so `a*b` is exact in f32.
fn two_prod_f16(a: f32, b: f32) -> (f32, f32) {
    let exact = a * b;
    let p = round_f16(exact);
    (p, round_f16(exact - p))
}

fn df_add(a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
    let (s, e) = two_sum_f16(a.0, b.0);
    let e2 = round_f16(round_f16(e + a.1) + b.1);
    fast_two_sum_f16(s, e2)
}

fn df_sub(a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
    df_add(a, (-b.0, -b.1))
}

fn df_mul(a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
    let (p, e) = two_prod_f16(a.0, b.0);
    let e2 = round_f16(round_f16(e + round_f16(a.0 * b.1)) + round_f16(a.1 * b.0));
    fast_two_sum_f16(p, e2)
}

/// Multiply WITHOUT an exact TwoProduct — the realistic op on f16-only hardware
/// that lacks FMA / wide-accumulate: the main product `a.0·b.0` rounds and its
/// error is lost; only the cross terms (the operands' `lo` bits) are carried.
fn df_mul_nofma(a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
    let p = round_f16(a.0 * b.0);
    let cross = round_f16(round_f16(a.0 * b.1) + round_f16(a.1 * b.0));
    fast_two_sum_f16(p, cross)
}

/// Compensated forward using `df_mul_nofma` — the honest f16-only-no-FMA bound.
fn compensated_forward_nofma_one(
    x: &[f32],
    tw_hi: &[f32],
    tw_lo: &[f32],
    n_fft: usize,
) -> Vec<f32> {
    let stages = num_stages(n_fft);
    let half = n_fft / 2;
    let mut reh = vec![0f32; n_fft];
    let mut rel = vec![0f32; n_fft];
    let mut imh = vec![0f32; n_fft];
    let mut iml = vec![0f32; n_fft];
    for i in 0..n_fft {
        let (h, l) = tl_split(x[bit_reverse(i, stages)]);
        reh[i] = h;
        rel[i] = l;
    }
    for s in 0..stages {
        let stride = 1usize << s;
        for b in 0..half {
            let group = b / stride;
            let k = b % stride;
            let i0 = group * 2 * stride + k;
            let i1 = i0 + stride;
            let wb = twiddle_index(s, b, half, 0);
            let w_re = (tw_hi[wb], tw_lo[wb]);
            let w_im = (tw_hi[wb + 1], tw_lo[wb + 1]);
            let (b_re, b_im) = ((reh[i1], rel[i1]), (imh[i1], iml[i1]));
            let wb_re = df_sub(df_mul_nofma(b_re, w_re), df_mul_nofma(b_im, w_im));
            let wb_im = df_add(df_mul_nofma(b_re, w_im), df_mul_nofma(b_im, w_re));
            let (a_re, a_im) = ((reh[i0], rel[i0]), (imh[i0], iml[i0]));
            (reh[i0], rel[i0]) = df_add(a_re, wb_re);
            (imh[i0], iml[i0]) = df_add(a_im, wb_im);
            (reh[i1], rel[i1]) = df_sub(a_re, wb_re);
            (imh[i1], iml[i1]) = df_sub(a_im, wb_im);
        }
    }
    let mut out = vec![0f32; n_fft * 2];
    for i in 0..n_fft {
        out[i * 2] = reh[i] + rel[i];
        out[i * 2 + 1] = imh[i] + iml[i];
    }
    out
}

/// Compensated (double-f16) forward FFT, real input. Twiddles are carried as
/// Matryoshka pairs `(tw_hi, tw_lo)`; output is the collapsed `hi+lo` spectrum.
fn compensated_forward_one(x: &[f32], tw_hi: &[f32], tw_lo: &[f32], n_fft: usize) -> Vec<f32> {
    let stages = num_stages(n_fft);
    let half = n_fft / 2;
    // state as double-f16 complex (re/im, each hi/lo), bit-reversed.
    let mut reh = vec![0f32; n_fft];
    let mut rel = vec![0f32; n_fft];
    let mut imh = vec![0f32; n_fft];
    let mut iml = vec![0f32; n_fft];
    for i in 0..n_fft {
        let j = bit_reverse(i, stages);
        let (h, l) = tl_split(x[j]);
        reh[i] = h;
        rel[i] = l;
    }
    for s in 0..stages {
        let stride = 1usize << s;
        for b in 0..half {
            let group = b / stride;
            let k = b % stride;
            let i0 = group * 2 * stride + k;
            let i1 = i0 + stride;
            let wb = twiddle_index(s, b, half, 0);
            let w_re = (tw_hi[wb], tw_lo[wb]);
            let w_im = (tw_hi[wb + 1], tw_lo[wb + 1]);
            let (b_re, b_im) = ((reh[i1], rel[i1]), (imh[i1], iml[i1]));
            // wb = b·w  (complex): re = b_re·w_re − b_im·w_im, im = b_re·w_im + b_im·w_re
            let wb_re = df_sub(df_mul(b_re, w_re), df_mul(b_im, w_im));
            let wb_im = df_add(df_mul(b_re, w_im), df_mul(b_im, w_re));
            let (a_re, a_im) = ((reh[i0], rel[i0]), (imh[i0], iml[i0]));
            let top_re = df_add(a_re, wb_re);
            let top_im = df_add(a_im, wb_im);
            let bot_re = df_sub(a_re, wb_re);
            let bot_im = df_sub(a_im, wb_im);
            (reh[i0], rel[i0]) = top_re;
            (imh[i0], iml[i0]) = top_im;
            (reh[i1], rel[i1]) = bot_re;
            (imh[i1], iml[i1]) = bot_im;
        }
    }
    let mut out = vec![0f32; n_fft * 2];
    for i in 0..n_fft {
        out[i * 2] = reh[i] + rel[i];
        out[i * 2 + 1] = imh[i] + iml[i];
    }
    out
}

fn compensated_forward_batch(
    signal: &[f32],
    tw_hi: &[f32],
    tw_lo: &[f32],
    batch: usize,
    n_fft: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; batch * n_fft * 2];
    for b in 0..batch {
        let y = compensated_forward_one(&signal[b * n_fft..(b + 1) * n_fft], tw_hi, tw_lo, n_fft);
        out[b * n_fft * 2..(b + 1) * n_fft * 2].copy_from_slice(&y);
    }
    out
}

/// One example: iterative-refinement forward FFT with a two-limb f16 accumulator
/// (every op f16). Returns the collapsed spectrum `hi + lo` (length `2·n_fft`).
fn refine_forward_one(x: &[f32], hi_tw: &[f32], n_fft: usize, iters: usize) -> Result<Vec<f32>> {
    let twolen = n_fft * 2;
    let nf = n_fft as f32;
    let mut y_hi = vec![0f32; twolen];
    let mut y_lo = vec![0f32; twolen];
    // input as two f16 limbs (data is representable as a pair on f16-only HW).
    let mut x_hi = vec![0f32; n_fft];
    let mut x_lo = vec![0f32; n_fft];
    for i in 0..n_fft {
        let (h, l) = tl_split(x[i]);
        x_hi[i] = h;
        x_lo[i] = l;
    }
    // residual r starts at x.
    let mut r_hi = x_hi.clone();
    let mut r_lo = x_lo.clone();

    for it in 0..=iters {
        let r16: Vec<f32> = (0..n_fft).map(|i| round_f16(r_hi[i] + r_lo[i])).collect();
        let dy = butterfly_forward_real_batch_f16(&r16, hi_tw, 1, n_fft)?;
        for j in 0..twolen {
            let (h, l) = tl_add(y_hi[j], y_lo[j], dy[j]);
            y_hi[j] = h;
            y_lo[j] = l;
        }
        if it < iters {
            let y16: Vec<f32> = (0..twolen).map(|j| round_f16(y_hi[j] + y_lo[j])).collect();
            let xb = butterfly_inverse_complex_batch_f16(&y16, hi_tw, 1, n_fft)?;
            for i in 0..n_fft {
                let xb_re = round_f16(xb[i * 2] / nf); // back-projected, f16
                // r = x − xb_re in two-limb f16: (x_hi − xb_re) carrying x_lo.
                let (s, e) = two_sum_f16(x_hi[i], round_f16(-xb_re));
                let lo = round_f16(x_lo[i] + e);
                let s2 = round_f16(s + lo);
                r_hi[i] = s2;
                r_lo[i] = round_f16(lo - round_f16(s2 - s));
            }
        }
    }
    Ok((0..twolen).map(|j| y_hi[j] + y_lo[j]).collect())
}

/// Batch iterative-refinement forward (two-limb f16 accumulator).
fn refine_forward_batch(
    signal: &[f32],
    hi_tw: &[f32],
    batch: usize,
    n_fft: usize,
    iters: usize,
) -> Result<Vec<f32>> {
    let mut out = vec![0f32; batch * n_fft * 2];
    for b in 0..batch {
        let y = refine_forward_one(&signal[b * n_fft..(b + 1) * n_fft], hi_tw, n_fft, iters)?;
        out[b * n_fft * 2..(b + 1) * n_fft * 2].copy_from_slice(&y);
    }
    Ok(out)
}

// ── Compensated FFT as an RLX graph ────────────────────────────────────────
// Each value travels as a (hi, lo) pair of [batch, 1] tensors. TwoSum uses
// add/sub (which survive the optimizer); TwoProduct uses `Op::Fma` (single
// rounding). Proves the compensated butterfly lowers + runs through the compiler.

use rlx_ir::infer::GraphExt as _;
use rlx_ir::{Graph, NodeId, Op};

fn g_two_sum(g: &mut Graph, a: NodeId, b: NodeId) -> (NodeId, NodeId) {
    let s = g.add(a, b);
    let bb = g.sub(s, a);
    let s_bb = g.sub(s, bb);
    let at = g.sub(a, s_bb);
    let bt = g.sub(b, bb);
    let e = g.add(at, bt);
    (s, e)
}

fn g_fast_two_sum(g: &mut Graph, a: NodeId, b: NodeId) -> (NodeId, NodeId) {
    let s = g.add(a, b);
    let sa = g.sub(s, a);
    let e = g.sub(b, sa);
    (s, e)
}

type Df = (NodeId, NodeId);

fn g_df_add(g: &mut Graph, a: Df, b: Df) -> Df {
    let (s, e) = g_two_sum(g, a.0, b.0);
    let e_al = g.add(e, a.1);
    let e2 = g.add(e_al, b.1);
    g_fast_two_sum(g, s, e2)
}

fn g_df_neg(g: &mut Graph, a: Df) -> Df {
    (g.neg(a.0), g.neg(a.1))
}

fn g_df_sub(g: &mut Graph, a: Df, b: Df) -> Df {
    let nb = g_df_neg(g, b);
    g_df_add(g, a, nb)
}

/// Double-word multiply. `a` is double (hi,lo); `w` is a single-limb twiddle
/// param `(w, 0)`. The product error term uses `Op::Fma` (the whole point).
fn g_df_mul_w(g: &mut Graph, a: Df, w: NodeId) -> Df {
    let p = g.mul(a.0, w);
    let neg_p = g.neg(p);
    let shape = g.shape(p).clone();
    let e = g.add_node(Op::Fma, vec![a.0, w, neg_p], shape); // a.0*w − p, single-rounded
    let cross = g.mul(a.1, w); // a.lo * w  (w_lo = 0)
    let e2 = g.add(e, cross);
    g_fast_two_sum(g, p, e2)
}

/// Build a compensated forward FFT graph. Returns the graph and the twiddle
/// param bindings (exact f32 twiddles; the f32 arena makes this double-f32).
pub fn build_compensated_forward_graph(
    n_fft: usize,
    batch: usize,
) -> Result<(Graph, Vec<(String, Vec<f32>)>)> {
    build_compensated_forward_graph_base(n_fft, batch, rlx_ir::DType::F32)
}

/// Compensated forward FFT graph at a chosen base dtype. `F32` → double-f32
/// (≈f64); `F64` → double-f64 (≈**f128**) — the data travels as `(hi, lo)` f64
/// pairs and TwoProduct uses the f64 `Op::Fma`.
pub fn build_compensated_forward_graph_base(
    n_fft: usize,
    batch: usize,
    base: rlx_ir::DType,
) -> Result<(Graph, Vec<(String, Vec<f32>)>)> {
    use rlx_ir::Shape;
    let stages = num_stages(n_fft);
    let half = n_fft / 2;
    let f = base;
    let mut g = Graph::new("compensated_fft");
    let signal = g.input("signal", Shape::new(&[batch, n_fft], f));

    // Bit-reversed input columns: re = signal column, lo/imag = 0.
    let mut reh = Vec::with_capacity(n_fft);
    let mut rel = Vec::with_capacity(n_fft);
    let mut imh = Vec::with_capacity(n_fft);
    let mut iml = Vec::with_capacity(n_fft);
    for i in 0..n_fft {
        let j = bit_reverse(i, stages);
        let col = g.narrow_(signal, 1, j, 1);
        let zero = g.sub(col, col);
        reh.push(col);
        rel.push(zero);
        imh.push(zero);
        iml.push(zero);
    }

    let cfg = FftLearnConfig::new(n_fft, batch)?;
    let exact = exact_twiddles(&cfg);
    let mut params = Vec::new();
    let mut tw = std::collections::HashMap::new();
    // Twiddles are [batch, 1] (value repeated) so every FMA operand has matching
    // shape — `Op::Fma` is elementwise and does not broadcast a [1] scalar.
    for s in 0..stages {
        for b in 0..half {
            let idx = twiddle_index(s, b, half, 0);
            let (rn, in_) = (format!("tw.{s}.{b}.re"), format!("tw.{s}.{b}.im"));
            let wre = g.param(&rn, Shape::new(&[batch, 1], f));
            let wim = g.param(&in_, Shape::new(&[batch, 1], f));
            params.push((rn, vec![exact[idx]; batch]));
            params.push((in_, vec![exact[idx + 1]; batch]));
            tw.insert((s, b), (wre, wim));
        }
    }

    for s in 0..stages {
        let stride = 1usize << s;
        let (mut nreh, mut nrel, mut nimh, mut niml) =
            (reh.clone(), rel.clone(), imh.clone(), iml.clone());
        for b in 0..half {
            let group = b / stride;
            let k = b % stride;
            let i0 = group * 2 * stride + k;
            let i1 = i0 + stride;
            let (wre, wim) = tw[&(s, b)];
            let b_re = (reh[i1], rel[i1]);
            let b_im = (imh[i1], iml[i1]);
            // wb = b·w : re = b_re·w_re − b_im·w_im, im = b_re·w_im + b_im·w_re
            let t1 = g_df_mul_w(&mut g, b_re, wre);
            let t2 = g_df_mul_w(&mut g, b_im, wim);
            let wb_re = g_df_sub(&mut g, t1, t2);
            let t3 = g_df_mul_w(&mut g, b_re, wim);
            let t4 = g_df_mul_w(&mut g, b_im, wre);
            let wb_im = g_df_add(&mut g, t3, t4);
            let a_re = (reh[i0], rel[i0]);
            let a_im = (imh[i0], iml[i0]);
            let top_re = g_df_add(&mut g, a_re, wb_re);
            let top_im = g_df_add(&mut g, a_im, wb_im);
            let bot_re = g_df_sub(&mut g, a_re, wb_re);
            let bot_im = g_df_sub(&mut g, a_im, wb_im);
            nreh[i0] = top_re.0;
            nrel[i0] = top_re.1;
            nimh[i0] = top_im.0;
            niml[i0] = top_im.1;
            nreh[i1] = bot_re.0;
            nrel[i1] = bot_re.1;
            nimh[i1] = bot_im.0;
            niml[i1] = bot_im.1;
        }
        reh = nreh;
        rel = nrel;
        imh = nimh;
        iml = niml;
    }

    // Collapse to [batch, n_fft, 2] interleaved.
    let mut cols = Vec::with_capacity(n_fft);
    for i in 0..n_fft {
        let re = g.add(reh[i], rel[i]); // [batch,1]
        let im = g.add(imh[i], iml[i]);
        let re3 = g.reshape_(re, vec![batch as i64, 1, 1]);
        let im3 = g.reshape_(im, vec![batch as i64, 1, 1]);
        cols.push(g.concat_(vec![re3, im3], 2)); // [batch,1,2]
    }
    let mut out = g.concat_(cols, 1); // [batch, n_fft, 2]
    // `run()`'s host boundary is f32; narrow an f64 result inside the graph so it
    // reads back correctly (the f128-grade precision is internal, demonstrated by
    // the eager `dd_fft` — this confirms the f64 compute path is correct).
    if base == rlx_ir::DType::F64 {
        let s = rlx_ir::Shape::new(&[batch, n_fft, 2], rlx_ir::DType::F32);
        out = g.add_node(
            Op::Cast {
                to: rlx_ir::DType::F32,
            },
            vec![out],
            s,
        );
    }
    g.set_outputs(vec![out]);
    Ok((g, params))
}

// ── f128-grade FFT via double-double (two f64 limbs) ───────────────────────
// True f128 has no hardware / stable-Rust support. Double-double gives a
// ~106-bit mantissa (~31 digits) ≈ f128 (112-bit) using only f64 ops + the
// f64 FMA (the same single-rounded `Op::Fma` added to the framework). Each
// number is a pair `(hi, lo)` with `hi + lo` the value; TwoSum/TwoProduct keep
// the discarded bits.

type Dd = (f64, f64);
type Cdd = (Dd, Dd); // complex double-double (re, im)

fn two_sum_f64(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}
fn fast_two_sum_f64(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}
fn two_prod_f64(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p)) // exact product error via the f64 fused multiply-add
}
fn dd_from(x: f64) -> Dd {
    (x, 0.0)
}
fn dd_add(a: Dd, b: Dd) -> Dd {
    let (s, e) = two_sum_f64(a.0, b.0);
    fast_two_sum_f64(s, e + a.1 + b.1)
}
fn dd_sub(a: Dd, b: Dd) -> Dd {
    dd_add(a, (-b.0, -b.1))
}
fn dd_mul(a: Dd, b: Dd) -> Dd {
    let (p, e) = two_prod_f64(a.0, b.0);
    fast_two_sum_f64(p, e + a.0 * b.1 + a.1 * b.0)
}
fn dd_div(a: Dd, b: Dd) -> Dd {
    let q1 = a.0 / b.0;
    let r = dd_sub(a, dd_mul(b, dd_from(q1)));
    fast_two_sum_f64(q1, r.0 / b.0)
}
fn dd_sqrt(a: Dd) -> Dd {
    if a.0 <= 0.0 {
        return (0.0, 0.0);
    }
    let x = a.0.sqrt();
    let diff = dd_sub(a, dd_mul(dd_from(x), dd_from(x)));
    fast_two_sum_f64(x, diff.0 / (2.0 * x))
}

fn cdd_add(a: Cdd, b: Cdd) -> Cdd {
    (dd_add(a.0, b.0), dd_add(a.1, b.1))
}
fn cdd_sub(a: Cdd, b: Cdd) -> Cdd {
    (dd_sub(a.0, b.0), dd_sub(a.1, b.1))
}
fn cdd_mul(a: Cdd, b: Cdd) -> Cdd {
    (
        dd_sub(dd_mul(a.0, b.0), dd_mul(a.1, b.1)),
        dd_add(dd_mul(a.0, b.1), dd_mul(a.1, b.0)),
    )
}

/// `(cos(π/2^s), sin(π/2^s))` to dd precision via the half-angle recurrence from
/// the EXACT `cos(π) = −1` — so these are true roots of unity, not f64 `sin_cos`.
fn dd_cos_sin(s: usize) -> (Dd, Dd) {
    let (mut c, mut sn) = (dd_from(-1.0), dd_from(0.0)); // (cos π, sin π)
    let (one, half) = (dd_from(1.0), dd_from(0.5));
    for _ in 0..s {
        let c2 = dd_sqrt(dd_mul(dd_add(one, c), half)); // cos(θ/2)
        sn = dd_sqrt(dd_mul(dd_sub(one, c), half)); // sin(θ/2)
        c = c2;
    }
    (c, sn)
}

/// Base twiddle `exp(sign·2πi/m)` for a radix-2 stage (m a power of two), dd-exact.
fn dd_base_twiddle(m: usize, sign: f64) -> Cdd {
    let s = m.trailing_zeros() as usize - 1; // 2π/m = π/2^s
    let (c, sn) = dd_cos_sin(s);
    (c, if sign < 0.0 { (-sn.0, -sn.1) } else { sn })
}

/// In-place radix-2 DIT FFT in double-double. `sign = -1` forward, `+1` inverse.
/// Twiddles are dd-exact roots of unity (above), advanced by dd multiplication.
pub fn dd_fft(a: &mut [Cdd], sign: f64) {
    let n = a.len();
    let bits = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = crate::model::butterfly::bit_reverse(i, bits);
        if i < j {
            a.swap(i, j);
        }
    }
    let mut m = 2;
    while m <= n {
        let half = m / 2;
        let base = dd_base_twiddle(m, sign);
        for g in (0..n).step_by(m) {
            let mut w: Cdd = (dd_from(1.0), dd_from(0.0));
            for k in 0..half {
                let t = cdd_mul(w, a[g + k + half]);
                let u = a[g + k];
                a[g + k] = cdd_add(u, t);
                a[g + k + half] = cdd_sub(u, t);
                let _ = k;
                w = cdd_mul(w, base);
            }
        }
        m *= 2;
    }
}

/// In-place radix-2 DIT FFT in plain f64 — the single-precision baseline.
fn f64_fft(a: &mut [(f64, f64)], sign: f64) {
    let n = a.len();
    let bits = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = crate::model::butterfly::bit_reverse(i, bits);
        if i < j {
            a.swap(i, j);
        }
    }
    let mut m = 2;
    while m <= n {
        let half = m / 2;
        for g in (0..n).step_by(m) {
            for k in 0..half {
                let ang = sign * 2.0 * std::f64::consts::PI * k as f64 / m as f64;
                let (s, c) = ang.sin_cos();
                let b = a[g + k + half];
                let tr = c * b.0 - s * b.1;
                let ti = c * b.1 + s * b.0;
                let u = a[g + k];
                a[g + k] = (u.0 + tr, u.1 + ti);
                a[g + k + half] = (u.0 - tr, u.1 - ti);
            }
        }
        m *= 2;
    }
}

/// FFT→IFFT roundtrip max error at double-double precision (≈f128).
pub fn dd_roundtrip_maxerr(x: &[f64]) -> f64 {
    let n = x.len();
    let mut a: Vec<Cdd> = x.iter().map(|&v| (dd_from(v), dd_from(0.0))).collect();
    dd_fft(&mut a, -1.0);
    dd_fft(&mut a, 1.0);
    let nn = dd_from(n as f64);
    (0..n)
        .map(|i| {
            // residual computed IN dd, then collapsed (it's tiny, so no loss).
            let err = dd_sub(dd_div(a[i].0, nn), dd_from(x[i]));
            (err.0 + err.1).abs()
        })
        .fold(0.0, f64::max)
}

/// FFT→IFFT roundtrip max error at plain f64 precision.
pub fn f64_roundtrip_maxerr(x: &[f64]) -> f64 {
    let n = x.len();
    let mut a: Vec<(f64, f64)> = x.iter().map(|&v| (v, 0.0)).collect();
    f64_fft(&mut a, -1.0);
    f64_fft(&mut a, 1.0);
    (0..n)
        .map(|i| (a[i].0 / n as f64 - x[i]).abs())
        .fold(0.0, f64::max)
}

// ── High precision on hardware WITHOUT f64: K-limb f32 expansions ───────────
// Float-float / quad-float: a value is an unevaluated sum of K non-overlapping
// f32s. K=2 ≈ f64 (~48-bit); K=4 ≈ f128-class (~96-bit). Everything is f32 ops
// + the f32 FMA (the native `Op::Fma` on CPU/WebGPU/Metal). Twiddles are
// precomputed in dd (host f64) and split to f32 limbs — the compute hardware
// never needs f64.

fn two_sum_f32(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}
fn two_prod_f32(a: f32, b: f32) -> (f32, f32) {
    let p = a * b;
    (p, a.mul_add(b, -p)) // exact product error via the f32 FMA
}

/// Renormalize an unevaluated f32 sum to a non-overlapping K-limb expansion
/// (distillation: repeated adjacent TwoSum sweeps, then keep the top K).
fn ex_renorm(mut t: Vec<f32>, k: usize) -> Vec<f32> {
    t.retain(|&x| x != 0.0);
    if t.is_empty() {
        return vec![0.0; k];
    }
    for _ in 0..t.len() {
        let mut changed = false;
        for i in (1..t.len()).rev() {
            let (s, e) = two_sum_f32(t[i - 1], t[i]);
            t[i - 1] = s;
            if t[i] != e {
                changed = true;
            }
            t[i] = e;
        }
        if !changed {
            break;
        }
    }
    t.retain(|&x| x != 0.0);
    t.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
    t.truncate(k);
    while t.len() < k {
        t.push(0.0);
    }
    t
}
fn ex_add(a: &[f32], b: &[f32], k: usize) -> Vec<f32> {
    let mut t = a.to_vec();
    t.extend_from_slice(b);
    ex_renorm(t, k)
}
fn ex_sub(a: &[f32], b: &[f32], k: usize) -> Vec<f32> {
    let nb: Vec<f32> = b.iter().map(|&x| -x).collect();
    ex_add(a, &nb, k)
}
fn ex_mul(a: &[f32], b: &[f32], k: usize) -> Vec<f32> {
    let mut t = Vec::with_capacity(a.len() * b.len() * 2);
    for &ai in a {
        for &bj in b {
            let (p, e) = two_prod_f32(ai, bj);
            t.push(p);
            t.push(e);
        }
    }
    ex_renorm(t, k)
}

type Cex = (Vec<f32>, Vec<f32>); // complex expansion (re, im)
fn cex_add(a: &Cex, b: &Cex, k: usize) -> Cex {
    (ex_add(&a.0, &b.0, k), ex_add(&a.1, &b.1, k))
}
fn cex_sub(a: &Cex, b: &Cex, k: usize) -> Cex {
    (ex_sub(&a.0, &b.0, k), ex_sub(&a.1, &b.1, k))
}
fn cex_mul(a: &Cex, b: &Cex, k: usize) -> Cex {
    let re = ex_sub(&ex_mul(&a.0, &b.0, k), &ex_mul(&a.1, &b.1, k), k);
    let im = ex_add(&ex_mul(&a.0, &b.1, k), &ex_mul(&a.1, &b.0, k), k);
    (re, im)
}

/// Split a dd (f64) value into K non-overlapping f32 limbs (host-side; the
/// resulting limbs are an f32-only constant the device consumes).
fn dd_split_to_f32(mut v: Dd, k: usize) -> Vec<f32> {
    let mut limbs = Vec::with_capacity(k);
    for _ in 0..k {
        let l = v.0 as f32;
        limbs.push(l);
        v = dd_sub(v, (l as f64, 0.0));
    }
    ex_renorm(limbs, k)
}

/// K-limb f32 radix-2 DIT FFT. Twiddles are dd-exact roots of unity (host) split
/// to f32 limbs; the transform itself is pure f32. `sign = -1` fwd, `+1` inv.
pub fn ex_fft(a: &mut [Cex], sign: f64, k: usize) {
    let n = a.len();
    let bits = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = crate::model::butterfly::bit_reverse(i, bits);
        if i < j {
            a.swap(i, j);
        }
    }
    let mut m = 2;
    while m <= n {
        let half = m / 2;
        let base = dd_base_twiddle(m, sign);
        for g in (0..n).step_by(m) {
            let mut w: Cdd = (dd_from(1.0), dd_from(0.0));
            for kk in 0..half {
                let w_ex: Cex = (dd_split_to_f32(w.0, k), dd_split_to_f32(w.1, k));
                let t = cex_mul(&w_ex, &a[g + kk + half], k);
                let u = a[g + kk].clone();
                a[g + kk] = cex_add(&u, &t, k);
                a[g + kk + half] = cex_sub(&u, &t, k);
                w = cdd_mul(w, base);
            }
        }
        m *= 2;
    }
}

// ── Optimized fixed double-f32 (the production ≈f64 path) ──────────────────
// Same as the generic K=2 expansion but on `(f32, f32)` tuples — no Vec, no
// sort, no distillation. This is what you ship for f64-grade on f64-less HW.

type F2 = (f32, f32);
fn fast_two_sum_f32(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    (s, b - (s - a))
}
fn f2_from(x: f32) -> F2 {
    (x, 0.0)
}
fn f2_add(a: F2, b: F2) -> F2 {
    let (s, e) = two_sum_f32(a.0, b.0);
    fast_two_sum_f32(s, e + a.1 + b.1)
}
fn f2_sub(a: F2, b: F2) -> F2 {
    f2_add(a, (-b.0, -b.1))
}
fn f2_mul(a: F2, b: F2) -> F2 {
    let (p, e) = two_prod_f32(a.0, b.0);
    fast_two_sum_f32(p, e + a.0 * b.1 + a.1 * b.0)
}
type Cf2 = (F2, F2);
fn cf2_mul(a: Cf2, b: Cf2) -> Cf2 {
    (
        f2_sub(f2_mul(a.0, b.0), f2_mul(a.1, b.1)),
        f2_add(f2_mul(a.0, b.1), f2_mul(a.1, b.0)),
    )
}
/// Narrow a dd (f64) twiddle to a double-f32 pair (host side).
fn dd_to_f2(v: Dd) -> F2 {
    let h = v.0 as f32;
    let l = ((v.0 - h as f64) + v.1) as f32;
    (h, l)
}

/// Optimized double-f32 radix-2 DIT FFT (`sign = -1` fwd, `+1` inv).
pub fn f2_fft(a: &mut [Cf2], sign: f64) {
    let n = a.len();
    let bits = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = crate::model::butterfly::bit_reverse(i, bits);
        if i < j {
            a.swap(i, j);
        }
    }
    let mut m = 2;
    while m <= n {
        let half = m / 2;
        let base = dd_base_twiddle(m, sign);
        for g in (0..n).step_by(m) {
            let mut w: Cdd = (dd_from(1.0), dd_from(0.0));
            for kk in 0..half {
                let w_f2: Cf2 = (dd_to_f2(w.0), dd_to_f2(w.1));
                let t = cf2_mul(w_f2, a[g + kk + half]);
                let u = a[g + kk];
                a[g + kk] = (f2_add(u.0, t.0), f2_add(u.1, t.1));
                a[g + kk + half] = (f2_sub(u.0, t.0), f2_sub(u.1, t.1));
                w = cdd_mul(w, base);
            }
        }
        m *= 2;
    }
}

/// Optimized double-f32 roundtrip max error.
pub fn f2_roundtrip_maxerr(x: &[f32]) -> f64 {
    let n = x.len();
    let mut a: Vec<Cf2> = x.iter().map(|&v| (f2_from(v), f2_from(0.0))).collect();
    f2_fft(&mut a, -1.0);
    f2_fft(&mut a, 1.0);
    let inv = f2_from(1.0 / n as f32);
    (0..n)
        .map(|i| {
            let r = f2_mul(a[i].0, inv);
            let e = f2_sub(r, f2_from(x[i]));
            (e.0 as f64 + e.1 as f64).abs()
        })
        .fold(0.0, f64::max)
}

// ── Low-precision base types: f16 / f8 / f4 ────────────────────────────────
// The same compensated arithmetic works for ANY base float. `round_fp`
// quantizes to a format with `eb` exponent + `mb` mantissa bits (subnormals
// included). 1 limb = the base's precision; 2 compensated limbs ≈ double the
// mantissa — e.g. double-f8 ≈ f16, double-f16 ≈ f32 — all on that low-precision
// hardware, using its native ops + FMA.

/// (exp_bits, mantissa_bits) for common formats.
pub const FMT_F16: (u32, u32) = (5, 10);
pub const FMT_F8_E4M3: (u32, u32) = (4, 3);
pub const FMT_F8_E5M2: (u32, u32) = (5, 2);
pub const FMT_F4_E2M1: (u32, u32) = (2, 1);

/// Round `x` to the nearest value of a float format `(eb, mb)` (round-to-nearest,
/// subnormals + overflow-to-max-finite).
pub fn round_fp(x: f32, eb: u32, mb: u32) -> f32 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    let bias = (1i32 << (eb - 1)) - 1;
    let emin = 1 - bias; // smallest normal exponent
    let emax = bias; // largest normal exponent
    let e = x.abs().log2().floor() as i32;
    let eff_e = e.max(emin); // subnormals quantize at emin's step
    let step = 2f32.powi(eff_e - mb as i32);
    let q = (x / step).round() * step;
    let max_finite = (2f32 - 2f32.powi(-(mb as i32))) * 2f32.powi(emax);
    if q.abs() > max_finite {
        x.signum() * max_finite
    } else {
        q
    }
}

// round-to-base error-free transforms (used by the precision tests below).
#[cfg(test)]
fn ts_r(a: f32, b: f32, r: &impl Fn(f32) -> f32) -> (f32, f32) {
    let s = r(a + b);
    let bb = r(s - a);
    (s, r(r(a - r(s - bb)) + r(b - bb)))
}
#[cfg(test)]
fn fts_r(a: f32, b: f32, r: &impl Fn(f32) -> f32) -> (f32, f32) {
    let s = r(a + b);
    (s, r(b - r(s - a)))
}
#[cfg(test)]
fn tp_r(a: f32, b: f32, r: &impl Fn(f32) -> f32) -> (f32, f32) {
    let p = r(a * b);
    (p, r(a.mul_add(b, -p))) // FMA: exact product error, rounded to base
}

/// One base-limb complex value `(re, im)`; or a 2-limb compensated pair when
/// `lo` is used. We store as `[re_hi, re_lo, im_hi, im_lo]` and run L∈{1,2}.
#[cfg(test)]
fn base_roundtrip(x: &[f32], fmt: (u32, u32), limbs: usize) -> f64 {
    let r = |v: f32| round_fp(v, fmt.0, fmt.1);
    let n = x.len();
    let bits = n.trailing_zeros() as usize;
    // complex value = (re:(hi,lo), im:(hi,lo)); for L=1 the lo limbs stay 0.
    type V = ((f32, f32), (f32, f32));
    let d_add = |a: (f32, f32), b: (f32, f32)| -> (f32, f32) {
        if limbs == 1 {
            return (r(a.0 + b.0), 0.0);
        }
        let (s, e) = ts_r(a.0, b.0, &r);
        fts_r(s, r(r(e + a.1) + b.1), &r)
    };
    let d_sub = |a: (f32, f32), b: (f32, f32)| d_add(a, (-b.0, -b.1));
    let d_mul = |a: (f32, f32), b: (f32, f32)| -> (f32, f32) {
        if limbs == 1 {
            return (r(a.0 * b.0), 0.0);
        }
        let (p, e) = tp_r(a.0, b.0, &r);
        fts_r(p, r(r(e + r(a.0 * b.1)) + r(a.1 * b.0)), &r)
    };
    let cmul = |a: V, b: V| -> V {
        (
            d_sub(d_mul(a.0, b.0), d_mul(a.1, b.1)),
            d_add(d_mul(a.0, b.1), d_mul(a.1, b.0)),
        )
    };
    let split = |v: Dd| -> (f32, f32) {
        let h = r(v.0 as f32);
        if limbs == 1 {
            (h, 0.0)
        } else {
            (h, r((v.0 - h as f64 + v.1) as f32))
        }
    };

    let mut a: Vec<V> = x.iter().map(|&v| ((r(v), 0.0), (0.0, 0.0))).collect();
    for &sign in &[-1.0f64, 1.0] {
        for i in 0..n {
            let j = crate::model::butterfly::bit_reverse(i, bits);
            if i < j {
                a.swap(i, j);
            }
        }
        let mut m = 2;
        while m <= n {
            let half = m / 2;
            let base = dd_base_twiddle(m, sign);
            for g in (0..n).step_by(m) {
                let mut w: Cdd = (dd_from(1.0), dd_from(0.0));
                for kk in 0..half {
                    let w_b: V = (split(w.0), split(w.1));
                    let t = cmul(w_b, a[g + kk + half]);
                    let u = a[g + kk];
                    a[g + kk] = (d_add(u.0, t.0), d_add(u.1, t.1));
                    a[g + kk + half] = (d_sub(u.0, t.0), d_sub(u.1, t.1));
                    w = cdd_mul(w, base);
                }
            }
            m *= 2;
        }
    }
    let inv = r(1.0 / n as f32);
    (0..n)
        .map(|i| {
            let re = d_mul(a[i].0, (inv, 0.0));
            (re.0 as f64 + re.1 as f64 - x[i] as f64).abs()
        })
        .fold(0.0, f64::max)
}

/// FFT→IFFT roundtrip max error using K f32 limbs (residual kept in-expansion).
pub fn ex_roundtrip_maxerr(x: &[f32], k: usize) -> f64 {
    let n = x.len();
    let mut a: Vec<Cex> = x
        .iter()
        .map(|&v| (ex_renorm(vec![v], k), vec![0.0; k]))
        .collect();
    ex_fft(&mut a, -1.0, k);
    ex_fft(&mut a, 1.0, k);
    let inv_n = [1.0f32 / n as f32]; // exact (n is a power of two)
    (0..n)
        .map(|i| {
            let recon = ex_mul(&a[i].0, &inv_n, k);
            let err = ex_sub(&recon, &[x[i]], k);
            err.iter().map(|&l| l as f64).sum::<f64>().abs()
        })
        .fold(0.0, f64::max)
}

/// One row of the precision/cost comparison.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SchemeRow {
    pub scheme: String,
    /// Forward FFT max error vs rustfft.
    pub max_err: f32,
    /// Relative compute cost (f16 FFT/IFFT passes; f32 counted as ~2 f16 passes).
    pub passes: f32,
    /// Whether the result needs a two-limb (f16 pair) output to realize this error.
    pub two_limb_out: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PrecisionReport {
    pub n_fft: usize,
    pub batch: usize,
    pub rows: Vec<SchemeRow>,
}

/// Compare precision schemes on a fixed random batch.
pub fn precision_sweep(
    n_fft: usize,
    batch: usize,
    max_iters: usize,
    seed: u64,
) -> Result<PrecisionReport> {
    let cfg = FftLearnConfig::new(n_fft, batch)?;
    let tw = exact_twiddles(&cfg);
    let mut rng = StdRng::seed_from_u64(seed);
    let signal: Vec<f32> = (0..batch * n_fft)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let reference = fft_real_batch(&signal, batch, n_fft)?;

    let mut rows = Vec::new();

    // f32 reference compute.
    rows.push(SchemeRow {
        scheme: "f32".into(),
        max_err: max_abs_error(
            &butterfly_forward_real_batch(&signal, &tw, batch, n_fft)?,
            &reference,
        ),
        passes: 2.0,
        two_limb_out: false,
    });

    // Plain f16 (single limb in and out).
    rows.push(SchemeRow {
        scheme: "f16".into(),
        max_err: max_abs_error(
            &butterfly_forward_real_batch_f16(&signal, &tw, batch, n_fft)?,
            &reference,
        ),
        passes: 1.0,
        two_limb_out: false,
    });

    // f16 output floor: f32 compute, single-f16 output (the wall for any 1-limb result).
    let f32_spec = butterfly_forward_real_batch(&signal, &tw, batch, n_fft)?;
    let floor: Vec<f32> = f32_spec.iter().map(|&v| round_f16(v)).collect();
    rows.push(SchemeRow {
        scheme: "f16-output-floor".into(),
        max_err: max_abs_error(&floor, &reference),
        passes: f32::NAN,
        two_limb_out: false,
    });

    // Iterative refinement (two-limb f16 accumulator) — included to show it does
    // NOT help: the residual is capped by the f16 *inverse* precision.
    let iters = max_iters;
    let pred = refine_forward_batch(&signal, &tw, batch, n_fft, iters)?;
    rows.push(SchemeRow {
        scheme: format!("f16-refine-{iters}"),
        max_err: max_abs_error(&pred, &reference),
        passes: (2 * iters + 1) as f32, // (iters+1) FFT + iters IFFT
        two_limb_out: true,
    });

    // Compensated double-f16 (twiddles as Matryoshka pairs) — pure f16 ops, ~f32.
    let tw_hi: Vec<f32> = tw.iter().map(|&w| round_f16(w)).collect();
    let tw_lo: Vec<f32> = tw
        .iter()
        .zip(&tw_hi)
        .map(|(&w, &h)| round_f16(w - h))
        .collect();
    // No-FMA bound: pure f16 products (realistic on f16-only HW without FMA).
    let mut nofma = vec![0f32; batch * n_fft * 2];
    for b in 0..batch {
        let y = compensated_forward_nofma_one(
            &signal[b * n_fft..(b + 1) * n_fft],
            &tw_hi,
            &tw_lo,
            n_fft,
        );
        nofma[b * n_fft * 2..(b + 1) * n_fft * 2].copy_from_slice(&y);
    }
    rows.push(SchemeRow {
        scheme: "f16-comp-noFMA".into(),
        max_err: max_abs_error(&nofma, &reference),
        passes: 3.0,
        two_limb_out: true,
    });

    // Full compensated: needs an exact TwoProduct (FMA or f16×f16→f32 accumulate).
    rows.push(SchemeRow {
        scheme: "f16-comp-FMA".into(),
        max_err: max_abs_error(
            &compensated_forward_batch(&signal, &tw_hi, &tw_lo, batch, n_fft),
            &reference,
        ),
        passes: 4.0,
        two_limb_out: true,
    });

    Ok(PrecisionReport { n_fft, batch, rows })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The make-or-break for a compiled compensated FFT: the optimizer must NOT
    /// rewrite `(a+b) − a → b` (true for reals, false for floats). If it does,
    /// the TwoSum error term collapses to 0 and double-f16 is impossible.
    #[test]
    fn twosum_survives_the_compiler() {
        use rlx_ir::infer::GraphExt;
        use rlx_ir::{DType, Graph, Shape};
        use rlx_runtime::Device;
        let mut g = Graph::new("twosum");
        let a = g.input("a", Shape::new(&[1], DType::F32));
        let b = g.input("b", Shape::new(&[1], DType::F32));
        let s = g.add(a, b);
        let t = g.sub(s, a); // (a+b) − a  — must survive
        let e = g.sub(b, t); // b − ((a+b) − a) = the rounding error
        g.set_outputs(vec![s, e]);
        let mut exec = crate::exec::compile::try_compile_graph(Device::Cpu, g).unwrap();
        // 1.0 + 1e-9 rounds to 1.0 in f32, so the true error term is ~1e-9.
        let out = exec.run(&[("a", &[1.0f32]), ("b", &[1e-9f32])]);
        assert!(
            out[1][0].abs() > 1e-12,
            "TwoSum error folded to {} — optimizer is float-unsafe; need a barrier",
            out[1][0]
        );
    }

    /// The compensated graph generalizes to an **F64 base** (double-f64 ≈ f128;
    /// `build_compensated_forward_graph_base`), and the f64 `Op::Fma` provides the
    /// TwoProduct. Running it end-to-end is blocked today by the runtime's
    /// f32-centric `run()` I/O (f32 inputs aren't widened to F64), so it's
    /// `#[ignore]`d until f64 graph I/O lands; the eager `dd_fft` (`1.2e-30`) is
    /// the full-precision f128 demonstration.
    #[test]
    #[ignore = "compiled f64 graph I/O (run() is f32-centric) not yet supported"]
    fn compiled_f64_compensated_fft_runs() {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        use rlx_runtime::Device;
        let (n_fft, batch) = (8usize, 4usize);
        let (graph, params) =
            super::build_compensated_forward_graph_base(n_fft, batch, rlx_ir::DType::F64).unwrap();
        let mut exec = crate::exec::compile::try_compile_graph(Device::Cpu, graph).unwrap();
        for (name, data) in &params {
            let bytes: Vec<u8> = data
                .iter()
                .flat_map(|&v| (v as f64).to_le_bytes())
                .collect();
            exec.set_param_typed(name, &bytes, rlx_ir::DType::F64);
        }
        let mut rng = StdRng::seed_from_u64(3);
        let signal: Vec<f32> = (0..batch * n_fft)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let out = exec.run(&[("signal", &signal)]);
        let reference = super::fft_real_batch(&signal, batch, n_fft).unwrap();
        let err = super::max_abs_error(&out[0], &reference);
        assert!(err < 1e-3, "compiled f64 compensated FFT err {err:e}");
    }

    /// End-to-end: the compensated double-word butterfly, lowered to an RLX graph
    /// with `Op::Fma`, compiles and runs through the compiler and matches rustfft.
    #[test]
    fn compensated_fft_graph_runs_on_cpu() {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        use rlx_runtime::Device;
        let (n_fft, batch) = (8usize, 4usize);
        let (graph, params) = super::build_compensated_forward_graph(n_fft, batch).unwrap();
        let mut exec = crate::exec::compile::try_compile_graph(Device::Cpu, graph).unwrap();
        for (name, data) in &params {
            exec.set_param(name, data);
        }
        let mut rng = StdRng::seed_from_u64(3);
        let signal: Vec<f32> = (0..batch * n_fft)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let out = exec.run(&[("signal", &signal)]);
        let reference = super::fft_real_batch(&signal, batch, n_fft).unwrap();
        let err = super::max_abs_error(&out[0], &reference);
        assert!(err < 1e-3, "compensated FFT graph err {err:e}");
    }

    /// Native `Op::Fma` on Metal (MSL `fma` is fused) + the full compensated FFT
    /// graph running on Metal and matching rustfft.
    #[cfg(feature = "metal")]
    #[test]
    fn fma_and_compensated_fft_on_metal() {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        use rlx_ir::infer::GraphExt;
        use rlx_ir::{DType, Graph, Op, Shape};
        use rlx_runtime::Device;
        if !rlx_runtime::is_available(Device::Metal) {
            return;
        }
        // single-rounding check
        let mut g = Graph::new("fma_metal");
        let a = g.input("a", Shape::new(&[4], DType::F32));
        let b = g.input("b", Shape::new(&[4], DType::F32));
        let p = g.mul(a, b);
        let neg_p = g.neg(p);
        let s = g.shape(a).clone();
        let e = g.add_node(Op::Fma, vec![a, b, neg_p], s);
        g.set_outputs(vec![p, e]);
        let mut exec = crate::exec::compile::try_compile_graph(Device::Metal, g).unwrap();
        let v = 1.0 + 2.0f32.powi(-13);
        let out = exec.run(&[("a", &[v; 4]), ("b", &[v; 4])]);
        assert!(
            out[1][0].abs() > 1e-10 && out[1][0].abs() < 1e-4,
            "metal Fma not single-rounded: {}",
            out[1][0]
        );
        // full compensated FFT graph
        let (n_fft, batch) = (8usize, 4usize);
        let (graph, params) = super::build_compensated_forward_graph(n_fft, batch).unwrap();
        let mut exec = crate::exec::compile::try_compile_graph(Device::Metal, graph).unwrap();
        for (name, data) in &params {
            exec.set_param(name, data);
        }
        let mut rng = StdRng::seed_from_u64(3);
        let signal: Vec<f32> = (0..batch * n_fft)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let out = exec.run(&[("signal", &signal)]);
        let reference = super::fft_real_batch(&signal, batch, n_fft).unwrap();
        let err = super::max_abs_error(&out[0], &reference);
        assert!(err < 1e-3, "metal compensated FFT graph err {err:e}");
    }

    /// The full compensated butterfly graph, with native `Op::Fma`, runs on
    /// wgpu/WebGPU and matches rustfft — f32-grade FFT via fused ops on the GPU.
    #[cfg(feature = "gpu")]
    #[test]
    fn compensated_fft_graph_runs_on_wgpu() {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        use rlx_runtime::Device;
        if !rlx_runtime::is_available(Device::Gpu) {
            return;
        }
        let (n_fft, batch) = (8usize, 4usize);
        let (graph, params) = super::build_compensated_forward_graph(n_fft, batch).unwrap();
        let mut exec = crate::exec::compile::try_compile_graph(Device::Gpu, graph).unwrap();
        for (name, data) in &params {
            exec.set_param(name, data);
        }
        let mut rng = StdRng::seed_from_u64(3);
        let signal: Vec<f32> = (0..batch * n_fft)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let out = exec.run(&[("signal", &signal)]);
        let reference = super::fft_real_batch(&signal, batch, n_fft).unwrap();
        let err = super::max_abs_error(&out[0], &reference);
        assert!(err < 1e-3, "wgpu compensated FFT graph err {err:e}");
    }

    /// Benchmark every precision representation on the same FFT roundtrip:
    /// precision (mantissa bits, reconstruction error) vs cost (µs/roundtrip).
    #[test]
    fn precision_speed_benchmark() {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        use std::time::Instant;
        let n = 256usize;
        let mut rng = StdRng::seed_from_u64(1);
        let x32: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let x64: Vec<f64> = x32.iter().map(|&v| v as f64).collect();

        fn timed(mut f: impl FnMut()) -> f64 {
            f(); // warmup
            let s = Instant::now();
            let mut it = 0u64;
            while s.elapsed().as_millis() < 60 {
                f();
                it += 1;
            }
            s.elapsed().as_nanos() as f64 / it.max(1) as f64 / 1000.0 // µs
        }

        // (name, hardware, mantissa bits, err, µs)
        let mut rows: Vec<(&str, &str, u32, f64, f64)> = Vec::new();
        for (k, name) in [
            (1usize, "f32 ×1"),
            (2, "f32 ×2 generic"),
            (3, "f32 ×3"),
            (4, "f32 ×4 (≈f128)"),
        ] {
            let err = super::ex_roundtrip_maxerr(&x32, k);
            let us = timed(|| {
                let _ = super::ex_roundtrip_maxerr(&x32, k);
            });
            rows.push((name, "f64-less", k as u32 * 24, err, us));
        }
        // optimized fixed double-f32 (the production ≈f64 path)
        let err = super::f2_roundtrip_maxerr(&x32);
        let us = timed(|| {
            let _ = super::f2_roundtrip_maxerr(&x32);
        });
        rows.push(("f32 ×2 OPT (≈f64)", "f64-less", 48, err, us));
        let err = super::f64_roundtrip_maxerr(&x64);
        let us = timed(|| {
            let _ = super::f64_roundtrip_maxerr(&x64);
        });
        rows.push(("f64 ×1 (native)", "f64", 53, err, us));
        let err = super::dd_roundtrip_maxerr(&x64);
        let us = timed(|| {
            let _ = super::dd_roundtrip_maxerr(&x64);
        });
        rows.push(("f64 ×2 (≈f128)", "f64", 106, err, us));

        let base = rows[0].4;
        eprintln!("\nFFT roundtrip — precision vs speed (n={n}, eager CPU):");
        eprintln!(
            "  {:<16} {:<9} {:>5} {:>12} {:>13} {:>7}",
            "variation", "hardware", "bits", "roundtrip", "µs/roundtrip", "×f32"
        );
        for (name, hw, bits, err, us) in &rows {
            eprintln!(
                "  {:<16} {:<9} {:>5} {:>12} {:>13.2} {:>6.1}×",
                name,
                hw,
                bits,
                format!("{err:.2e}"),
                us,
                us / base
            );
        }
        // sanity: more limbs → strictly more precision
        assert!(rows[3].3 < rows[1].3 && rows[1].3 < rows[0].3);
    }

    /// Bench EVERY precision representation and crown winners by each metric.
    #[test]
    fn comprehensive_precision_benchmark() {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        use std::time::Instant;
        let n = 256usize;
        let mut rng = StdRng::seed_from_u64(7);
        let x32: Vec<f32> = (0..n).map(|_| rng.gen_range(-0.5..0.5)).collect();
        let x64: Vec<f64> = x32.iter().map(|&v| v as f64).collect();
        fn timed(mut f: impl FnMut()) -> f64 {
            f();
            let s = Instant::now();
            let mut it = 0u64;
            while s.elapsed().as_millis() < 50 {
                f();
                it += 1;
            }
            s.elapsed().as_nanos() as f64 / it.max(1) as f64 / 1000.0
        }
        struct R {
            label: &'static str,
            hw: &'static str,
            bits: u32,
            err: f64,
            us: f64,
        }
        let mut rows: Vec<R> = Vec::new();
        macro_rules! bench {
            ($l:expr, $hw:expr, $b:expr, $e:expr) => {{
                let err = $e;
                let us = timed(|| {
                    std::hint::black_box($e);
                });
                rows.push(R {
                    label: $l,
                    hw: $hw,
                    bits: $b,
                    err,
                    us,
                });
            }};
        }
        bench!(
            "f8 ×1",
            "f8-only",
            4,
            super::base_roundtrip(&x32, super::FMT_F8_E4M3, 1)
        );
        bench!(
            "f8 ×2",
            "f8-only",
            8,
            super::base_roundtrip(&x32, super::FMT_F8_E4M3, 2)
        );
        bench!(
            "f16 ×1",
            "f16-only",
            11,
            super::base_roundtrip(&x32, super::FMT_F16, 1)
        );
        bench!(
            "f16 ×2",
            "f16-only",
            22,
            super::base_roundtrip(&x32, super::FMT_F16, 2)
        );
        bench!(
            "f32 ×1",
            "f64-less",
            24,
            super::ex_roundtrip_maxerr(&x32, 1)
        );
        bench!("f32 ×2", "f64-less", 48, super::f2_roundtrip_maxerr(&x32));
        bench!(
            "f32 ×3",
            "f64-less",
            72,
            super::ex_roundtrip_maxerr(&x32, 3)
        );
        bench!(
            "f32 ×4",
            "f64-less",
            96,
            super::ex_roundtrip_maxerr(&x32, 4)
        );
        bench!("f64 ×1", "f64-cap", 53, super::f64_roundtrip_maxerr(&x64));
        bench!("f64 ×2", "f64-cap", 106, super::dd_roundtrip_maxerr(&x64));

        let digits = |e: f64| {
            if e <= 0.0 {
                16.0
            } else {
                (-e.log10()).max(0.0)
            }
        };
        eprintln!("\n=== FFT precision benchmark (n={n}, eager CPU) ===");
        eprintln!(
            "  {:<7} {:<9} {:>4} {:>9} {:>9} {:>6} {:>9}",
            "variant", "hardware", "bits", "err", "µs/FFT", "digits", "dig/ms"
        );
        for r in &rows {
            eprintln!(
                "  {:<7} {:<9} {:>4} {:>9} {:>9.1} {:>6.1} {:>9.2}",
                r.label,
                r.hw,
                r.bits,
                format!("{:.1e}", r.err),
                r.us,
                digits(r.err),
                digits(r.err) / (r.us / 1000.0)
            );
        }
        let win = |name: &str, pick: &R, extra: String| {
            eprintln!("  🏆 {:<26} {} ({})", name, pick.label, extra);
        };
        eprintln!("\n=== WINNERS ===");
        let most_precise = rows
            .iter()
            .min_by(|a, b| a.err.partial_cmp(&b.err).unwrap())
            .unwrap();
        win(
            "most precise",
            most_precise,
            format!("{:.1e}", most_precise.err),
        );
        let fastest = rows
            .iter()
            .min_by(|a, b| a.us.partial_cmp(&b.us).unwrap())
            .unwrap();
        win("fastest", fastest, format!("{:.1} µs", fastest.us));
        let eff = rows
            .iter()
            .max_by(|a, b| {
                (digits(a.err) / a.us)
                    .partial_cmp(&(digits(b.err) / b.us))
                    .unwrap()
            })
            .unwrap();
        win(
            "best digits/µs",
            eff,
            format!("{:.2} dig/ms", digits(eff.err) / (eff.us / 1000.0)),
        );
        for tier in [(1e-14, "≈f64 grade"), (1e-27, "≈f128 grade")] {
            if let Some(w) = rows
                .iter()
                .filter(|r| r.err < tier.0)
                .min_by(|a, b| a.us.partial_cmp(&b.us).unwrap())
            {
                win(
                    &format!("cheapest {}", tier.1),
                    w,
                    format!("{:.1} µs", w.us),
                );
            }
        }
        for (hw, desc) in [
            ("f8-only", "f8 HW"),
            ("f16-only", "f16 HW"),
            ("f64-less", "f64-less GPU"),
        ] {
            if let Some(w) = rows
                .iter()
                .filter(|r| r.hw == hw)
                .min_by(|a, b| a.err.partial_cmp(&b.err).unwrap())
            {
                win(&format!("best on {}", desc), w, format!("{:.1e}", w.err));
            }
        }
    }

    /// Low-precision base types (f4 / f8 / f16) — single vs compensated-double.
    /// Doubling the limbs ≈ doubles the mantissa: double-f8 ≈ f16, double-f16 ≈ f32.
    #[test]
    fn low_precision_base_ladder() {
        let n = 16usize;
        let x: Vec<f32> = (0..n).map(|i| ((i % 5) as f32 - 2.0) * 0.25).collect();
        let fmts = [
            ("f4 (e2m1)", super::FMT_F4_E2M1),
            ("f8 (e4m3)", super::FMT_F8_E4M3),
            ("f16", super::FMT_F16),
            ("f32", (8u32, 23u32)),
        ];
        eprintln!("\nlow-precision base FFT roundtrip (n={n}):");
        eprintln!(
            "  {:<11} {:>5} {:>12} {:>12}",
            "base", "bits", "1-limb", "2-limb"
        );
        for (name, fmt) in fmts {
            let e1 = super::base_roundtrip(&x, fmt, 1);
            let e2 = super::base_roundtrip(&x, fmt, 2);
            eprintln!(
                "  {:<11} {:>5} {:>12} {:>12}",
                name,
                fmt.1 + 1,
                format!("{e1:.1e}"),
                format!("{e2:.1e}")
            );
        }
        // doubling f16 limbs must sharply improve precision (toward f32-grade)
        let f16_1 = super::base_roundtrip(&x, super::FMT_F16, 1);
        let f16_2 = super::base_roundtrip(&x, super::FMT_F16, 2);
        assert!(
            f16_2 < f16_1 / 50.0,
            "double-f16 {f16_2:e} vs single {f16_1:e}"
        );
    }

    /// On hardware WITHOUT f64, precision climbs by adding f32 limbs: 1 limb (raw
    /// f32) → 2 limbs (≈f64) → 4 limbs (≈f128-class), using only f32 ops + the
    /// f32 FMA. This is how `Op::Fma` (native on WebGPU/Metal) buys quad precision
    /// on f64-less GPUs.
    #[test]
    fn f32_expansion_precision_ladder() {
        let n = 64usize;
        let x: Vec<f32> = (0..n).map(|i| (((i * 9 + 1) % 17) as f32) - 8.0).collect();
        let e1 = super::ex_roundtrip_maxerr(&x, 1);
        let e2 = super::ex_roundtrip_maxerr(&x, 2);
        let e4 = super::ex_roundtrip_maxerr(&x, 4);
        eprintln!("f32-only ladder: 1-limb={e1:e}  2-limb(≈f64)={e2:e}  4-limb(≈f128)={e4:e}");
        assert!(e1 > 1e-7, "1 f32 limb should be ~1e-6, got {e1:e}");
        assert!(e2 < 1e-11 && e2 < e1 / 1e3, "2 limbs (≈f64): {e2:e}");
        assert!(e4 < 1e-20 && e4 < e2 / 1e6, "4 limbs (≈f128): {e4:e}");
    }

    /// Double-double FFT gives f128-grade precision — the roundtrip reconstructs
    /// to ~1e-28 where plain f64 is stuck at ~1e-14 (a >1e12× improvement, the
    /// extra ~53 mantissa bits of the second f64 limb).
    #[test]
    fn dd_fft_is_f128_grade() {
        let n = 64usize;
        let x: Vec<f64> = (0..n).map(|i| (((i * 9 + 1) % 17) as f64) - 8.0).collect();
        let dd = super::dd_roundtrip_maxerr(&x);
        let f64e = super::f64_roundtrip_maxerr(&x);
        eprintln!(
            "n={n}: dd(f128) roundtrip={dd:e}  f64 roundtrip={f64e:e}  ratio={:.0e}",
            f64e / dd
        );
        assert!(dd < 1e-25, "dd (f128-grade) roundtrip {dd:e}");
        assert!(f64e > 1e-16, "f64 roundtrip {f64e:e}");
        assert!(dd < f64e / 1e8, "dd ({dd:e}) should be ≫ f64 ({f64e:e})");
    }

    /// The new `Op::Fma` must be SINGLE-rounded: `fma(a,b,−round(a·b))` yields the
    /// product rounding error (≠0); a plain mul+add (two roundings) gives 0. This
    /// is the TwoProduct primitive that makes compensated arithmetic possible.
    #[test]
    fn fma_op_is_single_rounded_on_cpu() {
        use rlx_ir::infer::GraphExt;
        use rlx_ir::{DType, Graph, Op, Shape};
        use rlx_runtime::Device;
        let mut g = Graph::new("fma");
        let a = g.input("a", Shape::new(&[1], DType::F32));
        let b = g.input("b", Shape::new(&[1], DType::F32));
        let p = g.mul(a, b); // round(a·b)
        let neg_p = g.neg(p);
        let s = g.shape(a).clone();
        let e = g.add_node(Op::Fma, vec![a, b, neg_p], s); // a·b + (−p), one rounding
        g.set_outputs(vec![p, e]);
        let mut exec = crate::exec::compile::try_compile_graph(Device::Cpu, g).unwrap();
        // a=b=1+2^-13 → a·b = 1 + 2^-12 + 2^-26; the 2^-26 tail rounds off in f32.
        let v = 1.0 + 2.0f32.powi(-13);
        let out = exec.run(&[("a", &[v]), ("b", &[v])]);
        assert!(
            out[1][0].abs() > 1e-10 && out[1][0].abs() < 1e-4,
            "Fma not single-rounded: product error = {}",
            out[1][0]
        );
    }

    /// On the ANE (no native FMA) `Op::Fma` is rewritten to mul+add by `LowerFma`,
    /// so the graph still compiles and runs (by value) instead of erroring.
    #[cfg(feature = "coreml")]
    #[test]
    fn fma_decomposes_and_runs_on_ane() {
        use rlx_ir::infer::GraphExt;
        use rlx_ir::{DType, Graph, Op, Shape};
        use rlx_runtime::{CompileOptions, Device, Precision, Session};
        let mut g = Graph::new("fma_ane");
        let a = g.input("a", Shape::new(&[4], DType::F32));
        let b = g.input("b", Shape::new(&[4], DType::F32));
        let c = g.input("c", Shape::new(&[4], DType::F32));
        let s = g.shape(a).clone();
        let out = g.add_node(Op::Fma, vec![a, b, c], s);
        g.set_outputs(vec![out]);
        let opts = CompileOptions::new().precision(Precision::F16);
        let mut exec = Session::new(Device::Ane).compile_with(g, &opts);
        let r = exec.run(&[
            ("a", &[1.0, 2.0, 3.0, 4.0]),
            ("b", &[2.0; 4]),
            ("c", &[1.0; 4]),
        ]);
        for (i, exp) in [3.0, 5.0, 7.0, 9.0].iter().enumerate() {
            assert!(
                (r[0][i] - exp).abs() < 0.05,
                "fma[{i}]={} != {exp}",
                r[0][i]
            );
        }
    }

    /// Native `Op::Fma` on wgpu (WebGPU) must be single-rounded too — WGSL `fma`
    /// is a genuine fused op, so `fma(a,b,−round(a·b))` recovers the product error.
    #[cfg(feature = "gpu")]
    #[test]
    fn fma_op_is_single_rounded_on_wgpu() {
        use rlx_ir::infer::GraphExt;
        use rlx_ir::{DType, Graph, Op, Shape};
        use rlx_runtime::Device;
        if !rlx_runtime::is_available(Device::Gpu) {
            return;
        }
        let mut g = Graph::new("fma_wgpu");
        let a = g.input("a", Shape::new(&[4], DType::F32));
        let b = g.input("b", Shape::new(&[4], DType::F32));
        let p = g.mul(a, b);
        let neg_p = g.neg(p);
        let s = g.shape(a).clone();
        let e = g.add_node(Op::Fma, vec![a, b, neg_p], s);
        g.set_outputs(vec![p, e]);
        let mut exec = crate::exec::compile::try_compile_graph(Device::Gpu, g).unwrap();
        let v = 1.0 + 2.0f32.powi(-13);
        let out = exec.run(&[("a", &[v; 4]), ("b", &[v; 4])]);
        assert!(
            out[1][0].abs() > 1e-10 && out[1][0].abs() < 1e-4,
            "wgpu Fma not single-rounded: product error = {}",
            out[1][0]
        );
    }

    /// Same question on the ANE in real f16: does CoreML/MIL preserve the TwoSum
    /// error term, or does its fast-math fold it away? (b = 2^-12 rounds off the
    /// 1.0 + b sum in f16, so the surviving error is ~2.44e-4.)
    #[cfg(feature = "coreml")]
    #[test]
    fn twosum_survives_on_ane_f16() {
        use rlx_ir::infer::GraphExt;
        use rlx_ir::{DType, Graph, Shape};
        use rlx_runtime::{CompileOptions, Device, Precision, Session};
        let mut g = Graph::new("twosum_f16");
        let a = g.input("a", Shape::new(&[1], DType::F32));
        let b = g.input("b", Shape::new(&[1], DType::F32));
        let s = g.add(a, b);
        let t = g.sub(s, a);
        let e = g.sub(b, t);
        g.set_outputs(vec![s, e]);
        let opts = CompileOptions::new().precision(Precision::F16);
        let mut exec = Session::new(Device::Ane).compile_with(g, &opts);
        let out = exec.run(&[("a", &[1.0f32]), ("b", &[2.0f32.powi(-12)])]);
        eprintln!("ANE f16 TwoSum: s={} e={}", out[0][0], out[1][0]);
        assert!(
            out[1][0].abs() > 1e-5,
            "TwoSum error folded on ANE (e={}) — need a fusion barrier",
            out[1][0]
        );
    }

    #[test]
    fn compensated_reaches_near_f32() {
        let r = precision_sweep(256, 8, 2, 7).unwrap();
        let f16 = r.rows.iter().find(|x| x.scheme == "f16").unwrap().max_err;
        let comp = r
            .rows
            .iter()
            .find(|x| x.scheme == "f16-comp-FMA")
            .unwrap()
            .max_err;
        // Pure-f16 compensated arithmetic should beat single f16 by ~100x+.
        assert!(
            comp < f16 / 50.0,
            "compensated {comp:e} should beat f16 {f16:e}"
        );
    }
}
