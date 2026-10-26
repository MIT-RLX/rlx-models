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

//! Export TEN-VAD as an `.rlxp` package rlx runs alone, through the one
//! exporter ([`rlx_core::export`]): a clip of 16 kHz PCM (int16 units, any
//! number of whole 256-sample hops) in, each hop's voice probability out —
//! what [`crate::TenVad::probabilities_i16`] returns.
//!
//! The runner's host DSP is in the package, as graphs:
//!
//! - the frontend ([`crate::frontend`]): pre-emphasis, the 768-sample
//!   Hann-windowed spectrum (a DFT as two matmuls), 40 log-mel features;
//! - the pitch estimate ([`crate::pitch`]): band energies, the cepstrum,
//!   LPC by Levinson-Durbin (unrolled, each frame stopping where the
//!   runner's loop breaks), the inverse filter, then two recurrences as
//!   `Op::Scan`: the 4 kHz anti-alias biquad cascade over every sample, and
//!   the Viterbi pitch track over the frames;
//! - the network, an `Op::Scan` over the frames whose body is the runner's
//!   own streaming graph, its LSTM state reset every
//!   [`crate::DEFAULT_RESET_FRAMES`] frames as the runner resets it.
//!
//! Where the runner sums in a fixed order the graph sums in the same order
//! (the LPC filter, the correlation windows, the Viterbi); the spectrum and
//! the matrix products round otherwise.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, anyhow, bail};
use rlx_core::export::graph::{Pipeline, Stage, symbolic_in};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport};
use rlx_ir::op::{Activation, BinaryOp, CmpOp, PadMode, ReduceOp};
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use rlx_ten_vad_core::biquad::LOWPASS_4KHZ;
use rlx_ten_vad_core::pitch::tables as pt;
use serde_json::json;

use crate::model::{Shape as NetShape, build_graph};
use crate::{
    CONTEXT_FRAMES, DEFAULT_RESET_FRAMES, FEATURE_LEN, FFT_SIZE, HIDDEN, HOP_SIZE, MEL_BANDS,
    SPECTRUM_BINS, TenVadWeights, WINDOW_SIZE,
};

/// The symbol for the number of hops.
const FRAMES: u32 = 0;

const EPS: f32 = 1e-20;
const POWER_NORM: f32 = 32768.0 * 32768.0;
const PRE_EMPHASIS: f32 = 0.97;
/// A masked Viterbi transition: below any path, and finite.
const MASKED: f32 = -1e38;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

/// Graph-building shorthands over [`Graph`].
struct B<'a> {
    g: &'a mut Graph,
}

impl B<'_> {
    fn k(&mut self, v: f32) -> NodeId {
        self.g.add_node(
            Op::Constant {
                data: v.to_le_bytes().to_vec(),
            },
            vec![],
            f32s(&[1]),
        )
    }
    fn table(&mut self, v: &[f32], dims: &[usize]) -> NodeId {
        let data = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.g.add_node(Op::Constant { data }, vec![], f32s(dims))
    }
    fn act(&mut self, a: Activation, x: NodeId) -> NodeId {
        let s = self.g.node(x).shape.clone();
        self.g.add_node(Op::Activation(a), vec![x], s)
    }
    fn bin(&mut self, op: BinaryOp, a: NodeId, b: NodeId) -> NodeId {
        let s = self
            .g
            .node(a)
            .shape
            .broadcast_with(&self.g.node(b).shape)
            .expect("broadcastable");
        self.g.binary(op, a, b, s)
    }
    fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.bin(BinaryOp::Add, a, b)
    }
    fn sub(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.bin(BinaryOp::Sub, a, b)
    }
    fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.bin(BinaryOp::Mul, a, b)
    }
    fn div(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.bin(BinaryOp::Div, a, b)
    }
    fn max(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.bin(BinaryOp::Max, a, b)
    }
    fn min(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.bin(BinaryOp::Min, a, b)
    }
    fn addk(&mut self, a: NodeId, v: f32) -> NodeId {
        let k = self.k(v);
        self.add(a, k)
    }
    fn mulk(&mut self, a: NodeId, v: f32) -> NodeId {
        let k = self.k(v);
        self.mul(a, k)
    }
    /// `a op b` as 0 / 1.
    fn flag(&mut self, op: CmpOp, a: NodeId, b: NodeId) -> NodeId {
        let s = self
            .g
            .node(a)
            .shape
            .broadcast_with(&self.g.node(b).shape)
            .expect("broadcastable");
        let dims = s.dims().to_vec();
        let c = self.g.add_node(
            Op::Compare(op),
            vec![a, b],
            Shape::from_dims(&dims, DType::Bool),
        );
        self.g.add_node(
            Op::Cast { to: DType::F32 },
            vec![c],
            Shape::from_dims(&dims, DType::F32),
        )
    }
    /// `c ? a : b` for a 0 / 1 `c` and finite `a`, `b`: exactly one of them.
    fn select(&mut self, c: NodeId, a: NodeId, b: NodeId) -> NodeId {
        let one = self.k(1.0);
        let not = self.sub(one, c);
        let a = self.mul(a, c);
        let b = self.mul(b, not);
        self.add(a, b)
    }
    fn dims(&self, x: NodeId) -> Vec<usize> {
        self.g
            .node(x)
            .shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect()
    }
    fn reshape(&mut self, x: NodeId, d: &[usize]) -> NodeId {
        self.g
            .reshape(x, d.iter().map(|&v| v as i64).collect(), f32s(d))
    }
    fn narrow(&mut self, x: NodeId, axis: usize, start: usize, len: usize) -> NodeId {
        self.g.narrow_(x, axis, start, len)
    }
    /// Column `c` of `x` `[rows, cols]`, as `[rows, 1]`.
    fn col(&mut self, x: NodeId, c: usize) -> NodeId {
        self.narrow(x, 1, c, 1)
    }
    fn cat(&mut self, xs: Vec<NodeId>, axis: usize) -> NodeId {
        self.g.concat_(xs, axis)
    }
    /// `x` `[n]` delayed `by` samples, zeros first, still `[n]`.
    fn delay(&mut self, x: NodeId, by: usize) -> NodeId {
        let n = self.dims(x)[0];
        let p = self.g.pad_(x, vec![[by, 0]], PadMode::Constant(0.0));
        self.narrow(p, 0, 0, n)
    }
    /// `x` `[rows, …]` with `by` zero rows first, its last `by` dropped.
    fn delay_rows(&mut self, x: NodeId, by: usize) -> NodeId {
        let d = self.dims(x);
        let mut pad = vec![[0usize, 0usize]; d.len()];
        pad[0] = [by, 0];
        let p = self.g.pad_(x, pad, PadMode::Constant(0.0));
        self.narrow(p, 0, 0, d[0])
    }
    fn reduce_last(&mut self, x: NodeId, op: ReduceOp) -> NodeId {
        let mut d = self.dims(x);
        let last = d.len() - 1;
        d[last] = 1;
        self.g.reduce(x, op, vec![last], true, f32s(&d))
    }
    fn argmax_last(&mut self, x: NodeId) -> NodeId {
        let mut d = self.dims(x);
        let last = d.len() - 1;
        d[last] = 1;
        self.g.add_node(
            Op::ArgMax {
                axis: last,
                keep_dim: true,
            },
            vec![x],
            f32s(&d),
        )
    }
    /// `x` `[…, n]` at the f32 index `at` `[1]` along its last axis.
    fn pick(&mut self, x: NodeId, at: NodeId) -> NodeId {
        let i = self.g.add_node(
            Op::Cast { to: DType::I32 },
            vec![at],
            Shape::new(&[1], DType::I32),
        );
        let mut d = self.dims(x);
        let last = d.len() - 1;
        d[last] = 1;
        self.g.gather(x, i, last, f32s(&d))
    }
}

/// The 4 kHz anti-alias biquad cascade as an `Op::Scan` over `x` `[n]`:
/// its output `[n]`. The carry is the five sections' two registers and the
/// sample's output.
fn lowpass(b: &mut B, x: NodeId, n: usize) -> NodeId {
    let sections = LOWPASS_4KHZ.len();
    let width = 2 * sections + 1;
    let mut body = Graph::new("lowpass_step");
    {
        let mut s = B { g: &mut body };
        let carry = s.g.input("carry", f32s(&[width]));
        let xt = s.g.input("x", f32s(&[1]));
        let mut v = xt;
        let mut regs = Vec::new();
        for (i, sec) in LOWPASS_4KHZ.iter().enumerate() {
            let w0 = s.narrow(carry, 0, 2 * i, 1);
            let w1 = s.narrow(carry, 0, 2 * i + 1, 1);
            // t = x − a1·w0 − a2·w1; y = g·(b0·t + b1·w0 + b2·w2).
            let a1 = s.mulk(w0, sec.a[1]);
            let t = s.sub(v, a1);
            let a2 = s.mulk(w1, sec.a[2]);
            let t = s.sub(t, a2);
            let p0 = s.mulk(t, sec.b[0]);
            let p1 = s.mulk(w0, sec.b[1]);
            let y = s.add(p0, p1);
            let p2 = s.mulk(w1, sec.b[2]);
            let y = s.add(y, p2);
            let gk = s.k(sec.g);
            v = s.mul(gk, y);
            regs.push(t);
            regs.push(w0);
        }
        regs.push(v);
        let next = s.cat(regs, 0);
        s.g.set_outputs(vec![next]);
    }
    let init = b.table(&vec![0.0; width], &[width]);
    let xs = b.reshape(x, &[n, 1]);
    let traj = b.g.add_node(
        Op::Scan {
            body: Box::new(body),
            length: n as u32,
            save_trajectory: true,
            num_bcast: 0,
            num_xs: 1,
            num_checkpoints: 0,
        },
        vec![init, xs],
        f32s(&[n, width]),
    );
    let y = b.narrow(traj, 1, width - 1, 1);
    b.reshape(y, &[n])
}

/// Levinson-Durbin on each frame's lags `ac` (`ORDER + 1` columns `[f, 1]`),
/// stopping where the runner's loop breaks: the LPC columns.
fn levinson(b: &mut B, ac: &[NodeId]) -> Vec<NodeId> {
    let order = pt::ORDER;
    let zero = b.k(0.0);
    let mut lpc: Vec<NodeId> = vec![zero; order];
    let mut error = ac[0];
    // 1 while the frame still iterates.
    let mut live = b.k(1.0);
    let floor = b.mulk(ac[0], 0.001);
    for i in 0..order {
        let mut rr = zero;
        for j in 0..i {
            let t = b.mul(lpc[j], ac[i - j]);
            rr = b.add(rr, t);
        }
        rr = b.add(rr, ac[i + 1]);
        let neg = b.mulk(rr, -1.0);
        let r = b.div(neg, error);
        let mut next = lpc.clone();
        next[i] = r;
        for j in 0..(i + 1) >> 1 {
            let (t1, t2) = (lpc[j], lpc[i - 1 - j]);
            let a = b.mul(r, t2);
            next[j] = b.add(t1, a);
            let c = b.mul(r, t1);
            next[i - 1 - j] = b.add(t2, c);
        }
        let rr2 = b.mul(r, r);
        let drop = b.mul(rr2, error);
        let err_next = b.sub(error, drop);
        for j in 0..order {
            lpc[j] = b.select(live, next[j], lpc[j]);
        }
        error = b.select(live, err_next, error);
        let below = b.flag(CmpOp::Lt, err_next, floor);
        let one = b.k(1.0);
        let goes_on = b.sub(one, below);
        live = b.mul(live, goes_on);
    }
    lpc
}

/// One frame's two cross-correlation half-frames from its excitation `exc`
/// `[f, EXC]`: `[f, 2, MAX_LAG]` sharpened correlations and `[f, 2]`
/// reference energies.
fn xcorr(b: &mut B, exc: NodeId, f: usize) -> (NodeId, NodeId) {
    let (half, lags) = (pt::HALF, pt::MAX_LAG);
    let sq = b.mul(exc, exc);
    let mut corr = Vec::new();
    let mut energy = Vec::new();
    for sub in 0..2 {
        let off = sub * half;
        let reference = b.narrow(exc, 1, lags + off, half);
        // Each lag's dot product, summed term by term as the runner sums.
        let mut inst = b.k(0.0);
        for k in 0..half {
            let r = b.col(reference, k);
            let moving = b.narrow(exc, 1, off + k, lags);
            let t = b.mul(r, moving);
            inst = b.add(inst, t);
        }
        // The energies, summed in order as the runner sums them.
        let col = |b: &mut B, k: usize| b.col(sq, k);
        let mut e0 = b.k(0.0);
        for k in 0..half {
            let c = col(b, lags + off + k);
            e0 = b.add(e0, c);
        }
        let mut window = b.k(0.0);
        for k in 0..half {
            let c = col(b, off + k);
            window = b.add(window, c);
        }
        let mut cols = Vec::with_capacity(lags);
        for i in 0..lags {
            if i > 0 {
                let out = col(b, off + i - 1);
                let w = b.sub(window, out);
                let z = b.k(0.0);
                let w = b.max(w, z);
                let inn = col(b, off + i + half - 1);
                window = b.add(w, inn);
            }
            let d = b.addk(window, 1.0);
            let d = b.add(d, e0);
            let tiny = b.k(1e-12);
            let d = b.max(d, tiny);
            let v = b.col(inst, i);
            let v = b.mulk(v, 2.0);
            cols.push(b.div(v, d));
        }
        // Sharpen: damp a lag not clearly above its half-period neighbours.
        // Every lag it reads is later than the one it writes, so it reads the
        // correlations as they were.
        for i in 0..pt::SHARPEN {
            let a = cols[(lags + i) / 2];
            let c = cols[(lags + i + 2) / 2];
            let d = cols[(lags + i - 1) / 2];
            let peak = b.max(a, c);
            let peak = b.max(peak, d);
            let peak = b.mulk(peak, 1.1);
            let low = b.flag(CmpOp::Lt, cols[i], peak);
            let damped = b.mulk(cols[i], 0.8);
            cols[i] = b.select(low, damped, cols[i]);
        }
        let x = b.cat(cols, 1);
        corr.push(b.reshape(x, &[f, 1, lags]));
        energy.push(e0);
    }
    (b.cat(corr, 1), b.cat(energy, 1))
}

/// The Viterbi pitch track as an `Op::Scan` over the frames: from each
/// frame's six half-frames' weights `w6` `[f, 6]` and correlations `x6`
/// `[f, 6·MAX_LAG]` (oldest first), its pitch in Hz `[f]`.
fn viterbi(b: &mut B, w6: NodeId, x6: NodeId, f: usize) -> NodeId {
    let (n, lags, subs) = (pt::STATES, pt::MAX_LAG, pt::SUBFRAMES);
    // carry: path [n], path_all, best, prev [subs·n], pitch.
    let width = n + 2 + subs * n + 1;
    let penalty = pt::path_penalty();
    let mut pen = vec![0.0f32; n * n];
    let mut mask = vec![MASKED; n * n];
    for i in 0..n {
        let lo = (4i32 - i as i32).min(0);
        for j in lo..=4 {
            let k = i as i32 + j;
            if k >= n as i32 {
                break;
            }
            pen[i * n + k as usize] = penalty[j.unsigned_abs() as usize];
            mask[i * n + k as usize] = 0.0;
        }
    }
    let mut body = Graph::new("viterbi_step");
    {
        let mut s = B { g: &mut body };
        let carry = s.g.input("carry", f32s(&[width]));
        let w = s.g.input("w", f32s(&[subs]));
        let x = s.g.input("x", f32s(&[subs * lags]));
        let mut path = s.narrow(carry, 0, 0, n);
        let mut path_all = s.narrow(carry, 0, n, 1);
        let mut best = s.narrow(carry, 0, n + 1, 1);
        let prev_old = s.narrow(carry, 0, n + 2, subs * n);
        // The weights normalized: `w · (6 / (1e-15 + Σ w))`, summed in order.
        let mut total = s.k(0.0);
        for k in 0..subs {
            let c = s.narrow(w, 0, k, 1);
            total = s.add(total, c);
        }
        let tiny = s.k(1e-15);
        let total = s.add(tiny, total);
        let six = s.k(subs as f32);
        let ratio = s.div(six, total);
        let wn = s.mul(w, ratio);
        let pen = s.table(&pen, &[n, n]);
        let mask = s.table(&mask, &[n, n]);
        // The prev ring moves on by two half-frames.
        let mut prev: Vec<NodeId> = (2..subs).map(|r| s.narrow(prev_old, 0, r * n, n)).collect();
        for sub in subs - 2..subs {
            let p = s.reshape(path, &[1, n]);
            let scores = s.sub(p, pen);
            let scores = s.add(scores, mask);
            let top = s.reduce_last(scores, ReduceOp::Max);
            let top = s.reshape(top, &[n]);
            let arg = s.argmax_last(scores);
            let arg = s.reshape(arg, &[n]);
            let floor = s.addk(path_all, -1e10);
            let up = s.flag(CmpOp::Gt, top, floor);
            let score = s.select(up, top, floor);
            let from = s.select(up, arg, best);
            prev.push(from);
            let ws = s.narrow(wn, 0, sub, 1);
            let xs = s.narrow(x, 0, sub * lags, n);
            let gain = s.mul(ws, xs);
            let next = s.add(score, gain);
            let row = s.reshape(next, &[1, n]);
            let m = s.reduce_last(row, ReduceOp::Max);
            let m = s.reshape(m, &[1]);
            let a = s.argmax_last(row);
            let a = s.reshape(a, &[1]);
            let lowest = s.k(-1e15);
            let found = s.flag(CmpOp::Gt, m, lowest);
            path_all = s.select(found, m, lowest);
            let zero = s.k(0.0);
            best = s.select(found, a, zero);
            path = s.sub(next, path_all);
        }
        // Back through the six half-frames: the contour and its correlation.
        let mut state = best;
        let mut corr = s.k(0.0);
        let mut contour = vec![state; subs];
        for sub in (0..subs).rev() {
            let lagk = s.k(lags as f32);
            contour[sub] = s.sub(lagk, state);
            let ws = s.narrow(wn, 0, sub, 1);
            let xs = s.narrow(x, 0, sub * lags, lags);
            let xv = s.pick(xs, state);
            let t = s.mul(ws, xv);
            corr = s.add(corr, t);
            state = s.pick(prev[sub], state);
        }
        let corr = s.div(corr, six);
        let zero = s.k(0.0);
        let corr = s.max(corr, zero);
        let voiced_at = s.k(pt::VOICED);
        let voiced = s.flag(CmpOp::Ge, corr, voiced_at);
        let (mut sw, mut sx, mut sxx, mut sxy, mut sy) = (zero, zero, zero, zero, zero);
        for (sub, &y) in contour.iter().enumerate() {
            let ws = s.narrow(wn, 0, sub, 1);
            let xf = s.k(sub as f32);
            sw = s.add(sw, ws);
            let wx = s.mul(ws, xf);
            sx = s.add(sx, wx);
            let wxx = s.mul(wx, xf);
            sxx = s.add(sxx, wxx);
            let wxy = s.mul(wx, y);
            sxy = s.add(sxy, wxy);
            let wy = s.mul(ws, y);
            sy = s.add(sy, wy);
        }
        let a = s.mul(sw, sxx);
        let c = s.mul(sx, sx);
        let denom = s.sub(a, c);
        let is_zero = s.flag(CmpOp::Eq, denom, zero);
        let eps = s.k(1e-15);
        let denom = s.select(is_zero, eps, denom);
        let a = s.mul(sw, sxy);
        let c = s.mul(sx, sy);
        let num = s.sub(a, c);
        let slope = s.div(num, denom);
        let mean = s.div(sy, sw);
        // `4 · 2 · N_FEAT`, three frames of two half-frames.
        let span = s.k((4 * 2 * (pt::SUBFRAMES / 2)) as f32);
        let lim = s.div(mean, span);
        let neg = s.mulk(lim, -1.0);
        let clamped = s.max(slope, neg);
        let clamped = s.min(clamped, lim);
        let slope = s.select(voiced, clamped, zero);
        let t = s.mul(slope, sx);
        let t = s.sub(sy, t);
        let intercept = s.div(t, sw);
        let t = s.mulk(slope, 5.5);
        let period = s.add(intercept, t);
        let one = s.k(1.0);
        let period = s.max(period, one);
        let rate = s.k(pt::RATE);
        let hz = s.div(rate, period);
        let pitch = s.select(voiced, hz, zero);
        let mut out = vec![path, path_all, best];
        out.extend(prev);
        out.push(pitch);
        let next = s.cat(out, 0);
        s.g.set_outputs(vec![next]);
    }
    let init = b.table(&vec![0.0; width], &[width]);
    let traj = b.g.add_node(
        Op::Scan {
            body: Box::new(body),
            length: f as u32,
            save_trajectory: true,
            num_bcast: 0,
            num_xs: 2,
            num_checkpoints: 0,
        },
        vec![init, w6, x6],
        f32s(&[f, width]),
    );
    let p = b.narrow(traj, 1, width - 1, 1);
    b.reshape(p, &[f])
}

/// The frontend over `f` hops: `pcm` `[256·f]` (int16 units) → `context`
/// `[f, 3·41]`, each hop's three-frame feature stack, as the runner feeds
/// its network.
fn features(f: usize) -> Graph {
    let n = f * HOP_SIZE;
    let mut g = Graph::new("features");
    let mut b = B { g: &mut g };
    let x = b.g.input("pcm", f32s(&[n]));

    // Pre-emphasis, then the 768-sample frames ending at each hop.
    let xp = b.delay(x, 1);
    let xp = b.mulk(xp, PRE_EMPHASIS);
    let e = b.sub(x, xp);
    let ep = b.g.pad_(e, vec![[2 * HOP_SIZE, 0]], PadMode::Constant(0.0));
    let hops = b.reshape(ep, &[f + 2, HOP_SIZE]);
    let frames: Vec<NodeId> = (0..3).map(|k| b.narrow(hops, 0, k, f)).collect();
    let frames = b.cat(frames, 1);
    let window = b.g.param("frontend.window", f32s(&[WINDOW_SIZE]));
    let frames = b.mul(frames, window);
    // |X[k]|² of the 1024-point transform (the frame zero-padded).
    let re_t =
        b.g.param("frontend.dft_cos", f32s(&[WINDOW_SIZE, SPECTRUM_BINS]));
    let im_t =
        b.g.param("frontend.dft_sin", f32s(&[WINDOW_SIZE, SPECTRUM_BINS]));
    let re = b.g.mm(frames, re_t);
    let im = b.g.mm(frames, im_t);
    let re2 = b.mul(re, re);
    let im2 = b.mul(im, im);
    let power = b.add(re2, im2);

    // 40 log-mel features, standardized.
    let mel_t = b.g.param("frontend.mel", f32s(&[SPECTRUM_BINS, MEL_BANDS]));
    let mel = b.g.mm(power, mel_t);
    // band / POWER_NORM, as the runner divides.
    let norm = b.k(POWER_NORM);
    let mel = b.div(mel, norm);
    let mel = b.addk(mel, EPS);
    let logmel = b.act(Activation::Log, mel);
    let mean = b.g.param("frontend.mean", f32s(&[FEATURE_LEN]));
    let std = b.g.param("frontend.std_eps", f32s(&[FEATURE_LEN]));
    let mean40 = b.narrow(mean, 0, 0, MEL_BANDS);
    let std40 = b.narrow(std, 0, 0, MEL_BANDS);
    let feat = b.sub(logmel, mean40);
    let feat = b.div(feat, std40);

    // Pitch: band energies → log, compressed band by band → cepstrum.
    let be = b.g.param("pitch.bands", f32s(&[SPECTRUM_BINS, pt::BANDS]));
    let band = b.g.mm(power, be);
    let ln10 = b.k(std::f32::consts::LN_10);
    let mut log_max = b.k(-2.0);
    let mut follow = b.k(-2.0);
    let mut ly = Vec::new();
    for i in 0..pt::BANDS {
        let c = b.col(band, i);
        let c = b.addk(c, 1e-2);
        let l = b.act(Activation::Log, c);
        let l = b.div(l, ln10);
        let fl = b.addk(follow, -2.5);
        let v = b.max(l, fl);
        let lm = b.addk(log_max, -8.0);
        let v = b.max(v, lm);
        log_max = b.max(log_max, v);
        follow = b.max(fl, v);
        ly.push(v);
    }
    let ly = b.cat(ly, 1);
    let (dct, scale) = pt::dct();
    let nb = pt::BANDS;
    // cepstrum[i] = Σ_j ly[j]·dct[j][i] · scale.
    let d = b.table(&dct, &[nb, nb]);
    let cep = b.g.mm(ly, d);
    let cep = b.mulk(cep, scale);
    // ex[i] = 10^(Σ_j cep[j]·dct[i][j] · scale) · comp[i].
    let dt: Vec<f32> = (0..nb)
        .flat_map(|j| (0..nb).map(move |i| (i, j)))
        .map(|(i, j)| dct[i * nb + j])
        .collect();
    let dt = b.table(&dt, &[nb, nb]);
    let idct = b.g.mm(cep, dt);
    let idct = b.mulk(idct, scale);
    let ten = b.k(10.0);
    let ex = b.bin(BinaryOp::Pow, ten, idct);
    let comp = b.table(&pt::lpc_compensation(), &[nb]);
    let ex = b.mul(ex, comp);
    let acm = b.table(&pt::autocorrelation(), &[nb, pt::ORDER + 1]);
    let ac = b.g.mm(ex, acm);
    let mut lags: Vec<NodeId> = (0..=pt::ORDER).map(|k| b.col(ac, k)).collect();
    let a0 = b.mulk(lags[0], 1e-4);
    let a0 = b.addk(a0, (WINDOW_SIZE / 12) as f32 / 38.0);
    lags[0] = b.add(lags[0], a0);
    for (i, lag) in lags.iter_mut().enumerate().skip(1) {
        *lag = b.mulk(*lag, 1.0 - 6e-5 * i as f32 * i as f32);
    }
    let lpc = levinson(&mut b, &lags);

    // The inverse filter over the aligned signal (the raw one, delayed),
    // each frame with its own coefficients, summed as the runner sums.
    let aligned = b.delay(x, pt::ALIGN);
    let mut sum = b.reshape(aligned, &[f, HOP_SIZE]);
    for (j, &c) in lpc.iter().enumerate() {
        let past = b.delay(aligned, 1 + j);
        let past = b.reshape(past, &[f, HOP_SIZE]);
        let t = b.mul(c, past);
        sum = b.add(sum, t);
    }
    let sum = b.reshape(sum, &[n]);
    let before = b.delay(sum, 1);
    let before = b.mulk(before, 0.7);
    let filtered = b.add(sum, before);
    let low = lowpass(&mut b, filtered, n);
    // Every fourth sample, then each frame's excitation: the last 129.
    let dec = b.reshape(low, &[n / pt::DECIMATE, pt::DECIMATE]);
    let dec = b.col(dec, 0);
    let per = HOP_SIZE / pt::DECIMATE;
    let dec = b.reshape(dec, &[f * per]);
    let dp = b.g.pad_(dec, vec![[2 * per, 0]], PadMode::Constant(0.0));
    let rows = b.reshape(dp, &[f + 2, per]);
    let parts: Vec<NodeId> = (0..3).map(|k| b.narrow(rows, 0, k, f)).collect();
    let span = b.cat(parts, 1);
    let exc = b.narrow(span, 1, 3 * per - pt::EXCITATION, pt::EXCITATION);
    let (corr, energy) = xcorr(&mut b, exc, f);
    // Six half-frames, oldest first: the two frames before and this one.
    let c1 = b.delay_rows(corr, 1);
    let c2 = b.delay_rows(corr, 2);
    let x6 = b.cat(vec![c2, c1, corr], 1);
    let x6 = b.reshape(x6, &[f, pt::SUBFRAMES * pt::MAX_LAG]);
    let e1 = b.delay_rows(energy, 1);
    let e2 = b.delay_rows(energy, 2);
    let w6 = b.cat(vec![e2, e1, energy], 1);
    let pitch = viterbi(&mut b, w6, x6, f);
    let pitch = b.reshape(pitch, &[f, 1]);
    let m = b.narrow(mean, 0, MEL_BANDS, 1);
    let sd = b.narrow(std, 0, MEL_BANDS, 1);
    let p = b.sub(pitch, m);
    let p = b.div(p, sd);
    let feat = b.cat(vec![feat, p], 1);

    // Three frames of context, newest last, zeros before the first.
    let f1 = b.delay_rows(feat, 1);
    let f2 = b.delay_rows(feat, 2);
    let ctx = b.cat(vec![f2, f1, feat], 1);
    let out = b.reshape(ctx, &[f, CONTEXT_FRAMES * FEATURE_LEN]);
    b.g.node_mut(out).name = Some("context".into());
    b.g.set_outputs(vec![out]);
    g
}

/// The network over `f` hops as an `Op::Scan` whose body is the runner's
/// streaming graph: `context` `[f, 3·41]` → `probability` `[f]`. The LSTM
/// state resets after every [`DEFAULT_RESET_FRAMES`] frames.
fn network(w: &TenVadWeights, f: usize) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let (step, params) = build_graph(NetShape::Streaming, w)?;
    let mut names: Vec<String> = params.iter().map(|(n, _)| n.clone()).collect();
    names.sort();
    let dims: HashMap<String, Shape> = step
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Param { name } => Some((name.clone(), n.shape.clone())),
            _ => None,
        })
        .collect();
    let width = 4 * HIDDEN + 2;
    let mut body = Graph::new("network_step");
    {
        let mut s = B { g: &mut body };
        let carry = s.g.input("carry", f32s(&[width]));
        let weights: Vec<NodeId> = names
            .iter()
            .map(|n| s.g.input(format!("w.{n}"), dims[n].clone()))
            .collect();
        let x = s.g.input("x", f32s(&[CONTEXT_FRAMES * FEATURE_LEN]));
        let feat = s.reshape(x, &[1, CONTEXT_FRAMES, FEATURE_LEN]);
        let state: Vec<NodeId> = (0..4)
            .map(|k| {
                let v = s.narrow(carry, 0, k * HIDDEN, HIDDEN);
                s.reshape(v, &[1, HIDDEN])
            })
            .collect();
        let mut splice = rlx_ir::Splice::new().bind("feat", feat);
        for (k, name) in ["h1", "c1", "h2", "c2"].iter().enumerate() {
            splice = splice.bind(*name, state[k]);
        }
        for (n, &id) in names.iter().zip(&weights) {
            splice = splice.bind(n.clone(), id);
        }
        let out =
            s.g.splice(&step, &splice)
                .map_err(|e| anyhow!("splicing the streaming network: {e}"))?;
        let count = s.narrow(carry, 0, 4 * HIDDEN, 1);
        let count = s.addk(count, 1.0);
        let limit = s.k(DEFAULT_RESET_FRAMES as f32);
        let keep = s.flag(CmpOp::Lt, count, limit);
        let mut next = Vec::new();
        for &o in &out.outputs[1..] {
            let o = s.reshape(o, &[HIDDEN]);
            next.push(s.mul(o, keep));
        }
        next.push(s.mul(count, keep));
        next.push(s.reshape(out.outputs[0], &[1]));
        let next = s.cat(next, 0);
        s.g.set_outputs(vec![next]);
    }
    let mut g = Graph::new("score");
    let mut b = B { g: &mut g };
    let ctx =
        b.g.input("context", f32s(&[f, CONTEXT_FRAMES * FEATURE_LEN]));
    let init = b.table(&vec![0.0; width], &[width]);
    let mut inputs = vec![init];
    for n in &names {
        inputs.push(b.g.param(n.clone(), dims[n].clone()));
    }
    inputs.push(ctx);
    let traj = b.g.add_node(
        Op::Scan {
            body: Box::new(body),
            length: f as u32,
            save_trajectory: true,
            num_bcast: names.len() as u32,
            num_xs: 1,
            num_checkpoints: 0,
        },
        inputs,
        f32s(&[f, width]),
    );
    let p = b.narrow(traj, 1, width - 1, 1);
    let p = b.reshape(p, &[f]);
    b.g.node_mut(p).name = Some("probability".into());
    b.g.set_outputs(vec![p]);
    Ok((g, params.into_iter().collect()))
}

/// The frontend's tables, as the package holds them.
fn frontend_params(w: &TenVadWeights) -> HashMap<String, Vec<f32>> {
    let core = w.core();
    let (mut cos, mut sin) = (
        vec![0f32; WINDOW_SIZE * SPECTRUM_BINS],
        vec![0f32; WINDOW_SIZE * SPECTRUM_BINS],
    );
    for t in 0..WINDOW_SIZE {
        for k in 0..SPECTRUM_BINS {
            let a = 2.0 * std::f64::consts::PI * ((t * k) % FFT_SIZE) as f64 / FFT_SIZE as f64;
            cos[t * SPECTRUM_BINS + k] = a.cos() as f32;
            sin[t * SPECTRUM_BINS + k] = a.sin() as f32;
        }
    }
    let mel = rlx_ten_vad_core::frontend::mel_matrix();
    let mel_t: Vec<f32> = (0..SPECTRUM_BINS)
        .flat_map(|k| (0..MEL_BANDS).map(move |j| (k, j)))
        .map(|(k, j)| mel[j * SPECTRUM_BINS + k])
        .collect();
    HashMap::from([
        ("frontend.window".to_string(), core.window.to_vec()),
        ("frontend.dft_cos".to_string(), cos),
        ("frontend.dft_sin".to_string(), sin),
        ("frontend.mel".to_string(), mel_t),
        ("frontend.mean".to_string(), core.feature_mean.to_vec()),
        (
            "frontend.std_eps".to_string(),
            core.feature_std.iter().map(|s| s + EPS).collect(),
        ),
        ("pitch.bands".to_string(), pt::band_energy()),
    ])
}

/// Export TEN-VAD — its embedded weights, or the `.safetensors` at `path`
/// when it is one — as the package `out`.
pub fn export_rlxp(path: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let owned;
    let w = if path.is_file() {
        owned = TenVadWeights::load(path)?;
        &owned
    } else {
        TenVadWeights::embedded()
    };
    let feats = symbolic_in(FRAMES, (5, 7, 4), |f| Ok(features(f)))?;
    let score = symbolic_in(FRAMES, (5, 7, 4), |f| Ok(network(w, f)?.0))?;
    let mut params = frontend_params(w);
    params.extend(network(w, 4)?.1);
    let stages = vec![
        Stage::new(feats)
            .input("pcm", "pcm")
            .outputs(&["context"])
            .on("cpu"),
        Stage::new(score)
            .input("context", "context")
            .outputs(&["probability"]),
    ];
    let mut p = Pipeline::new("ten-vad", "voice-activity-detection", stages);
    p.inputs = vec![
        json!({"name": "pcm", "kind": "audio", "dtype": "f32", "shape": [null],
        "sample_rate": 16000, "channels": 1, "range": [-32768, 32767],
        "multiple_of": HOP_SIZE}),
    ];
    p.dims = vec![("frames".into(), FRAMES)];
    p.metadata = vec![
        ("hop".into(), json!(HOP_SIZE)),
        ("threshold".into(), json!(crate::DEFAULT_THRESHOLD)),
        (
            "input".into(),
            json!("16 kHz mono PCM in int16 units, whole 256-sample hops; one probability per hop"),
        ),
    ];
    p.exported_by = "rlx-ten-vad".into();
    let ck = Checkpoint::open(path).ok();
    if ck.is_none() && path.exists() && !path.is_dir() {
        bail!("{}: not a TEN-VAD .safetensors", path.display());
    }
    p.write(params, ck.as_ref(), out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("ten-vad", "voice-activity-detection", export_rlxp).claims(&["ten-vad"]),
    );
}
