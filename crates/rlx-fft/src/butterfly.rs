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

//! Learnable Cooley–Tukey butterfly network (eager CPU + optional IR graph).

use crate::config::{FftLearnConfig, TransformDir};
use crate::twiddle::{TwiddleSet, twiddle_index, twiddle_packed_name};
use anyhow::{Result, ensure};
use half::f16;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};
use std::collections::HashMap;

pub(crate) fn split_complex_planes(
    g: &mut Graph,
    state: NodeId,
    batch: usize,
    n: usize,
) -> (NodeId, NodeId) {
    let re_n = g.narrow_(state, 2, 0, 1);
    let im_n = g.narrow_(state, 2, 1, 1);
    let re = g.reshape_(re_n, vec![batch as i64, n as i64]);
    let im = g.reshape_(im_n, vec![batch as i64, n as i64]);
    (re, im)
}

pub(crate) fn merge_complex_planes(
    g: &mut Graph,
    re: NodeId,
    im: NodeId,
    batch: usize,
    n: usize,
) -> NodeId {
    let re3 = g.reshape_(re, vec![batch as i64, n as i64, 1]);
    let im3 = g.reshape_(im, vec![batch as i64, n as i64, 1]);
    g.concat_(vec![re3, im3], 2)
}

/// Slice the packed `[stages, half, 2]` twiddle tensor for one stage into the
/// `[1, groups, 1, stride]` re/im operands the vectorized butterfly expects.
/// (Replaces the former `concat` of `half` per-scalar twiddle param nodes.)
pub(crate) fn build_stage_twiddles(
    g: &mut Graph,
    packed: NodeId,
    stage: usize,
    groups: usize,
    stride: usize,
) -> (NodeId, NodeId) {
    let stage_slice = g.narrow_(packed, 0, stage, 1); // [1, half, 2]
    let re = g.narrow_(stage_slice, 2, 0, 1); // [1, half, 1]
    let im = g.narrow_(stage_slice, 2, 1, 1); // [1, half, 1]
    let w_re = g.reshape_(re, vec![1, groups as i64, 1, stride as i64]);
    let w_im = g.reshape_(im, vec![1, groups as i64, 1, stride as i64]);
    (w_re, w_im)
}

/// Create the single packed twiddle param `[stages, half, 2]` for a set.
pub(crate) fn packed_twiddle_param(
    g: &mut Graph,
    set: TwiddleSet,
    stages: usize,
    half: usize,
) -> (NodeId, ParamSlot) {
    let name = twiddle_packed_name(set);
    let node = g.param(&name, Shape::new(&[stages, half, 2], DType::F32));
    (
        node,
        ParamSlot {
            name,
            param: node,
            grad: None,
        },
    )
}

/// Concat `half` per-scalar twiddle nodes into the `[1, groups, 1, stride]`
/// stage operands. Used **only** by the ternary *deploy* path, whose twiddle
/// params are intentionally sparse (skipped butterflies are omitted), so it
/// cannot share the dense packed tensor. Dense paths use [`build_stage_twiddles`].
pub(crate) fn build_stage_twiddles_mapped(
    g: &mut Graph,
    stage: usize,
    groups: usize,
    stride: usize,
    twiddle_nodes: &HashMap<(usize, usize), (NodeId, NodeId)>,
) -> (NodeId, NodeId) {
    let mut re_scalars = Vec::with_capacity(groups * stride);
    let mut im_scalars = Vec::with_capacity(groups * stride);
    for g_idx in 0..groups {
        for k in 0..stride {
            let b = g_idx * stride + k;
            let (wr, wi) = twiddle_nodes[&(stage, b)];
            re_scalars.push(wr);
            im_scalars.push(wi);
        }
    }
    let re_cat = g.concat_(re_scalars, 0);
    let im_cat = g.concat_(im_scalars, 0);
    let w_re = g.reshape_(re_cat, vec![1, groups as i64, 1, stride as i64]);
    let w_im = g.reshape_(im_cat, vec![1, groups as i64, 1, stride as i64]);
    (w_re, w_im)
}

/// Map-based dense stage application — the ternary deploy path's all-forward
/// stages. Mirrors [`apply_butterfly_stage_vectorized`] but sources twiddles
/// from a per-scalar map instead of the packed tensor.
pub(crate) fn apply_butterfly_stage_vectorized_mapped(
    g: &mut Graph,
    re: NodeId,
    im: NodeId,
    stage: usize,
    batch: usize,
    n: usize,
    twiddle_nodes: &HashMap<(usize, usize), (NodeId, NodeId)>,
) -> (NodeId, NodeId) {
    let stride = 1usize << stage;
    let groups = n / (2 * stride);

    let re4 = g.reshape_(re, vec![batch as i64, groups as i64, 2, stride as i64]);
    let im4 = g.reshape_(im, vec![batch as i64, groups as i64, 2, stride as i64]);

    let a_re = g.narrow_(re4, 2, 0, 1);
    let b_re = g.narrow_(re4, 2, 1, 1);
    let a_im = g.narrow_(im4, 2, 0, 1);
    let b_im = g.narrow_(im4, 2, 1, 1);

    let (w_re, w_im) = build_stage_twiddles_mapped(g, stage, groups, stride, twiddle_nodes);

    let wb_re_a = g.mul(b_re, w_re);
    let wb_re_b = g.mul(b_im, w_im);
    let wb_re = g.sub(wb_re_a, wb_re_b);
    let wb_im_a = g.mul(b_re, w_im);
    let wb_im_b = g.mul(b_im, w_re);
    let wb_im = g.add(wb_im_a, wb_im_b);

    let top_re = g.add(a_re, wb_re);
    let top_im = g.add(a_im, wb_im);
    let bot_re = g.sub(a_re, wb_re);
    let bot_im = g.sub(a_im, wb_im);

    let out_re_cat = g.concat_(vec![top_re, bot_re], 2);
    let out_im_cat = g.concat_(vec![top_im, bot_im], 2);
    let out_re = g.reshape_(out_re_cat, vec![batch as i64, n as i64]);
    let out_im = g.reshape_(out_im_cat, vec![batch as i64, n as i64]);
    (out_re, out_im)
}

/// Slice one `[1]` (re, im) twiddle scalar out of the packed param — for paths
/// (e.g. Stockham) that consume individual butterfly twiddles. Unused slices DCE.
pub(crate) fn packed_twiddle_scalar(
    g: &mut Graph,
    packed: NodeId,
    stage: usize,
    butterfly: usize,
) -> (NodeId, NodeId) {
    let s_slice = g.narrow_(packed, 0, stage, 1); // [1, half, 2]
    let b_slice = g.narrow_(s_slice, 1, butterfly, 1); // [1, 1, 2]
    let re_n = g.narrow_(b_slice, 2, 0, 1);
    let re = g.reshape_(re_n, vec![1]);
    let im_n = g.narrow_(b_slice, 2, 1, 1);
    let im = g.reshape_(im_n, vec![1]);
    (re, im)
}

pub(crate) fn apply_butterfly_stage_vectorized(
    g: &mut Graph,
    re: NodeId,
    im: NodeId,
    stage: usize,
    batch: usize,
    n: usize,
    packed: NodeId,
) -> (NodeId, NodeId) {
    let stride = 1usize << stage;
    let groups = n / (2 * stride);

    let re4 = g.reshape_(re, vec![batch as i64, groups as i64, 2, stride as i64]);
    let im4 = g.reshape_(im, vec![batch as i64, groups as i64, 2, stride as i64]);

    let a_re = g.narrow_(re4, 2, 0, 1);
    let b_re = g.narrow_(re4, 2, 1, 1);
    let a_im = g.narrow_(im4, 2, 0, 1);
    let b_im = g.narrow_(im4, 2, 1, 1);

    let (w_re, w_im) = build_stage_twiddles(g, packed, stage, groups, stride);

    let wb_re_a = g.mul(b_re, w_re);
    let wb_re_b = g.mul(b_im, w_im);
    let wb_re = g.sub(wb_re_a, wb_re_b);
    let wb_im_a = g.mul(b_re, w_im);
    let wb_im_b = g.mul(b_im, w_re);
    let wb_im = g.add(wb_im_a, wb_im_b);

    let top_re = g.add(a_re, wb_re);
    let top_im = g.add(a_im, wb_im);
    let bot_re = g.sub(a_re, wb_re);
    let bot_im = g.sub(a_im, wb_im);

    let out_re_cat = g.concat_(vec![top_re, bot_re], 2);
    let out_im_cat = g.concat_(vec![top_im, bot_im], 2);
    let out_re = g.reshape_(out_re_cat, vec![batch as i64, n as i64]);
    let out_im = g.reshape_(out_im_cat, vec![batch as i64, n as i64]);
    (out_re, out_im)
}

#[inline]
fn cmul(re_a: f32, im_a: f32, re_w: f32, im_w: f32) -> (f32, f32) {
    (re_a * re_w - im_a * im_w, re_a * im_w + im_a * re_w)
}

#[inline]
fn cadd(re_a: f32, im_a: f32, re_b: f32, im_b: f32) -> (f32, f32) {
    (re_a + re_b, im_a + im_b)
}

#[inline]
fn csub(re_a: f32, im_a: f32, re_b: f32, im_b: f32) -> (f32, f32) {
    (re_a - re_b, im_a - im_b)
}

/// Backprop through complex multiply `c = a * w` (w treated as constant for signal backprop).
fn cmul_bw_a(re_dc: f32, im_dc: f32, re_w: f32, im_w: f32) -> (f32, f32) {
    (re_dc * re_w + im_dc * im_w, im_dc * re_w - re_dc * im_w)
}

/// Backprop through complex multiply w.r.t. w.
fn cmul_bw_w(re_dc: f32, im_dc: f32, re_a: f32, im_a: f32) -> (f32, f32) {
    (re_dc * re_a - im_dc * im_a, re_dc * im_a + im_dc * re_a)
}

pub fn num_stages(n_fft: usize) -> usize {
    n_fft.trailing_zeros() as usize
}

pub(crate) fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut rev = 0usize;
    for _ in 0..bits {
        rev = (rev << 1) | (x & 1);
        x >>= 1;
    }
    rev
}

pub(crate) fn bit_reverse_permute(buf: &mut [f32], n_fft: usize) {
    let bits = num_stages(n_fft);
    for i in 0..n_fft {
        let j = bit_reverse(i, bits);
        if i < j {
            buf.swap(i * 2, j * 2);
            buf.swap(i * 2 + 1, j * 2 + 1);
        }
    }
}

pub(crate) fn apply_stage(
    buf: &[f32],
    next: &mut [f32],
    twiddles: &[f32],
    n_fft: usize,
    stage: usize,
) {
    let half = n_fft / 2;
    let stride = 1usize << stage;
    for b in 0..half {
        let group = b / stride;
        let k = b % stride;
        let i0 = (group * 2 * stride + k) * 2;
        let i1 = i0 + stride * 2;
        let w_base = twiddle_index(stage, b, half, 0);
        let w_re = twiddles[w_base];
        let w_im = twiddles[w_base + 1];
        let (b_re, b_im) = cmul(buf[i1], buf[i1 + 1], w_re, w_im);
        let (top_re, top_im) = cadd(buf[i0], buf[i0 + 1], b_re, b_im);
        let (bot_re, bot_im) = csub(buf[i0], buf[i0 + 1], b_re, b_im);
        next[i0] = top_re;
        next[i0 + 1] = top_im;
        next[i1] = bot_re;
        next[i1 + 1] = bot_im;
    }
}

fn butterfly_transform_eager(input: &[f32], twiddles: &[f32], n_fft: usize) -> Result<Vec<f32>> {
    ensure!(
        input.len() == n_fft * 2,
        "input len {} != n_fft*2",
        input.len()
    );
    let half = n_fft / 2;
    let stages = num_stages(n_fft);
    ensure!(
        twiddles.len() >= stages * half * 2,
        "twiddle buffer too small"
    );
    let mut buf = input.to_vec();
    bit_reverse_permute(&mut buf, n_fft);
    for s in 0..stages {
        let mut next = vec![0f32; n_fft * 2];
        apply_stage(&buf, &mut next, twiddles, n_fft, s);
        buf = next;
    }
    Ok(buf)
}

/// Forward FFT on interleaved complex `[n_fft, 2]`.
pub fn butterfly_forward_eager(input: &[f32], twiddles: &[f32], n_fft: usize) -> Result<Vec<f32>> {
    butterfly_transform_eager(input, twiddles, n_fft)
}

/// Inverse FFT on interleaved complex `[n_fft, 2]`.
///
/// Uses `conj(FFT(conj(x)))` with the forward twiddle table — matches unnormalized
/// `rustfft` inverse when forward butterflies match `rustfft` forward.
pub fn butterfly_inverse_eager(input: &[f32], twiddles: &[f32], n_fft: usize) -> Result<Vec<f32>> {
    let mut conj = input.to_vec();
    for i in 0..n_fft {
        conj[i * 2 + 1] *= -1.0;
    }
    let mut out = butterfly_forward_eager(&conj, twiddles, n_fft)?;
    for i in 0..n_fft {
        out[i * 2 + 1] *= -1.0;
    }
    Ok(out)
}

/// Batch eager forward: `signal` is `[batch, n_fft]` real (imag assumed zero).
pub fn butterfly_forward_real_batch(
    signal: &[f32],
    twiddles: &[f32],
    batch: usize,
    n_fft: usize,
) -> Result<Vec<f32>> {
    ensure!(signal.len() == batch * n_fft);
    let mut out = vec![0f32; batch * n_fft * 2];
    for b in 0..batch {
        let mut complex = vec![0f32; n_fft * 2];
        for i in 0..n_fft {
            complex[i * 2] = signal[b * n_fft + i];
        }
        let y = butterfly_forward_eager(&complex, twiddles, n_fft)?;
        out[b * n_fft * 2..(b + 1) * n_fft * 2].copy_from_slice(&y);
    }
    Ok(out)
}

/// Batch inverse FFT: `spectrum` is `[batch, n_fft, 2]` complex.
pub fn butterfly_inverse_complex_batch(
    spectrum: &[f32],
    twiddles: &[f32],
    batch: usize,
    n_fft: usize,
) -> Result<Vec<f32>> {
    ensure!(spectrum.len() == batch * n_fft * 2);
    let mut out = vec![0f32; batch * n_fft * 2];
    for b in 0..batch {
        let chunk = &spectrum[b * n_fft * 2..(b + 1) * n_fft * 2];
        let y = butterfly_inverse_eager(chunk, twiddles, n_fft)?;
        out[b * n_fft * 2..(b + 1) * n_fft * 2].copy_from_slice(&y);
    }
    Ok(out)
}

/// Roundtrip encoder → decoder on synthetic real signals.
pub fn butterfly_encdec_roundtrip_batch(
    signal: &[f32],
    encoder_tw: &[f32],
    decoder_tw: &[f32],
    batch: usize,
    n_fft: usize,
) -> Result<Vec<f32>> {
    let spectrum = butterfly_forward_real_batch(signal, encoder_tw, batch, n_fft)?;
    butterfly_inverse_complex_batch(&spectrum, decoder_tw, batch, n_fft)
}

#[derive(Debug, Clone)]
pub(crate) struct ButterflyTrace {
    pub(crate) stage_inputs: Vec<(usize, Vec<f32>)>,
    pub(crate) output: Vec<f32>,
}

pub(crate) fn apply_conj(buf: &mut [f32], n_fft: usize) {
    for i in 0..n_fft {
        buf[i * 2 + 1] *= -1.0;
    }
}

/// Round an f32 to the nearest f16 and back — the unit fp16-simulation step.
pub(crate) fn round_f16(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
}

/// Like [`apply_stage`] but simulates fp16: twiddles and butterfly outputs are
/// rounded to f16 (compute stays f32 — matches f16-storage / f32-accumulate HW
/// like the Apple Neural Engine).
pub(crate) fn apply_stage_f16(
    buf: &[f32],
    next: &mut [f32],
    twiddles: &[f32],
    n_fft: usize,
    stage: usize,
) {
    let half = n_fft / 2;
    let stride = 1usize << stage;
    for b in 0..half {
        let group = b / stride;
        let k = b % stride;
        let i0 = (group * 2 * stride + k) * 2;
        let i1 = i0 + stride * 2;
        let w_base = twiddle_index(stage, b, half, 0);
        let w_re = round_f16(twiddles[w_base]);
        let w_im = round_f16(twiddles[w_base + 1]);
        let (b_re, b_im) = cmul(buf[i1], buf[i1 + 1], w_re, w_im);
        let (top_re, top_im) = cadd(buf[i0], buf[i0 + 1], b_re, b_im);
        let (bot_re, bot_im) = csub(buf[i0], buf[i0 + 1], b_re, b_im);
        next[i0] = round_f16(top_re);
        next[i0 + 1] = round_f16(top_im);
        next[i1] = round_f16(bot_re);
        next[i1 + 1] = round_f16(bot_im);
    }
}

/// fp16-simulated traced forward (rounds input, twiddles, and every stage
/// output to f16). The trace carries the f16 operating point so the standard
/// [`backward_butterfly_twiddles`] gives the straight-through (STE) gradient.
pub(crate) fn forward_butterfly_traced_f16(
    mut buf: Vec<f32>,
    twiddles: &[f32],
    n_fft: usize,
    bit_reverse_input: bool,
) -> Result<ButterflyTrace> {
    ensure!(buf.len() == n_fft * 2);
    if bit_reverse_input {
        bit_reverse_permute(&mut buf, n_fft);
    }
    for v in buf.iter_mut() {
        *v = round_f16(*v);
    }
    let stages = num_stages(n_fft);
    let mut stage_inputs: Vec<(usize, Vec<f32>)> = Vec::with_capacity(stages);
    for s in 0..stages {
        stage_inputs.push((s, buf.clone()));
        let mut next = vec![0f32; n_fft * 2];
        apply_stage_f16(&buf, &mut next, twiddles, n_fft, s);
        buf = next;
    }
    Ok(ButterflyTrace {
        stage_inputs,
        output: buf,
    })
}

/// Batch fp16-simulated forward FFT (real input) — for measuring true f16 error.
pub fn butterfly_forward_real_batch_f16(
    signal: &[f32],
    twiddles: &[f32],
    batch: usize,
    n_fft: usize,
) -> Result<Vec<f32>> {
    ensure!(signal.len() == batch * n_fft);
    let mut out = vec![0f32; batch * n_fft * 2];
    for b in 0..batch {
        let mut complex = vec![0f32; n_fft * 2];
        for i in 0..n_fft {
            complex[i * 2] = signal[b * n_fft + i];
        }
        let trace = forward_butterfly_traced_f16(complex, twiddles, n_fft, true)?;
        out[b * n_fft * 2..(b + 1) * n_fft * 2].copy_from_slice(&trace.output);
    }
    Ok(out)
}

/// Batch fp16-simulated inverse FFT (`conj(FFT_f16(conj(x)))`, unnormalized).
pub fn butterfly_inverse_complex_batch_f16(
    spectrum: &[f32],
    twiddles: &[f32],
    batch: usize,
    n_fft: usize,
) -> Result<Vec<f32>> {
    ensure!(spectrum.len() == batch * n_fft * 2);
    let mut out = vec![0f32; batch * n_fft * 2];
    for b in 0..batch {
        let mut conj = spectrum[b * n_fft * 2..(b + 1) * n_fft * 2].to_vec();
        apply_conj(&mut conj, n_fft);
        let mut y = forward_butterfly_traced_f16(conj, twiddles, n_fft, true)?.output;
        apply_conj(&mut y, n_fft);
        out[b * n_fft * 2..(b + 1) * n_fft * 2].copy_from_slice(&y);
    }
    Ok(out)
}

pub(crate) fn forward_butterfly_traced(
    mut buf: Vec<f32>,
    twiddles: &[f32],
    n_fft: usize,
    bit_reverse_input: bool,
) -> Result<ButterflyTrace> {
    ensure!(buf.len() == n_fft * 2);
    if bit_reverse_input {
        bit_reverse_permute(&mut buf, n_fft);
    }
    let stages = num_stages(n_fft);
    let mut stage_inputs: Vec<(usize, Vec<f32>)> = Vec::with_capacity(stages);
    for s in 0..stages {
        stage_inputs.push((s, buf.clone()));
        let mut next = vec![0f32; n_fft * 2];
        apply_stage(&buf, &mut next, twiddles, n_fft, s);
        buf = next;
    }
    Ok(ButterflyTrace {
        stage_inputs,
        output: buf,
    })
}

pub(crate) fn backward_butterfly_twiddles(
    mut grad: Vec<f32>,
    trace: &ButterflyTrace,
    twiddles: &[f32],
    n_fft: usize,
    tw_grad: &mut [f32],
    conj_output: bool,
) -> Vec<f32> {
    let half = n_fft / 2;
    if conj_output {
        apply_conj(&mut grad, n_fft);
    }
    for (s, input) in trace.stage_inputs.iter().rev() {
        let mut grad_in = vec![0f32; n_fft * 2];
        let stride = 1usize << *s;
        for bi in 0..half {
            let group = bi / stride;
            let k = bi % stride;
            let i0 = (group * 2 * stride + k) * 2;
            let i1 = i0 + stride * 2;
            let w_base = twiddle_index(*s, bi, half, 0);
            let w_re = twiddles[w_base];
            let w_im = twiddles[w_base + 1];

            let d_top_re = grad[i0];
            let d_top_im = grad[i0 + 1];
            let d_bot_re = grad[i1];
            let d_bot_im = grad[i1 + 1];

            let d_wb_re = d_top_re - d_bot_re;
            let d_wb_im = d_top_im - d_bot_im;
            let d_a_re = d_top_re + d_bot_re;
            let d_a_im = d_top_im + d_bot_im;

            grad_in[i0] += d_a_re;
            grad_in[i0 + 1] += d_a_im;

            let (d_b_re, d_b_im) = cmul_bw_a(d_wb_re, d_wb_im, w_re, w_im);
            grad_in[i1] += d_b_re;
            grad_in[i1 + 1] += d_b_im;

            let (d_w_re, d_w_im) = cmul_bw_w(d_wb_re, d_wb_im, input[i1], input[i1 + 1]);
            tw_grad[w_base] += d_w_re;
            tw_grad[w_base + 1] += d_w_im;
        }
        grad = grad_in;
    }
    grad
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EncDecStepLoss {
    pub reconstruction: f32,
    pub spectrum: f32,
}

/// Train encoder (FFT) + decoder (IFFT) on synthetic roundtrip data.
pub fn butterfly_train_step_encdec(
    signal: &[f32],
    encoder_tw: &mut [f32],
    decoder_tw: &mut [f32],
    batch: usize,
    n_fft: usize,
    lr: f32,
    spectrum_weight: f32,
) -> Result<EncDecStepLoss> {
    ensure!(signal.len() == batch * n_fft);
    let half = n_fft / 2;
    let stages = num_stages(n_fft);
    let scale = n_fft as f32;
    let norm = (batch * n_fft * 2) as f32;

    let mut enc_tw_grad = vec![0f32; stages * half * 2];
    let mut dec_tw_grad = vec![0f32; stages * half * 2];
    let mut recon_loss = 0f32;
    let mut spectrum_loss = 0f32;

    for b in 0..batch {
        let x = &signal[b * n_fft..(b + 1) * n_fft];
        let mut enc_in = vec![0f32; n_fft * 2];
        for i in 0..n_fft {
            enc_in[i * 2] = x[i];
        }

        let enc_trace = forward_butterfly_traced(enc_in, encoder_tw, n_fft, true)?;
        let spectrum = enc_trace.output.clone();

        let mut dec_in = spectrum.clone();
        apply_conj(&mut dec_in, n_fft);
        let dec_trace = forward_butterfly_traced(dec_in, decoder_tw, n_fft, true)?;
        let mut recovered = dec_trace.output.clone();
        apply_conj(&mut recovered, n_fft);

        for i in 0..n_fft {
            let d_re = recovered[i * 2] - x[i] * scale;
            let d_im = recovered[i * 2 + 1];
            recon_loss += d_re * d_re + d_im * d_im;
        }

        let ref_spec = crate::reference::fft_real_batch(x, 1, n_fft)?;
        if spectrum_weight > 0.0 {
            for i in 0..n_fft * 2 {
                let d = spectrum[i] - ref_spec[i];
                spectrum_loss += d * d;
            }
        }

        let mut grad_rec = vec![0f32; n_fft * 2];
        for i in 0..n_fft {
            grad_rec[i * 2] = 2.0 * (recovered[i * 2] - x[i] * scale) / norm;
            grad_rec[i * 2 + 1] = 2.0 * recovered[i * 2 + 1] / norm;
        }

        let mut grad_spec = backward_butterfly_twiddles(
            grad_rec,
            &dec_trace,
            decoder_tw,
            n_fft,
            &mut dec_tw_grad,
            true,
        );
        apply_conj(&mut grad_spec, n_fft);

        if spectrum_weight > 0.0 {
            for i in 0..n_fft * 2 {
                grad_spec[i] += 2.0 * spectrum_weight * (spectrum[i] - ref_spec[i]) / norm;
            }
        }

        let _ = backward_butterfly_twiddles(
            grad_spec,
            &enc_trace,
            encoder_tw,
            n_fft,
            &mut enc_tw_grad,
            false,
        );
    }

    for (t, g) in encoder_tw.iter_mut().zip(enc_tw_grad.iter()) {
        *t -= lr * g;
    }
    for (t, g) in decoder_tw.iter_mut().zip(dec_tw_grad.iter()) {
        *t -= lr * g;
    }

    Ok(EncDecStepLoss {
        reconstruction: recon_loss / (batch * n_fft) as f32,
        spectrum: if spectrum_weight > 0.0 {
            spectrum_loss / (batch * n_fft * 2) as f32
        } else {
            0.0
        },
    })
}

/// Train twiddles for forward or inverse transform.
pub fn butterfly_train_step_dir(
    signal: &[f32],
    twiddles: &mut [f32],
    batch: usize,
    n_fft: usize,
    lr: f32,
    dir: TransformDir,
) -> Result<f32> {
    match dir {
        TransformDir::Forward => butterfly_train_step(signal, twiddles, batch, n_fft, lr),
        TransformDir::Inverse => butterfly_train_step_inverse(signal, twiddles, batch, n_fft, lr),
    }
}

/// MSE loss backward w.r.t. twiddle parameters (forward FFT, real input).
pub fn butterfly_train_step(
    signal: &[f32],
    twiddles: &mut [f32],
    batch: usize,
    n_fft: usize,
    lr: f32,
) -> Result<f32> {
    butterfly_train_complex_dir(
        |b| {
            let mut state = vec![0f32; n_fft * 2];
            for i in 0..n_fft {
                state[i * 2] = signal[b * n_fft + i];
            }
            Ok(state)
        },
        |b| crate::reference::fft_real_batch(&signal[b * n_fft..(b + 1) * n_fft], 1, n_fft),
        twiddles,
        batch,
        n_fft,
        lr,
        true,
        false,
    )
}

/// MSE loss backward w.r.t. twiddle parameters (inverse FFT, complex spectrum input).
pub fn butterfly_train_step_inverse(
    spectrum: &[f32],
    twiddles: &mut [f32],
    batch: usize,
    n_fft: usize,
    lr: f32,
) -> Result<f32> {
    ensure!(spectrum.len() == batch * n_fft * 2);
    butterfly_train_complex_dir(
        |b| {
            let mut v = spectrum[b * n_fft * 2..(b + 1) * n_fft * 2].to_vec();
            for i in 0..n_fft {
                v[i * 2 + 1] *= -1.0;
            }
            Ok(v)
        },
        |b| {
            crate::reference::ifft_complex_batch(
                &spectrum[b * n_fft * 2..(b + 1) * n_fft * 2],
                1,
                n_fft,
            )
        },
        twiddles,
        batch,
        n_fft,
        lr,
        true,
        true,
    )
}

fn butterfly_train_complex_dir(
    mut input_for_batch: impl FnMut(usize) -> Result<Vec<f32>>,
    mut target_for_batch: impl FnMut(usize) -> Result<Vec<f32>>,
    twiddles: &mut [f32],
    batch: usize,
    n_fft: usize,
    lr: f32,
    bit_reverse_input: bool,
    conj_output: bool,
) -> Result<f32> {
    let half = n_fft / 2;
    let stages = num_stages(n_fft);
    let mut loss = 0f32;
    let mut tw_grad = vec![0f32; stages * half * 2];

    for b in 0..batch {
        let buf = input_for_batch(b)?;
        ensure!(buf.len() == n_fft * 2);
        let target = target_for_batch(b)?;

        let trace = forward_butterfly_traced(buf, twiddles, n_fft, bit_reverse_input)?;
        let mut buf = trace.output.clone();
        if conj_output {
            apply_conj(&mut buf, n_fft);
        }

        for i in 0..n_fft * 2 {
            let d = buf[i] - target[i];
            loss += d * d;
        }

        let mut grad = vec![0f32; n_fft * 2];
        for i in 0..n_fft * 2 {
            grad[i] = 2.0 * (buf[i] - target[i]) / (batch * n_fft * 2) as f32;
        }
        if conj_output {
            apply_conj(&mut grad, n_fft);
        }

        backward_butterfly_twiddles(grad, &trace, twiddles, n_fft, &mut tw_grad, false);
    }

    for (t, g) in twiddles.iter_mut().zip(tw_grad.iter()) {
        *t -= lr * g;
    }
    Ok(loss / batch as f32)
}

#[derive(Debug, Clone)]
pub struct ParamSlot {
    pub name: String,
    pub param: NodeId,
    pub grad: Option<NodeId>,
}

#[derive(Debug)]
pub struct ButterflyGraph {
    pub graph: Graph,
    pub params: Vec<ParamSlot>,
    pub signal_in: NodeId,
    pub spectrum_out: NodeId,
}

/// Build a compiled RLX forward FFT graph (real input).
pub fn build_butterfly_forward_graph(cfg: &FftLearnConfig) -> Result<ButterflyGraph> {
    let n = cfg.n_fft;
    let batch = cfg.batch;
    let f = DType::F32;
    let mut g = Graph::new("fft_butterfly");
    let signal_in = g.input("signal", Shape::new(&[batch, n], f));
    let zeros = g.sub(signal_in, signal_in);
    let re = g.reshape_(signal_in, vec![batch as i64, n as i64, 1]);
    let im = g.reshape_(zeros, vec![batch as i64, n as i64, 1]);
    let state = g.concat_(vec![re, im], 2);
    let (params, state) = append_forward_butterfly(&mut g, cfg, state, TwiddleSet::Shared)?;
    g.set_outputs(vec![state]);
    Ok(ButterflyGraph {
        graph: g,
        params,
        signal_in,
        spectrum_out: state,
    })
}

/// Build a compiled RLX inverse FFT graph (complex spectrum input).
pub fn build_butterfly_inverse_graph(cfg: &FftLearnConfig) -> Result<ButterflyGraph> {
    let n = cfg.n_fft;
    let batch = cfg.batch;
    let f = DType::F32;
    let mut g = Graph::new("ifft_butterfly");
    let spectrum_in = g.input("spectrum", Shape::new(&[batch, n, 2], f));
    let re = g.narrow_(spectrum_in, 2, 0, 1);
    let im = g.narrow_(spectrum_in, 2, 1, 1);
    let conj_im = g.neg(im);
    let state = g.concat_(vec![re, conj_im], 2);
    let (params, state) = append_forward_butterfly(&mut g, cfg, state, TwiddleSet::Shared)?;
    let out_re = g.narrow_(state, 2, 0, 1);
    let out_im = g.narrow_(state, 2, 1, 1);
    let out_conj_im = g.neg(out_im);
    let output = g.concat_(vec![out_re, out_conj_im], 2);
    g.set_outputs(vec![output]);
    Ok(ButterflyGraph {
        graph: g,
        params,
        signal_in: spectrum_in,
        spectrum_out: output,
    })
}

pub(crate) fn append_forward_butterfly(
    g: &mut Graph,
    cfg: &FftLearnConfig,
    mut state: NodeId,
    tw_set: TwiddleSet,
) -> Result<(Vec<ParamSlot>, NodeId)> {
    let n = cfg.n_fft;
    let batch = cfg.batch;
    let half = n / 2;
    let stages = cfg.num_stages();

    let bits = cfg.num_stages();
    let mut reordered = Vec::with_capacity(n);
    for i in 0..n {
        let j = bit_reverse(i, bits);
        reordered.push(g.narrow_(state, 1, j, 1));
    }
    state = g.concat_(reordered, 1);
    let (mut re, mut im) = split_complex_planes(g, state, batch, n);

    let (packed, slot) = packed_twiddle_param(g, tw_set, stages, half);
    let params = vec![slot];

    for s in 0..stages {
        (re, im) = apply_butterfly_stage_vectorized(g, re, im, s, batch, n, packed);
    }

    Ok((params, merge_complex_planes(g, re, im, batch, n)))
}
