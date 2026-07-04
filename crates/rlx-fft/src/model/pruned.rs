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

//! Gated / pruned butterfly — skip butterflies via learnable gates (Tier D).

use crate::distill::ternary_gates::GateMode;
use crate::model::butterfly::{
    ParamSlot, apply_butterfly_stage_vectorized_mapped, bit_reverse, bit_reverse_permute,
    build_stage_twiddles, merge_complex_planes, num_stages, packed_twiddle_param,
    split_complex_planes,
};
use crate::model::config::FftLearnConfig;
use crate::model::twiddle::{TwiddleSet, exact_twiddles, twiddle_index, twiddle_name_set};
use anyhow::{Result, ensure};
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};
use std::collections::HashMap;

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

#[inline]
fn cmul_bw_a(re_dc: f32, im_dc: f32, re_w: f32, im_w: f32) -> (f32, f32) {
    (re_dc * re_w + im_dc * im_w, im_dc * re_w - re_dc * im_w)
}

/// One soft gate per butterfly node (`stages * n_fft/2`), init ≈ 1.
pub fn init_gates(n_fft: usize) -> Vec<f32> {
    vec![1.0; gate_count(n_fft)]
}

pub fn gate_count(n_fft: usize) -> usize {
    num_stages(n_fft) * (n_fft / 2)
}

pub fn gate_index(stage: usize, butterfly: usize, half: usize) -> usize {
    stage * half + butterfly
}

pub fn gate_param_name(stage: usize, butterfly: usize) -> String {
    format!("gate.{stage}.{butterfly}")
}

#[derive(Debug, Clone)]
struct GatedButterflyNode {
    stage: usize,
    butterfly: usize,
    i0: usize,
    i1: usize,
    in_a_re: f32,
    in_a_im: f32,
    in_b_re: f32,
    in_b_im: f32,
    top_re: f32,
    top_im: f32,
    bot_re: f32,
    bot_im: f32,
    g: f32,
    gi: usize,
}

#[derive(Debug, Clone)]
struct GatedStageTrace {
    _input: Vec<f32>,
    nodes: Vec<GatedButterflyNode>,
}

#[derive(Debug, Clone)]
pub struct GatedButterflyTrace {
    stages: Vec<GatedStageTrace>,
}

fn apply_stage_gated_traced(
    buf: &[f32],
    next: &mut [f32],
    twiddles: &[f32],
    gates: &[f32],
    n_fft: usize,
    stage: usize,
) -> GatedStageTrace {
    let half = n_fft / 2;
    let stride = 1usize << stage;
    let mut nodes = Vec::with_capacity(half);
    for b in 0..half {
        let group = b / stride;
        let k = b % stride;
        let i0 = (group * 2 * stride + k) * 2;
        let i1 = i0 + stride * 2;
        let w_base = twiddle_index(stage, b, half, 0);
        let w_re = twiddles[w_base];
        let w_im = twiddles[w_base + 1];
        let gi = gate_index(stage, b, half);
        let g = gates[gi].clamp(0.0, 1.0);
        let in_a_re = buf[i0];
        let in_a_im = buf[i0 + 1];
        let in_b_re = buf[i1];
        let in_b_im = buf[i1 + 1];
        let (b_re, b_im) = cmul(in_b_re, in_b_im, w_re, w_im);
        let (top_re, top_im) = cadd(in_a_re, in_a_im, b_re, b_im);
        let (bot_re, bot_im) = csub(in_a_re, in_a_im, b_re, b_im);
        next[i0] = g * top_re + (1.0 - g) * in_a_re;
        next[i0 + 1] = g * top_im + (1.0 - g) * in_a_im;
        next[i1] = g * bot_re + (1.0 - g) * in_b_re;
        next[i1 + 1] = g * bot_im + (1.0 - g) * in_b_im;
        nodes.push(GatedButterflyNode {
            stage,
            butterfly: b,
            i0,
            i1,
            in_a_re,
            in_a_im,
            in_b_re,
            in_b_im,
            top_re,
            top_im,
            bot_re,
            bot_im,
            g,
            gi,
        });
    }
    GatedStageTrace {
        _input: buf.to_vec(),
        nodes,
    }
}

pub fn forward_pruned_traced(
    input: &[f32],
    twiddles: &[f32],
    gates: &[f32],
    n_fft: usize,
) -> Result<(Vec<f32>, GatedButterflyTrace)> {
    ensure!(input.len() == n_fft * 2);
    let half = n_fft / 2;
    let stages = num_stages(n_fft);
    ensure!(gates.len() >= gate_count(n_fft));
    ensure!(twiddles.len() >= stages * half * 2);
    let mut buf = input.to_vec();
    bit_reverse_permute(&mut buf, n_fft);
    let mut trace_stages = Vec::with_capacity(stages);
    for s in 0..stages {
        let mut next = vec![0f32; n_fft * 2];
        let trace = apply_stage_gated_traced(&buf, &mut next, twiddles, gates, n_fft, s);
        trace_stages.push(trace);
        buf = next;
    }
    Ok((
        buf,
        GatedButterflyTrace {
            stages: trace_stages,
        },
    ))
}

pub fn pruned_forward_eager(
    input: &[f32],
    twiddles: &[f32],
    gates: &[f32],
    n_fft: usize,
) -> Result<Vec<f32>> {
    Ok(forward_pruned_traced(input, twiddles, gates, n_fft)?.0)
}

pub fn pruned_forward_real_batch(
    signal: &[f32],
    twiddles: &[f32],
    gates: &[f32],
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
        let y = pruned_forward_eager(&complex, twiddles, gates, n_fft)?;
        out[b * n_fft * 2..(b + 1) * n_fft * 2].copy_from_slice(&y);
    }
    Ok(out)
}

/// Backprop output gradient → gate gradients (twiddles treated as constants).
pub fn backward_pruned_gates(
    mut grad: Vec<f32>,
    trace: &GatedButterflyTrace,
    twiddles: &[f32],
    n_fft: usize,
    gate_grad: &mut [f32],
) {
    let half = n_fft / 2;
    for stage in trace.stages.iter().rev() {
        let mut grad_in = vec![0f32; n_fft * 2];
        for node in &stage.nodes {
            let gi = node.gi;
            let g = node.g;
            let d_out_a_re = grad[node.i0];
            let d_out_a_im = grad[node.i0 + 1];
            let d_out_b_re = grad[node.i1];
            let d_out_b_im = grad[node.i1 + 1];

            gate_grad[gi] += d_out_a_re * (node.top_re - node.in_a_re)
                + d_out_a_im * (node.top_im - node.in_a_im)
                + d_out_b_re * (node.bot_re - node.in_b_re)
                + d_out_b_im * (node.bot_im - node.in_b_im);

            let d_top_re = d_out_a_re * g;
            let d_top_im = d_out_a_im * g;
            let d_bot_re = d_out_b_re * g;
            let d_bot_im = d_out_b_im * g;

            let d_in_a_re = d_out_a_re * (1.0 - g) + d_top_re + d_bot_re;
            let d_in_a_im = d_out_a_im * (1.0 - g) + d_top_im + d_bot_im;

            let d_wb_re = d_top_re - d_bot_re;
            let d_wb_im = d_top_im - d_bot_im;

            let w_base = twiddle_index(node.stage, node.butterfly, half, 0);
            let w_re = twiddles[w_base];
            let w_im = twiddles[w_base + 1];

            let (d_b_re, d_b_im) = cmul_bw_a(d_wb_re, d_wb_im, w_re, w_im);
            grad_in[node.i0] += d_in_a_re;
            grad_in[node.i0 + 1] += d_in_a_im;
            grad_in[node.i1] += d_out_b_re * (1.0 - g) + d_b_re;
            grad_in[node.i1 + 1] += d_out_b_im * (1.0 - g) + d_b_im;
        }
        grad = grad_in;
    }
}

/// Apply task-gradient gate update with L1 sparsity subgradient (no blind shrink).
pub fn gate_train_step(
    signal: &[f32],
    twiddles: &[f32],
    gates: &mut [f32],
    grad_denoised: &[f32],
    freq_mask: &[f32],
    denoiser_scale: &[f32],
    batch: usize,
    n_fft: usize,
    gate_lr: f32,
    sparsity_weight: f32,
) -> Result<()> {
    gate_train_step_with_delta(
        signal,
        twiddles,
        gates,
        grad_denoised,
        freq_mask,
        denoiser_scale,
        batch,
        n_fft,
        gate_lr,
        sparsity_weight,
        DEFAULT_MAX_GATE_DELTA,
    )
}

pub fn gate_train_step_with_delta(
    signal: &[f32],
    twiddles: &[f32],
    gates: &mut [f32],
    grad_denoised: &[f32],
    freq_mask: &[f32],
    denoiser_scale: &[f32],
    batch: usize,
    n_fft: usize,
    gate_lr: f32,
    _sparsity_weight: f32,
    max_gate_delta: f32,
) -> Result<()> {
    ensure!(signal.len() == batch * n_fft);
    ensure!(grad_denoised.len() == batch * n_fft * 2);
    let n_gates = gate_count(n_fft);
    ensure!(gates.len() >= n_gates);
    let mut gate_grad = vec![0f32; n_gates];
    let norm = (batch * n_fft * 2) as f32;

    for b in 0..batch {
        let mut complex = vec![0f32; n_fft * 2];
        for i in 0..n_fft {
            complex[i * 2] = signal[b * n_fft + i];
        }
        let mut grad_raw = vec![0f32; n_fft * 2];
        for i in 0..n_fft * 2 {
            let gi = b * n_fft * 2 + i;
            grad_raw[i] = grad_denoised[gi] * denoiser_scale[i] * freq_mask[i] / norm;
        }
        let (_, trace) = forward_pruned_traced(&complex, twiddles, gates, n_fft)?;
        backward_pruned_gates(grad_raw, &trace, twiddles, n_fft, &mut gate_grad);
    }

    let max_gg = gate_grad.iter().map(|g| g.abs()).fold(0f32, f32::max);
    let gg_scale = 1.0 / (1.0 + max_gg);

    for (g, gg) in gates.iter_mut().zip(gate_grad.iter()) {
        // Task gradient only — L1 sparsity stays in the scalar loss, not the gate update,
        // so gates are not blind-shrunk toward zero (which collapses mel/welch accuracy).
        let mut delta = gate_lr * gg_scale * *gg;
        delta = delta.clamp(-max_gate_delta, max_gate_delta);
        *g = (*g - delta).clamp(0.0, 1.0);
    }
    Ok(())
}

pub fn mean_gate(gates: &[f32]) -> f32 {
    if gates.is_empty() {
        return 1.0;
    }
    gates.iter().sum::<f32>() / gates.len() as f32
}

/// Default hard-gate threshold at inference (butterfly active if gate ≥ threshold).
pub const DEFAULT_GATE_THRESHOLD: f32 = 0.5;

/// Learning rate for gate updates (separate from twiddle lr).
pub const DEFAULT_GATE_LR: f32 = 1e-3;

/// Max gate change per gradient step — prevents collapse from large task gradients.
pub const DEFAULT_MAX_GATE_DELTA: f32 = 0.01;

/// Binarize soft gates for fast inference: `g < threshold` → 0, else 1.
pub fn hard_gates(gates: &[f32], threshold: f32) -> Vec<f32> {
    gates
        .iter()
        .map(|&g| if g >= threshold { 1.0 } else { 0.0 })
        .collect()
}

pub fn gate_sparsity_loss(gates: &[f32]) -> f32 {
    gates.iter().map(|&g| g.abs()).sum::<f32>() / gates.len().max(1) as f32
}

pub fn pruned_forward_real_batch_with_mode(
    signal: &[f32],
    twiddles: &[f32],
    gates: &[f32],
    batch: usize,
    n_fft: usize,
    hard: bool,
    threshold: f32,
) -> Result<Vec<f32>> {
    let g = if hard {
        hard_gates(gates, threshold)
    } else {
        gates.to_vec()
    };
    pruned_forward_real_batch(signal, twiddles, &g, batch, n_fft)
}

pub fn active_gate_count(gates: &[f32], threshold: f32) -> usize {
    gates.iter().filter(|&&g| g >= threshold).count()
}

pub fn exact_init(cfg: &FftLearnConfig) -> (Vec<f32>, Vec<f32>) {
    (exact_twiddles(cfg), init_gates(cfg.n_fft))
}

#[derive(Debug)]
pub struct GatedButterflyGraph {
    pub graph: Graph,
    pub twiddle_params: Vec<ParamSlot>,
    pub gate_params: Vec<ParamSlot>,
    pub signal_in: NodeId,
    pub spectrum_out: NodeId,
}

/// Gated butterfly FFT on a real-valued `[batch, n]` signal node (already in `g`).
pub fn append_gated_butterfly_from_real_signal(
    g: &mut Graph,
    cfg: &FftLearnConfig,
    signal: NodeId,
) -> Result<(Vec<ParamSlot>, Vec<ParamSlot>, NodeId)> {
    let n = cfg.n_fft;
    let batch = cfg.batch;
    let zeros = g.sub(signal, signal);
    let re = g.reshape_(signal, vec![batch as i64, n as i64, 1]);
    let im = g.reshape_(zeros, vec![batch as i64, n as i64, 1]);
    let state = g.concat_(vec![re, im], 2);
    append_gated_forward_butterfly(g, cfg, state, TwiddleSet::Shared)
}

/// Compiled RLX graph: gated butterfly FFT + optional hard gates baked as params.
pub fn build_gated_butterfly_forward_graph(cfg: &FftLearnConfig) -> Result<GatedButterflyGraph> {
    let n = cfg.n_fft;
    let batch = cfg.batch;
    let f = DType::F32;
    let mut g = Graph::new("fft_gated_butterfly");
    let signal_in = g.input("signal", Shape::new(&[batch, n], f));
    let (twiddle_params, gate_params, state) =
        append_gated_butterfly_from_real_signal(&mut g, cfg, signal_in)?;
    g.set_outputs(vec![state]);
    Ok(GatedButterflyGraph {
        graph: g,
        twiddle_params,
        gate_params,
        signal_in,
        spectrum_out: state,
    })
}

fn build_stage_gates(
    g: &mut Graph,
    stage: usize,
    groups: usize,
    stride: usize,
    gate_nodes: &HashMap<(usize, usize), NodeId>,
) -> NodeId {
    let mut scalars = Vec::with_capacity(groups * stride);
    for g_idx in 0..groups {
        for k in 0..stride {
            let b = g_idx * stride + k;
            scalars.push(gate_nodes[&(stage, b)]);
        }
    }
    let gate_cat = g.concat_(scalars, 0);
    g.reshape_(gate_cat, vec![1, groups as i64, 1, stride as i64])
}

fn apply_gated_butterfly_stage_vectorized(
    g: &mut Graph,
    re: NodeId,
    im: NodeId,
    stage: usize,
    batch: usize,
    n: usize,
    packed: NodeId,
    gate_nodes: &HashMap<(usize, usize), NodeId>,
    one: NodeId,
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
    let gate = build_stage_gates(g, stage, groups, stride, gate_nodes);
    let inv_g = g.sub(one, gate);

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

    let g_top_re = g.mul(gate, top_re);
    let g_a_re = g.mul(inv_g, a_re);
    let top_out_re = g.add(g_top_re, g_a_re);
    let g_top_im = g.mul(gate, top_im);
    let g_a_im = g.mul(inv_g, a_im);
    let top_out_im = g.add(g_top_im, g_a_im);
    let g_bot_re = g.mul(gate, bot_re);
    let g_b_re = g.mul(inv_g, b_re);
    let bot_out_re = g.add(g_bot_re, g_b_re);
    let g_bot_im = g.mul(gate, bot_im);
    let g_b_im = g.mul(inv_g, b_im);
    let bot_out_im = g.add(g_bot_im, g_b_im);

    let out_re_cat = g.concat_(vec![top_out_re, bot_out_re], 2);
    let out_im_cat = g.concat_(vec![top_out_im, bot_out_im], 2);
    let out_re = g.reshape_(out_re_cat, vec![batch as i64, n as i64]);
    let out_im = g.reshape_(out_im_cat, vec![batch as i64, n as i64]);
    (out_re, out_im)
}

pub(crate) fn append_gated_forward_butterfly(
    g: &mut Graph,
    cfg: &FftLearnConfig,
    mut state: NodeId,
    tw_set: TwiddleSet,
) -> Result<(Vec<ParamSlot>, Vec<ParamSlot>, NodeId)> {
    let n = cfg.n_fft;
    let batch = cfg.batch;
    let half = n / 2;
    let stages = cfg.num_stages();
    let f = DType::F32;
    let one = g.param("const.one", Shape::new(&[1], f));

    let bits = cfg.num_stages();
    let mut reordered = Vec::with_capacity(n);
    for i in 0..n {
        let j = bit_reverse(i, bits);
        reordered.push(g.narrow_(state, 1, j, 1));
    }
    state = g.concat_(reordered, 1);
    let (mut re, mut im) = split_complex_planes(g, state, batch, n);

    // Twiddles pack into one tensor; gates stay per-scalar (each is multiplied
    // into a distinct stage/butterfly and bound individually by the loader).
    let (packed, twiddle_slot) = packed_twiddle_param(g, tw_set, stages, half);
    let twiddle_params = vec![twiddle_slot];
    let mut gate_nodes: HashMap<(usize, usize), NodeId> = HashMap::new();
    let mut gate_params = Vec::new();

    for s in 0..stages {
        for b in 0..half {
            let gate_name = gate_param_name(s, b);
            let gate = g.param(&gate_name, Shape::new(&[1], f));
            gate_params.push(ParamSlot {
                name: gate_name,
                param: gate,
                grad: None,
            });
            gate_nodes.insert((s, b), gate);
        }
    }

    for s in 0..stages {
        (re, im) = apply_gated_butterfly_stage_vectorized(
            g,
            re,
            im,
            s,
            batch,
            n,
            packed,
            &gate_nodes,
            one,
        );
    }

    Ok((
        twiddle_params,
        gate_params,
        merge_complex_planes(g, re, im, batch, n),
    ))
}

fn build_ternary_stage_gate_tensor(
    g: &mut Graph,
    stage: usize,
    groups: usize,
    stride: usize,
    half: usize,
    gates: &[i8],
    reverse: bool,
) -> NodeId {
    let f = DType::F32;
    let mut scalars = Vec::with_capacity(groups * stride);
    for g_idx in 0..groups {
        for k in 0..stride {
            let b = g_idx * stride + k;
            let gi = gate_index(stage, b, half);
            let val = match GateMode::from_i8(gates[gi]) {
                GateMode::Skip => 0.0,
                GateMode::Forward => {
                    if reverse {
                        0.0
                    } else {
                        1.0
                    }
                }
                GateMode::Reverse => {
                    if reverse {
                        1.0
                    } else {
                        0.0
                    }
                }
            };
            let name = format!("const.{}.{stage}.{b}", if reverse { "rev" } else { "gate" });
            let node = g.param(&name, Shape::new(&[1], f));
            scalars.push(node);
            let _ = val;
        }
    }
    let cat = g.concat_(scalars, 0);
    g.reshape_(cat, vec![1, groups as i64, 1, stride as i64])
}

fn build_ternary_stage_twiddles(
    g: &mut Graph,
    stage: usize,
    groups: usize,
    stride: usize,
    half: usize,
    gates: &[i8],
    twiddle_params: &mut Vec<ParamSlot>,
) -> (NodeId, NodeId) {
    let f = DType::F32;
    let mut re_scalars = Vec::with_capacity(groups * stride);
    let mut im_scalars = Vec::with_capacity(groups * stride);
    for g_idx in 0..groups {
        for k in 0..stride {
            let b = g_idx * stride + k;
            let gi = gate_index(stage, b, half);
            let mode = GateMode::from_i8(gates[gi]);
            let (w_re_name, w_im_name) = if mode == GateMode::Skip {
                (
                    format!("const.zero.re.{stage}.{b}"),
                    format!("const.zero.im.{stage}.{b}"),
                )
            } else {
                (
                    twiddle_name_set(TwiddleSet::Shared, stage, b, "re"),
                    twiddle_name_set(TwiddleSet::Shared, stage, b, "im"),
                )
            };
            let w_re = g.param(&w_re_name, Shape::new(&[1], f));
            let w_im = g.param(&w_im_name, Shape::new(&[1], f));
            if mode != GateMode::Skip {
                twiddle_params.push(ParamSlot {
                    name: w_re_name,
                    param: w_re,
                    grad: None,
                });
                twiddle_params.push(ParamSlot {
                    name: w_im_name,
                    param: w_im,
                    grad: None,
                });
            }
            re_scalars.push(w_re);
            im_scalars.push(w_im);
        }
    }
    let re_cat = g.concat_(re_scalars, 0);
    let im_cat = g.concat_(im_scalars, 0);
    let w_re = g.reshape_(re_cat, vec![1, groups as i64, 1, stride as i64]);
    let w_im = g.reshape_(im_cat, vec![1, groups as i64, 1, stride as i64]);
    (w_re, w_im)
}

/// One butterfly slot: skip passes inputs through; forward/reverse run twiddle math only.
fn apply_single_ternary_butterfly(
    g: &mut Graph,
    a_re: NodeId,
    a_im: NodeId,
    b_re: NodeId,
    b_im: NodeId,
    stage: usize,
    butterfly: usize,
    mode: GateMode,
    twiddle_map: &HashMap<(usize, usize), (NodeId, NodeId)>,
) -> (NodeId, NodeId, NodeId, NodeId) {
    match mode {
        GateMode::Skip => (a_re, a_im, b_re, b_im),
        GateMode::Forward | GateMode::Reverse => {
            let (w_re, w_im) = twiddle_map[&(stage, butterfly)];
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
            if mode == GateMode::Forward {
                (top_re, top_im, bot_re, bot_im)
            } else {
                (bot_re, bot_im, top_re, top_im)
            }
        }
    }
}

/// Deploy path: emit butterfly math only for forward/reverse gates (skip = narrow passthrough).
fn apply_ternary_butterfly_stage_sparse(
    g: &mut Graph,
    re: NodeId,
    im: NodeId,
    stage: usize,
    _batch: usize,
    n: usize,
    half: usize,
    gates: &[i8],
    twiddle_map: &HashMap<(usize, usize), (NodeId, NodeId)>,
) -> (NodeId, NodeId) {
    let stride = 1usize << stage;
    let groups = n / (2 * stride);
    let mut re_slots: Vec<Option<NodeId>> = vec![None; n];
    let mut im_slots: Vec<Option<NodeId>> = vec![None; n];
    for g_idx in 0..groups {
        for k in 0..stride {
            let b = g_idx * stride + k;
            let top_pos = g_idx * 2 * stride + k;
            let bot_pos = top_pos + stride;
            let mode = GateMode::from_i8(gates[gate_index(stage, b, half)]);
            let a_re = g.narrow_(re, 1, top_pos, 1);
            let a_im = g.narrow_(im, 1, top_pos, 1);
            let b_re = g.narrow_(re, 1, bot_pos, 1);
            let b_im = g.narrow_(im, 1, bot_pos, 1);
            let (o_top_re, o_top_im, o_bot_re, o_bot_im) = apply_single_ternary_butterfly(
                g,
                a_re,
                a_im,
                b_re,
                b_im,
                stage,
                b,
                mode,
                twiddle_map,
            );
            re_slots[top_pos] = Some(o_top_re);
            im_slots[top_pos] = Some(o_top_im);
            re_slots[bot_pos] = Some(o_bot_re);
            im_slots[bot_pos] = Some(o_bot_im);
        }
    }
    let re_parts: Vec<NodeId> = re_slots.into_iter().map(|s| s.expect("re slot")).collect();
    let im_parts: Vec<NodeId> = im_slots.into_iter().map(|s| s.expect("im slot")).collect();
    let out_re = g.concat_(re_parts, 1);
    let out_im = g.concat_(im_parts, 1);
    (out_re, out_im)
}

fn stage_skip_count(gates: &[i8], stage: usize, half: usize) -> usize {
    (0..half)
        .filter(|&b| GateMode::from_i8(gates[gate_index(stage, b, half)]) == GateMode::Skip)
        .count()
}

fn stage_all_forward(gates: &[i8], stage: usize, half: usize) -> bool {
    (0..half).all(|b| GateMode::from_i8(gates[gate_index(stage, b, half)]) == GateMode::Forward)
}

fn build_deploy_twiddle_map(
    g: &mut Graph,
    cfg: &FftLearnConfig,
    gates: &[i8],
    twiddle_params: &mut Vec<ParamSlot>,
) -> HashMap<(usize, usize), (NodeId, NodeId)> {
    let f = DType::F32;
    let half = cfg.n_fft / 2;
    let mut map = HashMap::new();
    for s in 0..cfg.num_stages() {
        for b in 0..half {
            if GateMode::from_i8(gates[gate_index(s, b, half)]) == GateMode::Skip {
                continue;
            }
            let w_re_name = twiddle_name_set(TwiddleSet::Shared, s, b, "re");
            let w_im_name = twiddle_name_set(TwiddleSet::Shared, s, b, "im");
            let w_re = g.param(&w_re_name, Shape::new(&[1], f));
            let w_im = g.param(&w_im_name, Shape::new(&[1], f));
            twiddle_params.push(ParamSlot {
                name: w_re_name,
                param: w_re,
                grad: None,
            });
            twiddle_params.push(ParamSlot {
                name: w_im_name,
                param: w_im,
                grad: None,
            });
            map.insert((s, b), (w_re, w_im));
        }
    }
    map
}

/// Vectorized stage: skip=identity, forward=normal butterfly, reverse=swapped outputs.
#[allow(dead_code)]
fn apply_ternary_butterfly_stage_vectorized(
    g: &mut Graph,
    re: NodeId,
    im: NodeId,
    stage: usize,
    batch: usize,
    n: usize,
    half: usize,
    gates: &[i8],
    one: NodeId,
    twiddle_params: &mut Vec<ParamSlot>,
) -> (NodeId, NodeId) {
    let stride = 1usize << stage;
    let groups = n / (2 * stride);

    if (0..half).all(|b| GateMode::from_i8(gates[gate_index(stage, b, half)]) == GateMode::Skip) {
        return (re, im);
    }

    let re4 = g.reshape_(re, vec![batch as i64, groups as i64, 2, stride as i64]);
    let im4 = g.reshape_(im, vec![batch as i64, groups as i64, 2, stride as i64]);

    let a_re = g.narrow_(re4, 2, 0, 1);
    let b_re = g.narrow_(re4, 2, 1, 1);
    let a_im = g.narrow_(im4, 2, 0, 1);
    let b_im = g.narrow_(im4, 2, 1, 1);

    let (w_re, w_im) =
        build_ternary_stage_twiddles(g, stage, groups, stride, half, gates, twiddle_params);
    let gate = build_ternary_stage_gate_tensor(g, stage, groups, stride, half, gates, false);
    let rev = build_ternary_stage_gate_tensor(g, stage, groups, stride, half, gates, true);
    let inv_g = g.sub(one, gate);
    let inv_r = g.sub(one, rev);

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

    let rev_top_re_a = g.mul(rev, bot_re);
    let rev_top_re_b = g.mul(inv_r, top_re);
    let rev_top_re = g.add(rev_top_re_a, rev_top_re_b);
    let rev_top_im_a = g.mul(rev, bot_im);
    let rev_top_im_b = g.mul(inv_r, top_im);
    let rev_top_im = g.add(rev_top_im_a, rev_top_im_b);
    let rev_bot_re_a = g.mul(rev, top_re);
    let rev_bot_re_b = g.mul(inv_r, bot_re);
    let rev_bot_re = g.add(rev_bot_re_a, rev_bot_re_b);
    let rev_bot_im_a = g.mul(rev, top_im);
    let rev_bot_im_b = g.mul(inv_r, bot_im);
    let rev_bot_im = g.add(rev_bot_im_a, rev_bot_im_b);

    let g_top_re = g.mul(gate, rev_top_re);
    let g_a_re = g.mul(inv_g, a_re);
    let top_out_re = g.add(g_top_re, g_a_re);
    let g_top_im = g.mul(gate, rev_top_im);
    let g_a_im = g.mul(inv_g, a_im);
    let top_out_im = g.add(g_top_im, g_a_im);
    let g_bot_re = g.mul(gate, rev_bot_re);
    let g_b_re = g.mul(inv_g, b_re);
    let bot_out_re = g.add(g_bot_re, g_b_re);
    let g_bot_im = g.mul(gate, rev_bot_im);
    let g_b_im = g.mul(inv_g, b_im);
    let bot_out_im = g.add(g_bot_im, g_b_im);

    let out_re_cat = g.concat_(vec![top_out_re, bot_out_re], 2);
    let out_im_cat = g.concat_(vec![top_out_im, bot_out_im], 2);
    let out_re = g.reshape_(out_re_cat, vec![batch as i64, n as i64]);
    let out_im = g.reshape_(out_im_cat, vec![batch as i64, n as i64]);
    (out_re, out_im)
}

/// Fixed deploy-time param values for compile-time specialization (gates, zeros, twiddles).
pub fn ternary_deploy_param_bindings(
    gates: &[i8],
    n_fft: usize,
    twiddles: &[f32],
    hann: &[f32],
    block_gain: &[f32],
    block_bias: &[f32],
    mel_filters: &[f32],
) -> std::collections::HashMap<String, Vec<f32>> {
    use crate::model::twiddle::{TwiddleSet, twiddle_index, twiddle_name_set};
    let mut out = std::collections::HashMap::new();
    let half = n_fft / 2;
    let stages = n_fft.trailing_zeros() as usize;
    out.insert("hann".into(), hann.to_vec());
    out.insert("corr.gain".into(), block_gain.to_vec());
    out.insert("corr.bias".into(), block_bias.to_vec());
    out.insert("mel.filters".into(), mel_filters.to_vec());
    for s in 0..stages {
        for b in 0..half {
            let gi = gate_index(s, b, half);
            if GateMode::from_i8(gates[gi]) == GateMode::Skip {
                continue;
            }
            let base = twiddle_index(s, b, half, 0);
            let re_name = twiddle_name_set(TwiddleSet::Shared, s, b, "re");
            let im_name = twiddle_name_set(TwiddleSet::Shared, s, b, "im");
            out.insert(re_name, vec![twiddles[base]]);
            out.insert(im_name, vec![twiddles[base + 1]]);
        }
    }
    out
}

fn load_ternary_const_params(
    compiled: &mut rlx_runtime::CompiledGraph,
    gates: &[i8],
    n_fft: usize,
) {
    let half = n_fft / 2;
    let stages = n_fft.trailing_zeros() as usize;
    for s in 0..stages {
        for b in 0..half {
            let gi = gate_index(s, b, half);
            let (gate, rev) = match GateMode::from_i8(gates[gi]) {
                GateMode::Skip => (0.0, 0.0),
                GateMode::Forward => (1.0, 0.0),
                GateMode::Reverse => (1.0, 1.0),
            };
            compiled.set_param(&format!("const.gate.{s}.{b}"), &[gate]);
            compiled.set_param(&format!("const.rev.{s}.{b}"), &[rev]);
            if gate == 0.0 {
                compiled.set_param(&format!("const.zero.re.{s}.{b}"), &[0.0]);
                compiled.set_param(&format!("const.zero.im.{s}.{b}"), &[0.0]);
            }
        }
    }
    compiled.set_param("const.one", &[1.0]);
}

/// Compile-time pruned ternary butterfly: skip gates are identity (no twiddle ops).
pub fn append_ternary_butterfly_from_real_signal(
    g: &mut Graph,
    cfg: &FftLearnConfig,
    signal: NodeId,
    gates: &[i8],
) -> Result<(Vec<ParamSlot>, NodeId)> {
    let n = cfg.n_fft;
    let batch = cfg.batch;
    let half = n / 2;
    ensure!(
        gates.len() >= gate_count(n),
        "gates len {} < {}",
        gates.len(),
        gate_count(n)
    );

    let zeros = g.sub(signal, signal);
    let re0 = g.reshape_(signal, vec![batch as i64, n as i64, 1]);
    let im0 = g.reshape_(zeros, vec![batch as i64, n as i64, 1]);
    let mut state = g.concat_(vec![re0, im0], 2);

    let bits = cfg.num_stages();
    let mut reordered = Vec::with_capacity(n);
    for i in 0..n {
        let j = bit_reverse(i, bits);
        reordered.push(g.narrow_(state, 1, j, 1));
    }
    state = g.concat_(reordered, 1);
    let mut twiddle_params = Vec::new();
    let twiddle_map = build_deploy_twiddle_map(g, cfg, gates, &mut twiddle_params);
    for s in 0..cfg.num_stages() {
        if stage_skip_count(gates, s, half) == half {
            continue;
        }
        let (mut re, mut im) = split_complex_planes(g, state, batch, n);
        if stage_all_forward(gates, s, half) {
            (re, im) =
                apply_butterfly_stage_vectorized_mapped(g, re, im, s, batch, n, &twiddle_map);
        } else {
            (re, im) = apply_ternary_butterfly_stage_sparse(
                g,
                re,
                im,
                s,
                batch,
                n,
                half,
                gates,
                &twiddle_map,
            );
        }
        state = merge_complex_planes(g, re, im, batch, n);
    }

    Ok((twiddle_params, state))
}

/// Bake compile-time gate / reverse / zero-twiddle constants into a compiled graph.
pub fn apply_ternary_const_params(
    compiled: &mut rlx_runtime::CompiledGraph,
    gates: &[i8],
    n_fft: usize,
) {
    load_ternary_const_params(compiled, gates, n_fft);
}

/// Count butterflies that run twiddle math (forward + reverse).
pub fn active_ternary_butterfly_count(gates: &[i8]) -> usize {
    gates
        .iter()
        .filter(|&&g| GateMode::from_i8(g) != GateMode::Skip)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn all_gates_one_matches_butterfly() {
        let cfg = FftLearnConfig::new(64, 2).unwrap();
        let (tw, gates) = exact_init(&cfg);
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let signal: Vec<f32> = (0..128).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let pr = pruned_forward_real_batch(&signal, &tw, &gates, 2, 64).unwrap();
        let bf =
            crate::model::butterfly::butterfly_forward_real_batch(&signal, &tw, 2, 64).unwrap();
        let err = crate::model::reference::max_abs_error(&pr, &bf);
        assert!(err < 1e-4, "err={err}");
    }

    #[test]
    fn gate_grad_does_not_nan() {
        let cfg = FftLearnConfig::new(64, 4).unwrap();
        let (tw, mut gates) = exact_init(&cfg);
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        let signal: Vec<f32> = (0..256).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let grad = vec![0.01; 4 * 64 * 2];
        let mask = vec![1.0; 64 * 2];
        let scale = vec![1.0; 64 * 2];
        gate_train_step(
            &signal, &tw, &mut gates, &grad, &mask, &scale, 4, 64, 1e-3, 1e-3,
        )
        .unwrap();
        assert!(gates.iter().all(|g| g.is_finite()));
    }

    #[test]
    fn gated_graph_builds() {
        let cfg = FftLearnConfig::new(64, 4).unwrap();
        build_gated_butterfly_forward_graph(&cfg).unwrap();
    }

    #[test]
    #[ignore = "slow compile; run with --ignored"]
    fn compile_gated_256_cpu() {
        let cfg = FftLearnConfig::new(256, 8).unwrap();
        let model = crate::learned::model::FastLearnedFftModel::new(&cfg, 40, 16_000.0);
        let t = std::time::Instant::now();
        crate::learned::compile::compile_learned_mel(&model, &cfg, rlx_runtime::Device::Cpu, 0.5)
            .unwrap();
        eprintln!("compile_256_ms={}", t.elapsed().as_millis());
        assert!(t.elapsed().as_secs() < 30);
    }
}
