// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Kimi Delta Attention (KDA) — a gated delta-net linear-attention layer.
//!
//! Per KimiDeltaAttention (modeling_kimi_linear.py): q/k/v projections, each
//! through a causal short conv (kernel 4) + silu; L2-normed q/k; a **per-channel**
//! log-decay gate `-exp(A_log)·softplus(f_b(f_a(x)) + dt_bias)` (clamped to
//! `gate_lower_bound`); sigmoid `beta`; the [`Op::GatedDeltaNet`] recurrence with
//! the per-channel gate (`gated_delta_net_pc`); a gated-RMSNorm output
//! (`rms_norm(scan · sigmoid(g_proj(x)))`); `o_proj`; residual add.

use crate::common::{act, scalar_const, sigmoid};
use anyhow::Result;
use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::op::{Activation, Op};
use rlx_ir::{DType, HirGraphExt, Shape};
use std::collections::HashMap;

type Params = HashMap<String, Vec<f32>>;

/// KDA shape parameters.
#[derive(Debug, Clone, Copy)]
pub struct KdaDims {
    pub hidden: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub conv_kernel: usize,
    pub gate_lower_bound: Option<f32>,
    pub eps: f32,
    pub batch: usize,
    pub seq: usize,
}

impl KdaDims {
    pub fn proj(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

/// Dense (unquantized) KDA weights, row-major in the layouts the graph consumes.
/// Projection weights are `[in, out]` (so `mm(x[.,in], w) = [.,out]`); the HF
/// checkpoint stores `nn.Linear` as `[out, in]`, so the loader transposes.
#[derive(Debug, Clone, Default)]
pub struct KdaWeights {
    pub q_proj: Vec<f32>, // [hidden, proj]
    pub k_proj: Vec<f32>, // [hidden, proj]
    pub v_proj: Vec<f32>, // [hidden, proj]
    pub q_conv: Vec<f32>, // [proj, k] (depthwise)
    pub k_conv: Vec<f32>,
    pub v_conv: Vec<f32>,
    pub f_a: Vec<f32>,     // [hidden, head_dim]
    pub f_b: Vec<f32>,     // [head_dim, proj]
    pub dt_bias: Vec<f32>, // [proj]
    pub a_log: Vec<f32>,   // [head_dim] (per channel-within-head)
    pub b_proj: Vec<f32>,  // [hidden, num_heads]
    pub g_proj: Vec<f32>,  // [hidden, proj]
    pub o_norm: Vec<f32>,  // [head_dim]
    pub o_proj: Vec<f32>,  // [proj, hidden]
}

fn reg(
    g: &mut HirMut,
    params: &mut Params,
    name: &str,
    data: Vec<f32>,
    shape: &[usize],
) -> HirNodeId {
    debug_assert_eq!(
        data.len(),
        shape.iter().product::<usize>(),
        "{name} shape mismatch"
    );
    params.insert(name.to_string(), data);
    g.param(name, Shape::new(shape, DType::F32))
}

/// `x[.,in] @ w[in,out]` where `w` is registered under `{prefix}.{name}`.
fn linear(
    g: &mut HirMut,
    params: &mut Params,
    prefix: &str,
    name: &str,
    x: HirNodeId,
    w: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> HirNodeId {
    let wid = reg(
        g,
        params,
        &format!("{prefix}.{name}"),
        w.to_vec(),
        &[in_dim, out_dim],
    );
    g.mm(x, wid)
}

/// L2-normalize over the last axis: `x / sqrt(sum(x^2, -1))` (clamped by `eps`).
fn l2_norm(g: &mut HirMut, x: HirNodeId, shape: &[usize], eps: f32) -> HirNodeId {
    let last = shape.len() - 1;
    let mut sq_shape = shape.to_vec();
    sq_shape[last] = 1;
    let sq = g.mul(x, x);
    let sumsq = g.sum(sq, vec![last], true);
    let rms = act(
        g,
        Activation::Sqrt,
        sumsq,
        Shape::new(&sq_shape, DType::F32),
    );
    let eps_p = scalar_const(g, eps);
    let diff = g.sub(rms, eps_p);
    let relu = g.relu(diff);
    let denom = g.add(eps_p, relu);
    g.div(x, denom)
}

/// `softplus(x) = log(1 + exp(x))`.
fn softplus(g: &mut HirMut, x: HirNodeId, shape: &[usize]) -> HirNodeId {
    let sh = Shape::new(shape, DType::F32);
    let ex = act(g, Activation::Exp, x, sh.clone());
    let one = scalar_const(g, 1.0);
    let sum = g.add(ex, one);
    act(g, Activation::Log, sum, sh)
}

/// Depthwise causal conv1d over the sequence axis (kernel `k`), no bias. Input
/// and output are `[b, s, channels]`. Left-pads `k-1` zeros (causal).
fn depthwise_conv1d_causal(
    g: &mut HirMut,
    params: &mut Params,
    name: &str,
    weight: &[f32], // [channels, k], depthwise
    input: HirNodeId,
    batch: usize,
    seq: usize,
    channels: usize,
    k: usize,
) -> HirNodeId {
    let pad = reg(
        g,
        params,
        &format!("{name}.causal_pad"),
        vec![0f32; batch * (k - 1) * channels],
        &[batch, k - 1, channels],
    );
    let padded = g.concat_(vec![pad, input], 1); // [b, s+k-1, channels]
    let width = seq + k - 1;
    // BSC -> BCW -> NCHW [N,C,L,1], kernel [k,1], depthwise groups=channels.
    let bcw = g.transpose_(padded, vec![0, 2, 1]);
    let nchw = g.reshape_(bcw, vec![batch as i64, channels as i64, width as i64, 1]);
    let w = reg(g, params, name, weight.to_vec(), &[channels, 1, k, 1]);
    let conv = g.add_node(
        Op::Conv {
            kernel_size: vec![k, 1],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: channels,
        },
        vec![nchw, w],
        Shape::new(&[batch, channels, seq, 1], DType::F32),
    );
    let bcs = g.reshape_(conv, vec![batch as i64, channels as i64, seq as i64]);
    g.transpose_(bcs, vec![0, 2, 1]) // -> [b, s, channels]
}

/// Build one KDA layer on the (already input-normed) `h_in` `[batch, seq, hidden]`;
/// returns the **raw** attention output `[batch, seq, hidden]` (no residual — the
/// caller / AttnRes accumulation owns it).
pub fn build_kda_layer(
    g: &mut HirMut,
    params: &mut Params,
    prefix: &str,
    h_in: HirNodeId,
    w: &KdaWeights,
    d: KdaDims,
) -> Result<HirNodeId> {
    let (b, s, hidden, h, hd) = (d.batch, d.seq, d.hidden, d.num_heads, d.head_dim);
    let proj = d.proj();
    let rows = b * s;
    let bshd = [b, s, h, hd];

    let x2d = g.reshape_(h_in, vec![rows as i64, hidden as i64]);

    // (1) q/k/v projections -> [rows, proj]
    let q = linear(g, params, prefix, "q_proj", x2d, &w.q_proj, hidden, proj);
    let k = linear(g, params, prefix, "k_proj", x2d, &w.k_proj, hidden, proj);
    let v = linear(g, params, prefix, "v_proj", x2d, &w.v_proj, hidden, proj);

    // (2) causal short conv + silu, each stream. Reshape to [b, s, proj] first.
    let q3 = g.reshape_(q, vec![b as i64, s as i64, proj as i64]);
    let k3 = g.reshape_(k, vec![b as i64, s as i64, proj as i64]);
    let v3 = g.reshape_(v, vec![b as i64, s as i64, proj as i64]);
    let qc = depthwise_conv1d_causal(
        g,
        params,
        &format!("{prefix}.q_conv1d"),
        &w.q_conv,
        q3,
        b,
        s,
        proj,
        d.conv_kernel,
    );
    let kc = depthwise_conv1d_causal(
        g,
        params,
        &format!("{prefix}.k_conv1d"),
        &w.k_conv,
        k3,
        b,
        s,
        proj,
        d.conv_kernel,
    );
    let vc = depthwise_conv1d_causal(
        g,
        params,
        &format!("{prefix}.v_conv1d"),
        &w.v_conv,
        v3,
        b,
        s,
        proj,
        d.conv_kernel,
    );
    let qc = g.silu(qc);
    let kc = g.silu(kc);
    let vc = g.silu(vc);

    // (3) split heads [b,s,h,hd]; L2-norm q,k (no GQA: num_k == num_v == h).
    let qh = g.reshape_(qc, vec![b as i64, s as i64, h as i64, hd as i64]);
    let kh = g.reshape_(kc, vec![b as i64, s as i64, h as i64, hd as i64]);
    let vh = g.reshape_(vc, vec![b as i64, s as i64, h as i64, hd as i64]);
    let q_l2 = l2_norm(g, qh, &bshd, d.eps);
    let k_l2 = l2_norm(g, kh, &bshd, d.eps);

    // (4) per-channel log-decay gate g_log [b,s,h,hd]
    let f_a = linear(g, params, prefix, "f_a_proj", x2d, &w.f_a, hidden, hd);
    let f_b = linear(g, params, prefix, "f_b_proj", f_a, &w.f_b, hd, proj);
    let gate = g.reshape_(f_b, vec![b as i64, s as i64, h as i64, hd as i64]);
    let dt = reg(
        g,
        params,
        &format!("{prefix}.dt_bias"),
        w.dt_bias.clone(),
        &[proj],
    );
    let dt = g.reshape_(dt, vec![1, 1, h as i64, hd as i64]);
    let biased = g.add(gate, dt);
    let sp = softplus(g, biased, &bshd);
    // A_log is `[head_dim]` (per channel-within-head, shared across heads) in the
    // real checkpoint — broadcast as `[1,1,1,hd]`, not per-head.
    let a_log = reg(
        g,
        params,
        &format!("{prefix}.A_log"),
        w.a_log.clone(),
        &[hd],
    );
    let a_log = g.reshape_(a_log, vec![1, 1, 1, hd as i64]);
    let a_exp = act(
        g,
        Activation::Exp,
        a_log,
        Shape::new(&[1, 1, 1, hd], DType::F32),
    );
    let neg_a = act(
        g,
        Activation::Neg,
        a_exp,
        Shape::new(&[1, 1, 1, hd], DType::F32),
    );
    let mut g_log = g.mul(neg_a, sp); // -exp(A_log) * softplus(...)  [b,s,h,hd]
    if let Some(lb) = d.gate_lower_bound {
        // max(g_log, lb) = lb + relu(g_log - lb)
        let lb_c = scalar_const(g, lb);
        let shifted = g.sub(g_log, lb_c);
        let relu = g.relu(shifted);
        g_log = g.add(lb_c, relu);
    }

    // (5) beta = sigmoid(b_proj(x)) [b,s,h]
    let beta = linear(g, params, prefix, "b_proj", x2d, &w.b_proj, hidden, h);
    let beta = sigmoid(g, beta, Shape::new(&[rows, h], DType::F32));
    let beta = g.reshape_(beta, vec![b as i64, s as i64, h as i64]);

    // (6) per-channel gated delta-net recurrence
    let scan = g.gated_delta_net_pc(
        q_l2,
        k_l2,
        vh,
        g_log,
        beta,
        hd,
        Shape::new(&bshd, DType::F32),
    );

    // (7) FusedRMSNormGated(sigmoid): rms_norm(scan * sigmoid(g_proj(x)))
    let g2 = linear(g, params, prefix, "g_proj", x2d, &w.g_proj, hidden, proj);
    let g2 = g.reshape_(g2, vec![b as i64, s as i64, h as i64, hd as i64]);
    let g2_sig = sigmoid(g, g2, Shape::new(&bshd, DType::F32));
    let gated = g.mul(scan, g2_sig);
    let o_norm_w = reg(
        g,
        params,
        &format!("{prefix}.o_norm"),
        w.o_norm.clone(),
        &[hd],
    );
    let zero_beta = reg(
        g,
        params,
        &format!("{prefix}.o_norm.zero_beta"),
        vec![0f32; hd],
        &[hd],
    );
    let o = g.rms_norm(gated, o_norm_w, zero_beta, d.eps);

    // (8) o_proj → raw attention output (the residual is added by the caller /
    // the AttnRes block-residual accumulation).
    let o2d = g.reshape_(o, vec![rows as i64, proj as i64]);
    let attn = linear(g, params, prefix, "o_proj", o2d, &w.o_proj, proj, hidden);
    Ok(g.reshape_(attn, vec![b as i64, s as i64, hidden as i64]))
}
