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

use crate::common::{
    WeightQuant, act, bf16_backbone_active, emit_int8_resident, fake_quant_weight,
    int8_backbone_active, resolve_quant, scalar_const, sigmoid,
};
use anyhow::Result;
use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::op::{Activation, Op};
use rlx_ir::{DType, HirGraphExt, Shape};
use std::collections::HashMap;

type Params = HashMap<String, Vec<f32>>;

/// Chunk size for the opt-in FlashKDA chunked-parallel prefill path (see
/// [`crate::kda_chunk`]), or `None` to keep the native sequential recurrence.
/// `RLX_KDA_CHUNK` enables it; a value in `2..=16` sets the chunk size, anything
/// else (e.g. `1`, `on`) uses FlashKDA's default of 16.
fn kda_chunk_size() -> Option<usize> {
    let v = std::env::var("RLX_KDA_CHUNK").ok()?;
    match v.trim().parse::<usize>() {
        Ok(n) if (2..=16).contains(&n) => Some(n),
        _ => Some(16),
    }
}

/// Whether the chunked KDA prefill uses the `Op::Scan`-based K2 (O(1) graph size)
/// instead of the unrolled chunk loop. `RLX_KDA_CHUNK_SCAN=1`. Bit-identical.
fn kda_chunk_use_scan() -> bool {
    std::env::var("RLX_KDA_CHUNK_SCAN")
        .ok()
        .map(|v| matches!(v.trim(), "1" | "on" | "true" | "yes"))
        .unwrap_or(false)
}

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
    // Delegate to the shared helper so KDA projections (incl. the fused input
    // projection) honor the bf16-resident backbone collector.
    crate::common::linear(g, params, prefix, name, x, w, in_dim, out_dim)
}

/// Fuse the 6 KDA projections that all read the normed input `x2d`
/// (q, k, v, f_a, beta, g) into ONE `[hidden, Σout]` matmul + column narrows —
/// 6 GEMVs collapse to 1 (the dominant KDA body compute), fewer kernel launches,
/// one weight pass. **Bit-exact**: each output column is the same independent
/// dot-product over `hidden` whether computed alone or alongside others. The
/// per-row weight repack is baked into the (cached) graph, done once. Returns the
/// 6 projections as 2D `[rows, out]` nodes, same shapes as the separate `linear`s.
#[allow(clippy::type_complexity)]
fn fused_input_proj(
    g: &mut HirMut,
    params: &mut Params,
    prefix: &str,
    x2d: HirNodeId,
    w: &KdaWeights,
    hidden: usize,
    proj: usize,
    hd: usize,
    h: usize,
) -> (
    HirNodeId,
    HirNodeId,
    HirNodeId,
    HirNodeId,
    HirNodeId,
    HirNodeId,
) {
    // The 6 same-input projections, each with its own NAME so the per-projection
    // quant policy (`RLX_KIMI_QUANT=mixed`) can pick a scheme per sub-projection —
    // quantizing HERE, before the concat, is what lets `mixed` protect v_proj/g_proj
    // (2nd/3rd most sensitive) even though they share one fused matmul. For fixed
    // int8/int4/off this is identical to quantizing the whole fused matrix (per-out-
    // channel scales are column-independent); the bf16-backbone path is untouched.
    let parts: [(&[f32], usize, &str); 6] = [
        (&w.q_proj, proj, "q_proj"),
        (&w.k_proj, proj, "k_proj"),
        (&w.v_proj, proj, "v_proj"),
        (&w.f_a, hd, "f_a"),
        (&w.b_proj, h, "b_proj"),
        (&w.g_proj, proj, "g_proj"),
    ];
    let total: usize = parts.iter().map(|(_, o, _)| *o).sum();
    let bf16 = bf16_backbone_active();
    // Prequant-load: the projections came in EMPTY (the loader skipped their bf16
    // read); the fused int8 codes are mmapped by name in `emit_int8_resident`, so
    // skip the 708 MB f32 assembly entirely and pass an empty placeholder.
    let load = crate::common::prequant_load_active();
    // repack per-row: fused[r, off_i .. off_i+out_i] = Q_i(W_i)[r, :]  (weights [in,out])
    let mut fused_w = if load {
        Vec::new()
    } else {
        vec![0f32; hidden * total]
    };
    let mut offs = [0usize; 6];
    let mut off = 0usize;
    for (i, (wi, out, name)) in parts.iter().enumerate() {
        offs[i] = off;
        if !load {
            // per-projection fake-quant at the source (skip under bf16-backbone, which
            // registers its own bf16 param below).
            // Raw parts under bf16- OR int8-resident (the whole fused matrix is packed
            // below); otherwise per-projection fake-quant at the source.
            let sch = if bf16 || int8_backbone_active() {
                WeightQuant::None
            } else {
                resolve_quant(name)
            };
            let qi = if sch == WeightQuant::None {
                None
            } else {
                Some(fake_quant_weight(wi, hidden, *out, sch))
            };
            let src: &[f32] = qi.as_deref().unwrap_or(wi);
            for r in 0..hidden {
                fused_w[r * total + off..r * total + off + out]
                    .copy_from_slice(&src[r * out..r * out + out]);
            }
        }
        off += out;
    }
    // Register the (already per-projection-quantized) fused weight WITHOUT further
    // quant — except the bf16-backbone path, which must go through `linear` to emit
    // a bf16 param. `linear` under a fixed/mixed policy would re-quantize the whole
    // matrix, so bypass it with a plain `reg` + `mm` when we've quantized here.
    let fused = if int8_backbone_active() {
        // int8-resident: pack the WHOLE fused input matmul (the dominant KDA GEMM).
        emit_int8_resident(
            g,
            params,
            &format!("{prefix}.in_proj_fused"),
            x2d,
            &fused_w,
            hidden,
            total,
        )
    } else if bf16 {
        linear(
            g,
            params,
            prefix,
            "in_proj_fused",
            x2d,
            &fused_w,
            hidden,
            total,
        )
    } else {
        let wid = reg(
            g,
            params,
            &format!("{prefix}.in_proj_fused"),
            fused_w,
            &[hidden, total],
        );
        g.mm(x2d, wid)
    };
    (
        g.narrow_(fused, 1, offs[0], proj),
        g.narrow_(fused, 1, offs[1], proj),
        g.narrow_(fused, 1, offs[2], proj),
        g.narrow_(fused, 1, offs[3], hd),
        g.narrow_(fused, 1, offs[4], h),
        g.narrow_(fused, 1, offs[5], proj),
    )
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

/// `softplus(x) = log(1 + exp(x))`, via the native `Activation::Softplus` op —
/// one kernel instead of exp→+1→log, and the backend uses the numerically
/// stable `max(0,x) + ln_1p(exp(-|x|))` form (no overflow for large x).
fn softplus(g: &mut HirMut, x: HirNodeId, shape: &[usize]) -> HirNodeId {
    act(g, Activation::Softplus, x, Shape::new(shape, DType::F32))
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
    // Causal left-pad `k-1` zeros → `[b, s+k-1, c]`, then depthwise conv:
    //   out[t,c] = Σ_j weight[c,j] · padded[t+j, c]
    // Build it as a shift-multiply-accumulate over the `k` taps directly in
    // the channels-last `[b,s,c]` layout: each tap is a `narrow` of the padded
    // sequence times a per-channel weight vector `[1,1,c]`. This avoids the two
    // full-tensor NCL transposes + `Op::Conv` the previous version used (which
    // moved the whole sequence tensor twice, strided) — the taps are cheap
    // contiguous slices and the mul/add are fusable elementwise ops. `k` is
    // small (3–4), and the per-tap vectors are split host-side from `[c,k]`.
    let pad = reg(
        g,
        params,
        &format!("{name}.causal_pad"),
        vec![0f32; batch * (k - 1) * channels],
        &[batch, k - 1, channels],
    );
    let padded = g.concat_(vec![pad, input], 1); // [b, s+k-1, channels]
    // Fallback to the old NCL-transpose + `Op::Conv` path (`RLX_KDA_CONV_OLD=1`).
    // The shift-multiply-accumulate default is ~1.6–2× faster on the full KDA
    // layer (real dims: proj=96·128=12288 channels, k=4) — the transposes moved
    // that wide tensor twice, strided. Kept as an escape hatch.
    if std::env::var_os("RLX_KDA_CONV_OLD").is_some() {
        let width = seq + k - 1;
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
        return g.transpose_(bcs, vec![0, 2, 1]);
    }
    let mut acc: Option<HirNodeId> = None;
    for j in 0..k {
        // Tap j across all channels: weight is [channels, k] (k-minor).
        let tap: Vec<f32> = (0..channels).map(|c| weight[c * k + j]).collect();
        let wj = reg(g, params, &format!("{name}.tap{j}"), tap, &[1, 1, channels]);
        let shifted = g.narrow_(padded, 1, j, seq); // [b, s, channels]
        let term = g.mul(shifted, wj); // broadcast [1,1,channels]
        acc = Some(match acc {
            None => term,
            Some(a) => g.add(a, term),
        });
    }
    acc.expect("conv kernel size >= 1") // [b, s, channels]
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

    // (1) fuse the 6 same-input projections (q,k,v,f_a,beta,g) into one matmul
    let (q, k, v, f_a, beta, g2) = fused_input_proj(g, params, prefix, x2d, w, hidden, proj, hd, h);

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

    // (4) per-channel log-decay gate g_log [b,s,h,hd]  (f_a from the fused proj)
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
        // g_log ≤ 0 always (−exp·softplus), so max(g_log, lb) == clamp(lb, 0):
        // one native Clamp instead of sub→relu→add (3 elementwise passes over
        // [b,s,h,hd] × 69 KDA layers).
        g_log = g.add_node(
            Op::Clamp { min: lb, max: 0.0 },
            vec![g_log],
            Shape::new(&bshd, DType::F32),
        );
    }

    // (5) beta = sigmoid(b_proj(x)) [b,s,h]  (b_proj from the fused proj)
    let beta = sigmoid(g, beta, Shape::new(&[rows, h], DType::F32));
    let beta = g.reshape_(beta, vec![b as i64, s as i64, h as i64]);

    // (6) per-channel gated delta-net recurrence. Opt-in FlashKDA chunked-parallel
    // form (`RLX_KDA_CHUNK`), else the native sequential `Op::GatedDeltaNet`.
    let scan = if let Some(chunk) = kda_chunk_size() {
        let (out, _final_state) = crate::kda_chunk::build_kda_chunked_scan(
            g,
            q_l2,
            k_l2,
            vh,
            g_log,
            beta,
            crate::kda_chunk::ChunkDims {
                batch: b,
                seq: s,
                heads: h,
                head_dim: hd,
                chunk,
                use_scan: kda_chunk_use_scan(),
            },
            None,
        );
        out
    } else {
        g.gated_delta_net_pc(
            q_l2,
            k_l2,
            vh,
            g_log,
            beta,
            hd,
            Shape::new(&bshd, DType::F32),
        )
    };

    // (7) FusedRMSNormGated(sigmoid): rms_norm(scan * sigmoid(g_proj(x)))  (g from fused)
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

/// Depthwise causal conv1d in **decode** mode: the causal left-pad is the carried
/// `conv_state` `[b, k-1, channels]` (the previous tokens' pre-conv values)
/// instead of zeros. Returns `(output [b, s_new, channels], new_conv_state
/// [b, k-1, channels])` — the last `k-1` pre-conv rows, to feed the next step.
#[allow(clippy::too_many_arguments)]
fn depthwise_conv1d_carry(
    g: &mut HirMut,
    params: &mut Params,
    name: &str,
    weight: &[f32],
    input: HirNodeId,      // [b, s_new, channels]
    conv_state: HirNodeId, // [b, k-1, channels]
    batch: usize,
    s_new: usize,
    channels: usize,
    k: usize,
) -> (HirNodeId, HirNodeId) {
    let padded = g.concat_(vec![conv_state, input], 1); // [b, s_new+k-1, channels]
    let width = s_new + k - 1;

    // DECODE fast path (s_new==1): the depthwise conv is just a per-channel weighted
    // sum over the length-k window — `out[c] = Σ_j padded[b,j,c] · weight[c,j]`. Do it
    // as a broadcast-mul + reduce, skipping the Conv op AND its two NCHW transposes
    // (6 transposes/KDA layer gone). Bit-exact to the Conv (same cross-correlation
    // sum, same order). The weight is host-transposed [channels,k]→[1,k,channels] once
    // (baked into the cached graph).
    if s_new == 1 {
        let mut wt = vec![0f32; k * channels];
        for c in 0..channels {
            for j in 0..k {
                wt[j * channels + c] = weight[c * k + j];
            }
        }
        let wnode = reg(g, params, &format!("{name}.dec"), wt, &[1, k, channels]);
        let prod = g.mul(padded, wnode); // [b,k,channels]
        let summed = g.sum(prod, vec![1], false); // [b, channels]
        let out = g.reshape_(summed, vec![batch as i64, 1, channels as i64]);
        let new_state = g.narrow_(padded, 1, s_new, k - 1); // [b, k-1, channels]
        return (out, new_state);
    }

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
        Shape::new(&[batch, channels, s_new, 1], DType::F32),
    );
    let bcs = g.reshape_(conv, vec![batch as i64, channels as i64, s_new as i64]);
    let out = g.transpose_(bcs, vec![0, 2, 1]); // [b, s_new, channels]
    // Next state = last k-1 rows of `padded`.
    let new_state = g.narrow_(padded, 1, s_new, k - 1); // [b, k-1, channels]
    (out, new_state)
}

/// One KDA **decode step** — process `d.seq` NEW tokens O(1) in the prefix length
/// by carrying the short-conv state and the recurrent scan state, instead of
/// re-scanning the whole sequence. `conv_state_{q,k,v}` are `[b, k-1, proj]`
/// (previous tokens' pre-conv projections), `scan_state` is `[b, h, hd, hd]`
/// (written back in place by the carry op — the caller adds it to `set_outputs`).
/// Returns `(out [b, s_new, hidden], new_conv_state_{q,k,v})`. Equivalent to
/// [`build_kda_layer`] run over the full prefix, but incremental.
#[allow(clippy::too_many_arguments)]
pub fn build_kda_decode_step(
    g: &mut HirMut,
    params: &mut Params,
    prefix: &str,
    h_in: HirNodeId,
    conv_state_q: HirNodeId,
    conv_state_k: HirNodeId,
    conv_state_v: HirNodeId,
    scan_state: HirNodeId,
    w: &KdaWeights,
    d: KdaDims,
) -> Result<(HirNodeId, HirNodeId, HirNodeId, HirNodeId)> {
    let (b, s, hidden, h, hd) = (d.batch, d.seq, d.hidden, d.num_heads, d.head_dim);
    let proj = d.proj();
    let rows = b * s;
    let bshd = [b, s, h, hd];
    let kk = d.conv_kernel;

    let x2d = g.reshape_(h_in, vec![rows as i64, hidden as i64]);
    // fuse the 6 same-input projections (q,k,v,f_a,beta,g) into one matmul
    let (q, k, v, f_a, beta, g2) = fused_input_proj(g, params, prefix, x2d, w, hidden, proj, hd, h);
    let q3 = g.reshape_(q, vec![b as i64, s as i64, proj as i64]);
    let k3 = g.reshape_(k, vec![b as i64, s as i64, proj as i64]);
    let v3 = g.reshape_(v, vec![b as i64, s as i64, proj as i64]);

    // Conv with carried state (returns the next state too).
    let (qc, ncs_q) = depthwise_conv1d_carry(
        g,
        params,
        &format!("{prefix}.q_conv1d"),
        &w.q_conv,
        q3,
        conv_state_q,
        b,
        s,
        proj,
        kk,
    );
    let (kc, ncs_k) = depthwise_conv1d_carry(
        g,
        params,
        &format!("{prefix}.k_conv1d"),
        &w.k_conv,
        k3,
        conv_state_k,
        b,
        s,
        proj,
        kk,
    );
    let (vc, ncs_v) = depthwise_conv1d_carry(
        g,
        params,
        &format!("{prefix}.v_conv1d"),
        &w.v_conv,
        v3,
        conv_state_v,
        b,
        s,
        proj,
        kk,
    );
    let qc = g.silu(qc);
    let kc = g.silu(kc);
    let vc = g.silu(vc);

    let qh = g.reshape_(qc, vec![b as i64, s as i64, h as i64, hd as i64]);
    let kh = g.reshape_(kc, vec![b as i64, s as i64, h as i64, hd as i64]);
    let vh = g.reshape_(vc, vec![b as i64, s as i64, h as i64, hd as i64]);
    let q_l2 = l2_norm(g, qh, &bshd, d.eps);
    let k_l2 = l2_norm(g, kh, &bshd, d.eps);

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
    let mut g_log = g.mul(neg_a, sp);
    if let Some(lb) = d.gate_lower_bound {
        let lb_c = scalar_const(g, lb);
        let shifted = g.sub(g_log, lb_c);
        let relu = g.relu(shifted);
        g_log = g.add(lb_c, relu);
    }

    let beta = sigmoid(g, beta, Shape::new(&[rows, h], DType::F32));
    let beta = g.reshape_(beta, vec![b as i64, s as i64, h as i64]);

    // Carried scan: resume from `scan_state`, write it back in place.
    let scan = g.gated_delta_net_carry_pc(
        q_l2,
        k_l2,
        vh,
        g_log,
        beta,
        scan_state,
        hd,
        Shape::new(&bshd, DType::F32),
    );

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
    let o2d = g.reshape_(o, vec![rows as i64, proj as i64]);
    let attn = linear(g, params, prefix, "o_proj", o2d, &w.o_proj, proj, hidden);
    let out = g.reshape_(attn, vec![b as i64, s as i64, hidden as i64]);
    Ok((out, ncs_q, ncs_k, ncs_v))
}
