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

//! `Glm5NextTextLinearAttention` — KDA, a gated delta-net linear attention.
//!
//! 34 of GLM-5.3-Flash's 45 layers. Because the DSA layers are pure NoPE, these
//! layers carry the model's *only* positional information.
//!
//! ```text
//!   q,k,v  = silu(shortconv_k4( {q,k,v}_proj(x) ))       [1,s,h,d]
//!   q,k   ←  l2norm(·)                                    (fused in the kernel)
//!   g_log  = lower_bound · σ(exp(A_log) · (f_b(f_a(x)) + dt_bias))
//!   beta   = σ(b_proj(x))                                 [1,s,h]
//!   o      = GatedDeltaNet(q, k, v, g_log, beta)          per-channel decay
//!   o      = rmsnorm(o)·o_norm.weight · σ(g_b(g_a(x)))    ← gate AFTER the norm
//!   out    = o_proj(o)
//! ```
//!
//! Details that are easy to get wrong:
//!
//! * **The gates read the pre-conv hidden state.** `f`, `beta` and the output
//!   gate all project `hidden_states`, not the convolved q/k/v.
//! * **`A_log` is per *head*** (`[num_heads]`), broadcast across that head's
//!   channels — while `dt_bias` and the gate itself are per *channel*.
//! * **The safe-gate form is a different function**, not a clamp: with
//!   `gate_lower_bound = -5` set (as GLM-5.3-Flash ships) the decay is
//!   `lb·σ(exp(A_log)·g)`, landing in `[lb, 0)` by construction. Only when the
//!   bound is absent does it become `−exp(A_log)·softplus(g)`.
//! * **The gated norm multiplies by `σ(gate)` *after* normalising and scaling**,
//!   so the gate never enters the variance.
//!
//! GGUF splits the reference's single `conv1d` over `3·qkv_dim` channels into
//! `ssm_conv1d_{q,k,v}`. The convolution is depthwise, so per-stream convolution
//! is bit-identical to convolving the concatenation.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::{Activation, Op, PadMode};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::common::linear;

/// Epsilon inside the reference `l2norm` (`1 / sqrt(Σx² + eps)`), independent of
/// `rms_norm_eps`.
const L2NORM_EPS: f32 = 1e-6;

#[derive(Debug, Clone, Copy)]
pub struct KdaDims {
    pub hidden: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub conv_kernel: usize,
    /// `linear_lower_bound` — `Some` selects the safe-gate (sigmoid) form.
    pub lower_bound: Option<f32>,
    pub eps: f32,
    pub seq: usize,
}

impl KdaDims {
    /// `num_heads * head_dim` — the q/k/v projection width.
    pub fn proj(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

fn act(gb: &mut HirMut<'_>, kind: Activation, x: HirNodeId, shape: Shape) -> HirNodeId {
    gb.add_node(Op::Activation(kind), vec![x], shape)
}

/// `x / sqrt(Σ_last x² + eps)`.
fn l2norm(gb: &mut HirMut<'_>, x: HirNodeId, dims: &[usize], eps: HirNodeId) -> HirNodeId {
    let last = dims.len() - 1;
    let mut red = dims.to_vec();
    red[last] = 1;
    let sq = gb.mul(x, x);
    let sumsq = gb.sum(sq, vec![last], true);
    let plus = gb.add(sumsq, eps);
    let denom = act(gb, Activation::Sqrt, plus, Shape::new(&red, DType::F32));
    gb.div(x, denom)
}

/// Causal depthwise conv1d over the sequence axis, channels-last `[1, s, c]`.
///
/// Emitted as `k` shifted slices of the left-padded input times a per-channel
/// tap, which keeps everything contiguous and elementwise — no NCL transposes
/// around `Op::Conv` for what is a 4-tap filter.
fn short_conv(
    emit: &mut Emit<'_>,
    key: &str,
    x: HirNodeId, // [1, s, c]
    seq: usize,
    channels: usize,
    k: usize,
) -> Result<HirNodeId> {
    // GGUF stores the depthwise kernel as `[channels, 1, k]`.
    let w = emit.load_param(key, false)?;
    let mut gb = HirMut::new(emit.hir());
    let w2 = gb.reshape_(w, vec![channels as i64, k as i64]);
    let padded = gb.pad_(x, vec![[0, 0], [k - 1, 0], [0, 0]], PadMode::Constant(0.0));
    let mut acc: Option<HirNodeId> = None;
    for j in 0..k {
        let tap = gb.narrow_(w2, 1, j, 1); // [c, 1]
        let tap = gb.reshape_(tap, vec![1, 1, channels as i64]);
        let slice = gb.narrow_(padded, 1, j, seq); // [1, s, c]
        let term = gb.mul(slice, tap);
        acc = Some(match acc {
            Some(a) => gb.add(a, term),
            None => term,
        });
    }
    Ok(acc.expect("conv kernel >= 1"))
}

/// Causal depthwise conv1d whose left-pad is a **carried state** instead of
/// zeros — the decode counterpart of [`short_conv`].
///
/// `state` is `[1, k-1, c]`: the previous `k-1` tokens' *pre-conv* projections.
/// Returns `(out [1, s, c], next_state [1, k-1, c])`, the next state being the
/// last `k-1` rows of `concat(state, x)`. With `s = 1` this makes the conv O(1)
/// per token instead of re-running the whole prefix.
fn short_conv_carried(
    emit: &mut Emit<'_>,
    key: &str,
    x: HirNodeId,     // [1, s, c]
    state: HirNodeId, // [1, k-1, c]
    seq: usize,
    channels: usize,
    k: usize,
) -> Result<(HirNodeId, HirNodeId)> {
    let w = emit.load_param(key, false)?;
    let mut gb = HirMut::new(emit.hir());
    let w2 = gb.reshape_(w, vec![channels as i64, k as i64]);
    // [1, (k-1)+s, c] — the carried prefix followed by the new tokens.
    let padded = gb.concat_(vec![state, x], 1);
    let mut acc: Option<HirNodeId> = None;
    for j in 0..k {
        let tap = gb.narrow_(w2, 1, j, 1);
        let tap = gb.reshape_(tap, vec![1, 1, channels as i64]);
        let slice = gb.narrow_(padded, 1, j, seq);
        let term = gb.mul(slice, tap);
        acc = Some(match acc {
            Some(a) => gb.add(a, term),
            None => term,
        });
    }
    let next_state = gb.narrow_(padded, 1, seq, k - 1);
    Ok((acc.expect("conv kernel >= 1"), next_state))
}

/// Per-layer recurrent state for [`emit_kda_decode`].
///
/// All four pieces are threaded explicitly: pass them in as graph inputs and
/// read them back out.
///
/// It is tempting to bind `scan` to a persistent param and rely on
/// [`rlx_ir::op::Op::GatedDeltaNet`]'s documented in-place update — CPU, Metal
/// and wgpu do mutate the buffer. **MLX does not**: it substitutes the new state
/// into its evaluation env, which does not survive to the next `run()`, so a
/// param-bound state silently stays at its initial value there. Reading the
/// state node as an output is portable, because after the op that node *is* the
/// updated state on every backend.
#[derive(Debug, Clone, Copy)]
pub struct KdaState {
    /// `[1, conv_kernel - 1, proj]` — previous tokens' pre-conv q projections.
    pub conv_q: HirNodeId,
    pub conv_k: HirNodeId,
    pub conv_v: HirNodeId,
    /// `[1, num_heads, head_dim, head_dim]`.
    pub scan: HirNodeId,
}

/// New conv states produced by one decode step; feed these back next step.
#[derive(Debug, Clone, Copy)]
pub struct KdaStateOut {
    pub conv_q: HirNodeId,
    pub conv_k: HirNodeId,
    pub conv_v: HirNodeId,
}

/// Emit one KDA **decode step**: `d.seq` new tokens, O(1) in the prefix length.
///
/// Identical arithmetic to [`emit_kda_attention`]; the only differences are that
/// the short conv's left-pad comes from `state.conv_*` instead of zeros, and the
/// recurrence resumes from `state.scan` instead of a zero matrix. Running this
/// token by token reproduces the prefill output exactly — see
/// `tests/decode_equivalence.rs`.
pub fn emit_kda_decode(
    emit: &mut Emit<'_>,
    prefix: &str,
    hidden: HirNodeId,
    state: KdaState,
    d: KdaDims,
) -> Result<(HirNodeId, KdaStateOut)> {
    let f = DType::F32;
    let (s, h, hd) = (d.seq, d.num_heads, d.head_dim);
    let proj = d.proj();
    let (si, hi, hdi, pi) = (s as i64, h as i64, hd as i64, proj as i64);
    let bshd = [1, s, h, hd];

    let x2d = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(hidden, vec![si, d.hidden as i64])
    };

    // ── (1) q/k/v projections → carried short conv → silu ──
    let mut streams = Vec::with_capacity(3);
    let mut next_states = Vec::with_capacity(3);
    for (proj_name, conv_name, st) in [
        ("attn_q", "ssm_conv1d_q", state.conv_q),
        ("attn_k", "ssm_conv1d_k", state.conv_k),
        ("attn_v", "ssm_conv1d_v", state.conv_v),
    ] {
        let p = linear(emit, &format!("{prefix}.{proj_name}.weight"), x2d)?;
        let p3 = {
            let mut gb = HirMut::new(emit.hir());
            gb.reshape_(p, vec![1, si, pi])
        };
        let (c, next) = short_conv_carried(
            emit,
            &format!("{prefix}.{conv_name}.weight"),
            p3,
            st,
            s,
            proj,
            d.conv_kernel,
        )?;
        next_states.push(next);
        let mut gb = HirMut::new(emit.hir());
        let c = gb.silu(c);
        streams.push(gb.reshape_(c, vec![1, si, hi, hdi]));
    }
    let (qh, kh, vh) = (streams[0], streams[1], streams[2]);

    // ── (2) per-channel log-decay gate (identical to prefill) ──
    let f_a = linear(emit, &format!("{prefix}.ssm_f_a.weight"), x2d)?;
    let f_raw = linear(emit, &format!("{prefix}.ssm_f_b.weight"), f_a)?;
    let a_log = emit.load_param(&format!("{prefix}.ssm_a"), false)?;
    let dt_bias = emit.load_param(&format!("{prefix}.ssm_dt.bias"), false)?;
    let eps_l2 = emit.synth_param(
        &format!("{prefix}.kda.dec.l2eps"),
        vec![L2NORM_EPS],
        Shape::new(&[1], f),
    );
    let lb_const = d.lower_bound.map(|lb| {
        emit.synth_param(
            &format!("{prefix}.kda.dec.gate_lb"),
            vec![lb],
            Shape::new(&[1], f),
        )
    });

    let (q_l2, k_l2, g_log) = {
        let mut gb = HirMut::new(emit.hir());
        let q_l2 = l2norm(&mut gb, qh, &bshd, eps_l2);
        let k_l2 = l2norm(&mut gb, kh, &bshd, eps_l2);

        let fg = gb.reshape_(f_raw, vec![1, si, hi, hdi]);
        let dt = gb.reshape_(dt_bias, vec![1, 1, hi, hdi]);
        let biased = gb.add(fg, dt);
        let a4 = gb.reshape_(a_log, vec![1, 1, hi, 1]);
        let a_exp = act(&mut gb, Activation::Exp, a4, Shape::new(&[1, 1, h, 1], f));
        let g_log = match lb_const {
            Some(lb) => {
                let scaled = gb.mul(biased, a_exp);
                let sig = act(&mut gb, Activation::Sigmoid, scaled, Shape::new(&bshd, f));
                gb.mul(sig, lb)
            }
            None => {
                let sp = act(&mut gb, Activation::Softplus, biased, Shape::new(&bshd, f));
                let neg = act(
                    &mut gb,
                    Activation::Neg,
                    a_exp,
                    Shape::new(&[1, 1, h, 1], f),
                );
                gb.mul(sp, neg)
            }
        };
        (q_l2, k_l2, g_log)
    };

    // ── (3) beta, then the recurrence resumed from the carried state ──
    let beta = linear(emit, &format!("{prefix}.ssm_beta.weight"), x2d)?;
    let scan = {
        let mut gb = HirMut::new(emit.hir());
        let beta = act(&mut gb, Activation::Sigmoid, beta, Shape::new(&[s, h], f));
        let beta = gb.reshape_(beta, vec![1, si, hi]);
        gb.gated_delta_net_carry_pc(
            q_l2,
            k_l2,
            vh,
            g_log,
            beta,
            state.scan,
            hd,
            Shape::new(&bshd, f),
        )
    };

    // ── (4) gated output norm + o_proj (identical to prefill) ──
    let g_a = linear(emit, &format!("{prefix}.ssm_g_a.weight"), x2d)?;
    let g_raw = linear(emit, &format!("{prefix}.ssm_g_b.weight"), g_a)?;
    let o_norm_w = emit.load_param(&format!("{prefix}.ssm_norm.weight"), false)?;
    let zb = emit.synth_zeros(&format!("{prefix}.ssm_norm.dzb"), hd);
    let o = {
        let mut gb = HirMut::new(emit.hir());
        let normed = gb.rms_norm(scan, o_norm_w, zb, d.eps);
        let g4 = gb.reshape_(g_raw, vec![1, si, hi, hdi]);
        let g_sig = act(&mut gb, Activation::Sigmoid, g4, Shape::new(&bshd, f));
        let gated = gb.mul(normed, g_sig);
        gb.reshape_(gated, vec![si, pi])
    };
    let out = linear(emit, &format!("{prefix}.attn_output.weight"), o)?;
    let mut gb = HirMut::new(emit.hir());
    let out = gb.reshape_(out, vec![1, si, d.hidden as i64]);
    Ok((
        out,
        KdaStateOut {
            conv_q: next_states[0],
            conv_k: next_states[1],
            conv_v: next_states[2],
        },
    ))
}

/// Emit one KDA layer for GGUF block `prefix` (`blk.{i}`) over `[1, seq, hidden]`.
pub fn emit_kda_attention(
    emit: &mut Emit<'_>,
    prefix: &str,
    hidden: HirNodeId,
    d: KdaDims,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let (s, h, hd) = (d.seq, d.num_heads, d.head_dim);
    let proj = d.proj();
    let (si, hi, hdi, pi) = (s as i64, h as i64, hd as i64, proj as i64);
    let bshd = [1, s, h, hd];

    let x2d = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(hidden, vec![si, d.hidden as i64])
    };

    // ── (1) q/k/v projections → causal short conv → silu ──
    let mut streams = Vec::with_capacity(3);
    for (proj_name, conv_name) in [
        ("attn_q", "ssm_conv1d_q"),
        ("attn_k", "ssm_conv1d_k"),
        ("attn_v", "ssm_conv1d_v"),
    ] {
        let p = linear(emit, &format!("{prefix}.{proj_name}.weight"), x2d)?;
        let p3 = {
            let mut gb = HirMut::new(emit.hir());
            gb.reshape_(p, vec![1, si, pi])
        };
        let c = short_conv(
            emit,
            &format!("{prefix}.{conv_name}.weight"),
            p3,
            s,
            proj,
            d.conv_kernel,
        )?;
        let mut gb = HirMut::new(emit.hir());
        let c = gb.silu(c);
        streams.push(gb.reshape_(c, vec![1, si, hi, hdi]));
    }
    let (qh, kh, vh) = (streams[0], streams[1], streams[2]);

    // ── (2) per-channel log-decay gate, from the *pre-conv* hidden state ──
    let f_a = linear(emit, &format!("{prefix}.ssm_f_a.weight"), x2d)?;
    let f_raw = linear(emit, &format!("{prefix}.ssm_f_b.weight"), f_a)?;
    let a_log = emit.load_param(&format!("{prefix}.ssm_a"), false)?;
    let dt_bias = emit.load_param(&format!("{prefix}.ssm_dt.bias"), false)?;
    let eps_l2 = emit.synth_param(
        &format!("{prefix}.kda.l2eps"),
        vec![L2NORM_EPS],
        Shape::new(&[1], f),
    );
    let lb_const = d.lower_bound.map(|lb| {
        emit.synth_param(
            &format!("{prefix}.kda.gate_lb"),
            vec![lb],
            Shape::new(&[1], f),
        )
    });

    let (q_l2, k_l2, g_log) = {
        let mut gb = HirMut::new(emit.hir());
        let q_l2 = l2norm(&mut gb, qh, &bshd, eps_l2);
        let k_l2 = l2norm(&mut gb, kh, &bshd, eps_l2);

        let fg = gb.reshape_(f_raw, vec![1, si, hi, hdi]);
        let dt = gb.reshape_(dt_bias, vec![1, 1, hi, hdi]);
        let biased = gb.add(fg, dt);
        // A_log is per head → [1,1,h,1], broadcast over that head's channels.
        let a4 = gb.reshape_(a_log, vec![1, 1, hi, 1]);
        let a_exp = act(&mut gb, Activation::Exp, a4, Shape::new(&[1, 1, h, 1], f));
        let g_log = match lb_const {
            // lower_bound · σ(exp(A_log) · (f + dt_bias)) ∈ [lower_bound, 0)
            Some(lb) => {
                let scaled = gb.mul(biased, a_exp);
                let sig = act(&mut gb, Activation::Sigmoid, scaled, Shape::new(&bshd, f));
                gb.mul(sig, lb)
            }
            // −exp(A_log) · softplus(f + dt_bias)
            None => {
                let sp = act(&mut gb, Activation::Softplus, biased, Shape::new(&bshd, f));
                let neg = act(
                    &mut gb,
                    Activation::Neg,
                    a_exp,
                    Shape::new(&[1, 1, h, 1], f),
                );
                gb.mul(sp, neg)
            }
        };
        (q_l2, k_l2, g_log)
    };

    // ── (3) beta = σ(b_proj(x)), then the delta-net recurrence ──
    let beta = linear(emit, &format!("{prefix}.ssm_beta.weight"), x2d)?; // [s, h]
    let scan = {
        let mut gb = HirMut::new(emit.hir());
        let beta = act(&mut gb, Activation::Sigmoid, beta, Shape::new(&[s, h], f));
        let beta = gb.reshape_(beta, vec![1, si, hi]);
        gb.gated_delta_net_pc(q_l2, k_l2, vh, g_log, beta, hd, Shape::new(&bshd, f))
    };

    // ── (4) gated RMSNorm (normalise, scale, *then* gate) + o_proj ──
    let g_a = linear(emit, &format!("{prefix}.ssm_g_a.weight"), x2d)?;
    let g_raw = linear(emit, &format!("{prefix}.ssm_g_b.weight"), g_a)?;
    let o_norm_w = emit.load_param(&format!("{prefix}.ssm_norm.weight"), false)?;
    let zb = emit.synth_zeros(&format!("{prefix}.ssm_norm.zb"), hd);
    let o = {
        let mut gb = HirMut::new(emit.hir());
        let normed = gb.rms_norm(scan, o_norm_w, zb, d.eps);
        let g4 = gb.reshape_(g_raw, vec![1, si, hi, hdi]);
        let g_sig = act(&mut gb, Activation::Sigmoid, g4, Shape::new(&bshd, f));
        let gated = gb.mul(normed, g_sig);
        gb.reshape_(gated, vec![si, pi])
    };
    let out = linear(emit, &format!("{prefix}.attn_output.weight"), o)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.reshape_(out, vec![1, si, d.hidden as i64]))
}
