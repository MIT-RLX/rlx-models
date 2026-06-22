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

//! Shared HIR builders for 1-D neural audio codecs (SEANet / HiFi-GAN conv
//! stacks + RVQ + transformer bottlenecks), so every codec crate (dac, mimi,
//! encodec, snac, …) reuses the *same* backend-portable graph primitives
//! instead of copy-pasting them.
//!
//! Backend-portability rules baked in here (learned the hard way porting
//! rlx-dac / rlx-mimi):
//!   * A `[C, T]` activation is carried as NCHW `[1, C, T, 1]` — **length in H**,
//!     singleton W, `[k, 1]` kernel. MLX reads the length-axis conv params from
//!     index 0, so the length MUST sit in H.
//!   * Padding is **explicit** (concat), never the conv's own `padding` arg —
//!     backends disagree on padding the singleton layout.
//!   * Transposed conv is **decomposed** to zero-insert + regular conv —
//!     wgpu / coreml ship no native `ConvTranspose` kernel.
//!
//! All helpers are dependency-light: weights are `&[f32]` (row-major) + explicit
//! dims, outputs are `Vec<f32>` + dims. Codec crates wrap those in `ndarray`.

use anyhow::{Context, Result};
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::op::{Activation, Op};
use rlx_ir::{DType, Graph, HirGraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};

const F32: DType = DType::F32;

/// How to fill the padded region of a 1-D conv input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode {
    /// Zero padding.
    Constant,
    /// Edge-replicate padding (repeat the first/last column).
    Replicate,
    /// Reflect padding (mirror around the edge, excluding it) — EnCodec's mode.
    /// Requires `pad < length` on each side.
    Reflect,
}

/// Symmetric "same length" padding for a stride-1 conv: `dilation*(k-1)/2`.
pub fn same_pad(k: usize, dilation: usize) -> usize {
    (dilation * (k - 1)) / 2
}

/// Causal padding (left-heavy) matching HF `MimiConv1d` / EnCodec causal mode.
/// Returns `(pad_left, pad_right)` for input length `t_in`.
pub fn causal_pad(t_in: usize, k: usize, stride: usize, dilation: usize) -> (usize, usize) {
    let eff_k = (k - 1) * dilation + 1;
    let padding_total = eff_k.saturating_sub(stride);
    let num = t_in as i64 - eff_k as i64 + padding_total as i64;
    let n_frames = if num <= 0 {
        0
    } else {
        (num as usize).div_ceil(stride)
    };
    let ideal_len = n_frames * stride + eff_k - padding_total;
    (padding_total, ideal_len.saturating_sub(t_in))
}

/// Non-causal (symmetric-ish) padding matching EnCodec/SpeechTokenizer non-causal
/// `SConv1d`: `pad_right = padding_total/2 + extra`, `pad_left = padding_total -
/// padding_total/2`. Returns `(pad_left, pad_right)`.
pub fn noncausal_pad(t_in: usize, k: usize, stride: usize, dilation: usize) -> (usize, usize) {
    let (padding_total, extra) = causal_pad(t_in, k, stride, dilation);
    let pad_right = padding_total / 2 + extra;
    let pad_left = padding_total - padding_total / 2;
    (pad_left, pad_right)
}

/// A 1-D convolution spec. Weight is `[c_out, c_in/groups, k]` row-major.
pub struct Conv1d<'w> {
    pub weight: &'w [f32],
    pub bias: Option<&'w [f32]>,
    pub c_out: usize,
    pub c_in: usize,
    pub k: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
    pub pad_left: usize,
    pub pad_right: usize,
    pub pad_mode: PadMode,
}

impl<'w> Conv1d<'w> {
    /// Stride-1 same-length conv (symmetric zero pad), no bias by default.
    pub fn same(weight: &'w [f32], c_out: usize, c_in: usize, k: usize, dilation: usize) -> Self {
        let p = same_pad(k, dilation);
        Self {
            weight,
            bias: None,
            c_out,
            c_in,
            k,
            stride: 1,
            dilation,
            groups: 1,
            pad_left: p,
            pad_right: p,
            pad_mode: PadMode::Constant,
        }
    }

    pub fn with_bias(mut self, bias: Option<&'w [f32]>) -> Self {
        self.bias = bias;
        self
    }
}

/// A 1-D transposed convolution spec. Weight is `[c_in, c_out/groups, k]`
/// row-major (PyTorch `ConvTranspose1d` layout). `trim_left`/`trim_right` are
/// the columns cropped from the full output on each side.
pub struct ConvTranspose1d<'w> {
    pub weight: &'w [f32],
    pub bias: Option<&'w [f32]>,
    pub c_in: usize,
    pub c_out_per_group: usize,
    pub k: usize,
    pub stride: usize,
    pub groups: usize,
    pub trim_left: usize,
    pub trim_right: usize,
}

/// Multi-head self-attention spec. q/k/v/o weights are `[c, c]` (`[out, in]`),
/// `c = num_heads * head_dim`.
pub struct Attention<'w> {
    pub q_w: &'w [f32],
    pub k_w: &'w [f32],
    pub v_w: &'w [f32],
    pub o_w: &'w [f32],
    pub num_heads: usize,
    pub head_dim: usize,
    pub scaling: f32,
}

/// HIR graph builder for audio codecs. Accumulates `(name, data)` params while
/// you build; hand them to [`compile`] afterwards.
pub struct AudioGraph<'a, 'b> {
    /// The underlying HIR builder (use `.input`/`.add_node`/etc. directly when needed).
    pub g: &'a mut HirMut<'b>,
    /// Collected params, keyed by auto-generated names. Move out with
    /// `std::mem::take(&mut ag.params)` before `set_outputs`.
    pub params: Vec<(String, Vec<f32>)>,
    next: usize,
}

impl<'a, 'b> AudioGraph<'a, 'b> {
    pub fn new(g: &'a mut HirMut<'b>) -> Self {
        Self {
            g,
            params: Vec::new(),
            next: 0,
        }
    }

    /// Declare a runtime input.
    pub fn input(&mut self, name: &str, shape: &[usize]) -> HirNodeId {
        self.g.input(name.to_string(), Shape::new(shape, F32))
    }

    /// Declare a weight param (auto-named) and stash its data.
    pub fn param(&mut self, data: Vec<f32>, shape: &[usize]) -> HirNodeId {
        let name = format!("w{}", self.next);
        self.next += 1;
        debug_assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "param {name} shape {shape:?} vs len {}",
            data.len()
        );
        let id = self.g.param(name.clone(), Shape::new(shape, F32));
        self.params.push((name, data));
        id
    }

    /// Broadcast a tensor to `target` shape via `Op::Expand`.
    pub fn expand(&mut self, id: HirNodeId, target: &[usize]) -> HirNodeId {
        let shape = Shape::new(target, F32);
        let tgt: Vec<i64> = target.iter().map(|&d| d as i64).collect();
        self.g
            .add_node(Op::Expand { target_shape: tgt }, vec![id], shape)
    }

    /// A constant scalar broadcast to `target` shape.
    pub fn scalar(&mut self, v: f32, target: &[usize]) -> HirNodeId {
        let ones = vec![1usize; target.len()];
        let s = self.param(vec![v], &ones);
        self.expand(s, target)
    }

    fn add_bias_cl(&mut self, y: HirNodeId, bias: &[f32], c: usize, t: usize) -> HirNodeId {
        let b = self.param(bias.to_vec(), &[1, c, 1, 1]);
        let be = self.expand(b, &[1, c, t, 1]);
        self.g.add(y, be)
    }

    /// Asymmetric pad on the length (H) axis, zero or edge-replicate.
    pub fn pad_len(
        &mut self,
        x: HirNodeId,
        c: usize,
        t: usize,
        pl: usize,
        pr: usize,
        mode: PadMode,
    ) -> (HirNodeId, usize) {
        if pl == 0 && pr == 0 {
            return (x, t);
        }
        let mut parts: Vec<HirNodeId> = Vec::new();
        match mode {
            PadMode::Constant => {
                if pl > 0 {
                    parts.push(self.param(vec![0.0; c * pl], &[1, c, pl, 1]));
                }
                parts.push(x);
                if pr > 0 {
                    parts.push(self.param(vec![0.0; c * pr], &[1, c, pr, 1]));
                }
            }
            PadMode::Replicate => {
                if pl > 0 {
                    let first = self.g.narrow_(x, 2, 0, 1);
                    parts.push(self.expand(first, &[1, c, pl, 1]));
                }
                parts.push(x);
                if pr > 0 {
                    let last = self.g.narrow_(x, 2, t - 1, 1);
                    parts.push(self.expand(last, &[1, c, pr, 1]));
                }
            }
            PadMode::Reflect => {
                // left: columns [pl, pl-1, ..., 1]; right: [T-2, T-3, ..., T-1-pr]
                for i in (1..=pl).rev() {
                    parts.push(self.g.narrow_(x, 2, i, 1));
                }
                parts.push(x);
                for i in 1..=pr {
                    parts.push(self.g.narrow_(x, 2, t - 1 - i, 1));
                }
            }
        }
        (self.g.concat_(parts, 2), t + pl + pr)
    }

    /// Zero-insert `s-1` samples between length steps: `[1,C,T,1]` → `[1,C,(T-1)s+1,1]`.
    pub fn inflate_len(
        &mut self,
        x: HirNodeId,
        c: usize,
        t: usize,
        s: usize,
    ) -> (HirNodeId, usize) {
        if s <= 1 {
            return (x, t);
        }
        let z = self.param(vec![0.0; c * t * (s - 1)], &[1, c, t, s - 1]);
        let cat = self.g.concat_(vec![x, z], 3);
        let resh = self.g.reshape_(cat, vec![1, c as i64, (t * s) as i64, 1]);
        let lu = (t - 1) * s + 1;
        (self.g.narrow_(resh, 2, 0, lu), lu)
    }

    fn conv_node(
        &mut self,
        x: HirNodeId,
        w: HirNodeId,
        c_out: usize,
        t_out: usize,
        k: usize,
        stride: usize,
        dil: usize,
        groups: usize,
    ) -> HirNodeId {
        self.g.add_node(
            Op::Conv {
                kernel_size: vec![k, 1],
                stride: vec![stride, 1],
                padding: vec![0, 0],
                dilation: vec![dil, 1],
                groups,
            },
            vec![x, w],
            Shape::new(&[1, c_out, t_out, 1], F32),
        )
    }

    /// Emit a 1-D conv on a `[1, c_in, x_t, 1]` tensor. Returns `(node, c_out, t_out)`.
    pub fn conv1d(&mut self, x: HirNodeId, x_t: usize, spec: &Conv1d) -> (HirNodeId, usize, usize) {
        let (xp, t_pad) = self.pad_len(
            x,
            spec.c_in,
            x_t,
            spec.pad_left,
            spec.pad_right,
            spec.pad_mode,
        );
        let dil = spec.dilation.max(1);
        let t_out = (t_pad.saturating_sub(dil * (spec.k - 1) + 1)) / spec.stride + 1;
        let w = self.param(
            spec.weight.to_vec(),
            &[spec.c_out, spec.c_in / spec.groups, spec.k, 1],
        );
        let mut y = self.conv_node(
            xp,
            w,
            spec.c_out,
            t_out,
            spec.k,
            spec.stride,
            dil,
            spec.groups,
        );
        if let Some(b) = spec.bias {
            y = self.add_bias_cl(y, b, spec.c_out, t_out);
        }
        (y, spec.c_out, t_out)
    }

    /// Emit a 1-D transposed conv as zero-insert + regular conv (bit-exact,
    /// runs on every backend). Returns `(node, c_out, t_out)`.
    pub fn conv_transpose1d(
        &mut self,
        x: HirNodeId,
        x_t: usize,
        spec: &ConvTranspose1d,
    ) -> (HirNodeId, usize, usize) {
        let (in_ch, c_out_per_g, k, groups) =
            (spec.c_in, spec.c_out_per_group, spec.k, spec.groups);
        let c_out = c_out_per_g * groups;
        let c_in_per_g = in_ch / groups;

        // Flip kernel + swap in/out per group: W[oc, ic_per_g, j] = weight[ic, oc_per_g, k-1-j]
        let mut wflip = vec![0f32; c_out * c_in_per_g * k];
        for grp in 0..groups {
            for ocp in 0..c_out_per_g {
                let oc = grp * c_out_per_g + ocp;
                for icp in 0..c_in_per_g {
                    let ic = grp * c_in_per_g + icp;
                    for j in 0..k {
                        wflip[(oc * c_in_per_g + icp) * k + j] =
                            spec.weight[(ic * c_out_per_g + ocp) * k + (k - 1 - j)];
                    }
                }
            }
        }

        let (u, lu) = self.inflate_len(x, in_ch, x_t, spec.stride);
        let (up, t_pad) = self.pad_len(u, in_ch, lu, k - 1, k - 1, PadMode::Constant);
        let t_raw = t_pad - (k - 1);
        let w = self.param(wflip, &[c_out, c_in_per_g, k, 1]);
        let mut y = self.conv_node(up, w, c_out, t_raw, k, 1, 1, groups);

        let t_out = t_raw - spec.trim_left - spec.trim_right;
        if spec.trim_left > 0 || spec.trim_right > 0 {
            y = self.g.narrow_(y, 2, spec.trim_left, t_out);
        }
        if let Some(b) = spec.bias {
            y = self.add_bias_cl(y, b, c_out, t_out);
        }
        (y, c_out, t_out)
    }

    /// ELU on a `[1,C,T,1]` tensor: `x>=0 ? x : exp(x)-1`.
    pub fn elu(&mut self, x: HirNodeId, c: usize, t: usize) -> HirNodeId {
        let shape = Shape::new(&[1, c, t, 1], F32);
        let r = self.g.activation(Activation::Relu, x, shape.clone());
        let negx = self.g.activation(Activation::Neg, x, shape.clone());
        let relu_neg = self.g.activation(Activation::Relu, negx, shape.clone());
        let minx = self.g.activation(Activation::Neg, relu_neg, shape.clone());
        let e = self.g.activation(Activation::Exp, minx, shape);
        let re = self.g.add(r, e);
        let ones = self.scalar(1.0, &[1, c, t, 1]);
        self.g.sub(re, ones)
    }

    /// Snake activation: `x + sin(α·x)² / (α + 1e-9)`, per channel.
    pub fn snake(&mut self, x: HirNodeId, c: usize, t: usize, alpha: &[f32]) -> HirNodeId {
        let inv: Vec<f32> = alpha.iter().map(|&a| 1.0 / (a + 1e-9)).collect();
        let a = self.param(alpha.to_vec(), &[1, c, 1, 1]);
        let ia = self.param(inv, &[1, c, 1, 1]);
        let a_e = self.expand(a, &[1, c, t, 1]);
        let ia_e = self.expand(ia, &[1, c, t, 1]);
        let full = Shape::new(&[1, c, t, 1], F32);
        let ax = self.g.mul(x, a_e);
        let s = self.g.activation(Activation::Sin, ax, full);
        let s2 = self.g.mul(s, s);
        let scaled = self.g.mul(s2, ia_e);
        self.g.add(x, scaled)
    }

    /// SnakeBeta (BigVGAN, `alpha_logscale=True`): `x + sin(eᵃ·x)² / (eᵇ + 1e-9)`,
    /// per channel. `alpha_raw`/`beta_raw` are the stored (pre-exp) parameters.
    pub fn snake_beta(
        &mut self,
        x: HirNodeId,
        c: usize,
        t: usize,
        alpha_raw: &[f32],
        beta_raw: &[f32],
    ) -> HirNodeId {
        let a_exp: Vec<f32> = alpha_raw.iter().map(|v| v.exp()).collect();
        let inv_b: Vec<f32> = beta_raw.iter().map(|v| 1.0 / (v.exp() + 1e-9)).collect();
        let a = self.param(a_exp, &[1, c, 1, 1]);
        let ib = self.param(inv_b, &[1, c, 1, 1]);
        let a_e = self.expand(a, &[1, c, t, 1]);
        let ib_e = self.expand(ib, &[1, c, t, 1]);
        let full = Shape::new(&[1, c, t, 1], F32);
        let ax = self.g.mul(x, a_e);
        let s = self.g.activation(Activation::Sin, ax, full);
        let s2 = self.g.mul(s, s);
        let scaled = self.g.mul(s2, ib_e);
        self.g.add(x, scaled)
    }

    /// HalfSnake (NeMo): snake on the first `c/2` channels, LeakyReLU(0.01) on the
    /// rest, concatenated. `alpha` has `c/2` entries.
    pub fn half_snake(&mut self, x: HirNodeId, c: usize, t: usize, alpha: &[f32]) -> HirNodeId {
        let h = c / 2;
        let xs = self.g.narrow_(x, 1, 0, h);
        let xl = self.g.narrow_(x, 1, h, c - h);
        let s = self.snake(xs, h, t, alpha);
        let l = self.leaky_relu(xl, c - h, t, 0.01);
        self.g.concat_(vec![s, l], 1)
    }

    /// Clamp to `[lo, hi]` elementwise: `min(max(x, lo), hi)`.
    pub fn clamp(&mut self, x: HirNodeId, c: usize, t: usize, lo: f32, hi: f32) -> HirNodeId {
        let shape = Shape::new(&[1, c, t, 1], F32);
        let lo_s = self.scalar(lo, &[1, c, t, 1]);
        let hi_s = self.scalar(hi, &[1, c, t, 1]);
        let xml = self.g.sub(x, lo_s);
        let r = self.g.activation(Activation::Relu, xml, shape.clone());
        let lo_s2 = self.scalar(lo, &[1, c, t, 1]);
        let mx = self.g.add(r, lo_s2); // max(x, lo)
        let hi_s2 = self.scalar(hi, &[1, c, t, 1]);
        let hmm = self.g.sub(hi_s, mx);
        let r2 = self.g.activation(Activation::Relu, hmm, shape);
        self.g.sub(hi_s2, r2) // hi - relu(hi - max(x,lo)) = min(.., hi)
    }

    pub fn tanh(&mut self, x: HirNodeId) -> HirNodeId {
        self.g.tanh(x)
    }

    /// Elementwise multiply (broadcasting per the IR's rules).
    pub fn mul(&mut self, a: HirNodeId, b: HirNodeId) -> HirNodeId {
        self.g.mul(a, b)
    }

    /// Elementwise add.
    pub fn add(&mut self, a: HirNodeId, b: HirNodeId) -> HirNodeId {
        self.g.add(a, b)
    }

    /// Broadcast a `[1, 1, t, 1]` tensor to `[1, c, t, 1]` (over channels).
    pub fn broadcast_to_channels(&mut self, x: HirNodeId, c: usize, t: usize) -> HirNodeId {
        self.expand(x, &[1, c, t, 1])
    }

    pub fn gelu(&mut self, x: HirNodeId) -> HirNodeId {
        self.g.gelu(x)
    }

    /// LeakyReLU(x, slope) = relu(x) - slope*relu(-x), for HiFi-GAN-style decoders.
    pub fn leaky_relu(&mut self, x: HirNodeId, c: usize, t: usize, slope: f32) -> HirNodeId {
        let shape = Shape::new(&[1, c, t, 1], F32);
        let r = self.g.activation(Activation::Relu, x, shape.clone());
        let negx = self.g.activation(Activation::Neg, x, shape.clone());
        let relu_neg = self.g.activation(Activation::Relu, negx, shape);
        let sc = self.scalar(slope, &[1, c, t, 1]);
        let neg_part = self.g.mul(relu_neg, sc);
        self.g.sub(r, neg_part)
    }

    /// `[1, C, T, 1]` → `[T, C]` (channel-major → time-major).
    pub fn ct_to_tc(&mut self, x: HirNodeId, c: usize, t: usize) -> HirNodeId {
        let r = self.g.reshape_(x, vec![c as i64, t as i64]);
        self.g.transpose_(r, vec![1, 0])
    }

    /// `[T, C]` → `[1, C, T, 1]`.
    pub fn tc_to_ct(&mut self, x: HirNodeId, t: usize, c: usize) -> HirNodeId {
        let r = self.g.transpose_(x, vec![1, 0]);
        self.g.reshape_(r, vec![1, c as i64, t as i64, 1])
    }

    /// GroupNorm on a `[1, C, T, 1]` NCHW tensor (gamma/beta are `[C]`).
    pub fn group_norm(
        &mut self,
        x: HirNodeId,
        c: usize,
        gamma: &[f32],
        beta: &[f32],
        num_groups: usize,
        eps: f32,
    ) -> HirNodeId {
        let g = self.param(gamma.to_vec(), &[c]);
        let b = self.param(beta.to_vec(), &[c]);
        self.g.group_norm(x, g, b, num_groups, eps)
    }

    /// SiLU activation.
    pub fn silu(&mut self, x: HirNodeId) -> HirNodeId {
        self.g.silu(x)
    }

    /// LayerNorm over the last axis of a `[T, C]` tensor.
    pub fn layer_norm(
        &mut self,
        x: HirNodeId,
        c: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> HirNodeId {
        let g = self.param(gamma.to_vec(), &[c]);
        let b = self.param(beta.to_vec(), &[c]);
        self.g.ln(x, g, b, eps)
    }

    /// RMSNorm over the last axis of a `[T, C]` tensor (weight-only, no bias).
    pub fn rms_norm(&mut self, x: HirNodeId, c: usize, gamma: &[f32], eps: f32) -> HirNodeId {
        let g = self.param(gamma.to_vec(), &[c]);
        let b = self.param(vec![0.0; c], &[c]);
        self.g.rms_norm(x, g, b, eps)
    }

    /// `x[rows, in] @ Wᵀ` where `w_oi` is `[out, in]` (torch Linear weight, no bias).
    pub fn linear(&mut self, x: HirNodeId, in_d: usize, out_d: usize, w_oi: &[f32]) -> HirNodeId {
        let mut wt = vec![0f32; in_d * out_d];
        for o in 0..out_d {
            for i in 0..in_d {
                wt[i * out_d + o] = w_oi[o * in_d + i];
            }
        }
        let wp = self.param(wt, &[in_d, out_d]);
        self.g.mm(x, wp)
    }

    /// `x[rows, in] @ Wᵀ + b`. `rows` is needed to broadcast the bias.
    pub fn linear_bias(
        &mut self,
        x: HirNodeId,
        rows: usize,
        in_d: usize,
        out_d: usize,
        w_oi: &[f32],
        bias: &[f32],
    ) -> HirNodeId {
        let y = self.linear(x, in_d, out_d, w_oi);
        let b = self.param(bias.to_vec(), &[1, out_d]);
        let be = self.expand(b, &[rows, out_d]);
        self.g.add(y, be)
    }

    /// Per-channel scale of a `[rows, c]` tensor by `[c]`.
    pub fn scale_rows(&mut self, x: HirNodeId, rows: usize, c: usize, scale: &[f32]) -> HirNodeId {
        let s = self.param(scale.to_vec(), &[1, c]);
        let se = self.expand(s, &[rows, c]);
        self.g.mul(x, se)
    }

    /// Raw access to reshape/transpose/matmul/softmax for custom blocks.
    pub fn reshape(&mut self, x: HirNodeId, shape: Vec<i64>) -> HirNodeId {
        self.g.reshape_(x, shape)
    }
    pub fn transpose(&mut self, x: HirNodeId, perm: Vec<usize>) -> HirNodeId {
        self.g.transpose_(x, perm)
    }
    pub fn matmul(&mut self, a: HirNodeId, b: HirNodeId) -> HirNodeId {
        self.g.mm(a, b)
    }
    pub fn softmax(&mut self, x: HirNodeId, axis: i32) -> HirNodeId {
        self.g.sm(x, axis)
    }

    /// Multi-head self-attention on a `[T, C]` tensor. `cos`/`sin` (each `[T,
    /// head_dim]`) enable rotate-half RoPE; `mask` (`[T, T]`, additive) enables
    /// causal/sliding masking.
    pub fn attention(
        &mut self,
        x: HirNodeId,
        t: usize,
        spec: &Attention,
        cos: Option<&[f32]>,
        sin: Option<&[f32]>,
        mask: Option<&[f32]>,
    ) -> HirNodeId {
        let h = spec.num_heads;
        let d = spec.head_dim;
        let c = h * d;
        let q = self.linear(x, c, c, spec.q_w);
        let k = self.linear(x, c, c, spec.k_w);
        let v = self.linear(x, c, c, spec.v_w);

        let to_htd = |this: &mut Self, n: HirNodeId| {
            let r = this.g.reshape_(n, vec![t as i64, h as i64, d as i64]);
            this.g.transpose_(r, vec![1, 0, 2])
        };
        let mut q = to_htd(self, q);
        let mut k = to_htd(self, k);
        let v = to_htd(self, v);

        if let (Some(cos), Some(sin)) = (cos, sin) {
            let cos_p = self.param(cos.to_vec(), &[1, t, d]);
            let sin_p = self.param(sin.to_vec(), &[1, t, d]);
            q = self.rope(q, h, t, d, cos_p, sin_p);
            k = self.rope(k, h, t, d, cos_p, sin_p);
        }

        let kt = self.g.transpose_(k, vec![0, 2, 1]);
        let scores = self.g.mm(q, kt);
        let sc = self.scalar(spec.scaling, &[h, t, t]);
        let mut scores = self.g.mul(scores, sc);
        if let Some(mask) = mask {
            let mp = self.param(mask.to_vec(), &[1, t, t]);
            let me = self.expand(mp, &[h, t, t]);
            scores = self.g.add(scores, me);
        }
        let probs = self.g.sm(scores, -1);
        let out = self.g.mm(probs, v);
        let out = self.g.transpose_(out, vec![1, 0, 2]);
        let out = self.g.reshape_(out, vec![t as i64, c as i64]);
        self.linear(out, c, c, spec.o_w)
    }

    fn rope(
        &mut self,
        x: HirNodeId,
        h: usize,
        t: usize,
        d: usize,
        cos_p: HirNodeId,
        sin_p: HirNodeId,
    ) -> HirNodeId {
        let half = d / 2;
        let cos_e = self.expand(cos_p, &[h, t, d]);
        let sin_e = self.expand(sin_p, &[h, t, d]);
        let a = self.g.narrow_(x, 2, 0, half);
        let b = self.g.narrow_(x, 2, half, half);
        let nb = self
            .g
            .activation(Activation::Neg, b, Shape::new(&[h, t, half], F32));
        let rot = self.g.concat_(vec![nb, a], 2);
        let xc = self.g.mul(x, cos_e);
        let rs = self.g.mul(rot, sin_e);
        self.g.add(xc, rs)
    }
}

/// Rotate-half RoPE cos/sin tables `[t, head_dim]` from `inv_freq` (`head_dim/2`).
pub fn rope_tables(inv_freq: &[f32], t: usize) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let d = half * 2;
    let mut cos = vec![0f32; t * d];
    let mut sin = vec![0f32; t * d];
    for ti in 0..t {
        for hi in 0..half {
            let f = ti as f32 * inv_freq[hi];
            let (s, c) = f.sin_cos();
            cos[ti * d + hi] = c;
            cos[ti * d + hi + half] = c;
            sin[ti * d + hi] = s;
            sin[ti * d + hi + half] = s;
        }
    }
    (cos, sin)
}

/// Additive sliding-window causal mask `[t, t]` (0 where attended, -inf else).
pub fn sliding_causal_mask(t: usize, window: usize) -> Vec<f32> {
    let mut m = vec![f32::NEG_INFINITY; t * t];
    for i in 0..t {
        let lo = i.saturating_sub(window.saturating_sub(1));
        for j in lo..=i {
            m[i * t + j] = 0.0;
        }
    }
    m
}

/// A compiled codec graph plus its `[out_c, out_t]` output dims.
pub struct CompiledAudioGraph {
    compiled: CompiledGraph,
    out_c: usize,
    out_t: usize,
    input_name: String,
}

/// Compile a built graph for `device`, binding all params.
pub fn compile(
    device: Device,
    graph: Graph,
    params: Vec<(String, Vec<f32>)>,
    out_c: usize,
    out_t: usize,
    input_name: &str,
) -> CompiledAudioGraph {
    let mut compiled = Session::new(device).compile(graph);
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    compiled.finalize_params();
    CompiledAudioGraph {
        compiled,
        out_c,
        out_t,
        input_name: input_name.to_string(),
    }
}

impl CompiledAudioGraph {
    /// Run with a row-major input, returning `(flat_output, (out_c, out_t))`.
    pub fn run(&mut self, input: &[f32]) -> Result<(Vec<f32>, (usize, usize))> {
        let outs = self.compiled.run(&[(self.input_name.as_str(), input)]);
        let flat = outs
            .into_iter()
            .next()
            .context("audio graph produced no output")?;
        Ok((flat, (self.out_c, self.out_t)))
    }

    /// Run with multiple named inputs (e.g. a latent plus per-block noise planes).
    pub fn run_many(&mut self, inputs: &[(&str, &[f32])]) -> Result<(Vec<f32>, (usize, usize))> {
        let outs = self.compiled.run(inputs);
        let flat = outs
            .into_iter()
            .next()
            .context("audio graph produced no output")?;
        Ok((flat, (self.out_c, self.out_t)))
    }

    pub fn out_dims(&self) -> (usize, usize) {
        (self.out_c, self.out_t)
    }
}

/// Finalize a single-output graph built into `hir`: set the output, lower, and
/// return the MIR `Graph`. Call after moving params out of the `AudioGraph`.
pub fn finish_graph(mut hir: HirModule, output: HirNodeId) -> Result<Graph> {
    hir.set_outputs(vec![output]);
    Graph::from_hir(hir).map_err(|e| anyhow::anyhow!("audio graph lower: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
        fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| self.f() * scale).collect()
        }
    }

    fn gpu_devices() -> Vec<Device> {
        let v = vec![Device::Cpu];
        #[cfg(feature = "metal")]
        if rlx_runtime::is_available(Device::Metal) {
            v.push(Device::Metal);
        }
        #[cfg(feature = "mlx")]
        if rlx_runtime::is_available(Device::Mlx) {
            v.push(Device::Mlx);
        }
        #[cfg(feature = "gpu")]
        if rlx_runtime::is_available(Device::Gpu) {
            v.push(Device::Gpu);
        }
        v
    }

    /// Reference: plain-loop conv1d (cross-correlation) with explicit asymmetric
    /// zero padding, dilation, stride, groups. Input `[c_in, t]`, weight
    /// `[c_out, c_in/groups, k]`. Returns `[c_out, t_out]` flat.
    fn ref_conv1d(
        x: &[f32],
        c_in: usize,
        t: usize,
        w: &[f32],
        c_out: usize,
        k: usize,
        stride: usize,
        dil: usize,
        groups: usize,
        pl: usize,
        pr: usize,
        bias: Option<&[f32]>,
    ) -> (Vec<f32>, usize) {
        let t_pad = t + pl + pr;
        let mut padded = vec![0f32; c_in * t_pad];
        for ci in 0..c_in {
            for i in 0..t {
                padded[ci * t_pad + pl + i] = x[ci * t + i];
            }
        }
        let t_out = (t_pad - (dil * (k - 1) + 1)) / stride + 1;
        let c_in_g = c_in / groups;
        let c_out_g = c_out / groups;
        let mut out = vec![0f32; c_out * t_out];
        for grp in 0..groups {
            for ocg in 0..c_out_g {
                let oc = grp * c_out_g + ocg;
                for to in 0..t_out {
                    let mut acc = bias.map(|b| b[oc]).unwrap_or(0.0);
                    for icg in 0..c_in_g {
                        let ic = grp * c_in_g + icg;
                        for ki in 0..k {
                            let src = to * stride + ki * dil;
                            acc += padded[ic * t_pad + src] * w[(oc * c_in_g + icg) * k + ki];
                        }
                    }
                    out[oc * t_out + to] = acc;
                }
            }
        }
        (out, t_out)
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// Build a single-conv graph, run on `dev`, return flat `[c_out, t_out]`.
    fn run_conv(dev: Device, x: &[f32], c_in: usize, t: usize, spec: &Conv1d) -> (Vec<f32>, usize) {
        let mut hir = HirModule::new("conv_test");
        let mut g = HirMut::new(&mut hir);
        let mut ag = AudioGraph::new(&mut g);
        let xin = ag.input("x", &[1, c_in, t, 1]);
        let (out, _oc, ot) = ag.conv1d(xin, t, spec);
        let params = std::mem::take(&mut ag.params);
        let graph = finish_graph(hir, out).unwrap();
        let mut comp = compile(dev, graph, params, spec.c_out, ot, "x");
        let (flat, _) = comp.run(x).unwrap();
        (flat, ot)
    }

    #[test]
    fn conv1d_dilated_grouped_matches_reference_all_backends() {
        let mut r = Lcg(42);
        let (c_in, t, c_out, k, stride, dil, groups) =
            (8usize, 60usize, 8usize, 3usize, 1usize, 2usize, 2usize);
        let x = r.vec(c_in * t, 0.5);
        let w = r.vec(c_out * (c_in / groups) * k, 0.3);
        let bias = r.vec(c_out, 0.1);
        let (pl, pr) = (same_pad(k, dil), same_pad(k, dil));
        let spec = Conv1d {
            weight: &w,
            bias: Some(&bias),
            c_out,
            c_in,
            k,
            stride,
            dilation: dil,
            groups,
            pad_left: pl,
            pad_right: pr,
            pad_mode: PadMode::Constant,
        };
        let (reference, t_out) = ref_conv1d(
            &x,
            c_in,
            t,
            &w,
            c_out,
            k,
            stride,
            dil,
            groups,
            pl,
            pr,
            Some(&bias),
        );

        for dev in gpu_devices() {
            let (got, ot) = run_conv(dev, &x, c_in, t, &spec);
            assert_eq!(ot, t_out);
            let err = max_abs(&got, &reference);
            assert!(err < 2e-3, "conv1d on {dev:?}: max|Δ| = {err}");
            eprintln!("conv1d {dev:?} ok: max|Δ| = {err:.2e}");
        }
    }

    #[test]
    fn conv_transpose1d_matches_reference_all_backends() {
        let mut r = Lcg(7);
        let (c_in, t, c_out, k, stride) = (4usize, 20usize, 6usize, 4usize, 2usize);
        let x = r.vec(c_in * t, 0.5);
        let w = r.vec(c_in * c_out * k, 0.3); // [c_in, c_out, k]
        let bias = r.vec(c_out, 0.1);
        let spec = ConvTranspose1d {
            weight: &w,
            bias: Some(&bias),
            c_in,
            c_out_per_group: c_out,
            k,
            stride,
            groups: 1,
            trim_left: 1,
            trim_right: 1,
        };

        // Reference: full transposed conv then trim.
        let t_full = (t - 1) * stride + k;
        let mut full = vec![0f32; c_out * t_full];
        for oc in 0..c_out {
            for p in 0..t_full {
                full[oc * t_full + p] = bias[oc];
            }
        }
        for ti in 0..t {
            for ic in 0..c_in {
                let xv = x[ic * t + ti];
                for ki in 0..k {
                    let p = ti * stride + ki;
                    for oc in 0..c_out {
                        full[oc * t_full + p] += xv * w[(ic * c_out + oc) * k + ki];
                    }
                }
            }
        }
        let t_out = t_full - 2;
        let mut reference = vec![0f32; c_out * t_out];
        for oc in 0..c_out {
            for to in 0..t_out {
                reference[oc * t_out + to] = full[oc * t_full + 1 + to];
            }
        }

        for dev in gpu_devices() {
            let mut hir = HirModule::new("ct_test");
            let mut g = HirMut::new(&mut hir);
            let mut ag = AudioGraph::new(&mut g);
            let xin = ag.input("x", &[1, c_in, t, 1]);
            let (out, _oc, ot) = ag.conv_transpose1d(xin, t, &spec);
            let params = std::mem::take(&mut ag.params);
            let graph = finish_graph(hir, out).unwrap();
            let mut comp = compile(dev, graph, params, c_out, ot, "x");
            let (got, _) = comp.run(&x).unwrap();
            assert_eq!(ot, t_out);
            let err = max_abs(&got, &reference);
            assert!(err < 2e-3, "conv_transpose1d on {dev:?}: max|Δ| = {err}");
            eprintln!("conv_transpose1d {dev:?} ok: max|Δ| = {err:.2e}");
        }
    }
}
