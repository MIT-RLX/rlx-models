// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! Shared low-level ops used by both the FlowLM and the Mimi decoder.
//!
//! All ops operate on `ndarray::Array*<f32>`. Conventions follow PyTorch:
//! Linear `y = x @ W^T + b` where `W` is `[out, in]`. LayerNorm/RMSNorm match
//! the standard formulations.

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};

// ─── Activations ──────────────────────────────────────────────────────────────

#[inline]
pub fn gelu_scalar(x: f32) -> f32 {
    // Exact GELU as used by PyTorch's `nn.GELU()`.
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

#[inline]
pub fn silu_scalar(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub fn elu_scalar(x: f32) -> f32 {
    if x >= 0.0 { x } else { x.exp() - 1.0 }
}

pub fn gelu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = gelu_scalar(*v);
    }
}

pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = silu_scalar(*v);
    }
}

pub fn elu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = elu_scalar(*v);
    }
}

/// Abramowitz–Stegun erf approximation (~ 1.5e-7 max error). Matches what
/// libm's float erf produces; good enough for inference.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    // A&S 7.1.26 constants.
    let a1 = 0.254_829_6_f32;
    let a2 = -0.284_496_72_f32;
    let a3 = 1.421_413_8_f32;
    let a4 = -1.453_152_1_f32;
    let a5 = 1.061_405_4_f32;
    let p = 0.327_591_1_f32;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

// ─── Linear (matmul + optional bias) ──────────────────────────────────────────

/// `y = x @ W^T + b`, with `x: [N, in]`, `W: [out, in]`, `b: [out]?`, returns `[N, out]`.
///
/// On Apple targets with the `accelerate` feature, routes through CBLAS sgemm.
/// Elsewhere falls back to ndarray's dot product.
pub fn linear(x: ArrayView2<f32>, w: ArrayView2<f32>, b: Option<ArrayView1<f32>>) -> Array2<f32> {
    let (n, c_in) = x.dim();
    let (c_out, c_in_w) = w.dim();
    debug_assert_eq!(c_in, c_in_w, "linear: in dims mismatch");

    let mut out = Array2::<f32>::zeros((n, c_out));
    if n == 0 || c_in == 0 || c_out == 0 {
        if let Some(b) = b {
            for mut row in out.axis_iter_mut(Axis(0)) {
                for j in 0..c_out {
                    row[j] = b[j];
                }
            }
        }
        return out;
    }

    #[cfg(all(feature = "accelerate", any(target_os = "macos", target_os = "ios")))]
    {
        // x: [N, K] row-major, lda = K
        // w: [M, K] row-major (we want W^T so set trans_b = T) → ldb = K
        // out: [N, M] row-major, ldc = M
        let x_slice = x.as_standard_layout();
        let w_slice = w.as_standard_layout();
        unsafe {
            blas_ffi::cblas_sgemm(
                blas_ffi::CBLAS_ROW_MAJOR,
                blas_ffi::CBLAS_NO_TRANS,
                blas_ffi::CBLAS_TRANS,
                n as i32,
                c_out as i32,
                c_in as i32,
                1.0,
                x_slice.as_ptr(),
                c_in as i32,
                w_slice.as_ptr(),
                c_in as i32,
                0.0,
                out.as_mut_ptr(),
                c_out as i32,
            );
        }
    }

    #[cfg(not(all(feature = "accelerate", any(target_os = "macos", target_os = "ios"))))]
    {
        out = x.dot(&w.t());
    }

    if let Some(b) = b {
        debug_assert_eq!(b.len(), c_out);
        for mut row in out.axis_iter_mut(Axis(0)) {
            for j in 0..c_out {
                row[j] += b[j];
            }
        }
    }
    out
}

#[cfg(all(feature = "accelerate", any(target_os = "macos", target_os = "ios")))]
mod blas_ffi {
    use std::os::raw::c_int;

    // CBLAS enum values mirror cblas.h.
    pub const CBLAS_ROW_MAJOR: c_int = 101;
    pub const CBLAS_NO_TRANS: c_int = 111;
    pub const CBLAS_TRANS: c_int = 112;

    #[allow(non_snake_case)]
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub fn cblas_sgemm(
            layout: c_int,
            trans_a: c_int,
            trans_b: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f32,
            a: *const f32,
            lda: c_int,
            b: *const f32,
            ldb: c_int,
            beta: f32,
            c: *mut f32,
            ldc: c_int,
        );
    }
}

// ─── Norms ────────────────────────────────────────────────────────────────────

pub fn layernorm(
    x: ArrayView2<f32>,
    weight: Option<ArrayView1<f32>>,
    bias: Option<ArrayView1<f32>>,
    eps: f32,
) -> Array2<f32> {
    let (n, c) = x.dim();
    let mut out = x.to_owned();
    let inv_c = 1.0 / c as f32;
    for i in 0..n {
        let mut mean = 0.0;
        for j in 0..c {
            mean += out[[i, j]];
        }
        mean *= inv_c;
        let mut var = 0.0;
        for j in 0..c {
            let d = out[[i, j]] - mean;
            var += d * d;
        }
        var *= inv_c;
        let inv = 1.0 / (var + eps).sqrt();
        for j in 0..c {
            let mut v = (out[[i, j]] - mean) * inv;
            if let Some(w) = weight {
                v *= w[j];
            }
            if let Some(b) = bias {
                v += b[j];
            }
            out[[i, j]] = v;
        }
    }
    out
}

/// "RMS norm" as defined in `pocket_tts/modules/mlp.py::_rms_norm`:
/// scales `x` by `1/sqrt(var(x, unbiased=False) + eps)` (i.e. the centered
/// second moment, NOT the raw mean-square) and applies `weight` per channel.
/// The numerator stays as `x` (not mean-subtracted) — only the denominator
/// uses centered variance. Matches `x * (alpha * torch.rsqrt(var))` exactly.
pub fn rmsnorm(x: ArrayView2<f32>, weight: ArrayView1<f32>, eps: f32) -> Array2<f32> {
    let (n, c) = x.dim();
    let mut out = x.to_owned();
    let inv_c = 1.0 / c as f32;
    for i in 0..n {
        let mut mean = 0.0;
        for j in 0..c {
            mean += out[[i, j]];
        }
        mean *= inv_c;
        let mut sq = 0.0;
        for j in 0..c {
            let d = out[[i, j]] - mean;
            sq += d * d;
        }
        let inv = 1.0 / (sq * inv_c + eps).sqrt();
        for j in 0..c {
            out[[i, j]] = out[[i, j]] * inv * weight[j];
        }
    }
    out
}

// ─── RoPE (interleaved pairs — matches pocket_tts.modules.rope) ───────────────

/// Build the cached inverse-frequency vector for RoPE of head dim `D`:
/// `inv_freq[k] = 1 / max_period ** (2k / D)` for `k in 0..D/2`.
pub fn rope_inv_freq(head_dim: usize, max_period: f32) -> Array1<f32> {
    let half = head_dim / 2;
    let mut buf = Array1::<f32>::zeros(half);
    for k in 0..half {
        let exponent = (2 * k) as f32 / head_dim as f32;
        buf[k] = 1.0 / max_period.powf(exponent);
    }
    buf
}

/// Apply RoPE in-place over a `[T, H, D]` slab, where the head dim packs
/// interleaved `(real, imag)` pairs (`x[..., 0]` real, `x[..., 1]` imag).
///
/// `positions` provides the absolute position per token (length T). This matches
/// pocket_tts's `rope.py` `apply_rotary_pos_emb`.
pub fn apply_rope(x: &mut Array3<f32>, positions: &[i64], inv_freq: ArrayView1<f32>) {
    let (t, h, d) = x.dim();
    debug_assert_eq!(positions.len(), t);
    debug_assert!(d % 2 == 0);
    let half = d / 2;
    for ti in 0..t {
        let pos = positions[ti] as f32;
        for hi in 0..h {
            for k in 0..half {
                let theta = pos * inv_freq[k];
                let cos = theta.cos();
                let sin = theta.sin();
                let re = x[[ti, hi, 2 * k]];
                let im = x[[ti, hi, 2 * k + 1]];
                x[[ti, hi, 2 * k]] = re * cos - im * sin;
                x[[ti, hi, 2 * k + 1]] = re * sin + im * cos;
            }
        }
    }
}

// ─── 1-D Conv helpers ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode {
    Constant,
    Replicate,
}

/// Causal 1-D convolution following pocket_tts's `StreamingConv1d`: the input
/// is left-padded by `(K - 1) * D - (S - 1)` (effectively `effective_K - stride`),
/// then `Conv1d(stride=S)` is applied with no right-side padding.
///
/// Shapes: `x: [Cin, T_in]`, `weight: [Cout, Cin, K]` → `[Cout, T_out]`.
pub fn causal_conv1d(
    x: ArrayView2<f32>,
    weight: ArrayView3<f32>,
    bias: Option<ArrayView1<f32>>,
    stride: usize,
    dilation: usize,
    pad_mode: PadMode,
    groups: usize,
) -> Array2<f32> {
    let (out_ch, in_ch_per_group, k) = weight.dim();
    let (cin_full, t_in) = x.dim();
    debug_assert_eq!(cin_full, in_ch_per_group * groups);
    debug_assert_eq!(out_ch % groups, 0);
    let out_ch_per_group = out_ch / groups;

    let eff_k = (k - 1) * dilation + 1;
    let pad_left = eff_k.saturating_sub(stride);
    let t_pad = t_in + pad_left;
    if t_pad < eff_k {
        return Array2::<f32>::zeros((out_ch, 0));
    }
    let t_out = (t_pad - eff_k) / stride + 1;

    let mut padded = vec![0f32; cin_full * t_pad];
    for ci in 0..cin_full {
        let base = ci * t_pad;
        // Left pad
        let left = match pad_mode {
            PadMode::Constant => 0.0,
            PadMode::Replicate => x[[ci, 0]],
        };
        for t in 0..pad_left {
            padded[base + t] = left;
        }
        for t in 0..t_in {
            padded[base + pad_left + t] = x[[ci, t]];
        }
    }

    let mut out = Array2::<f32>::zeros((out_ch, t_out));
    for g in 0..groups {
        let in_lo = g * in_ch_per_group;
        let out_lo = g * out_ch_per_group;
        for oc in 0..out_ch_per_group {
            let oc_full = out_lo + oc;
            for ti in 0..t_out {
                let mut acc = 0.0;
                for ic in 0..in_ch_per_group {
                    let ic_full = in_lo + ic;
                    let row = ic_full * t_pad;
                    for kk in 0..k {
                        let tt = ti * stride + kk * dilation;
                        acc += weight[[oc_full, ic, kk]] * padded[row + tt];
                    }
                }
                if let Some(b) = bias {
                    acc += b[oc_full];
                }
                out[[oc_full, ti]] = acc;
            }
        }
    }
    out
}

/// 1-D transposed convolution matching pocket_tts's `StreamingConvTranspose1d`
/// one-shot path: compute the full transposed conv, drop the trailing
/// `K - S` samples (held back as `partial` in streaming mode). Output length
/// is `T_in * S`.
///
/// Shapes: `x: [Cin, T_in]`, `weight: [Cin, Cout/groups, K]` → `[Cout, T_in*S]`.
pub fn causal_conv_transpose1d(
    x: ArrayView2<f32>,
    weight: ArrayView3<f32>,
    bias: Option<ArrayView1<f32>>,
    stride: usize,
    groups: usize,
) -> Array2<f32> {
    let (cin_full, t_in) = x.dim();
    let (cin_w, cout_per_group, k) = weight.dim();
    debug_assert_eq!(cin_full, cin_w);
    debug_assert_eq!(cin_full % groups, 0);
    let in_per_group = cin_full / groups;
    let out_ch = cout_per_group * groups;
    if t_in == 0 {
        return Array2::<f32>::zeros((out_ch, 0));
    }
    let full_t = (t_in - 1) * stride + k;
    // PyTorch ConvTranspose1d weight layout: `[in_ch, out_ch/groups, K]`.
    let mut full = Array2::<f32>::zeros((out_ch, full_t));
    for g in 0..groups {
        let in_lo = g * in_per_group;
        let out_lo = g * cout_per_group;
        for ic in 0..in_per_group {
            let ic_full = in_lo + ic;
            for ti in 0..t_in {
                let v = x[[ic_full, ti]];
                if v == 0.0 {
                    continue;
                }
                let t_base = ti * stride;
                for oc in 0..cout_per_group {
                    let oc_full = out_lo + oc;
                    let w_row = weight.slice(ndarray::s![ic_full, oc, ..]);
                    for kk in 0..k {
                        full[[oc_full, t_base + kk]] += v * w_row[kk];
                    }
                }
            }
        }
    }
    if let Some(b) = bias {
        for oc in 0..out_ch {
            let by = b[oc];
            for t in 0..full_t {
                full[[oc, t]] += by;
            }
        }
    }
    // Drop the trailing `K - S` samples (held back as `partial`); output is
    // `T_in * S` samples.
    let drop = k.saturating_sub(stride);
    let trimmed_len = full_t - drop;
    let mut out = Array2::<f32>::zeros((out_ch, trimmed_len));
    for oc in 0..out_ch {
        for t in 0..trimmed_len {
            out[[oc, t]] = full[[oc, t]];
        }
    }
    out
}
