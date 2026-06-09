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

//! Small ndarray helpers for eager CPU speech-tokenizer decode.

use ndarray::{Array2, ArrayView1, ArrayView2, ArrayView3};
use rayon::prelude::*;
use rlx_cpu::blas::{sgemm, sgemm_bt};

pub fn rms_norm(x: ArrayView2<f32>, weight: ArrayView1<f32>, eps: f32) -> Array2<f32> {
    let (t, d) = x.dim();
    let mut out = Array2::<f32>::zeros((t, d));
    for i in 0..t {
        let row = x.row(i);
        let mut sum = 0f32;
        for v in row.iter() {
            sum += v * v;
        }
        let inv = 1.0 / (sum / d as f32 + eps).sqrt();
        for j in 0..d {
            out[[i, j]] = row[j] * inv * weight[j];
        }
    }
    out
}

pub fn silu(x: ArrayView2<f32>) -> Array2<f32> {
    x.mapv(|v| v / (1.0 + (-v).exp()))
}

pub fn swiglu(w1: ArrayView2<f32>, w3: ArrayView2<f32>, w2: &Array2<f32>) -> Array2<f32> {
    let h = silu(w1) * w3.to_owned();
    linear2(h.view(), w2.view(), None)
}

pub fn linear2(
    x: ArrayView2<f32>,
    w: ArrayView2<f32>,
    bias: Option<ArrayView1<f32>>,
) -> Array2<f32> {
    // HF Linear weight is usually `[out, in]`; some checkpoints store `[in, out]` (e.g. output_proj).
    let mut out = if x.ncols() == w.nrows() {
        let w_owned = w.to_owned();
        x.dot(&w_owned)
    } else {
        x.dot(&w.t())
    };
    if let Some(b) = bias {
        for mut row in out.rows_mut() {
            row += &b;
        }
    }
    out
}

pub fn gelu(x: ArrayView2<f32>) -> Array2<f32> {
    x.mapv(|v| 0.5 * v * (1.0 + (v * std::f32::consts::FRAC_2_SQRT_PI * 0.5).tanh()))
}

pub fn layer_norm(
    x: ArrayView2<f32>,
    weight: ArrayView1<f32>,
    bias: ArrayView1<f32>,
    eps: f32,
) -> Array2<f32> {
    let (t, d) = x.dim();
    let mut out = Array2::<f32>::zeros((t, d));
    for i in 0..t {
        let row = x.row(i);
        let mut mean = 0f32;
        for v in row.iter() {
            mean += v;
        }
        mean /= d as f32;
        let mut var = 0f32;
        for v in row.iter() {
            let dlt = v - mean;
            var += dlt * dlt;
        }
        var /= d as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for j in 0..d {
            out[[i, j]] = (row[j] - mean) * inv * weight[j] + bias[j];
        }
    }
    out
}

#[allow(dead_code)]
pub fn pad1d_reflect(x: ArrayView2<f32>, pad_left: usize, pad_right: usize) -> Array2<f32> {
    let (c, t) = x.dim();
    let out_len = t + pad_left + pad_right;
    let mut out = Array2::<f32>::zeros((c, out_len));
    for ci in 0..c {
        for ti in 0..t {
            out[[ci, ti + pad_left]] = x[[ci, ti]];
        }
        for pi in 0..pad_left {
            let src = pi.min(t.saturating_sub(1));
            out[[ci, pad_left - 1 - pi]] = x[[ci, src]];
        }
        for pi in 0..pad_right {
            let src = (t.saturating_sub(1)).saturating_sub(pi);
            out[[ci, pad_left + t + pi]] = x[[ci, src]];
        }
    }
    out
}

fn pad1d_constant(x: ArrayView2<f32>, pad_left: usize, pad_right: usize) -> Array2<f32> {
    let (c, t) = x.dim();
    let out_len = t + pad_left + pad_right;
    let mut out = Array2::<f32>::zeros((c, out_len));
    for ci in 0..c {
        for ti in 0..t {
            out[[ci, ti + pad_left]] = x[[ci, ti]];
        }
    }
    out
}

/// Pre-flattened `[out_ch, in_ch * k]` causal conv weights (built once at load).
#[derive(Clone)]
pub struct FlatConv1d {
    pub w: Array2<f32>,
    pub bias: Option<Vec<f32>>,
    pub stride: usize,
    pub dilation: usize,
    pub in_ch: usize,
    pub k: usize,
    pub out_ch: usize,
}

impl FlatConv1d {
    pub fn from_view(
        weight: ArrayView3<f32>,
        bias: Option<ArrayView1<f32>>,
        stride: usize,
        dilation: usize,
    ) -> Self {
        let (out_ch, in_ch, k) = weight.dim();
        let patch = in_ch * k;
        let mut w = Array2::<f32>::zeros((out_ch, patch));
        for oc in 0..out_ch {
            for ic in 0..in_ch {
                for ki in 0..k {
                    w[[oc, ic * k + ki]] = weight[[oc, ic, ki]];
                }
            }
        }
        Self {
            w,
            bias: bias.map(|b| b.to_vec()),
            stride,
            dilation,
            in_ch,
            k,
            out_ch,
        }
    }
}

/// Pre-flattened transposed conv: per kernel tap `W_k` is `[out_ch, in_ch]`.
#[derive(Clone)]
pub struct FlatTransConv1d {
    pub w_k: Vec<Array2<f32>>,
    pub bias: Option<Vec<f32>>,
    pub stride: usize,
    pub in_ch: usize,
    pub out_ch: usize,
    pub k: usize,
}

impl FlatTransConv1d {
    pub fn from_view(
        weight: ArrayView3<f32>,
        bias: Option<ArrayView1<f32>>,
        stride: usize,
    ) -> Self {
        let (in_ch, out_ch, k) = weight.dim();
        let mut w_k = Vec::with_capacity(k);
        for ki in 0..k {
            let mut w = Array2::<f32>::zeros((out_ch, in_ch));
            for ic in 0..in_ch {
                for oc in 0..out_ch {
                    w[[oc, ic]] = weight[[ic, oc, ki]];
                }
            }
            w_k.push(w);
        }
        Self {
            w_k,
            bias: bias.map(|b| b.to_vec()),
            stride,
            in_ch,
            out_ch,
            k,
        }
    }
}

/// Channel-major `[ch, t]` buffer for ping-pong conv decode (no per-layer `Array2` alloc).
#[derive(Default, Clone)]
pub struct ChT {
    pub ch: usize,
    pub t: usize,
    pub(crate) data: Vec<f32>,
}

impl ChT {
    pub fn ensure(&mut self, ch: usize, t: usize) {
        let n = ch.saturating_mul(t);
        if self.data.len() < n {
            self.data.resize(n, 0.0);
        }
        self.ch = ch;
        self.t = t;
    }

    #[inline]
    pub fn view(&self) -> ArrayView2<'_, f32> {
        ArrayView2::from_shape((self.ch, self.t), &self.data[..self.ch * self.t]).expect("cht view")
    }

    pub fn adopt_from_array2(&mut self, a: &Array2<f32>) {
        let (ch, t) = a.dim();
        self.ensure(ch, t);
        self.data[..ch * t].copy_from_slice(a.as_slice_memory_order().expect("cht adopt"));
    }
}

/// Reusable scratch for [`causal_conv1d_flat`] / [`conv_transpose1d_flat`].
pub struct ConvWorkspace {
    padded: Vec<f32>,
    col: Vec<f32>,
    gemm: Vec<f32>,
    trans_out: Vec<f32>,
}

impl ConvWorkspace {
    pub fn new() -> Self {
        Self {
            padded: Vec::new(),
            col: Vec::new(),
            gemm: Vec::new(),
            trans_out: Vec::new(),
        }
    }
}

/// Cached-weight causal conv with optional GPU gemm (same numerics as [`causal_conv1d_flat`]).
pub fn causal_conv1d_flat_maybe_gpu(
    x: ArrayView2<f32>,
    flat: &FlatConv1d,
    ws: &mut ConvWorkspace,
    gpu: Option<&mut super::gpu_matmul::GpuMatmulCache>,
) -> Array2<f32> {
    let mut tmp = ChT::default();
    causal_conv1d_flat_cht_maybe_gpu(x, flat, ws, gpu, &mut tmp);
    let mut out = Array2::<f32>::zeros((tmp.ch, tmp.t));
    out.as_slice_memory_order_mut()
        .expect("contiguous conv out")
        .copy_from_slice(&tmp.data[..tmp.ch * tmp.t]);
    out
}

/// Cached-weight causal conv into [`ChT`] (parallel im2col + optional GPU gemm).
pub fn causal_conv1d_flat_cht_maybe_gpu(
    x: ArrayView2<f32>,
    flat: &FlatConv1d,
    ws: &mut ConvWorkspace,
    gpu: Option<&mut super::gpu_matmul::GpuMatmulCache>,
    out: &mut ChT,
) {
    let (in_ch, t_in) = x.dim();
    debug_assert_eq!(in_ch, flat.in_ch);
    let k = flat.k;
    let dilation = flat.dilation;
    let stride = flat.stride;
    let effective_k = (k - 1) * dilation + 1;
    let pad_left = effective_k.saturating_sub(stride);
    let n_frames = ((t_in as f32 - effective_k as f32 + pad_left as f32) / stride as f32 + 1.0)
        .floor() as isize;
    let n_frames = n_frames.max(1) as usize;
    let ideal_len = (n_frames - 1) * stride + effective_k - pad_left;
    let extra_right = ideal_len.saturating_sub(t_in);
    let t_pad = t_in + pad_left + extra_right;
    let t_out = (t_pad - effective_k) / stride + 1;
    let patch = flat.in_ch * flat.k;
    let out_ch = flat.out_ch;

    let pad_len = in_ch * t_pad;
    if ws.padded.len() < pad_len {
        ws.padded.resize(pad_len, 0.0);
    }
    ws.padded[..pad_len].fill(0.0);
    for ic in 0..in_ch {
        let row = &mut ws.padded[ic * t_pad..ic * t_pad + t_pad];
        for ti in 0..t_in {
            row[pad_left + ti] = x[[ic, ti]];
        }
    }

    let col_len = t_out * patch;
    if ws.col.len() < col_len {
        ws.col.resize(col_len, 0.0);
    }
    ws.col[..col_len].fill(0.0);
    if t_out > 4 {
        let padded = &ws.padded[..pad_len];
        ws.col[..col_len]
            .par_chunks_mut(patch)
            .enumerate()
            .for_each(|(ti, row)| {
                for ic in 0..in_ch {
                    for ki in 0..k {
                        let src_t = ti * stride + ki * dilation;
                        if src_t < t_pad {
                            row[ic * k + ki] = padded[ic * t_pad + src_t];
                        }
                    }
                }
            });
    } else {
        for ti in 0..t_out {
            for ic in 0..in_ch {
                for ki in 0..k {
                    let src_t = ti * stride + ki * dilation;
                    if src_t < t_pad {
                        ws.col[ti * patch + ic * k + ki] = ws.padded[ic * t_pad + src_t];
                    }
                }
            }
        }
    }

    if let Some(cache) = gpu {
        let col =
            ArrayView2::from_shape((t_out, patch), &ws.col[..col_len]).expect("flat conv col");
        if let Ok(mut out_tc) = cache.matmul_bt(col, flat.w.view()) {
            if let Some(b) = &flat.bias {
                for mut row in out_tc.rows_mut() {
                    for (v, &bi) in row.iter_mut().zip(b.iter()) {
                        *v += bi;
                    }
                }
            }
            out.ensure(out_ch, t_out);
            for ti in 0..t_out {
                for oc in 0..out_ch {
                    out.data[oc * t_out + ti] = out_tc[[ti, oc]];
                }
            }
            return;
        }
    }

    out.ensure(out_ch, t_out);
    let out_len = out_ch * t_out;
    let w = flat.w.as_slice().expect("flat conv weight");
    sgemm_bt(
        w,
        &ws.col[..col_len],
        &mut out.data[..out_len],
        out_ch,
        patch,
        t_out,
        1.0,
    );
    if let Some(b) = &flat.bias {
        for oc in 0..out_ch {
            let bi = b[oc];
            let row = &mut out.data[oc * t_out..(oc + 1) * t_out];
            for v in row.iter_mut() {
                *v += bi;
            }
        }
    }
}

/// Cached-weight causal conv (same numerics as [`causal_conv1d`] for `groups == 1`).
#[cfg(test)]
pub fn causal_conv1d_flat(
    x: ArrayView2<f32>,
    flat: &FlatConv1d,
    ws: &mut ConvWorkspace,
) -> Array2<f32> {
    let mut tmp = ChT::default();
    causal_conv1d_flat_cht_maybe_gpu(x, flat, ws, None, &mut tmp);
    let mut out = Array2::<f32>::zeros((tmp.ch, tmp.t));
    out.as_slice_memory_order_mut()
        .expect("contiguous conv out")
        .copy_from_slice(&tmp.data[..tmp.ch * tmp.t]);
    out
}

/// Cached-weight causal transposed conv into [`ChT`].
pub fn conv_transpose1d_flat_cht(
    x: ArrayView2<f32>,
    flat: &FlatTransConv1d,
    ws: &mut ConvWorkspace,
    out: &mut ChT,
) {
    let (in_ch, t_in) = x.dim();
    debug_assert_eq!(in_ch, flat.in_ch);
    let stride = flat.stride;
    let k = flat.k;
    let out_ch = flat.out_ch;
    let trim_right = k.saturating_sub(stride);
    let t_raw = (t_in - 1) * stride + k;
    let end = t_raw - trim_right;

    let buf_len = out_ch * t_raw;
    if ws.trans_out.len() < buf_len {
        ws.trans_out.resize(buf_len, 0.0);
    }
    ws.trans_out[..buf_len].fill(0.0);
    if let Some(b) = &flat.bias {
        for oc in 0..out_ch {
            let bi = b[oc];
            let row = &mut ws.trans_out[oc * t_raw..(oc + 1) * t_raw];
            for v in row.iter_mut() {
                *v = bi;
            }
        }
    }
    let x_flat = x.as_slice_memory_order().expect("trans conv x contiguous");
    let tap_len = out_ch * t_in;
    if ws.gemm.len() < tap_len {
        ws.gemm.resize(tap_len, 0.0);
    }
    for (ki, w) in flat.w_k.iter().enumerate() {
        let w_slice = w.as_slice().expect("trans conv tap");
        sgemm(
            w_slice,
            x_flat,
            &mut ws.gemm[..tap_len],
            out_ch,
            in_ch,
            t_in,
        );
        for oc in 0..out_ch {
            let y_row = &ws.gemm[oc * t_in..(oc + 1) * t_in];
            let out_row = &mut ws.trans_out[oc * t_raw..(oc + 1) * t_raw];
            for ti in 0..t_in {
                let out_t = ti * stride + ki;
                if out_t < t_raw {
                    out_row[out_t] += y_row[ti];
                }
            }
        }
    }

    out.ensure(out_ch, end);
    for oc in 0..out_ch {
        let src = &ws.trans_out[oc * t_raw..oc * t_raw + end];
        out.data[oc * end..(oc + 1) * end].copy_from_slice(src);
    }
}

/// Cached-weight causal transposed conv (same numerics as [`conv_transpose1d`]).
pub fn conv_transpose1d_flat(
    x: ArrayView2<f32>,
    flat: &FlatTransConv1d,
    ws: &mut ConvWorkspace,
) -> Array2<f32> {
    let mut local_ws = ConvWorkspace::new();
    let mut tmp = ChT::default();
    conv_transpose1d_flat_cht(x, flat, &mut local_ws, &mut tmp);
    let _ = ws;
    let mut out = Array2::<f32>::zeros((tmp.ch, tmp.t));
    out.as_slice_memory_order_mut()
        .expect("contiguous trans conv out")
        .copy_from_slice(&tmp.data[..tmp.ch * tmp.t]);
    out
}

/// HF `Qwen3TTSTokenizerV2CausalConvNet` — constant left pad, optional extra right pad.
pub fn causal_conv1d(
    x: ArrayView2<f32>,
    weight: ArrayView3<f32>,
    bias: Option<ArrayView1<f32>>,
    stride: usize,
    dilation: usize,
) -> Array2<f32> {
    let (out_ch, in_per_group, k) = weight.dim();
    let (in_ch, t_in) = x.dim();
    let groups = in_ch.checked_div(in_per_group).unwrap_or(1);
    let effective_k = (k - 1) * dilation + 1;
    let pad_left = effective_k.saturating_sub(stride);
    let n_frames = ((t_in as f32 - effective_k as f32 + pad_left as f32) / stride as f32 + 1.0)
        .floor() as isize;
    let n_frames = n_frames.max(1) as usize;
    let ideal_len = (n_frames - 1) * stride + effective_k - pad_left;
    let extra_right = ideal_len.saturating_sub(t_in);
    let padded = pad1d_constant(x, pad_left, extra_right);
    let t_pad = padded.dim().1;
    let t_out = (t_pad - effective_k) / stride + 1;

    if groups == 1 {
        return causal_conv1d_gemm(
            padded.view(),
            weight,
            bias,
            stride,
            dilation,
            out_ch,
            in_ch,
            k,
            t_out,
        );
    }

    let mut rows = vec![vec![0f32; t_out]; out_ch];
    rows.par_iter_mut().enumerate().for_each(|(oc, row)| {
        let g = oc / (out_ch / groups);
        let ic_base = g * in_per_group;
        for ti in 0..t_out {
            let mut sum = 0f32;
            for ic in 0..in_per_group {
                for ki in 0..k {
                    let src_t = ti * stride + ki * dilation;
                    if src_t < t_pad {
                        sum += padded[[ic_base + ic, src_t]] * weight[[oc, ic, ki]];
                    }
                }
            }
            row[ti] = sum;
        }
        if let Some(b) = bias {
            for ti in 0..t_out {
                row[ti] += b[oc];
            }
        }
    });
    let mut out = Array2::<f32>::zeros((out_ch, t_out));
    for oc in 0..out_ch {
        for ti in 0..t_out {
            out[[oc, ti]] = rows[oc][ti];
        }
    }
    out
}

pub(crate) fn causal_conv1d_gemm(
    padded: ArrayView2<f32>,
    weight: ArrayView3<f32>,
    bias: Option<ArrayView1<f32>>,
    stride: usize,
    dilation: usize,
    out_ch: usize,
    in_ch: usize,
    k: usize,
    t_out: usize,
) -> Array2<f32> {
    let t_pad = padded.dim().1;
    let patch = in_ch * k;
    let mut col = Array2::<f32>::zeros((t_out, patch));
    for ti in 0..t_out {
        for ic in 0..in_ch {
            for ki in 0..k {
                let src_t = ti * stride + ki * dilation;
                if src_t < t_pad {
                    col[[ti, ic * k + ki]] = padded[[ic, src_t]];
                }
            }
        }
    }
    let mut w = Array2::<f32>::zeros((out_ch, patch));
    for oc in 0..out_ch {
        for ic in 0..in_ch {
            for ki in 0..k {
                w[[oc, ic * k + ki]] = weight[[oc, ic, ki]];
            }
        }
    }
    let mut out_tc = col.dot(&w.t());
    if let Some(b) = bias {
        for mut row in out_tc.rows_mut() {
            for (v, &bi) in row.iter_mut().zip(b.iter()) {
                *v += bi;
            }
        }
    }
    let mut out = Array2::<f32>::zeros((out_ch, t_out));
    for ti in 0..t_out {
        for oc in 0..out_ch {
            out[[oc, ti]] = out_tc[[ti, oc]];
        }
    }
    out
}

#[allow(dead_code)]
pub fn conv1d(
    x: ArrayView2<f32>,
    weight: ArrayView3<f32>,
    bias: Option<ArrayView1<f32>>,
    stride: usize,
    pad_left: usize,
) -> Array2<f32> {
    let padded = pad1d_reflect(x, pad_left, 0);
    let (out_ch, in_ch, k) = weight.dim();
    let t_pad = padded.dim().1;
    let t_out = (t_pad - k) / stride + 1;
    let mut out = Array2::<f32>::zeros((out_ch, t_out));
    for oc in 0..out_ch {
        for ti in 0..t_out {
            let mut sum = 0f32;
            for ic in 0..in_ch {
                for ki in 0..k {
                    sum += padded[[ic, ti * stride + ki]] * weight[[oc, ic, ki]];
                }
            }
            out[[oc, ti]] = sum;
        }
    }
    if let Some(b) = bias {
        for oc in 0..out_ch {
            for ti in 0..t_out {
                out[[oc, ti]] += b[oc];
            }
        }
    }
    out
}

/// HF `Qwen3TTSTokenizerV2CausalTransConvNet` — trim `kernel_size - stride` from the right.
pub fn conv_transpose1d(
    x: ArrayView2<f32>,
    weight: ArrayView3<f32>,
    bias: Option<ArrayView1<f32>>,
    stride: usize,
) -> Array2<f32> {
    let (in_ch, out_ch, k) = weight.dim();
    let (_, t_in) = x.dim();
    let trim_right = k.saturating_sub(stride);
    let t_raw = (t_in - 1) * stride + k;
    let end = t_raw - trim_right;

    let rows: Vec<Vec<f32>> = (0..out_ch)
        .into_par_iter()
        .map(|oc| {
            let mut row = vec![0f32; t_raw];
            if let Some(b) = bias {
                for v in row.iter_mut() {
                    *v = b[oc];
                }
            }
            for ti in 0..t_in {
                for ic in 0..in_ch {
                    let src = x[[ic, ti]];
                    for ki in 0..k {
                        row[ti * stride + ki] += src * weight[[ic, oc, ki]];
                    }
                }
            }
            row
        })
        .collect();

    let mut out = Array2::<f32>::zeros((out_ch, end));
    for oc in 0..out_ch {
        for ti in 0..end {
            out[[oc, ti]] = rows[oc][ti];
        }
    }
    out
}

pub fn snake_beta_cht(x: &mut ChT, alpha: ArrayView1<f32>, beta: ArrayView1<f32>) {
    let (c, t) = (x.ch, x.t);
    let eps = 1e-9_f32;
    for ci in 0..c {
        let a = alpha[ci].exp();
        let b = beta[ci].exp();
        let inv_b = 1.0 / (b + eps);
        let row = &mut x.data[ci * t..(ci + 1) * t];
        for v in row.iter_mut() {
            let s = (*v * a).sin();
            *v += inv_b * s * s;
        }
    }
}

#[cfg(test)]
mod transconv_tests {
    use super::*;
    use ndarray::{Array3, ArrayView1};

    #[test]
    fn conv_transpose_matches_hf_up0_dump() {
        let x_path = "/tmp/hf_before_upsample.bin";
        let w_path = "/tmp/up0_w.bin";
        let y_path = "/tmp/hf_up0_trans.bin";
        if !std::path::Path::new(x_path).is_file() {
            eprintln!("skip: {x_path}");
            return;
        }
        let x_bytes = std::fs::read(x_path).expect("x");
        let w_bytes = std::fs::read(w_path).expect("w");
        let y_bytes = std::fs::read(y_path).expect("y");
        let in_ch = 1024usize;
        let t_in = 22usize;
        let out_ch = 1024usize;
        let k = 2usize;
        let stride = 2usize;
        let t_out = 44usize;
        let x: Vec<f32> = x_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let x = Array2::from_shape_vec((in_ch, t_in), x).expect("x shape");
        let w: Vec<f32> = w_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let w = Array3::from_shape_vec((in_ch, out_ch, k), w).expect("w shape");
        let b: Vec<f32> = std::fs::read("/tmp/up0_b.bin")
            .expect("b")
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let hf: Vec<f32> = y_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let ref_out = conv_transpose1d(x.view(), w.view(), Some(ArrayView1::from(&b)), stride);
        let flat = FlatTransConv1d::from_view(w.view(), Some(ArrayView1::from(&b)), stride);
        let mut ws = ConvWorkspace::new();
        let flat_out = conv_transpose1d_flat(x.view(), &flat, &mut ws);
        let mut max_ref = 0f32;
        let mut max_flat = 0f32;
        for ic in 0..out_ch {
            for ti in 0..t_out {
                let hf_v = hf[ic * t_out + ti];
                let d1 = (ref_out[[ic, ti]] - hf_v).abs();
                let d2 = (flat_out[[ic, ti]] - hf_v).abs();
                max_ref = max_ref.max(d1);
                max_flat = max_flat.max(d2);
            }
        }
        eprintln!("up0 vs HF: ref_max={max_ref:.6} flat_max={max_flat:.6}");
        assert!(max_ref < 1e-3, "reference transconv wrong vs HF: {max_ref}");
        assert!(max_flat < 1e-3, "flat transconv wrong vs HF: {max_flat}");
    }

    #[test]
    fn conv_transpose_flat_matches_reference() {
        let in_ch = 4usize;
        let out_ch = 3;
        let k = 2;
        let stride = 2;
        let t_in = 5;
        let mut w = Array3::<f32>::zeros((in_ch, out_ch, k));
        for ic in 0..in_ch {
            for oc in 0..out_ch {
                for ki in 0..k {
                    w[[ic, oc, ki]] = ((ic * 17 + oc * 3 + ki) as f32) * 0.01;
                }
            }
        }
        let mut x = Array2::<f32>::zeros((in_ch, t_in));
        for ic in 0..in_ch {
            for ti in 0..t_in {
                x[[ic, ti]] = ((ic + ti) as f32) * 0.1 - 0.2;
            }
        }
        let bias = ArrayView1::from(&[0.01f32, -0.02, 0.03]);
        let ref_out = conv_transpose1d(x.view(), w.view(), Some(bias.view()), stride);
        let flat = FlatTransConv1d::from_view(w.view(), Some(bias.view()), stride);
        let mut ws = ConvWorkspace::new();
        let flat_out = conv_transpose1d_flat(x.view(), &flat, &mut ws);
        assert_eq!(ref_out.dim(), flat_out.dim());
        let mut max_d = 0f32;
        for ((idx, ref_v), (_, flat_v)) in ref_out.indexed_iter().zip(flat_out.indexed_iter()) {
            let dlt = (*ref_v - *flat_v).abs();
            max_d = max_d.max(dlt);
            assert!(
                dlt < 1e-5,
                "mismatch at {:?}: ref={ref_v} flat={flat_v}",
                idx
            );
        }
        assert!(max_d < 1e-5, "max_d={max_d}");
    }
}
