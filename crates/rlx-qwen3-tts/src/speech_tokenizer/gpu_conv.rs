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

//! GPU paths for speech transposed conv and ConvNeXt (matmul + causal conv gemm).

use super::gpu_matmul::GpuMatmulCache;
use super::ops::{
    ConvWorkspace, FlatConv1d, FlatTransConv1d, causal_conv1d_flat_maybe_gpu, gelu, layer_norm,
    linear2,
};
use anyhow::Result;
use ndarray::{Array2, ArrayView1, ArrayView2, ArrayView3};

/// HF transposed conv with per-tap GPU matmul (cached tap weights).
pub fn conv_transpose1d_gpu_flat(
    cache: &mut GpuMatmulCache,
    x: ArrayView2<f32>,
    flat: &FlatTransConv1d,
) -> Array2<f32> {
    let in_ch = flat.in_ch;
    let out_ch = flat.out_ch;
    let k = flat.k;
    let stride = flat.stride;
    let (_, t_in) = x.dim();
    let trim_right = k.saturating_sub(stride);
    let t_raw = (t_in - 1) * stride + k;
    let end = t_raw - trim_right;

    let mut trans_out = vec![0f32; out_ch * t_raw];
    if let Some(b) = &flat.bias {
        for oc in 0..out_ch {
            let bi = b[oc];
            for v in trans_out[oc * t_raw..(oc + 1) * t_raw].iter_mut() {
                *v = bi;
            }
        }
    }

    let mut x_row = vec![0f32; t_in * in_ch];
    for ic in 0..in_ch {
        for ti in 0..t_in {
            x_row[ti * in_ch + ic] = x[[ic, ti]];
        }
    }
    let x_t = ArrayView2::from_shape((t_in, in_ch), &x_row).expect("transconv x_row");
    for (ki, w) in flat.w_k.iter().enumerate() {
        let _ = ki;
        match cache.matmul_bt(x_t, w.view()) {
            Ok(gemm) => {
                for ti in 0..t_in {
                    for oc in 0..out_ch {
                        let out_t = ti * stride + ki;
                        if out_t < t_raw {
                            trans_out[oc * t_raw + out_t] += gemm[[ti, oc]];
                        }
                    }
                }
            }
            Err(_) => {
                return super::ops::conv_transpose1d_flat(x, flat, &mut ConvWorkspace::new());
            }
        }
    }

    let mut out = Array2::<f32>::zeros((out_ch, end));
    for oc in 0..out_ch {
        for ti in 0..end {
            out[[oc, ti]] = trans_out[oc * t_raw + ti];
        }
    }
    out
}

/// HF transposed conv; GPU when `gpu` is set.
pub fn conv_transpose1d_maybe_gpu(
    x: ArrayView2<f32>,
    weight: ArrayView3<f32>,
    bias: Option<ArrayView1<f32>>,
    stride: usize,
    gpu: Option<&mut GpuMatmulCache>,
) -> Array2<f32> {
    match gpu {
        Some(cache) => {
            let flat = FlatTransConv1d::from_view(weight, bias, stride);
            conv_transpose1d_gpu_flat(cache, x, &flat)
        }
        None => super::ops::conv_transpose1d(x, weight, bias, stride),
    }
}

/// ConvNeXt block on GPU.
pub fn run_convnext_gpu(
    cache: &mut GpuMatmulCache,
    ws: &mut ConvWorkspace,
    x: &Array2<f32>,
    dw_flat: &FlatConv1d,
    norm_w: ArrayView1<f32>,
    norm_b: ArrayView1<f32>,
    pw1_w: ArrayView2<f32>,
    pw1_b: ArrayView1<f32>,
    pw2_w: ArrayView2<f32>,
    pw2_b: ArrayView1<f32>,
    gamma: ArrayView1<f32>,
) -> Result<Array2<f32>> {
    let residual = x.to_owned();
    let h = causal_conv1d_flat_maybe_gpu(x.view(), dw_flat, ws, Some(cache));
    let mut seq = h.t().to_owned();
    seq = layer_norm(seq.view(), norm_w, norm_b, 1e-6);
    seq = linear2_gpu(cache, seq.view(), pw1_w, Some(pw1_b))?;
    seq = gelu(seq.view());
    seq = linear2_gpu(cache, seq.view(), pw2_w, Some(pw2_b))?;
    for mut row in seq.rows_mut() {
        for (v, &g) in row.iter_mut().zip(gamma.iter()) {
            *v *= g;
        }
    }
    Ok(residual + seq.t())
}

/// ConvNeXt block; GPU dw conv + pw matmuls when `gpu` is set.
pub fn run_convnext_maybe_gpu(
    x: &Array2<f32>,
    dw_weight: ArrayView3<f32>,
    dw_bias: ArrayView1<f32>,
    norm_w: ArrayView1<f32>,
    norm_b: ArrayView1<f32>,
    pw1_w: ArrayView2<f32>,
    pw1_b: ArrayView1<f32>,
    pw2_w: ArrayView2<f32>,
    pw2_b: ArrayView1<f32>,
    gamma: ArrayView1<f32>,
    gpu: Option<&mut GpuMatmulCache>,
) -> Result<Array2<f32>> {
    match gpu {
        Some(cache) => {
            let mut ws = ConvWorkspace::new();
            let dw_flat = FlatConv1d::from_view(dw_weight, Some(dw_bias), 1, 1);
            run_convnext_gpu(
                cache, &mut ws, x, &dw_flat, norm_w, norm_b, pw1_w, pw1_b, pw2_w, pw2_b, gamma,
            )
        }
        None => Ok(run_convnext_cpu(
            x, dw_weight, dw_bias, norm_w, norm_b, pw1_w, pw1_b, pw2_w, pw2_b, gamma,
        )),
    }
}

fn run_convnext_cpu(
    x: &Array2<f32>,
    dw_weight: ArrayView3<f32>,
    dw_bias: ArrayView1<f32>,
    norm_w: ArrayView1<f32>,
    norm_b: ArrayView1<f32>,
    pw1_w: ArrayView2<f32>,
    pw1_b: ArrayView1<f32>,
    pw2_w: ArrayView2<f32>,
    pw2_b: ArrayView1<f32>,
    gamma: ArrayView1<f32>,
) -> Array2<f32> {
    let residual = x.to_owned();
    let h = super::ops::causal_conv1d(x.view(), dw_weight, Some(dw_bias), 1, 1);
    let mut seq = h.t().to_owned();
    seq = layer_norm(seq.view(), norm_w, norm_b, 1e-6);
    seq = linear2(seq.view(), pw1_w, Some(pw1_b));
    seq = gelu(seq.view());
    seq = linear2(seq.view(), pw2_w, Some(pw2_b));
    for mut row in seq.rows_mut() {
        for (v, &g) in row.iter_mut().zip(gamma.iter()) {
            *v *= g;
        }
    }
    residual + seq.t()
}

fn linear2_gpu(
    cache: &mut GpuMatmulCache,
    x: ArrayView2<f32>,
    w: ArrayView2<f32>,
    bias: Option<ArrayView1<f32>>,
) -> Result<Array2<f32>> {
    let (in_dim, w_in) = (x.ncols(), w.nrows());
    if in_dim != w_in {
        return Ok(linear2(x, w, bias));
    }
    match cache.matmul_bt(x, w) {
        Ok(mut out) => {
            if let Some(b) = bias {
                for mut row in out.rows_mut() {
                    row += &b;
                }
            }
            Ok(out)
        }
        Err(_) => Ok(linear2(x, w, bias)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};
    use rlx_runtime::{Device, is_available};

    #[test]
    fn gpu_transconv_matches_cpu_flat() {
        let device = Device::Metal;
        if !GpuMatmulCache::available(device) || !is_available(device) {
            return;
        }
        let in_ch = 8usize;
        let out_ch = 6;
        let k = 2;
        let stride = 2;
        let t_in = 11;
        let mut w = Array3::<f32>::zeros((in_ch, out_ch, k));
        for ((ic, oc, ki), v) in w.indexed_iter_mut() {
            *v = (ic * 17 + oc * 3 + ki) as f32 * 0.01;
        }
        let b = ndarray::Array1::from_vec((0..out_ch).map(|i| i as f32 * 0.001).collect());
        let mut x = Array2::<f32>::zeros((in_ch, t_in));
        for ((ic, ti), v) in x.indexed_iter_mut() {
            *v = ((ic + ti) % 7) as f32 * 0.02 - 0.05;
        }
        let flat = FlatTransConv1d::from_view(w.view(), Some(b.view()), stride);
        let mut ws = ConvWorkspace::new();
        let cpu = super::super::ops::conv_transpose1d_flat(x.view(), &flat, &mut ws);
        let mut cache = GpuMatmulCache::new(device);
        let gpu = conv_transpose1d_gpu_flat(&mut cache, x.view(), &flat);
        assert_eq!(cpu.dim(), gpu.dim());
        let mut max_abs = 0f32;
        for ((ic, ti), cv) in cpu.indexed_iter() {
            max_abs = max_abs.max((cv - gpu[[ic, ti]]).abs());
        }
        assert!(max_abs < 1e-3, "gpu transconv max_abs={max_abs}");
    }

    #[test]
    fn gpu_causal_conv_matches_cpu_flat() {
        let device = Device::Metal;
        if !GpuMatmulCache::available(device) || !is_available(device) {
            return;
        }
        let in_ch = 8usize;
        let out_ch = 6;
        let k = 3;
        let t_in = 15;
        let mut w = Array3::<f32>::zeros((out_ch, in_ch, k));
        for ((oc, ic, ki), v) in w.indexed_iter_mut() {
            *v = (oc * 11 + ic * 5 + ki) as f32 * 0.01;
        }
        let b = ndarray::Array1::from_vec((0..out_ch).map(|i| i as f32 * 0.002).collect());
        let mut x = Array2::<f32>::zeros((in_ch, t_in));
        for ((ic, ti), v) in x.indexed_iter_mut() {
            *v = ((ic + ti) % 5) as f32 * 0.03 - 0.04;
        }
        let flat = FlatConv1d::from_view(w.view(), Some(b.view()), 1, 1);
        let mut ws = ConvWorkspace::new();
        let cpu = super::super::ops::causal_conv1d_flat(x.view(), &flat, &mut ws);
        let mut cache = GpuMatmulCache::new(device);
        let gpu = causal_conv1d_flat_maybe_gpu(x.view(), &flat, &mut ws, Some(&mut cache));
        assert_eq!(cpu.dim(), gpu.dim());
        let mut max_abs = 0f32;
        for ((ic, ti), cv) in cpu.indexed_iter() {
            max_abs = max_abs.max((cv - gpu[[ic, ti]]).abs());
        }
        assert!(max_abs < 1e-3, "gpu causal conv max_abs={max_abs}");
    }
}
