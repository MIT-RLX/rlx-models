// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Ndarray primitives for the SNAC decoder (channels-first `[C, T]`).

use ndarray::{Array2, ArrayView1, ArrayView2, ArrayView3, Axis, s};

/// Snake1d: `x + sin(alpha * x)^2 / (alpha + 1e-9)` per channel.
pub fn snake1d(x: ArrayView2<f32>, alpha: ArrayView1<f32>) -> Array2<f32> {
    let (c, t) = (x.shape()[0], x.shape()[1]);
    assert_eq!(alpha.len(), c);
    let mut out = Array2::<f32>::zeros((c, t));
    for ci in 0..c {
        let a = alpha[ci];
        let inv_a = 1.0 / (a + 1e-9);
        for ti in 0..t {
            let v = x[[ci, ti]];
            let s = (a * v).sin();
            out[[ci, ti]] = v + inv_a * s * s;
        }
    }
    out
}

/// Conv1d with same-length output (symmetric zero pad).
///
/// * `x`: `[C_in, T]`
/// * `w`: `[C_out, C_in / groups, K]`
pub fn conv1d(
    x: ArrayView2<f32>,
    w: ArrayView3<f32>,
    b: Option<ArrayView1<f32>>,
    pad: usize,
    groups: usize,
    dilation: usize,
) -> Array2<f32> {
    let (c_in, t) = (x.shape()[0], x.shape()[1]);
    let (c_out, _w_in_per_g, k) = (w.shape()[0], w.shape()[1], w.shape()[2]);
    assert_eq!(c_in % groups, 0);
    assert_eq!(c_out % groups, 0);
    let c_in_per_g = c_in / groups;
    let dil = dilation.max(1);

    let mut out = Array2::<f32>::zeros((c_out, t));
    for g in 0..groups {
        let c_in_start = g * c_in_per_g;
        let c_out_start = g * (c_out / groups);
        for co in 0..(c_out / groups) {
            let oc = c_out_start + co;
            for ti in 0..t {
                let mut acc = 0.0f32;
                for ci in 0..c_in_per_g {
                    let ic = c_in_start + ci;
                    for ki in 0..k {
                        let src = ti + ki * dil;
                        if src >= pad && src < t + pad {
                            acc += x[[ic, src - pad]] * w[[oc, ci, ki]];
                        }
                    }
                }
                out[[oc, ti]] = acc;
            }
        }
    }

    if let Some(b) = b {
        out += &b.view().insert_axis(Axis(1));
    }
    out
}

/// ConvTranspose1d matching PyTorch `ConvTranspose1d` (channels-first).
///
/// * `x`: `[C_in, T_in]`
/// * `w`: `[C_in, C_out / groups, K]`
pub fn conv_transpose1d(
    x: ArrayView2<f32>,
    w: ArrayView3<f32>,
    b: Option<ArrayView1<f32>>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    groups: usize,
) -> Array2<f32> {
    let (c_in, t_in) = (x.shape()[0], x.shape()[1]);
    let (w_in, w_out_per_g, k) = (w.shape()[0], w.shape()[1], w.shape()[2]);
    assert_eq!(c_in, w_in);
    assert_eq!(c_in % groups, 0);

    let c_out = w_out_per_g * groups;
    let t_out = (t_in - 1) * stride + k - 2 * padding + output_padding;
    let mut out = Array2::<f32>::zeros((c_out, t_out));

    for g in 0..groups {
        let c_in_start = g * (c_in / groups);
        let c_out_start = g * w_out_per_g;
        for ti in 0..t_in {
            for ci in 0..(c_in / groups) {
                let ic = c_in_start + ci;
                for ki in 0..k {
                    let to = ti * stride + ki;
                    if to < padding || to >= t_out + padding {
                        continue;
                    }
                    let out_t = to - padding;
                    if out_t >= t_out {
                        continue;
                    }
                    for co in 0..w_out_per_g {
                        let oc = c_out_start + co;
                        out[[oc, out_t]] += x[[ic, ti]] * w[[ic, co, ki]];
                    }
                }
            }
        }
    }

    if let Some(b) = b {
        out += &b.view().insert_axis(Axis(1));
    }
    out
}

/// `repeat_interleave` along the time axis (last dim of `[C, T]`).
pub fn repeat_interleave_time(x: ArrayView2<f32>, repeats: usize) -> Array2<f32> {
    if repeats <= 1 {
        return x.to_owned();
    }
    let (c, t) = (x.shape()[0], x.shape()[1]);
    let mut out = Array2::<f32>::zeros((c, t * repeats));
    for ci in 0..c {
        for ti in 0..t {
            let v = x[[ci, ti]];
            for r in 0..repeats {
                out[[ci, ti * repeats + r]] = v;
            }
        }
    }
    out
}

/// Embedding lookup: `codes[T]` + `codebook[n, d]` → `[d, T]`.
pub fn embed_codes(codes: &[i32], codebook: ArrayView2<f32>) -> Array2<f32> {
    let d = codebook.shape()[1];
    let mut out = Array2::<f32>::zeros((d, codes.len()));
    for (ti, &code) in codes.iter().enumerate() {
        let row = codebook.row(code as usize);
        for di in 0..d {
            out[[di, ti]] = row[di];
        }
    }
    out
}

/// ResidualUnit center-crop add (matches SNAC / DAC).
pub fn residual_add(x: ArrayView2<f32>, y: ArrayView2<f32>) -> Array2<f32> {
    let t_x = x.shape()[1];
    let t_y = y.shape()[1];
    if t_y == t_x {
        return &x + &y;
    }
    let pad = (t_x - t_y) / 2;
    let x_crop = x.slice(s![.., pad..pad + t_y]);
    &x_crop + &y
}

/// NoiseBlock: `x + randn(1,T) * conv1x1(x)` (matches SNAC PyTorch).
pub fn noise_block(
    x: ArrayView2<f32>,
    w: ArrayView3<f32>,
    noise_time: ArrayView1<f32>,
) -> Array2<f32> {
    let mut h = conv1d(x, w, None, 0, 1, 1);
    let t = x.shape()[1];
    debug_assert_eq!(noise_time.len(), t);
    for ti in 0..t {
        let n = noise_time[ti];
        for ci in 0..h.shape()[0] {
            h[[ci, ti]] *= n;
        }
    }
    &x + &h
}

/// Deterministic decode (legacy / tests without noise fixtures).
pub fn noise_block_zero(x: ArrayView2<f32>) -> Array2<f32> {
    x.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array3};

    #[test]
    fn conv1d_same_length() {
        let x = Array2::<f32>::ones((4, 8));
        let w = Array3::<f32>::ones((8, 4, 3));
        let b = Array1::<f32>::zeros(8);
        let out = conv1d(x.view(), w.view(), Some(b.view()), 1, 1, 1);
        assert_eq!(out.shape(), &[8, 8]);
    }

    #[test]
    fn conv_transpose_doubles_length_stride2() {
        let x = Array2::<f32>::ones((2, 4));
        let w = Array3::<f32>::ones((2, 2, 4));
        let out = conv_transpose1d(x.view(), w.view(), None, 2, 1, 0, 1);
        assert_eq!(out.shape()[1], 8);
    }
}
