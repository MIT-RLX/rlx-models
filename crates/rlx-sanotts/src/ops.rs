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

//! Host-eager kernels for the sanoTTS student stack — the CPU reference path.
//!
//! Everything is channel-major `[C, T]` (see [`Mat`]), matching both the numpy
//! reference and the `[1, C, T, 1]` NCHW layout the graph path feeds to rlx.

/// A row-major `[rows, cols]` matrix; rows are channels, cols are time steps.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mat {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn from_vec(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        debug_assert_eq!(data.len(), rows * cols);
        Self { data, rows, cols }
    }

    #[inline]
    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    #[inline]
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        let c = self.cols;
        &mut self.data[r * c..(r + 1) * c]
    }

    /// Stack `self` on top of `other` (same column count).
    pub fn vstack(&self, other: &Mat) -> Mat {
        debug_assert_eq!(self.cols, other.cols);
        let mut data = Vec::with_capacity(self.data.len() + other.data.len());
        data.extend_from_slice(&self.data);
        data.extend_from_slice(&other.data);
        Mat::from_vec(data, self.rows + other.rows, self.cols)
    }
}

/// `x / (1 + exp(-x))`.
#[inline]
pub fn silu_(x: &mut [f32]) {
    for v in x {
        *v /= 1.0 + (-*v).exp();
    }
}

/// `x if x > 0 else slope * x` — note the strict `>`, matching the reference.
#[inline]
pub fn leaky_relu_(x: &mut [f32], slope: f32) {
    for v in x {
        if *v <= 0.0 {
            *v *= slope;
        }
    }
}

#[inline]
pub fn tanh_(x: &mut [f32]) {
    for v in x {
        *v = v.tanh();
    }
}

/// `n` evenly spaced values in `[0, 1]` (`0` when `n == 1`), as numpy's
/// `linspace(0, 1, n)` computes them in float64 before the f32 cast.
pub fn linspace01(n: usize) -> Vec<f32> {
    match n {
        0 => Vec::new(),
        1 => vec![0.0],
        _ => {
            let last = (n - 1) as f64;
            (0..n).map(|i| (i as f64 / last) as f32).collect()
        }
    }
}

/// PyTorch `Conv1d` with "same" padding: `pad = dilation * (K / 2)`.
///
/// `x: [in_ch, T]`, `w: [out_ch, in_ch, K]` (flat), `b: [out_ch]` → `[out_ch, T]`.
pub fn conv1d_same(x: &Mat, w: &[f32], b: &[f32], out_ch: usize, k: usize, dilation: usize) -> Mat {
    let in_ch = x.rows;
    let t = x.cols;
    debug_assert_eq!(w.len(), out_ch * in_ch * k);
    let pad = (dilation * (k / 2)) as isize;
    let mut out = Mat::zeros(out_ch, t);
    for oc in 0..out_ch {
        out.row_mut(oc).fill(b[oc]);
    }
    let ti = t as isize;
    for kk in 0..k {
        let off = (kk * dilation) as isize - pad;
        let lo = (-off).max(0);
        let hi = (ti - off).min(ti);
        if hi <= lo {
            continue;
        }
        let (lo, hi) = (lo as usize, hi as usize);
        let span = hi - lo;
        let src_lo = (lo as isize + off) as usize;
        for oc in 0..out_ch {
            for ic in 0..in_ch {
                let wv = w[(oc * in_ch + ic) * k + kk];
                if wv == 0.0 {
                    continue;
                }
                let src = &x.data[ic * t + src_lo..ic * t + src_lo + span];
                let dst = &mut out.data[oc * t + lo..oc * t + lo + span];
                for (d, s) in dst.iter_mut().zip(src) {
                    *d += wv * s;
                }
            }
        }
    }
    out
}

/// `Conv1d` with `kernel_size = 1`: a per-timestep linear projection.
pub fn conv1d_1x1(x: &Mat, w: &[f32], b: &[f32], out_ch: usize) -> Mat {
    let in_ch = x.rows;
    let t = x.cols;
    debug_assert_eq!(w.len(), out_ch * in_ch);
    let mut out = Mat::zeros(out_ch, t);
    for oc in 0..out_ch {
        let dst = &mut out.data[oc * t..(oc + 1) * t];
        dst.fill(b[oc]);
        for ic in 0..in_ch {
            let wv = w[oc * in_ch + ic];
            if wv == 0.0 {
                continue;
            }
            let src = &x.data[ic * t..(ic + 1) * t];
            for (d, s) in dst.iter_mut().zip(src) {
                *d += wv * s;
            }
        }
    }
    out
}

/// PyTorch `ConvTranspose1d`. `x: [in_ch, T]`, `w: [in_ch, out_ch, K]`, `b: [out_ch]`.
///
/// Output length is `(T - 1) * stride - 2 * padding + K`.
pub fn conv_transpose1d(
    x: &Mat,
    w: &[f32],
    b: &[f32],
    out_ch: usize,
    k: usize,
    stride: usize,
    padding: usize,
) -> Mat {
    let in_ch = x.rows;
    let t = x.cols;
    debug_assert_eq!(w.len(), in_ch * out_ch * k);
    let l = (t - 1) * stride + k - 2 * padding;
    let mut out = Mat::zeros(out_ch, l);
    for oc in 0..out_ch {
        out.row_mut(oc).fill(b[oc]);
    }
    for kk in 0..k {
        let shift = kk as isize - padding as isize;
        // Input steps whose contribution lands inside [0, l).
        let t_lo = if shift < 0 {
            ((-shift) as usize).div_ceil(stride)
        } else {
            0
        };
        let t_hi = if shift >= l as isize {
            0
        } else {
            (((l as isize - 1 - shift) as usize) / stride + 1).min(t)
        };
        if t_hi <= t_lo {
            continue;
        }
        let j_start = (t_lo * stride) as isize + shift;
        debug_assert!(j_start >= 0);
        let j_start = j_start as usize;
        for ic in 0..in_ch {
            let xrow = &x.data[ic * t..(ic + 1) * t];
            for oc in 0..out_ch {
                let wv = w[(ic * out_ch + oc) * k + kk];
                if wv == 0.0 {
                    continue;
                }
                let orow = &mut out.data[oc * l..(oc + 1) * l];
                let mut j = j_start;
                for &xv in &xrow[t_lo..t_hi] {
                    orow[j] += wv * xv;
                    j += stride;
                }
            }
        }
    }
    out
}

/// One `ResidualConvBlock`: `x + scale * conv2(silu(conv1(x)))`.
///
/// Kernel size is read from the stored weight rather than the config, so it
/// always matches what the checkpoint actually holds.
pub fn residual_conv_block(
    x: &Mat,
    ts: &crate::voicepack::TensorStore,
    prefix: &str,
) -> anyhow::Result<Mat> {
    let scale = ts.scalar(&format!("{prefix}.scale"))?;
    let (w1, s1) = ts.get(&format!("{prefix}.net.0.weight"))?;
    let b1 = ts.data(&format!("{prefix}.net.0.bias"))?;
    let mut t = conv1d_same(x, w1, b1, s1[0], s1[2], 1);
    silu_(&mut t.data);
    let (w2, s2) = ts.get(&format!("{prefix}.net.2.weight"))?;
    let b2 = ts.data(&format!("{prefix}.net.2.bias"))?;
    let u = conv1d_same(&t, w2, b2, s2[0], s2[2], 1);
    let mut out = x.clone();
    for (o, uv) in out.data.iter_mut().zip(&u.data) {
        *o += scale * uv;
    }
    Ok(out)
}
