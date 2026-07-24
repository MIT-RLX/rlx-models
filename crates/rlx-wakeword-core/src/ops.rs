// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Pure f32 ops (no BLAS) for embedded ports.

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// `y = A @ x` with A row-major `[m, n]`.
pub fn gemv(m: usize, n: usize, a: &[f32], x: &[f32], y: &mut [f32]) {
    for i in 0..m {
        let mut s = 0.0f32;
        let row = i * n;
        for j in 0..n {
            s += a[row + j] * x[j];
        }
        y[i] = s;
    }
}

pub fn gemv_bias(m: usize, n: usize, a: &[f32], x: &[f32], bias: &[f32], y: &mut [f32]) {
    gemv(m, n, a, x, y);
    for i in 0..m {
        y[i] += bias[i];
    }
}

/// Conv1d on channel-major `[in_ch, t]` → `[out_ch, t_out]`.
pub fn conv1d_nchw(
    x: &[f32],
    in_ch: usize,
    t_in: usize,
    w: &[f32],
    out_ch: usize,
    k: usize,
    stride: usize,
    pad: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> usize {
    let t_out = if t_in + 2 * pad >= k {
        (t_in + 2 * pad - k) / stride + 1
    } else {
        0
    };
    out.fill(0.0);
    for oc in 0..out_ch {
        for ot in 0..t_out {
            let mut sum = bias.map(|b| b[oc]).unwrap_or(0.0);
            for ic in 0..in_ch {
                for ki in 0..k {
                    let ti = ot * stride + ki;
                    let ti = ti as isize - pad as isize;
                    if ti < 0 || ti >= t_in as isize {
                        continue;
                    }
                    let x_idx = ic * t_in + ti as usize;
                    let w_idx = oc * (in_ch * k) + ic * k + ki;
                    sum += x[x_idx] * w[w_idx];
                }
            }
            out[oc * t_out + ot] = sum;
        }
    }
    t_out
}

pub fn global_mean_pool_chw(x: &[f32], ch: usize, spatial: usize, out: &mut [f32]) {
    let inv = if spatial == 0 {
        0.0
    } else {
        1.0 / spatial as f32
    };
    for c in 0..ch {
        let mut s = 0.0f32;
        let base = c * spatial;
        for i in 0..spatial {
            s += x[base + i];
        }
        out[c] = s * inv;
    }
}
