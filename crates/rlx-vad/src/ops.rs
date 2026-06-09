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

//! Shared CPU ops for VAD models (RLX BLAS backend).

use rlx_cpu::blas::{sgemm, sgemm_accumulate};

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `y = alpha * A @ x + beta * y` with A row-major `[m, n]`.
pub fn gemv(m: usize, n: usize, a: &[f32], x: &[f32], beta: f32, y: &mut [f32]) {
    if m == 0 || n == 0 {
        return;
    }
    if beta == 0.0 {
        y[..m].fill(0.0);
        sgemm(a, x, y, m, n, 1);
    } else if beta == 1.0 {
        sgemm_accumulate(a, x, y, m, n, 1);
    } else {
        for v in y.iter_mut().take(m) {
            *v *= beta;
        }
        sgemm_accumulate(a, x, y, m, n, 1);
    }
}

/// PyTorch Conv1d on channel-major `[in_ch, t]` → `[out_ch, t_out]`.
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

/// Write ONNX/NumPy-style reflect extension into `out` (length = pad count).
pub fn fill_reflect_pad_right(x: &[f32], out: &mut [f32]) {
    let t = x.len();
    let pad = out.len();
    if pad == 0 {
        return;
    }
    if t <= 1 {
        out.fill(x[0]);
        return;
    }
    let period = 2 * (t - 1);
    for i in 0..pad {
        let mut p = (t - 1) + i + 1;
        while p >= t {
            p = period - p;
        }
        while p > t - 1 {
            p = period - p;
        }
        out[i] = x[p];
    }
}

/// Reflect-pad the right edge of mono `[t]` by `pad` samples (ONNX / NumPy `reflect`, no edge repeat).
pub fn pad1d_reflect_right(x: &[f32], pad: usize, out: &mut [f32]) {
    let t = x.len();
    out[..t].copy_from_slice(x);
    fill_reflect_pad_right(x, &mut out[t..t + pad]);
}

/// LSTM cell step: `x` `[input]`, `(h, c)` each `[hidden]` → new `(h, c)`.
/// `gates_scratch` must hold at least `4 * hidden_size` elements.
pub fn lstm_cell_step(
    x: &[f32],
    h: &[f32],
    c: &[f32],
    w_ih: &[f32],
    w_hh: &[f32],
    b_ih: &[f32],
    b_hh: &[f32],
    input_size: usize,
    hidden_size: usize,
    h_out: &mut [f32],
    c_out: &mut [f32],
    gates_scratch: &mut [f32],
) {
    let gates = hidden_size * 4;
    debug_assert!(gates_scratch.len() >= gates);
    let g = &mut gates_scratch[..gates];
    gemv(gates, input_size, w_ih, x, 0.0, g);
    for i in 0..gates {
        g[i] += b_ih[i];
    }
    gemv(gates, hidden_size, w_hh, h, 1.0, g);
    for i in 0..gates {
        g[i] += b_hh[i];
    }
    for i in 0..hidden_size {
        let i_gate = sigmoid(g[i]);
        let f_gate = sigmoid(g[hidden_size + i]);
        let o_gate = sigmoid(g[2 * hidden_size + i]);
        let c_gate = g[3 * hidden_size + i].tanh();
        c_out[i] = f_gate * c[i] + i_gate * c_gate;
        h_out[i] = o_gate * c_out[i].tanh();
    }
}
