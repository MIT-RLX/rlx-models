//! Host-eager NN ops (ndarray). These are the parity reference and the CPU
//! fallback; the rlx-ir graph path mirrors them op-for-op for other backends.
//!
//! Conventions: time-major activations `[T, C]` (matching PyTorch `[B=1, T, C]`);
//! conv tensors are channel-major `[C, T]`.

use ndarray::{Array2, ArrayView2};

/// Wrap a flat `[rows*cols]` slice as a `[rows, cols]` view.
pub fn view2<'a>(data: &'a [f32], rows: usize, cols: usize) -> ArrayView2<'a, f32> {
    ArrayView2::from_shape((rows, cols), data).expect("view2 shape")
}

/// `y = x @ wᵀ + b`, with `x: [T, in]`, `w: [out, in]` (PyTorch Linear layout).
pub fn linear(
    x: &Array2<f32>,
    w: &[f32],
    out: usize,
    inp: usize,
    b: Option<&[f32]>,
) -> Array2<f32> {
    let w = view2(w, out, inp); // [out, in]
    let mut y = x.dot(&w.t()); // [T, out]
    if let Some(b) = b {
        for mut row in y.rows_mut() {
            for (v, bi) in row.iter_mut().zip(b.iter()) {
                *v += *bi;
            }
        }
    }
    y
}

/// LayerNorm over the last axis (channels). `eps` default 1e-5 (nn.LayerNorm).
pub fn layer_norm(x: &Array2<f32>, gamma: &[f32], beta: &[f32], eps: f32) -> Array2<f32> {
    let (t, c) = x.dim();
    let mut y = Array2::<f32>::zeros((t, c));
    for i in 0..t {
        let row = x.row(i);
        let mean = row.sum() / c as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for j in 0..c {
            y[[i, j]] = (row[j] - mean) * inv * gamma[j] + beta[j];
        }
    }
    y
}

#[inline]
pub fn sigmoid_(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

pub fn silu_inplace(x: &mut Array2<f32>) {
    x.mapv_inplace(|v| v * sigmoid_(v));
}

/// Grouped/standard 1-D convolution via im2col + matmul (fast + backend-portable).
/// `x: [c_in, t]`, `w: [c_out, c_in/groups, k]` (flat), `b: [c_out]`.
#[allow(clippy::too_many_arguments)]
pub fn conv1d(
    x: &Array2<f32>,
    w: &[f32],
    c_out: usize,
    c_in_g: usize,
    k: usize,
    bias: Option<&[f32]>,
    stride: usize,
    pad: usize,
    dilation: usize,
    groups: usize,
) -> Array2<f32> {
    let (c_in, t) = x.dim();
    debug_assert_eq!(c_in, c_in_g * groups);
    let c_out_g = c_out / groups;
    let t_out = (t + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
    let mut y = Array2::<f32>::zeros((c_out, t_out));
    for g in 0..groups {
        // im2col: cols[c_in_g*k, t_out]
        let mut cols = Array2::<f32>::zeros((c_in_g * k, t_out));
        for icg in 0..c_in_g {
            let xrow = x.row(g * c_in_g + icg);
            for kk in 0..k {
                let row = icg * k + kk;
                let off = kk * dilation;
                for ot in 0..t_out {
                    let pos = ot * stride + off;
                    if pos >= pad && pos - pad < t {
                        cols[[row, ot]] = xrow[pos - pad];
                    }
                }
            }
        }
        // wg[c_out_g, c_in_g*k] @ cols → [c_out_g, t_out]
        let wg = view2(&w[g * c_out_g * c_in_g * k..], c_out_g, c_in_g * k);
        let res = wg.dot(&cols);
        for ocg in 0..c_out_g {
            let oc = g * c_out_g + ocg;
            let bz = bias.map(|b| b[oc]).unwrap_or(0.0);
            for ot in 0..t_out {
                y[[oc, ot]] = res[[ocg, ot]] + bz;
            }
        }
    }
    y
}

/// 1-D transposed convolution (groups=1 path is all we need).
/// `x: [c_in, t]`, `w: [c_in, c_out, k]` (PyTorch ConvTranspose1d layout), `b: [c_out]`.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d(
    x: &Array2<f32>,
    w: &[f32],
    c_in: usize,
    c_out: usize,
    k: usize,
    bias: Option<&[f32]>,
    stride: usize,
    pad: usize,
) -> Array2<f32> {
    let (cin, t) = x.dim();
    debug_assert_eq!(cin, c_in);
    let t_full = (t - 1) * stride + k; // before cropping padding
    let t_out = t_full - 2 * pad;
    // M[t, c_out*k] = xᵀ[t, c_in] @ w[c_in, c_out*k]
    let xt = x.t().to_owned(); // [t, c_in]
    let wmat = view2(w, c_in, c_out * k); // [c_in, c_out*k]
    let m = xt.dot(&wmat); // [t, c_out*k]
    let mut y = Array2::<f32>::zeros((c_out, t_out));
    // col2im overlap-add with stride
    for it in 0..t {
        let base = it * stride;
        for kk in 0..k {
            let full_pos = base + kk;
            if full_pos < pad || full_pos - pad >= t_out {
                continue;
            }
            let ot = full_pos - pad;
            for oc in 0..c_out {
                y[[oc, ot]] += m[[it, oc * k + kk]];
            }
        }
    }
    if let Some(b) = bias {
        for oc in 0..c_out {
            let bz = b[oc];
            for ot in 0..t_out {
                y[[oc, ot]] += bz;
            }
        }
    }
    y
}

/// Bidirectional single-layer GRU. `x: [T, in]`. Returns `[T, 2*hidden]`
/// (forward dir in `[..hidden]`, reverse in `[hidden..]`), matching PyTorch
/// `nn.GRU(batch_first=True, bidirectional=True)`.
pub struct GruWeights<'a> {
    pub w_ih: &'a [f32], // [3*hidden, in]
    pub w_hh: &'a [f32], // [3*hidden, hidden]
    pub b_ih: &'a [f32], // [3*hidden]
    pub b_hh: &'a [f32], // [3*hidden]
}

fn gru_dir(
    x: &Array2<f32>,
    w: &GruWeights,
    hidden: usize,
    inp: usize,
    reverse: bool,
) -> Array2<f32> {
    let t = x.dim().0;
    let w_ih = view2(w.w_ih, 3 * hidden, inp);
    let w_hh = view2(w.w_hh, 3 * hidden, hidden);
    let mut out = Array2::<f32>::zeros((t, hidden));
    let mut h = vec![0f32; hidden];
    for step in 0..t {
        let ti = if reverse { t - 1 - step } else { step };
        let xt = x.row(ti);
        // gi = W_ih x + b_ih ; gh = W_hh h + b_hh   (PyTorch GRU gate order: r, z, n)
        let mut gi = vec![0f32; 3 * hidden];
        let mut gh = vec![0f32; 3 * hidden];
        for o in 0..3 * hidden {
            let mut a = w.b_ih[o];
            let irow = w_ih.row(o);
            for j in 0..inp {
                a += irow[j] * xt[j];
            }
            gi[o] = a;
            let mut c = w.b_hh[o];
            let hrow = w_hh.row(o);
            for j in 0..hidden {
                c += hrow[j] * h[j];
            }
            gh[o] = c;
        }
        let mut new_h = vec![0f32; hidden];
        for j in 0..hidden {
            let r = sigmoid_(gi[j] + gh[j]);
            let z = sigmoid_(gi[hidden + j] + gh[hidden + j]);
            let n = (gi[2 * hidden + j] + r * gh[2 * hidden + j]).tanh();
            new_h[j] = (1.0 - z) * n + z * h[j];
        }
        h = new_h;
        for j in 0..hidden {
            out[[ti, j]] = h[j];
        }
    }
    out
}

pub fn bidirectional_gru(
    x: &Array2<f32>,
    fwd: &GruWeights,
    rev: &GruWeights,
    hidden: usize,
    inp: usize,
) -> Array2<f32> {
    let f = gru_dir(x, fwd, hidden, inp, false);
    let r = gru_dir(x, rev, hidden, inp, true);
    let t = x.dim().0;
    let mut out = Array2::<f32>::zeros((t, 2 * hidden));
    for i in 0..t {
        for j in 0..hidden {
            out[[i, j]] = f[[i, j]];
            out[[i, hidden + j]] = r[[i, j]];
        }
    }
    out
}
