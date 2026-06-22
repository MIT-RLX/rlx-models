use ndarray::{Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};

/// Snake1d: `x + sin(alpha * x)^2 / (alpha + 1e-9)` per channel.
pub fn snake1d(x: ArrayView2<f32>, alpha: ArrayView1<f32>) -> Array2<f32> {
    snake1d_mode(x, alpha, false)
}

/// Snake activation. `c_style=false` is standard per-channel DAC snake
/// (`x + sin(αx)²/(α+1e-9)`). `c_style=true` reproduces TSAC's vendored
/// `snake_s`, which indexes `alpha[(ci*T+ti) % n_alpha]` over the channel-major
/// `[C][T]` buffer (i.e. alpha is *not* strictly per-channel) and clamps α to
/// 1e-6. The q8 weights were calibrated against that decoder, so TSAC needs it.
pub fn snake1d_mode(x: ArrayView2<f32>, alpha: ArrayView1<f32>, c_style: bool) -> Array2<f32> {
    let (c, t) = (x.shape()[0], x.shape()[1]);
    let na = alpha.len();
    let mut out = Array2::<f32>::zeros((c, t));
    for ci in 0..c {
        for ti in 0..t {
            let a = if c_style {
                let a = alpha[(ci * t + ti) % na];
                if a < 1e-6 { 1e-6 } else { a }
            } else {
                alpha[ci]
            };
            let inv_a = if c_style { 1.0 / a } else { 1.0 / (a + 1e-9) };
            let v = x[[ci, ti]];
            let s = (a * v).sin();
            out[[ci, ti]] = v + inv_a * s * s;
        }
    }
    out
}

/// Apply PyTorch `weight_norm` on Conv1d weights: `g * v / ||v||`.
pub fn weight_norm(g: ArrayView3<f32>, v: ArrayView3<f32>) -> Array3<f32> {
    let (out_ch, in_ch, k) = v.dim();
    let mut w = Array3::<f32>::zeros((out_ch, in_ch, k));
    for oc in 0..out_ch {
        let mut norm_sq = 0.0f32;
        for ic in 0..in_ch {
            for ki in 0..k {
                let x = v[[oc, ic, ki]];
                norm_sq += x * x;
            }
        }
        let scale = g[[oc, 0, 0]] / norm_sq.sqrt();
        for ic in 0..in_ch {
            for ki in 0..k {
                w[[oc, ic, ki]] = v[[oc, ic, ki]] * scale;
            }
        }
    }
    w
}

/// Build the im2col matrix `[c_in*k, t_out]` (row-major) for a 1-D conv whose
/// input tap for output column `to`, kernel index `ki` is `to*stride + ki*dil -
/// pad` (out-of-range ⇒ 0). Row order `ic*k + ki` matches the weight's
/// `[c_out, c_in, k]` flattening, so `W2d @ cols` is the convolution.
fn im2col_1d(
    x: ArrayView2<f32>,
    c_in: usize,
    t_in: usize,
    k: usize,
    t_out: usize,
    stride: usize,
    dil: usize,
    pad: usize,
) -> Vec<f32> {
    let mut cols = vec![0f32; c_in * k * t_out];
    for ic in 0..c_in {
        for ki in 0..k {
            let base = (ic * k + ki) * t_out;
            let off = ki * dil;
            for to in 0..t_out {
                let src = to * stride + off;
                if src >= pad {
                    let xi = src - pad;
                    if xi < t_in {
                        cols[base + to] = x[[ic, xi]];
                    }
                }
            }
        }
    }
    cols
}

/// Core conv as `W2d[c_out, c_in*k] @ im2col[c_in*k, t_out]` via BLAS sgemm
/// (Accelerate/OpenBLAS through `rlx-tensor`), replacing the naive
/// `O(c_out·c_in·k·t)` loop. Groups are not used by DAC, so this is the groups=1
/// path; callers with `groups > 1` fall back to the scalar reference.
fn conv1d_gemm(
    x: ArrayView2<f32>,
    w: ArrayView3<f32>,
    t_out: usize,
    stride: usize,
    dil: usize,
    pad: usize,
) -> Array2<f32> {
    let (c_in, t_in) = (x.shape()[0], x.shape()[1]);
    let (c_out, _, k) = (w.shape()[0], w.shape()[1], w.shape()[2]);
    let cols = im2col_1d(x, c_in, t_in, k, t_out, stride, dil, pad);
    // Weight as a contiguous `[c_out, c_in*k]` matrix.
    let w_std = w.as_standard_layout();
    let wflat = w_std.as_slice().expect("weight standard layout");
    let mut out = vec![0f32; c_out * t_out];
    rlx_cpu::blas::sgemm(wflat, &cols, &mut out, c_out, c_in * k, t_out);
    Array2::from_shape_vec((c_out, t_out), out).expect("conv out reshape")
}

/// Conv1d with symmetric zero padding (same length when `stride == 1`).
pub fn conv1d_same(
    x: ArrayView2<f32>,
    w: ArrayView3<f32>,
    b: Option<ArrayView1<f32>>,
    pad: usize,
    groups: usize,
    dilation: usize,
) -> Array2<f32> {
    let t = x.shape()[1];
    let dil = dilation.max(1);
    let mut out = if groups == 1 {
        conv1d_gemm(x, w, t, 1, dil, pad)
    } else {
        conv1d_same_naive(x, w, pad, groups, dil)
    };
    if let Some(b) = b {
        out += &b.view().insert_axis(Axis(1));
    }
    out
}

/// Strided Conv1d: `L_out = (L_in + 2*pad - k) / stride + 1`.
pub fn conv1d_stride(
    x: ArrayView2<f32>,
    w: ArrayView3<f32>,
    b: Option<ArrayView1<f32>>,
    stride: usize,
    pad: usize,
) -> Array2<f32> {
    let (_, t_in) = (x.shape()[0], x.shape()[1]);
    let k = w.shape()[2];
    let t_out = (t_in + 2 * pad).saturating_sub(k) / stride + 1;
    let mut out = conv1d_gemm(x, w, t_out, stride, 1, pad);
    if let Some(b) = b {
        out += &b.view().insert_axis(Axis(1));
    }
    out
}

/// ConvTranspose1d matching PyTorch (channels-first, `output_padding = 0`).
/// `out_offset` shifts the (length-preserving) crop window right — 0 is PyTorch;
/// TSAC's C decoder uses `out_offset = padding` (kernel centered at `K/2`).
///
/// Computed as `M[c_out*k, t_in] = Wt[c_out*k, c_in] @ x[c_in, t_in]` (BLAS
/// sgemm) followed by a strided scatter-add into the output, instead of the
/// naive `O(c_in·c_out·k·t)` loop.
pub fn conv_transpose1d(
    x: ArrayView2<f32>,
    w: ArrayView3<f32>,
    b: Option<ArrayView1<f32>>,
    stride: usize,
    padding: usize,
    out_offset: usize,
) -> Array2<f32> {
    let (c_in, t_in) = (x.shape()[0], x.shape()[1]);
    let (w_in, c_out, k) = (w.shape()[0], w.shape()[1], w.shape()[2]);
    debug_assert_eq!(c_in, w_in);
    let t_out = (t_in - 1) * stride + k - 2 * padding;
    let off = padding + out_offset;

    // Repack weight `[c_in, c_out, k]` → `Wt[c_out*k, c_in]` so a single sgemm
    // yields every `(oc, ki)` contribution for all input columns at once.
    let mut wt = vec![0f32; c_out * k * c_in];
    for ic in 0..c_in {
        for oc in 0..c_out {
            for ki in 0..k {
                wt[(oc * k + ki) * c_in + ic] = w[[ic, oc, ki]];
            }
        }
    }
    let x_std = x.as_standard_layout();
    let xflat = x_std.as_slice().expect("convT input standard layout");
    let mut m = vec![0f32; c_out * k * t_in];
    rlx_cpu::blas::sgemm(&wt, xflat, &mut m, c_out * k, c_in, t_in);

    let mut out = Array2::<f32>::zeros((c_out, t_out));
    for oc in 0..c_out {
        for ki in 0..k {
            let row = (oc * k + ki) * t_in;
            for ti in 0..t_in {
                let to = ti * stride + ki;
                if to < off || to >= t_out + off {
                    continue;
                }
                out[[oc, to - off]] += m[row + ti];
            }
        }
    }

    if let Some(b) = b {
        out += &b.view().insert_axis(Axis(1));
    }
    out
}

/// Scalar reference conv used only for grouped convs (DAC uses groups=1) and the
/// BLAS-parity unit test.
fn conv1d_same_naive(
    x: ArrayView2<f32>,
    w: ArrayView3<f32>,
    pad: usize,
    groups: usize,
    dil: usize,
) -> Array2<f32> {
    let (c_in, t) = (x.shape()[0], x.shape()[1]);
    let (c_out, _w_in_per_g, k) = (w.shape()[0], w.shape()[1], w.shape()[2]);
    let c_in_per_g = c_in / groups;
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
    out
}

/// ResidualUnit center-crop add (matches DAC / SNAC).
pub fn residual_add(x: ArrayView2<f32>, y: ArrayView2<f32>) -> Array2<f32> {
    let t_x = x.shape()[1];
    let t_y = y.shape()[1];
    if t_y == t_x {
        return &x + &y;
    }
    let pad = (t_x - t_y) / 2;
    let mut out = y.to_owned();
    for c in 0..x.shape()[0] {
        for ti in 0..t_y {
            out[[c, ti]] += x[[c, pad + ti]];
        }
    }
    out
}

pub fn tanh_inplace(x: &mut Array2<f32>) {
    for v in x.iter_mut() {
        *v = v.tanh();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn conv1d_stride_lengths() {
        let x = Array2::<f32>::ones((8, 100));
        let w = Array3::<f32>::ones((16, 8, 4));
        let b = ndarray::Array1::<f32>::zeros(16);
        let out = conv1d_stride(x.view(), w.view(), Some(b.view()), 2, 1);
        assert_eq!(out.shape()[1], 50);
    }

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
        fn arr2(&mut self, a: usize, b: usize) -> Array2<f32> {
            Array2::from_shape_fn((a, b), |_| self.f())
        }
        fn arr3(&mut self, a: usize, b: usize, c: usize) -> Array3<f32> {
            Array3::from_shape_fn((a, b, c), |_| self.f() * 0.3)
        }
    }

    fn max_abs(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        assert_eq!(a.dim(), b.dim());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// Naive strided conv reference (the pre-BLAS implementation).
    fn conv1d_stride_naive(
        x: ArrayView2<f32>,
        w: ArrayView3<f32>,
        stride: usize,
        pad: usize,
    ) -> Array2<f32> {
        let (c_in, t_in) = (x.shape()[0], x.shape()[1]);
        let (c_out, _, k) = (w.shape()[0], w.shape()[1], w.shape()[2]);
        let t_out = (t_in + 2 * pad).saturating_sub(k) / stride + 1;
        let mut out = Array2::<f32>::zeros((c_out, t_out));
        for oc in 0..c_out {
            for ti in 0..t_out {
                let mut acc = 0.0f32;
                for ic in 0..c_in {
                    for ki in 0..k {
                        let src = ti * stride + ki;
                        if src >= pad && src < t_in + pad {
                            acc += x[[ic, src - pad]] * w[[oc, ic, ki]];
                        }
                    }
                }
                out[[oc, ti]] = acc;
            }
        }
        out
    }

    fn conv_transpose1d_naive(
        x: ArrayView2<f32>,
        w: ArrayView3<f32>,
        stride: usize,
        padding: usize,
        out_offset: usize,
    ) -> Array2<f32> {
        let (c_in, t_in) = (x.shape()[0], x.shape()[1]);
        let (_, c_out, k) = (w.shape()[0], w.shape()[1], w.shape()[2]);
        let t_out = (t_in - 1) * stride + k - 2 * padding;
        let off = padding + out_offset;
        let mut out = Array2::<f32>::zeros((c_out, t_out));
        for ti in 0..t_in {
            for ic in 0..c_in {
                for ki in 0..k {
                    let to = ti * stride + ki;
                    if to < off || to >= t_out + off {
                        continue;
                    }
                    for oc in 0..c_out {
                        out[[oc, to - off]] += x[[ic, ti]] * w[[ic, oc, ki]];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn blas_conv1d_same_matches_naive() {
        let mut r = Lcg(0xC0FFEE);
        // dilated K=7 same-conv (the residual-unit shape) + a K=1 conv.
        for (c_in, c_out, k, dil, pad) in [(12, 12, 7, 3, 9usize), (8, 16, 1, 1, 0)] {
            let x = r.arr2(c_in, 50);
            let w = r.arr3(c_out, c_in, k);
            let got = conv1d_same(x.view(), w.view(), None, pad, 1, dil);
            let want = conv1d_same_naive(x.view(), w.view(), pad, 1, dil);
            assert!(
                max_abs(&got, &want) < 1e-4,
                "same conv Δ={}",
                max_abs(&got, &want)
            );
        }
    }

    #[test]
    fn blas_conv1d_stride_matches_naive() {
        let mut r = Lcg(0x1234);
        let x = r.arr2(8, 100);
        let w = r.arr3(16, 8, 4);
        let got = conv1d_stride(x.view(), w.view(), None, 2, 1);
        let want = conv1d_stride_naive(x.view(), w.view(), 2, 1);
        assert!(
            max_abs(&got, &want) < 1e-4,
            "stride conv Δ={}",
            max_abs(&got, &want)
        );
    }

    #[test]
    fn blas_conv_transpose_matches_naive() {
        let mut r = Lcg(0xABCD);
        let x = r.arr2(12, 30);
        let w = r.arr3(12, 6, 4); // [c_in, c_out, k]
        for out_offset in [0usize, 1] {
            let got = conv_transpose1d(x.view(), w.view(), None, 2, 1, out_offset);
            let want = conv_transpose1d_naive(x.view(), w.view(), 2, 1, out_offset);
            assert_eq!(got.dim(), want.dim());
            assert!(
                max_abs(&got, &want) < 1e-4,
                "convT Δ={}",
                max_abs(&got, &want)
            );
        }
    }
}
