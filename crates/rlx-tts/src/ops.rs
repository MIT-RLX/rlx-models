use ndarray::{Array2, ArrayView2, Axis};

pub fn view2<'a>(data: &'a [f32], rows: usize, cols: usize) -> ArrayView2<'a, f32> {
    ArrayView2::from_shape((rows, cols), data).expect("view2 shape")
}

/// quantizes activations (and stores f16 weights) while accumulating in f32.
#[inline]
pub fn f16_act(v: f32) -> f32 {
    half::f16::from_f32(v).to_f32()
}

///
/// On macOS uses Accelerate `cblas_sgemm` so attention out-proj (and other IPs)
pub fn linear_in_out(
    x: &Array2<f32>,
    w: &[f32],
    inp: usize,
    out: usize,
    b: Option<&[f32]>,
) -> Array2<f32> {
    let (t, xin) = x.dim();
    debug_assert_eq!(xin, inp);
    let mut y = Array2::<f32>::zeros((t, out));
    let x_slice = x.as_slice().expect("linear x contiguous");
    let y_slice = y.as_slice_mut().expect("linear y contiguous");
    #[cfg(target_os = "macos")]
    {
        // C = α A B + β C  with A[T,in], B[in,out], C[T,out], row-major.
        unsafe {
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                t as i32,
                out as i32,
                inp as i32,
                1.0,
                x_slice.as_ptr(),
                inp as i32,
                w.as_ptr(),
                out as i32,
                0.0,
                y_slice.as_mut_ptr(),
                out as i32,
            );
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let wv = view2(w, inp, out);
        y = x.dot(&wv);
    }
    if let Some(b) = b {
        for mut row in y.rows_mut() {
            for (v, bi) in row.iter_mut().zip(b.iter()) {
                *v += *bi;
            }
        }
    }
    y
}

#[cfg(target_os = "macos")]
#[allow(non_upper_case_globals)]
const CblasRowMajor: u32 = 101;
#[cfg(target_os = "macos")]
#[allow(non_upper_case_globals)]
const CblasNoTrans: u32 = 111;

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: u32,
        trans_a: u32,
        trans_b: u32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// Like [`linear_in_out`], but f16-rounds each activation before the GEMV.
pub fn linear_in_out_lpa(
    x: &Array2<f32>,
    w: &[f32],
    inp: usize,
    out: usize,
    b: Option<&[f32]>,
) -> Array2<f32> {
    let mut x16 = x.clone();
    x16.mapv_inplace(f16_act);
    linear_in_out(&x16, w, inp, out, b)
}

/// LayerNorm over the last axis.
/// When `unbiased` is true (Kaldi `UnbiasedVar T`), variance uses `N-1` and
/// `inv = 1/sqrt(var+eps)`.
///
/// `inv = 1 / (sqrt(var) + eps)` — **not** `1/sqrt(var+eps)`. Mean via
/// `vDSP_meanv`; variance via `vDSP_svesq(x-mean)/C` on macOS.
pub fn layer_norm(
    x: &Array2<f32>,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    unbiased: bool,
) -> Array2<f32> {
    let (t, c) = x.dim();
    let mut y = Array2::<f32>::zeros((t, c));
    let denom = if unbiased && c > 1 {
        (c - 1) as f32
    } else {
        c as f32
    };
    for i in 0..t {
        let row = x.row(i);
        let mean = {
            #[cfg(target_os = "macos")]
            {
                if let Some(slice) = row.as_slice() {
                    let mut m = 0.0f32;
                    unsafe {
                        vDSP_meanv(slice.as_ptr(), 1, &mut m, c);
                    }
                    m
                } else {
                    row.sum() / c as f32
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                row.sum() / c as f32
            }
        };
        let var = {
            #[cfg(target_os = "macos")]
            {
                if let Some(slice) = row.as_slice() {
                    let mut centered = slice.to_vec();
                    for v in &mut centered {
                        *v -= mean;
                    }
                    let mut sumsq = 0.0f32;
                    unsafe {
                        vDSP_svesq(centered.as_ptr(), 1, &mut sumsq, c);
                    }
                    sumsq / denom
                } else {
                    row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / denom
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / denom
            }
        };
        let inv = if unbiased {
            1.0 / (var + eps).sqrt()
        } else {
            1.0 / (var.sqrt() + eps)
        };
        for j in 0..c {
            let scaled = (row[j] - mean) * inv;
            y[[i, j]] = scaled * gamma[j] + beta[j];
        }
    }
    y
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    /// Accelerate.framework `vDSP_meanv`.
    fn vDSP_meanv(a: *const f32, ia: isize, result: *mut f32, n: usize);
    /// Accelerate.framework `vDSP_svesq` (sum of squares).
    fn vDSP_svesq(a: *const f32, ia: isize, result: *mut f32, n: usize);
}

#[inline]
pub fn sigmoid_(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

pub fn relu_inplace(x: &mut Array2<f32>) {
    x.mapv_inplace(|v| v.max(0.0));
}

/// Grouped/standard 1-D convolution. `x: [c_in, t]`, `w: [c_out, c_in, 1, k]` flat.
///
/// `oc→ot→ic→k` MAC bit-matches Hello prenet but is ~3–8e-6 off on FFN0 (1024×256×9).
#[allow(clippy::too_many_arguments)]
pub fn conv1d(
    x: &Array2<f32>,
    w: &[f32],
    c_out: usize,
    c_in: usize,
    k: usize,
    bias: Option<&[f32]>,
    stride: usize,
    pad: usize,
    dilation: usize,
) -> Array2<f32> {
    let (cin, t) = x.dim();
    debug_assert_eq!(cin, c_in);
    let t_out = (t + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
    #[cfg(target_os = "macos")]
    {
        // im2col: [Cin*K, T_out], then Y = W @ col.
        let mut col = Array2::<f32>::zeros((c_in * k, t_out));
        for ot in 0..t_out {
            for ic in 0..c_in {
                for kk in 0..k {
                    let pos = ot * stride + kk * dilation;
                    let v = if pos >= pad && pos - pad < t {
                        x[[ic, pos - pad]]
                    } else {
                        0.0
                    };
                    col[[ic * k + kk, ot]] = v;
                }
            }
        }
        let mut y = Array2::<f32>::zeros((c_out, t_out));
        unsafe {
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                c_out as i32,
                t_out as i32,
                (c_in * k) as i32,
                1.0,
                w.as_ptr(),
                (c_in * k) as i32,
                col.as_slice().unwrap().as_ptr(),
                t_out as i32,
                0.0,
                y.as_slice_mut().unwrap().as_mut_ptr(),
                t_out as i32,
            );
        }
        if let Some(b) = bias {
            for oc in 0..c_out {
                for ot in 0..t_out {
                    y[[oc, ot]] += b[oc];
                }
            }
        }
        y
    }
    #[cfg(not(target_os = "macos"))]
    {
        let mut y = Array2::<f32>::zeros((c_out, t_out));
        let wg = view2(w, c_out, c_in * k);
        for oc in 0..c_out {
            let bz = bias.map(|b| b[oc]).unwrap_or(0.0);
            for ot in 0..t_out {
                let mut acc = 0.0f32;
                for ic in 0..c_in {
                    for kk in 0..k {
                        let pos = ot * stride + kk * dilation;
                        if pos >= pad && pos - pad < t {
                            acc += wg[[oc, ic * k + kk]] * x[[ic, pos - pad]];
                        }
                    }
                }
                y[[oc, ot]] = acc + bz;
            }
        }
        y
    }
}

/// Same-pad 1-D conv preserving length (`pad = dilation * (k-1) / 2` for odd k).
pub fn conv1d_same(
    x: &Array2<f32>,
    w: &[f32],
    c_out: usize,
    c_in: usize,
    k: usize,
    bias: Option<&[f32]>,
    dilation: usize,
) -> Array2<f32> {
    let pad = dilation * (k - 1) / 2;
    conv1d(x, w, c_out, c_in, k, bias, 1, pad, dilation)
}

/// max → subtract → exp → sum → divide.
fn softmax_row(row: &mut [f32]) {
    #[cfg(target_os = "macos")]
    {
        let n = row.len();
        let mut max = f32::NEG_INFINITY;
        unsafe {
            vDSP_maxv(row.as_ptr(), 1, &mut max, n);
        }
        let neg_max = -max;
        unsafe {
            vDSP_vsadd(row.as_ptr(), 1, &neg_max, row.as_mut_ptr(), 1, n);
        }
        rlx_cpu::vmath::vvexpf_inplace(row);
        let mut sum = 0.0f32;
        unsafe {
            vDSP_sve(row.as_ptr(), 1, &mut sum, n);
        }
        let exp = row.to_vec();
        for (v, e) in row.iter_mut().zip(exp.iter()) {
            *v = e / sum;
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for v in row.iter_mut() {
            *v -= max;
        }
        rlx_cpu::vmath::vvexpf_inplace(row);
        let sum: f32 = row.iter().sum();
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn vDSP_maxv(a: *const f32, ia: isize, result: *mut f32, n: usize);
    fn vDSP_vsadd(a: *const f32, ia: isize, b: *const f32, c: *mut f32, ic: isize, n: usize);
    fn vDSP_sve(a: *const f32, ia: isize, result: *mut f32, n: usize);
}

/// Multi-head self-attention: Q/K/V are `[T, heads*dim]`, scale `1/sqrt(dim)`.
///
/// post-multiply of `QKᵀ` is algebraically equal and bit-matches Hello scores.
pub fn multihead_self_attention(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    heads: usize,
    scale: f32,
) -> Array2<f32> {
    let (t, d) = q.dim();
    debug_assert_eq!(d % heads, 0);
    let dim = d / heads;
    let mut out = Array2::<f32>::zeros((t, d));
    for h in 0..heads {
        let qh = q.slice(ndarray::s![.., h * dim..(h + 1) * dim]);
        let kh = k.slice(ndarray::s![.., h * dim..(h + 1) * dim]);
        let vh = v.slice(ndarray::s![.., h * dim..(h + 1) * dim]);
        let mut scores = qh.dot(&kh.t());
        scores.mapv_inplace(|v| v * scale);
        for mut row in scores.rows_mut() {
            let slice = row.as_slice_mut().expect("softmax row contiguous");
            softmax_row(slice);
        }
        let ctx = scores.dot(&vh);
        out.slice_mut(ndarray::s![.., h * dim..(h + 1) * dim])
            .assign(&ctx);
    }
    out
}

/// Length regulation: repeat each phone embedding `dur[i]` times.
pub fn length_regulate(x: &Array2<f32>, durs: &[usize]) -> Array2<f32> {
    let (_, c) = x.dim();
    let total: usize = durs.iter().sum();
    let mut y = Array2::<f32>::zeros((total, c));
    let mut o = 0;
    for (i, &d) in durs.iter().enumerate() {
        let row = x.row(i);
        for _ in 0..d {
            y.row_mut(o).assign(&row);
            o += 1;
        }
    }
    y
}

pub fn embed_lookup(ids: &[usize], table: &[f32], vocab: usize, dim: usize) -> Array2<f32> {
    let w = view2(table, vocab, dim);
    let mut y = Array2::<f32>::zeros((ids.len(), dim));
    for (i, &id) in ids.iter().enumerate() {
        let id = id.min(vocab - 1);
        y.row_mut(i).assign(&w.row(id));
    }
    y
}

/// Transpose `[T, C]` ↔ `[C, T]`.
pub fn to_channels_first(x: &Array2<f32>) -> Array2<f32> {
    x.t().as_standard_layout().to_owned()
}

pub fn to_time_major(x: &Array2<f32>) -> Array2<f32> {
    x.t().as_standard_layout().to_owned()
}

pub fn add_inplace(a: &mut Array2<f32>, b: &Array2<f32>) {
    *a = &*a + b;
}

pub fn stack_rows(rows: &[Array2<f32>]) -> Array2<f32> {
    if rows.is_empty() {
        return Array2::zeros((0, 0));
    }
    let mut out = rows[0].clone();
    for r in rows.iter().skip(1) {
        out = ndarray::concatenate(Axis(0), &[out.view(), r.view()]).expect("concat");
    }
    out
}
