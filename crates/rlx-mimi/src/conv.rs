use ndarray::{Array1, Array2, Array3, ArrayView2};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode {
    Constant,
    Replicate,
}

#[inline]
pub fn elu(x: f32) -> f32 {
    if x >= 0.0 { x } else { x.exp() - 1.0 }
}

pub fn elu_inplace(a: &mut Array2<f32>) {
    for v in a.iter_mut() {
        *v = elu(*v);
    }
}

/// Causal 1D conv matching HF `MimiConv1d`.
pub fn mimi_causal_conv1d(
    x: ArrayView2<f32>,
    weight: &Array3<f32>,
    bias: Option<&Array1<f32>>,
    stride: usize,
    dilation: usize,
    pad_mode: PadMode,
) -> Array2<f32> {
    let (out_ch, in_ch, k) = weight.dim();
    debug_assert_eq!(x.dim().0, in_ch);
    let t_in = x.dim().1;
    let effective_k = (k - 1) * dilation + 1;
    let padding_total = effective_k.saturating_sub(stride);
    let num = t_in as i64 - effective_k as i64 + padding_total as i64;
    let n_frames = if num <= 0 {
        0usize
    } else {
        (num as usize).div_ceil(stride)
    };
    let ideal_len = n_frames * stride + effective_k - padding_total;
    let extra_right = ideal_len.saturating_sub(t_in);
    let pad_left = padding_total;
    let pad_right = extra_right;
    let t_pad = t_in + pad_left + pad_right;

    let mut padded = vec![0f32; in_ch * t_pad];
    for ci in 0..in_ch {
        let row = x.row(ci);
        let base = ci * t_pad;
        match pad_mode {
            PadMode::Constant => {
                for i in 0..t_in {
                    padded[base + pad_left + i] = row[i];
                }
            }
            PadMode::Replicate => {
                let edge_left = row[0];
                let edge_right = row[t_in - 1];
                for i in 0..pad_left {
                    padded[base + i] = edge_left;
                }
                for i in 0..t_in {
                    padded[base + pad_left + i] = row[i];
                }
                for i in 0..pad_right {
                    padded[base + pad_left + t_in + i] = edge_right;
                }
            }
        }
    }
    let t_out = (t_pad - effective_k) / stride + 1;
    let patch = in_ch * k;
    let mut col = vec![0f32; t_out * patch];
    for ti in 0..t_out {
        let row_base = ti * patch;
        for ci in 0..in_ch {
            let pad_base = ci * t_pad + ti * stride;
            let dst_base = row_base + ci * k;
            for ki in 0..k {
                col[dst_base + ki] = padded[pad_base + ki * dilation];
            }
        }
    }
    let mut w_flat = vec![0f32; out_ch * patch];
    for oc in 0..out_ch {
        for ci in 0..in_ch {
            for ki in 0..k {
                w_flat[oc * patch + ci * k + ki] = weight[[oc, ci, ki]];
            }
        }
    }
    use ndarray::ArrayView2 as V;
    let col_a = V::from_shape((t_out, patch), &col).unwrap();
    let w_a = V::from_shape((out_ch, patch), &w_flat).unwrap();
    let out_tc = col_a.dot(&w_a.t());
    let mut out = Array2::<f32>::zeros((out_ch, t_out));
    for oc in 0..out_ch {
        let b = bias.map(|b| b[oc]).unwrap_or(0.0);
        for ti in 0..t_out {
            out[[oc, ti]] = out_tc[[ti, oc]] + b;
        }
    }
    out
}

/// HF `MimiConvTranspose1d` (non-grouped).
pub fn conv_transpose1d(
    x: ArrayView2<f32>,
    weight: &Array3<f32>,
    bias: Option<&Array1<f32>>,
    stride: usize,
    trim_right_ratio: f32,
) -> Array2<f32> {
    let (in_ch, out_ch, k) = weight.dim();
    let (_, t_in) = x.dim();
    let padding_total = k.saturating_sub(stride);
    let padding_right = ((padding_total as f32) * trim_right_ratio).ceil() as usize;
    let padding_left = padding_total - padding_right;
    let t_raw = (t_in - 1) * stride + k;
    let end = t_raw.saturating_sub(padding_right);

    let mut out = Array2::<f32>::zeros((out_ch, end));
    for oc in 0..out_ch {
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
        for ti in padding_left..end {
            out[[oc, ti - padding_left]] = row[ti];
        }
    }
    out
}

/// Depthwise grouped transposed conv (`groups = in_channels = out_channels`).
pub fn grouped_conv_transpose1d(
    x: ArrayView2<f32>,
    weight: &Array3<f32>,
    stride: usize,
    trim_right_ratio: f32,
) -> Array2<f32> {
    let (channels, _, _k) = weight.dim();
    debug_assert_eq!(x.dim().0, channels);
    let mut out = Array2::<f32>::zeros((channels, 0));
    for c in 0..channels {
        let row = x.row(c);
        let x1 = row.insert_axis(ndarray::Axis(0));
        let w1 = weight.slice(ndarray::s![c..c + 1, 0..1, ..]);
        let y = conv_transpose1d(x1, &w1.to_owned(), None, stride, trim_right_ratio);
        if out.dim().1 == 0 {
            out = Array2::<f32>::zeros((channels, y.dim().1));
        }
        for ti in 0..y.dim().1 {
            out[[c, ti]] = y[[0, ti]];
        }
    }
    out
}
