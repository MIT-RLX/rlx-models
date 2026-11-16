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

/// Conv1d outputs `[o_lo, o_hi)` (of `t_out` per channel), one time position at a time: the
/// position's input patch `[in_ch, k]` (zeros where it reads padding) gathered once, then one
/// dot product per output channel against the weight row `w[oc]` (`[in_ch, k]`, contiguous).
/// For the narrow layers of a wake CNN this beats walking every (channel, channel, tap) for a
/// few positions, and the same function serves a whole window, so a streaming caller that
/// recomputes some positions gets values identical to a full pass. `patch` is scratch.
pub fn conv1d_positions(
    x: &[f32],
    in_ch: usize,
    t_in: usize,
    w: &[f32],
    out_ch: usize,
    k: usize,
    stride: usize,
    pad: usize,
    bias: &[f32],
    out: &mut [f32],
    t_out: usize,
    o_lo: usize,
    o_hi: usize,
    patch: &mut Vec<f32>,
) {
    let n = in_ch * k;
    patch.resize(n, 0.0);
    for j in o_lo..o_hi.min(t_out) {
        for ic in 0..in_ch {
            for ki in 0..k {
                let i = (j * stride + ki) as isize - pad as isize;
                patch[ic * k + ki] = if i >= 0 && (i as usize) < t_in {
                    x[ic * t_in + i as usize]
                } else {
                    0.0
                };
            }
        }
        for oc in 0..out_ch {
            out[oc * t_out + j] = bias[oc] + dot(&w[oc * n..(oc + 1) * n], patch);
        }
    }
}

/// Dot product with four running sums (vectorizes; a fixed order, so deterministic).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = [0.0f32; 4];
    let (ac, ar) = a.split_at(a.len() / 4 * 4);
    let (bc, br) = b.split_at(ac.len());
    for (x, y) in ac.as_chunks::<4>().0.iter().zip(bc.as_chunks::<4>().0) {
        for l in 0..4 {
            s[l] += x[l] * y[l];
        }
    }
    let mut r = (s[0] + s[1]) + (s[2] + s[3]);
    for (x, y) in ar.iter().zip(br) {
        r += x * y;
    }
    r
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
    // Per tap, the outputs whose input index `ot·stride + ki − pad` falls inside `[0, t_in)`:
    // the padding test leaves the inner loop, which then runs over contiguous time (and
    // vectorizes). Each output still sums bias, then channels, then taps, in that order.
    for oc in 0..out_ch {
        let row = &mut out[oc * t_out..(oc + 1) * t_out];
        row.fill(bias.map(|b| b[oc]).unwrap_or(0.0));
        for ic in 0..in_ch {
            let xr = &x[ic * t_in..(ic + 1) * t_in];
            let wr = &w[oc * (in_ch * k) + ic * k..oc * (in_ch * k) + (ic + 1) * k];
            for (ki, &wv) in wr.iter().enumerate() {
                let lo = if pad > ki {
                    (pad - ki).div_ceil(stride)
                } else {
                    0
                };
                let hi = if t_in + pad > ki {
                    ((t_in + pad - ki - 1) / stride + 1).min(t_out)
                } else {
                    0
                };
                if lo >= hi {
                    continue;
                }
                let start = lo * stride + ki - pad;
                for (o, &xv) in row[lo..hi]
                    .iter_mut()
                    .zip(xr[start..].iter().step_by(stride))
                {
                    *o += wv * xv;
                }
            }
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

#[cfg(test)]
mod conv_tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// The direct definition, with the padding test inside.
    fn reference(
        x: &[f32],
        in_ch: usize,
        t_in: usize,
        w: &[f32],
        out_ch: usize,
        k: usize,
        stride: usize,
        pad: usize,
        b: &[f32],
    ) -> Vec<f32> {
        let t_out = (t_in + 2 * pad - k) / stride + 1;
        let mut out = vec![0.0; out_ch * t_out];
        for oc in 0..out_ch {
            for ot in 0..t_out {
                let mut sum = b[oc];
                for ic in 0..in_ch {
                    for ki in 0..k {
                        let ti = (ot * stride + ki) as isize - pad as isize;
                        if ti >= 0 && (ti as usize) < t_in {
                            sum += x[ic * t_in + ti as usize] * w[oc * in_ch * k + ic * k + ki];
                        }
                    }
                }
                out[oc * t_out + ot] = sum;
            }
        }
        out
    }

    #[test]
    fn matches_the_direct_definition_bit_for_bit() {
        let val = |i: usize| ((i * 2654435761) % 1000) as f32 / 500.0 - 1.0;
        for (t_in, k, stride, pad) in [
            (120, 3, 1, 1),
            (120, 3, 2, 1),
            (61, 3, 2, 1),
            (9, 5, 2, 2),
            (7, 3, 3, 0),
            (4, 3, 1, 1),
        ] {
            let (in_ch, out_ch) = (3, 4);
            let x: Vec<f32> = (0..in_ch * t_in).map(val).collect();
            let w: Vec<f32> = (0..out_ch * in_ch * k).map(|i| val(i + 7)).collect();
            let b: Vec<f32> = (0..out_ch).map(|i| val(i + 99)).collect();
            let want = reference(&x, in_ch, t_in, &w, out_ch, k, stride, pad, &b);
            let mut got = vec![0.0; want.len()];
            let t_out = conv1d_nchw(
                &x,
                in_ch,
                t_in,
                &w,
                out_ch,
                k,
                stride,
                pad,
                Some(&b),
                &mut got,
            );
            assert_eq!(t_out * out_ch, want.len());
            assert_eq!(got, want, "t_in={t_in} k={k} stride={stride} pad={pad}");
        }
    }
}
