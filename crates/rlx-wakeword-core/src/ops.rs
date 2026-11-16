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
        y[i] = dot(&a[i * n..(i + 1) * n], &x[..n]);
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
    blocked: bool,
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
        let mut oc = 0;
        if blocked {
            // Four output channels per pass over the patch (each patch load serves four rows).
            while oc + 4 <= out_ch {
                let r = dot4(
                    [
                        &w[oc * n..(oc + 1) * n],
                        &w[(oc + 1) * n..(oc + 2) * n],
                        &w[(oc + 2) * n..(oc + 3) * n],
                        &w[(oc + 3) * n..(oc + 4) * n],
                    ],
                    patch,
                );
                for (q, v) in r.iter().enumerate() {
                    out[(oc + q) * t_out + j] = bias[oc + q] + v;
                }
                oc += 4;
            }
        }
        for oc in oc..out_ch {
            out[oc * t_out + j] = bias[oc] + dot(&w[oc * n..(oc + 1) * n], patch);
        }
    }
}

/// Dot product: on aarch64 NEON, four 4-lane fused multiply-add sums over 16 floats a step;
/// elsewhere four scalar sums. A fixed order either way, so deterministic.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        neon::dot(a, b)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
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
}

/// Four dot products against the same `b`: `[rows[q] · b]`. NEON on aarch64 (one load of `b`
/// per four fused multiply-adds), the same fused operations in the same order elsewhere.
#[inline]
pub fn dot4(rows: [&[f32]; 4], b: &[f32]) -> [f32; 4] {
    let n = b.len().min(rows.iter().map(|r| r.len()).min().unwrap_or(0));
    let mut i = 0;
    let mut acc = [[0.0f32; 4]; 4];
    #[cfg(target_arch = "aarch64")]
    // SAFETY: loads of `i..i + 4` with `i + 4 <= n`, inside every row and `b`.
    unsafe {
        use core::arch::aarch64::*;
        let mut a = [vdupq_n_f32(0.0); 4];
        while i + 4 <= n {
            let bv = vld1q_f32(b.as_ptr().add(i));
            for q in 0..4 {
                a[q] = vfmaq_f32(a[q], vld1q_f32(rows[q].as_ptr().add(i)), bv);
            }
            i += 4;
        }
        for q in 0..4 {
            vst1q_f32(acc[q].as_mut_ptr(), a[q]);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    while i + 4 <= n {
        for q in 0..4 {
            for l in 0..4 {
                acc[q][l] = rows[q][i + l].mul_add(b[i + l], acc[q][l]);
            }
        }
        i += 4;
    }
    let mut out = [0.0f32; 4];
    for q in 0..4 {
        let mut r = (acc[q][0] + acc[q][1]) + (acc[q][2] + acc[q][3]);
        for j in i..n {
            r = rows[q][j].mul_add(b[j], r);
        }
        out[q] = r;
    }
    out
}

/// NEON kernels (aarch64: NEON is part of the base architecture, so no runtime check).
#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        // SAFETY: every load reads `i..i + 4` (or `i..i + 16`) with that end at most `n`.
        unsafe {
            let (mut s0, mut s1, mut s2, mut s3) = (
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
            );
            let mut i = 0;
            while i + 16 <= n {
                s0 = vfmaq_f32(s0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
                s1 = vfmaq_f32(s1, vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
                s2 = vfmaq_f32(s2, vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
                s3 = vfmaq_f32(s3, vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
                i += 16;
            }
            while i + 4 <= n {
                s0 = vfmaq_f32(s0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
                i += 4;
            }
            let mut r = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
            while i < n {
                r = a[i].mul_add(b[i], r);
                i += 1;
            }
            r
        }
    }

    /// [`super::conv1d_dilated`]'s outputs `[lo, hi)` from a zero-padded input `xp`
    /// (`[c_in, tp]`, the input at column `pad·d`): 4 output channels × 8 positions in
    /// registers per pass, one fused multiply-add per (channel, tap) and lane. Each output sums
    /// its bias, then channels, then taps, as the scalar edge does: lanes and edge agree bit for
    /// bit, whichever positions a caller asks for.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_dilated(
        xp: &[f32],
        tp: usize,
        c_in: usize,
        w: &[f32],
        c_out: usize,
        k: usize,
        d: usize,
        bias: &[f32],
        out: &mut [f32],
        t: usize,
        lo: usize,
        hi: usize,
    ) {
        let blocks = c_out / 4 * 4;
        let mut ob = 0;
        while ob < blocks {
            let mut p = lo;
            while p + 8 <= hi {
                // SAFETY: rows of `xp` are `tp = t + 2·pad·d` long and position `p + 7`
                // reads at most column `p + 7 + (k − 1)·d < tp`; `out` rows are `t` long.
                unsafe {
                    let mut acc = [[vdupq_n_f32(0.0); 2]; 4];
                    for (c, a) in acc.iter_mut().enumerate() {
                        *a = [vdupq_n_f32(bias[ob + c]); 2];
                    }
                    for i in 0..c_in {
                        let row = xp.as_ptr().add(i * tp + p);
                        for j in 0..k {
                            let x0 = vld1q_f32(row.add(j * d));
                            let x1 = vld1q_f32(row.add(j * d + 4));
                            for (c, a) in acc.iter_mut().enumerate() {
                                let wv = *w.get_unchecked(((ob + c) * c_in + i) * k + j);
                                a[0] = vfmaq_n_f32(a[0], x0, wv);
                                a[1] = vfmaq_n_f32(a[1], x1, wv);
                            }
                        }
                    }
                    for (c, a) in acc.iter().enumerate() {
                        let o = out.as_mut_ptr().add((ob + c) * t + p);
                        vst1q_f32(o, a[0]);
                        vst1q_f32(o.add(4), a[1]);
                    }
                }
                p += 8;
            }
            super::conv_dilated_scalar(xp, tp, c_in, w, ob..ob + 4, k, d, bias, out, t, p, hi);
            ob += 4;
        }
        super::conv_dilated_scalar(xp, tp, c_in, w, blocks..c_out, k, d, bias, out, t, lo, hi);
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

/// Same-length conv1d, dilation `d`: `out[o, t] = b[o] + Σ_i Σ_j w[o, i, j] · x[i, t + (j − k/2)·d]`
/// (zero outside the input), for outputs `[lo, hi)` only (the rest of `out` is left alone: a
/// streaming caller recomputes what changed). `x [c_in, t]`, `w [c_out, c_in, k]`,
/// `out [c_out, t]`. The residual blocks' layer: NEON on aarch64, fused multiply-adds in the
/// same order elsewhere, so a position's value does not depend on which positions were asked.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_dilated(
    x: &[f32],
    c_in: usize,
    t: usize,
    w: &[f32],
    c_out: usize,
    k: usize,
    d: usize,
    bias: &[f32],
    out: &mut [f32],
    lo: usize,
    hi: usize,
    xp: &mut Vec<f32>,
) {
    let hi = hi.min(t);
    if lo >= hi {
        return;
    }
    // The input with `pad·d` zeros each side (in the caller's scratch `xp`, kept between
    // calls): no bounds tests in the inner loops.
    let edge = (k / 2) * d;
    let tp = t + 2 * edge;
    xp.clear();
    xp.resize(c_in * tp, 0.0);
    for i in 0..c_in {
        xp[i * tp + edge..i * tp + edge + t].copy_from_slice(&x[i * t..(i + 1) * t]);
    }
    let xp = &xp[..];
    #[cfg(target_arch = "aarch64")]
    neon::conv_dilated(xp, tp, c_in, w, c_out, k, d, bias, out, t, lo, hi);
    #[cfg(not(target_arch = "aarch64"))]
    conv_dilated_scalar(xp, tp, c_in, w, 0..c_out, k, d, bias, out, t, lo, hi);
}

/// Outputs `[lo, hi)` of channels `chans`, one at a time (the order of the NEON lanes).
#[allow(clippy::too_many_arguments)]
fn conv_dilated_scalar(
    xp: &[f32],
    tp: usize,
    c_in: usize,
    w: &[f32],
    chans: core::ops::Range<usize>,
    k: usize,
    d: usize,
    bias: &[f32],
    out: &mut [f32],
    t: usize,
    lo: usize,
    hi: usize,
) {
    for o in chans {
        for p in lo..hi {
            let mut acc = bias[o];
            for i in 0..c_in {
                for j in 0..k {
                    acc = xp[i * tp + p + j * d].mul_add(w[(o * c_in + i) * k + j], acc);
                }
            }
            out[o * t + p] = acc;
        }
    }
}

#[cfg(test)]
mod kernel_tests {
    use super::*;

    /// The dilated conv (NEON on aarch64) matches its definition, and a position's value is the
    /// same whichever range of positions is computed (streaming recomputes parts).
    #[test]
    fn dilated_conv_matches_its_definition_and_any_range() {
        let val = |i: usize| ((i * 2654435761) % 1000) as f32 / 500.0 - 1.0;
        for (c_in, c_out, t, d) in [
            (64, 64, 50, 1),
            (64, 64, 50, 8),
            (5, 6, 13, 2),
            (3, 9, 7, 4),
        ] {
            let k = 3;
            let x: Vec<f32> = (0..c_in * t).map(val).collect();
            let w: Vec<f32> = (0..c_out * c_in * k).map(|i| val(i + 3) * 0.1).collect();
            let b: Vec<f32> = (0..c_out).map(|i| val(i + 77)).collect();
            let mut full = vec![0.0; c_out * t];
            conv1d_dilated(&x, c_in, t, &w, c_out, k, d, &b, &mut full, 0, t, &mut Vec::new());
            for o in 0..c_out {
                for p in 0..t {
                    let mut want = b[o] as f64;
                    for i in 0..c_in {
                        for j in 0..k {
                            let q = p as isize + (j as isize - 1) * d as isize;
                            if (0..t as isize).contains(&q) {
                                want += (w[(o * c_in + i) * k + j] * x[i * t + q as usize]) as f64;
                            }
                        }
                    }
                    let got = full[o * t + p] as f64;
                    assert!(
                        (got - want).abs() < 1e-4,
                        "c_in {c_in} d {d} o {o} p {p}: {got} vs {want}"
                    );
                }
            }
            for (lo, hi) in [(0, 3), (1, 12), (t / 2, t), (t - 1, t)] {
                let mut part = vec![f32::NAN; c_out * t];
                conv1d_dilated(&x, c_in, t, &w, c_out, k, d, &b, &mut part, lo, hi, &mut Vec::new());
                for o in 0..c_out {
                    assert_eq!(
                        &part[o * t + lo..o * t + hi.min(t)],
                        &full[o * t + lo..o * t + hi.min(t)]
                    );
                }
            }
        }
    }

    #[test]
    fn dot_matches_the_sum() {
        let val = |i: usize| ((i * 2654435761) % 1000) as f32 / 500.0 - 1.0;
        for n in [0usize, 3, 4, 15, 16, 17, 96, 320] {
            let a: Vec<f32> = (0..n).map(val).collect();
            let b: Vec<f32> = (0..n).map(|i| val(i + 5)).collect();
            let want: f64 = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (*x as f64) * (*y as f64))
                .sum();
            assert!((dot(&a, &b) as f64 - want).abs() < 1e-4, "n {n}");
        }
    }
}
