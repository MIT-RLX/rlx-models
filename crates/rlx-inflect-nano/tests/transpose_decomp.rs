//! Localize the transposed-conv rewrite: host conv_transpose1d vs
//! (zero-insert + regular conv1d with reversed/transposed weight).

use ndarray::Array2;
use rlx_inflect_nano::ops::{conv_transpose1d, conv1d};

#[test]
fn zero_insert_conv_equals_transpose() {
    let (c_in, c_out, k, stride, t) = (2usize, 3usize, 4usize, 2usize, 5usize);
    let pad = (k - stride) / 2;
    // deterministic pseudo-random data
    let mut x = Array2::<f32>::zeros((c_in, t));
    for c in 0..c_in {
        for j in 0..t {
            x[[c, j]] = ((c * 7 + j * 3) % 11) as f32 * 0.1 - 0.5;
        }
    }
    let mut w = vec![0f32; c_in * c_out * k]; // [c_in, c_out, k]
    for (i, v) in w.iter_mut().enumerate() {
        *v = ((i * 5) % 13) as f32 * 0.1 - 0.6;
    }
    let bias = vec![0.0f32; c_out];

    let reference = conv_transpose1d(&x, &w, c_in, c_out, k, Some(&bias), stride, pad);

    // decomposition: reversed/transposed weight, zero-insert, conv1d, pad'=k-1-pad
    let mut wrev = vec![0f32; c_out * c_in * k]; // [c_out, c_in, k]
    for ic in 0..c_in {
        for oc in 0..c_out {
            for kk in 0..k {
                wrev[oc * c_in * k + ic * k + kk] = w[ic * c_out * k + oc * k + (k - 1 - kk)];
            }
        }
    }
    let l_up = (t - 1) * stride + 1;
    let mut xu = Array2::<f32>::zeros((c_in, l_up));
    for c in 0..c_in {
        for i in 0..t {
            xu[[c, i * stride]] = x[[c, i]];
        }
    }
    let got = conv1d(
        &xu,
        &wrev,
        c_out,
        c_in,
        k,
        Some(&bias),
        1,
        k - 1 - pad,
        1,
        1,
    );

    assert_eq!(
        reference.dim(),
        got.dim(),
        "shape {:?} vs {:?}",
        reference.dim(),
        got.dim()
    );
    let d = reference
        .iter()
        .zip(got.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("transpose-decomp maxdiff = {d:.3e}");
    assert!(d < 1e-5, "decomposition mismatch: {d:.3e}");
}
