// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! Sanity-check the Accelerate sgemm wrapper against a hand-rolled reference.

use ndarray::{Array1, Array2};
use rlx_pocket_tts::ops::linear;

fn reference_linear(x: &Array2<f32>, w: &Array2<f32>, b: Option<&Array1<f32>>) -> Array2<f32> {
    let (n, k) = x.dim();
    let (m, k2) = w.dim();
    assert_eq!(k, k2);
    let mut out = Array2::<f32>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let mut s = 0.0_f32;
            for p in 0..k {
                s += x[[i, p]] * w[[j, p]];
            }
            out[[i, j]] = s;
            if let Some(b) = b {
                out[[i, j]] += b[j];
            }
        }
    }
    out
}

#[test]
fn linear_parity_small() {
    // N=3, K=5, M=4 — easy to inspect by hand.
    let x = Array2::from_shape_vec((3, 5), (0..15).map(|v| v as f32 * 0.1).collect()).unwrap();
    let w =
        Array2::from_shape_vec((4, 5), (0..20).map(|v| (v as f32 * 0.05).sin()).collect()).unwrap();
    let b = Array1::from(vec![0.1_f32, -0.2, 0.3, -0.4]);

    let got = linear(x.view(), w.view(), Some(b.view()));
    let want = reference_linear(&x, &w, Some(&b));
    assert_eq!(got.dim(), want.dim());
    for i in 0..3 {
        for j in 0..4 {
            let d = (got[[i, j]] - want[[i, j]]).abs();
            assert!(
                d < 1e-5,
                "[{i},{j}]: got {} want {} (Δ={d:.2e})",
                got[[i, j]],
                want[[i, j]]
            );
        }
    }
}

#[test]
fn linear_parity_lm_shape() {
    // Match the FlowLM in_proj shape: K=1024, M=3072, batch=1.
    let n = 1;
    let k = 256; // shrink for test speed; ratio of K to M preserved
    let m = 768;
    let x = Array2::from_shape_vec(
        (n, k),
        (0..n * k).map(|v| ((v as f32) * 0.001).sin()).collect(),
    )
    .unwrap();
    let w = Array2::from_shape_vec(
        (m, k),
        (0..m * k).map(|v| ((v as f32) * 0.0007).cos()).collect(),
    )
    .unwrap();

    let got = linear(x.view(), w.view(), None);
    let want = reference_linear(&x, &w, None);
    let max_err = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_err < 1e-3, "max err {max_err:.2e}");
}
