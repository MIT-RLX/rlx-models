//! The readout head: `z_j = (K(h_opt_j) · Q(h_decide)) / sqrt(dp)`.

use rlx_kev::pointer::{PointerHead, softmax};

/// Identity projections, `dp = 2`, so the logits are the raw dot products
/// scaled by `1/sqrt(2)` and can be checked by hand.
fn identity_head(temperature: f32) -> PointerHead {
    PointerHead::new(
        vec![1.0, 0.0, 0.0, 1.0],
        vec![0.0, 0.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![0.0, 0.0],
        2,
        2,
        temperature,
    )
    .expect("well-formed head")
}

#[test]
fn logits_are_scaled_dot_products() {
    let head = identity_head(1.0);
    let z = head
        .logits(&[1.0, 0.0], &[1.0, 0.0, 0.0, 1.0])
        .expect("logits");
    let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
    assert!((z[0] - inv_sqrt2).abs() < 1e-6, "z[0] = {}", z[0]);
    assert!(z[1].abs() < 1e-6, "z[1] = {}", z[1]);

    let p = softmax(&z);
    assert!((p[0] - 0.669_8).abs() < 1e-3, "p = {p:?}");
    assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
}

#[test]
fn bias_terms_are_applied() {
    let head = PointerHead::new(
        vec![0.0, 0.0, 0.0, 0.0],
        vec![1.0, 0.0], // q(h) = [1, 0] regardless of input
        vec![0.0, 0.0, 0.0, 0.0],
        vec![3.0, 0.0], // k(h) = [3, 0] regardless of input
        2,
        2,
        1.0,
    )
    .expect("well-formed head");
    let z = head.logits(&[9.0, 9.0], &[0.0, 0.0]).expect("logits");
    // (3*1 + 0*0) / sqrt(2)
    assert!((z[0] - 3.0 / 2.0_f32.sqrt()).abs() < 1e-6, "z = {z:?}");
}

/// Temperature rescales the distribution but can never reorder it — that is
/// what makes a fitted value safe to apply at inference only.
#[test]
fn temperature_sharpens_without_changing_the_argmax() {
    let h_opts = [1.0, 0.0, 0.3, 0.6, 0.0, 1.0];
    let raw = identity_head(1.0)
        .probs(&[1.0, 0.2], &h_opts)
        .expect("probs");
    let warm = identity_head(2.2)
        .probs(&[1.0, 0.2], &h_opts)
        .expect("probs");

    let arg = |p: &[f32]| {
        p.iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, v)| {
                if *v > bv { (i, *v) } else { (bi, bv) }
            })
            .0
    };
    assert_eq!(arg(&raw), arg(&warm), "temperature reordered the options");

    let peak_raw = raw.iter().copied().fold(f32::MIN, f32::max);
    let peak_warm = warm.iter().copied().fold(f32::MIN, f32::max);
    assert!(
        peak_warm < peak_raw,
        "T > 1 must flatten the distribution: {peak_warm} vs {peak_raw}"
    );
    assert!((warm.iter().sum::<f32>() - 1.0).abs() < 1e-6);
}

/// A transposed or mis-sized checkpoint must fail at load, not produce
/// plausible numbers.
#[test]
fn malformed_heads_are_rejected() {
    // [dp, d] = [2, 3] declared as head_dim=2, hidden=2.
    assert!(
        PointerHead::new(
            vec![0.0; 6],
            vec![0.0; 2],
            vec![0.0; 6],
            vec![0.0; 2],
            2,
            2,
            1.0
        )
        .is_err(),
        "weight length mismatch must be rejected"
    );
    // Bias of the wrong width.
    assert!(
        PointerHead::new(
            vec![0.0; 4],
            vec![0.0; 3],
            vec![0.0; 4],
            vec![0.0; 2],
            2,
            2,
            1.0
        )
        .is_err(),
        "bias length mismatch must be rejected"
    );
    // Non-positive temperature would flip or annihilate the distribution.
    assert!(
        PointerHead::new(
            vec![0.0; 4],
            vec![0.0; 2],
            vec![0.0; 4],
            vec![0.0; 2],
            2,
            2,
            0.0
        )
        .is_err()
    );
    assert!(
        PointerHead::new(
            vec![0.0; 4],
            vec![0.0; 2],
            vec![0.0; 4],
            vec![0.0; 2],
            2,
            2,
            -1.0
        )
        .is_err()
    );
}

#[test]
fn ragged_option_blocks_are_rejected() {
    let head = identity_head(1.0);
    assert!(
        head.logits(&[1.0], &[1.0, 0.0]).is_err(),
        "a decide vector of the wrong width must be rejected"
    );
    assert!(
        head.logits(&[1.0, 0.0], &[1.0, 0.0, 0.5]).is_err(),
        "an option block that is not a multiple of hidden must be rejected"
    );
}

#[test]
fn softmax_is_stable_at_large_magnitudes() {
    let p = softmax(&[1000.0, 999.0, -1000.0]);
    assert!(p.iter().all(|v| v.is_finite()), "p = {p:?}");
    assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert!(p[0] > p[1] && p[1] > p[2]);
}
