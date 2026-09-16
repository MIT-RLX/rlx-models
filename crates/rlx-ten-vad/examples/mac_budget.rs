// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Where the multiply-accumulates are, and whether any of them can be removed.
//!
//! Three questions, answered with measurements rather than assumption:
//!
//! 1. **Are any weights exactly zero?** Those are free to skip, the way the mel
//!    filterbank's were.
//! 2. **Are the weight matrices low-rank?** Replacing `W [k, n]` with
//!    `U [k, r] · V [r, n]` costs `r(k + n)` instead of `k·n`, so it only wins
//!    below a break-even rank. Trained matrices are usually near full-rank;
//!    this says whether these are.
//! 3. **How much does each stage actually cost?**
//!
//! ```text
//! cargo run -p rlx-ten-vad --release --example mac_budget
//! ```

use rlx_ten_vad_core::HIDDEN;
use rlx_ten_vad_core::weights::embedded_net;

/// Singular values by one-sided Jacobi, descending.
///
/// `a` is row-major `[m, n]` and must have `m >= n`; transpose first otherwise
/// (the spectrum is the same). Columns are rotated until mutually orthogonal,
/// at which point their norms are the singular values.
fn singular_values(a: &[f32], m: usize, n: usize) -> Vec<f64> {
    assert!(m >= n, "one-sided Jacobi needs m >= n");
    let mut w: Vec<f64> = a.iter().map(|&v| f64::from(v)).collect();
    for _ in 0..60 {
        let mut off = 0.0f64;
        for p in 0..n - 1 {
            for q in p + 1..n {
                let (mut alpha, mut beta, mut gamma) = (0.0f64, 0.0f64, 0.0f64);
                for i in 0..m {
                    let (ap, aq) = (w[i * n + p], w[i * n + q]);
                    alpha += ap * ap;
                    beta += aq * aq;
                    gamma += ap * aq;
                }
                if gamma == 0.0 || alpha == 0.0 || beta == 0.0 {
                    continue;
                }
                off += gamma * gamma / (alpha * beta);
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..m {
                    let (ap, aq) = (w[i * n + p], w[i * n + q]);
                    w[i * n + p] = c * ap - s * aq;
                    w[i * n + q] = s * ap + c * aq;
                }
            }
        }
        if off < 1e-24 {
            break;
        }
    }
    let mut sv: Vec<f64> = (0..n)
        .map(|j| (0..m).map(|i| w[i * n + j].powi(2)).sum::<f64>().sqrt())
        .collect();
    sv.sort_by(|x, y| y.partial_cmp(x).unwrap());
    sv
}

/// Smallest rank whose singular values carry `frac` of the squared energy.
fn rank_for(sv: &[f64], frac: f64) -> usize {
    let total: f64 = sv.iter().map(|s| s * s).sum();
    let mut acc = 0.0;
    for (i, s) in sv.iter().enumerate() {
        acc += s * s;
        if acc >= frac * total {
            return i + 1;
        }
    }
    sv.len()
}

/// `[out, in]` → `[in, out]`, so Jacobi gets `m >= n`.
fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; a.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = a[r * cols + c];
        }
    }
    t
}

/// Fraction of the conv stack's output that is exactly zero, measured on real
/// features. Post-ReLU zeros multiply whole columns of the largest matrix in
/// the model, so skipping them is exact, not an approximation.
fn conv_out_sparsity(rows: &[Vec<f32>]) -> (f64, f64) {
    use rlx_ten_vad_core::net::Net;
    use rlx_ten_vad_core::{CONTEXT_FRAMES, FEATURE_LEN};

    let mut net = Net::new(embedded_net());
    let mut stack = vec![0.0f32; CONTEXT_FRAMES * FEATURE_LEN];
    let (mut zero, mut total) = (0usize, 0usize);
    let mut worst = 0usize;
    for row in rows {
        stack.copy_within(FEATURE_LEN.., 0);
        stack[(CONTEXT_FRAMES - 1) * FEATURE_LEN..].copy_from_slice(row);
        net.forward(&stack);
        let f = net.conv_out();
        let z = f.iter().filter(|v| **v == 0.0).count();
        zero += z;
        total += f.len();
        worst = worst.max(z);
    }
    (zero as f64 / total as f64, worst as f64 / 80.0)
}

fn main() {
    let w = embedded_net();
    let gates = 4 * HIDDEN;

    println!("MACs per 16 ms frame\n");
    let stages: [(&str, usize); 10] = [
        ("conv0 depthwise 3x3", 39 * 9),
        ("conv0 pointwise", 16 * 39),
        ("sep1 depthwise", 16 * 10 * 3),
        ("sep1 pointwise", 16 * 16 * 10),
        ("sep2 depthwise", 16 * 5 * 3),
        ("sep2 pointwise", 16 * 16 * 5),
        ("lstm1 (x||h) @ W", gates * (80 + HIDDEN)),
        ("lstm2 (h1||h2) @ W", gates * (2 * HIDDEN)),
        ("dense1", 2 * HIDDEN * 32),
        ("dense2", 32),
    ];
    let total: usize = stages.iter().map(|(_, m)| m).sum();
    for (name, macs) in stages {
        let bar = "#".repeat((macs * 40 / total).max(if macs * 40 / total == 0 { 0 } else { 1 }));
        println!(
            "  {name:22} {macs:6}  {:5.1}%  {bar}",
            100.0 * macs as f64 / total as f64
        );
    }
    println!("  {:22} {total:6}", "total");

    println!("\nexactly-zero weights (free to skip)\n");
    let tensors: [(&str, &[f32]); 6] = [
        ("lstm1.weight_ih", w.lstm1_weight_ih),
        ("lstm1.weight_hh", w.lstm1_weight_hh),
        ("lstm2.weight_ih", w.lstm2_weight_ih),
        ("lstm2.weight_hh", w.lstm2_weight_hh),
        ("dense1.weight", w.dense1_weight),
        ("sep1.pointwise.weight", w.sep1_pointwise),
    ];
    for (name, t) in tensors {
        let z = t.iter().filter(|v| **v == 0.0).count();
        println!(
            "  {name:22} {z:6} of {:6}  ({:.2}%)",
            t.len(),
            100.0 * z as f64 / t.len() as f64
        );
    }

    // Activation sparsity, on the real reference clip.
    let raw = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/reference_features.f32"),
    )
    .expect("reference features");
    let rows: Vec<Vec<f32>> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect::<Vec<_>>()
        .chunks_exact(41)
        .map(<[f32]>::to_vec)
        .collect();
    let (mean_z, best_frame) = conv_out_sparsity(&rows);
    println!("\nactivation sparsity into lstm1 (post-ReLU, exact zeros)\n");
    println!(
        "  conv output zero {:.1}% on average, up to {:.1}% in the best frame",
        100.0 * mean_z,
        100.0 * best_frame
    );
    let saved = (mean_z * (4.0 * HIDDEN as f64) * 80.0) as usize;
    println!(
        "  skipping them removes {saved} of {total} MACs ({:.1}%)",
        100.0 * saved as f64 / total as f64
    );

    println!("\nlow-rank potential — W [k, n] costs k*n; U*V costs r*(k+n)\n");
    // The gate projections as the datapath sees them: [in+H, 4H].
    let mut l1 = vec![0.0f32; (80 + HIDDEN) * gates];
    for r in 0..gates {
        for i in 0..80 {
            l1[i * gates + r] = w.lstm1_weight_ih[r * 80 + i];
        }
        for i in 0..HIDDEN {
            l1[(80 + i) * gates + r] = w.lstm1_weight_hh[r * HIDDEN + i];
        }
    }
    let mut l2 = vec![0.0f32; (2 * HIDDEN) * gates];
    for r in 0..gates {
        for i in 0..HIDDEN {
            l2[i * gates + r] = w.lstm2_weight_ih[r * HIDDEN + i];
            l2[(HIDDEN + i) * gates + r] = w.lstm2_weight_hh[r * HIDDEN + i];
        }
    }

    for (name, mat, k, n) in [
        ("lstm1 [144, 256]", l1.as_slice(), 80 + HIDDEN, gates),
        ("lstm2 [128, 256]", l2.as_slice(), 2 * HIDDEN, gates),
        ("dense1 [128, 32]", w.dense1_weight, 2 * HIDDEN, 32),
    ] {
        // Jacobi wants m >= n: feed whichever orientation is tall.
        let sv = if k >= n {
            singular_values(mat, k, n)
        } else {
            singular_values(&transpose(mat, k, n), n, k)
        };
        let full = k * n;
        let breakeven = full / (k + n);
        println!(
            "  {name:18} full {full:6} MACs   break-even rank {breakeven:3} of {}",
            sv.len()
        );
        for frac in [0.99, 0.999] {
            let r = rank_for(&sv, frac);
            let cost = r * (k + n);
            let verdict = if cost < full {
                format!("{:.2}x cheaper", full as f64 / cost as f64)
            } else {
                format!("{:.2}x MORE expensive", cost as f64 / full as f64)
            };
            println!(
                "      {:.1}% energy needs rank {r:3} -> {cost:6} MACs, {verdict}",
                frac * 100.0
            );
        }
        println!(
            "      sigma_max/sigma_min = {:.1}",
            sv[0] / sv.last().copied().unwrap_or(1.0).max(1e-12)
        );
    }
}
