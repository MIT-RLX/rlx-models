// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Ridge regression: `(XᵀX + λI) w = Xᵀy` for row-major `X` `[m × n]`.

use anyhow::{Result, ensure};

/// Fit ridge coefficients for design matrix `x` (`m` rows, `n` cols) and targets `y` (`m`).
pub fn fit_ridge(x: &[f32], m: usize, n: usize, y: &[f32], lambda: f32) -> Result<Vec<f32>> {
    ensure!(m > 0 && n > 0, "ridge fit needs positive dimensions");
    ensure!(x.len() == m * n && y.len() == m, "ridge fit shape mismatch");

    let lam = lambda.max(1e-8) as f64;
    let mut xtx = vec![0f64; n * n];
    for row in x.chunks_exact(n) {
        for i in 0..n {
            let ri = row[i] as f64;
            for j in 0..=i {
                xtx[i * n + j] += ri * row[j] as f64;
            }
        }
    }
    for i in 0..n {
        for j in 0..i {
            xtx[j * n + i] = xtx[i * n + j];
        }
        xtx[i * n + i] += lam;
    }

    let mut xty = vec![0f64; n];
    for (row, &yt) in x.chunks_exact(n).zip(y) {
        let yt = yt as f64;
        for (xi, b) in row.iter().zip(xty.iter_mut()) {
            *b += *xi as f64 * yt;
        }
    }

    let w = solve_spd(&xtx, n, &xty)?;
    Ok(w.into_iter().map(|v| v as f32).collect())
}

/// Apply fitted weights to rows of `x` (`m × n`).
pub fn predict(x: &[f32], m: usize, n: usize, w: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; m];
    for (row, o) in x.chunks_exact(n).zip(out.iter_mut()) {
        *o = row.iter().zip(w).map(|(a, b)| a * b).sum();
    }
    out
}

fn solve_spd(a: &[f64], n: usize, b: &[f64]) -> Result<Vec<f64>> {
    let mut l = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                ensure!(s > 0.0, "ridge normal matrix is not SPD (λ too small?)");
                l[i * n + j] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }

    let mut y = b.to_vec();
    for i in 0..n {
        let mut s = y[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_linear_relation() {
        let x = vec![1.0f32, 1.0, 2.0, 1.0, 3.0, 1.0];
        let y = vec![3.0f32, 5.0];
        let w = fit_ridge(&x, 2, 3, &y, 1e-6).unwrap();
        let pred = predict(&x, 2, 3, &w);
        assert!((pred[0] - 3.0).abs() < 1e-4);
        assert!((pred[1] - 5.0).abs() < 1e-4);
    }
}
