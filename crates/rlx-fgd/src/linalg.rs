// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Small dense symmetric solvers.
//!
//! Both users are tiny and symmetric: Proposition A.2's `λ_min(K)` over an
//! `n × n` kernel matrix, and the per-voxel spherical-harmonic least-squares
//! fit of Section 4.3, which is `(deg+1)² ≤ 25` wide. Cyclic Jacobi covers
//! both — it is accurate on small symmetric matrices, needs no pivoting, and
//! gives the eigenvectors a pseudo-inverse needs when the fit is rank
//! deficient (fewer sampled directions than harmonics).

/// Eigenvalues and eigenvectors of a symmetric `n × n` matrix, by cyclic
/// Jacobi rotations.
///
/// `matrix` is row-major and is not modified. Returns `(values, vectors)`
/// with `vectors` row-major and column `k` the eigenvector for `values[k]`.
pub fn symmetric_eigen(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = matrix.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..100 {
        let mut off = 0.0;
        for p in 0..n {
            for q in p + 1..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off <= 1e-30 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-18 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), v)
}

/// The smallest eigenvalue of a symmetric `n × n` matrix.
pub fn smallest_eigenvalue(matrix: &[f64], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    symmetric_eigen(matrix, n)
        .0
        .into_iter()
        .fold(f64::INFINITY, f64::min)
}

/// Solve `Ax = b` for symmetric positive semidefinite `A` in the
/// least-squares, minimum-norm sense: `x = A⁺b`.
///
/// Directions that `A` does not constrain are left at zero rather than being
/// driven by whatever noise landed in `b` — which is what makes the
/// spherical-harmonic fit safe when a voxel is seen from fewer directions than
/// it has harmonics. `tolerance` is relative to the largest eigenvalue.
pub fn pseudo_inverse_solve(matrix: &[f64], rhs: &[f64], n: usize, tolerance: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let (values, vectors) = symmetric_eigen(matrix, n);
    let largest = values.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let cutoff = tolerance * largest;
    let mut out = vec![0.0; n];
    for k in 0..n {
        if values[k].abs() <= cutoff {
            continue;
        }
        // Project b onto eigenvector k, scale by 1/λ, project back.
        let projection: f64 = (0..n).map(|i| vectors[i * n + k] * rhs[i]).sum();
        let scaled = projection / values[k];
        for i in 0..n {
            out[i] += scaled * vectors[i * n + k];
        }
    }
    out
}
