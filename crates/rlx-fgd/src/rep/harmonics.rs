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

//! Real spherical harmonics, and the RKHS over `S²` they generate.
//!
//! Section 4.3 needs colour to be view-dependent, and the paper puts that
//! dependence in an RKHS `H^S` over the sphere rather than in plain `L²(S²)` —
//! `L²` functions cannot be evaluated at a direction, and the renderer does
//! exactly that. Truncating at degree `d` gives the reproducing kernel
//!
//! ```text
//! K(ω, ω') = Σ_{i < (d+1)²} Y_i(ω) Y_i(ω')
//! ```
//!
//! whose RKHS is the span of those harmonics, with the `L²(S²)` inner product
//! restricted to it. That is what makes the colour norm of a voxel simply the
//! sum of its squared coefficients: the basis is orthonormal, so the RKHS norm
//! and the `L²(S²)` norm agree on the band-limited functions the
//! representation can hold.
//!
//! Basis and constants follow the PlenOctree convention, which is also what the
//! authors' code and its CUDA renderer use.

const C0: f64 = 0.28209479177387814;
const C1: f64 = 0.4886025119029199;
const C2: [f64; 5] = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
];
const C3: [f64; 7] = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
];
const C4: [f64; 9] = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
];

/// The highest degree implemented.
pub const MAX_DEGREE: u32 = 4;

/// Number of coefficients for a degree-`degree` expansion, `(degree + 1)²`.
pub fn coefficient_count(degree: u32) -> usize {
    let n = degree as usize + 1;
    n * n
}

/// The real spherical harmonics up to `degree`, evaluated at a **unit**
/// direction, written into `out` (which must hold [`coefficient_count`]).
pub fn basis_into(degree: u32, direction: [f64; 3], out: &mut [f64]) {
    assert!(degree <= MAX_DEGREE, "degree {degree} exceeds {MAX_DEGREE}");
    debug_assert_eq!(out.len(), coefficient_count(degree));
    let [x, y, z] = direction;

    out[0] = C0;
    if degree == 0 {
        return;
    }
    out[1] = -C1 * y;
    out[2] = C1 * z;
    out[3] = -C1 * x;
    if degree == 1 {
        return;
    }
    let (xx, yy, zz) = (x * x, y * y, z * z);
    let (xy, yz, xz) = (x * y, y * z, x * z);
    out[4] = C2[0] * xy;
    out[5] = C2[1] * yz;
    out[6] = C2[2] * (2.0 * zz - xx - yy);
    out[7] = C2[3] * xz;
    out[8] = C2[4] * (xx - yy);
    if degree == 2 {
        return;
    }
    out[9] = C3[0] * y * (3.0 * xx - yy);
    out[10] = C3[1] * xy * z;
    out[11] = C3[2] * y * (4.0 * zz - xx - yy);
    out[12] = C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
    out[13] = C3[4] * x * (4.0 * zz - xx - yy);
    out[14] = C3[5] * z * (xx - yy);
    out[15] = C3[6] * x * (xx - 3.0 * yy);
    if degree == 3 {
        return;
    }
    out[16] = C4[0] * xy * (xx - yy);
    out[17] = C4[1] * yz * (3.0 * xx - yy);
    out[18] = C4[2] * xy * (7.0 * zz - 1.0);
    out[19] = C4[3] * yz * (7.0 * zz - 3.0);
    out[20] = C4[4] * (zz * (35.0 * zz - 30.0) + 3.0);
    out[21] = C4[5] * xz * (7.0 * zz - 3.0);
    out[22] = C4[6] * (xx - yy) * (7.0 * zz - 1.0);
    out[23] = C4[7] * xz * (xx - 3.0 * yy);
    out[24] = C4[8] * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy));
}

/// [`basis_into`] into a fresh vector.
pub fn basis(degree: u32, direction: [f64; 3]) -> Vec<f64> {
    let mut out = vec![0.0; coefficient_count(degree)];
    basis_into(degree, direction, &mut out);
    out
}

/// `Σ_i coefficients[i] · Y_i(direction)`.
pub fn evaluate(degree: u32, coefficients: &[f64], direction: [f64; 3]) -> f64 {
    let mut scratch = [0.0f64; 25];
    let n = coefficient_count(degree);
    basis_into(degree, direction, &mut scratch[..n]);
    coefficients[..n]
        .iter()
        .zip(&scratch[..n])
        .map(|(c, b)| c * b)
        .sum()
}

/// The coefficient of `Y_0` that makes a constant-`value` colour, `value / C0`.
pub fn constant_coefficient(value: f64) -> f64 {
    value / C0
}

/// The reproducing kernel `K(ω, ω') = Σ_{i < (d+1)²} Y_i(ω) Y_i(ω')` of the
/// degree-`degree` RKHS over `S²`.
///
/// Directions are normalised first, as the authors' `SphericalHarmonicsRKHS`
/// does — the kernel is only defined on the sphere.
pub fn kernel(degree: u32, a: [f64; 3], b: [f64; 3]) -> f64 {
    let a = normalize(a);
    let b = normalize(b);
    let mut ba = [0.0f64; 25];
    let mut bb = [0.0f64; 25];
    let n = coefficient_count(degree);
    basis_into(degree, a, &mut ba[..n]);
    basis_into(degree, b, &mut bb[..n]);
    ba[..n].iter().zip(&bb[..n]).map(|(p, q)| p * q).sum()
}

/// Scale a vector to unit length; a zero vector is returned unchanged.
pub fn normalize(v: [f64; 3]) -> [f64; 3] {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm > 0.0 {
        [v[0] / norm, v[1] / norm, v[2] / norm]
    } else {
        v
    }
}
