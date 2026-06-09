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

//! Linear rectified-flow interpolant (paper convention: t=0 noise, t=1 data).

const EPS: f32 = 1e-8;

#[inline]
pub fn clip_zero(x: f32) -> f32 {
    x.max(EPS)
}

/// α(t) = t
#[inline]
pub fn alpha(t: f32) -> f32 {
    t
}

/// σ(t) = 1 - t
#[inline]
pub fn sigma(t: f32) -> f32 {
    1.0 - t
}

#[inline]
pub fn alpha_dot(_t: f32) -> f32 {
    1.0
}

#[inline]
pub fn sigma_dot(_t: f32) -> f32 {
    -1.0
}

/// g(t) = σ(t)² / α(t)²
#[inline]
pub fn g(t: f32) -> f32 {
    let a = alpha(t);
    let s = sigma(t);
    (s * s) / clip_zero(a * a)
}

/// Inverse of g for the linear schedule: g(t) = ((1-t)/t)² → t = 1 / (1 + √g)
#[inline]
pub fn g_inv(g_val: f32) -> f32 {
    let g_val = g_val.max(0.0);
    1.0 / (1.0 + g_val.sqrt())
}

/// Denoiser from velocity: D_t(x) = (σ u - σ̇ x) / (α̇ σ - α σ̇)
#[inline]
pub fn denoiser_from_velocity(t: f32, x: f32, velocity: f32) -> f32 {
    let a = alpha(t);
    let s = sigma(t);
    let ad = alpha_dot(t);
    let sd = sigma_dot(t);
    let denom = ad * s - a * sd;
    (s * velocity - sd * x) / clip_zero(denom)
}

/// Velocity from denoiser: u = (α̇ D + σ̇ x) / σ with the paper parameterization.
#[inline]
pub fn velocity_from_denoiser(t: f32, x: f32, denoised: f32) -> f32 {
    let s = sigma(t);
    let ad = alpha_dot(t);
    let sd = sigma_dot(t);
    (ad * denoised + sd * x) / clip_zero(s)
}

/// Guidance coefficient b_t = σ² α̇/α - σ̇ σ (Eq. 10).
#[inline]
pub fn guidance_coefficient(t: f32) -> f32 {
    let a = alpha(t);
    let s = sigma(t);
    s * s * alpha_dot(t) / clip_zero(a) - sigma_dot(t) * s
}

/// Noised sample x_t = α(t) z + σ(t) ε.
#[inline]
pub fn noise_sample(z: f32, t: f32, eps: f32) -> f32 {
    alpha(t) * z + sigma(t) * eps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundaries() {
        assert!((alpha(0.0)).abs() < 1e-6);
        assert!((alpha(1.0) - 1.0).abs() < 1e-6);
        assert!((sigma(0.0) - 1.0).abs() < 1e-6);
        assert!((sigma(1.0)).abs() < 1e-6);
    }

    #[test]
    fn g_inv_roundtrip() {
        for &t in &[0.1f32, 0.3, 0.5, 0.9] {
            let gi = g_inv(g(t));
            assert!((gi - t).abs() < 1e-5, "t={t} gi={gi}");
        }
    }

    #[test]
    fn denoiser_velocity_roundtrip() {
        let t = 0.4;
        let x = 0.7;
        let u = 1.2;
        let d = denoiser_from_velocity(t, x, u);
        let u2 = velocity_from_denoiser(t, x, d);
        assert!((u - u2).abs() < 1e-5);
    }
}
