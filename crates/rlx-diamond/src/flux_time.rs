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

//! Map between paper flow time (0=noise, 1=data) and FLUX rectified-flow sigma (1→0).

use crate::interpolant;

/// FLUX passes `sigma` as timestep where 1.0 is pure noise and 0.0 is data.
#[inline]
pub fn flux_sigma_to_paper_t(sigma: f32) -> f32 {
    (1.0 - sigma).clamp(0.0, 1.0)
}

#[inline]
pub fn paper_t_to_flux_sigma(t: f32) -> f32 {
    (1.0 - t).clamp(0.0, 1.0)
}

/// FLUX x0 prediction: `x0 = x - sigma * v`.
#[inline]
pub fn flux_x0_from_velocity(x: f32, sigma: f32, velocity: f32) -> f32 {
    x - sigma * velocity
}

/// Paper denoiser from FLUX velocity at the matching noise level.
#[inline]
pub fn paper_denoiser_from_flux(x: f32, sigma: f32, velocity: f32) -> f32 {
    flux_x0_from_velocity(x, sigma, velocity)
}

/// Guidance b_t in FLUX coordinates (matches reference FluxDiamondMap).
#[inline]
pub fn flux_guidance_coefficient(sigma: f32, max_abs: f32) -> f32 {
    let t = sigma.clamp(1e-6, 1.0 - 1e-6);
    let alpha = 1.0 - t;
    (t / alpha).clamp(0.0, max_abs)
}

/// Paper b_t from FLUX sigma.
#[inline]
pub fn paper_guidance_coefficient_from_flux_sigma(sigma: f32) -> f32 {
    interpolant::guidance_coefficient(flux_sigma_to_paper_t(sigma))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_time() {
        for s in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            assert!((paper_t_to_flux_sigma(flux_sigma_to_paper_t(s)) - s).abs() < 1e-6);
        }
    }

    #[test]
    fn guidance_matches_flux_formula() {
        let sigma = 0.4f32;
        let b_flux = flux_guidance_coefficient(sigma, 100.0);
        let expected = sigma / (1.0 - sigma);
        assert!((b_flux - expected).abs() < 1e-5);
    }
}
