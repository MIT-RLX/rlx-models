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

//! Reproducible Gaussian noise for flow-matching (seeded PCG).

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::StandardNormal;

/// Fill `out` with i.i.d. standard normal samples from `seed`.
pub fn fill_standard_normal(out: &mut [f32], seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    for v in out.iter_mut() {
        *v = rng.sample::<f32, _>(StandardNormal);
    }
}

/// Per-frame seed mixing (prompt seed + frame index).
pub fn frame_seed(base: u64, frame_index: usize) -> u64 {
    base.wrapping_add((frame_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_noise_is_stable() {
        let mut a = [0f32; 8];
        let mut b = [0f32; 8];
        fill_standard_normal(&mut a, 42);
        fill_standard_normal(&mut b, 42);
        assert_eq!(a, b);
    }
}
