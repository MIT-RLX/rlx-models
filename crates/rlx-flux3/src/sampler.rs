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

//! Schedule and initial noise, matching BFL `flux3/sampling.py` and
//! `flux3/pipeline.py` value for value.

/// `torch.linspace(start, end, n)` in f32: the CPU kernel's two-sided form.
fn linspace_f32(start: f32, end: f32, n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f32;
    let half = n / 2;
    (0..n)
        .map(|i| {
            if i < half {
                start + step * i as f32
            } else {
                end - step * (n - i - 1) as f32
            }
        })
        .collect()
}

/// Flow times `t_0 = 1 > … > t_steps = 0`: `α·t / (1 + (α − 1)·t)` over a
/// uniform grid, in f32.
pub fn timesteps(steps: usize, alpha: f32) -> Vec<f32> {
    let am1 = (alpha as f64 - 1.0) as f32;
    linspace_f32(1.0, 0.0, steps.max(1) + 1)
        .into_iter()
        .map(|t| alpha * t / (1.0 + am1 * t))
        .collect()
}

/// `torch.randn((n,), generator=torch.Generator().manual_seed(seed))`,
/// rounded to bf16 as the reference does before sampling.
pub fn initial_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = rlx_rng::torch::RandomState::new(seed as u32);
    rng.normal_tensor_f32(n, 0.0, 1.0)
        .into_iter()
        .map(crate::embed::round_bf16)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_matches_torch() {
        // torch: alpha 6.93, 50 steps → s[1..4] (hex floats of the f32 values).
        let s = timesteps(50, 6.93);
        assert_eq!(s.len(), 51);
        assert_eq!(s[0], 1.0);
        assert_eq!(s[1] as f64, f64::from_bits(0x3FEFE7F240000000));
        assert_eq!(s[2] as f64, f64::from_bits(0x3FEFCF0A40000000));
        assert_eq!(s[3] as f64, f64::from_bits(0x3FEFB53C20000000));
        assert_eq!(*s.last().unwrap(), 0.0);
        let s7 = timesteps(7, 6.93);
        assert_eq!(s7[3] as f64, f64::from_bits(0x3FECE00020000000));
    }
}
