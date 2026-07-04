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

//! Host CPU reference for the NARMA-10 recurrence and metrics.

/// NARMA memory order (canonical RC benchmark uses 10).
pub const ORDER: usize = 10;

/// Default recurrence coefficients for NARMA-10.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coefficients {
    /// Autoregressive gain on `y[t]` (default `0.3`).
    pub alpha: f64,
    /// Nonlinear history gain multiplied by `y[t]` (default `0.05`).
    pub beta: f64,
    /// Input product gain on `u[t−9]·u[t]` (default `1.5`).
    pub gamma: f64,
    /// Bias (default `0.1`).
    pub delta: f64,
}

impl Default for Coefficients {
    fn default() -> Self {
        Self {
            alpha: 0.3,
            beta: 0.05,
            gamma: 1.5,
            delta: 0.1,
        }
    }
}

/// Input/target pair from [`generate`].
#[derive(Debug, Clone, PartialEq)]
pub struct Series {
    /// Uniform inputs in `[0, 0.5]`, length `n_timesteps + ORDER`.
    pub inputs: Vec<f64>,
    /// Full output trajectory `y[0..inputs.len()]`, with `y[0] = 0`.
    pub outputs: Vec<f64>,
    /// Target outputs, length `n_timesteps` (aligned with `inputs[ORDER..]`).
    pub targets: Vec<f64>,
}

impl Series {
    /// Output `y[t]` (`0` when `t` is out of range).
    #[inline]
    pub fn y_at(&self, t: usize) -> f64 {
        self.outputs.get(t).copied().unwrap_or(0.0)
    }
}

/// Deterministic splitmix64 stream for `Uniform(0, 0.5)` inputs.
#[derive(Debug, Clone, Copy)]
pub struct Rng(u64);

impl Rng {
    /// Seed the generator (`seed = 0` is valid).
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Next value in `[0, 0.5)`.
    pub fn uniform_half_open(&mut self) -> f64 {
        self.uniform01() * 0.5
    }

    /// Next value in `[0, 1)`.
    pub fn uniform01(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        self.0 = z;
        (z >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    }
}

/// One NARMA-10 step: given histories ending at time `t`, return `y[t+1]`.
///
/// `y_hist[t]` is `y[t]`; missing past values are treated as `0`.
/// `u_hist[t]` is `u[t]`; missing past inputs are treated as `0`.
pub fn step(coeff: Coefficients, y_hist: &[f64], u_hist: &[f64], t: usize) -> f64 {
    let yt = y_hist.get(t).copied().unwrap_or(0.0);
    let mut sum = 0.0;
    for i in 0..ORDER {
        if t >= i {
            sum += y_hist[t - i];
        }
    }
    let u_t = u_hist.get(t).copied().unwrap_or(0.0);
    let u_lag = if t >= ORDER - 1 {
        u_hist[t - (ORDER - 1)]
    } else {
        0.0
    };
    coeff.alpha * yt
        + coeff.beta * yt * sum
        + coeff.gamma * u_lag * u_t
        + coeff.delta
}

/// Generate a reproducible NARMA-10 dataset.
///
/// `inputs` has length `n_timesteps + ORDER`; `targets` has length `n_timesteps`.
/// Initial state is `y[0] = 0`.
pub fn generate(n_timesteps: usize, seed: u64) -> Series {
    generate_with_coeff(n_timesteps, seed, Coefficients::default())
}

/// Like [`generate`] with custom coefficients (order fixed at [`ORDER`]).
pub fn generate_with_coeff(n_timesteps: usize, seed: u64, coeff: Coefficients) -> Series {
    assert!(n_timesteps > 0, "n_timesteps must be positive");
    let total = n_timesteps + ORDER;
    let mut rng = Rng::new(seed);
    let inputs: Vec<f64> = (0..total).map(|_| rng.uniform_half_open()).collect();
    let mut y = vec![0.0; total];
    for t in 0..total - 1 {
        y[t + 1] = step(coeff, &y, &inputs, t);
    }
    let targets = y[ORDER..].to_vec();
    Series {
        inputs,
        outputs: y,
        targets,
    }
}

/// Map raw NARMA inputs from `[0, 0.5]` to `[-1, 1]` (optional reservoir preprocessing).
///
/// Predictors in [`crate::models`] inject raw `u[t]` with `input_scaling` on `W_in` instead.
pub fn scale_inputs_for_reservoir(inputs: &[f64]) -> Vec<f64> {
    inputs.iter().map(|&u| 4.0 * u - 1.0).collect()
}

/// Normalized root mean squared error between prediction and target.
///
/// `σ²(y)` is the variance of `target`; returns `0` when variance is zero and
/// predictions match exactly.
pub fn nrmse(predicted: &[f64], target: &[f64]) -> f64 {
    let n = predicted.len().min(target.len());
    assert!(n > 0, "nrmse needs at least one sample");
    let mean = target[..n].iter().sum::<f64>() / n as f64;
    let var = target[..n]
        .iter()
        .map(|&y| {
            let d = y - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    if var == 0.0 {
        return if predicted[..n]
            .iter()
            .zip(&target[..n])
            .all(|(p, t)| (p - t).abs() <= f64::EPSILON)
        {
            0.0
        } else {
            f64::INFINITY
        };
    }
    let mse = predicted[..n]
        .iter()
        .zip(&target[..n])
        .map(|(p, t)| {
            let d = p - t;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    (mse / var).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_matches_hand_computed_values() {
        let coeff = Coefficients::default();
        let u = [0.25];
        let y = [0.0];
        assert!((step(coeff, &y, &u, 0) - 0.1).abs() < 1e-12);

        let u = [0.25, 0.25];
        let y = [0.0, 0.1];
        let next = step(coeff, &y, &u, 1);
        let expected = 0.3 * 0.1 + 0.05 * 0.1 * 0.1 + 0.1;
        assert!((next - expected).abs() < 1e-12);
    }

    #[test]
    fn splitmix_first_ten_match_reference() {
        let mut rng = Rng::new(42);
        let vals: Vec<f64> = (0..10).map(|_| rng.uniform01()).collect();
        eprintln!("rust {:?}", vals);
        let expected = [
            0.7415648787718233,
            0.34329192209867343,
            0.3869742762400409,
            0.7553888514674879,
            0.7959440121759976,
        ];
        for (a, b) in vals.iter().zip(expected) {
            assert!((a - b).abs() < 1e-12, "got {a} expected {b}");
        }
    }

    #[test]
    fn generate_is_deterministic() {
        let a = generate(100, 42);
        let b = generate(100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn series_lengths() {
        let s = generate(50, 1);
        assert_eq!(s.inputs.len(), 50 + ORDER);
        assert_eq!(s.outputs.len(), 50 + ORDER);
        assert_eq!(s.targets.len(), 50);
        assert_eq!(s.outputs[ORDER..], s.targets[..]);
    }

    #[test]
    fn nrmse_perfect_is_zero() {
        let y: Vec<f64> = (0..20).map(|i| i as f64 * 0.01).collect();
        assert!((nrmse(&y, &y) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn scale_inputs_maps_range() {
        assert!((scale_inputs_for_reservoir(&[0.0])[0] + 1.0).abs() < 1e-12);
        assert!((scale_inputs_for_reservoir(&[0.5])[0] - 1.0).abs() < 1e-12);
    }
}
