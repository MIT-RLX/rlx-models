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

//! Stable Audio rectified-flow sampler schedule.
//!
//! Stable Audio Open denoises a latent with a rectified-flow DiT: the sampler
//! walks a `sigma` schedule from `1 → 0`, where each linear point is remapped by a
//! **length-dependent timestep shift** (SD3-style) so longer clips get a different
//! noise curve. This module is the pure schedule math; the resulting sigmas feed
//! [`rlx_audio_blocks::sampling::FlowMatchEuler`] for the Euler denoise step.

use core::f32::consts::PI;

// LogSNR shift anchors (Stable Audio Open defaults).
const LOGSNR_ANCHOR: f32 = -6.2;
const LOGSNR_END: f32 = 2.0;
const LOGSNR_RATE: f32 = 0.0;
const LOGSNR_ANCHOR_LENGTH: f32 = 2000.0;

/// Which timestep-shift curve to apply to the linear sigma schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftKind {
    /// SD3 "full" shift: `mu` interpolated by sequence length, logit-space remap.
    Full,
    /// LogSNR shift (Stable Audio Open default).
    LogSnr,
    /// Identity (linear schedule).
    None,
}

/// Distribution-shift parameters for the sampler schedule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistributionShift {
    pub kind: ShiftKind,
    pub base_shift: f32,
    pub max_shift: f32,
    pub min_length: i64,
    pub max_length: i64,
    pub use_sine: bool,
}

impl Default for DistributionShift {
    fn default() -> Self {
        Self {
            kind: ShiftKind::LogSnr,
            base_shift: 0.5,
            max_shift: 1.15,
            min_length: 256,
            max_length: 4096,
            use_sine: false,
        }
    }
}

impl DistributionShift {
    /// Remap a linear timestep `t ∈ [0, 1]` to the shifted schedule value for a
    /// latent of `seq_len` frames.
    pub fn shift_timestep(&self, t: f32, seq_len: i64) -> f32 {
        match self.kind {
            ShiftKind::Full => self.shifted_full(t, seq_len),
            ShiftKind::LogSnr => shifted_logsnr(t, seq_len),
            ShiftKind::None => t,
        }
    }

    fn shifted_full(&self, t: f32, seq_len: i64) -> f32 {
        if t <= 0.0 {
            return 0.0;
        }
        if t >= 1.0 {
            return 1.0;
        }
        let clamped = seq_len.clamp(self.min_length, self.max_length) as f32;
        let min_length = self.min_length as f32;
        let max_length = self.max_length as f32;
        let ratio = (clamped - min_length) / (max_length - min_length);
        let mu = -(self.base_shift + (self.max_shift - self.base_shift) * ratio);
        let exp_mu = mu.exp();
        let odds = 1.0 / (1.0 - t) - 1.0;
        let mut out = 1.0 - exp_mu / (exp_mu + odds);
        if self.use_sine {
            out = (out * 0.5 * PI).sin();
        }
        out
    }
}

fn shifted_logsnr(t: f32, seq_len: i64) -> f32 {
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }
    let clamped = seq_len.max(1) as f32;
    let log2_ratio = (clamped / LOGSNR_ANCHOR_LENGTH).log2();
    let logsnr_start = LOGSNR_ANCHOR - LOGSNR_RATE * log2_ratio;
    let logsnr = LOGSNR_END - t * (LOGSNR_END - logsnr_start);
    1.0 / (1.0 + logsnr.exp())
}

/// The latent sequence length for a clip of `seconds` at `sample_rate`, given the
/// autoencoder `downsampling_ratio` (ceil division).
pub fn effective_latent_length(seconds: f32, sample_rate: usize, downsampling_ratio: i64) -> i64 {
    let audio_samples = (seconds * sample_rate as f32) as i64;
    let d = downsampling_ratio.max(1);
    (audio_samples + d - 1) / d
}

/// Build the `sigma` schedule of `steps + 1` points, descending from `sigma_max`
/// (default `1.0`) to `0.0`, with the distribution shift applied to each interior
/// point. Suitable for [`rlx_audio_blocks::sampling::FlowMatchEuler::from_sigmas`].
pub fn make_schedule(
    shift: &DistributionShift,
    steps: usize,
    seq_len: i64,
    sigma_max: f32,
) -> Vec<f32> {
    let points = steps + 1;
    let mut schedule = vec![0.0f32; points];
    for (i, s) in schedule.iter_mut().enumerate() {
        let linear = sigma_max * (1.0 - i as f32 / (points - 1).max(1) as f32);
        *s = shift.shift_timestep(linear, seq_len);
    }
    // Pin the endpoints exactly.
    schedule[0] = sigma_max;
    schedule[points - 1] = 0.0;
    schedule
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logsnr_is_monotonic_with_pinned_endpoints() {
        let ds = DistributionShift::default();
        assert_eq!(ds.shift_timestep(0.0, 2000), 0.0);
        assert_eq!(ds.shift_timestep(1.0, 2000), 1.0);
        let mut prev = -1.0;
        for k in 0..=10 {
            let t = k as f32 / 10.0;
            let v = ds.shift_timestep(t, 2000);
            assert!(v >= prev - 1e-6, "non-monotonic at t={t}: {v} < {prev}");
            assert!((0.0..=1.0).contains(&v));
            prev = v;
        }
    }

    #[test]
    fn full_shift_endpoints_and_monotonic() {
        let ds = DistributionShift {
            kind: ShiftKind::Full,
            ..Default::default()
        };
        assert_eq!(ds.shift_timestep(0.0, 1024), 0.0);
        assert_eq!(ds.shift_timestep(1.0, 1024), 1.0);
        let a = ds.shift_timestep(0.3, 1024);
        let b = ds.shift_timestep(0.6, 1024);
        assert!(b > a, "full shift not increasing: {a} !< {b}");
    }

    #[test]
    fn none_shift_is_identity() {
        let ds = DistributionShift {
            kind: ShiftKind::None,
            ..Default::default()
        };
        assert!((ds.shift_timestep(0.42, 1000) - 0.42).abs() < 1e-6);
    }

    #[test]
    fn schedule_descends_from_one_to_zero() {
        let ds = DistributionShift::default();
        let steps = 20;
        let sched = make_schedule(&ds, steps, 1500, 1.0);
        assert_eq!(sched.len(), steps + 1);
        assert_eq!(sched[0], 1.0);
        assert_eq!(sched[steps], 0.0);
        assert!(
            sched.windows(2).all(|w| w[1] <= w[0] + 1e-6),
            "not descending"
        );
    }

    #[test]
    fn effective_length_ceil_divides() {
        // 1 s @ 44100 Hz, downsample 2048 → ceil(44100/2048) = 22.
        assert_eq!(effective_latent_length(1.0, 44_100, 2048), 22);
        assert_eq!(effective_latent_length(0.0, 44_100, 2048), 0);
    }
}
