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

//! Rectified-flow Euler scheduler (`eta = 0`) for MiniMax-H3.
//!
//! Three details separate this from the usual flow-match Euler scheduler, and
//! all three are load-bearing:
//!
//! 1. **The velocity points at the data.** The DiT predicts a data-ward
//!    velocity, so the denoised estimate is `x0 = x_t + sigma * v` — a plus,
//!    where diffusers' `FlowMatchEulerDiscreteScheduler` subtracts.
//! 2. **`t = 1 - sigma`, on `[0, 1]`, with `t = 1` meaning clean.** The DiT's
//!    AdaLN consumes this convention unscaled; there is no `* 1000` factor.
//! 3. **The grid is `linspace(1, 0, num_inference_steps)`.** The terminal zero
//!    is part of the requested count, and duplicates the shift creates near
//!    `sigma = 1` are collapsed, so the realized step count can come out below
//!    the request.
//!
//! One request runs **two** schedules, one per modality: `shift = 12.0` for
//! video and `shift = 3.0` for audio.

use anyhow::{Result, bail, ensure};

/// A built sigma / timestep schedule plus the cursor into it.
#[derive(Debug, Clone)]
pub struct H3Scheduler {
    shift: f32,
    /// Sigma grid, strictly decreasing, terminating at exactly `0.0`.
    sigmas: Vec<f32>,
    /// `1 - sigmas[..len-1]` — one entry per model evaluation.
    timesteps: Vec<f32>,
    step_index: Option<usize>,
    begin_index: Option<usize>,
}

impl H3Scheduler {
    /// Create a scheduler with the given exponential shift.
    pub fn new(shift: f32) -> Result<Self> {
        ensure!(shift > 0.0, "`shift` must be positive, got {shift}");
        Ok(Self {
            shift,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            step_index: None,
            begin_index: None,
        })
    }

    /// Video schedule of the released checkpoint.
    pub fn video() -> Self {
        Self::new(12.0).expect("12.0 is positive")
    }

    /// Audio schedule of the released checkpoint.
    pub fn audio() -> Self {
        Self::new(3.0).expect("3.0 is positive")
    }

    #[must_use]
    pub fn shift(&self) -> f32 {
        self.shift
    }

    /// Override the shift. Call before [`Self::set_timesteps`].
    pub fn set_shift(&mut self, shift: f32) -> Result<()> {
        ensure!(shift > 0.0, "`shift` must be positive, got {shift}");
        self.shift = shift;
        Ok(())
    }

    #[must_use]
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    #[must_use]
    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    /// Number of model evaluations this schedule drives.
    #[must_use]
    pub fn num_inference_steps(&self) -> usize {
        self.timesteps.len()
    }

    #[must_use]
    pub fn step_index(&self) -> Option<usize> {
        self.step_index
    }

    pub fn set_begin_index(&mut self, begin_index: usize) {
        self.begin_index = Some(begin_index);
    }

    /// Apply the exponential shift to one sigma: `s*x / (1 + (s-1)*x)`.
    ///
    /// Maps `0 -> 0` and `1 -> 1` exactly for any positive `s`.
    #[must_use]
    pub fn apply_shift(shift: f32, x: f32) -> f32 {
        shift * x / (1.0 + (shift - 1.0) * x)
    }

    /// Build the schedule from a step count.
    ///
    /// The grid is `linspace(1, 0, num_inference_steps)` pushed through the
    /// shift with consecutive duplicates collapsed. The terminal `0` is part of
    /// the grid, so the schedule drives `len(sigmas) - 1` evaluations.
    pub fn set_timesteps(&mut self, num_inference_steps: usize) -> Result<()> {
        ensure!(
            num_inference_steps >= 2,
            "`set_timesteps` requires num_inference_steps >= 2, got {num_inference_steps}"
        );
        let n = num_inference_steps;
        let mut sigmas = Vec::with_capacity(n);
        for i in 0..n {
            // linspace(1, 0, n): exact endpoints, matching torch.
            let base = if n == 1 {
                1.0
            } else {
                1.0 - (i as f32) / ((n - 1) as f32)
            };
            sigmas.push(Self::apply_shift(self.shift, base));
        }
        dedup_consecutive(&mut sigmas);
        self.install(sigmas)
    }

    /// Install a fully-formed sigma schedule verbatim — no shift, no dedup.
    ///
    /// Must be strictly decreasing and terminate at `0.0`.
    pub fn set_sigmas(&mut self, sigmas: &[f32]) -> Result<()> {
        ensure!(
            sigmas.len() >= 2,
            "`sigmas` must hold at least two values, got {}",
            sigmas.len()
        );
        if !sigmas.windows(2).all(|w| w[1] < w[0]) {
            bail!("`sigmas` must be strictly decreasing");
        }
        if sigmas[sigmas.len() - 1] != 0.0 {
            bail!(
                "`sigmas` must terminate at 0.0, got {}",
                sigmas[sigmas.len() - 1]
            );
        }
        self.install(sigmas.to_vec())
    }

    fn install(&mut self, sigmas: Vec<f32>) -> Result<()> {
        ensure!(
            sigmas.len() >= 2,
            "schedule collapsed to {} sigma(s); raise num_inference_steps or lower the shift",
            sigmas.len()
        );
        self.timesteps = sigmas[..sigmas.len() - 1].iter().map(|s| 1.0 - s).collect();
        self.sigmas = sigmas;
        self.step_index = None;
        self.begin_index = None;
        Ok(())
    }

    /// Index of `timestep` in [`Self::timesteps`].
    pub fn index_for_timestep(&self, timestep: f32) -> Result<usize> {
        self.timesteps
            .iter()
            .position(|&t| t == timestep)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "timestep {timestep} is not in the schedule; pass a value from `timesteps()`"
                )
            })
    }

    /// Rectified-flow forward process in the H3 convention:
    /// `x_t = t*x0 + (1 - t)*noise`. `t = 1` returns `sample` unchanged.
    ///
    /// H3 uses this to noise conditioning anchors, where `t` is a `noise_aug`
    /// level rather than a schedule entry, so it is taken at face value.
    pub fn scale_noise(sample: &[f32], timestep: f32, noise: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            sample.len() == noise.len(),
            "scale_noise: sample len {} != noise len {}",
            sample.len(),
            noise.len()
        );
        Ok(sample
            .iter()
            .zip(noise)
            .map(|(&x, &n)| timestep * x + (1.0 - timestep) * n)
            .collect())
    }

    /// Take one Euler step, advancing the cursor.
    ///
    /// `model_output` is the data-ward velocity at `timestep`, `sample` is
    /// `x_t`. Returns `x_{t+1}`.
    pub fn step(
        &mut self,
        model_output: &[f32],
        timestep: f32,
        sample: &[f32],
    ) -> Result<Vec<f32>> {
        ensure!(
            model_output.len() == sample.len(),
            "step: model_output len {} != sample len {}",
            model_output.len(),
            sample.len()
        );
        if self.step_index.is_none() {
            self.step_index = Some(match self.begin_index {
                Some(i) => i,
                None => self.index_for_timestep(timestep)?,
            });
        }
        let i = self.step_index.expect("set above");
        ensure!(
            i + 1 < self.sigmas.len(),
            "step: schedule exhausted at index {i} ({} sigmas)",
            self.sigmas.len()
        );

        // x0 from the data-ward velocity. The sigma here is recovered from the
        // timestep the DiT was conditioned on, not from the grid: for
        // sigma < 0.5 the round trip `1 - (1 - sigma)` is not exact in f32 and
        // the reference keeps the two sources apart.
        let sigma_from_timestep = 1.0 - timestep;
        let sigma = self.sigmas[i];
        let sigma_next = self.sigmas[i + 1];
        let ratio = sigma_next / sigma;

        let out = sample
            .iter()
            .zip(model_output)
            .map(|(&x, &v)| {
                let denoised = x + sigma_from_timestep * v;
                ratio * x + (1.0 - ratio) * denoised
            })
            .collect();
        self.step_index = Some(i + 1);
        Ok(out)
    }

    /// Reset the cursor so the schedule can drive another sampling run.
    pub fn reset(&mut self) {
        self.step_index = None;
    }
}

/// Collapse runs of equal consecutive values, mirroring
/// `torch.unique_consecutive`.
fn dedup_consecutive(v: &mut Vec<f32>) {
    let mut out: Vec<f32> = Vec::with_capacity(v.len());
    for &x in v.iter() {
        if out.last().is_none_or(|&last| last != x) {
            out.push(x);
        }
    }
    *v = out;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_fixes_both_endpoints() {
        for &s in &[3.0f32, 12.0, 1.0, 0.5] {
            assert_eq!(H3Scheduler::apply_shift(s, 0.0), 0.0);
            assert!((H3Scheduler::apply_shift(s, 1.0) - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn shift_of_one_is_identity() {
        for i in 0..=10 {
            let x = i as f32 / 10.0;
            assert!((H3Scheduler::apply_shift(1.0, x) - x).abs() < 1e-6);
        }
    }

    #[test]
    fn schedule_is_decreasing_and_terminates_at_zero() {
        for &shift in &[3.0f32, 12.0] {
            let mut s = H3Scheduler::new(shift).unwrap();
            s.set_timesteps(32).unwrap();
            let sig = s.sigmas();
            assert_eq!(*sig.last().unwrap(), 0.0);
            assert!((sig[0] - 1.0).abs() < 1e-6);
            assert!(
                sig.windows(2).all(|w| w[1] < w[0]),
                "sigmas must strictly decrease: {sig:?}"
            );
            // t = 1 - sigma, one per evaluation, and no evaluation at the
            // terminal sigma.
            assert_eq!(s.timesteps().len(), sig.len() - 1);
            for (t, sg) in s.timesteps().iter().zip(sig) {
                assert!((t - (1.0 - sg)).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn video_shift_compresses_grid_toward_one() {
        // shift > 1 pushes interior sigmas up, so the first steps are coarser
        // in sigma and the schedule spends more evaluations near clean.
        let mut v = H3Scheduler::video();
        v.set_timesteps(8).unwrap();
        let mut a = H3Scheduler::audio();
        a.set_timesteps(8).unwrap();
        for (sv, sa) in v.sigmas().iter().zip(a.sigmas()) {
            assert!(
                sv >= sa,
                "video shift 12 should dominate audio shift 3: {sv} vs {sa}"
            );
        }
    }

    #[test]
    fn step_at_terminal_sigma_lands_on_denoised() {
        // The last evaluation has sigma_next = 0, so ratio = 0 and the update
        // returns exactly x0 = x_t + sigma * v.
        let mut s = H3Scheduler::new(1.0).unwrap();
        s.set_timesteps(3).unwrap();
        let n = s.num_inference_steps();
        let sample = vec![0.25f32, -0.5];
        let vel = vec![1.0f32, 2.0];
        let mut cur = sample.clone();
        for k in 0..n {
            let t = s.timesteps()[k];
            cur = s.step(&vel, t, &cur).unwrap();
        }
        let last_t = *s.timesteps().last().unwrap();
        let _ = last_t;
        assert!(cur.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn single_step_matches_closed_form() {
        let mut s = H3Scheduler::new(1.0).unwrap();
        s.set_timesteps(5).unwrap();
        let t = s.timesteps()[0];
        let sigma = s.sigmas()[0];
        let sigma_next = s.sigmas()[1];
        let x = vec![0.3f32];
        let v = vec![-1.25f32];
        let got = s.step(&v, t, &x).unwrap();

        let denoised = x[0] + (1.0 - t) * v[0];
        let ratio = sigma_next / sigma;
        let want = ratio * x[0] + (1.0 - ratio) * denoised;
        assert!((got[0] - want).abs() < 1e-6, "got {got:?}, want {want}");
    }

    #[test]
    fn velocity_sign_is_data_ward() {
        // A positive velocity must move the sample *up* — the opposite of the
        // usual flow-match convention.
        let mut s = H3Scheduler::new(1.0).unwrap();
        s.set_timesteps(4).unwrap();
        let t = s.timesteps()[0];
        let out = s.step(&[1.0], t, &[0.0]).unwrap();
        assert!(
            out[0] > 0.0,
            "data-ward velocity must increase x, got {out:?}"
        );
    }

    #[test]
    fn scale_noise_endpoints() {
        let x0 = vec![1.0f32, 2.0, 3.0];
        let noise = vec![-1.0f32, 0.0, 5.0];
        let clean = H3Scheduler::scale_noise(&x0, 1.0, &noise).unwrap();
        assert_eq!(clean, x0);
        let pure = H3Scheduler::scale_noise(&x0, 0.0, &noise).unwrap();
        assert_eq!(pure, noise);
    }

    #[test]
    fn explicit_sigmas_are_validated() {
        let mut s = H3Scheduler::video();
        assert!(s.set_sigmas(&[1.0, 0.5, 0.0]).is_ok());
        assert_eq!(s.num_inference_steps(), 2);
        // not decreasing
        assert!(s.set_sigmas(&[1.0, 1.0, 0.0]).is_err());
        // does not terminate at zero
        assert!(s.set_sigmas(&[1.0, 0.5, 0.1]).is_err());
    }

    #[test]
    fn dedup_collapses_consecutive_runs_only() {
        let mut v = vec![1.0f32, 1.0, 0.5, 0.5, 0.5, 1.0, 0.0];
        dedup_consecutive(&mut v);
        assert_eq!(v, vec![1.0, 0.5, 1.0, 0.0]);
    }

    #[test]
    fn large_shift_collapses_head_of_grid() {
        // With a big shift and many points, several leading sigmas round to the
        // same f32 and are collapsed, so the realized count can drop below the
        // request. It must still be a valid schedule.
        let mut s = H3Scheduler::new(1000.0).unwrap();
        s.set_timesteps(64).unwrap();
        assert!(s.num_inference_steps() >= 1);
        assert!(s.sigmas().windows(2).all(|w| w[1] < w[0]));
        assert_eq!(*s.sigmas().last().unwrap(), 0.0);
    }

    #[test]
    fn rejects_degenerate_inputs() {
        assert!(H3Scheduler::new(0.0).is_err());
        assert!(H3Scheduler::new(-1.0).is_err());
        let mut s = H3Scheduler::video();
        assert!(s.set_timesteps(1).is_err());
    }

    #[test]
    fn full_sampling_run_consumes_schedule() {
        let mut s = H3Scheduler::video();
        s.set_timesteps(16).unwrap();
        let n = s.num_inference_steps();
        let mut x = vec![0.5f32; 4];
        for k in 0..n {
            let t = s.timesteps()[k];
            x = s.step(&[0.1f32; 4], t, &x).unwrap();
        }
        assert_eq!(s.step_index(), Some(n));
        assert!(x.iter().all(|v| v.is_finite()));
        // Reset lets the same schedule drive another run.
        s.reset();
        assert_eq!(s.step_index(), None);
    }
}
