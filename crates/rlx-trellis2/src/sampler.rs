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

//! Flow-matching Euler sampler with classifier-free guidance and a guidance
//! interval (`FlowEulerGuidanceIntervalSampler`,
//! `trellis2/pipelines/samplers/flow_euler.py`).
//!
//! The model predicts velocity `v`; the Euler update is
//! `x_{t-1} = x_t - (t - t_prev)·v`. Time steps follow
//! `t = rescale_t·τ / (1 + (rescale_t-1)·τ)` for `τ = linspace(1, 0, steps+1)`,
//! and the model is queried at `1000·t`. CFG combines a positive and negative
//! prediction (`gs·pos + (1-gs)·neg`), optionally re-scaled to match the
//! positive branch's std, and is only active while `t ∈ guidance_interval`.

use crate::config::SamplerParams;

/// Everything the sampler needs beyond the per-model params.
#[derive(Debug, Clone, Copy)]
pub struct SamplerConfig {
    pub sigma_min: f32,
    pub steps: usize,
    pub guidance_strength: f32,
    pub guidance_rescale: f32,
    pub guidance_interval: [f32; 2],
    pub rescale_t: f32,
}

impl SamplerConfig {
    pub fn from_params(sigma_min: f32, p: &SamplerParams) -> Self {
        Self {
            sigma_min,
            steps: p.steps,
            guidance_strength: p.guidance_strength,
            guidance_rescale: p.guidance_rescale,
            guidance_interval: p.guidance_interval,
            rescale_t: p.rescale_t,
        }
    }
}

/// The time schedule `[(t, t_prev); steps]` used by the Euler integrator.
pub fn time_pairs(steps: usize, rescale_t: f32) -> Vec<(f32, f32)> {
    let n = steps + 1;
    let seq: Vec<f32> = (0..n)
        .map(|i| {
            let tau = 1.0 - i as f32 / (n as f32 - 1.0); // linspace(1,0,steps+1)
            rescale_t * tau / (1.0 + (rescale_t - 1.0) * tau)
        })
        .collect();
    (0..steps).map(|i| (seq[i], seq[i + 1])).collect()
}

fn std_all(x: &[f32]) -> f32 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f32;
    let mean = x.iter().sum::<f32>() / nf;
    // PyTorch std is unbiased (n-1); batch tensors here are large so it is
    // numerically irrelevant, but match it exactly.
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / (nf - 1.0);
    var.sqrt()
}

/// `x_0 = (1-σ)·x_t - (σ + (1-σ)·t)·pred`.
fn pred_to_xstart(x_t: &[f32], t: f32, pred: &[f32], sigma_min: f32) -> Vec<f32> {
    let a = 1.0 - sigma_min;
    let b = sigma_min + (1.0 - sigma_min) * t;
    x_t.iter().zip(pred).map(|(x, p)| a * x - b * p).collect()
}

/// `pred = ((1-σ)·x_t - x_0) / (σ + (1-σ)·t)`.
fn xstart_to_pred(x_t: &[f32], t: f32, x0: &[f32], sigma_min: f32) -> Vec<f32> {
    let a = 1.0 - sigma_min;
    let b = sigma_min + (1.0 - sigma_min) * t;
    x_t.iter().zip(x0).map(|(x, x0)| (a * x - x0) / b).collect()
}

/// One CFG velocity prediction at time `t`, honoring the guidance interval.
///
/// `model_v(x_t, t_scaled, cond)` returns the raw velocity prediction; the
/// sampler queries it at `t_scaled = 1000·t` with the positive and (if needed)
/// negative conditioning.
fn guided_prediction<F>(
    model_v: &mut F,
    x_t: &[f32],
    t: f32,
    cond: &[f32],
    neg_cond: &[f32],
    cfg: &SamplerConfig,
) -> Vec<f32>
where
    F: FnMut(&[f32], f32, &[f32]) -> Vec<f32>,
{
    let in_interval = cfg.guidance_interval[0] <= t && t <= cfg.guidance_interval[1];
    let gs = if in_interval {
        cfg.guidance_strength
    } else {
        1.0
    };
    let ts = 1000.0 * t;

    if gs == 1.0 {
        return model_v(x_t, ts, cond);
    }
    if gs == 0.0 {
        return model_v(x_t, ts, neg_cond);
    }
    let pos = model_v(x_t, ts, cond);
    let neg = model_v(x_t, ts, neg_cond);
    let mut pred: Vec<f32> = pos
        .iter()
        .zip(&neg)
        .map(|(p, n)| gs * p + (1.0 - gs) * n)
        .collect();

    if cfg.guidance_rescale > 0.0 {
        let x0_pos = pred_to_xstart(x_t, t, &pos, cfg.sigma_min);
        let x0_cfg = pred_to_xstart(x_t, t, &pred, cfg.sigma_min);
        let std_pos = std_all(&x0_pos);
        let std_cfg = std_all(&x0_cfg);
        let ratio = if std_cfg > 0.0 {
            std_pos / std_cfg
        } else {
            1.0
        };
        let gr = cfg.guidance_rescale;
        let x0: Vec<f32> = x0_cfg
            .iter()
            .map(|v| gr * (v * ratio) + (1.0 - gr) * v)
            .collect();
        pred = xstart_to_pred(x_t, t, &x0, cfg.sigma_min);
    }
    pred
}

/// Integrate the flow ODE with Euler steps, returning the final sample (same
/// shape/layout as `noise`).
///
/// `model_v(x_t, t_scaled, cond) -> v` is the flow-matching model applied in
/// whatever tokenization the caller uses (dense grid or sparse voxels).
pub fn flow_euler_sample<F>(
    mut model_v: F,
    noise: &[f32],
    cond: &[f32],
    neg_cond: &[f32],
    cfg: &SamplerConfig,
) -> Vec<f32>
where
    F: FnMut(&[f32], f32, &[f32]) -> Vec<f32>,
{
    let mut x = noise.to_vec();
    for (t, t_prev) in time_pairs(cfg.steps, cfg.rescale_t) {
        let v = guided_prediction(&mut model_v, &x, t, cond, neg_cond, cfg);
        let dt = t - t_prev;
        for (xi, vi) in x.iter_mut().zip(&v) {
            *xi -= dt * vi;
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_endpoints() {
        let pairs = time_pairs(12, 5.0);
        assert_eq!(pairs.len(), 12);
        assert!((pairs[0].0 - 1.0).abs() < 1e-6, "first t should be 1.0");
        assert!(
            (pairs[11].1 - 0.0).abs() < 1e-6,
            "last t_prev should be 0.0"
        );
        // monotonically decreasing
        for w in pairs.windows(2) {
            assert!(w[0].0 > w[1].0);
        }
    }

    #[test]
    fn rescale_identity_when_one() {
        // rescale_t = 1 -> plain linspace(1,0,steps+1)
        let pairs = time_pairs(4, 1.0);
        let expect = [1.0, 0.75, 0.5, 0.25, 0.0];
        for (i, &(t, _)) in pairs.iter().enumerate() {
            assert!((t - expect[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn euler_linear_field_hits_zero() {
        // v(x) = x with rescale_t=1: exact flow x_{t-1}=x_t-(dt)x_t; not exactly
        // zero but must shrink monotonically toward 0.
        let cfg = SamplerConfig {
            sigma_min: 1e-5,
            steps: 50,
            guidance_strength: 1.0,
            guidance_rescale: 0.0,
            guidance_interval: [0.0, 1.0],
            rescale_t: 1.0,
        };
        let noise = vec![3.0f32; 4];
        let out = flow_euler_sample(|x, _t, _c| x.to_vec(), &noise, &[], &[], &cfg);
        assert!(out.iter().all(|&v| v.abs() < 3.0));
    }
}
