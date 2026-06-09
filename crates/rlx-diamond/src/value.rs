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

//! Value function estimators (Diamond Maps Proposition 4.1).

const NEG_INF: f32 = -1e30;

/// log (1/K Σ exp(r_k)) — stable log-mean-exp.
pub fn log_mean_exp(rewards: &[f32]) -> f32 {
    assert!(!rewards.is_empty());
    let max_r = rewards.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = rewards.iter().map(|r| (r - max_r).exp()).sum();
    max_r + (sum / rewards.len() as f32).ln()
}

/// Softmax weights exp(r_k) / Σ exp(r_j).
pub fn softmax_weights(rewards: &[f32]) -> Vec<f32> {
    assert!(!rewards.is_empty());
    let max_r = rewards.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = rewards.iter().map(|r| (r - max_r).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|e| e / sum).collect()
}

/// Online log-sum-exp accumulation (reference `make_guidance_value_and_grad_fn`).
#[derive(Debug, Clone)]
pub struct LogSumExpGrad {
    logsumexp: f32,
    grad: Vec<f32>,
}

impl LogSumExpGrad {
    pub fn new(dim: usize) -> Self {
        Self {
            logsumexp: NEG_INF,
            grad: vec![0.0; dim],
        }
    }

    /// Incorporate one particle: reward value and ∂r/∂x_t.
    pub fn accumulate(&mut self, reward: f32, grad: &[f32]) {
        assert_eq!(self.grad.len(), grad.len());
        if !self.logsumexp.is_finite() || self.logsumexp <= NEG_INF / 2.0 {
            self.logsumexp = reward;
            self.grad.copy_from_slice(grad);
            return;
        }
        let logsumexp_next = logaddexp(self.logsumexp, reward);
        let w_prev = (self.logsumexp - logsumexp_next).exp();
        let w_curr = (reward - logsumexp_next).exp();
        for (g, &dg) in self.grad.iter_mut().zip(grad.iter()) {
            *g = *g * w_prev + dg * w_curr;
        }
        self.logsumexp = logsumexp_next;
    }

    pub fn value(&self) -> f32 {
        self.logsumexp
    }

    pub fn grad(&self) -> &[f32] {
        &self.grad
    }
}

#[inline]
pub fn logaddexp(a: f32, b: f32) -> f32 {
    if a > b {
        a + (1.0 + (b - a).exp()).ln()
    } else {
        b + (1.0 + (a - b).exp()).ln()
    }
}

/// Aggregate particle gradients: Σ softmax(r)_k ∇_{x_t} r(z^k).
pub fn softmax_grad_aggregate(rewards: &[f32], grads: &[Vec<f32>]) -> Vec<f32> {
    assert!(!rewards.is_empty());
    assert_eq!(rewards.len(), grads.len());
    let weights = softmax_weights(rewards);
    let dim = grads[0].len();
    let mut out = vec![0.0f32; dim];
    for (w, g) in weights.iter().zip(grads.iter()) {
        for (o, &gi) in out.iter_mut().zip(g.iter()) {
            *o += w * gi;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_mean_exp_two_particles() {
        let r = [0.0f32, 1.0];
        let v = log_mean_exp(&r);
        let expected = (0.5f32 * (1.0 + std::f32::consts::E)).ln();
        assert!((v - expected).abs() < 1e-4);
    }

    #[test]
    fn online_matches_batch_softmax() {
        let rewards = [0.1f32, 0.5, 0.2];
        let grads = [vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
        let batch = softmax_grad_aggregate(&rewards, &grads);
        let mut online = LogSumExpGrad::new(2);
        for (r, g) in rewards.iter().zip(grads.iter()) {
            online.accumulate(*r, g);
        }
        for i in 0..2 {
            assert!((batch[i] - online.grad()[i]).abs() < 1e-5);
        }
    }
}
