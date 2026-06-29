// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use super::dynamics::{
    compute_mu, compute_token_entropies, distribution_entropy, select_adaptive_mask_ratio,
};
use super::mask::{VisionKeySpan, block_highest_entropy_keys};

/// Output of the probe forward (Fig. 6 step b).
#[derive(Debug, Clone)]
pub struct AifProbe {
    /// Eq. 2 — D_v = {d_v^l}: `[vision_idx][layer]`.
    pub dynamics: Vec<Vec<f32>>,
    /// Eq. 3.
    pub mu: Vec<f32>,
    /// Eq. 4 — higher ⇒ less important ⇒ mask first.
    pub token_entropy: Vec<f32>,
    /// Eq. 5 — S0 before masking.
    pub s0: f32,
    /// Sec. 4.3 — selected mask ratio r.
    pub mask_ratio: f32,
}

impl AifProbe {
    /// Build probe state from layer dynamics (Eq. 2–5).
    pub fn build(dynamics: Vec<Vec<f32>>) -> Self {
        let mu = compute_mu(&dynamics);
        let token_entropy = compute_token_entropies(&dynamics, &mu);
        let s0 = distribution_entropy(&mu);
        let mask_ratio = select_adaptive_mask_ratio(&mu, &token_entropy);
        Self {
            dynamics,
            mu,
            token_entropy,
            s0,
            mask_ratio,
        }
    }

    /// Sec. 4.3 — KV key indices to block for text decode queries.
    pub fn blocked_keys(&self, span: VisionKeySpan) -> Vec<usize> {
        block_highest_entropy_keys(span, &self.token_entropy, self.mask_ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_matches_equations() {
        let dynamics = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![0.1, 0.9]];
        let p = AifProbe::build(dynamics);
        assert_eq!(p.mu.len(), 3);
        assert!((p.mu[0] - 0.5).abs() < 1e-5);
        assert!(p.s0 > 0.0);
        assert!((0.1..=0.9).contains(&p.mask_ratio));
    }
}
