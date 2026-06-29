// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

/// Candidate mask ratios (Sec. 4.3): 0.1..=0.9 step 0.1.
pub const MASK_RATIO_CANDIDATES: &[f32] = &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

/// Eq. 3 — μ_v = (1/L) Σ_l d_v^l.
pub fn compute_mu(dynamics: &[Vec<f32>]) -> Vec<f32> {
    dynamics
        .iter()
        .map(|layers| {
            if layers.is_empty() {
                0.0
            } else {
                layers.iter().sum::<f32>() / layers.len() as f32
            }
        })
        .collect()
}

/// Eq. 4 — Ent_v = Σ_l -(d_v^l / (L·μ_v)) log(d_v^l / (L·μ_v)).
pub fn compute_token_entropies(dynamics: &[Vec<f32>], mu: &[f32]) -> Vec<f32> {
    assert_eq!(dynamics.len(), mu.len());
    dynamics
        .iter()
        .zip(mu.iter())
        .map(|(layers, &m)| token_entropy_eq4(layers, m))
        .collect()
}

fn token_entropy_eq4(layers: &[f32], mu: f32) -> f32 {
    let l = layers.len() as f64;
    let denom = l * mu as f64;
    if denom <= 0.0 {
        return 0.0;
    }
    let mut ent = 0.0f64;
    for &d in layers {
        let p = (d as f64) / denom;
        if p > 0.0 {
            ent -= p * p.ln();
        }
    }
    ent as f32
}

/// Eq. 5 — S = -Σ_i (μ_i / Σμ) log(μ_i / Σμ).
pub fn distribution_entropy(mu: &[f32]) -> f32 {
    let total: f64 = mu.iter().map(|&x| x.max(0.0) as f64).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut ent = 0.0f64;
    for &m in mu {
        let p = (m.max(0.0) as f64) / total;
        if p > 0.0 {
            ent -= p * p.ln();
        }
    }
    ent as f32
}

/// Sec. 4.3 — mask highest-Ent tokens; pick ratio r with max |S(r) - S0|.
pub fn select_adaptive_mask_ratio(mu: &[f32], token_entropy: &[f32]) -> f32 {
    assert_eq!(mu.len(), token_entropy.len());
    let n = mu.len();
    if n == 0 {
        return 0.5;
    }
    let s0 = distribution_entropy(mu);
    let mut by_ent: Vec<usize> = (0..n).collect();
    by_ent.sort_by(|&a, &b| {
        token_entropy[b]
            .partial_cmp(&token_entropy[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mut best_ratio = 0.5f32;
    let mut best_dist = -1.0f64;
    for &ratio in MASK_RATIO_CANDIDATES {
        let block_n = block_count(n, ratio);
        if block_n == 0 || block_n >= n {
            continue;
        }
        let mu_keep: Vec<f32> = mu
            .iter()
            .enumerate()
            .filter(|(i, _)| !by_ent.iter().take(block_n).any(|&b| b == *i))
            .map(|(_, &m)| m)
            .collect();
        let s = distribution_entropy(&mu_keep);
        let dist = (f64::from(s) - f64::from(s0)).abs();
        if dist > best_dist {
            best_dist = dist;
            best_ratio = ratio;
        }
    }
    best_ratio
}

pub(crate) fn block_count(n: usize, ratio: f32) -> usize {
    if n <= 1 || ratio <= 0.0 {
        return 0;
    }
    let r = ratio.clamp(0.0, 1.0);
    let raw = ((n as f32) * r).ceil() as usize;
    raw.max(1).min(n.saturating_sub(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eq4_hand_example() {
        let layers = vec![0.2, 0.8];
        let mu = 0.5f32;
        let ent = token_entropy_eq4(&layers, mu);
        // p = [0.2, 0.8] after norm by L*mu=1.0
        let expected = -(0.2f64 * 0.2f64.ln() + 0.8f64 * 0.8f64.ln());
        assert!((ent as f64 - expected).abs() < 1e-5);
    }

    #[test]
    fn block_count_small_grid() {
        assert_eq!(block_count(1, 0.5), 0);
        assert_eq!(block_count(2, 0.1), 1);
        assert_eq!(block_count(4, 0.1), 1);
        assert_eq!(block_count(4, 0.9), 3);
    }

    #[test]
    fn eq5_uniform_nonzero() {
        let mu = vec![0.25; 4];
        let s = distribution_entropy(&mu);
        assert!((s as f64 - (4.0_f64.ln())).abs() < 1e-4);
    }
}
