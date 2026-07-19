// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Global correlation term via an exponential-family Natural Evolution Strategy
//! (SnapViT §3.3). The block-scaling vector `c = exp(g)` is optimized in
//! log-space by **separable NES** — per-dimension mean/variance natural-gradient
//! updates driven by the label-free fitness. (The paper's full-covariance xNES
//! additionally models cross-block off-diagonals; the separable variant keeps
//! the search tractable for hundreds of blocks and is a documented deviation.)

use anyhow::Result;

use crate::dino::Rng;
use crate::vit::config::VitConfig;

use super::fitness::Fitness;
use super::local::LocalScores;
use super::prune::{coeffs_len, prunability, prune_at};

/// xNES / sNES search settings.
#[derive(Clone)]
pub struct XnesConfig {
    pub population: usize,
    pub iterations: usize,
    /// Sparsity levels the fitness averages over (Eq. 6).
    pub sparsities: Vec<f32>,
    pub eta_mu: f32,
    /// `0` ⇒ the NES default `(3 + ln B) / (5·√B)`.
    pub eta_sigma: f32,
    pub sigma0: f32,
    pub seed: u64,
}

impl Default for XnesConfig {
    fn default() -> Self {
        Self {
            population: 8,
            iterations: 30,
            sparsities: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            eta_mu: 1.0,
            eta_sigma: 0.0,
            sigma0: 0.3,
            seed: 0xACE1,
        }
    }
}

/// Result of the search.
pub struct XnesResult {
    /// Best block-scaling `c = exp(g)` found (positive multipliers).
    pub best_c: Vec<f32>,
    pub best_fitness: f32,
    /// Fitness at `c = 1` (pure local ranking) — the baseline.
    pub baseline_fitness: f32,
    /// Best fitness per iteration.
    pub history: Vec<f32>,
}

/// Mean fitness over the sparsity set for a candidate `c`.
fn eval_c(
    cfg: &VitConfig,
    local: &LocalScores,
    fitness: &mut Fitness,
    sparsities: &[f32],
    c: &[f32],
) -> Result<f32> {
    let p = prunability(cfg, local, c);
    let mut acc = 0.0f32;
    for &s in sparsities {
        let pr = prune_at(cfg, &p, s);
        acc += fitness.eval(pr.head_mask, pr.ffn_mask)?;
    }
    Ok(acc / sparsities.len().max(1) as f32)
}

/// NES rank utilities (best-first): `u_i = max(0, ln(λ/2+1) − ln i)`,
/// normalized to sum 1, minus `1/λ` (so they sum to 0).
fn utilities(lambda: usize) -> Vec<f32> {
    let mut u: Vec<f32> = (1..=lambda)
        .map(|i| (((lambda as f32) / 2.0 + 1.0).ln() - (i as f32).ln()).max(0.0))
        .collect();
    let sum: f32 = u.iter().sum::<f32>().max(1e-12);
    for x in u.iter_mut() {
        *x = *x / sum - 1.0 / lambda as f32;
    }
    u
}

/// Optimize the block scaling `c` to maximize the label-free fitness.
pub fn optimize(
    cfg: &VitConfig,
    local: &LocalScores,
    fitness: &mut Fitness,
    xc: &XnesConfig,
) -> Result<XnesResult> {
    let b = coeffs_len(cfg);
    let lambda = xc.population.max(4);
    let eta_sigma = if xc.eta_sigma > 0.0 {
        xc.eta_sigma
    } else {
        (3.0 + (b as f32).ln()) / (5.0 * (b as f32).sqrt())
    };
    let util = utilities(lambda);

    // Baseline: c = 1 (pure local ranking).
    let ones = vec![1.0f32; b];
    let baseline = eval_c(cfg, local, fitness, &xc.sparsities, &ones)?;

    let mut mu = vec![0.0f32; b]; // log-space; c = exp(mu) = 1 at start
    let mut sigma = vec![xc.sigma0; b];
    let mut rng = Rng::new(xc.seed);

    let mut best_c = ones.clone();
    let mut best_fitness = baseline;
    let mut history = Vec::with_capacity(xc.iterations);

    for _ in 0..xc.iterations {
        // Sample population.
        let mut samples: Vec<Vec<f32>> = Vec::with_capacity(lambda);
        let mut cands: Vec<Vec<f32>> = Vec::with_capacity(lambda);
        let mut fits: Vec<f32> = Vec::with_capacity(lambda);
        for _ in 0..lambda {
            let s: Vec<f32> = (0..b).map(|_| rng.gauss()).collect();
            let c: Vec<f32> = (0..b).map(|d| (mu[d] + sigma[d] * s[d]).exp()).collect();
            let f = eval_c(cfg, local, fitness, &xc.sparsities, &c)?;
            if f > best_fitness {
                best_fitness = f;
                best_c = c.clone();
            }
            samples.push(s);
            cands.push(c);
            fits.push(f);
        }

        // Rank descending (best first) → utility per sample.
        let mut order: Vec<usize> = (0..lambda).collect();
        order.sort_by(|&a, &b| fits[b].total_cmp(&fits[a]));

        // Natural-gradient estimates.
        let mut grad_mu = vec![0.0f32; b];
        let mut grad_sigma = vec![0.0f32; b];
        for (rank, &k) in order.iter().enumerate() {
            let u = util[rank];
            for d in 0..b {
                grad_mu[d] += u * samples[k][d];
                grad_sigma[d] += u * (samples[k][d] * samples[k][d] - 1.0);
            }
        }
        for d in 0..b {
            mu[d] += xc.eta_mu * sigma[d] * grad_mu[d];
            sigma[d] *= (0.5 * eta_sigma * grad_sigma[d]).exp();
        }

        history.push(best_fitness);
    }

    Ok(XnesResult {
        best_c,
        best_fitness,
        baseline_fitness: baseline,
        history,
    })
}
