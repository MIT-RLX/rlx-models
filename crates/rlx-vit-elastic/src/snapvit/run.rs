// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! End-to-end SnapViT: local Hessian diagonal → xNES global scaling → a
//! continuum of elastic pruned sub-networks (one score, many sparsities).

use anyhow::Result;
use rlx_runtime::Device;

use crate::vit::config::VitConfig;
use crate::vit::weights::LoadedVit;

use super::fitness::Fitness;
use super::local::{CalibImage, LocalScores, SnapVitConfig, compute_local_scores};
use super::prune::{PruneResult, prunability, prune_at};
use super::xnes::{XnesConfig, optimize};

/// One elastic operating point.
#[derive(Clone, Debug)]
pub struct ElasticEntry {
    pub sparsity: f32,
    pub head_mask: Vec<f32>,
    pub ffn_mask: Vec<f32>,
    /// PCA-cosine retention vs the original model (higher is better).
    pub fitness: f32,
    pub heads_pruned: usize,
    pub ffn_pruned: usize,
    pub param_reduction: f32,
}

/// Full SnapViT output.
pub struct SnapVitResult {
    pub local: LocalScores,
    /// Evolved block scaling `c` (positive multipliers).
    pub c: Vec<f32>,
    pub baseline_fitness: f32,
    pub best_fitness: f32,
    pub xnes_history: Vec<f32>,
    /// One entry per requested elastic sparsity.
    pub elastic: Vec<ElasticEntry>,
}

/// SnapViT run parameters.
#[derive(Clone)]
pub struct SnapVitParams {
    pub ssl: SnapVitConfig,
    pub xnes: XnesConfig,
    /// PCA dimension for the fitness (`0` / `>= hidden` ⇒ raw cosine).
    pub pca_dim: usize,
    /// Sparsities to materialize in the elastic export.
    pub elastic_sparsities: Vec<f32>,
}

impl SnapVitParams {
    pub fn new(img_size: usize) -> Self {
        Self {
            ssl: SnapVitConfig::new(img_size),
            xnes: XnesConfig::default(),
            pca_dim: 192,
            elastic_sparsities: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    }
}

/// Run SnapViT: compute local scores on `calib`, evolve the global scaling on
/// `fit`, then materialize the elastic pruned sub-networks.
pub fn run(
    cfg: &VitConfig,
    loaded: &LoadedVit,
    calib: &[CalibImage],
    fit: &[CalibImage],
    params: &SnapVitParams,
    device: Device,
) -> Result<SnapVitResult> {
    let local = compute_local_scores(cfg, loaded, calib, &params.ssl, device)?;

    // The fitness forward uses batch = |fit|; RLX's Metal/MLX batched
    // transpose/narrow forward NaNs on this graph, so the SSL fitness runs on
    // CPU (as does the gradient). Exported pruned models still deploy on any
    // backend via `VitRunner` (batch-1 forward is backend-parity-verified).
    let fit_device = crate::snapvit::local::backward_device(device, "snapvit fitness");
    let fit_loaded = LoadedVit {
        params: loaded.params.clone(),
        preprocess: loaded.preprocess.clone(),
    };
    let mut fitness = Fitness::new(cfg, fit_loaded, fit, fit_device, params.pca_dim)?;

    let xr = optimize(cfg, &local, &mut fitness, &params.xnes)?;

    let p = prunability(cfg, &local, &xr.best_c);
    let mut elastic = Vec::with_capacity(params.elastic_sparsities.len());
    for &s in &params.elastic_sparsities {
        let PruneResult {
            head_mask,
            ffn_mask,
            heads_pruned,
            ffn_pruned,
            param_reduction,
            ..
        } = prune_at(cfg, &p, s);
        let fitness_s = fitness.eval(head_mask.clone(), ffn_mask.clone())?;
        elastic.push(ElasticEntry {
            sparsity: s,
            head_mask,
            ffn_mask,
            fitness: fitness_s,
            heads_pruned,
            ffn_pruned,
            param_reduction,
        });
    }

    Ok(SnapVitResult {
        local,
        c: xr.best_c,
        baseline_fitness: xr.baseline_fitness,
        best_fitness: xr.best_fitness,
        xnes_history: xr.history,
        elastic,
    })
}
