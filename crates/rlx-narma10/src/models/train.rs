// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

use crate::host::{ORDER, Series, nrmse};
use crate::ridge::{fit_ridge, predict};
use crate::reservoir::Reservoir;
use anyhow::Result;

/// Quick-check timesteps (Nakajima-style dense ESN).
pub const QUICK_TIMESTEPS: usize = 5_000;
/// Quick-check washout.
pub const QUICK_WASHOUT: usize = 100;

/// LCESN paper washout (Matzner & Mráz, ICLR 2025).
pub const LCESN_WASHOUT: usize = 1000;
/// Post-washout readout training pairs.
pub const LCESN_TRAIN_SAMPLES: usize = 12_000;
/// Held-out test target steps after the train segment.
pub const LCESN_TEST_SAMPLES: usize = 1_000;
/// Total target timesteps (train segment + test segment; washout is a prefix).
pub const LCESN_TIMESTEPS: usize = 14_000;

/// Training and evaluation protocol (washout + train/test split).
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Discard this many timesteps before collecting reservoir states.
    pub washout: usize,
    /// Fraction of target indices `j` used for readout training (`j < train_n`).
    pub train_frac: f64,
    /// Ridge penalty λ on readout weights (`(XᵀX + λI)⁻¹Xᵀy`).
    pub ridge_lambda: f32,
    /// Seed for reservoir weight initialization (independent of [`crate::host::generate`]).
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            washout: 100,
            train_frac: 0.7,
            ridge_lambda: 1e-8,
            seed: 42,
        }
    }
}

impl TrainConfig {
    /// LCESN paper protocol: washout 1000, ~12k train / 1k test at [`LCESN_TIMESTEPS`].
    pub fn lcesn() -> Self {
        Self {
            washout: LCESN_WASHOUT,
            train_frac: Self::train_frac_for_collected(
                LCESN_TIMESTEPS,
                LCESN_WASHOUT,
                LCESN_TRAIN_SAMPLES,
            ),
            ridge_lambda: 1e-8,
            seed: 42,
        }
    }

    /// Nakajima-style dense ESN check: [`QUICK_TIMESTEPS`] steps, washout [`QUICK_WASHOUT`].
    pub fn quick() -> Self {
        Self {
            washout: QUICK_WASHOUT,
            train_frac: 0.75,
            ridge_lambda: 1e-8,
            seed: 42,
        }
    }

    /// Alias for [`Self::lcesn`].
    pub fn long_sequence() -> Self {
        Self::lcesn()
    }

    /// `train_frac` so that `n_timesteps` yields `train_collect` post-washout samples.
    ///
    /// Post-washout count is `floor(train_frac * n) - (washout - ORDER)` when `washout >= ORDER`.
    pub fn train_frac_for_collected(
        n_timesteps: usize,
        washout: usize,
        train_collect: usize,
    ) -> f64 {
        let train_end = washout.saturating_sub(ORDER) + train_collect;
        train_end as f64 / n_timesteps as f64
    }

    /// Expected post-washout training samples when `series` has `n_timesteps` targets.
    pub fn expected_train_samples(&self, n_timesteps: usize) -> usize {
        let train_n = ((n_timesteps as f64) * self.train_frac).floor() as usize;
        train_n.saturating_sub(self.washout.saturating_sub(ORDER))
    }
}

/// Fit statistics from [`Narma10Predictor::fit`].
#[derive(Debug, Clone)]
pub struct TrainReport {
    /// Post-washout state–target pairs used for ridge regression.
    pub train_samples: usize,
    /// Target steps in the held-out test segment.
    pub test_samples: usize,
    /// In-sample NRMSE on the training design matrix.
    pub train_nrmse: f64,
    /// First target index in the held-out test segment.
    pub split_index: usize,
}

/// One row from [`bench_predictors`].
#[derive(Debug, Clone)]
pub struct BenchRow {
    /// Predictor name (e.g. `esn_ridge`).
    pub name: &'static str,
    pub train_nrmse: f64,
    /// NRMSE on `targets[split_index..]` after [`Narma10Predictor::fit`].
    pub test_nrmse: f64,
    pub train_samples: usize,
    pub test_samples: usize,
}

/// Trait for NARMA-10 one-step-ahead forecasting models.
pub trait Narma10Predictor {
    /// Short identifier used in [`bench_predictors`] output.
    fn name(&self) -> &'static str;

    /// Fit readout weights; returns in-sample train NRMSE on collected states.
    fn fit(&mut self, series: &Series, cfg: &TrainConfig) -> Result<TrainReport>;

    /// Predict `y[t]` for every target timestep (input-driven reservoir replay).
    fn predict_all(&mut self, series: &Series, cfg: &TrainConfig) -> Result<Vec<f64>>;
}

pub(crate) struct ReadoutTrainer {
    reservoir: Reservoir,
    weights: Vec<f32>,
    feature_dim: usize,
}

pub(crate) trait FeatureMap {
    fn dim(&self, units: usize) -> usize;
    fn append(&self, state: &[f32], out: &mut Vec<f32>);
}

pub(crate) struct LinearFeatures;

impl FeatureMap for LinearFeatures {
    fn dim(&self, units: usize) -> usize {
        units + 1
    }

    fn append(&self, state: &[f32], out: &mut Vec<f32>) {
        out.push(1.0);
        out.extend_from_slice(state);
    }
}

pub(crate) struct QuadraticFeatures;

impl FeatureMap for QuadraticFeatures {
    fn dim(&self, units: usize) -> usize {
        1 + 2 * units
    }

    fn append(&self, state: &[f32], out: &mut Vec<f32>) {
        out.push(1.0);
        out.extend_from_slice(state);
        for &x in state {
            out.push(x * x);
        }
    }
}

impl ReadoutTrainer {
    pub(crate) fn new(reservoir: Reservoir, features: &dyn FeatureMap) -> Self {
        let feature_dim = features.dim(reservoir.units());
        Self {
            reservoir,
            weights: Vec::new(),
            feature_dim,
        }
    }

    pub(crate) fn fit_inner(
        &mut self,
        series: &Series,
        cfg: &TrainConfig,
        features: &dyn FeatureMap,
    ) -> Result<TrainReport> {
        let inputs_f32: Vec<f32> = series.inputs.iter().map(|&u| u as f32).collect();
        let n = series.targets.len();
        let train_n = ((n as f64) * cfg.train_frac).floor() as usize;
        let train_n = train_n.max(ORDER).min(n.saturating_sub(1));
        self.reservoir.reset();

        let mut design = Vec::new();
        let mut targets = Vec::new();

        for t in 0..series.inputs.len() {
            if t + 1 < series.inputs.len() {
                self.reservoir.step(inputs_f32[t], 0.0);
            }
            let time = t + 1;
            if time >= ORDER {
                let j = time - ORDER;
                if time >= cfg.washout && j < train_n {
                    let start = design.len();
                    features.append(self.reservoir.state(), &mut design);
                    debug_assert_eq!(design.len() - start, self.feature_dim);
                    targets.push(series.targets[j] as f32);
                }
            }
        }

        let m = targets.len();
        self.weights = fit_ridge(&design, m, self.feature_dim, &targets, cfg.ridge_lambda)?;

        let train_pred = predict(&design, m, self.feature_dim, &self.weights);
        let train_pred64: Vec<f64> = train_pred.iter().map(|&v| v as f64).collect();
        let targets64: Vec<f64> = targets.iter().map(|&v| v as f64).collect();
        let train_nrmse = nrmse(&train_pred64, &targets64);

        Ok(TrainReport {
            train_samples: m,
            test_samples: n - train_n,
            train_nrmse,
            split_index: train_n,
        })
    }

    pub(crate) fn predict_all_inner(
        &mut self,
        series: &Series,
        _cfg: &TrainConfig,
        features: &dyn FeatureMap,
    ) -> Result<Vec<f64>> {
        let inputs_f32: Vec<f32> = series.inputs.iter().map(|&u| u as f32).collect();
        let n = series.targets.len();
        self.reservoir.reset();

        let mut out = Vec::with_capacity(n);
        let mut feat = Vec::with_capacity(self.feature_dim);

        for t in 0..series.inputs.len() {
            if t + 1 < series.inputs.len() {
                self.reservoir.step(inputs_f32[t], 0.0);
            }
            let time = t + 1;
            if time >= ORDER {
                let j = time - ORDER;
                if j < n {
                    feat.clear();
                    features.append(self.reservoir.state(), &mut feat);
                    let y_hat: f32 = feat.iter().zip(&self.weights).map(|(a, b)| a * b).sum();
                    out.push(y_hat as f64);
                }
            }
        }

        Ok(out)
    }
}

/// Naive persistence baseline NRMSE: `ŷ(t) = y(t−1)`.
pub fn persistence_nrmse(targets: &[f64]) -> f64 {
    if targets.len() < 2 {
        return f64::INFINITY;
    }
    let pred: Vec<f64> = std::iter::once(targets[0])
        .chain(targets[..targets.len() - 1].iter().copied())
        .collect();
    nrmse(&pred, targets)
}

/// Run all three predictors on one series; rows are sorted by test NRMSE (best first).
pub fn bench_predictors(series: &Series, cfg: &TrainConfig) -> Result<Vec<BenchRow>> {
    let mut rows = Vec::new();
    let models: Vec<Box<dyn Narma10Predictor>> = vec![
        Box::new(super::EsnRidge::new()),
        Box::new(super::LocalEsn::new()),
        Box::new(super::PolyReadoutEsn::new()),
    ];
    for mut model in models {
        let report = model.fit(series, cfg)?;
        let pred = model.predict_all(series, cfg)?;
        let n = series.targets.len();
        let test_start = report.split_index.min(n);
        let test_nrmse = if test_start < n {
            nrmse(&pred[test_start..n], &series.targets[test_start..n])
        } else {
            f64::INFINITY
        };
        rows.push(BenchRow {
            name: model.name(),
            train_nrmse: report.train_nrmse,
            test_nrmse,
            train_samples: report.train_samples,
            test_samples: report.test_samples,
        });
    }
    rows.sort_by(|a, b| {
        a.test_nrmse
            .partial_cmp(&b.test_nrmse)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(rows)
}
