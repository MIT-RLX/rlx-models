// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

use crate::host::Series;
use crate::models::train::{
    LinearFeatures, Narma10Predictor, ReadoutTrainer, TrainConfig, TrainReport,
};
use crate::reservoir::{Reservoir, ReservoirConfig};
use anyhow::Result;

/// Classic echo-state network with linear ridge readout.
///
/// Reservoir: [`ReservoirConfig::dense_standard`] (N=300, ρ=0.9, Nakajima RC-tutorial /
/// Kodali et al. 2025 NARMA-10 baseline).
pub struct EsnRidge {
    inner: Option<ReadoutTrainer>,
}

impl Default for EsnRidge {
    fn default() -> Self {
        Self { inner: None }
    }
}

impl EsnRidge {
    /// New untrained predictor; call [`Narma10Predictor::fit`] before predict.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Narma10Predictor for EsnRidge {
    fn name(&self) -> &'static str {
        "esn_ridge"
    }

    fn fit(&mut self, series: &Series, cfg: &TrainConfig) -> Result<TrainReport> {
        let res = Reservoir::new(ReservoirConfig::dense_standard(), cfg.seed);
        let mut inner = ReadoutTrainer::new(res, &LinearFeatures);
        let report = inner.fit_inner(series, cfg, &LinearFeatures)?;
        self.inner = Some(inner);
        Ok(report)
    }

    fn predict_all(&mut self, series: &Series, cfg: &TrainConfig) -> Result<Vec<f64>> {
        let inner = self.inner.as_mut().expect("fit before predict");
        inner.predict_all_inner(series, cfg, &LinearFeatures)
    }
}
