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

/// Locally connected echo-state network with linear ridge readout.
///
/// Reservoir: [`ReservoirConfig::local_lcesn`] (800 units, 20×40 toroidal grid, kernel 7).
/// Use [`TrainConfig::lcesn`] for the paper protocol (washout 1000, 12k train).
pub struct LocalEsn {
    inner: Option<ReadoutTrainer>,
}

impl Default for LocalEsn {
    fn default() -> Self {
        Self { inner: None }
    }
}

impl LocalEsn {
    /// New untrained predictor; call [`Narma10Predictor::fit`] before predict.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Narma10Predictor for LocalEsn {
    fn name(&self) -> &'static str {
        "local_esn"
    }

    fn fit(&mut self, series: &Series, cfg: &TrainConfig) -> Result<TrainReport> {
        let mut cfg = cfg.clone();
        cfg.ridge_lambda = cfg.ridge_lambda.max(1e-8);
        let res = Reservoir::new(ReservoirConfig::local_lcesn(), cfg.seed);
        let mut inner = ReadoutTrainer::new(res, &LinearFeatures);
        let report = inner.fit_inner(series, &cfg, &LinearFeatures)?;
        self.inner = Some(inner);
        Ok(report)
    }

    fn predict_all(&mut self, series: &Series, cfg: &TrainConfig) -> Result<Vec<f64>> {
        let inner = self.inner.as_mut().expect("fit before predict");
        inner.predict_all_inner(series, cfg, &LinearFeatures)
    }
}
