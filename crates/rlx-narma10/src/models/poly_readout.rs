// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

use crate::host::Series;
use crate::models::train::{
    Narma10Predictor, QuadraticFeatures, ReadoutTrainer, TrainConfig, TrainReport,
};
use crate::reservoir::{Reservoir, ReservoirConfig};
use anyhow::Result;

/// Dense ESN with quadratic feature readout (HCNN-inspired nonlinear readout).
///
/// Reservoir: [`ReservoirConfig::dense_poly`] (N=400). Appends `x²` features before ridge.
#[derive(Default)]
pub struct PolyReadoutEsn {
    inner: Option<ReadoutTrainer>,
}

impl PolyReadoutEsn {
    /// New untrained predictor; call [`Narma10Predictor::fit`] before predict.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Narma10Predictor for PolyReadoutEsn {
    fn name(&self) -> &'static str {
        "poly_readout_esn"
    }

    fn fit(&mut self, series: &Series, cfg: &TrainConfig) -> Result<TrainReport> {
        let mut cfg = cfg.clone();
        cfg.ridge_lambda = cfg.ridge_lambda.max(1e-8);
        let res = Reservoir::new(ReservoirConfig::dense_poly(), cfg.seed.wrapping_add(1));
        let mut inner = ReadoutTrainer::new(res, &QuadraticFeatures);
        let report = inner.fit_inner(series, &cfg, &QuadraticFeatures)?;
        self.inner = Some(inner);
        Ok(report)
    }

    fn predict_all(&mut self, series: &Series, cfg: &TrainConfig) -> Result<Vec<f64>> {
        let inner = self.inner.as_mut().expect("fit before predict");
        inner.predict_all_inner(series, cfg, &QuadraticFeatures)
    }
}
