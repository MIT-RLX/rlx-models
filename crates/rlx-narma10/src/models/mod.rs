// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! NARMA-10 predictors: dense ESN, locally connected ESN (LCESN), and polynomial readout.
//!
//! Training protocol and benchmarks live in [`train`]; see the [crate README](../README.md).

mod esn_ridge;
mod local_esn;
mod poly_readout;
mod train;

pub use esn_ridge::EsnRidge;
pub use local_esn::LocalEsn;
pub use poly_readout::PolyReadoutEsn;
pub use train::{
    BenchRow, LCESN_TEST_SAMPLES, LCESN_TIMESTEPS, LCESN_TRAIN_SAMPLES, LCESN_WASHOUT,
    Narma10Predictor, QUICK_TIMESTEPS, QUICK_WASHOUT, TrainConfig, TrainReport, bench_predictors,
    persistence_nrmse,
};
