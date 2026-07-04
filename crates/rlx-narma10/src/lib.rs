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

//! Canonical **NARMA-10** (order-10 nonlinear autoregressive moving average) reference.
//!
//! See the [crate README](README.md) for benchmark protocols, predictor tables, and commands.
//!
//! Standard reservoir-computing benchmark recurrence (Atiya & Parlos, 2000;
//! Schrauwen et al., 2008):
//!
//! ```text
//! y[t+1] = α·y[t] + β·y[t]·Σ_{i=0}^{9} y[t−i] + γ·u[t−9]·u[t] + δ
//! ```
//!
//! with `u[t] ~ Uniform(0, 0.5)`, `(α, β, γ, δ) = (0.3, 0.05, 1.5, 0.1)`, and
//! `y[t] = 0` for `t < 0`.
//!
//! # Layers
//!
//! - [`host`] — CPU reference recurrence, [`Series`] generation, [`nrmse`]
//! - [`rlx`] — same recurrence via compiled RLX graphs on any backend
//! - [`models`] — [`EsnRidge`], [`LocalEsn`], [`PolyReadoutEsn`] + [`bench_predictors`]
//! - [`reservoir`] — fixed random ESN dynamics (dense and locally connected)
//! - [`ridge`] — ridge regression readout solver
//!
//! # Protocols
//!
//! - [`TrainConfig::lcesn`] — washout 1000, 12k train, ~1k test at [`LCESN_TIMESTEPS`]
//! - [`TrainConfig::quick`] — 5k steps, washout 100 (fast dense-ESN check)

pub mod host;
pub mod models;
pub mod reservoir;
pub mod ridge;
pub mod rlx;

pub use host::{
    Coefficients, ORDER, Rng, Series, generate, generate_with_coeff, nrmse,
    scale_inputs_for_reservoir, step,
};
pub use models::{
    BenchRow, EsnRidge, LocalEsn, Narma10Predictor, PolyReadoutEsn, TrainConfig, TrainReport,
    bench_predictors, persistence_nrmse, LCESN_TEST_SAMPLES, LCESN_TIMESTEPS, LCESN_TRAIN_SAMPLES,
    LCESN_WASHOUT, QUICK_TIMESTEPS, QUICK_WASHOUT,
};
pub use reservoir::{Reservoir, ReservoirConfig};
pub use rlx::{
    BACKEND_DEVICES, SeriesRunner, build_series_graph, generate_on_device,
    generate_on_device_with_coeff, max_abs_diff,
};
