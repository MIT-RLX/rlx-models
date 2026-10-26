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

//! Mistral **Timestral** — zero-shot probabilistic time-series forecasting.
//!
//! Native port of [`mistralai/Timestral-209M`](https://huggingface.co/mistralai/Timestral-209M)
//! and its 5M / 28M / 123M siblings: a patch transformer alternating time attention
//! (per series, causal, partial RoPE) with variate attention (across the variates and
//! covariates of a series), decoding 21 monotone quantiles block by block.
//!
//! ```no_run
//! use rlx_timestral::TimestralForecaster;
//! use rlx_runtime::Device;
//!
//! let mut fc = TimestralForecaster::load("weights/Timestral-209M".as_ref(), Device::Cpu)?;
//! let context: Vec<f32> = (0..1000).map(|t| (t as f32 * 0.1).sin()).collect();
//! let forecast = fc.forecast_series(&context, 96)?;
//! let median = forecast.median(); // (1, 1, 96)
//! # Ok::<(), anyhow::Error>(())
//! ```

pub mod cli;
pub mod config;
pub mod covariates;
#[cfg(feature = "export")]
pub mod export;
pub mod forecaster;
pub mod graph;
pub mod masks;
pub mod model;
pub mod net;
pub mod scaler;
pub mod train;
pub mod weights;

pub use config::{LayerKind, TimestralConfig};
pub use forecaster::{Forecast, ForecastOptions, TimestralForecaster, resolve_checkpoint};
pub use model::{Role, Rows, TimestralModel};
pub use weights::TimestralWeights;

/// Devices this build can run on: compiled in and present.
pub fn available_devices() -> Vec<rlx_runtime::Device> {
    rlx_core::device_capabilities::available_devices()
}

/// Lower-case device name for reports.
pub fn device_label(device: rlx_runtime::Device) -> String {
    match device {
        rlx_runtime::Device::Gpu => "wgpu".into(),
        d => format!("{d:?}").to_lowercase(),
    }
}
