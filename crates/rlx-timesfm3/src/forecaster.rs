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

//! High-level forecasting API (mirrors upstream `TimesFM3Forecaster`).

use crate::config::{MAX_CONTEXT_LENGTH, TimesFM3Config};
use crate::device::resolve_device;
use crate::host::model::{TimesFM3Model, validate_context};
use crate::session::TimesFM3Session;
use anyhow::{Result, bail};
use ndarray::{Array2, Array3, Array4};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

/// Point + optional quantile forecast for one query.
#[derive(Debug, Clone)]
pub struct ForecastOutput {
    pub forecast: Vec<f32>,
    pub quantiles: Option<Vec<f32>>,
}

/// Inference wrapper around [`TimesFM3Session`].
pub struct TimesFM3Forecaster {
    session: TimesFM3Session,
    median_idx: usize,
}

impl TimesFM3Forecaster {
    pub fn from_weights_dir(dir: &Path, device: Device) -> Result<Self> {
        let cfg = TimesFM3Config::from_dir(dir)?;
        let session = TimesFM3Session::open(dir, cfg, device)?;
        Ok(Self::from_session(session))
    }

    pub fn from_session(session: TimesFM3Session) -> Self {
        let median_idx = session.config().median_quantile_index();
        Self {
            session,
            median_idx,
        }
    }

    pub fn from_model(model: TimesFM3Model, device: Device) -> Self {
        Self::from_session(TimesFM3Session::from_model(model, device))
    }

    pub fn config(&self) -> &TimesFM3Config {
        self.session.config()
    }

    pub fn device(&self) -> Device {
        self.session.device()
    }

    #[cfg(feature = "hf-download")]
    pub fn from_pretrained(repo: &str, cache_dir: Option<&Path>, device: Device) -> Result<Self> {
        use anyhow::Context;
        use hf_hub::api::sync::Api;
        let api = Api::new()?;
        let repo = if let Some(c) = cache_dir {
            api.model(repo.to_string()).with_cache_dir(c.to_path_buf())
        } else {
            api.model(repo.to_string())
        };
        let config = repo.get("config.json")?;
        let weights = repo.get("model.safetensors")?;
        let cfg = TimesFM3Config::from_file(&config)?;
        let dir = weights
            .parent()
            .context("safetensors path has no parent")?
            .to_path_buf();
        Self::from_weights_dir(&dir, device)
    }

    /// Forecast a univariate series (`context`) for `horizon` steps.
    pub fn predict(
        &mut self,
        context: &[f32],
        horizon: usize,
        return_quantiles: bool,
    ) -> Result<ForecastOutput> {
        validate_context(context.len())?;
        if horizon == 0 {
            bail!("horizon must be > 0");
        }
        let ctx = Array3::from_shape_vec((1, 1, context.len()), context.to_vec())?;
        let out = self.session.decode(ctx.view(), horizon, None, None, None)?;
        Ok(output_from_array(
            &out,
            0,
            horizon,
            self.median_idx,
            return_quantiles,
        ))
    }

    /// Multivariate forecast: `context` shape `[num_targets, context_len]`.
    pub fn predict_multivariate(
        &mut self,
        context: Array2<f32>,
        horizon: usize,
        past_only_covariates: Option<Array2<f32>>,
        past_future_covariates: Option<Array2<f32>>,
        return_quantiles: bool,
    ) -> Result<ForecastOutput> {
        let ctx_len = context.shape()[1];
        validate_context(ctx_len)?;
        let u = context.shape()[0];
        let target = context.insert_axis(ndarray::Axis(0));
        let po = past_only_covariates.map(|a| a.insert_axis(ndarray::Axis(0)));
        let pf = past_future_covariates.map(|a| a.insert_axis(ndarray::Axis(0)));
        let out = self.session.decode(
            target.view(),
            horizon,
            po.as_ref().map(|x| x.view()),
            pf.as_ref().map(|x| x.view()),
            None,
        )?;
        Ok(pack_variate_forecast(
            &out,
            u,
            horizon,
            self.median_idx,
            return_quantiles,
        ))
    }
}

fn output_from_array(
    out: &Array4<f32>,
    target_idx: usize,
    horizon: usize,
    median_idx: usize,
    return_quantiles: bool,
) -> ForecastOutput {
    let forecast: Vec<f32> = (0..horizon)
        .map(|t| out[[0, target_idx, t, median_idx]])
        .collect();
    let quantiles = if return_quantiles {
        Some(
            (0..horizon)
                .flat_map(|t| (0..out.shape()[3]).map(move |q| out[[0, target_idx, t, q]]))
                .collect(),
        )
    } else {
        None
    };
    ForecastOutput {
        forecast,
        quantiles,
    }
}

fn pack_variate_forecast(
    out: &Array4<f32>,
    u: usize,
    horizon: usize,
    median_idx: usize,
    return_quantiles: bool,
) -> ForecastOutput {
    let mut forecast = Vec::with_capacity(u * horizon);
    for vi in 0..u {
        for t in 0..horizon {
            forecast.push(out[[0, vi, t, median_idx]]);
        }
    }
    let quantiles = if return_quantiles {
        let q = out.shape()[3];
        let mut qs = Vec::with_capacity(u * horizon * q);
        for vi in 0..u {
            for t in 0..horizon {
                for qi in 0..q {
                    qs.push(out[[0, vi, t, qi]]);
                }
            }
        }
        Some(qs)
    } else {
        None
    };
    ForecastOutput {
        forecast,
        quantiles,
    }
}

/// Builder for [`TimesFM3Forecaster`].
pub struct TimesFM3ForecasterBuilder {
    weights: Option<PathBuf>,
    config: Option<TimesFM3Config>,
    device: Option<Device>,
}

impl TimesFM3ForecasterBuilder {
    pub fn new() -> Self {
        Self {
            weights: None,
            config: None,
            device: None,
        }
    }

    pub fn weights(mut self, path: impl Into<PathBuf>) -> Self {
        self.weights = Some(path.into());
        self
    }

    pub fn config(mut self, cfg: TimesFM3Config) -> Self {
        self.config = Some(cfg);
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    pub fn device_name(mut self, name: &str) -> Result<Self> {
        self.device = Some(resolve_device(name)?);
        Ok(self)
    }

    pub fn build(self) -> Result<TimesFM3Forecaster> {
        let weights = self
            .weights
            .ok_or_else(|| anyhow::anyhow!("weights path required"))?;
        let cfg = match self.config {
            Some(c) => c,
            None if weights.is_dir() => TimesFM3Config::from_dir(&weights)?,
            None => bail!("pass --config or a directory containing config.json"),
        };
        let device = self.device.unwrap_or(Device::Cpu);
        let session = TimesFM3Session::open(&weights, cfg, device)?;
        Ok(TimesFM3Forecaster::from_session(session))
    }
}

impl Default for TimesFM3ForecasterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub fn global_context(cfg: &TimesFM3Config) -> usize {
    let p = cfg.input_patch_len;
    MAX_CONTEXT_LENGTH.div_ceil(p) * p
}
