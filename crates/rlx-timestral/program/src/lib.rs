// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! The forecaster's host logic as an RLX program, answering `run`: what
//! `TimestralForecaster::forecast` does around the network, from the crate's
//! own files — the packed rows' scaling, masks and positions
//! (`model.rs`), the block-by-block decode, covariates, symmetric averaging
//! and clipping (`forecaster.rs`) — with each block's graph run by the
//! package ([`net::Net`]).
//!
//! In: `context`, f32 `[B, V, T]` (NaN: missing); optionally
//! `observation_mask` (u8 or bool, like the context), `past_covariates`
//! `[B, C, T]`, `known_covariates` `[B, C, T + H]` and their
//! `categorical_…` codes. Options: `horizon` (`H`, required),
//! `symmetric_averaging` and `clip_nonnegative` (both on by default).
//! Out: `quantiles`, f32 `[B, V, H, Q]`, and `levels`, f32 `[Q]`.

#[allow(dead_code)]
#[path = "../../src/config.rs"]
mod config;
#[allow(dead_code)]
#[path = "../../src/covariates.rs"]
mod covariates;
#[allow(dead_code)]
#[path = "../../src/forecaster.rs"]
mod forecaster;
#[allow(dead_code)]
#[path = "../../src/masks.rs"]
mod masks;
#[allow(dead_code)]
#[path = "../../src/model.rs"]
mod model;
#[allow(dead_code)]
#[path = "../../src/scaler.rs"]
mod scaler;

mod net {
    //! The package's graphs in place of the crate's compiled ones.

    use anyhow::{Result, anyhow};
    use rlx_program::{Tensor, run};

    use crate::config::TimestralConfig;
    use crate::model::GraphInputs;

    pub struct Net {
        /// Graphs symbolic in rows and context patches, one per decoder
        /// length; and the covariate window's, as many decoder tokens as
        /// context patches.
        pub blocks: Vec<(usize, String)>,
        pub window: String,
    }

    impl Net {
        pub fn run(&mut self, cfg: &TimestralConfig, block: GraphInputs) -> Result<Vec<Vec<f32>>> {
            let GraphInputs { dims, inputs } = block;
            let (r, l, n) = (dims.r, dims.l, dims.n_dec);
            let s = l + n;
            let function = match self.blocks.iter().find(|(d, _)| *d == n) {
                Some((_, f)) => f,
                None if l == n => &self.window,
                None => return Err(anyhow!("the package has no graph for {n} decoder tokens")),
            };
            let shape = |name: &str| -> Vec<usize> {
                match name {
                    "patches" => vec![r * l, 2 * cfg.patch_size],
                    "dec" => vec![r * n, cfg.d_model],
                    "rope_cos" | "rope_sin" => vec![r * s, cfg.head_dim() / 2],
                    "time_bias" => vec![r, s, s],
                    _ => vec![s, r, r],
                }
            };
            let tensors = inputs
                .iter()
                .map(|(name, v)| Tensor::f32(name.as_str(), shape(name), v))
                .collect();
            let outs = run(function, tensors).map_err(|e| anyhow!(e))?;
            Ok(outs.iter().map(Tensor::to_f32).collect())
        }
    }
}

use forecaster::{ForecastOptions, TimestralForecaster};
use model::TimestralModel;
use ndarray::Array3;
use rlx_program::{Message, Program, Tensor};
use serde_json::Value;

struct Forecaster(TimestralForecaster);

fn array(t: &Tensor, what: &str) -> Result<Array3<f32>, String> {
    if t.shape.len() != 3 {
        return Err(format!("`{what}` is [B, ·, T], not {:?}", t.shape));
    }
    Array3::from_shape_vec((t.shape[0], t.shape[1], t.shape[2]), t.to_f32())
        .map_err(|e| format!("`{what}`: {e}"))
}

impl Program for Forecaster {
    fn init(config: Value) -> Result<Self, String> {
        let cfg: config::TimestralConfig = serde_json::from_value(config["model"].clone())
            .map_err(|e| format!("the model config: {e}"))?;
        cfg.validate().map_err(|e| e.to_string())?;
        let forecast_query: Vec<f32> = serde_json::from_value(config["forecast_query"].clone())
            .map_err(|e| format!("forecast_query: {e}"))?;
        let blocks = config["blocks"]
            .as_array()
            .ok_or("config has no blocks")?
            .iter()
            .filter_map(|b| {
                Some((
                    b["decoder"].as_u64()? as usize,
                    b["function"].as_str()?.to_string(),
                ))
            })
            .collect();
        let window = config["window"]
            .as_str()
            .ok_or("config has no window")?
            .to_string();
        let model = TimestralModel {
            cfg,
            forecast_query,
            net: net::Net { blocks, window },
            bucketing: true,
            capture_hidden: false,
            last_hidden: None,
        };
        Ok(Forecaster(TimestralForecaster::new(model)))
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "run" {
            return Err(format!("the forecaster answers run, not {task}"));
        }
        let context = array(
            req.tensor("context")
                .ok_or("run takes a `context` tensor, f32 [B, V, T]")?,
            "context",
        )?;
        let o = &req.body["options"];
        let horizon = o["horizon"]
            .as_u64()
            .ok_or("run takes the `horizon` option: the steps to forecast")?
            as usize;
        let optional = |name: &str| req.tensor(name).map(|t| array(t, name)).transpose();
        let defaults = ForecastOptions::default();
        let opts = ForecastOptions {
            observation_mask: optional("observation_mask")?.map(|m| m.mapv(|v| v != 0.0)),
            past_covariates: optional("past_covariates")?,
            known_covariates: optional("known_covariates")?,
            categorical_past_covariates: optional("categorical_past_covariates")?,
            categorical_known_covariates: optional("categorical_known_covariates")?,
            symmetric_averaging: o["symmetric_averaging"]
                .as_bool()
                .unwrap_or(defaults.symmetric_averaging),
            clip_nonnegative: o["clip_nonnegative"]
                .as_bool()
                .unwrap_or(defaults.clip_nonnegative),
        };
        let f = self
            .0
            .forecast(context.view(), horizon, &opts)
            .map_err(|e| format!("{e:#}"))?;
        let shape = f.quantiles.shape().to_vec();
        let q: Vec<f32> = f.quantiles.iter().copied().collect();
        let n = f.quantile_levels.len();
        Ok(Message::tensors(vec![
            Tensor::f32("quantiles", shape, &q),
            Tensor::f32("levels", vec![n], &f.quantile_levels),
        ]))
    }
}

rlx_program::program!(Forecaster);
