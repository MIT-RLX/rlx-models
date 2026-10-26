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

//! What runs the network: [`build_graph`] compiled per shape on one RLX
//! device. Everything around it ([`crate::model`], [`crate::forecaster`])
//! is host math with no device of its own, so the exported package's
//! program builds those files with its own `Net`, calling the package's
//! graphs.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use rlx_core::flow_util::{built_from_hir, compile_cache_ensure_built_with_options};
use rlx_runtime::Device;
use rlx_runtime::compile_cache::CompileCache;

use crate::config::TimestralConfig;
use crate::forecaster::{TimestralForecaster, resolve_checkpoint};
use crate::graph::{build_graph, graph_params};
use crate::model::{GraphInputs, TimestralModel};
use crate::weights::TimestralWeights;

/// Compiled shapes kept alive. Each holds its own copy of the weights.
const GRAPH_CACHE_CAPACITY: usize = 3;

/// The compiled graphs of one model on one device.
pub struct Net {
    device: Device,
    params: Arc<HashMap<String, Vec<f32>>>,
    cache: CompileCache,
    number_format: Option<rlx_runtime::NumberFormat>,
}

impl Net {
    /// Run the graph for `block.dims` (compiled on first use): its outputs,
    /// the head first.
    pub fn run(&mut self, cfg: &TimestralConfig, block: GraphInputs) -> Result<Vec<Vec<f32>>> {
        let GraphInputs { dims, inputs } = block;
        let key = dims.key();
        if !self.cache.contains(key) {
            let hir = build_graph(cfg, dims);
            let built = built_from_hir(hir, (*self.params).clone())?;
            let mut opts = rlx_runtime::CompileOptions::new();
            if let Some(f) = self.number_format {
                opts.number_format = Some(f);
            }
            compile_cache_ensure_built_with_options(&mut self.cache, key, built, &opts)?;
        }
        let graph = self
            .cache
            .get_or_compile(key, || unreachable!("graph compiled above"));
        let inputs: Vec<(&str, &[f32])> = inputs
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        let outs = graph.run(&inputs);
        outs.first().context("graph produced no output")?;
        Ok(outs)
    }
}

impl TimestralModel {
    /// Load `config.json` and `model.safetensors` from a checkpoint directory.
    pub fn load(dir: &Path, device: Device) -> Result<Self> {
        let cfg = TimestralConfig::from_file(&dir.join("config.json"))?;
        let weights = TimestralWeights::load(dir, &cfg)?;
        Ok(Self::new(cfg, &weights, device))
    }

    pub fn new(cfg: TimestralConfig, weights: &TimestralWeights, device: Device) -> Self {
        let params = Arc::new(graph_params(&cfg, weights));
        Self {
            forecast_query: weights.forecast_query.to_vec(),
            cfg,
            net: Net {
                device,
                params,
                cache: CompileCache::new(device, GRAPH_CACHE_CAPACITY),
                number_format: None,
            },
            bucketing: true,
            capture_hidden: false,
            last_hidden: None,
        }
    }

    /// Seeded random weights for tests that need no download.
    pub fn synth(cfg: TimestralConfig, seed: u64, device: Device) -> Self {
        let w = TimestralWeights::synth(&cfg, seed);
        Self::new(cfg, &w, device)
    }

    pub fn device(&self) -> Device {
        self.net.device
    }

    /// Run the network with every weight and intermediate value in `format`
    /// (`None`: the process setting, `RLX_NUMBER_FORMAT`, default `f32`).
    /// Drops compiled graphs, which were built for the previous format.
    pub fn set_number_format(&mut self, format: Option<rlx_runtime::NumberFormat>) -> Result<()> {
        // f64 runs natively on the CPU and in double-single on GPUs; f64x2 is
        // double-single everywhere.
        self.net.number_format = format;
        self.net.cache = CompileCache::new(self.net.device, GRAPH_CACHE_CAPACITY);
        Ok(())
    }

    pub fn number_format(&self) -> Option<rlx_runtime::NumberFormat> {
        self.net.number_format
    }
}

impl TimestralForecaster {
    /// Load a checkpoint directory (`config.json` + `model.safetensors`).
    pub fn load(dir: &Path, device: Device) -> Result<Self> {
        Ok(Self::new(TimestralModel::load(dir, device)?))
    }

    /// Resolve a local directory or download a Hugging Face repo id
    /// (e.g. `mistralai/Timestral-209M`; needs the `hf-download` feature).
    pub fn from_pretrained(path_or_repo: &str, device: Device) -> Result<Self> {
        let dir = resolve_checkpoint(path_or_repo)?;
        Self::load(&dir, device)
    }

    pub fn device(&self) -> Device {
        self.model().device()
    }
}
