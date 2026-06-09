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

//! Compiled acoustic velocity stack (3-token FM sequence, bidirectional attention).

use crate::acoustic::FM_SEQ;
use crate::acoustic_flow::build_acoustic_velocity_built;
use crate::config::AcousticTransformerArgs;
use crate::load::PREFIX_ACOUSTIC;
use crate::load::VoxtralTtsWeightStore;
use crate::weights::SnapshotLoader;
use anyhow::{Context, Result};
use rlx_core::flow_bridge::compile_options_from_profile;
use rlx_core::weight_map::WeightMap;
use rlx_ir::logical_kernel::KernelDispatchConfig;
use rlx_runtime::{CompiledGraph, Device, Session};
use std::collections::HashMap;

const STACK_BATCH: usize = 1;

pub struct CompiledAcousticStack {
    graph: CompiledGraph,
    token_dim: usize,
}

impl CompiledAcousticStack {
    pub fn open(
        store: &VoxtralTtsWeightStore,
        args: &AcousticTransformerArgs,
        device: Device,
    ) -> Result<Self> {
        let mut wm = store.load_acoustic()?;
        let keys: Vec<String> = wm.keys().map(str::to_string).collect();
        let mut snapshot = HashMap::with_capacity(keys.len());
        for key in keys {
            let short = key
                .strip_prefix(PREFIX_ACOUSTIC)
                .unwrap_or(key.as_str())
                .to_string();
            snapshot.insert(short, wm.take(&key)?);
        }
        let mut loader = SnapshotLoader::new(snapshot);
        let mut wm_build = WeightMap::from_weight_loader(&mut loader)?;
        let built = build_acoustic_velocity_built(args, &mut wm_build, STACK_BATCH, FM_SEQ)
            .context("build acoustic velocity graph")?;
        let profile = built.profile().clone();
        let params = built.params().clone();
        let opts = compile_options_from_profile(&profile, device, KernelDispatchConfig::default());
        let (graph, _) = built.into_graph_parts()?;
        let mut graph = Session::new(device).compile_with(graph, &opts);
        for (name, data) in &params {
            graph.set_param(name, data);
        }
        Ok(Self {
            graph,
            token_dim: args.dim,
        })
    }

    pub fn forward(&mut self, tokens: &[f32]) -> Result<Vec<f32>> {
        Ok(self.graph.run(&[("hidden", tokens)])[0].clone())
    }

    pub fn input_len(&self) -> usize {
        FM_SEQ * self.token_dim
    }
}
