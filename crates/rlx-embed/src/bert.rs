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

//! RLX-compiled BERT encoder for text embeddings.

use std::path::Path;

use anyhow::Result;
use rlx_runtime::{CompiledGraph, Device, Precision, PrecisionPolicy, Session};

use rlx_bert::flow::build_bert_built;
use rlx_core::config::BertConfig;
use rlx_core::flow_bridge::compile_options_from_profile;
use rlx_core::flow_util::{compile_built, graph_from_built};
use rlx_core::weight_map::WeightMap;
use rlx_ir::logical_kernel::KernelDispatchConfig;

/// RLX-compiled BERT model ready for inference.
pub struct RlxBertModel {
    compiled: CompiledGraph,
    config: BertConfig,
    weights_path: String,
    compiled_bs: (usize, usize),
    device: Device,
    precision: Precision,
    policy: Option<PrecisionPolicy>,
}

impl RlxBertModel {
    pub fn load_sized(
        config_path: &Path,
        weights_path: &str,
        batch: usize,
        seq: usize,
    ) -> Result<Self> {
        Self::load_sized_on(config_path, weights_path, batch, seq, Device::Cpu)
    }

    pub fn load_sized_on(
        config_path: &Path,
        weights_path: &str,
        batch: usize,
        seq: usize,
        device: Device,
    ) -> Result<Self> {
        Self::load_sized_with_policy(
            config_path,
            weights_path,
            batch,
            seq,
            device,
            Precision::F32,
            None,
        )
    }

    pub fn load_sized_with_policy(
        config_path: &Path,
        weights_path: &str,
        batch: usize,
        seq: usize,
        device: Device,
        precision: Precision,
        policy: Option<PrecisionPolicy>,
    ) -> Result<Self> {
        let config = BertConfig::from_file(config_path)?;
        let compiled = Self::compile_flow(
            &config,
            weights_path,
            batch,
            seq,
            device,
            precision,
            &policy,
        )?;
        Ok(Self {
            compiled,
            config,
            weights_path: weights_path.to_string(),
            compiled_bs: (batch, seq),
            device,
            precision,
            policy,
        })
    }

    pub fn load(config_path: &Path, weights_path: &str) -> Result<Self> {
        Self::load_sized(config_path, weights_path, 1, 1)
    }

    pub fn recompile(&mut self, batch: usize, seq: usize) -> Result<()> {
        if self.compiled_bs == (batch, seq) {
            return Ok(());
        }
        self.compiled = Self::compile_flow(
            &self.config,
            &self.weights_path,
            batch,
            seq,
            self.device,
            self.precision,
            &self.policy,
        )?;
        self.compiled_bs = (batch, seq);
        Ok(())
    }

    fn compile_flow(
        config: &BertConfig,
        weights_path: &str,
        batch: usize,
        seq: usize,
        device: Device,
        precision: Precision,
        policy: &Option<PrecisionPolicy>,
    ) -> Result<CompiledGraph> {
        let mut wm = WeightMap::from_file(weights_path)?;
        let built = build_bert_built(config, &mut wm, batch, seq)?;
        if device == Device::Cpu && precision == Precision::F32 && policy.is_none() {
            return compile_built(built, device);
        }
        let profile = built.profile().clone();
        let (graph, params) = graph_from_built(built)?;
        let mut opts =
            compile_options_from_profile(&profile, device, KernelDispatchConfig::default());
        opts.precision = precision;
        opts.policy = policy.clone();
        let mut compiled = Session::new(device).compile_with(graph, &opts);
        for (name, data) in params {
            compiled.set_param(&name, &data);
        }
        Ok(compiled)
    }

    pub fn forward(
        &mut self,
        input_ids: &[f32],
        attention_mask: &[f32],
        token_type_ids: &[f32],
        position_ids: &[f32],
    ) -> Vec<f32> {
        let batch = self.compiled_bs.0;
        let seq = self.compiled_bs.1;
        let _ = self.recompile(batch, seq);
        let outputs = self.compiled.run(&[
            ("input_ids", input_ids),
            ("attention_mask", attention_mask),
            ("token_type_ids", token_type_ids),
            ("position_ids", position_ids),
        ]);
        outputs.into_iter().next().unwrap_or_default()
    }

    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
}
