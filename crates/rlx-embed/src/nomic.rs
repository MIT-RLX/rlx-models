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

//! RLX-compiled NomicBERT encoder for text embeddings.

use std::collections::HashSet;
use std::path::Path;

use anyhow::Result;
use rlx_runtime::{CompileCache, Device, PrecisionPolicy};

use rlx_core::config::NomicBertConfig;
use rlx_core::flow_util::graph_from_built;
use rlx_core::weight_map::WeightMap;
use rlx_nomic::flow::build_nomic_built;

/// RLX-compiled NomicBERT with shape-bucketed compile cache.
pub struct RlxNomicModel {
    cache: CompileCache,
    params_loaded: HashSet<u64>,
    config: NomicBertConfig,
    weights_path: String,
    current_key: u64,
    #[allow(dead_code)]
    device: Device,
    #[allow(dead_code)]
    policy: Option<PrecisionPolicy>,
}

impl RlxNomicModel {
    fn key(batch: usize, seq: usize) -> u64 {
        ((batch as u64) << 32) | (seq as u64)
    }

    pub fn load_sized_on(
        config_path: &Path,
        weights_path: &str,
        batch: usize,
        seq: usize,
        device: Device,
    ) -> Result<Self> {
        Self::load_sized_with_policy(config_path, weights_path, batch, seq, device, None)
    }

    pub fn load_sized_with_policy(
        config_path: &Path,
        weights_path: &str,
        batch: usize,
        seq: usize,
        device: Device,
        policy: Option<PrecisionPolicy>,
    ) -> Result<Self> {
        let config = NomicBertConfig::from_file(config_path)?;
        let mut model = Self {
            cache: CompileCache::with_policy(device, 16, policy.clone()),
            params_loaded: HashSet::new(),
            config,
            weights_path: weights_path.to_string(),
            current_key: Self::key(batch, seq),
            device,
            policy,
        };
        model.recompile(batch, seq)?;
        Ok(model)
    }

    pub fn load_sized(
        config_path: &Path,
        weights_path: &str,
        batch: usize,
        seq: usize,
    ) -> Result<Self> {
        Self::load_sized_on(config_path, weights_path, batch, seq, Device::Cpu)
    }

    pub fn load(config_path: &Path, weights_path: &str) -> Result<Self> {
        Self::load_sized(config_path, weights_path, 1, 1)
    }

    pub fn recompile(&mut self, batch: usize, seq: usize) -> Result<()> {
        let key = Self::key(batch, seq);
        self.current_key = key;
        if self.cache.contains(key) && self.params_loaded.contains(&key) {
            return Ok(());
        }
        let mut wm = WeightMap::from_file(&self.weights_path)?;
        let (graph, params) =
            graph_from_built(build_nomic_built(&self.config, &mut wm, batch, seq)?)?;
        let compiled = self.cache.get_or_compile(key, || graph);
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        self.params_loaded.insert(key);
        Ok(())
    }

    pub fn forward(
        &mut self,
        input_ids: &[f32],
        attention_mask: &[f32],
        token_type_ids: &[f32],
    ) -> Vec<f32> {
        let key = self.current_key;
        let compiled = self.cache.get_or_compile(key, || {
            unreachable!("forward called without prior recompile/load_sized")
        });
        let outputs = compiled.run(&[
            ("input_ids", input_ids),
            ("attention_mask", attention_mask),
            ("token_type_ids", token_type_ids),
        ]);
        outputs.into_iter().next().unwrap_or_default()
    }

    pub fn forward_slots(
        &mut self,
        input_ids: &[f32],
        attention_mask: &[f32],
        token_type_ids: &[f32],
    ) -> (*const f32, usize) {
        let key = self.current_key;
        let compiled = self.cache.get_or_compile(key, || unreachable!());
        let slots = compiled.run_slots(&[input_ids, attention_mask, token_type_ids]);
        if slots.is_empty() {
            return (std::ptr::null(), 0);
        }
        let (off, len) = slots[0];
        unsafe {
            let ptr = compiled.arena_ptr().add(off) as *const f32;
            (ptr, len)
        }
    }

    pub fn forward_pipelined(
        &mut self,
        input_sets: &[(Vec<f32>, Vec<f32>, Vec<f32>)],
    ) -> Vec<Vec<Vec<f32>>> {
        let key = self.current_key;
        let compiled = self.cache.get_or_compile(key, || unreachable!());
        let prepared: Vec<Vec<(&str, &[f32])>> = input_sets
            .iter()
            .map(|(ids, mask, tt)| {
                vec![
                    ("input_ids", ids.as_slice()),
                    ("attention_mask", mask.as_slice()),
                    ("token_type_ids", tt.as_slice()),
                ]
            })
            .collect();
        compiled.run_pipelined(&prepared)
    }

    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
}
