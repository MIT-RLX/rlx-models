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

//! Export a MiniMax M3 checkpoint's language model as an `.rlxp` package rlx
//! runs alone, through the one exporter ([`rlx_core::export`]): the
//! runner's prefill graph (block-sparse attention, MoE), symbolic in its
//! length and run over the whole context for each token, as the runner's
//! `predict_logits` does. Per-expert weights are stacked as the runner
//! stacks them, so the package holds those; the rest are referenced.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use rlx_core::export::decoder::{Built, Forward, export_forward};
use rlx_core::weight_loader::WeightLoader;
use rlx_core::weight_map::WeightMap;

pub use rlx_core::export::{ExportOptions, ExportReport};

use super::config::MiniMaxM3Config;
use super::flow::build_m3_text_flow;
use super::weights::normalize_snapshot;
use super::{ROPE_COS, ROPE_SIN, rope_tables};

struct M3Forward {
    cfg: MiniMaxM3Config,
    max_positions: usize,
}

impl Forward for M3Forward {
    fn family(&self) -> String {
        "minimax-m3".into()
    }
    fn exported_by(&self) -> &str {
        "rlx-minimax"
    }
    fn max_positions(&self) -> usize {
        self.max_positions
    }
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    fn forward(&self, weights: &mut dyn WeightLoader, seq: usize) -> Result<Built> {
        // The text weights as `load_m3_text_snapshot` names and stacks them.
        let mut raw: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        for k in weights.remaining_keys() {
            let v = weights.take(&k)?;
            let name = k.strip_prefix("language_model.").unwrap_or(&k).to_string();
            if !name.starts_with("vision") && !name.starts_with("multi_modal") {
                raw.insert(name, v);
            }
        }
        let mut wm = WeightMap::from_tensors(normalize_snapshot(&self.cfg, raw)?);
        rlx_core::flow_util::graph_from_built(build_m3_text_flow(&self.cfg, &mut wm, seq, true)?)
    }
    fn tables(&self, rows: usize) -> Vec<(String, Vec<f32>, usize)> {
        let (cos, sin) = rope_tables(rows, self.cfg.n_rot(), self.cfg.rope_theta);
        let half = self.cfg.n_rot() / 2;
        vec![(ROPE_COS.into(), cos, half), (ROPE_SIN.into(), sin, half)]
    }
}

/// Export the MiniMax M3 checkpoint in `dir` (`config.json`, safetensors,
/// `tokenizer.json`) as the package `out`: its language model; the vision
/// tower is not exported.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let config_path = dir.join("config.json");
    let cfg = MiniMaxM3Config::from_hf_config_json(&config_path)?;
    let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;
    let text = raw.get("text_config").unwrap_or(&raw);
    let max_positions = text
        .get("max_position_embeddings")
        .and_then(serde_json::Value::as_u64)
        .map_or(usize::MAX, |n| n as usize);
    export_forward(&M3Forward { cfg, max_positions }, dir, out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`])
/// under the `minimax-m3` runner's name.
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("minimax-m3", "text-generation", export_rlxp)
            .claims(&["minimax_m3", "minimax_m3_vl"]),
    );
}
