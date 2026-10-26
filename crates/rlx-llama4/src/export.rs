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

//! Export a Llama 4 checkpoint's language model as an `.rlxp` package rlx
//! runs alone, through the one exporter ([`rlx_core::export`]): the
//! runner's full-sequence graph (MoE, RoPE and NoPE layers), symbolic in its
//! length and run over the whole context for each token, as the runner
//! runs it. The context stops below `attention_chunk_size`, where chunked
//! attention and temperature tuning start.

use std::path::Path;

use anyhow::Result;
use rlx_core::export::decoder::{Built, Forward, export_forward};
use rlx_core::weight_loader::WeightLoader;

pub use rlx_core::export::{ExportOptions, ExportReport};

use crate::attention::{ROPE_COS, ROPE_SIN};
use crate::config::Llama4TextConfig;
use crate::flow::build_llama4_text_flow;
use crate::rope::build_rope_tables;
use crate::runner::text_weights;

struct Llama4Forward {
    cfg: Llama4TextConfig,
}

impl Forward for Llama4Forward {
    fn family(&self) -> String {
        "llama4".into()
    }
    fn exported_by(&self) -> &str {
        "rlx-llama4"
    }
    fn max_positions(&self) -> usize {
        let chunk = self.cfg.attention_chunk_size.unwrap_or(usize::MAX);
        self.cfg
            .max_position_embeddings
            .min(chunk.saturating_sub(1))
    }
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    fn forward(&self, weights: &mut dyn WeightLoader, seq: usize) -> Result<Built> {
        let mut wm = text_weights(weights)?;
        rlx_core::flow_util::graph_from_built(build_llama4_text_flow(
            &self.cfg, &mut wm, seq, true, false,
        )?)
    }
    fn tables(&self, rows: usize) -> Vec<(String, Vec<f32>, usize)> {
        let half = self.cfg.head_dim() / 2;
        let (cos, sin) = build_rope_tables(self.cfg.head_dim(), self.cfg.rope_theta(), rows);
        vec![(ROPE_COS.into(), cos, half), (ROPE_SIN.into(), sin, half)]
    }
}

/// Export the Llama 4 checkpoint in `dir` (`config.json`, safetensors,
/// `tokenizer.json`) as the package `out`: its language model; the vision
/// tower is not exported.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let cfg = Llama4TextConfig::from_file(dir.join("config.json"))?;
    export_forward(&Llama4Forward { cfg }, dir, out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`])
/// under the `llama4` runner's name.
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("llama4", "text-generation", export_rlxp).claims(&["llama4", "llama4_text"]),
    );
}
