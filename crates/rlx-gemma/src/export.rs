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

//! Export a Hugging Face Gemma checkpoint as an `.rlxp` package rlx runs
//! alone, through rlx-qwen3's decoder exporter: the step (symbolic in its
//! cache and batch, its state inside), a prompt function, both RoPE tables
//! (sliding and full layers'), the tokenizer and chat template.
//!
//! Sliding-window layers mask with a second input, `mask_local`, which the
//! stateful step computes from its position (`StatefulDecode::windows`), so
//! the context runs past `sliding_window` exactly.

use std::path::Path;

use anyhow::{Result, bail};
use rlx_core::export::decoder::{Built, Decoder, export_decoder};
pub use rlx_core::export::{ExportOptions, ExportReport};
use rlx_core::weight_loader::WeightLoader;

use crate::config::GemmaConfig;
use crate::rope::{resolve_global_inv_freq, resolve_inv_freq, rope_slice};

struct GemmaDecoder {
    cfg: GemmaConfig,
    family: String,
    inv_freq: Vec<f64>,
    global_inv_freq: Option<Vec<f64>>,
}

impl Decoder for GemmaDecoder {
    fn family(&self) -> String {
        self.family.clone()
    }
    fn exported_by(&self) -> &str {
        "rlx-gemma"
    }
    fn max_positions(&self) -> usize {
        self.cfg.max_position_embeddings
    }
    fn layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    fn rope_inputs(&self) -> Vec<(String, String, usize)> {
        let mut t = vec![("rope_cos".into(), "rope_sin".into(), self.inv_freq.len())];
        if let Some(g) = &self.global_inv_freq {
            t.push(("rope_cos_global".into(), "rope_sin_global".into(), g.len()));
        }
        t
    }
    fn rope_row(&self, pos: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
        let mut r = vec![rope_slice(&self.inv_freq, pos)];
        if let Some(g) = &self.global_inv_freq {
            r.push(rope_slice(g, pos));
        }
        r
    }
    fn step(&self, weights: &mut dyn WeightLoader, batch: usize, upper: usize) -> Result<Built> {
        let flow = crate::flow::GemmaFlow::for_decode(&self.cfg, batch, upper)
            .lm_head()
            .custom_mask()
            .windowed_mask();
        rlx_core::flow_util::graph_from_built(flow.build(weights)?)
    }
    fn windows(&self) -> Vec<(String, usize)> {
        let sliding = (0..self.cfg.num_hidden_layers).any(|i| !self.cfg.is_full_attention_layer(i));
        match self.cfg.sliding_window {
            Some(w) if sliding => vec![("mask_local".into(), w)],
            _ => Vec::new(),
        }
    }
    fn prompt(&self, weights: &mut dyn WeightLoader, batch: usize, seq: usize) -> Result<Built> {
        crate::builder::build_gemma_graph_sized_last_logits(&self.cfg, weights, batch, seq, true)
    }
    fn prompt_rope_params(&self) -> Vec<(String, String)> {
        vec![
            ("rope.cos".into(), "rope.sin".into()),
            ("rope.global.cos".into(), "rope.global.sin".into()),
        ]
    }
}

/// Export the Gemma checkpoint in `dir` (`config.json`,
/// `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, optional
/// `chat_template.jinja`, `generation_config.json` and `README.md`) as the
/// package `out`. Text-only Gemma 2 / 3 checkpoints.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let cfg = GemmaConfig::from_file(&dir.join("config.json"))?;
    let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let family = raw
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("gemma")
        .to_string();
    if cfg.is_moe() {
        bail!("{family}: mixture-of-experts Gemma is not exported");
    }
    if raw.get("rope_scaling").is_some_and(|v| !v.is_null()) {
        bail!("rope_scaling {}: not supported", raw["rope_scaling"]);
    }
    let inv_freq = resolve_inv_freq(&cfg, None);
    let global_inv_freq = resolve_global_inv_freq(&cfg, None);
    let model = GemmaDecoder {
        cfg,
        family,
        inv_freq,
        global_inv_freq,
    };
    export_decoder(&model, dir, out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`])
/// under the `gemma` runner's name.
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("gemma", "text-generation", export_rlxp).claims(&[
            "gemma3",
            "gemma3_text",
            "gemma2",
            "gemma",
        ]),
    );
}
