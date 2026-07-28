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

//! Llama-4 text runner. v1: full-sequence (no KV-cache) decode — the graph is
//! compiled once at `max_len`, each step re-runs the padded sequence and reads
//! logits at the true last position (causal ⇒ pad is harmless). Correct, O(L²).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rlx_core::flow_util::compile_built;
use rlx_core::weight_loader::WeightLoader;
use rlx_core::weight_map::WeightMap;
use rlx_runtime::{CompiledGraph, Device};

use crate::config::Llama4TextConfig;
use crate::flow::build_llama4_text_flow;
use crate::rope::build_rope_tables;

pub struct Llama4Runner {
    cfg: Llama4TextConfig,
    device: Device,
    weights: Option<WeightMap>, // model.* + lm_head.*; consumed on first build
    graph: Option<(usize, CompiledGraph)>, // (max_len, compiled)
    rope: HashMap<usize, (Vec<f32>, Vec<f32>)>, // by max_len
}

impl Llama4Runner {
    /// Load a Llama-4 checkpoint directory (text weights; vision ignored).
    pub fn from_checkpoint(dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        let dir = dir.as_ref();
        let cfg = Llama4TextConfig::from_file(dir.join("config.json"))?;
        let mut full = WeightMap::from_safetensors_dir(dir)
            .with_context(|| format!("loading llama4 safetensors from {}", dir.display()))?;

        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        for k in full.remaining_keys() {
            if k.starts_with("vision_model.") || k.starts_with("multi_modal_projector.") {
                let _ = full.take(&k)?; // drop vision weights (text runner)
                continue;
            }
            let (d, s) = full.take(&k)?;
            if let Some(r) = k.strip_prefix("language_model.model.") {
                t.insert(format!("model.{r}"), (d, s));
            } else if let Some(r) = k.strip_prefix("language_model.lm_head.") {
                t.insert(format!("lm_head.{r}"), (d, s));
            } else {
                t.insert(k, (d, s)); // model.* / lm_head.weight (text-only checkpoints)
            }
        }
        Ok(Self {
            cfg,
            device,
            weights: Some(WeightMap::from_tensors(t)),
            graph: None,
            rope: HashMap::new(),
        })
    }

    pub fn config(&self) -> &Llama4TextConfig {
        &self.cfg
    }

    fn ensure_graph(&mut self, max_len: usize) -> Result<()> {
        if self.graph.as_ref().map(|(l, _)| *l) == Some(max_len) {
            return Ok(());
        }
        let mut wm = self.weights.take().ok_or_else(|| {
            anyhow!("llama4 text weights already consumed (one max_len per session in v1)")
        })?;
        let built = build_llama4_text_flow(&self.cfg, &mut wm, max_len, true, false)?;
        let compiled = compile_built(built, self.device)?;
        self.graph = Some((max_len, compiled));
        self.rope.entry(max_len).or_insert_with(|| {
            build_rope_tables(self.cfg.head_dim(), self.cfg.rope_theta(), max_len)
        });
        Ok(())
    }

    /// Greedy generation from a tokenized prompt. Returns the new token ids.
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        max_new: usize,
        eos: Option<u32>,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        let max_len = prompt_ids.len() + max_new;
        self.ensure_graph(max_len)?;
        let vocab = self.cfg.vocab_size;
        let (cos, sin) = self.rope.get(&max_len).unwrap();
        let compiled = &mut self.graph.as_mut().unwrap().1;

        let mut ids = prompt_ids.to_vec();
        let mut generated = Vec::new();
        for _ in 0..max_new {
            let real = ids.len();
            if real > max_len {
                break;
            }
            let mut input = vec![0f32; max_len];
            for (i, &t) in ids.iter().enumerate() {
                input[i] = t as f32;
            }
            let out = compiled
                .run(&[
                    ("input_ids", input.as_slice()),
                    ("rope_cos", cos.as_slice()),
                    ("rope_sin", sin.as_slice()),
                ])
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("llama4 forward returned no output"))?;
            let row = &out[(real - 1) * vocab..real * vocab];
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (v, &x) in row.iter().enumerate() {
                if x > best_v {
                    best_v = x;
                    best = v;
                }
            }
            let next = best as u32;
            generated.push(next);
            if !on_token(next) || Some(next) == eos || ids.len() + 1 >= max_len {
                break;
            }
            ids.push(next);
        }
        Ok(generated)
    }
}
