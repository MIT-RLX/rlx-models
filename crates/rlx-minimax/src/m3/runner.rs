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

//! MiniMax-M3 text runner — a correctness-first **prefill** LM.
//!
//! [`MiniMaxM3Runner`] loads the normalized weight snapshot, builds the text
//! prefill graph for a given prompt length (caching one compiled graph per
//! length), runs it, and returns the last token's logits. Implementing
//! [`LmRunner::predict_logits`] makes the trait's default re-prefill
//! `generate` loop work directly.
//!
//! This is deliberately not KV-cached: incremental decode reprocesses the whole
//! context each step (`O(seq²)` and one compiled graph per distinct length).
//! It is correct and sufficient to validate the graph end-to-end; a KV-cache
//! decode path (per the plan) is the perf follow-up. It is also the only path
//! that fits in bounded RAM for the real 428B checkpoint (which cannot run
//! locally regardless).

use anyhow::{Result, anyhow};
use rlx_cli::LmRunner;
use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;
use std::path::Path;

use super::config::MiniMaxM3Config;
use super::decode::{DecodeOutputLayout, build_m3_decode_flow};
use super::flow::build_m3_text_flow;
use super::weights::{Snapshot, load_m3_text_snapshot};
use super::{rope_row, rope_tables};

pub struct MiniMaxM3Runner {
    cfg: MiniMaxM3Config,
    snapshot: Snapshot,
    device: Device,
    cache: HashMap<usize, CompiledGraph>,
    // KV-cache decode state.
    decode_cache: HashMap<usize, CompiledGraph>,
    decode_layout: Option<DecodeOutputLayout>,
    k_buf: Vec<Vec<f32>>,
    v_buf: Vec<Vec<f32>>,
    idxk_buf: Vec<Vec<f32>>,
    past_len: usize,
}

impl MiniMaxM3Runner {
    /// Build from an already-normalized f32 weight snapshot (used by tests).
    pub fn from_snapshot(cfg: MiniMaxM3Config, snapshot: Snapshot, device: Device) -> Self {
        let layers = cfg.num_hidden_layers;
        Self {
            cfg,
            snapshot,
            device,
            cache: HashMap::new(),
            decode_cache: HashMap::new(),
            decode_layout: None,
            k_buf: vec![Vec::new(); layers],
            v_buf: vec![Vec::new(); layers],
            idxk_buf: vec![Vec::new(); layers],
            past_len: 0,
        }
    }

    /// Load config + weights from disk. `config_path` may be `None` to derive
    /// the config from a HF `config.json` sitting next to `weights_path`.
    pub fn from_pretrained(
        weights_path: &str,
        config_path: Option<&Path>,
        device: Device,
    ) -> Result<Self> {
        let cfg = match config_path {
            Some(p) => MiniMaxM3Config::from_hf_config_json(p)?,
            None => {
                let sib = Path::new(weights_path)
                    .parent()
                    .map(|d| d.join("config.json"))
                    .ok_or_else(|| anyhow!("cannot locate config.json next to {weights_path}"))?;
                MiniMaxM3Config::from_hf_config_json(&sib)?
            }
        };
        let snapshot = load_m3_text_snapshot(&cfg, weights_path)?;
        Ok(Self::from_snapshot(cfg, snapshot, device))
    }

    pub fn config(&self) -> &MiniMaxM3Config {
        &self.cfg
    }

    /// Ensure a compiled prefill graph exists for `seq`.
    fn ensure_compiled(&mut self, seq: usize) -> Result<()> {
        if self.cache.contains_key(&seq) {
            return Ok(());
        }
        let mut wm = WeightMap::from_tensors(self.snapshot.clone());
        let built = build_m3_text_flow(&self.cfg, &mut wm, seq, true)?;
        let compiled = compile_built(built, self.device)?;
        self.cache.insert(seq, compiled);
        Ok(())
    }

    /// Run a prefill over `ids` and return the last token's logits (`[vocab]`).
    pub fn forward_last_logits(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        if ids.is_empty() {
            return Err(anyhow!("empty prompt"));
        }
        let seq = ids.len();
        let vocab = self.cfg.vocab_size;
        let n_rot = self.cfg.n_rot();
        let theta = self.cfg.rope_theta;
        self.ensure_compiled(seq)?;
        let (cos, sin) = rope_tables(seq, n_rot, theta);
        let idf: Vec<f32> = ids.iter().map(|&t| t as f32).collect();
        let compiled = self.cache.get_mut(&seq).expect("compiled graph present");
        let mut out = compiled.run(&[
            ("input_ids", idf.as_slice()),
            ("rope_cos", cos.as_slice()),
            ("rope_sin", sin.as_slice()),
        ]);
        let logits = out
            .drain(..)
            .next()
            .ok_or_else(|| anyhow!("m3 forward returned no output"))?;
        if logits.len() != seq * vocab {
            return Err(anyhow!(
                "m3 logits len {} != seq*vocab {}",
                logits.len(),
                seq * vocab
            ));
        }
        Ok(logits[(seq - 1) * vocab..seq * vocab].to_vec())
    }

    // ── KV-cache incremental decode ─────────────────────────────────────────

    /// Clear the KV cache and reset the decode position.
    pub fn decode_reset(&mut self) {
        self.past_len = 0;
        for b in self.k_buf.iter_mut() {
            b.clear();
        }
        for b in self.v_buf.iter_mut() {
            b.clear();
        }
        for b in self.idxk_buf.iter_mut() {
            b.clear();
        }
    }

    fn ensure_decode_compiled(&mut self, past_len: usize) -> Result<()> {
        if self.decode_cache.contains_key(&past_len) {
            return Ok(());
        }
        let mut wm = WeightMap::from_tensors(self.snapshot.clone());
        let (built, layout) = build_m3_decode_flow(&self.cfg, &mut wm, past_len)?;
        self.decode_layout = Some(layout);
        self.decode_cache
            .insert(past_len, compile_built(built, self.device)?);
        Ok(())
    }

    /// Decode one token given the current cache; append fresh K/V/idx_k rows to
    /// the cache and return the token's logits (`[vocab]`).
    pub fn decode_step(&mut self, token: u32) -> Result<Vec<f32>> {
        let pl = self.past_len;
        let n_rot = self.cfg.n_rot();
        let theta = self.cfg.rope_theta;
        let vocab = self.cfg.vocab_size;
        let layers = self.cfg.num_hidden_layers;
        self.ensure_decode_compiled(pl)?;
        let layout = self.decode_layout.clone().expect("decode layout");

        let (cos, sin) = rope_row(pl, n_rot, theta);
        let tid = [token as f32];
        // Input names must outlive the borrow held by `run`.
        let names: Vec<(String, String, Option<String>)> = (0..layers)
            .map(|i| {
                (
                    format!("past_k_{i}"),
                    format!("past_v_{i}"),
                    self.cfg
                        .is_sparse_layer(i)
                        .then(|| format!("past_idxk_{i}")),
                )
            })
            .collect();

        let outs = {
            let mut inputs: Vec<(&str, &[f32])> = Vec::with_capacity(3 + layers * 3);
            inputs.push(("input_ids", &tid));
            inputs.push(("rope_cos", &cos));
            inputs.push(("rope_sin", &sin));
            for i in 0..layers {
                inputs.push((names[i].0.as_str(), self.k_buf[i].as_slice()));
                inputs.push((names[i].1.as_str(), self.v_buf[i].as_slice()));
                if let Some(n) = &names[i].2 {
                    inputs.push((n.as_str(), self.idxk_buf[i].as_slice()));
                }
            }
            let compiled = self.decode_cache.get_mut(&pl).expect("decode graph");
            compiled.run(&inputs)
        };

        // Append the fresh rows to the cache.
        for (i, &(ki, vi, idxki)) in layout.iter().enumerate() {
            self.k_buf[i].extend_from_slice(&outs[ki]);
            self.v_buf[i].extend_from_slice(&outs[vi]);
            if let Some(x) = idxki {
                self.idxk_buf[i].extend_from_slice(&outs[x]);
            }
        }
        self.past_len += 1;

        let logits = outs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("decode returned no logits"))?;
        if logits.len() != vocab {
            return Err(anyhow!(
                "decode logits len {} != vocab {}",
                logits.len(),
                vocab
            ));
        }
        Ok(logits)
    }

    /// Reset the cache, decode the whole prompt token-by-token, and return the
    /// last token's logits — equivalent to `forward_last_logits` (prefill).
    pub fn decode_prefill(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        if ids.is_empty() {
            return Err(anyhow!("empty prompt"));
        }
        self.decode_reset();
        let mut last = Vec::new();
        for &t in ids {
            last = self.decode_step(t)?;
        }
        Ok(last)
    }

    /// KV-cached greedy generation: decode the prompt, then extend.
    pub fn decode_generate(
        &mut self,
        prompt: &[u32],
        n_new: usize,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        let mut last = self.decode_prefill(prompt)?;
        let mut out = Vec::with_capacity(n_new);
        for _ in 0..n_new {
            let next = argmax(&last);
            out.push(next);
            if !on_token(next) {
                break;
            }
            last = self.decode_step(next)?;
        }
        Ok(out)
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best as u32
}

impl LmRunner for MiniMaxM3Runner {
    fn family(&self) -> &'static str {
        "minimax-m3"
    }
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        self.forward_last_logits(prompt_ids)
    }
}
