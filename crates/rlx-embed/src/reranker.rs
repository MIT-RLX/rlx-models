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

//! Cross-encoder **reranker** — jointly scores a `(query, passage)` pair.
//!
//! The bi-encoders elsewhere in this crate embed the query and the passage
//! independently and compare the two vectors: fast (doc vectors cache, one
//! index scan), but blind to fine-grained query↔passage interactions, so the
//! true answer is often near — not at — rank 1. A cross-encoder instead runs
//! BERT over the concatenation `[CLS] query [SEP] passage [SEP]` and reads a
//! single relevance logit off a classification head. It sees every query-token
//! ↔ passage-token interaction, so it separates a *dog* question from a *cat*
//! fact that a bi-encoder lumps under "pet". The cost is one forward per pair
//! (no cacheable vectors), so it's used to **re-order a bi-encoder's top-N**,
//! not to scan the whole store.
//!
//! Loads a `BertForSequenceClassification` checkpoint (e.g.
//! `cross-encoder/ms-marco-MiniLM-L-6-v2`, `num_labels = 1`): the encoder runs
//! through [`RlxBertModel`] (which auto-detects the `bert.` weight prefix), and
//! the pooler (`dense` + `tanh` on `[CLS]`) and the single-logit classifier head
//! are applied here in host code from the checkpoint's own weights.

use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::weight_map::WeightMap;
use rlx_runtime::Device;

use crate::bert::RlxBertModel;
use crate::tokenizer::BertTokenizer;

/// A loaded cross-encoder reranker. Call [`rerank`](Self::rerank) to re-order a
/// candidate list by joint relevance to a query.
pub struct RlxReranker {
    model: RlxBertModel,
    tok: BertTokenizer,
    hidden: usize,
    /// Fixed padded sequence length the encoder is compiled for (avoids a
    /// per-pair recompile; pad tokens are zeroed out by the attention mask).
    max_seq: usize,
    /// BERT pooler (`dense` `[H,H]` + `tanh`), applied to `[CLS]` before the
    /// head. `None` for checkpoints whose classifier reads `[CLS]` directly.
    pooler_w: Option<Vec<f32>>,
    pooler_b: Option<Vec<f32>>,
    /// Classification head: `logit = pooled · Wᵀ + b`, `W` shape `[num_labels, H]`.
    cls_w: Vec<f32>,
    cls_b: Vec<f32>,
    num_labels: usize,
}

/// First matching key's `(data, shape)`, cloned out of the map.
fn get_any(wm: &WeightMap, keys: &[&str]) -> Option<(Vec<f32>, Vec<usize>)> {
    for k in keys {
        if let Some((d, s)) = wm.get(k) {
            return Some((d.to_vec(), s.to_vec()));
        }
    }
    None
}

impl RlxReranker {
    /// Build from already-materialized files: a BERT `config.json`, a
    /// `model.safetensors`, and a tokenizer directory (holding `tokenizer.json`,
    /// `config.json`, `special_tokens_map.json`, `tokenizer_config.json`).
    /// `max_seq` is the padded pair length (query + passage + specials).
    pub fn from_files(
        config_path: &Path,
        weights_path: &str,
        tokenizer_dir: &Path,
        device: Device,
        max_seq: usize,
    ) -> Result<Self> {
        let model = RlxBertModel::load_sized_on(config_path, weights_path, 1, max_seq, device)
            .context("load cross-encoder BERT encoder")?;
        let hidden = model.hidden_size();
        let tok = BertTokenizer::from_dir(tokenizer_dir, max_seq)
            .with_context(|| format!("load reranker tokenizer from {tokenizer_dir:?}"))?;

        // Read the head weights from the same checkpoint. The pooler lives under
        // the encoder's `bert.` prefix (if any); the classifier is top-level.
        let wm = WeightMap::from_file(weights_path).context("open reranker weights")?;
        let pooler = get_any(&wm, &["bert.pooler.dense.weight", "pooler.dense.weight"]);
        let pooler_b = get_any(&wm, &["bert.pooler.dense.bias", "pooler.dense.bias"]);
        let (pooler_w, pooler_b) = match (pooler, pooler_b) {
            (Some((w, _)), Some((b, _))) => (Some(w), Some(b)),
            _ => (None, None),
        };
        let (cls_w, cls_shape) = get_any(&wm, &["classifier.weight", "classifier.dense.weight"])
            .context("reranker checkpoint has no `classifier.weight` (not a sequence-classification model?)")?;
        let (cls_b, _) = get_any(&wm, &["classifier.bias", "classifier.dense.bias"])
            .unwrap_or_else(|| (vec![0.0], vec![1]));
        // classifier weight is [num_labels, hidden] row-major.
        let num_labels = if cls_shape.len() == 2 && cls_shape[1] == hidden {
            cls_shape[0]
        } else {
            (cls_w.len() / hidden.max(1)).max(1)
        };

        Ok(Self {
            model,
            tok,
            hidden,
            max_seq,
            pooler_w,
            pooler_b,
            cls_w,
            cls_b,
            num_labels,
        })
    }

    /// Download `repo` (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) via hf-hub
    /// and build a reranker on `device`.
    #[cfg(feature = "hf-download")]
    pub fn from_pretrained(repo: &str, device: Device, max_seq: usize) -> Result<Self> {
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_progress(true)
            .build()
            .context("hf-hub ApiBuilder::build")?;
        let r = api.model(repo.to_string());
        let config = r
            .get("config.json")
            .with_context(|| format!("fetch {repo} config.json"))?;
        let weights = r
            .get("model.safetensors")
            .with_context(|| format!("fetch {repo} model.safetensors"))?;
        // BertTokenizer::from_dir needs these four in the same directory.
        let _ = r
            .get("tokenizer.json")
            .with_context(|| format!("fetch {repo} tokenizer.json"))?;
        let _ = r
            .get("special_tokens_map.json")
            .with_context(|| format!("fetch {repo} special_tokens_map.json"))?;
        let _ = r
            .get("tokenizer_config.json")
            .with_context(|| format!("fetch {repo} tokenizer_config.json"))?;
        let dir = config
            .parent()
            .ok_or_else(|| anyhow::anyhow!("reranker config has no parent dir"))?;
        let weights_str = weights
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("non-utf8 reranker weights path"))?;
        Self::from_files(&config, weights_str, dir, device, max_seq)
    }

    /// Relevance logit for one `(query, passage)` pair — higher = more relevant.
    /// Only the ordering across passages for a fixed query is meaningful (the
    /// absolute value is the raw classifier logit, uncalibrated).
    pub fn score(&mut self, query: &str, passage: &str) -> Result<f32> {
        let (ids, mask, tt) = self.tok.encode_pair(query, passage)?;
        let seq = self.max_seq;
        let mut ids_f = vec![0f32; seq];
        let mut mask_f = vec![0f32; seq];
        let mut tt_f = vec![0f32; seq];
        let mut pos_f = vec![0f32; seq];
        let n = ids.len().min(seq);
        for i in 0..n {
            ids_f[i] = ids[i] as f32;
            mask_f[i] = mask[i] as f32;
            tt_f[i] = tt[i] as f32;
            pos_f[i] = i as f32;
        }
        // Keep single-pair scoring at batch=1 even if a prior batched rerank left
        // the encoder compiled for a larger batch.
        self.model.recompile(1, seq)?;
        let hidden = self.model.forward(&ids_f, &mask_f, &tt_f, &pos_f);
        let h = self.hidden;
        if hidden.len() < h {
            return Ok(f32::NEG_INFINITY);
        }
        Ok(self.head(&hidden[0..h]))
    }

    /// `[CLS]` → optional BERT pooler (`dense` + `tanh`) → single-logit classifier.
    fn head(&self, cls: &[f32]) -> f32 {
        let h = self.hidden;
        let pooled: Vec<f32> = match (&self.pooler_w, &self.pooler_b) {
            (Some(pw), Some(pb)) if pw.len() >= h * h && pb.len() >= h => {
                let mut o = vec![0f32; h];
                for (j, oj) in o.iter_mut().enumerate() {
                    let mut s = pb[j];
                    let base = j * h;
                    for k in 0..h {
                        s += cls[k] * pw[base + k];
                    }
                    *oj = s.tanh();
                }
                o
            }
            _ => cls.to_vec(),
        };
        let _ = self.num_labels;
        let mut logit = self.cls_b.first().copied().unwrap_or(0.0);
        for (k, pk) in pooled.iter().enumerate().take(h) {
            logit += pk * self.cls_w.get(k).copied().unwrap_or(0.0);
        }
        logit
    }

    /// Score all `passages` against `query` in ONE fused batched forward
    /// (`batch = passages.len()`, fixed `max_seq`) instead of N batch=1 forwards
    /// — a single kernel launch + one weight load for the whole candidate set
    /// (the reranker's throughput fix; the pairs share weights and padded seq).
    /// Returns one logit per passage, in input order.
    fn score_batch(&mut self, query: &str, passages: &[&str]) -> Result<Vec<f32>> {
        let n = passages.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let seq = self.max_seq;
        let mut ids = vec![0f32; n * seq];
        let mut mask = vec![0f32; n * seq];
        let mut tt = vec![0f32; n * seq];
        let mut pos = vec![0f32; n * seq];
        for (r, p) in passages.iter().enumerate() {
            let (pi, pm, ptt) = self.tok.encode_pair(query, p)?;
            let m = pi.len().min(seq);
            let base = r * seq;
            for i in 0..m {
                ids[base + i] = pi[i] as f32;
                mask[base + i] = pm[i] as f32;
                tt[base + i] = ptt[i] as f32;
                pos[base + i] = i as f32;
            }
        }
        self.model.recompile(n, seq)?;
        let hidden = self.model.forward(&ids, &mask, &tt, &pos);
        let h = self.hidden;
        let mut out = Vec::with_capacity(n);
        for r in 0..n {
            let off = r * seq * h;
            if off + h > hidden.len() {
                out.push(f32::NEG_INFINITY);
            } else {
                // CLS = position 0 of row r in the [batch, seq, hidden] output.
                out.push(self.head(&hidden[off..off + h]));
            }
        }
        Ok(out)
    }

    /// Score every passage against `query` and return `(original_index, score)`
    /// sorted by descending relevance. The caller keeps the top few. Uses one
    /// fused batched forward over all candidates.
    pub fn rerank(&mut self, query: &str, passages: &[&str]) -> Result<Vec<(usize, f32)>> {
        let mut scored: Vec<(usize, f32)> = self
            .score_batch(query, passages)?
            .into_iter()
            .enumerate()
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored)
    }

    /// Encoder hidden size.
    pub fn hidden_size(&self) -> usize {
        self.hidden
    }
}
