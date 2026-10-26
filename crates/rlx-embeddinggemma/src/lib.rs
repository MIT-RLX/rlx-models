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

//! `google/embeddinggemma-300m` — text embeddings from a bidirectional Gemma 3.
//!
//! ```text
//!   prompt + text ─► Gemma 3 trunk (24 layers, bidirectional) ─► norm
//!                         │
//!               mean over real tokens
//!                         │
//!        2_Dense [3072, 768] · 3_Dense [768, 3072]   (no bias, no activation)
//!                         │
//!                        L2
//! ```
//!
//! The trunk is `rlx-gemma`'s Gemma 3 prefill flow with one change: every
//! layer's attention reads an additive bias instead of a causal mask, because
//! the checkpoint sets `use_bidirectional_attention`. transformers then lets a
//! full-attention layer see every real token, and a sliding layer every real
//! token with `|q − k| < sliding_window / 2 + 1` (257 for the configured 512,
//! see `bidirectional_window`) — so the two layer kinds differ only past 257
//! tokens, and get separate bias inputs.
//!
//! The head is two bias-free linears with an identity activation between them,
//! which compose into one `[768, 768]` matrix; and a linear map commutes with
//! a mean, so the pooled vector is projected once rather than every token
//! ([`rlx_gemma::embedding_gemma2::pool_project_normalize`]).
//!
//! Matryoshka: the model is trained so a prefix of the 768-wide vector is
//! itself an embedding; [`truncate`] cuts to 512/256/128 and renormalizes.

#[cfg(feature = "export")]
pub mod export;

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use rlx_core::weight_loader::WeightLoader;
use rlx_core::{AssetSource, SafetensorsSource, TensorRows};
use rlx_flow::blocks::gemma_prefill_layer_composed;
use rlx_gemma::GemmaConfig;
use rlx_gemma::embedding_gemma2::{l2_normalize, pool_project_normalize};
use rlx_gemma::flow::{GemmaFlow, GemmaLayerCtx};
use rlx_ir::{DType, Shape};
use rlx_runtime::{CompiledGraph, Device};

/// Sequences are padded up to a multiple of this, so a range of lengths
/// shares one compiled graph.
const SEQ_BUCKET: usize = 64;

/// Additive bias for a masked attention score. Finite, so a fully padded
/// query row (never pooled) stays finite instead of producing NaN.
const MASKED: f32 = -1e9;

/// A loaded EmbeddingGemma checkpoint.
pub struct EmbeddingGemma {
    weights: SafetensorsSource,
    /// Token embedding rows, gathered on the host: the 262144 × 768 table is
    /// two-thirds of the checkpoint, and in-graph it would be a second full
    /// copy — past wasm32's 4 GiB together with the file bytes.
    embed: TensorRows,
    cfg: GemmaConfig,
    device: Device,
    /// `3_Dense · 2_Dense`, `[hidden, hidden]` row-major.
    head: Vec<f32>,
    max_seq: usize,
    prompts: HashMap<String, String>,
    compiled: HashMap<usize, CompiledGraph>,
    #[cfg(feature = "tokenizer")]
    tokenizer: tokenizers::Tokenizer,
}

impl EmbeddingGemma {
    /// Open a checkpoint directory (`config.json`, `model.safetensors`,
    /// `2_Dense/`, `3_Dense/`, `tokenizer.json`).
    pub fn open(dir: impl Into<PathBuf>, device: Device) -> Result<Self> {
        Self::open_source(&AssetSource::dir(dir.into()), device)
    }

    /// Open from any [`AssetSource`] — a directory, a `.rlxpack`, or in-memory
    /// files (the browser path) — under the checkpoint's own file names.
    pub fn open_source(src: &AssetSource, device: Device) -> Result<Self> {
        rlx_core::validate_browser_capable_device("embeddinggemma", device)?;
        let raw = src.read_to_string("config.json")?;
        let value: serde_json::Value =
            serde_json::from_str(&raw).context("embeddinggemma: parse config.json")?;
        let cfg = GemmaConfig::from_hf_value(&value, &raw)?;
        if cfg.sliding_window.is_none() {
            bail!("embeddinggemma: config has no sliding_window — not a Gemma 3 embedder");
        }
        let head = compose_dense_head(src, cfg.hidden_size)?;
        let weights = SafetensorsSource::from_asset(src, "model.safetensors")?;
        let max_seq = src
            .read_json::<serde_json::Value>("sentence_bert_config.json")
            .ok()
            .and_then(|v| v.get("max_seq_length")?.as_u64())
            .map_or(cfg.max_position_embeddings, |v| v as usize);
        let prompts = src
            .read_json::<serde_json::Value>("config_sentence_transformers.json")
            .ok()
            .and_then(|v| v.get("prompts").cloned())
            .and_then(|p| serde_json::from_value(p).ok())
            .unwrap_or_default();
        #[cfg(feature = "tokenizer")]
        let tokenizer = tokenizers::Tokenizer::from_bytes(src.read("tokenizer.json")?)
            .map_err(|e| anyhow!("embeddinggemma: load tokenizer.json: {e}"))?;
        Ok(Self {
            embed: TensorRows::open(&weights, "embed_tokens.weight")?,
            weights,
            cfg,
            device,
            head,
            max_seq,
            prompts,
            compiled: HashMap::new(),
            #[cfg(feature = "tokenizer")]
            tokenizer,
        })
    }

    pub fn config(&self) -> &GemmaConfig {
        &self.cfg
    }

    /// Embedding width (768).
    pub fn dim(&self) -> usize {
        self.cfg.hidden_size
    }

    /// A named prompt from `config_sentence_transformers.json` — `"query"`,
    /// `"document"`, `"Classification"`, `"STS"`, …
    pub fn prompt(&self, name: &str) -> Option<&str> {
        self.prompts.get(name).map(String::as_str)
    }

    /// Unit-length embedding of one token sequence (BOS … EOS included).
    pub fn embed_ids(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        let job = self.prepare(ids)?;
        let compiled = self.compiled.get_mut(&job.seq).expect("prepared");
        let out = compiled.run(&job.inputs());
        self.finish(out, &job)
    }

    /// [`embed_ids`](Self::embed_ids) on wasm WebGPU, where reading results
    /// back from the GPU cannot block.
    #[cfg(target_arch = "wasm32")]
    pub async fn embed_ids_async(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        let job = self.prepare(ids)?;
        let compiled = self.compiled.get_mut(&job.seq).expect("prepared");
        let out = compiled
            .run_async(&job.inputs())
            .await
            .map_err(|e| anyhow!(e))?;
        self.finish(out, &job)
    }

    /// Graph inputs for `ids`, compiling the graph for their bucket if needed.
    fn prepare(&mut self, ids: &[u32]) -> Result<Job> {
        if ids.is_empty() {
            bail!("embeddinggemma: empty input");
        }
        let ids = &ids[..ids.len().min(self.max_seq)];
        let n = ids.len();
        let seq = n.div_ceil(SEQ_BUCKET) * SEQ_BUCKET;
        let window = self
            .cfg
            .sliding_window
            .map_or(usize::MAX, bidirectional_window);
        let nh = self.cfg.num_attention_heads;

        // Gemma scales embeddings by √hidden (an f32 constant, as in-graph).
        let h = self.cfg.hidden_size;
        let scale = (h as f64).sqrt() as f32;
        let mut hidden_in = self.embed.gather(ids)?;
        hidden_in.iter_mut().for_each(|x| *x *= scale);
        // Padding positions are masked out of attention and pooling.
        hidden_in.resize(seq * h, 0.0);
        let full = attention_bias(nh, seq, n, usize::MAX);
        let sliding = attention_bias(nh, seq, n, window);

        if !self.compiled.contains_key(&seq) {
            // On wasm32 keep one graph: each holds a full copy of the weights.
            if cfg!(target_arch = "wasm32") {
                self.compiled.clear();
            }
            let compiled = self.compile(seq)?;
            self.compiled.insert(seq, compiled);
        }
        Ok(Job {
            seq,
            n,
            hidden_in,
            full,
            sliding,
        })
    }

    /// Mean-pool the real tokens, project, normalize.
    fn finish(&self, out: Vec<Vec<f32>>, job: &Job) -> Result<Vec<f32>> {
        let hidden = out
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("embeddinggemma: graph produced no output"))?;
        let h = self.cfg.hidden_size;
        let mask: Vec<bool> = (0..job.seq).map(|i| i < job.n).collect();
        pool_project_normalize(&hidden[..job.seq * h], &mask, h, &self.head, h)
    }

    /// Tokenize `prompt + text` and embed it.
    #[cfg(feature = "tokenizer")]
    pub fn embed_text(&mut self, prompt: &str, text: &str) -> Result<Vec<f32>> {
        let ids = self.tokenize(&format!("{prompt}{text}"))?;
        self.embed_ids(&ids)
    }

    /// Token ids as sentence-transformers produces them (BOS + text + EOS).
    #[cfg(feature = "tokenizer")]
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("embeddinggemma: tokenize: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    fn compile(&self, seq: usize) -> Result<CompiledGraph> {
        let built = build_encoder(&self.cfg, self.weights.loader()?, seq)?;
        rlx_core::flow_util::compile_built(built, self.device)
    }
}

/// The encoder at `seq` tokens: `prefill_hidden` (the scaled token
/// embeddings) and the full and sliding attention biases in, the final
/// hidden states out.
fn build_encoder(
    cfg: &GemmaConfig,
    weights: Box<dyn WeightLoader>,
    seq: usize,
) -> Result<rlx_flow::BuiltModel> {
    let cfg = cfg.clone();
    let nh = cfg.num_attention_heads;
    let full_layers: Vec<bool> = (0..cfg.active_num_layers())
        .map(|i| cfg.is_full_attention_layer(i))
        .collect();
    let bias_shape = Shape::new(&[1, nh, seq, seq], DType::F32);
    let mut loader = ModelPrefix(weights);
    GemmaFlow::for_prefill(&cfg, 1, seq)
        .prefill_from_hidden()
        .hidden_only()
        .profile_prefill()
        .layer(move |ctx| match ctx {
            GemmaLayerCtx::Prefill {
                index,
                style,
                attn,
                eps,
                ..
            } => {
                let bias = if full_layers[index] {
                    "attn_bias_full"
                } else {
                    "attn_bias_sliding"
                };
                gemma_prefill_layer_composed(index, style, attn.with_bias_input(bias), eps, None)
            }
            decode => decode.default_stage(),
        })
        .patch_flow(move |flow| {
            flow.input("attn_bias_full", bias_shape.clone())
                .input("attn_bias_sliding", bias_shape.clone())
        })
        .build(&mut loader)
}

/// Inputs for one forward pass, built by `EmbeddingGemma::prepare`.
struct Job {
    seq: usize,
    n: usize,
    hidden_in: Vec<f32>,
    full: Vec<f32>,
    sliding: Vec<f32>,
}

impl Job {
    fn inputs(&self) -> [(&str, &[f32]); 3] {
        [
            ("prefill_hidden", &self.hidden_in),
            ("attn_bias_full", &self.full),
            ("attn_bias_sliding", &self.sliding),
        ]
    }
}

/// Renormalized Matryoshka prefix of an embedding (`dim` ∈ {768, 512, 256, 128}).
pub fn truncate(embedding: &[f32], dim: usize) -> Vec<f32> {
    let mut v = embedding[..dim.min(embedding.len())].to_vec();
    l2_normalize(&mut v);
    v
}

/// Cosine similarity of two unit vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The `|q − k|` bound a sliding layer actually uses.
///
/// With `use_bidirectional_attention`, transformers rewrites the configured
/// window on load to `window / 2 + 1` ("due to fa we set exclusive bounds"):
/// a 512-token window looks 256 tokens either way, not 512. Using 512 as
/// written measured cosine 0.9875 on an 899-token input; this matches to 2e-7.
fn bidirectional_window(configured: usize) -> usize {
    configured / 2 + 1
}

/// `[1, heads, seq, seq]` additive bias: query `q` sees key `k` when `k` is a
/// real token (`k < live`) and `|q − k| < window`.
fn attention_bias(heads: usize, seq: usize, live: usize, window: usize) -> Vec<f32> {
    let mut one = vec![MASKED; seq * seq];
    for q in 0..seq {
        for k in 0..live {
            if q.abs_diff(k) < window {
                one[q * seq + k] = 0.0;
            }
        }
    }
    one.repeat(heads)
}

/// `3_Dense.weight · 2_Dense.weight` — `[hidden, hidden]`. Both linears are
/// bias-free with an identity activation, so they are one linear map.
fn compose_dense_head(src: &AssetSource, hidden: usize) -> Result<Vec<f32>> {
    let read = |sub: &str| -> Result<(Vec<f32>, Vec<usize>)> {
        SafetensorsSource::from_asset(src, &format!("{sub}/model.safetensors"))?
            .loader()
            .with_context(|| format!("embeddinggemma: open {sub}/model.safetensors"))?
            .take("linear.weight")
    };
    let (w1, s1) = read("2_Dense")?; // [mid, hidden]
    let (w2, s2) = read("3_Dense")?; // [hidden, mid]
    let mid = s1[0];
    if s1 != [mid, hidden] || s2 != [hidden, mid] {
        bail!("embeddinggemma: dense head shapes {s1:?} / {s2:?} do not chain through {hidden}");
    }
    let mut out = vec![0f32; hidden * hidden];
    for o in 0..hidden {
        for m in 0..mid {
            let a = w2[o * mid + m];
            let row = &w1[m * hidden..(m + 1) * hidden];
            for (acc, &b) in out[o * hidden..(o + 1) * hidden].iter_mut().zip(row) {
                *acc += a * b;
            }
        }
    }
    Ok(out)
}

/// The checkpoint is a bare `Gemma3TextModel` (`layers.0…`, `embed_tokens`);
/// the flow asks for the causal-LM names (`model.layers.0…`).
struct ModelPrefix(Box<dyn WeightLoader>);

impl ModelPrefix {
    fn key(key: &str) -> &str {
        key.strip_prefix("model.").unwrap_or(key)
    }
}

impl WeightLoader for ModelPrefix {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        // The flow registers the table even when fed hidden states (for a
        // tied LM head this crate does not build); a placeholder keeps the
        // real one out of the graph.
        if key == "model.embed_tokens.weight" {
            return Ok((vec![0.0], vec![1, 1]));
        }
        self.0.take(Self::key(key))
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.0.take_transposed(Self::key(key))
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.0.remaining_keys()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bias_is_bidirectional_windowed_and_masks_padding() {
        let b = attention_bias(1, 6, 4, 2);
        let at = |q: usize, k: usize| b[q * 6 + k];
        assert_eq!(at(1, 0), 0.0, "past within window");
        assert_eq!(at(1, 2), 0.0, "future within window — bidirectional");
        assert_eq!(at(0, 2), MASKED, "|q-k| = window is outside (exclusive)");
        assert_eq!(at(3, 4), MASKED, "padding key");
        assert_eq!(at(4, 3), 0.0, "padding query still sees real keys");
    }

    #[test]
    fn truncate_renormalizes() {
        let v = truncate(&[3.0, 4.0, 12.0], 2);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
    }
}
