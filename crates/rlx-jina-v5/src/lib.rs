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

//! `jinaai/jina-embeddings-v5-text-nano` — text embeddings from a
//! bidirectional EuroBERT trunk with a per-task LoRA.
//!
//! ```text
//!   "Query: " | "Document: " + text + <|end_of_text|>
//!        │
//!   EuroBERT (12 Llama-shaped layers, bidirectional) + task LoRA ─► norm
//!        │
//!   hidden state of the LAST real token (the appended EOS)
//!        │
//!       L2
//! ```
//!
//! EuroBERT is a Llama decoder layer with the causal mask removed, so the
//! trunk is `rlx-llama32`'s prefill flow with each layer's attention reading an
//! additive key-padding bias instead. Pooling is not a mean: the custom
//! sentence-transformers module takes the hidden state at `len − 1`.
//!
//! The checkpoint ships four LoRA adapters (`retrieval`, `text-matching`,
//! `clustering`, `classification`; rank 32, alpha 32, on q/k/v/o and the
//! three MLP projections). One is merged into the base weights at load —
//! `W + (alpha / r) · B·A` — so inference runs the plain trunk.

#[cfg(feature = "export")]
pub mod export;

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use rlx_core::weight_loader::WeightLoader;
use rlx_core::{AssetSource, SafetensorsSource, TensorRows};
use rlx_flow::LayerStack;
use rlx_flow::blocks::SelfAttnPrefillSpec;
use rlx_ir::{DType, Shape};
use rlx_llama32::{Llama32Config, Llama32Flow, LlamaLayerCtx};
use rlx_runtime::{CompiledGraph, Device};

/// Task adapters shipped under `adapters/`.
pub const TASKS: [&str; 4] = ["retrieval", "text-matching", "clustering", "classification"];

/// Sequences are padded up to a multiple of this, so a range of lengths
/// shares one compiled graph.
const SEQ_BUCKET: usize = 64;

/// Additive bias for a masked (padding) key; finite so padded query rows
/// stay finite.
const MASKED: f32 = -1e9;

/// A loaded jina-embeddings-v5 checkpoint with one task adapter merged.
pub struct JinaV5 {
    weights: SafetensorsSource,
    /// Token embeddings, gathered on the host so the 128256-row table is not
    /// copied into every compiled bucket (a second bucket would otherwise
    /// exhaust wasm32's 4 GiB).
    embed: TensorRows,
    adapter: SafetensorsSource,
    adapter_cfg: serde_json::Value,
    cfg: Llama32Config,
    device: Device,
    task: String,
    prompts: HashMap<String, String>,
    compiled: HashMap<usize, CompiledGraph>,
    #[cfg(feature = "tokenizer")]
    tokenizer: tokenizers::Tokenizer,
}

impl JinaV5 {
    /// Open a checkpoint directory with the adapter for `task` (one of [`TASKS`]).
    pub fn open(dir: impl Into<PathBuf>, task: &str, device: Device) -> Result<Self> {
        Self::open_source(&AssetSource::dir(dir.into()), task, device)
    }

    /// Open from any [`AssetSource`] — a directory, a `.rlxpack`, or in-memory
    /// files (the browser path) — under the checkpoint's own file names:
    /// `config.json`, `model.safetensors`, `tokenizer.json`,
    /// `adapters/<task>/adapter_{config.json,model.safetensors}`.
    pub fn open_source(src: &AssetSource, task: &str, device: Device) -> Result<Self> {
        rlx_core::validate_browser_capable_device("jina-v5", device)?;
        if !TASKS.contains(&task) {
            bail!("jina-v5: unknown task {task:?}; expected one of {TASKS:?}");
        }
        let cfg: Llama32Config = serde_json::from_str(&src.read_to_string("config.json")?)
            .context("jina-v5: parse config.json")?;
        let prompts = src
            .read_json::<serde_json::Value>("config_sentence_transformers.json")
            .ok()
            .and_then(|v| v.get("prompts").cloned())
            .and_then(|p| serde_json::from_value(p).ok())
            .unwrap_or_default();
        #[cfg(feature = "tokenizer")]
        let tokenizer = tokenizers::Tokenizer::from_bytes(src.read("tokenizer.json")?)
            .map_err(|e| anyhow!("jina-v5: load tokenizer.json: {e}"))?;
        let adapter_dir = format!("adapters/{task}");
        let weights = SafetensorsSource::from_asset(src, "model.safetensors")?;
        Ok(Self {
            embed: TensorRows::open(&weights, "embed_tokens.weight")?,
            weights,
            adapter: SafetensorsSource::from_asset(
                src,
                &format!("{adapter_dir}/adapter_model.safetensors"),
            )?,
            adapter_cfg: src
                .read_json(&format!("{adapter_dir}/adapter_config.json"))
                .with_context(|| format!("jina-v5: read {adapter_dir}/adapter_config.json"))?,
            cfg,
            device,
            task: task.to_string(),
            prompts,
            compiled: HashMap::new(),
            #[cfg(feature = "tokenizer")]
            tokenizer,
        })
    }

    pub fn config(&self) -> &Llama32Config {
        &self.cfg
    }

    pub fn task(&self) -> &str {
        &self.task
    }

    /// Embedding width (768).
    pub fn dim(&self) -> usize {
        self.cfg.hidden_size
    }

    /// `"query"` → `"Query: "`, `"document"` → `"Document: "`.
    pub fn prompt(&self, name: &str) -> Option<&str> {
        self.prompts.get(name).map(String::as_str)
    }

    /// Unit-length embedding of one token sequence (text ids + the EOS the
    /// tokenizer appends).
    pub fn embed_ids(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        let job = self.prepare(ids)?;
        let compiled = self.compiled.get_mut(&job.seq).expect("prepared");
        let out = compiled.run(&job.inputs());
        self.finish(out, job.n)
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
        self.finish(out, job.n)
    }

    /// Graph inputs for `ids`, compiling the graph for their bucket if needed.
    fn prepare(&mut self, ids: &[u32]) -> Result<Job> {
        if ids.is_empty() {
            bail!("jina-v5: empty input");
        }
        let ids = &ids[..ids.len().min(self.cfg.max_position_embeddings)];
        let n = ids.len();
        let seq = n.div_ceil(SEQ_BUCKET) * SEQ_BUCKET;
        let mut embeds = self.embed.gather(ids)?;
        // Padding positions are masked out of attention and never pooled.
        embeds.resize(seq * self.cfg.hidden_size, 0.0);
        let bias = key_padding_bias(self.cfg.num_attention_heads, seq, n);

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
            embeds,
            bias,
        })
    }

    /// Last real token's state, normalized.
    fn finish(&self, out: Vec<Vec<f32>>, n: usize) -> Result<Vec<f32>> {
        let hidden = out
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("jina-v5: graph produced no output"))?;
        let h = self.cfg.hidden_size;
        let mut v = hidden[(n - 1) * h..n * h].to_vec();
        l2_normalize(&mut v);
        Ok(v)
    }

    /// Tokenize `prompt + text` and embed it.
    #[cfg(feature = "tokenizer")]
    pub fn embed_text(&mut self, prompt: &str, text: &str) -> Result<Vec<f32>> {
        let ids = self.tokenize(&format!("{prompt}{text}"))?;
        self.embed_ids(&ids)
    }

    /// Token ids as sentence-transformers produces them (text + EOS).
    #[cfg(feature = "tokenizer")]
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("jina-v5: tokenize: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    fn compile(&self, seq: usize) -> Result<CompiledGraph> {
        let built = build_encoder(
            &self.cfg,
            self.weights.loader()?,
            self.adapter.loader()?,
            &self.adapter_cfg,
            seq,
        )?;
        rlx_core::flow_util::compile_built(built, self.device)
    }
}

/// The encoder at `seq` tokens, the adapter merged into the base weights:
/// `inputs_embeds` and the key-padding `attn_bias` in, the final hidden
/// states out.
fn build_encoder(
    cfg: &Llama32Config,
    base: Box<dyn WeightLoader>,
    adapter: Box<dyn WeightLoader>,
    adapter_cfg: &serde_json::Value,
    seq: usize,
) -> Result<rlx_flow::BuiltModel> {
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let head_dim = cfg.head_dim.unwrap_or(cfg.hidden_size / nh);
    let eps = cfg.rms_norm_eps as f32;
    let bias_shape = Shape::new(&[1, nh, seq, seq], DType::F32);
    let mut loader = AdapterMerged::open(base, adapter, adapter_cfg)?;
    Llama32Flow::for_prefill(cfg, 1, seq)
        .inputs_embeds()
        .hidden_only()
        .profile_prefill()
        .layer(move |ctx| match ctx {
            LlamaLayerCtx::Prefill { weight_index, .. } => {
                bidirectional_layer(weight_index, nh, head_dim, nkv, eps)
            }
            decode => decode.default_stage(),
        })
        .patch_flow(move |flow| flow.input("attn_bias", bias_shape.clone()))
        .build(&mut loader)
}

/// Inputs for one forward pass, built by `JinaV5::prepare`.
struct Job {
    seq: usize,
    n: usize,
    embeds: Vec<f32>,
    bias: Vec<f32>,
}

impl Job {
    fn inputs(&self) -> [(&str, &[f32]); 2] {
        [("inputs_embeds", &self.embeds), ("attn_bias", &self.bias)]
    }
}

/// One EuroBERT layer: a Llama pre-norm block whose attention reads the
/// additive `attn_bias` input instead of a causal mask.
fn bidirectional_layer(
    layer: usize,
    heads: usize,
    head_dim: usize,
    kv_heads: usize,
    eps: f32,
) -> rlx_flow::FlowStage {
    let prefix = format!("model.layers.{layer}");
    LayerStack::named(format!("layer{layer}"))
        .residual_save()
        .rms_norm(format!("{prefix}.input_layernorm.weight"), eps)
        .self_attn_prefill(
            SelfAttnPrefillSpec::hf_layer(&prefix, heads, head_dim, kv_heads)
                .with_bias_input("attn_bias"),
        )
        .linear(format!("{prefix}.self_attn.o_proj.weight"), true)
        .residual_add()
        .residual_save()
        .rms_norm(format!("{prefix}.post_attention_layernorm.weight"), eps)
        .swiglu_hf_mlp(format!("{prefix}.mlp"))
        .residual_add()
        .build()
}

/// `[1, heads, seq, seq]`: every query sees the first `live` keys.
fn key_padding_bias(heads: usize, seq: usize, live: usize) -> Vec<f32> {
    let mut row = vec![MASKED; seq];
    row[..live].fill(0.0);
    row.repeat(seq * heads)
}

/// Renormalized Matryoshka prefix of an embedding.
pub fn truncate(embedding: &[f32], dim: usize) -> Vec<f32> {
    let mut v = embedding[..dim.min(embedding.len())].to_vec();
    l2_normalize(&mut v);
    v
}

/// Cosine similarity of two unit vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= n);
}

/// Base weights with one LoRA adapter folded in.
///
/// The checkpoint is a bare EuroBERT (`layers.0…`); the flow asks for
/// `model.layers.0…`. Adapter tensors are
/// `base_model.model.<module>.lora_{A,B}.weight`.
struct AdapterMerged {
    base: Box<dyn WeightLoader>,
    /// `<module>.weight` → (`A` `[r, in]`, `B` `[out, r]`, `r`).
    lora: HashMap<String, (Vec<f32>, Vec<f32>, usize)>,
    scale: f32,
}

impl AdapterMerged {
    /// `base` weights, the adapter's tensors `src` and its `adapter_config.json`.
    fn open(
        base: Box<dyn WeightLoader>,
        mut src: Box<dyn WeightLoader>,
        cfg: &serde_json::Value,
    ) -> Result<Self> {
        let r = cfg["r"]
            .as_f64()
            .ok_or_else(|| anyhow!("jina-v5: adapter r"))?;
        let alpha = cfg["lora_alpha"]
            .as_f64()
            .ok_or_else(|| anyhow!("jina-v5: adapter lora_alpha"))?;
        if cfg["use_rslora"].as_bool() == Some(true) || cfg["use_dora"].as_bool() == Some(true) {
            bail!("jina-v5: rsLoRA / DoRA adapters are not supported");
        }
        let mut lora = HashMap::new();
        for key in src.remaining_keys() {
            let Some(module) = key
                .strip_prefix("base_model.model.")
                .and_then(|k| k.strip_suffix(".lora_A.weight"))
            else {
                continue;
            };
            let (a, a_shape) = src.take(&key)?;
            let (b, b_shape) = src.take(&format!("base_model.model.{module}.lora_B.weight"))?;
            let rank = a_shape[0];
            if b_shape[1] != rank {
                bail!("jina-v5: {module}: lora_A {a_shape:?} / lora_B {b_shape:?}");
            }
            lora.insert(format!("{module}.weight"), (a, b, rank));
        }
        if lora.is_empty() {
            bail!("jina-v5: adapter has no LoRA tensors");
        }
        Ok(Self {
            base,
            lora,
            scale: (alpha / r) as f32,
        })
    }

    fn key(key: &str) -> &str {
        key.strip_prefix("model.").unwrap_or(key)
    }
}

impl WeightLoader for AdapterMerged {
    fn len(&self) -> usize {
        self.base.len()
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let key = Self::key(key);
        let (mut w, shape) = self.base.take(key)?;
        if let Some((a, b, r)) = self.lora.remove(key) {
            let (out, inp) = (shape[0], shape[1]);
            if a.len() != r * inp || b.len() != out * r {
                bail!("jina-v5: {key} {shape:?} does not match its rank-{r} adapter");
            }
            for o in 0..out {
                let row = &mut w[o * inp..(o + 1) * inp];
                for k in 0..r {
                    let bk = self.scale * b[o * r + k];
                    for (x, &ak) in row.iter_mut().zip(&a[k * inp..(k + 1) * inp]) {
                        *x += bk * ak;
                    }
                }
            }
        }
        Ok((w, shape))
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (w, shape) = self.take(key)?;
        let [rows, cols] = shape[..] else {
            bail!("jina-v5: take_transposed on {key} with shape {shape:?}");
        };
        let mut t = vec![0f32; w.len()];
        for i in 0..rows {
            for j in 0..cols {
                t[j * rows + i] = w[i * cols + j];
            }
        }
        Ok((t, vec![cols, rows]))
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.base.remaining_keys()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_padding_bias_masks_only_padding_keys() {
        let b = key_padding_bias(2, 4, 3);
        assert_eq!(b.len(), 2 * 4 * 4);
        for q in 0..4 {
            assert_eq!(&b[q * 4..q * 4 + 4], &[0.0, 0.0, 0.0, MASKED]);
        }
    }
}
