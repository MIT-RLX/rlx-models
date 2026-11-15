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

//! `google/embeddinggemma-2` — a Gemma 4 trunk used as a multimodal embedder.
//!
//! ```text
//!   text ──► embed_tokens ─┐
//!   image ─► vision_tower ─┼─► embedding_projection ─► Gemma 4 trunk (24 layers)
//!   audio ─► audio_tower ──┘     (into the 512-wide text space)
//!                                          │
//!                           language_model.norm
//!                                          │
//!                   language_model.embedding_projection  [768, 512]
//!                                          │
//!                            mean over tokens, then L2
//! ```
//!
//! The trunk is an ordinary small Gemma 4 — 24 layers, `hidden_size = 512`,
//! 4 heads of 256, sliding/full alternation with a RoPE base per kind, and
//! per-layer input embeddings (`hidden_size_per_layer_input = 512`, so
//! `ple.per_layer_model_projection` is `[24 · 512, 512]`). All of that is
//! [`crate::GemmaConfig`] and the existing builder. What is specific to this
//! checkpoint is the two ends: how the modalities get *into* the trunk, and the
//! pooling head that turns token states into one unit vector.
//!
//! ## The head, and why its order does not matter
//!
//! sentence-transformers describes it as three modules: the transformer, then
//! `Pooling{mode: mean}`, then `Normalize`. The projection to
//! `embedding_dim = 768` lives inside the model, so a reader has to decide
//! whether to project each token and then average, or average and then project.
//!
//! It makes no difference, and that is worth stating rather than discovering:
//! `embedding_projection` carries **no bias**, so it is a linear map, and a
//! linear map commutes with an average — `W·(Σxᵢ/n) = Σ(W·xᵢ)/n`. Projecting
//! the pooled vector is the cheaper half (one `[768, 512]` matvec instead of
//! `seq` of them) and [`pool_project_normalize`] does that.
//! `tests/embedding_gemma2.rs` pins the equivalence so the shortcut cannot
//! quietly stop being one.
//!
//! ## Status
//!
//! Text and audio embeddings run end to end in
//! [`crate::embedding_gemma2_model::EmbeddingGemma2`], with the null-space audio
//! head in [`crate::embedding_gemma2_nullspace`]. Both match HF +
//! sentence-transformers at every stage on CPU, Metal and MLX
//! (`tests/embedding_gemma2_parity.rs`). The vision tower is not wired.
//!
//! [`EmbeddingGemma2Names`] maps this checkpoint's tensor names and keeps the
//! builders' `w − 1` norm convention, which a plain loader would not.

use anyhow::{Result, bail};
use rlx_core::weight_loader::WeightLoader;
use std::path::Path;

use crate::GemmaConfig;

/// Hyper-parameters specific to the embedding wrapper.
///
/// The trunk's own parameters are in [`Self::text`]; this is what the wrapper
/// adds around it.
#[derive(Debug, Clone)]
pub struct EmbeddingGemma2Config {
    /// The decoder trunk — an ordinary Gemma 4.
    pub text: GemmaConfig,
    /// Width of the pooled sentence embedding (`language_model.embedding_projection`).
    pub embedding_dim: usize,
    /// Placeholder ids the modality embeddings are spliced over.
    pub image_token_id: u32,
    pub audio_token_id: u32,
    pub video_token_id: u32,
    /// Begin/end-of-image and begin/end-of-audio markers.
    pub boi_token_id: u32,
    pub eoi_token_id: u32,
    pub boa_token_id: u32,
    pub eoa_token_id: u32,
    /// `text_config.eos_token_id`: closes every input, after the audio.
    pub eos_token_id: u32,
    /// Soft tokens one image occupies in the sequence.
    pub vision_soft_tokens_per_image: usize,
    /// Whether the checkpoint carries the respective tower.
    pub has_vision: bool,
    pub has_audio: bool,
    /// `audio_config.use_clipped_linears`: every audio-tower linear clamps its
    /// input and output to learned bounds ([`crate::gemma4_audio::AudioConfig`]).
    pub audio_use_clipped_linears: bool,
}

impl EmbeddingGemma2Config {
    /// Parse the published `config.json`.
    pub fn from_hf_json(raw: &str) -> Result<Self> {
        let v: serde_json::Value =
            serde_json::from_str(raw).map_err(|e| anyhow::anyhow!("embeddinggemma2: {e}"))?;
        let top = v
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("embeddinggemma2: config is not an object"))?;

        let model_type = top.get("model_type").and_then(|m| m.as_str()).unwrap_or("");
        if model_type != "embedding_gemma2" {
            bail!("embeddinggemma2: expected model_type `embedding_gemma2`, got {model_type:?}");
        }

        let text_v = top
            .get("text_config")
            .ok_or_else(|| anyhow::anyhow!("embeddinggemma2: no `text_config`"))?;
        // Hand over the subtree directly — no temp file, so nothing races
        // with itself when two threads parse at once. `raw` is passed for the
        // arch sniff, which keys off the nested `model_type`.
        let mut text = GemmaConfig::from_hf_value(text_v, raw)?;
        // The trunk is a Gemma 4, but nothing in this config names one, so
        // the arch sniff would fall back to plain Gemma.
        text.arch = crate::config::GemmaArch::Gemma4;
        apply_per_layer_config(&mut text, text_v)?;

        let u = |k: &str| -> Option<u64> { top.get(k).and_then(|x| x.as_u64()) };
        let embedding_dim = text_v
            .get("embedding_dim")
            .and_then(|x| x.as_u64())
            .ok_or_else(|| anyhow::anyhow!("embeddinggemma2: no `text_config.embedding_dim`"))?
            as usize;

        let audio = top.get("audio_config").and_then(|a| a.as_object());
        Ok(EmbeddingGemma2Config {
            text,
            embedding_dim,
            image_token_id: u("image_token_id").unwrap_or(0) as u32,
            audio_token_id: u("audio_token_id").unwrap_or(0) as u32,
            video_token_id: u("video_token_id").unwrap_or(0) as u32,
            boi_token_id: u("boi_token_id").unwrap_or(0) as u32,
            eoi_token_id: u("eoi_token_id").unwrap_or(0) as u32,
            boa_token_id: u("boa_token_id").unwrap_or(0) as u32,
            // NOTE the published key is `eoa_token_index`, not `..._id`.
            eoa_token_id: u("eoa_token_index")
                .or_else(|| u("eoa_token_id"))
                .unwrap_or(0) as u32,
            eos_token_id: text_v
                .get("eos_token_id")
                .and_then(|x| x.as_u64())
                .unwrap_or(1) as u32,
            vision_soft_tokens_per_image: u("vision_soft_tokens_per_image").unwrap_or(0) as usize,
            has_vision: top.contains_key("vision_config"),
            has_audio: audio.is_some(),
            audio_use_clipped_linears: audio
                .and_then(|a| a.get("use_clipped_linears"))
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
        })
    }

    /// Parse from a `config.json` path.
    pub fn from_hf_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("embeddinggemma2: read {path:?}: {e}"))?;
        Self::from_hf_json(&raw)
    }

    /// Tensors outside the per-layer stacks, which is where this checkpoint
    /// differs from a plain Gemma.
    ///
    /// Note the absence of an `lm_head`: an embedder has no vocabulary head,
    /// and `tie_word_embeddings` is irrelevant to it.
    pub fn head_tensor_names(&self) -> Vec<String> {
        let mut v = vec![
            "language_model.embed_tokens.weight".to_string(),
            "language_model.norm.weight".to_string(),
            "language_model.embedding_projection.weight".to_string(),
        ];
        if self.text.has_ple() {
            v.push("language_model.ple.per_layer_model_projection.weight".into());
            v.push("language_model.ple.per_layer_projection_norm.weight".into());
        }
        if self.has_vision {
            v.push("embed_vision.embedding_projection.weight".into());
        }
        if self.has_audio {
            v.push("embed_audio.embedding_projection.weight".into());
        }
        v
    }

    /// Shape of `ple.per_layer_model_projection`: one slice per layer.
    pub fn per_layer_projection_shape(&self) -> Option<[usize; 2]> {
        self.text.has_ple().then(|| {
            [
                self.text.num_hidden_layers * self.text.ple_width(),
                self.text.hidden_size,
            ]
        })
    }
}

/// `text_config.per_layer_config`: per-layer overrides keyed by the layer
/// index (`"05"`). This checkpoint uses them for one thing, the wider heads of
/// its full-attention layers (`head_dim` 512, one KV head), which
/// [`GemmaConfig`] states as `global_head_dim` / `num_global_key_value_heads`.
/// Anything else, or an override on a sliding layer, is refused rather than
/// ignored.
fn apply_per_layer_config(text: &mut GemmaConfig, text_v: &serde_json::Value) -> Result<()> {
    let Some(per_layer) = text_v.get("per_layer_config").and_then(|p| p.as_object()) else {
        return Ok(());
    };
    let mut global: Option<(usize, usize)> = None;
    for (key, over) in per_layer {
        let layer: usize = key.parse().map_err(|_| {
            anyhow::anyhow!("embeddinggemma2: per_layer_config key {key:?} is not a layer index")
        })?;
        if !text.is_full_attention_layer(layer) {
            bail!("embeddinggemma2: per_layer_config overrides sliding layer {layer}");
        }
        let obj = over.as_object().ok_or_else(|| {
            anyhow::anyhow!("embeddinggemma2: per_layer_config[{key}] is not an object")
        })?;
        if let Some(k) = obj
            .keys()
            .find(|k| *k != "head_dim" && *k != "num_key_value_heads")
        {
            bail!("embeddinggemma2: per_layer_config[{key}].{k} is not supported");
        }
        let hd = obj
            .get("head_dim")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize);
        let kv = obj
            .get("num_key_value_heads")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize);
        let this = (
            hd.unwrap_or(text.head_dim()),
            kv.unwrap_or(text.num_key_value_heads),
        );
        if global.is_some_and(|g| g != this) {
            bail!("embeddinggemma2: full-attention layers disagree on head_dim / kv heads");
        }
        global = Some(this);
    }
    if let Some((hd, kv)) = global {
        // Every full-attention layer must carry the override, or the uniform
        // `global_*` fields would misdescribe the ones that do not.
        let full = (0..text.num_hidden_layers)
            .filter(|&l| text.is_full_attention_layer(l))
            .count();
        if full != per_layer.len() {
            bail!(
                "embeddinggemma2: per_layer_config covers {} of {full} full-attention layers",
                per_layer.len()
            );
        }
        text.global_head_dim = Some(hd);
        text.num_global_key_value_heads = Some(kv);
    }
    Ok(())
}

/// A [`WeightLoader`] that serves the names [`crate::builder`] and
/// [`crate::gemma4_audio`] ask for (`model.layers.N.…`,
/// `model.audio_tower.…`) from this checkpoint's (`language_model.layers.N.…`,
/// `audio_tower.…`, PLE weights under `ple_block.`).
///
/// It also keeps the norm convention those builders share with
/// [`crate::qat_loader::GemmaQatLoader`]: a tensor whose checkpoint name ends
/// in `norm.weight` is served as `w − 1`, because the builders apply
/// `1 + w` (Gemma ≤ 3) while Gemma 4 norms multiply by `w` itself.
pub struct EmbeddingGemma2Names<L: WeightLoader> {
    inner: L,
}

impl<L: WeightLoader> EmbeddingGemma2Names<L> {
    pub fn new(inner: L) -> Self {
        Self { inner }
    }

    /// The checkpoint name of a builder-side `key`.
    pub fn map(key: &str) -> String {
        for tower in [
            "audio_tower.",
            "embed_audio.",
            "vision_tower.",
            "embed_vision.",
        ] {
            if let Some(rest) = key.strip_prefix("model.").filter(|r| r.starts_with(tower)) {
                return rest.to_string();
            }
        }
        if let Some(rest) = key.strip_prefix("model.layers.")
            && let Some((layer, tail)) = rest.split_once('.')
            && (tail.starts_with("per_layer_input_gate.")
                || tail.starts_with("per_layer_projection.")
                || tail.starts_with("post_per_layer_input_norm."))
        {
            return format!("language_model.layers.{layer}.ple_block.{tail}");
        }
        match key.strip_prefix("model.") {
            Some(rest) => format!("language_model.{rest}"),
            None => key.to_string(),
        }
    }
}

impl<L: WeightLoader> WeightLoader for EmbeddingGemma2Names<L> {
    fn format_id(&self) -> &'static str {
        self.inner.format_id()
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let name = Self::map(key);
        let (mut data, shape) = self.inner.take(&name)?;
        if name.ends_with("norm.weight") {
            data.iter_mut().for_each(|v| *v -= 1.0);
        }
        Ok((data, shape))
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let name = Self::map(key);
        if name.ends_with("norm.weight") {
            bail!("embeddinggemma2: {name} is a norm, not a matrix to transpose");
        }
        self.inner.take_transposed(&name)
    }
    fn stored_dtype(&self, key: &str) -> Option<rlx_ir::DType> {
        self.inner.stored_dtype(&Self::map(key))
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.inner.remaining_keys()
    }
}

/// Mean-pool `[seq, hidden]` token states, project to `embedding_dim`, and L2
/// normalize — the `Pooling{mean}` + `Normalize` head, on the host.
///
/// `mask` is one entry per token; `false` positions are excluded from the mean
/// rather than contributing a zero, which is the difference between padding
/// being ignored and padding dragging the embedding toward the origin.
///
/// `proj` is `embedding_projection` in the orientation safetensors reports it,
/// `[embedding_dim, hidden]`.
pub fn pool_project_normalize(
    tokens: &[f32],
    mask: &[bool],
    hidden: usize,
    proj: &[f32],
    embedding_dim: usize,
) -> Result<Vec<f32>> {
    if hidden == 0 || embedding_dim == 0 {
        bail!("embeddinggemma2: hidden and embedding_dim must be > 0");
    }
    if !tokens.len().is_multiple_of(hidden) {
        bail!(
            "embeddinggemma2: {} token values do not divide into {hidden}-wide states",
            tokens.len()
        );
    }
    let seq = tokens.len() / hidden;
    if mask.len() != seq {
        bail!(
            "embeddinggemma2: {} mask entries for {seq} tokens",
            mask.len()
        );
    }
    if proj.len() != embedding_dim * hidden {
        bail!(
            "embeddinggemma2: projection is {} values, expected {embedding_dim}x{hidden}",
            proj.len()
        );
    }
    let live = mask.iter().filter(|m| **m).count();
    if live == 0 {
        bail!("embeddinggemma2: every token is masked out, nothing to pool");
    }

    // Pool first, then project: the projection has no bias, so it commutes
    // with the mean, and this way it runs once instead of `seq` times.
    let mut pooled = vec![0f32; hidden];
    for (t, keep) in mask.iter().enumerate() {
        if !keep {
            continue;
        }
        for (acc, v) in pooled.iter_mut().zip(&tokens[t * hidden..(t + 1) * hidden]) {
            *acc += *v;
        }
    }
    let inv = 1.0 / live as f32;
    for v in pooled.iter_mut() {
        *v *= inv;
    }

    let mut out = vec![0f32; embedding_dim];
    for (o, row) in out.iter_mut().zip(proj.chunks_exact(hidden)) {
        *o = row.iter().zip(&pooled).map(|(w, x)| w * x).sum();
    }
    l2_normalize(&mut out);
    Ok(out)
}

/// Scale a vector to unit L2 norm, leaving an all-zero vector alone.
pub fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        let inv = 1.0 / n;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}
