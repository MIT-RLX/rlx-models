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

//! `google/embeddinggemma-2` text and audio embeddings, end to end.
//!
//! ```text
//!   text ids ──► embed_tokens · √512 ────────────────────────┐
//!   audio ──► log-mel ──► audio tower ──► embed_audio ──────┤ spliced at <|audio|>
//!                       (12 Conformer layers)  (RMSNorm, 1536→512)
//!                                                           ▼
//!        projection-only PLE ──► Gemma 4 trunk, bidirectional ──► norm
//!                                       ──► embedding_projection (768) ──► mean ──► L2
//! ```
//!
//! The trunk is [`crate::builder`]'s Gemma 4 graph in `inputs_embeds` mode with
//! additive attention biases instead of a causal mask: the model is an encoder.
//! Full-attention layers see every token; sliding layers see every token within
//! 512 positions on either side, through a second bias once the input is longer
//! than that. Graphs are built at the exact sequence length and kept, so a batch
//! of same-length inputs compiles once. Audio is cut at 30 s
//! ([`MAX_AUDIO_SAMPLES`]), as the reference feature extractor cuts it.
//!
//! Per-layer embeddings are *projection-only* here (no token table):
//! `rmsnorm(per_layer_model_projection(x) · 512^-0.5)` per layer slice,
//! computed by a small graph and fed to the trunk as `per_layer_inputs`.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use rlx_core::flow_util::compile_graph_gemma_prefill_with_params;
use rlx_core::safetensors_checkpoint::SafetensorsCheckpoint;
use rlx_core::weight_loader::{ArcCacheLoader, ArcF32Tensor};
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{CompiledGraph, Device};

use crate::embedding_gemma2::{
    EmbeddingGemma2Config, EmbeddingGemma2Names, pool_project_normalize,
};
use crate::gemma4_audio::{AudioConfig, audio_block_mask, audio_rel_pos, build_audio_tower_taps};
use crate::gemma4_audio_features::{AudioFeatureConfig, AudioFeatureExtractor};

/// The prompt the sentence-transformers configuration uses for queries; the
/// null-space head was fitted on audio embedded after it.
pub const QUERY_PROMPT: &str = "task: search result | query: ";

/// Longest clip the reference feature extractor keeps (`max_length` 480 000
/// samples, 30 s at 16 kHz); it cuts the rest, and so does this.
pub const MAX_AUDIO_SAMPLES: usize = 480_000;

/// Additive attention bias of a blocked key (the multimodal masks' value).
const MASKED: f32 = -1e4;

/// An audio clip's embedding and the audio-tower states asked for.
#[derive(Debug, Clone)]
pub struct AudioEmbedding {
    /// The model's own embedding, `[embedding_dim]`, unit norm.
    pub embedding: Vec<f32>,
    /// Audio-tower hidden states after each requested layer, `[seq · 1024]`.
    pub taps: Vec<Vec<f32>>,
    /// Audio soft tokens (rows of each tap).
    pub audio_tokens: usize,
}

/// Compiled graphs kept per kind (trunk, PLE, audio tower). Each is built for
/// one input length and holds its own arena and uploaded weights, so an
/// unbounded cache grows with every new clip length until the device runs out.
const CACHED_GRAPHS: usize = 2;

/// Drop one entry of `m` when it is full, before a new one goes in.
fn make_room<K: Clone + Eq + std::hash::Hash, V>(m: &mut HashMap<K, V>) {
    if m.len() >= CACHED_GRAPHS
        && let Some(k) = m.keys().next().cloned()
    {
        m.remove(&k);
    }
}

/// A loaded `embeddinggemma-2` (text + audio; the vision tower is not loaded).
pub struct EmbeddingGemma2 {
    cfg: EmbeddingGemma2Config,
    audio_cfg: AudioConfig,
    features: AudioFeatureExtractor,
    /// Checkpoint tensors by their own names, shared by every graph build.
    weights: HashMap<String, ArcF32Tensor>,
    device: Device,
    trunks: HashMap<usize, CompiledGraph>,
    ples: HashMap<usize, CompiledGraph>,
    towers: HashMap<(usize, Vec<usize>), (CompiledGraph, usize)>,
}

impl EmbeddingGemma2 {
    /// Load `config.json`, `processor_config.json` and `model.safetensors`
    /// from `dir`.
    pub fn open(dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        let dir = dir.as_ref();
        let cfg = EmbeddingGemma2Config::from_hf_path(dir.join("config.json"))?;
        if !cfg.has_audio {
            bail!("embeddinggemma2: {dir:?} has no audio tower");
        }
        let features = match std::fs::read_to_string(dir.join("processor_config.json")) {
            Ok(raw) => AudioFeatureConfig::from_json(&serde_json::from_str(&raw)?),
            Err(_) => AudioFeatureConfig::default(),
        };
        let audio_cfg = AudioConfig {
            use_clipped_linears: cfg.audio_use_clipped_linears,
            ..AudioConfig::default()
        };
        // The vision tower is a third of the file and unused here.
        let ckpt = SafetensorsCheckpoint::open_file(&dir.join("model.safetensors"))?;
        let want: HashSet<String> = ckpt
            .keys()
            .filter(|k| !k.starts_with("vision_tower.") && !k.starts_with("embed_vision."))
            .map(str::to_string)
            .collect();
        let mut map = ckpt
            .load_selected(&want)
            .context("load embeddinggemma-2 weights")?;
        let weights = map
            .take_all()
            .into_iter()
            .map(|(k, (v, s))| (k, (Arc::new(v), s)))
            .collect();
        Ok(Self {
            cfg,
            audio_cfg,
            features: AudioFeatureExtractor::new(features),
            weights,
            device,
            trunks: HashMap::new(),
            ples: HashMap::new(),
            towers: HashMap::new(),
        })
    }

    pub fn config(&self) -> &EmbeddingGemma2Config {
        &self.cfg
    }

    fn tensor(&self, name: &str) -> Result<&ArcF32Tensor> {
        self.weights
            .get(name)
            .ok_or_else(|| anyhow!("embeddinggemma2: no tensor {name}"))
    }

    /// The `[embedding_dim, hidden]` output projection, whose range is the
    /// only part of the embedding space text can reach.
    pub fn embedding_projection(&self) -> Result<&[f32]> {
        Ok(&self.tensor("language_model.embedding_projection.weight")?.0)
    }

    /// Embed token ids (with `<bos>` / `<eos>` already in place, as the
    /// tokenizer produces them).
    pub fn embed_ids(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        self.embed_sequence(ids, &[])
    }

    /// The token ids of `prompt_ids` (`<bos>` + prompt, no `<eos>`) followed
    /// by an audio clip of `audio_tokens` soft tokens, as the processor lays
    /// them out: `<|audio>`, the placeholders, `<audio|>`, `<eos>`.
    pub fn audio_prompt_ids(&self, prompt_ids: &[u32], audio_tokens: usize) -> Vec<u32> {
        let mut ids = prompt_ids.to_vec();
        ids.push(self.cfg.boa_token_id);
        ids.extend(std::iter::repeat_n(self.cfg.audio_token_id, audio_tokens));
        ids.push(self.cfg.eoa_token_id);
        ids.push(self.cfg.eos_token_id);
        ids
    }

    /// Embed 16 kHz mono `samples` after `prompt_ids` (`<bos>` + prompt, no
    /// `<eos>`), returning the audio-tower states after the layers in `taps`
    /// too (1-based; `[2, 6]` for the null-space head).
    pub fn embed_audio(
        &mut self,
        prompt_ids: &[u32],
        samples: &[f32],
        taps: &[usize],
    ) -> Result<AudioEmbedding> {
        let (feats, frames) = self.audio_features(samples);
        if frames == 0 {
            bail!(
                "embeddinggemma2: {} samples is too short for one mel frame",
                samples.len()
            );
        }
        let (outs, seq) = self.run_tower(&feats, frames, taps)?;
        let soft = self.embed_audio_tokens(&outs[0], seq)?;
        let ids = self.audio_prompt_ids(prompt_ids, seq);
        let embedding = self.embed_sequence(&ids, &soft)?;
        Ok(AudioEmbedding {
            embedding,
            taps: outs[1..].to_vec(),
            audio_tokens: seq,
        })
    }

    /// The log-mel features of `samples` (valid frames only), after the
    /// [`MAX_AUDIO_SAMPLES`] cut.
    pub fn audio_features(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        self.features
            .extract(&samples[..samples.len().min(MAX_AUDIO_SAMPLES)])
    }

    /// Run the audio tower over `frames` log-mel frames: its output
    /// `[seq · 1536]`, then the state after each layer in `taps`.
    pub fn run_tower(
        &mut self,
        feats: &[f32],
        frames: usize,
        taps: &[usize],
    ) -> Result<(Vec<Vec<f32>>, usize)> {
        let key = (frames, taps.to_vec());
        if !self.towers.contains_key(&key) {
            make_room(&mut self.towers);
            let mut w = EmbeddingGemma2Names::new(ArcCacheLoader::new(&self.weights));
            let (g, params, seq) = build_audio_tower_taps(
                &self.audio_cfg,
                &mut w,
                1,
                frames,
                self.audio_cfg.layers,
                true,
                taps,
            )?;
            let c = crate::gemma4_audio::compile_audio_tower(self.device, g, params)?;
            self.towers.insert(key.clone(), (c, seq));
        }
        let (c, seq) = self.towers.get_mut(&key).expect("just built");
        let seq = *seq;
        let rel = audio_rel_pos(&self.audio_cfg);
        let (mask, _) = audio_block_mask(&self.audio_cfg, seq);
        let outs = c.run(&[
            ("audio_feats", feats),
            ("audio_rel_pos", &rel),
            ("audio_mask", &mask),
        ]);
        Ok((outs, seq))
    }

    /// `embed_audio`: RMSNorm without scale, then `[hidden, out_dims]`.
    pub fn embed_audio_tokens(&self, tower_out: &[f32], seq: usize) -> Result<Vec<f32>> {
        let (w, shape) = self.tensor("embed_audio.embedding_projection.weight")?;
        let (h, d) = (shape[0], shape[1]);
        if tower_out.len() != seq * d {
            bail!(
                "embeddinggemma2: tower output is {} values, expected {seq}x{d}",
                tower_out.len()
            );
        }
        let eps = self.audio_cfg.eps;
        let mut out = vec![0f32; seq * h];
        let mut normed = vec![0f32; d];
        for t in 0..seq {
            let x = &tower_out[t * d..(t + 1) * d];
            let ms = x.iter().map(|v| v * v).sum::<f32>() / d as f32;
            let inv = (ms + eps).powf(-0.5);
            for (n, v) in normed.iter_mut().zip(x) {
                *n = v * inv;
            }
            for (o, row) in out[t * h..(t + 1) * h].iter_mut().zip(w.chunks_exact(d)) {
                *o = row.iter().zip(&normed).map(|(a, b)| a * b).sum();
            }
        }
        Ok(out)
    }

    /// The trunk over `ids`, with `audio` soft tokens (`[n · hidden]`) taking
    /// the audio placeholders in order; mean-pooled, projected, normalized.
    fn embed_sequence(&mut self, ids: &[u32], audio: &[f32]) -> Result<Vec<f32>> {
        let h = self.cfg.text.hidden_size;
        let seq = ids.len();
        if seq == 0 {
            bail!("embeddinggemma2: nothing to embed");
        }
        let n_audio = ids
            .iter()
            .filter(|&&t| t == self.cfg.audio_token_id)
            .count();
        if n_audio * h != audio.len() {
            bail!(
                "embeddinggemma2: {n_audio} audio placeholders but {} soft-token values",
                audio.len()
            );
        }
        let scale = (h as f32).sqrt();
        let embed = &self.tensor("language_model.embed_tokens.weight")?.0;
        // `prescale` feeds the builder, which multiplies by √hidden; `scaled`
        // is what HF's trunk sees and what PLE projects.
        let mut prescale = vec![0f32; seq * h];
        let mut scaled = vec![0f32; seq * h];
        let mut a = 0;
        for (t, &id) in ids.iter().enumerate() {
            let (pre, sc) = (
                &mut prescale[t * h..(t + 1) * h],
                &mut scaled[t * h..(t + 1) * h],
            );
            if id == self.cfg.audio_token_id {
                let row = &audio[a * h..(a + 1) * h];
                a += 1;
                for i in 0..h {
                    pre[i] = row[i] / scale;
                    sc[i] = row[i];
                }
            } else {
                let row = embed
                    .get(id as usize * h..(id as usize + 1) * h)
                    .ok_or_else(|| anyhow!("embeddinggemma2: token id {id} past the vocabulary"))?;
                pre.copy_from_slice(row);
                for i in 0..h {
                    sc[i] = row[i] * scale;
                }
            }
        }
        let ple = self.per_layer_inputs(&scaled, seq)?;
        let hidden = self.run_trunk(&prescale, &ple, seq)?;
        let proj = &self.tensor("language_model.embedding_projection.weight")?.0;
        pool_project_normalize(&hidden, &vec![true; seq], h, proj, self.cfg.embedding_dim)
    }

    /// Projection-only PLE: `[seq · layers · ple]`.
    fn per_layer_inputs(&mut self, scaled: &[f32], seq: usize) -> Result<Vec<f32>> {
        if !self.ples.contains_key(&seq) {
            make_room(&mut self.ples);
            let t = &self.cfg.text;
            let (h, layers, pw) = (t.hidden_size, t.num_hidden_layers, t.ple_width());
            let mut g = Graph::new("embeddinggemma2_ple");
            let mut params = HashMap::new();
            let x = g.input("x", Shape::new(&[1, seq, h], DType::F32));
            let wname = "language_model.ple.per_layer_model_projection.weight";
            let mut w = ArcCacheLoader::new(&self.weights);
            let (wt, wshape) =
                rlx_core::weight_loader::WeightLoader::take_transposed(&mut w, wname)?;
            let wid = g.param(wname, Shape::new(&wshape, DType::F32));
            params.insert(wname.to_string(), wt);
            let mut y = g.mm(x, wid);
            let sname = "ple.projection_scale";
            let sid = g.param(sname, Shape::new(&[1], DType::F32));
            params.insert(sname.to_string(), vec![(h as f32).powf(-0.5)]);
            y = g.mul(y, sid);
            let y = g.reshape_(y, vec![1, seq as i64, layers as i64, pw as i64]);
            let nname = "language_model.ple.per_layer_projection_norm.weight";
            let gid = g.param(nname, Shape::new(&[pw], DType::F32));
            params.insert(nname.to_string(), self.tensor(nname)?.0.as_ref().clone());
            let bname = "ple.norm_beta";
            let bid = g.param(bname, Shape::new(&[pw], DType::F32));
            params.insert(bname.to_string(), vec![0.0; pw]);
            let n = g.rms_norm(y, gid, bid, t.rms_norm_eps as f32);
            let out = g.reshape_(n, vec![1, seq as i64, (layers * pw) as i64]);
            g.set_outputs(vec![out]);
            let c = compile_graph_gemma_prefill_with_params(self.device, g, params)?;
            self.ples.insert(seq, c);
        }
        let c = self.ples.get_mut(&seq).expect("just built");
        Ok(c.run(&[("x", scaled)]).swap_remove(0))
    }

    /// The trunk up to `language_model.norm`: `[seq · hidden]`.
    ///
    /// Attention is bidirectional: full-attention layers see every token,
    /// sliding layers every token within `sliding_window` positions (HF's
    /// `abs(q - k) <= sliding_window`). Up to `sliding_window + 1` tokens the
    /// two are the same and one all-zero bias serves both; past that the
    /// sliding layers get their own.
    fn run_trunk(&mut self, prescale: &[f32], ple: &[f32], seq: usize) -> Result<Vec<f32>> {
        let window = self.cfg.text.sliding_window.unwrap_or(usize::MAX);
        let windowed = seq > window.saturating_add(1);
        if !self.trunks.contains_key(&seq) {
            make_room(&mut self.trunks);
            let mut marker = crate::gemma4_e2b_mm::embed_lazy_marker_with(&self.cfg.text, true);
            if windowed {
                let (_, scheme, _) = marker["__media_bias__"].clone();
                marker.insert(
                    "__sliding_bias__".to_string(),
                    (crate::builder::PackedSrc::F32, scheme, vec![]),
                );
            }
            let mut packed = HashMap::new();
            let mut w = EmbeddingGemma2Names::new(ArcCacheLoader::new(&self.weights));
            let (g, params) = crate::builder::build_gemma_graph_sized_packed_ext(
                &self.cfg.text,
                &mut w,
                1,
                seq,
                /*with_lm_head*/ false,
                /*last_token_from_input*/ false,
                /*with_kv_outputs*/ false,
                &mut packed,
                Some(&marker),
                None,
            )?;
            // The builder records dense projections as f32 sentinels; real
            // packed bytes would mean a quantized checkpoint this path cannot run.
            if let Some(k) = packed
                .iter()
                .find(|(_, (src, _, _))| !src.is_f32())
                .map(|(k, _)| k)
            {
                bail!("embeddinggemma2: {k} is packed; this path runs dense checkpoints only");
            }
            let c = crate::gemma_e2b::compile_e2b_prefill(self.device, g, params)?;
            self.trunks.insert(seq, c);
        }
        let heads = self.cfg.text.num_attention_heads;
        let bias = vec![0f32; heads * seq * seq];
        let sliding = windowed.then(|| {
            let mut b = vec![0f32; heads * seq * seq];
            for (i, v) in b.iter_mut().enumerate() {
                let (q, k) = ((i / seq) % seq, i % seq);
                if q.abs_diff(k) > window {
                    *v = MASKED;
                }
            }
            b
        });
        let c = self.trunks.get_mut(&seq).expect("just built");
        let mut inputs: Vec<(&str, &[f32])> = vec![
            ("input_embeddings", prescale),
            ("per_layer_inputs", ple),
            ("attn_bias", &bias),
        ];
        if let Some(sb) = &sliding {
            inputs.push(("attn_bias_sliding", sb));
        }
        let outs = c.run(&inputs);
        outs.into_iter()
            .next()
            .ok_or_else(|| anyhow!("embeddinggemma2: the trunk produced no output"))
    }
}
