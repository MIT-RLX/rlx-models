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

//! The Llama-3.2 backbone, entered through `inputs_embeds`.
//!
//! TADA adds nothing to the decoder stack itself — the whole extension lives in
//! what gets summed into the input embedding:
//!
//! ```text
//!   embed_tokens[token]
//! + acoustic_proj(latent)        512 → hidden, the previous frame's acoustics
//! + acoustic_mask_emb[has_latent]
//! + time_start_embed[frames_before]
//! + time_end_embed[frames_after]
//! ```
//!
//! So this module composes that vector host-side and hands it to
//! `rlx-llama32`'s flow, which supplies the decoder, the KV cache and every
//! backend. The LM head is never built: for text-to-speech the token sequence
//! is known up front, so only the hidden states matter, and skipping the head
//! drops a 128 k-wide matmul from every step.

use crate::config::TadaConfig;
use crate::prof::{self, trace};
use crate::weights::{Linear, TensorStore};
use anyhow::{Context, Result, bail};
use rlx_core::weight_map::WeightMap;
use rlx_ir::DType;
use rlx_llama32::{Llama32Config, Llama32Flow};
use rlx_runtime::{AotCache, CompileOptions, CompiledGraph, Device};
use std::collections::HashMap;
use std::sync::Arc;

/// On-disk LIR cache shared by every TADA graph of a given kind.
///
/// Compiling the backbone, the solver and the codec transformer from scratch
/// costs several seconds — more than the inference itself for a short
/// utterance. Caching the lowered LIR turns that into a backend-compile on
/// every run after the first.
pub fn aot_cache(kind: &str) -> AotCache {
    let dir = std::env::var_os("RLX_TADA_CACHE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("rlx_tada_aot"));
    AotCache::new(dir.join(kind))
}

/// The additive embeddings TADA layers on top of the token embedding.
pub struct InputEmbedder {
    /// The token table (128 256 × hidden) is left in the mapping and read a row
    /// at a time — an utterance touches a few dozen of its rows, and
    /// materializing it as f32 costs a gigabyte for nothing.
    store: Arc<TensorStore>,
    acoustic_proj: Linear,
    acoustic_mask_emb: Vec<f32>,
    time_start: Vec<f32>,
    time_end: Vec<f32>,
    hidden: usize,
    acoustic_dim: usize,
    num_time_classes: usize,
}

const EMBED_KEY: &str = "model.embed_tokens.weight";

impl InputEmbedder {
    pub fn load(store: Arc<TensorStore>, cfg: &TadaConfig) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let shape = store
            .shape(EMBED_KEY)
            .context("backbone: token embedding table")?;
        if shape != [cfg.vocab_size, hidden] {
            bail!(
                "embed_tokens is {shape:?}, expected [{}, {hidden}]",
                cfg.vocab_size
            );
        }
        Ok(Self {
            acoustic_proj: store.linear("acoustic_proj.weight")?,
            acoustic_mask_emb: store.get("acoustic_mask_emb.weight")?,
            time_start: store.get("time_start_embed.weight")?,
            time_end: store.get("time_end_embed.weight")?,
            store,
            hidden,
            acoustic_dim: cfg.acoustic_dim,
            num_time_classes: cfg.num_time_classes,
        })
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// `acoustic_proj(latent) + acoustic_mask_emb[mask] + time embeddings`.
    ///
    /// Split out from [`Self::with_token`] because the guidance branch reuses the
    /// exact same acoustic and time conditioning and differs only in its token
    /// — so this half is computed once per step, not twice.
    pub fn conditioning(
        &self,
        latent: &[f32],
        mask: u8,
        frames_before: u32,
        frames_after: u32,
    ) -> Result<Vec<f32>> {
        if latent.len() != self.acoustic_dim {
            bail!(
                "acoustic latent is {} wide, model wants {}",
                latent.len(),
                self.acoustic_dim
            );
        }
        let c = self.num_time_classes as u32;
        if frames_before >= c || frames_after >= c {
            bail!("frame gap {frames_before}/{frames_after} exceeds {c} time classes");
        }
        let mut out = vec![0f32; self.hidden];
        // acoustic_proj: [hidden, acoustic_dim] row-major, with bias.
        for (o, slot) in out.iter_mut().enumerate() {
            let row =
                &self.acoustic_proj.weight[o * self.acoustic_dim..(o + 1) * self.acoustic_dim];
            *slot = row.iter().zip(latent).map(|(w, x)| w * x).sum();
        }
        if let Some(b) = &self.acoustic_proj.bias {
            for (slot, v) in out.iter_mut().zip(b) {
                *slot += v;
            }
        }
        for (slot, v) in out
            .iter_mut()
            .zip(&self.acoustic_mask_emb[mask as usize * self.hidden..])
        {
            *slot += v;
        }
        for (slot, v) in out
            .iter_mut()
            .zip(&self.time_start[frames_before as usize * self.hidden..])
        {
            *slot += v;
        }
        for (slot, v) in out
            .iter_mut()
            .zip(&self.time_end[frames_after as usize * self.hidden..])
        {
            *slot += v;
        }
        Ok(out)
    }

    /// Add the token embedding for `token` to a conditioning vector, writing
    /// into `out` (`hidden` wide).
    pub fn with_token(&self, conditioning: &[f32], token: u32, out: &mut [f32]) -> Result<()> {
        self.store.row(EMBED_KEY, token as usize, out)?;
        for (o, c) in out.iter_mut().zip(conditioning) {
            *o += c;
        }
        Ok(())
    }
}

/// Round `n` up to the next multiple of `bucket`.
fn bucket_up(n: usize, bucket: usize) -> usize {
    n.div_ceil(bucket) * bucket
}

/// Prefill sequence lengths are rounded to this, so one compiled graph serves a
/// range of utterances. Padding goes at the *end* and attention is causal, so
/// the real positions cannot see it.
const PREFILL_BUCKET: usize = 32;
/// KV bucket granularity for decode. `bucket_decode_mask` already masks the
/// unused tail, so any bucket ≥ the step count is correct.
const DECODE_BUCKET: usize = 64;

/// KV bucket size that covers `steps`. Callers size their cache buffers from
/// this, so it has to be the same rounding the graph was compiled for.
pub fn decode_bucket(steps: usize) -> usize {
    bucket_up(steps, DECODE_BUCKET)
}

/// Compiled prefill + decode graphs for the backbone.
pub struct Backbone {
    cfg: Llama32Config,
    device: Device,
    /// Weights stay in the mapped checkpoint; a `WeightMap` is materialized
    /// only for the duration of a graph build and dropped immediately after.
    store: Arc<TensorStore>,
    names: Vec<String>,
    cache: AotCache,
    tag: String,
    prefill: Option<(usize, CompiledGraph)>,
    decode: Option<(usize, CompiledGraph)>,
    pub batch: usize,
}

impl Backbone {
    /// `store` must hold the `model.*` decoder tensors. The token embedding is
    /// excluded — [`InputEmbedder`] reads it from the same mapping, and leaving
    /// it in would make the flow build a tied LM head we never run.
    ///
    /// `tag` distinguishes checkpoints in the on-disk compile cache; it must
    /// change whenever the weights do.
    pub fn new(
        store: Arc<TensorStore>,
        cfg: Llama32Config,
        device: Device,
        batch: usize,
        tag: &str,
    ) -> Result<Self> {
        let mut names: Vec<String> = store
            .keys()
            .filter(|n| n.starts_with("model.layers.") || *n == "model.norm.weight")
            .map(str::to_string)
            .collect();
        if names.is_empty() {
            bail!("no `model.*` decoder tensors found in the checkpoint");
        }
        names.sort();
        Ok(Self {
            cfg,
            device,
            store,
            names,
            cache: aot_cache("backbone"),
            tag: tag.to_string(),
            prefill: None,
            decode: None,
            batch,
        })
    }

    pub fn kv_dim(&self) -> usize {
        self.cfg.num_key_value_heads * self.cfg.head_dim()
    }

    pub fn num_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }

    pub fn hidden(&self) -> usize {
        self.cfg.hidden_size
    }

    /// Materialize the decoder weights as f32 for one graph build.
    ///
    /// This is the peak-memory moment of the whole model, so it is deliberately
    /// short-lived: the map is consumed by the flow, copied into the graph's
    /// param arena, and dropped before the next build starts. The previous
    /// version kept a permanent copy *and* cloned it per build — three
    /// simultaneous copies of a 3.9 GB tensor set.
    fn weight_map(&self) -> Result<WeightMap> {
        let mut tensors = HashMap::with_capacity(self.names.len());
        for name in &self.names {
            let shape = self.store.shape(name)?.to_vec();
            tensors.insert(name.clone(), (self.store.get(name)?, shape));
        }
        Ok(WeightMap::from_tensors(tensors))
    }

    /// Prefill `seq` positions, returning `(hidden, per-layer K, per-layer V)`.
    ///
    /// `embeds` is `[batch, seq, hidden]` row-major. Internally the sequence is
    /// padded up to a `PREFILL_BUCKET` multiple so one compiled graph covers
    /// a range of utterance lengths; the padding sits past the end of a causal
    /// mask and cannot reach the real positions. Returned rows are strided by
    /// the *bucket*, which [`PrefillOut::row_stride`] reports.
    pub fn prefill(&mut self, embeds: &[f32], seq: usize) -> Result<PrefillOut> {
        let h = self.cfg.hidden_size;
        let want = self.batch * seq * h;
        if embeds.len() != want {
            bail!(
                "prefill embeds are {} values, expected {want}",
                embeds.len()
            );
        }
        let padded = bucket_up(seq, PREFILL_BUCKET);
        if self.prefill.as_ref().map(|(s, _)| *s) != Some(padded) {
            // Free the previous graph's arena before building the next.
            self.prefill = None;
            let key = format!("{}_prefill_b{}_s{padded}", self.tag, self.batch);
            let g = self.compile(&key, |cfg| {
                Llama32Flow::new(cfg)
                    .prefill()
                    .batch(self.batch)
                    .seq(padded)
                    .inputs_embeds()
                    .hidden_only()
                    .export_kv()
            })?;
            self.prefill = Some((padded, g));
        }
        let mut buf = vec![0f32; self.batch * padded * h];
        for b in 0..self.batch {
            buf[b * padded * h..b * padded * h + seq * h]
                .copy_from_slice(&embeds[b * seq * h..(b + 1) * seq * h]);
        }
        let g = &mut self.prefill.as_mut().expect("prefill graph").1;
        let outs = g.run(&[("inputs_embeds", &buf)]);
        Ok(PrefillOut {
            outs,
            row_stride: padded,
        })
    }

    /// Build a flow, compile it through the on-disk LIR cache, and bind params.
    fn compile<F>(&self, key: &str, make: F) -> Result<CompiledGraph>
    where
        F: for<'c> FnOnce(&'c Llama32Config) -> Llama32Flow<'c>,
    {
        let t = std::time::Instant::now();
        let mut wm = self.weight_map()?;
        let t_wm = t.elapsed();
        let t = std::time::Instant::now();
        let built = make(&self.cfg)
            .build(&mut wm)
            .with_context(|| format!("build TADA backbone flow {key}"))?;
        drop(wm);
        let (hir, mut params) = built.into_parts()?;
        let t_flow = t.elapsed();
        let t = std::time::Instant::now();
        let mut g = self
            .cache
            .compile_hir_cached(key, self.device, hir, &CompileOptions::default())
            .map_err(|e| anyhow::anyhow!("compile {key}: {e}"))?;
        let t_lir = t.elapsed();
        let t_bind = std::time::Instant::now();
        for (name, data) in params.drain() {
            g.set_param(&name, &data);
        }
        g.finalize_params();
        trace!(
            "    {key}: weights {t_wm:?} flow {t_flow:?} compile {t_lir:?} bind {:?} rss {} MB",
            t_bind.elapsed(),
            prof::rss_mb()
        );
        Ok(g)
    }

    /// One decode step at absolute position `pos`.
    ///
    /// `upper` is the KV bucket size the graph was compiled for; `past_k` /
    /// `past_v` are `[batch, upper, kv_dim]` f32 buffers that the caller keeps
    /// across steps, and the new row comes back at bucket index `upper`.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        embeds: &[f32],
        pos: usize,
        upper: usize,
        past_k: &[Vec<f32>],
        past_v: &[Vec<f32>],
    ) -> Result<DecodeOut> {
        if self.decode.as_ref().map(|(u, _)| *u) != Some(upper) {
            self.decode = None;
            let key = format!("{}_decode_b{}_u{upper}", self.tag, self.batch);
            let g = self.compile(&key, |cfg| {
                Llama32Flow::new(cfg)
                    .decode()
                    .batch(self.batch)
                    .past(upper)
                    .custom_mask()
                    .inputs_embeds()
                    .hidden_only()
                    .export_kv()
            })?;
            self.decode = Some((upper, g));
        }
        let layers = self.num_layers();
        let one = rlx_runtime::attn_mask::bucket_decode_mask(pos, upper);
        let mut mask = Vec::with_capacity(one.len() * self.batch);
        for _ in 0..self.batch {
            mask.extend_from_slice(&one);
        }
        let pos_v = [pos as f32];

        let k_names: Vec<String> = (0..layers).map(|l| format!("past_k_{l}")).collect();
        let v_names: Vec<String> = (0..layers).map(|l| format!("past_v_{l}")).collect();
        let mut inputs: Vec<(&str, &[f32])> = Vec::with_capacity(3 + 2 * layers);
        inputs.push(("inputs_embeds", embeds));
        inputs.push(("mask", &mask));
        inputs.push(("position", &pos_v));
        for l in 0..layers {
            inputs.push((k_names[l].as_str(), &past_k[l]));
            inputs.push((v_names[l].as_str(), &past_v[l]));
        }
        let g = &mut self.decode.as_mut().expect("decode graph").1;
        Ok(DecodeOut {
            outs: g.run(&inputs),
        })
    }

    /// Release the prefill graph's arena. Prefill and decode never run at the
    /// same time, and holding both doubles the backbone's resident footprint.
    pub fn drop_prefill(&mut self) {
        self.prefill = None;
    }

    /// Release every compiled graph. The codec runs after the last decode step
    /// and needs its own multi-gigabyte arena; holding the backbone's at the
    /// same time is what sets the process's peak.
    pub fn release(&mut self) {
        self.prefill = None;
        self.decode = None;
    }
}

/// Outputs of [`Backbone::prefill`]: hidden states then `(K, V)` per layer.
pub struct PrefillOut {
    outs: Vec<Vec<f32>>,
    row_stride: usize,
}

impl PrefillOut {
    /// Positions per batch row in every returned tensor — the *padded* bucket,
    /// not the requested sequence length.
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }
    pub fn hidden(&self) -> &[f32] {
        &self.outs[0]
    }
    pub fn k(&self, layer: usize) -> &[f32] {
        &self.outs[1 + 2 * layer]
    }
    pub fn v(&self, layer: usize) -> &[f32] {
        &self.outs[2 + 2 * layer]
    }
}

/// Outputs of [`Backbone::step`], same ordering as [`PrefillOut`].
pub struct DecodeOut {
    outs: Vec<Vec<f32>>,
}

impl DecodeOut {
    pub fn hidden(&self) -> &[f32] {
        &self.outs[0]
    }
    pub fn k(&self, layer: usize) -> &[f32] {
        &self.outs[1 + 2 * layer]
    }
    pub fn v(&self, layer: usize) -> &[f32] {
        &self.outs[2 + 2 * layer]
    }
}

/// `DType` re-export so callers do not need a direct `rlx-ir` dependency.
pub const F32: DType = DType::F32;
