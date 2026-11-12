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

//! Staged execution: host-side embedding → encoder (+ cross K/V) → KV-cached
//! decode steps → host LM head.
//!
//! Memory is the design constraint. On CPU every compiled graph owns its
//! weights, so the runner keeps **one** encoder graph and **one** decode-step
//! graph resident, and none of the rest on the host: graphs are built by
//! streaming tensors from the checkpoint on disk ([`Checkpoint`]). The host
//! holds only the tied `[vocab, d]` table, which is both the embedding and
//! the LM head. Source lengths are padded to a few buckets (masked keys) and
//! the step graph's KV capacity covers a whole generation, so a run of
//! sentences reuses the same two graphs.

use crate::config::NllbConfig;
use crate::flow;
use crate::generate::GenerateConfig;
use crate::tokenizer::NllbTokenizer;
use crate::weight_source::{Checkpoint, ShapeRecorder};
use crate::weights::lang as lk;
use anyhow::{Result, anyhow, bail, ensure};
use rlx_core::autoregressive::run_bucketed_kv_decode_keyed_batched;
use rlx_core::flow_bridge::compile_options_for_profile;
use rlx_core::flow_util::{compile_built, graph_from_built};
use rlx_core::gpu_kv::{
    GpuKvBinding, KvLayout, device_supports_resident_kv_row_feed, run_bucketed_kv_decode_gpu,
};
use rlx_core::weight_map::WeightMap;
use rlx_flow::CompileProfile;
use rlx_runtime::attn_mask::bucket_decode_mask;
use rlx_runtime::compile_cache::{BucketedCompileCache, CacheRunInput};
use rlx_runtime::{CompiledGraph, Device, LayerKvCache, Session};
use std::path::Path;

/// Encoder source-length buckets.
///
/// Every sentence the decode step sees is padded to at least
/// [`MIN_STEP_SRC`] rows (pad keys are masked, and the step's cross-attention
/// skips them, so padding there is nearly free); one step graph serves all of
/// them. The encoder is different: it computes every row, and at 128 rows a
/// 10-token sentence spent ~50 ms on padding, most of its whole translation.
/// So the first bucket is 64 rows (~25 ms): small enough for short input,
/// large enough that nearly every sentence lands in it and one encoder graph
/// serves a whole document. Longer sources get a second encoder graph that
/// shares the first's weight pages (one param-sharing scope); note macOS
/// per-process counters (RSS, `phys_footprint`) charge shared pages once per
/// mapping, so they show that second graph as a full copy even though system
/// memory does not drop.
const SRC_BUCKETS: [usize; 4] = [64, 128, 512, 1024];

/// Fewest source rows a decode-step graph is built for.
const MIN_STEP_SRC: usize = 128;

/// Encoder graphs kept resident (most recent first).
const RESIDENT_ENCODERS: usize = 2;

/// Smallest bucket holding `n` source tokens.
fn src_bucket(n: usize) -> usize {
    SRC_BUCKETS.iter().copied().find(|&b| b >= n).unwrap_or(n)
}

/// Distinct param-sharing scope per loaded model, so graphs of two different
/// checkpoints in one process never share weight pages.
static NEXT_MODEL_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// KV capacity (past rows) of the step graph for `max_len` decoder tokens:
/// the last fed token sits at `max_len - 2`. Powers of two from 64, so the
/// usual `max_new_tokens` all land on one graph.
fn step_capacity(max_len: usize) -> usize {
    max_len.saturating_sub(1).next_power_of_two().max(64)
}

/// Encoder output for one source sentence: what decoding needs from it.
pub struct Encoded {
    /// Real source tokens.
    pub src_len: usize,
    /// Rows of the cross K/V a decode step reads: the encoder bucket, but at
    /// least 128, so one step graph serves the short buckets (pad rows are
    /// zero and masked).
    pub bucket: usize,
    /// Encoder output, real rows only (`[src_len · d]`).
    pub hidden: Vec<f32>,
    /// Per decoder layer, cross-attention K and V (`[bucket · d]`).
    cross_k: Vec<Vec<f32>>,
    cross_v: Vec<Vec<f32>>,
}

/// Distinct id per [`DecodeState`], so the step graph knows whose source
/// parameters it holds.
static NEXT_STATE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// KV-cached decoding state for `batch` sequences of one source.
pub struct DecodeState {
    id: u64,
    /// A single sequence keeps its self-attention K/V resident in the step
    /// graph (only the new row moves each step); a batch (beam search, whose
    /// rows get reordered) keeps it on the host in `kv`.
    resident: Option<GpuKvBinding>,
    batch: usize,
    d: usize,
    bucket: usize,
    capacity: usize,
    kv: LayerKvCache,
    /// Cross K/V replicated per sequence (`[batch · bucket · d]`).
    cross_k: Vec<Vec<f32>>,
    cross_v: Vec<Vec<f32>>,
    src_mask: Vec<f32>,
}

impl DecodeState {
    /// Tokens already decoded (the next one goes to this position).
    pub fn len(&self) -> usize {
        self.kv.past_len
    }

    pub fn is_empty(&self) -> bool {
        self.kv.past_len == 0
    }

    /// Reorder the batch so sequence `i` continues from old sequence
    /// `parents[i]` (beam search).
    pub fn reorder(&mut self, parents: &[usize]) {
        assert_eq!(parents.len(), self.batch);
        assert!(self.resident.is_none(), "reorder needs the host-side cache");
        // Each layer buffer is `[batch, past, d]`.
        let row = self.kv.past_len * self.d;
        let gather = |buf: &Vec<f32>| -> Vec<f32> {
            let mut out = Vec::with_capacity(buf.len());
            for &p in parents {
                out.extend_from_slice(&buf[p * row..(p + 1) * row]);
            }
            out
        };
        for l in 0..self.kv.layers_k.len() {
            self.kv.layers_k[l] = gather(&self.kv.layers_k[l]);
            self.kv.layers_v[l] = gather(&self.kv.layers_v[l]);
        }
    }
}

/// The resident decode-step graph and the geometry it was compiled for.
struct StepSlot {
    /// [`DecodeState::id`] whose source mask and cross K/V are bound.
    bound: Option<u64>,
    batch: usize,
    bucket: usize,
    capacity: usize,
    cache: BucketedCompileCache,
}

/// Loaded NLLB / M2M100 model.
pub struct NllbModel {
    cfg: NllbConfig,
    device: Device,
    weights: Checkpoint,
    /// Host `model.shared.weight` `[vocab, d]` — embedding + tied LM head.
    shared_table: Vec<f32>,
    /// Host `final_logits_bias` `[vocab]` (empty if absent).
    final_logits_bias: Vec<f32>,
    /// Sinusoidal positions `[max_positions + 2, d]`.
    positions: Vec<f32>,
    /// Encoder graphs by bucket, most recent first, sharing weight pages.
    encoders: Vec<(usize, CompiledGraph)>,
    /// Param-sharing scope of this model's encoder graphs.
    share_scope: String,
    step: Option<StepSlot>,
    tokenizer: Option<NllbTokenizer>,
}

impl NllbModel {
    pub fn config(&self) -> &NllbConfig {
        &self.cfg
    }
    pub fn device(&self) -> Device {
        self.device
    }
    pub fn tokenizer(&self) -> Option<&NllbTokenizer> {
        self.tokenizer.as_ref()
    }

    /// Load from a checkpoint directory or file: safetensors (file or shards),
    /// or the `pytorch_model.bin` that `facebook/nllb-200-*` actually publish.
    /// Only the tied embedding table is read now; the rest stays on disk
    /// until a graph is built. `tokenizer.json` beside the weights is loaded.
    pub fn load(weights_path: &Path, cfg: NllbConfig, device: Device) -> Result<Self> {
        let weights = Checkpoint::open(weights_path)?;
        let dir = if weights_path.is_dir() {
            Some(weights_path)
        } else {
            weights_path.parent()
        };
        let tokenizer = match dir.map(|d| d.join("tokenizer.json")) {
            Some(p) if p.is_file() => Some(NllbTokenizer::from_file(&p)?),
            _ => None,
        };
        Self::from_checkpoint(weights, cfg, device, tokenizer)
    }

    /// Construct from an in-memory [`WeightMap`] (tests / synthetic checkpoints).
    pub fn from_weight_map(
        weights: WeightMap,
        cfg: NllbConfig,
        device: Device,
        tokenizer: Option<NllbTokenizer>,
    ) -> Result<Self> {
        Self::from_checkpoint(Checkpoint::Memory(weights), cfg, device, tokenizer)
    }

    fn from_checkpoint(
        mut weights: Checkpoint,
        cfg: NllbConfig,
        device: Device,
        tokenizer: Option<NllbTokenizer>,
    ) -> Result<Self> {
        use rlx_flow::WeightSource;
        rlx_core::validate_standard_device("nllb", device)?;
        let shared_key = [lk::SHARED, lk::ENC_EMBED_TOKENS]
            .into_iter()
            .find(|k| weights.has(k))
            .ok_or_else(|| {
                anyhow!(
                    "nllb: missing shared embedding (`{}` or `{}`)",
                    lk::SHARED,
                    lk::ENC_EMBED_TOKENS
                )
            })?;
        let shared_table = match &mut weights {
            // Move it out: the map must not keep a second 1 GB copy.
            Checkpoint::Memory(m) => m.take(shared_key)?,
            other => other.read(shared_key)?,
        };
        let (shared_table, shape) = shared_table;
        if shape.len() != 2 || shape[0] != cfg.vocab_size || shape[1] != cfg.d_model {
            bail!(
                "nllb: shared embedding shape {shape:?} != [{}, {}]",
                cfg.vocab_size,
                cfg.d_model
            );
        }
        let final_logits_bias = if weights.has(lk::FINAL_LOGITS_BIAS) {
            weights.read(lk::FINAL_LOGITS_BIAS)?.0
        } else {
            Vec::new()
        };
        let positions =
            cfg.sinusoidal_positions(cfg.max_position_embeddings + NllbConfig::POS_OFFSET);
        Ok(Self {
            cfg,
            device,
            weights,
            shared_table,
            final_logits_bias,
            positions,
            encoders: Vec::new(),
            share_scope: format!(
                "rlx-nllb-{}",
                NEXT_MODEL_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ),
            step: None,
            tokenizer,
        })
    }

    /// Attach / replace the tokenizer (e.g. after loading weights from a file).
    pub fn set_tokenizer(&mut self, tokenizer: NllbTokenizer) {
        self.tokenizer = Some(tokenizer);
    }

    /// Token embeddings `× embed_scale` (`[ids · d]`).
    pub fn embed_text(&self, token_ids: &[u32]) -> Vec<f32> {
        let mut out = vec![0f32; token_ids.len() * self.cfg.d_model];
        for (i, &tok) in token_ids.iter().enumerate() {
            self.embed_into(tok, None, &mut out[i * self.cfg.d_model..]);
        }
        out
    }

    /// `out[..d] = shared[tok] · scale (+ position row of decoder/encoder
    /// index `pos`)`.
    fn embed_into(&self, tok: u32, pos: Option<usize>, out: &mut [f32]) {
        let d = self.cfg.d_model;
        let scale = self.cfg.embed_scale();
        let emb = &self.shared_table[tok as usize * d..(tok as usize + 1) * d];
        match pos {
            Some(p) => {
                let row = (p + self.cfg.pad_token_id as usize + 1) * d;
                let pe = &self.positions[row..row + d];
                for j in 0..d {
                    out[j] = emb[j] * scale + pe[j];
                }
            }
            None => {
                for j in 0..d {
                    out[j] = emb[j] * scale;
                }
            }
        }
    }

    /// Encode source token ids: encoder output plus every decoder layer's
    /// cross K/V, at a padded bucket length with the pad keys masked.
    pub fn encode_tokens(&mut self, token_ids: &[u32]) -> Result<Encoded> {
        let d = self.cfg.d_model;
        let n = token_ids.len();
        ensure!(
            n >= 1 && n <= self.cfg.max_position_embeddings,
            "nllb: source length {n} outside 1..={}",
            self.cfg.max_position_embeddings
        );
        let bucket = src_bucket(n);
        let mut x = vec![0f32; bucket * d];
        for (i, &tok) in token_ids.iter().enumerate() {
            self.embed_into(tok, Some(i), &mut x[i * d..(i + 1) * d]);
        }
        let mask: Vec<f32> = (0..bucket).map(|i| (i < n) as u8 as f32).collect();
        if let Some(i) = self.encoders.iter().position(|(b, _)| *b == bucket) {
            let hit = self.encoders.remove(i);
            self.encoders.insert(0, hit);
        } else {
            // Make room first, so no more than the cap coexist while building.
            self.encoders.truncate(RESIDENT_ENCODERS - 1);
            let mut rec = ShapeRecorder::new(&self.weights);
            let built = flow::build_encoder_cross_built(&self.cfg, &mut rec, bucket)?;
            let taken = rec.into_taken();
            let options = compile_options_for_profile(built.profile(), self.device);
            let (graph, _) = graph_from_built(built)?;
            let device = self.device;
            let mut g = rlx_runtime::with_param_sharing_scope(&self.share_scope, || {
                Session::new(device).compile_with(graph, &options)
            });
            self.weights.stream_into(&mut g, &taken)?;
            self.encoders.insert(0, (bucket, g));
        }
        let g = &mut self.encoders[0].1;
        let mut out = g
            .run(&[("x", x.as_slice()), ("mask", mask.as_slice())])
            .into_iter();
        let mut hidden = out
            .next()
            .ok_or_else(|| anyhow!("encoder graph produced no output"))?;
        hidden.truncate(n * d);
        // The decode step reads at least `MIN_STEP_SRC` rows: zero-pad.
        let rows = bucket.max(MIN_STEP_SRC);
        let (mut cross_k, mut cross_v) = (Vec::new(), Vec::new());
        for _ in 0..self.cfg.decoder_layers {
            let mut k = out.next().ok_or_else(|| anyhow!("missing cross K"))?;
            let mut v = out.next().ok_or_else(|| anyhow!("missing cross V"))?;
            k.resize(rows * d, 0.0);
            v.resize(rows * d, 0.0);
            cross_k.push(k);
            cross_v.push(v);
        }
        Ok(Encoded {
            src_len: n,
            bucket: rows,
            hidden,
            cross_k,
            cross_v,
        })
    }

    /// Fresh decoding state for `batch` sequences of `enc`, able to hold
    /// `max_len` decoder tokens.
    pub fn start_decode(&self, enc: &Encoded, batch: usize, max_len: usize) -> Result<DecodeState> {
        ensure!(batch >= 1, "nllb: decode batch must be ≥ 1");
        ensure!(
            max_len <= self.cfg.max_position_embeddings,
            "nllb: {max_len} decoder tokens exceed {} positions",
            self.cfg.max_position_embeddings
        );
        let rep =
            |bufs: &[Vec<f32>]| -> Vec<Vec<f32>> { bufs.iter().map(|b| b.repeat(batch)).collect() };
        let one: Vec<f32> = (0..enc.bucket)
            .map(|i| (i < enc.src_len) as u8 as f32)
            .collect();
        let n = self.cfg.decoder_layers;
        Ok(DecodeState {
            id: NEXT_STATE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            // The step returns `concat(past, new)` per layer.
            resident: (batch == 1 && device_supports_resident_kv_row_feed(self.device))
                .then(|| GpuKvBinding::new(KvLayout::AppendLast, 1)),
            batch,
            d: self.cfg.d_model,
            bucket: enc.bucket,
            capacity: step_capacity(max_len),
            kv: LayerKvCache {
                past_len: 0,
                layers_k: vec![Vec::new(); n],
                layers_v: vec![Vec::new(); n],
                layers_kv_base: vec![0; n],
            },
            cross_k: rep(&enc.cross_k),
            cross_v: rep(&enc.cross_v),
            src_mask: one.repeat(batch),
        })
    }

    /// Feed one token per sequence; returns the final decoder hidden
    /// `[batch · d]` for those tokens and appends them to the KV cache.
    pub fn step_hidden(&mut self, state: &mut DecodeState, tokens: &[u32]) -> Result<Vec<f32>> {
        let d = self.cfg.d_model;
        let n = self.cfg.decoder_layers;
        let (batch, past) = (state.batch, state.kv.past_len);
        ensure!(
            tokens.len() == batch,
            "nllb: {} tokens for batch {batch}",
            tokens.len()
        );
        ensure!(
            past <= state.capacity,
            "nllb: decode position {past} beyond capacity {}",
            state.capacity
        );
        let mut x = vec![0f32; batch * d];
        for (b, &tok) in tokens.iter().enumerate() {
            self.embed_into(tok, Some(past), &mut x[b * d..(b + 1) * d]);
        }
        let mask = bucket_decode_mask(past, state.capacity).repeat(batch);

        let want = (batch, state.bucket, state.capacity);
        if self.step.as_ref().map(|s| (s.batch, s.bucket, s.capacity)) != Some(want) {
            self.step = None;
            let cap = state.capacity as u64;
            self.step = Some(StepSlot {
                bound: None,
                batch,
                bucket: state.bucket,
                capacity: state.capacity,
                #[allow(clippy::single_range_in_vec_init)]
                cache: BucketedCompileCache::new(self.device, vec![0..cap + 1]),
            });
        }
        let specs = [
            CacheRunInput {
                name: "x",
                data: &x,
                row_inner: None,
            },
            CacheRunInput {
                name: "mask",
                data: &mask,
                row_inner: None,
            },
        ];
        let options = compile_options_for_profile(&CompileProfile::encoder(), self.device);
        let slot = self.step.as_mut().unwrap();
        if slot.cache.compiled_count() == 0 {
            // Compile first, then stream the weights in (see `ShapeRecorder`).
            let mut rec = ShapeRecorder::new(&self.weights);
            let built = flow::build_decode_step_built(
                &self.cfg,
                &mut rec,
                batch,
                state.capacity,
                state.bucket,
            )?;
            let taken = rec.into_taken();
            let (graph, _) = graph_from_built(built)?;
            let (_, compiled) = slot
                .cache
                .get_or_compile_with_options(past as u64, |_| graph, &options)
                .ok_or_else(|| anyhow!("nllb: position {past} outside the step graph"))?;
            self.weights.stream_into(compiled, &taken)?;
            slot.bound = None;
        }
        let rebound = slot.bound != Some(state.id);
        if rebound {
            let compiled = slot
                .cache
                .compiled_for_key_mut(past as u64)
                .ok_or_else(|| anyhow!("nllb: step graph missing"))?;
            compiled.set_param(flow::SRC_MASK_PARAM, &state.src_mask);
            for (i, name) in flow::cross_kv_param_names(n).chunks(2).enumerate() {
                compiled.set_param(&name[0], &state.cross_k[i]);
                compiled.set_param(&name[1], &state.cross_v[i]);
            }
            slot.bound = Some(state.id);
        }
        if let Some(binding) = state.resident.as_mut() {
            // Handles are (re)installed from the host cache whenever this
            // graph last served another state.
            let hidden = run_bucketed_kv_decode_gpu(
                &mut slot.cache,
                past as u64,
                past,
                &mut state.kv,
                binding,
                d,
                n,
                &specs,
                |_| unreachable!("nllb step graph is compiled above"),
                &options,
                rebound,
            )?;
            // Mirror only the new row per layer (24 × 4 KB, not the cache)
            // so the host copy stays the true history: another state may
            // take this graph, and a reinstall must restore this one's.
            let compiled = slot
                .cache
                .compiled_for_key_mut(past as u64)
                .ok_or_else(|| anyhow!("nllb: step graph missing"))?;
            for l in 0..n {
                let row = |out| {
                    compiled
                        .read_output_row(out, state.capacity, d)
                        .ok_or_else(|| anyhow!("nllb: K/V row of layer {l} not readable"))
                };
                let (k, v) = (row(1 + 2 * l)?, row(2 + 2 * l)?);
                state.kv.layers_k[l].extend_from_slice(&k);
                state.kv.layers_v[l].extend_from_slice(&v);
            }
            return Ok(hidden);
        }
        let (hidden, new_k, new_v) = run_bucketed_kv_decode_keyed_batched(
            &mut slot.cache,
            past as u64,
            past,
            batch,
            &state.kv,
            d,
            n,
            &specs,
            |_| unreachable!("nllb step graph is compiled above"),
            &options,
        )?;
        state.kv.layers_k = new_k;
        state.kv.layers_v = new_v;
        state.kv.past_len = past + 1;
        Ok(hidden)
    }

    /// [`Self::step_hidden`] + LM head: next-token logits `[batch · vocab]`.
    pub fn step(&mut self, state: &mut DecodeState, tokens: &[u32]) -> Result<Vec<f32>> {
        let hidden = self.step_hidden(state, tokens)?;
        Ok(self.lm_logits_batch(&hidden, tokens.len()))
    }

    /// Encoder over the unpadded source with in-graph positions, compiled for
    /// this call and dropped — the independent spelling tests compare
    /// [`Self::encode_tokens`] to.
    pub fn encode_full(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let embeds = self.embed_text(token_ids);
        let built = flow::build_encoder_built(&self.cfg, &mut self.weights, 1, token_ids.len())?;
        let mut g = compile_built(built, self.device)?;
        g.run(&[("inputs_embeds", embeds.as_slice())])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("encoder graph produced no output"))
    }

    /// Teacher-forced decoder hidden for every position of `token_ids`
    /// (`[len · d]`) in one causal pass — compiled for this call and dropped.
    /// Independent of the KV-cached step path.
    pub fn decode_hidden(&mut self, token_ids: &[u32], enc: &Encoded) -> Result<Vec<f32>> {
        let dec_embeds = self.embed_text(token_ids);
        let built = flow::build_decoder_hidden_built(
            &self.cfg,
            &mut self.weights,
            1,
            token_ids.len(),
            enc.src_len,
        )?;
        let mut g = compile_built(built, self.device)?;
        g.run(&[
            ("decoder_inputs_embeds", dec_embeds.as_slice()),
            ("encoder_hidden", enc.hidden.as_slice()),
        ])
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("decoder graph produced no output"))
    }

    /// Drop the resident encoder and decode-step graphs (weights stay on
    /// disk; the next call rebuilds what it needs).
    pub fn evict_compile_cache(&mut self) {
        self.encoders.clear();
        self.step = None;
    }

    /// Translate `src_text` from `src_lang` to `tgt_lang` (ISO or FLORES codes).
    pub fn translate(
        &mut self,
        src_text: &str,
        src_lang: &str,
        tgt_lang: &str,
        opts: &GenerateConfig,
    ) -> Result<String> {
        let src_code = crate::resolve_flores(src_lang)?;
        let tgt_code = crate::resolve_flores(tgt_lang)?;
        let (src_ids, tgt_lang_id) = {
            let tk = self
                .tokenizer
                .as_ref()
                .ok_or_else(|| anyhow!("nllb: tokenizer required for translate()"))?;
            (
                tk.encode(src_text, &src_code)?,
                tk.lang_token_id(&tgt_code)?,
            )
        };
        let enc = self.encode_tokens(&src_ids)?;
        let mut gencfg = opts.clone();
        gencfg.forced_bos_token_id = Some(tgt_lang_id);
        let out_ids = self.generate_beam(&enc, &gencfg)?;
        // Drop decoder_start + forced lang code; strip trailing EOS.
        let start = out_ids.len().min(2);
        let mut body = out_ids[start..].to_vec();
        if body.last() == Some(&self.cfg.eos_token_id) {
            body.pop();
        }
        self.tokenizer
            .as_ref()
            .ok_or_else(|| anyhow!("nllb: tokenizer required for translate()"))?
            .decode(&body)
    }

    /// Tied LM head + `final_logits_bias` for one hidden row `[d]`.
    pub fn lm_logits(&self, hidden_row: &[f32]) -> Vec<f32> {
        self.lm_logits_batch(hidden_row, 1)
    }

    /// Tied LM head for `rows` hidden rows (`[rows · d]` → `[rows · vocab]`).
    pub fn lm_logits_batch(&self, hidden: &[f32], rows: usize) -> Vec<f32> {
        let d = self.cfg.d_model;
        let vocab = self.cfg.vocab_size;
        debug_assert_eq!(hidden.len(), rows * d);
        let mut logits = vec![0f32; rows * vocab];
        // One pass over the 1 GB table for all rows, split across the cores.
        rlx_core::host_kernels::matmul_bt_few_rows(
            hidden,
            &self.shared_table,
            &mut logits,
            rows,
            d,
            vocab,
        );
        if self.final_logits_bias.len() == vocab {
            for row in logits.chunks_mut(vocab) {
                for (o, &b) in row.iter_mut().zip(&self.final_logits_bias) {
                    *o += b;
                }
            }
        }
        logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_cover_lengths() {
        assert_eq!(src_bucket(1), 64);
        assert_eq!(src_bucket(64), 64);
        assert_eq!(src_bucket(65), 128);
        assert_eq!(src_bucket(129), 512);
        // One step graph serves the 64- and 128-row encoders.
        assert_eq!(
            src_bucket(5).max(MIN_STEP_SRC),
            src_bucket(100).max(MIN_STEP_SRC)
        );
        assert_eq!(src_bucket(1024), 1024);
        // max_new_tokens 1..=64 share one graph; the last fed token fits.
        assert_eq!(step_capacity(65), 64);
        assert_eq!(step_capacity(2), 64);
        assert_eq!(step_capacity(129), 128);
        for max_len in 2..600 {
            assert!(step_capacity(max_len) >= max_len - 2);
        }
    }
}
