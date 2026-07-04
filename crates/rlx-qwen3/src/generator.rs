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

//! Host-side generation loop for Qwen3.
//!
//! This is the **naive** generator: each `step()` rebuilds the prefill
//! graph for the full token history and runs it from scratch
//! (O(N²) compute over N generated tokens). The API is shaped to
//! match the upcoming KV-cache version exactly so callers don't have
//! to change anything when the cached path lands — only the internal
//! implementation swaps.
//!
//! Why ship the naive version first:
//!   - Establishes the public API contract before the IR/kernel
//!     changes that the cached version needs land.
//!   - Lets you run end-to-end generation against a real checkpoint
//!     today and validate the prefill graph is numerically correct.
//!   - Provides a reference baseline for the cached version's own
//!     numerical-parity test (cached vs recompute must match).

use crate::builder::{
    build_qwen3_decode_graph_sized, build_qwen3_decode_graph_sized_ext,
    build_qwen3_decode_graph_sized_ragged, build_qwen3_graph_sized,
    build_qwen3_graph_sized_last_logits,
};
use crate::capabilities::validate_device;
use crate::config::Qwen3Config;
use crate::profile::qwen3_profile_near_weights;
use crate::sampling::{SampleOpts, sample_token};
use anyhow::{Context, Result};
use rlx_core::autoregressive::{
    DecodeLogitsKv, KvCacheState, compile_cache_ensure_graph, kv_from_prefill_outputs,
    prefill_cache_key, run_bucketed_kv_decode, split_decode_logits_kv,
};
use rlx_core::flow_bridge::compile_options_from_profile;
use rlx_core::weight_loader::WeightLoader;
use rlx_core::weight_map::WeightMap;
use rlx_flow::CompileProfile;
use rlx_ir::logical_kernel::KernelDispatchConfig;
use rlx_runtime::attn_mask::bucket_decode_mask;
use rlx_runtime::compile_cache::{BucketedCompileCache, CacheRunInput, CompileCache};
use rlx_runtime::{CompileOptions, Device, Session};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Stateful Qwen3 generation handle.
///
/// Holds the (config, weight bytes, token history) and rebuilds a
/// prefill graph on each [`step`] call. Cheap to construct after
/// initial weight load; tokens stay in-memory between calls.
pub struct Qwen3Generator {
    cfg: Qwen3Config,
    /// Map of weight key → (f32 data, shape), behind an `Arc` so the
    /// per-step setup is an O(1) refcount bump, not a deep copy of every
    /// weight. `WeightMap::take` is destructive, so a fresh `WeightMap` is
    /// still materialized (`(*arc).clone()`) when a graph is *built* — but
    /// with the bucketed decode cache that happens only on a compile miss,
    /// not on every decode step (the hot path captures just the `Arc`).
    weights_cache: Arc<HashMap<String, (Vec<f32>, Vec<usize>)>>,
    tokens: Vec<u32>,
    device: Device,
    /// Populated lazily on the first `step_cached` call (seeded from
    /// the prompt via prefill-with-cache); thereafter advanced by each
    /// decode step.
    cache: Option<KvCacheState>,
    /// Per-key LRU compile cache for prefill graphs. Keyed by `seq`.
    /// Set to `None` to disable (default for new instances; opt in via
    /// [`Qwen3Generator::with_prefill_cache`]).
    prefill_compile_cache: Option<CompileCache>,
    /// Bucketed compile cache for decode-mode graphs. Each bucket
    /// holds one compiled graph specialized at its upper-bound
    /// `past_seq`; the host pads `past_k`/`past_v` and supplies a
    /// per-step mask so a single bucket serves every `past_seq` in
    /// its range. Opt in via [`Qwen3Generator::with_decode_cache`].
    decode_compile_cache: Option<BucketedCompileCache>,
    /// Bucketed decode compile caches for **fused batched** decode, one per
    /// batch size `B` (the decode graph shape is specialized on `B`). Built
    /// lazily by [`Qwen3Generator::decode_batched_uniform`].
    batched_decode_caches: HashMap<usize, BucketedCompileCache>,
    /// Like [`Self::batched_decode_caches`] but for **ragged** batched decode
    /// (per-sequence RoPE rows), keyed by batch size `B`.
    batched_ragged_caches: HashMap<usize, BucketedCompileCache>,
    prefill_profile: CompileProfile,
    decode_profile: CompileProfile,
}

impl Qwen3Generator {
    /// Construct from any [`WeightLoader`] — drains it into an
    /// internal cache so the loader is free after this call.
    pub fn from_loader(
        cfg: Qwen3Config,
        loader: &mut dyn WeightLoader,
        device: Device,
    ) -> Result<Self> {
        validate_device(&cfg, device, false)?;
        let keys = loader.remaining_keys();
        let mut weights_cache = HashMap::with_capacity(keys.len());
        for k in keys {
            let v = loader
                .take(&k)
                .with_context(|| format!("draining weight {k}"))?;
            // Normalize the cache key to the safetensors / HuggingFace
            // naming convention so subsequent builder calls that ask
            // for `model.embed_tokens.weight` (the canonical name baked
            // into the qwen3 builder) hit the cache whether the
            // loader was safetensors-native or GGUF-native.
            let canonical =
                rlx_core::weight_loader::gguf_to_hf_name(&k).unwrap_or_else(|| k.clone());
            weights_cache.insert(canonical, v);
        }
        let max_past = cfg.max_position_embeddings.clamp(1, 4096);
        Ok(Self {
            cfg,
            weights_cache: Arc::new(weights_cache),
            tokens: Vec::new(),
            device,
            cache: None,
            prefill_compile_cache: Some(CompileCache::new(device, 8)),
            decode_compile_cache: Some(BucketedCompileCache::power_of_two_ladder(
                device,
                1,
                max_past as u64,
            )),
            batched_decode_caches: HashMap::new(),
            batched_ragged_caches: HashMap::new(),
            prefill_profile: CompileProfile::qwen3_prefill(),
            decode_profile: CompileProfile::qwen3_decode(),
        })
    }

    /// Like [`Self::from_loader`] but loads `qwen3.rlx.toml` from the weights directory when present.
    pub fn from_loader_at(
        cfg: Qwen3Config,
        loader: &mut dyn WeightLoader,
        device: Device,
        weights_path: &Path,
    ) -> Result<Self> {
        let mut g = Self::from_loader(cfg, loader, device)?;
        g.prefill_profile = qwen3_profile_near_weights(weights_path, false);
        g.decode_profile = qwen3_profile_near_weights(weights_path, true);
        Ok(g)
    }

    pub fn with_compile_profiles(
        mut self,
        prefill: CompileProfile,
        decode: CompileProfile,
    ) -> Self {
        self.prefill_profile = prefill;
        self.decode_profile = decode;
        self
    }

    pub fn prefill_profile(&self) -> &CompileProfile {
        &self.prefill_profile
    }

    pub fn decode_profile(&self) -> &CompileProfile {
        &self.decode_profile
    }

    fn profile_compile_options(&self, decode: bool) -> CompileOptions {
        let profile = if decode {
            &self.decode_profile
        } else {
            &self.prefill_profile
        };
        compile_options_from_profile(profile, self.device, KernelDispatchConfig::default())
    }

    /// Enable the prefill compile cache with the given LRU capacity.
    /// Useful when the same prompt length is used across multiple
    /// generation runs — the second + Nth run skip the compile +
    /// param-attach roundtrip (~30-50ms per call on CPU).
    pub fn with_prefill_cache(mut self, capacity: usize) -> Self {
        self.prefill_compile_cache = Some(CompileCache::new(self.device, capacity));
        self
    }

    /// Enable the bucketed decode compile cache spanning past-seq
    /// values in `[1, max_past]`. Buckets are power-of-two
    /// `[1..2, 2..3, 3..5, 5..9, 9..17, …]`. Each bucket compiles
    /// one graph at its upper bound; a steady-state generation loop
    /// across `N` tokens compiles `O(log N)` graphs instead of `N`.
    ///
    /// Padding compute waste is bounded at 2×: actual `past_seq` is
    /// at least half the bucket's upper bound (except possibly the
    /// smallest bucket).
    /// Override the bucketed decode compile cache after construction.
    /// Passing `None` forces the naive (O(N²)) generation path.
    pub fn set_decode_compile_cache(&mut self, cache: Option<BucketedCompileCache>) {
        self.decode_compile_cache = cache;
    }

    pub fn with_decode_cache(mut self, max_past: usize) -> Self {
        let cache = BucketedCompileCache::power_of_two_ladder(
            self.device,
            /*min*/ 1,
            max_past.max(1) as u64,
        );
        self.decode_compile_cache = Some(cache);
        self
    }

    /// Convenience: load weights from a safetensors or GGUF path
    /// (dispatch by extension; see `rlx_core::weight_loader::load_from_path`).
    pub fn from_path(cfg: Qwen3Config, path: &str, device: Device) -> Result<Self> {
        Self::from_path_at(cfg, path, device, Path::new("."))
    }

    /// Like [`Self::from_path`] with an explicit weights path for `qwen3.rlx.toml` discovery.
    pub fn from_path_at(
        cfg: Qwen3Config,
        path: &str,
        device: Device,
        weights_path: &Path,
    ) -> Result<Self> {
        let mut loader = rlx_core::weight_loader::load_from_path(path)?;
        Self::from_loader_at(cfg, loader.as_mut(), device, weights_path)
    }

    /// Same as [`from_path`] but with MTP-head visibility control.
    /// When `include_mtp=true` and the file is GGUF, MTP weights are
    /// drained into the generator's cache alongside the base
    /// weights. The base inference path still ignores them — they
    /// sit in cache for a future MTP-aware decoder. Non-GGUF formats
    /// silently ignore the flag (safetensors files publish all
    /// tensors uniformly; downstream code distinguishes by name).
    pub fn from_path_with_mtp(
        cfg: Qwen3Config,
        path: &str,
        device: Device,
        include_mtp: bool,
    ) -> Result<Self> {
        // Branch on extension so we can flip the GGUF-specific
        // visibility option. Safetensors has no equivalent — it
        // doesn't isolate MTP tensors at the loader level.
        if path.ends_with(".gguf") {
            let mut gguf = rlx_core::weight_loader::GgufLoader::from_file(path)?;
            gguf.include_mtp(include_mtp);
            Self::from_loader(cfg, &mut gguf, device)
        } else {
            Self::from_path(cfg, path, device)
        }
    }

    /// Replace the token history with `prompt_ids`. Does not run the
    /// model — the next [`step`] call processes the full sequence.
    /// Clears any KV cache from a prior generation.
    pub fn prefill(&mut self, prompt_ids: &[u32]) {
        self.tokens.clear();
        self.tokens.extend_from_slice(prompt_ids);
        self.cache = None;
    }

    /// Run one prefill over the current token history and sample the
    /// next token. The sampled token is appended to the history and
    /// returned. Call repeatedly to generate.
    pub fn step(&mut self, opts: SampleOpts) -> Result<u32> {
        if self.tokens.is_empty() {
            anyhow::bail!("step() called with empty token history; call prefill() first");
        }
        let seq = self.tokens.len();
        let mut wm = WeightMap::from_tensors((*self.weights_cache).clone());
        let (graph, params) = build_qwen3_graph_sized_last_logits(
            &self.cfg, &mut wm, /*batch*/ 1, seq, /*with_kv_outputs*/ false,
        )?;
        let compile_opts = self.profile_compile_options(false);
        let mut compiled = Session::new(self.device).compile_with(graph, &compile_opts);
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        let ids_f32: Vec<f32> = self.tokens.iter().map(|&i| i as f32).collect();
        let outputs = compiled.run(&[("input_ids", ids_f32.as_slice())]);
        let logits = outputs
            .into_iter()
            .next()
            .context("compiled.run returned no outputs")?;

        let vocab = self.cfg.vocab_size;
        let expected = vocab;
        if logits.len() < expected {
            anyhow::bail!(
                "logits length {} < expected {} (last logits, seq {seq}, vocab {vocab})",
                logits.len(),
                expected
            );
        }
        // Last-logits graph returns [B=1, 1, vocab].
        let last_row = &logits[..vocab];
        let tok = sample_token(last_row, opts) as u32;
        self.tokens.push(tok);
        Ok(tok)
    }

    /// Run `n` steps and return the newly generated token ids
    /// (excludes the prefill prompt).
    pub fn generate(&mut self, n: usize, opts: SampleOpts) -> Result<Vec<u32>> {
        if self.decode_compile_cache.is_some() {
            return self.generate_cached(n, opts);
        }
        let start = self.tokens.len();
        for _ in 0..n {
            self.step(opts)?;
        }
        Ok(self.tokens[start..].to_vec())
    }

    /// Cached step: O(L) per token instead of O(L²). First call seeds
    /// the KV cache from the prompt via prefill-with-cache; subsequent
    /// calls run the decode-mode graph on just the last token + cached
    /// past. Output is bit-identical to [`step`] modulo reduction
    /// order in the SDPA kernel.
    ///
    /// Invariant after each call: `cache.past_seq == tokens.len() - 1`
    /// (the just-sampled token is appended but not yet in the cache;
    /// it becomes the input for the next decode step).
    pub fn step_cached(&mut self, opts: SampleOpts) -> Result<u32> {
        if self.tokens.is_empty() {
            anyhow::bail!("step_cached() called with empty token history; call prefill() first");
        }
        if self.cache.is_none() {
            // The seed runs prefill, populates the cache, samples from
            // the last position, and appends the token. Return that
            // token directly — no decode step on this call.
            let tok = self.seed_cache_from_prompt(opts)?;
            // Bound the prompt's cache to the window before the first decode.
            self.rotate_cache_if_sliding();
            return Ok(tok);
        }
        let cache = self.cache.as_ref().unwrap();
        let past_seq = cache.past_len;
        // The token we feed into decode is whatever's after the cached
        // prefix in `self.tokens`. After a prior cached step this is
        // the just-sampled token; after seeding it's the same.
        if self.tokens.len() <= past_seq {
            anyhow::bail!(
                "cache invariant violated: tokens.len() {} <= past_seq {}",
                self.tokens.len(),
                past_seq
            );
        }
        // The token to feed is always the most recent one, at its *absolute*
        // position. With a rotated sliding-window cache `past_seq` (cache
        // length) is smaller than `abs_pos`; for non-windowed models they are
        // equal, so this is behavior-preserving.
        let abs_pos = self.tokens.len() - 1;
        let input_tok = self.tokens[abs_pos];

        // Branch: bucketed compile cache vs one-shot compile per step.
        let (logits, new_k, new_v) = if self.decode_compile_cache.is_some()
            && self
                .decode_compile_cache
                .as_ref()
                .unwrap()
                .bucket_for(past_seq as u64)
                .is_some()
        {
            self.decode_step_bucketed(past_seq, abs_pos, input_tok)?
        } else {
            self.decode_step_oneshot(past_seq, abs_pos, input_tok)?
        };

        let cache_mut = self.cache.as_mut().unwrap();
        cache_mut.past_len = past_seq + 1;
        cache_mut.layers_k = new_k;
        cache_mut.layers_v = new_v;
        // Sliding-window: keep only the last `window` cached positions.
        self.rotate_cache_if_sliding();

        let vocab = self.cfg.vocab_size;
        if logits.len() != vocab {
            anyhow::bail!("decode logits length {} != vocab {}", logits.len(), vocab);
        }
        let tok = sample_token(&logits, opts) as u32;
        self.tokens.push(tok);
        Ok(tok)
    }

    /// For sliding-window models, trim the KV cache to its last `window`
    /// positions (a rotating cache). With absolute-position RoPE in the decode
    /// step, this gives windowed attention at O(window) memory instead of
    /// O(sequence). A no-op for non-windowed models.
    fn rotate_cache_if_sliding(&mut self) {
        let w = match (self.cfg.use_sliding_window, self.cfg.sliding_window) {
            (true, Some(w)) if w > 0 => w,
            _ => return,
        };
        let kv_dim = self.cfg.kv_proj_dim();
        if let Some(cache) = self.cache.as_mut() {
            if cache.past_len > w {
                let drop = (cache.past_len - w) * kv_dim;
                for k in cache.layers_k.iter_mut() {
                    k.drain(0..drop.min(k.len()));
                }
                for v in cache.layers_v.iter_mut() {
                    v.drain(0..drop.min(v.len()));
                }
                cache.past_len = w;
            }
        }
    }

    /// Decode path that compiles a fresh graph for the exact `past_seq`
    /// every call. Slower but always-correct fallback.
    fn decode_step_oneshot(
        &mut self,
        past_seq: usize,
        abs_pos: usize,
        input_tok: u32,
    ) -> Result<DecodeLogitsKv> {
        let cache = self.cache.as_ref().unwrap();

        let mut wm = WeightMap::from_tensors((*self.weights_cache).clone());
        let (graph, params) =
            build_qwen3_decode_graph_sized(&self.cfg, &mut wm, /*batch*/ 1, past_seq)?;
        let opts = self.profile_compile_options(true);
        let mut compiled = Session::new(self.device).compile_with(graph, &opts);
        for (name, data) in &params {
            compiled.set_param(name, data);
        }

        let (cos, sin) = compute_rope_slice(&self.cfg, abs_pos);
        let input_ids_f32 = [input_tok as f32];
        let key_strs: Vec<String> = (0..self.cfg.num_hidden_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        let mut inputs: Vec<(&str, &[f32])> =
            Vec::with_capacity(3 + 2 * self.cfg.num_hidden_layers);
        inputs.push(("input_ids", input_ids_f32.as_slice()));
        inputs.push(("rope_cos", cos.as_slice()));
        inputs.push(("rope_sin", sin.as_slice()));
        for i in 0..self.cfg.num_hidden_layers {
            inputs.push((&key_strs[2 * i], cache.layers_k[i].as_slice()));
            inputs.push((&key_strs[2 * i + 1], cache.layers_v[i].as_slice()));
        }

        let outputs = compiled.run(&inputs);
        split_decode_logits_kv(outputs, self.cfg.num_hidden_layers)
    }

    fn decode_step_bucketed(
        &mut self,
        past_seq: usize,
        abs_pos: usize,
        input_tok: u32,
    ) -> Result<DecodeLogitsKv> {
        let kv = self.cache.as_ref().unwrap().clone();
        let kv_dim = self.cfg.kv_proj_dim();
        let n_layers = self.cfg.num_hidden_layers;
        // RoPE uses the token's *absolute* position; with a rotated
        // (sliding-window) cache this differs from the cache length `past_seq`.
        let (cos, sin) = compute_rope_slice(&self.cfg, abs_pos);
        let input_ids_f32 = [input_tok as f32];
        let decode_opts = self.profile_compile_options(true);
        let upper = self
            .decode_compile_cache
            .as_ref()
            .and_then(|cache_dec| {
                cache_dec.bucket_for(past_seq as u64).map(|idx| {
                    cache_dec
                        .buckets()
                        .nth(idx)
                        .map(|r| (r.end - 1) as usize)
                        .unwrap_or(past_seq)
                })
            })
            .unwrap_or(past_seq);
        // Standard full causal decode mask: every cached key is valid. For
        // sliding-window models the cache is *rotated* to hold only the last
        // `window` keys (see `rotate_cache_if_sliding`), so the windowing is
        // already in the cache and the plain mask suffices.
        let mask = bucket_decode_mask(past_seq, upper);
        let fixed = [
            CacheRunInput {
                name: "input_ids",
                data: &input_ids_f32,
                row_inner: None,
            },
            CacheRunInput {
                name: "rope_cos",
                data: &cos,
                row_inner: None,
            },
            CacheRunInput {
                name: "rope_sin",
                data: &sin,
                row_inner: None,
            },
            CacheRunInput {
                name: "mask",
                data: &mask,
                row_inner: None,
            },
        ];
        let cfg = self.cfg.clone();
        let weights = self.weights_cache.clone();
        let cache_dec = self.decode_compile_cache.as_mut().unwrap();
        run_bucketed_kv_decode(
            cache_dec,
            past_seq,
            &kv,
            kv_dim,
            n_layers,
            &fixed,
            |upper| {
                // Deep-copy weights only when this bucket actually compiles
                // (cache miss); steady-state decode captures just the `Arc`.
                let mut wm = WeightMap::from_tensors((*weights).clone());
                build_qwen3_decode_graph_sized_ext(&cfg, &mut wm, 1, upper as usize, true)
                    .expect("qwen3 bucketed decode graph")
            },
            &decode_opts,
        )
    }

    /// **Fused batched decode** — the throughput primitive for serving.
    ///
    /// Decodes `B = entries.len()` sequences in ONE forward, so every weight
    /// matmul (QKV/O projections, MLP, LM head) runs once over the whole batch
    /// instead of `B` times. Each `entries[i] = (token, kv)` gives sequence
    /// `i`'s next input token and its cache; **all caches must share
    /// `past_seq`** (the same `past_len`) and the same absolute position
    /// `abs_pos`, because the decode graph's RoPE slice and causal-mask bucket
    /// are per-step values shared across the batch. The continuous batcher
    /// groups decode work by cache length to satisfy this.
    ///
    /// Returns, per input sequence, `(next_token_logits[vocab], advanced_kv)`
    /// where `advanced_kv.past_len == past_seq + 1`.
    ///
    /// KV layout note: the decode graph reads/writes K/V **batch-major**
    /// `[batch, seq_pos, kv_dim]` (each sequence's positions contiguous). We
    /// pad each sequence to the compiled bucket `upper` and assemble that
    /// layout directly, then run the cached graph and de-interleave the
    /// `[batch, upper+1, kv_dim]` output (real positions `0..past_seq` plus the
    /// new token at row `upper`). The bucketed primitive's host-side
    /// pad/compact helpers are position-major and only coincide with this at
    /// batch=1, so we drive the compiled graph directly. Logits are batch-major
    /// `[batch, vocab]`.
    pub fn decode_batched_uniform(
        &mut self,
        entries: &[(u32, &KvCacheState)],
        abs_pos: usize,
        past_seq: usize,
    ) -> Result<Vec<(Vec<f32>, KvCacheState)>> {
        let b = entries.len();
        if b == 0 {
            return Ok(Vec::new());
        }
        let kv_dim = self.cfg.kv_proj_dim();
        let n_layers = self.cfg.num_hidden_layers;
        let vocab = self.cfg.vocab_size;
        let max_past = self.cfg.max_position_embeddings.clamp(1, 4096) as u64;
        let device = self.device;
        // Hoist every `self` read before the mutable cache borrow below.
        let decode_opts = self.profile_compile_options(true);
        let cfg = self.cfg.clone();
        let weights = self.weights_cache.clone();

        let input_ids_f32: Vec<f32> = entries.iter().map(|(t, _)| *t as f32).collect();
        let (cos, sin) = compute_rope_slice(&cfg, abs_pos);

        // One bucketed compile cache per batch size (graph shape depends on B).
        // Fetch the compiled graph for this past_seq bucket and its `upper`.
        let cache_b = self
            .batched_decode_caches
            .entry(b)
            .or_insert_with(|| BucketedCompileCache::power_of_two_ladder(device, 1, max_past));
        let (upper_u64, compiled) = cache_b
            .ensure_graph_with_params(
                past_seq as u64,
                |upper| {
                    let mut wm = WeightMap::from_tensors((*weights).clone());
                    build_qwen3_decode_graph_sized_ext(&cfg, &mut wm, b, upper as usize, true)
                        .expect("qwen3 batched decode graph")
                },
                &decode_opts,
            )
            .ok_or_else(|| anyhow::anyhow!("past_seq {past_seq} outside decode buckets"))?;
        let upper = upper_u64 as usize;

        // Batch-major padded KV `[batch, upper, kv_dim]`: each sequence's
        // `past_seq` real positions followed by `upper - past_seq` zero rows.
        let real = past_seq * kv_dim;
        let mut padded_k: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
        let mut padded_v: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let mut pk = vec![0.0f32; b * upper * kv_dim];
            let mut pv = vec![0.0f32; b * upper * kv_dim];
            for (i, (_tok, kv)) in entries.iter().enumerate() {
                debug_assert_eq!(
                    kv.past_len, past_seq,
                    "decode_batched_uniform: ragged past_len"
                );
                let base = i * upper * kv_dim;
                pk[base..base + real].copy_from_slice(&kv.layers_k[l][..real]);
                pv[base..base + real].copy_from_slice(&kv.layers_v[l][..real]);
            }
            padded_k.push(pk);
            padded_v.push(pv);
        }

        // Per-row mask `[batch, upper + 1]`; rows are identical at uniform past_seq.
        let row_mask = bucket_decode_mask(past_seq, upper);
        let mut mask = Vec::with_capacity(b * row_mask.len());
        for _ in 0..b {
            mask.extend_from_slice(&row_mask);
        }

        let key_strs: Vec<String> = (0..n_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        let mut inputs: Vec<(&str, &[f32])> = Vec::with_capacity(4 + 2 * n_layers);
        inputs.push(("input_ids", &input_ids_f32));
        inputs.push(("rope_cos", &cos));
        inputs.push(("rope_sin", &sin));
        inputs.push(("mask", &mask));
        for l in 0..n_layers {
            inputs.push((&key_strs[2 * l], &padded_k[l]));
            inputs.push((&key_strs[2 * l + 1], &padded_v[l]));
        }

        let raw = compiled.run(&inputs);
        let (logits, new_k, new_v) = split_decode_logits_kv(raw, n_layers)?;

        // De-interleave batch-major: logits `[batch, vocab]`; per layer the KV
        // output is `[batch, upper+1, kv_dim]` — take positions `0..past_seq`
        // and the new token at row `upper`.
        let new_len = past_seq + 1;
        let mut out = Vec::with_capacity(b);
        for i in 0..b {
            let lo = logits[i * vocab..(i + 1) * vocab].to_vec();
            let mut kv = KvCacheState {
                past_len: new_len,
                layers_k: Vec::with_capacity(n_layers),
                layers_v: Vec::with_capacity(n_layers),
                layers_kv_base: vec![0; n_layers],
            };
            for l in 0..n_layers {
                let stride = (upper + 1) * kv_dim;
                let base = i * stride;
                let nt = base + upper * kv_dim;
                let mut sk = vec![0.0f32; new_len * kv_dim];
                let mut sv = vec![0.0f32; new_len * kv_dim];
                sk[..real].copy_from_slice(&new_k[l][base..base + real]);
                sv[..real].copy_from_slice(&new_v[l][base..base + real]);
                sk[real..real + kv_dim].copy_from_slice(&new_k[l][nt..nt + kv_dim]);
                sv[real..real + kv_dim].copy_from_slice(&new_v[l][nt..nt + kv_dim]);
                kv.layers_k.push(sk);
                kv.layers_v.push(sv);
            }
            out.push((lo, kv));
        }
        Ok(out)
    }

    /// **Ragged fused batched decode** — decodes `B = entries.len()` sequences
    /// in ONE forward even when they sit at **different** cache lengths.
    ///
    /// Unlike [`Self::decode_batched_uniform`], each sequence carries its own
    /// past length (`kv.past_len`) and absolute position. The decode graph is
    /// built with per-sequence RoPE rows (`rope_cos`/`rope_sin` shaped
    /// `[batch, head_dim/2]`) and a per-row causal mask, and every sequence's KV
    /// is padded to the shared bucket `upper`. This is what lets a real server
    /// fuse arbitrary concurrent requests (varied prompt lengths, staggered
    /// arrivals) into a single batched forward — the full throughput win.
    ///
    /// Each `entries[i] = (token, kv)`; returns `(next_token_logits, advanced_kv)`
    /// per sequence with `advanced_kv.past_len == kv.past_len + 1`. Assumes
    /// non-sliding RoPE (absolute position == cache length).
    pub fn decode_batched_ragged(
        &mut self,
        entries: &[(u32, &KvCacheState)],
    ) -> Result<Vec<(Vec<f32>, KvCacheState)>> {
        let b = entries.len();
        if b == 0 {
            return Ok(Vec::new());
        }
        // A single sequence has nothing to fuse and no ragged RoPE need — the
        // shared-row uniform path is simpler and identical numerically.
        if b == 1 {
            let (tok, kv) = entries[0];
            let past = kv.past_len;
            return self.decode_batched_uniform(&[(tok, kv)], past, past);
        }
        let kv_dim = self.cfg.kv_proj_dim();
        let n_layers = self.cfg.num_hidden_layers;
        let vocab = self.cfg.vocab_size;
        let half = self.cfg.head_dim / 2;
        let max_past = self.cfg.max_position_embeddings.clamp(1, 4096) as u64;
        let device = self.device;
        let decode_opts = self.profile_compile_options(true);
        let cfg = self.cfg.clone();
        let weights = self.weights_cache.clone();

        // Per-sequence input token, RoPE row (at the sequence's own position),
        // and mask row (each sequence's own real past length).
        let input_ids_f32: Vec<f32> = entries.iter().map(|(t, _)| *t as f32).collect();
        let max_past_seq = entries.iter().map(|(_, kv)| kv.past_len).max().unwrap_or(0);

        // Graph + bucket `upper` for the longest sequence in the batch.
        let cache_b = self
            .batched_ragged_caches
            .entry(b)
            .or_insert_with(|| BucketedCompileCache::power_of_two_ladder(device, 1, max_past));
        let (upper_u64, compiled) = cache_b
            .ensure_graph_with_params(
                max_past_seq as u64,
                |upper| {
                    let mut wm = WeightMap::from_tensors((*weights).clone());
                    build_qwen3_decode_graph_sized_ragged(&cfg, &mut wm, b, upper as usize)
                        .expect("qwen3 ragged decode graph")
                },
                &decode_opts,
            )
            .ok_or_else(|| anyhow::anyhow!("past_seq {max_past_seq} outside decode buckets"))?;
        let upper = upper_u64 as usize;

        // Per-row RoPE `[batch, half]` and mask `[batch, upper+1]`.
        let mut cos = Vec::with_capacity(b * half);
        let mut sin = Vec::with_capacity(b * half);
        let mut mask = Vec::with_capacity(b * (upper + 1));
        for (_tok, kv) in entries {
            let (c, s) = compute_rope_slice(&cfg, kv.past_len);
            cos.extend_from_slice(&c);
            sin.extend_from_slice(&s);
            mask.extend_from_slice(&bucket_decode_mask(kv.past_len, upper));
        }

        // Batch-major padded KV `[batch, upper, kv_dim]`; each sequence's own
        // `past_len` real rows then zero padding to `upper`.
        let mut padded_k: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
        let mut padded_v: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let mut pk = vec![0.0f32; b * upper * kv_dim];
            let mut pv = vec![0.0f32; b * upper * kv_dim];
            for (i, (_tok, kv)) in entries.iter().enumerate() {
                let real = kv.past_len * kv_dim;
                let base = i * upper * kv_dim;
                pk[base..base + real].copy_from_slice(&kv.layers_k[l][..real]);
                pv[base..base + real].copy_from_slice(&kv.layers_v[l][..real]);
            }
            padded_k.push(pk);
            padded_v.push(pv);
        }

        let key_strs: Vec<String> = (0..n_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        let mut inputs: Vec<(&str, &[f32])> = Vec::with_capacity(4 + 2 * n_layers);
        inputs.push(("input_ids", &input_ids_f32));
        inputs.push(("rope_cos", &cos));
        inputs.push(("rope_sin", &sin));
        inputs.push(("mask", &mask));
        for l in 0..n_layers {
            inputs.push((&key_strs[2 * l], &padded_k[l]));
            inputs.push((&key_strs[2 * l + 1], &padded_v[l]));
        }

        let raw = compiled.run(&inputs);
        let (logits, new_k, new_v) = split_decode_logits_kv(raw, n_layers)?;

        // De-interleave per sequence: each has its own new length and the new
        // token's K/V at the padded row `upper`.
        let mut out = Vec::with_capacity(b);
        for i in 0..b {
            let past_i = entries[i].1.past_len;
            let real = past_i * kv_dim;
            let new_len = past_i + 1;
            let lo = logits[i * vocab..(i + 1) * vocab].to_vec();
            let mut kv = KvCacheState {
                past_len: new_len,
                layers_k: Vec::with_capacity(n_layers),
                layers_v: Vec::with_capacity(n_layers),
                layers_kv_base: vec![0; n_layers],
            };
            for l in 0..n_layers {
                let stride = (upper + 1) * kv_dim;
                let base = i * stride;
                let nt = base + upper * kv_dim;
                let mut sk = vec![0.0f32; new_len * kv_dim];
                let mut sv = vec![0.0f32; new_len * kv_dim];
                sk[..real].copy_from_slice(&new_k[l][base..base + real]);
                sv[..real].copy_from_slice(&new_v[l][base..base + real]);
                sk[real..real + kv_dim].copy_from_slice(&new_k[l][nt..nt + kv_dim]);
                sv[real..real + kv_dim].copy_from_slice(&new_v[l][nt..nt + kv_dim]);
                kv.layers_k.push(sk);
                kv.layers_v.push(sv);
            }
            out.push((lo, kv));
        }
        Ok(out)
    }

    /// Run prefill-with-cache and return the raw outputs. Uses the
    /// LRU `CompileCache` when enabled; otherwise compiles fresh each
    /// call. Keyed by `seq` because graph shape is seq-specialized.
    fn run_prefill_with_cache(
        &mut self,
        batch: usize,
        seq: usize,
        ids_f32: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        let prefill_opts = self.profile_compile_options(false);
        if let Some(cache) = &mut self.prefill_compile_cache {
            let key = prefill_cache_key(batch, seq);
            let mut wm = WeightMap::from_tensors((*self.weights_cache).clone());
            let (graph, params) = build_qwen3_graph_sized_last_logits(
                &self.cfg, &mut wm, batch, seq, /*with_kv_outputs*/ true,
            )?;
            let compiled = compile_cache_ensure_graph(cache, key, graph, params, &prefill_opts);
            Ok(compiled.run(&[("input_ids", ids_f32)]))
        } else {
            let mut wm = WeightMap::from_tensors((*self.weights_cache).clone());
            let (graph, params) = build_qwen3_graph_sized_last_logits(
                &self.cfg, &mut wm, batch, seq, /*with_kv_outputs*/ true,
            )?;
            let opts = self.profile_compile_options(false);
            let mut compiled = Session::new(self.device).compile_with(graph, &opts);
            for (name, data) in &params {
                compiled.set_param(name, data);
            }
            Ok(compiled.run(&[("input_ids", ids_f32)]))
        }
    }

    /// Run `n` cached steps and return the newly generated tokens.
    pub fn generate_cached(&mut self, n: usize, opts: SampleOpts) -> Result<Vec<u32>> {
        self.generate_cached_with(n, opts, |_| {})
    }

    /// Same as [`generate_cached`] but invokes `on_token` once per
    /// freshly sampled id, inside the decode loop. The whole `n` step
    /// loop shares the bucketed compile cache — callers wanting a
    /// streaming UI should prefer this to calling
    /// `generate_cached(1, …)` `n` times (which forces a fresh
    /// compile per token at the bucket boundaries).
    pub fn generate_cached_with(
        &mut self,
        n: usize,
        opts: SampleOpts,
        on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        self.generate_cached_until(n, opts, |_| true, on_token)
    }

    /// Like [`generate_cached_with`] but stops early when `should_continue`
    /// returns `false` after sampling a token.
    pub fn generate_cached_until(
        &mut self,
        n: usize,
        opts: SampleOpts,
        mut should_continue: impl FnMut(u32) -> bool,
        mut on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        let start = self.tokens.len();
        for _ in 0..n {
            let tok = self.step_cached(opts)?;
            on_token(tok);
            if !should_continue(tok) {
                break;
            }
        }
        Ok(self.tokens[start..].to_vec())
    }

    /// Run prefill-with-cache on the current `self.tokens` (the
    /// prompt), populate `self.cache`, sample the next token from the
    /// last position's logits, and append it. Returns the sampled
    /// token. Invariant after: `cache.past_seq == tokens.len() - 1`.
    fn seed_cache_from_prompt(&mut self, opts: SampleOpts) -> Result<u32> {
        let seq = self.tokens.len();
        let batch = 1usize;
        let kv_dim = self.cfg.kv_proj_dim();

        let ids_f32: Vec<f32> = self.tokens.iter().map(|&i| i as f32).collect();
        let outputs = self.run_prefill_with_cache(batch, seq, &ids_f32)?;
        let (logits, kv) =
            kv_from_prefill_outputs(outputs, batch, seq, kv_dim, self.cfg.num_hidden_layers)?;
        self.cache = Some(kv);

        let vocab = self.cfg.vocab_size;
        let needed = vocab;
        if logits.len() < needed {
            anyhow::bail!("prefill logits length {} < {}", logits.len(), needed);
        }
        let last_row = &logits[..vocab];
        let tok = sample_token(last_row, opts) as u32;
        self.tokens.push(tok);
        Ok(tok)
    }

    /// Full token history (prompt + generated).
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// One teacher-forced forward over `tokens`; returns the flattened
    /// `[seq, vocab]` logits (row `i` predicts token `i+1`). A single graph
    /// run — used by the eval harness instead of O(N) single decode steps.
    pub fn sequence_logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            anyhow::bail!("sequence_logits: empty token sequence");
        }
        let seq = tokens.len();
        let mut wm = WeightMap::from_tensors((*self.weights_cache).clone());
        let (graph, params) = build_qwen3_graph_sized(
            &self.cfg, &mut wm, 1, seq, /*with_lm_head*/ true, false,
        )?;
        let opts = self.profile_compile_options(false);
        let mut compiled = Session::new(self.device).compile_with(graph, &opts);
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        let ids_f32: Vec<f32> = tokens.iter().map(|&i| i as f32).collect();
        let mut out = compiled.run(&[("input_ids", &ids_f32)]);
        out.drain(..)
            .next()
            .ok_or_else(|| anyhow::anyhow!("sequence_logits: graph produced no output"))
    }

    /// Teacher-forced next-token log-probabilities: `log P(tokens[i+1] |
    /// tokens[..=i])` for `i` in `0..seq-1`. One forward; host-side
    /// log-softmax. Returns `seq - 1` values (empty for a 1-token input).
    pub fn sequence_logprobs(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let seq = tokens.len();
        if seq < 2 {
            return Ok(Vec::new());
        }
        let vocab = self.cfg.vocab_size;
        let logits = self.sequence_logits(tokens)?;
        if logits.len() < seq * vocab {
            anyhow::bail!(
                "sequence_logits len {} < seq*vocab {}",
                logits.len(),
                seq * vocab
            );
        }
        let mut out = Vec::with_capacity(seq - 1);
        for i in 0..seq - 1 {
            let row = &logits[i * vocab..(i + 1) * vocab];
            out.push(log_softmax_at(row, tokens[i + 1] as usize));
        }
        Ok(out)
    }

    /// Snapshot the seeded KV cache + token history for prompt-cache reuse.
    /// `None` if no cache has been seeded yet (no `step_cached`/prefill-cache
    /// call has run).
    pub fn export_cache(&self) -> Option<(KvCacheState, Vec<u32>)> {
        self.cache
            .as_ref()
            .map(|c| (c.clone(), self.tokens.clone()))
    }

    /// Restore a previously exported KV cache + token history so generation
    /// resumes from a cached prefix instead of re-prefilling it.
    pub fn restore_cache(&mut self, cache: KvCacheState, tokens: Vec<u32>) {
        self.cache = Some(cache);
        self.tokens = tokens;
    }

    /// Prefill `prompt`, **reusing** a `restored` KV cache that already covers
    /// the first `reuse_len` tokens — only the suffix is processed (one decode
    /// step per suffix token). Returns the last-position logits and leaves
    /// `self.cache` seeded at `past_len == prompt.len()`, ready for decode.
    ///
    /// `reuse_len` is capped to `prompt.len() - 1` so the final prompt token
    /// is always re-processed: its *next-token* prediction logits are not
    /// stored in the cache, so we must recompute them. When the cache covers
    /// more than the cap, it is trimmed to the reused row count.
    pub fn prefill_with_reuse(
        &mut self,
        prompt: &[u32],
        restored: KvCacheState,
        reuse_len: usize,
    ) -> Result<Vec<f32>> {
        if prompt.is_empty() {
            anyhow::bail!("prefill_with_reuse: empty prompt");
        }
        let kv_dim = self.cfg.kv_proj_dim();
        let reuse = reuse_len.min(prompt.len() - 1).min(restored.past_len);
        let kv = if reuse == restored.past_len {
            restored
        } else {
            trim_kv(&restored, reuse, kv_dim)
        };
        self.cache = Some(kv);
        self.tokens = prompt[..reuse].to_vec();
        let mut logits = Vec::new();
        for &t in &prompt[reuse..] {
            logits = self.decode_get_logits(t)?;
        }
        Ok(logits)
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.cfg
    }

    /// The device this generator compiles + runs on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Low-level primitive: reset internal state, run prefill-with-cache
    /// over `context`, and return the *last position's* logits row
    /// (`P(next_token | context)`). Does NOT sample or append. The
    /// internal `tokens` buffer is set to `context` and the KV cache
    /// is populated to `past_seq = context.len()`.
    ///
    /// Used by [`crate::spec::Qwen3Speculator`] to compute the
    /// first row of a `Speculator::verify` / `propose` result before
    /// the decode loop runs.
    pub fn prefill_get_last_logits(&mut self, context: &[u32]) -> Result<Vec<f32>> {
        if context.is_empty() {
            anyhow::bail!("prefill_get_last_logits: empty context");
        }
        self.tokens.clear();
        self.tokens.extend_from_slice(context);
        self.cache = None;

        let seq = context.len();
        let batch = 1usize;
        let kv_dim = self.cfg.kv_proj_dim();

        let ids_f32: Vec<f32> = context.iter().map(|&i| i as f32).collect();
        let outputs = self.run_prefill_with_cache(batch, seq, &ids_f32)?;
        let (logits, kv) =
            kv_from_prefill_outputs(outputs, batch, seq, kv_dim, self.cfg.num_hidden_layers)?;
        self.cache = Some(kv);

        let vocab = self.cfg.vocab_size;
        let needed = vocab;
        if logits.len() < needed {
            anyhow::bail!("logits short: {} < {}", logits.len(), needed);
        }
        Ok(logits[..vocab].to_vec())
    }

    /// Low-level primitive: run one decode step with the caller-
    /// supplied input token (no sampling), advance the KV cache, and
    /// return the resulting logits row `P(next | history ++ input)`.
    /// Appends `input` to the `tokens` buffer so the invariant
    /// `cache.past_seq == tokens.len()` holds after this call (note:
    /// differs from `step_cached` invariant because this method does
    /// not append a sampled token).
    pub fn decode_get_logits(&mut self, input: u32) -> Result<Vec<f32>> {
        let cache = self.cache.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "decode_get_logits: cache not seeded; call prefill_get_last_logits first"
            )
        })?;
        let past_seq = cache.past_len;

        let mut wm = WeightMap::from_tensors((*self.weights_cache).clone());
        let (graph, params) =
            build_qwen3_decode_graph_sized(&self.cfg, &mut wm, /*batch*/ 1, past_seq)?;
        let opts = self.profile_compile_options(true);
        let mut compiled = Session::new(self.device).compile_with(graph, &opts);
        for (name, data) in &params {
            compiled.set_param(name, data);
        }

        let (cos, sin) = compute_rope_slice(&self.cfg, past_seq);
        let input_ids_f32 = [input as f32];
        let key_strs: Vec<String> = (0..self.cfg.num_hidden_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        let mut inputs: Vec<(&str, &[f32])> =
            Vec::with_capacity(3 + 2 * self.cfg.num_hidden_layers);
        inputs.push(("input_ids", input_ids_f32.as_slice()));
        inputs.push(("rope_cos", cos.as_slice()));
        inputs.push(("rope_sin", sin.as_slice()));
        for i in 0..self.cfg.num_hidden_layers {
            let pk = &cache.layers_k[i];
            let pv = &cache.layers_v[i];
            inputs.push((&key_strs[2 * i], pk.as_slice()));
            inputs.push((&key_strs[2 * i + 1], pv.as_slice()));
        }

        let outputs = compiled.run(&inputs);
        let (logits, new_k, new_v) = split_decode_logits_kv(outputs, self.cfg.num_hidden_layers)?;

        let cache_mut = self.cache.as_mut().unwrap();
        cache_mut.past_len = past_seq + 1;
        cache_mut.layers_k = new_k;
        cache_mut.layers_v = new_v;
        self.tokens.push(input);

        Ok(logits)
    }
}

/// `log softmax(row)[idx]` computed stably (`row[idx] - logsumexp(row)`).
fn log_softmax_at(row: &[f32], idx: usize) -> f32 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sumexp: f32 = row.iter().map(|&x| (x - max).exp()).sum();
    let lse = max + sumexp.ln();
    row.get(idx).copied().unwrap_or(f32::NEG_INFINITY) - lse
}

/// Trim a KV cache to its first `rows` sequence positions (uniform `kv_dim`).
/// Each layer's `[past_len * kv_dim]` row-major buffer keeps its first
/// `rows * kv_dim` elements.
fn trim_kv(kv: &KvCacheState, rows: usize, kv_dim: usize) -> KvCacheState {
    let take = rows * kv_dim;
    let cut = |layers: &[Vec<f32>]| -> Vec<Vec<f32>> {
        layers
            .iter()
            .map(|buf| buf[..take.min(buf.len())].to_vec())
            .collect()
    };
    KvCacheState {
        past_len: rows,
        layers_k: cut(&kv.layers_k),
        layers_v: cut(&kv.layers_v),
        layers_kv_base: kv.layers_kv_base.clone(),
    }
}

/// Compute the single-row (cos, sin) RoPE slice for absolute position
/// `pos`. Matches the formula in the prefill builder so cached decode
/// and recompute prefill produce the same RoPE rotation.
fn compute_rope_slice(cfg: &Qwen3Config, pos: usize) -> (Vec<f32>, Vec<f32>) {
    let dh = cfg.head_dim;
    let half = dh / 2;
    let mut cos = vec![0f32; half];
    let mut sin = vec![0f32; half];
    for i in 0..half {
        let freq = 1.0 / cfg.rope_theta.powf((2 * i) as f64 / dh as f64);
        let angle = pos as f64 * freq;
        let (s, c) = angle.sin_cos();
        cos[i] = c as f32;
        sin[i] = s as f32;
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Qwen3Config;

    fn tiny_cfg() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 16,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            attention_bias: false,
            qk_norm: true,
            sliding_window: None,
            max_window_layers: usize::MAX,
            use_sliding_window: false,
            num_experts: 0,
            num_experts_used: 0,
            expert_ffn_size: 0,
            shared_expert_ffn_size: 0,
            expert_weights_scale: 1.0,
        }
    }

    fn synthetic_weights(cfg: &Qwen3Config) -> WeightMap {
        let h = cfg.hidden_size;
        let q_dim = cfg.q_proj_dim();
        let kv_dim = cfg.kv_proj_dim();
        let int_dim = cfg.intermediate_size;
        let dh = cfg.head_dim;
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        // Use a deterministic non-zero pattern so logits aren't all 0
        // (sampling on an all-zero row is undefined order).
        let pat = |n: usize, salt: u32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(salt)) >> 8;
                    (x as f32 / (1u32 << 24) as f32) - 0.5
                })
                .collect()
        };
        t.insert(
            "model.embed_tokens.weight".into(),
            (pat(cfg.vocab_size * h, 1), vec![cfg.vocab_size, h]),
        );
        for i in 0..cfg.num_hidden_layers {
            let lp = format!("model.layers.{i}");
            t.insert(
                format!("{lp}.input_layernorm.weight"),
                (pat(h, 100 + i as u32), vec![h]),
            );
            t.insert(
                format!("{lp}.post_attention_layernorm.weight"),
                (pat(h, 200 + i as u32), vec![h]),
            );
            t.insert(
                format!("{lp}.self_attn.q_proj.weight"),
                (pat(q_dim * h, 300 + i as u32), vec![q_dim, h]),
            );
            t.insert(
                format!("{lp}.self_attn.k_proj.weight"),
                (pat(kv_dim * h, 400 + i as u32), vec![kv_dim, h]),
            );
            t.insert(
                format!("{lp}.self_attn.v_proj.weight"),
                (pat(kv_dim * h, 500 + i as u32), vec![kv_dim, h]),
            );
            t.insert(
                format!("{lp}.self_attn.o_proj.weight"),
                (pat(h * q_dim, 600 + i as u32), vec![h, q_dim]),
            );
            t.insert(
                format!("{lp}.self_attn.q_norm.weight"),
                (pat(dh, 700 + i as u32), vec![dh]),
            );
            t.insert(
                format!("{lp}.self_attn.k_norm.weight"),
                (pat(dh, 800 + i as u32), vec![dh]),
            );
            t.insert(
                format!("{lp}.mlp.gate_proj.weight"),
                (pat(int_dim * h, 900 + i as u32), vec![int_dim, h]),
            );
            t.insert(
                format!("{lp}.mlp.up_proj.weight"),
                (pat(int_dim * h, 1000 + i as u32), vec![int_dim, h]),
            );
            t.insert(
                format!("{lp}.mlp.down_proj.weight"),
                (pat(h * int_dim, 1100 + i as u32), vec![h, int_dim]),
            );
        }
        t.insert("model.norm.weight".into(), (pat(h, 2000), vec![h]));
        t.insert(
            "lm_head.weight".into(),
            (pat(cfg.vocab_size * h, 3000), vec![cfg.vocab_size, h]),
        );
        WeightMap::from_tensors(t)
    }

    #[test]
    fn generator_drains_loader_and_runs_one_step() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let mut gn = Qwen3Generator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap();
        assert_eq!(wm.len(), 0, "loader should be drained");
        gn.prefill(&[1, 2, 3]);
        let t = gn.step(SampleOpts::greedy()).unwrap();
        assert!((t as usize) < cfg.vocab_size);
        assert_eq!(gn.tokens().len(), 4);
    }

    /// Fused batched decode must equal independent single-sequence decode,
    /// token-for-token in logits AND in the advanced KV cache. This is the
    /// throughput primitive's correctness oracle: it pins the position-major
    /// `[seq, batch, kv_dim]` KV layout, the per-row mask, and the
    /// bucket-padded custom-mask graph at batch > 1, all on CPU.
    #[test]
    fn batched_decode_matches_single_sequence() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let mut g = Qwen3Generator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap();

        let close = |a: &[f32], b: &[f32], who: &str| {
            assert_eq!(a.len(), b.len(), "{who}: length");
            for (j, (x, y)) in a.iter().zip(b).enumerate() {
                assert!(
                    (x - y).abs() <= 1e-3 + 1e-3 * y.abs(),
                    "{who} elem[{j}]: batched {x} vs single {y}"
                );
            }
        };

        // Cover several prompt lengths: len 4 → bucket 3..5 (upper 4, no pad);
        // len 3 → bucket 3..5 (upper 4, ONE padded row, exercises the mask);
        // len 2 → bucket 2..3 (upper 2, no pad).
        for len in [2usize, 3, 4] {
            let prompt_a: Vec<u32> = (1..=len as u32).collect();
            let prompt_b: Vec<u32> = (1..=len as u32).map(|t| t + 5).collect();
            let tok_a = 5u32;
            let tok_b = 11u32;
            let past = len;

            // Reference A: single-seq decode + its advanced cache.
            g.prefill_get_last_logits(&prompt_a).unwrap();
            let (kv_a, _) = g.export_cache().unwrap();
            let exp_a = g.decode_get_logits(tok_a).unwrap();
            let (kv_a_next, _) = g.export_cache().unwrap();

            // Reference B (fresh prefill resets the cache).
            g.prefill_get_last_logits(&prompt_b).unwrap();
            let (kv_b, _) = g.export_cache().unwrap();
            let exp_b = g.decode_get_logits(tok_b).unwrap();
            let (kv_b_next, _) = g.export_cache().unwrap();

            // B=1 through the batched path must match single-seq decode exactly
            // (isolates mask/rope/graph from the multi-sequence interleaving).
            let solo = g
                .decode_batched_uniform(&[(tok_a, &kv_a)], past, past)
                .unwrap();
            close(&solo[0].0, &exp_a, &format!("len{len} B=1 logits"));

            // Fused: both sequences in ONE batched forward.
            let out = g
                .decode_batched_uniform(&[(tok_a, &kv_a), (tok_b, &kv_b)], past, past)
                .unwrap();
            assert_eq!(out.len(), 2);
            close(&out[0].0, &exp_a, &format!("len{len} seq A logits"));
            close(&out[1].0, &exp_b, &format!("len{len} seq B logits"));

            // The de-interleaved KV must equal each single-seq advanced cache.
            assert_eq!(out[0].1.past_len, past + 1);
            assert_eq!(out[1].1.past_len, past + 1);
            for l in 0..cfg.num_hidden_layers {
                close(
                    &out[0].1.layers_k[l],
                    &kv_a_next.layers_k[l],
                    &format!("len{len} A K"),
                );
                close(
                    &out[0].1.layers_v[l],
                    &kv_a_next.layers_v[l],
                    &format!("len{len} A V"),
                );
                close(
                    &out[1].1.layers_k[l],
                    &kv_b_next.layers_k[l],
                    &format!("len{len} B K"),
                );
                close(
                    &out[1].1.layers_v[l],
                    &kv_b_next.layers_v[l],
                    &format!("len{len} B V"),
                );
            }
        }
    }

    /// Ragged fused decode (sequences at DIFFERENT cache lengths) must equal
    /// independent single-sequence decode — logits and advanced KV. This pins
    /// the per-sequence RoPE table + per-row mask path that lets arbitrary
    /// concurrent requests fuse into one forward.
    #[test]
    fn ragged_batched_decode_matches_single_sequence() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let mut g = Qwen3Generator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap();

        let close = |a: &[f32], b: &[f32], who: &str| {
            assert_eq!(a.len(), b.len(), "{who} len");
            for (j, (x, y)) in a.iter().zip(b).enumerate() {
                assert!(
                    (x - y).abs() <= 1e-3 + 1e-3 * y.abs(),
                    "{who}[{j}]: {x} vs {y}"
                );
            }
        };

        // Seq A at past_len 2, seq B at past_len 5 — different RoPE positions,
        // different mask rows, padded to a shared bucket upper.
        let prompt_a = vec![1u32, 2];
        let prompt_b = vec![3u32, 4, 5, 6, 7];
        let tok_a = 8u32;
        let tok_b = 9u32;

        g.prefill_get_last_logits(&prompt_a).unwrap();
        let (kv_a, _) = g.export_cache().unwrap();
        let exp_a = g.decode_get_logits(tok_a).unwrap();
        let (kv_a_next, _) = g.export_cache().unwrap();

        g.prefill_get_last_logits(&prompt_b).unwrap();
        let (kv_b, _) = g.export_cache().unwrap();
        let exp_b = g.decode_get_logits(tok_b).unwrap();
        let (kv_b_next, _) = g.export_cache().unwrap();

        let out = g
            .decode_batched_ragged(&[(tok_a, &kv_a), (tok_b, &kv_b)])
            .unwrap();
        assert_eq!(out.len(), 2);
        close(&out[0].0, &exp_a, "ragged A logits");
        close(&out[1].0, &exp_b, "ragged B logits");

        // Order-independence: swapping the batch must swap the outputs, not
        // corrupt them (guards the per-sequence RoPE/mask indexing).
        let sw = g
            .decode_batched_ragged(&[(tok_b, &kv_b), (tok_a, &kv_a)])
            .unwrap();
        close(&sw[0].0, &exp_b, "swapped B logits");
        close(&sw[1].0, &exp_a, "swapped A logits");

        assert_eq!(out[0].1.past_len, prompt_a.len() + 1);
        assert_eq!(out[1].1.past_len, prompt_b.len() + 1);
        for l in 0..cfg.num_hidden_layers {
            close(&out[0].1.layers_k[l], &kv_a_next.layers_k[l], "ragged A K");
            close(&out[0].1.layers_v[l], &kv_a_next.layers_v[l], "ragged A V");
            close(&out[1].1.layers_k[l], &kv_b_next.layers_k[l], "ragged B K");
            close(&out[1].1.layers_v[l], &kv_b_next.layers_v[l], "ragged B V");
        }
    }

    #[test]
    fn generate_n_appends_n_tokens() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let mut gn = Qwen3Generator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap();
        gn.prefill(&[5, 6]);
        let new_tokens = gn.generate(3, SampleOpts::greedy()).unwrap();
        assert_eq!(new_tokens.len(), 3);
        assert_eq!(gn.tokens().len(), 5);
        for t in &new_tokens {
            assert!((*t as usize) < cfg.vocab_size);
        }
    }

    #[test]
    fn step_without_prefill_errors() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let mut gn = Qwen3Generator::from_loader(cfg, &mut wm, Device::Cpu).unwrap();
        let r = gn.step(SampleOpts::greedy());
        assert!(r.is_err());
    }

    #[test]
    fn cached_matches_naive_on_greedy() {
        // The cached and naive paths must produce the same token
        // sequence given the same prompt + opts. This is the
        // load-bearing test for the KV-cache implementation: if the
        // decode-mode graph, the kernel's Lq!=Lk fix, the cache
        // wiring, or the RoPE position-slice is wrong, the sequences
        // diverge here.
        let cfg = tiny_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5];
        let steps = 4;

        let mut wm_n = synthetic_weights(&cfg);
        let mut gn_naive =
            Qwen3Generator::from_loader(cfg.clone(), &mut wm_n, Device::Cpu).unwrap();
        gn_naive.prefill_compile_cache = None;
        gn_naive.decode_compile_cache = None;
        gn_naive.prefill(&prompt);
        let naive_tokens = gn_naive.generate(steps, SampleOpts::greedy()).unwrap();

        let mut wm_c = synthetic_weights(&cfg);
        let mut gn_cached =
            Qwen3Generator::from_loader(cfg.clone(), &mut wm_c, Device::Cpu).unwrap();
        gn_cached.prefill(&prompt);
        let cached_tokens = gn_cached
            .generate_cached(steps, SampleOpts::greedy())
            .unwrap();

        assert_eq!(
            cached_tokens, naive_tokens,
            "cached vs naive token mismatch — KV cache or kernel-Lq!=Lk bug"
        );
    }

    #[test]
    fn sliding_window_cached_decode_matches_naive() {
        // With a sliding window, the KV-cached decode path (windowed custom
        // mask) must reproduce naive prefill-recompute (SlidingWindow op),
        // generating past the window so the masking actually bites.
        let mut cfg = tiny_cfg();
        cfg.use_sliding_window = true;
        cfg.sliding_window = Some(3);
        let prompt: Vec<u32> = vec![1, 2, 3, 5, 4];
        let steps = 5;

        let mut wm_n = synthetic_weights(&cfg);
        let mut gn_naive =
            Qwen3Generator::from_loader(cfg.clone(), &mut wm_n, Device::Cpu).unwrap();
        gn_naive.prefill_compile_cache = None;
        gn_naive.decode_compile_cache = None;
        gn_naive.prefill(&prompt);
        let naive = gn_naive.generate(steps, SampleOpts::greedy()).unwrap();

        let mut wm_c = synthetic_weights(&cfg);
        let mut gn_cached =
            Qwen3Generator::from_loader(cfg.clone(), &mut wm_c, Device::Cpu).unwrap();
        gn_cached.prefill(&prompt);
        let cached = gn_cached
            .generate_cached(steps, SampleOpts::greedy())
            .unwrap();

        assert_eq!(
            cached, naive,
            "windowed cached decode must match naive windowed prefill"
        );
    }

    #[test]
    fn sliding_window_rotates_cache_to_bound_memory() {
        // The rotating cache stays at `window` rows no matter how long the
        // prompt + generation get — O(window) memory instead of O(sequence).
        let mut cfg = tiny_cfg();
        cfg.use_sliding_window = true;
        cfg.sliding_window = Some(3);
        let mut wm = synthetic_weights(&cfg);
        let mut gn = Qwen3Generator::from_loader(cfg, &mut wm, Device::Cpu).unwrap();
        gn.prefill(&[1, 2, 3, 5, 4, 6, 7]); // prompt len 7 > window 3
        let _ = gn.generate_cached(5, SampleOpts::greedy()).unwrap();
        let cache = gn.cache.as_ref().unwrap();
        assert_eq!(cache.past_len, 3, "cache must be bounded to the window");
        // And each layer's K/V buffer holds exactly `window` rows.
        let kv_dim = gn.cfg.kv_proj_dim();
        assert_eq!(cache.layers_k[0].len(), 3 * kv_dim);
    }

    #[test]
    fn sliding_window_reduces_to_causal_when_wide() {
        // A sliding window ≥ seq must reproduce full causal attention; a
        // narrow window must change the result (the mask actually restricts).
        let prompt: Vec<u32> = vec![1, 2, 3, 5, 4];
        let base = tiny_cfg();

        let logits = |mut cfg: Qwen3Config, use_sw: bool, win: Option<usize>| {
            cfg.use_sliding_window = use_sw;
            cfg.sliding_window = win;
            let mut wm = synthetic_weights(&cfg);
            let mut g = Qwen3Generator::from_loader(cfg, &mut wm, Device::Cpu).unwrap();
            g.sequence_logits(&prompt).unwrap()
        };

        let causal = logits(base.clone(), false, None);
        let wide = logits(base.clone(), true, Some(100)); // ≥ seq
        let narrow = logits(base.clone(), true, Some(2)); // < seq

        assert_eq!(causal.len(), wide.len());
        for (a, b) in causal.iter().zip(&wide) {
            assert!(
                (a - b).abs() < 1e-4,
                "wide window must equal causal: {a} vs {b}"
            );
        }
        let diff: f32 = causal.iter().zip(&narrow).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-3,
            "narrow window should differ from causal (sum |Δ|={diff})"
        );
    }

    #[test]
    fn sequence_logprobs_shape_and_consistency() {
        let cfg = tiny_cfg();
        let vocab = cfg.vocab_size;
        let tokens: Vec<u32> = vec![1, 2, 3, 5, 4];
        let mut wm = synthetic_weights(&cfg);
        let mut g = Qwen3Generator::from_loader(cfg, &mut wm, Device::Cpu).unwrap();

        let logits = g.sequence_logits(&tokens).unwrap();
        assert_eq!(logits.len(), tokens.len() * vocab);

        let lps = g.sequence_logprobs(&tokens).unwrap();
        assert_eq!(lps.len(), tokens.len() - 1);
        // All log-probs are <= 0 and finite.
        assert!(lps.iter().all(|&x| x <= 1e-4 && x.is_finite()));
        // Consistent with a hand log-softmax of the corresponding row.
        let row0 = &logits[0..vocab];
        let hand = log_softmax_at(row0, tokens[1] as usize);
        assert!((lps[0] - hand).abs() < 1e-5);
    }

    #[test]
    fn prefill_with_reuse_matches_full_prefill() {
        // Reusing a cached prefix + decoding the suffix must predict the
        // same next token as a full prefill of the whole prompt.
        let cfg = tiny_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5, 4, 6];

        let mut wm_ref = synthetic_weights(&cfg);
        let mut g_ref = Qwen3Generator::from_loader(cfg.clone(), &mut wm_ref, Device::Cpu)
            .unwrap()
            .with_decode_cache(64);
        let ref_logits = g_ref.prefill_get_last_logits(&prompt).unwrap();
        let ref_tok = sample_token(&ref_logits, SampleOpts::greedy());

        // Build a prefix cache covering the first 3 tokens.
        let mut wm_p = synthetic_weights(&cfg);
        let mut g_pref = Qwen3Generator::from_loader(cfg.clone(), &mut wm_p, Device::Cpu)
            .unwrap()
            .with_decode_cache(64);
        let _ = g_pref.prefill_get_last_logits(&prompt[..3]).unwrap();
        let (prefix_kv, prefix_tokens) = g_pref.export_cache().unwrap();
        assert_eq!(prefix_kv.past_len, 3);
        assert_eq!(prefix_tokens, prompt[..3].to_vec());

        // Reuse the prefix, prefill only the suffix.
        let mut wm_r = synthetic_weights(&cfg);
        let mut g_reuse = Qwen3Generator::from_loader(cfg.clone(), &mut wm_r, Device::Cpu)
            .unwrap()
            .with_decode_cache(64);
        let reuse_logits = g_reuse.prefill_with_reuse(&prompt, prefix_kv, 3).unwrap();
        let reuse_tok = sample_token(&reuse_logits, SampleOpts::greedy());

        assert_eq!(
            reuse_tok, ref_tok,
            "suffix-prefill must predict the same next token as full prefill"
        );
        // Cache is left ready for decode at the full prompt length.
        assert_eq!(g_reuse.cache.as_ref().unwrap().past_len, prompt.len());
    }

    #[test]
    fn cached_step_advances_cache_invariant() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let mut gn = Qwen3Generator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap();
        gn.prefill(&[1, 2, 3]);
        let _ = gn.step_cached(SampleOpts::greedy()).unwrap();
        // After seed: tokens.len() == 4, cache.past_seq == 3 (cache holds prompt).
        assert_eq!(gn.tokens().len(), 4);
        assert_eq!(gn.cache.as_ref().unwrap().past_len, 3);
        let _ = gn.step_cached(SampleOpts::greedy()).unwrap();
        // After one decode: tokens.len() == 5, cache.past_seq == 4.
        assert_eq!(gn.tokens().len(), 5);
        assert_eq!(gn.cache.as_ref().unwrap().past_len, 4);
    }

    #[test]
    fn bucketed_decode_matches_oneshot() {
        // The bucketed compile-cache path (padded K/V + custom mask)
        // must produce the same token sequence as the one-shot
        // path. Load-bearing for the bucketed cache feature: if the
        // mask, padding, or output slicing is wrong, sequences
        // diverge here.
        let cfg = tiny_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5];
        let steps = 6;

        let mut wm_one = synthetic_weights(&cfg);
        let mut gn_one =
            Qwen3Generator::from_loader(cfg.clone(), &mut wm_one, Device::Cpu).unwrap();
        gn_one.prefill(&prompt);
        let oneshot_tokens = gn_one.generate_cached(steps, SampleOpts::greedy()).unwrap();

        let mut wm_buc = synthetic_weights(&cfg);
        let mut gn_buc = Qwen3Generator::from_loader(cfg.clone(), &mut wm_buc, Device::Cpu)
            .unwrap()
            .with_decode_cache(/*max_past*/ 32);
        gn_buc.prefill(&prompt);
        let bucketed_tokens = gn_buc.generate_cached(steps, SampleOpts::greedy()).unwrap();

        assert_eq!(
            bucketed_tokens, oneshot_tokens,
            "bucketed-cache decode diverged from one-shot decode — \
             mask, padding, or output-slice bug"
        );
    }

    #[test]
    fn bucketed_decode_q_proj_seq_is_one() {
        use rlx_ir::Op;

        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let (graph, _) = build_qwen3_decode_graph_sized_ext(&cfg, &mut wm, 1, 4, true).unwrap();
        for node in graph.nodes() {
            if let Op::MatMul = &node.op {
                let sh = graph.shape(node.id);
                if sh.rank() == 3 && sh.dim(2).unwrap_static() == cfg.q_proj_dim() {
                    assert_eq!(
                        sh.dim(1).unwrap_static(),
                        1,
                        "decode q_proj matmul seq dim must be 1, got {sh} on node {}",
                        node.id
                    );
                }
            }
        }

        let fused = rlx_opt::CompilePipeline::new(rlx_opt::FusionTarget::Metal)
            .with_assert_fusion_clean(false)
            .compile_graph(graph)
            .lir
            .into_graph();
        for node in fused.nodes() {
            if let Op::Narrow { len, .. } = &node.op {
                let sh = fused.shape(node.id);
                if sh.rank() == 3 && *len == cfg.q_proj_dim() {
                    assert_eq!(
                        sh.dim(1).unwrap_static(),
                        1,
                        "fused decode q narrow seq dim must be 1, got {sh} on node {}",
                        node.id
                    );
                }
            }
        }
    }

    #[test]
    fn prefill_compile_cache_does_not_change_output() {
        let cfg = tiny_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5];
        let mut wm_a = synthetic_weights(&cfg);
        let mut gn_a = Qwen3Generator::from_loader(cfg.clone(), &mut wm_a, Device::Cpu).unwrap();
        gn_a.prefill(&prompt);
        let a = gn_a.generate_cached(4, SampleOpts::greedy()).unwrap();

        let mut wm_b = synthetic_weights(&cfg);
        let mut gn_b = Qwen3Generator::from_loader(cfg.clone(), &mut wm_b, Device::Cpu)
            .unwrap()
            .with_prefill_cache(/*capacity*/ 4);
        gn_b.prefill(&prompt);
        let b = gn_b.generate_cached(4, SampleOpts::greedy()).unwrap();

        assert_eq!(a, b, "enabling prefill_cache must not change output");
    }

    #[test]
    fn greedy_is_deterministic_across_runs() {
        let cfg = tiny_cfg();
        let weights = synthetic_weights(&cfg);
        let mk = || {
            let mut wm = WeightMap::from_tensors(weights_as_hashmap(&weights));
            Qwen3Generator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap()
        };
        let mut a = mk();
        let mut b = mk();
        a.prefill(&[1, 2, 3]);
        b.prefill(&[1, 2, 3]);
        let ta = a.generate(4, SampleOpts::greedy()).unwrap();
        let tb = b.generate(4, SampleOpts::greedy()).unwrap();
        assert_eq!(ta, tb);
    }

    fn weights_as_hashmap(wm: &WeightMap) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
        // Reconstruct the underlying map by re-running synthetic_weights
        // — WeightMap doesn't expose its inner map. Sufficient for the
        // determinism test since synthetic_weights is itself
        // deterministic.
        let _ = wm; // silence unused
        let cfg = tiny_cfg();
        let mut new = synthetic_weights(&cfg);
        let keys: Vec<String> = new.keys().map(|s| s.to_string()).collect();
        let mut out = HashMap::new();
        for k in keys {
            out.insert(k.clone(), new.take(&k).unwrap());
        }
        out
    }
}
