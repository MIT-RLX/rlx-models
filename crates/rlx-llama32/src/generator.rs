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

//! Host-side generation loop for LLaMA-3.2.
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
//!   - Provides a reference oracle for the cached version's own
//!     numerical-parity test (cached vs recompute must match).

use crate::builder::{
    build_llama32_decode_hir_dynamic_ext, build_llama32_decode_hir_sized,
    build_llama32_decode_hir_sized_ext, build_llama32_graph_sized_last_logits,
    build_llama32_prefill_hir_dynamic_ext,
};
use crate::config::Llama32Config;
use crate::rope::{resolve_inv_freq, rope_slice};
use anyhow::{Context, Result};
use rlx_core::flow_bridge::compile_options_from_profile;
use rlx_core::weight_loader::WeightLoader;
use rlx_core::weight_map::WeightMap;
use rlx_flow::CompileProfile;
use rlx_ir::DimBinding;
use rlx_ir::logical_kernel::KernelDispatchConfig;
use rlx_qwen3::sampling::{SampleOpts, sample_token};
use rlx_runtime::compile_cache::{BucketedCompileCache, CompileCache, DynamicDimCompileCache};
use rlx_runtime::{CompileOptions, Device, Session};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// MPSGraph on Metal has two known gaps in `rlx-metal` 0.2.1:
///
/// 1. **Decode attention** — assumes `seq_q == seq_kv`; KV-cache decode breaks.
/// 2. **Packed GGUF `DequantMatMul`** — lowers U8 quant bytes as F32 constants
///    (`w_bytes.len() == k*n*4`), destroying parity vs CPU thunks.
///
/// Force the Metal thunk path for affected compiles until `rlx-metal` ≥ 0.2.2.
fn metal_thunk_compile_guard<R, F>(device: Device, f: F) -> R
where
    F: FnOnce() -> R,
{
    if device == Device::Metal {
        rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");
        let out = f();
        rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
        out
    } else {
        f()
    }
}

fn metal_decode_compile_guard<R, F>(device: Device, decode: bool, f: F) -> R
where
    F: FnOnce() -> R,
{
    if decode {
        metal_thunk_compile_guard(device, f)
    } else {
        f()
    }
}

/// Per-layer KV cache state for incremental decoding. Each `Vec<f32>`
/// is a flat `[batch, past_seq, kv_proj_dim]` tensor.
#[derive(Clone)]
struct KvCacheState {
    past_seq: usize,
    layers_k: Vec<Vec<f32>>,
    layers_v: Vec<Vec<f32>>,
}

/// Stateful LLaMA-3.2 generation handle.
///
/// Holds the (config, weight bytes, token history) and rebuilds a
/// prefill graph on each [`step`] call. Cheap to construct after
/// initial weight load; tokens stay in-memory between calls.
pub struct Llama32Generator {
    cfg: Llama32Config,
    /// Map of weight key → (f32 data, shape). Cloned on each step
    /// into a fresh `WeightMap` because `WeightMap::take` is
    /// destructive — see the cached-generator notes for the path
    /// that avoids the clone.
    weights_cache: HashMap<String, (Vec<f32>, Vec<usize>)>,
    tokens: Vec<u32>,
    device: Device,
    /// Populated lazily on the first `step_cached` call (seeded from
    /// the prompt via prefill-with-cache); thereafter advanced by each
    /// decode step.
    cache: Option<KvCacheState>,
    /// Per-key LRU compile cache for prefill graphs. Keyed by `seq`.
    /// Set to `None` to disable (default for new instances; opt in via
    /// [`Llama32Generator::with_prefill_cache`]).
    prefill_compile_cache: Option<CompileCache>,
    /// Compile prefill once with `sym::SEQ`, specialize per prompt length.
    prefill_dynamic_cache: Option<DynamicDimCompileCache>,
    /// Bucketed compile cache for decode-mode graphs. Each bucket
    /// holds one compiled graph specialized at its upper-bound
    /// `past_seq`; the host pads `past_k`/`past_v` and supplies a
    /// per-step mask so a single bucket serves every `past_seq` in
    /// its range. Opt in via [`Llama32Generator::with_decode_cache`].
    decode_compile_cache: Option<BucketedCompileCache>,
    decode_dynamic_cache: Option<DynamicDimCompileCache>,
    /// Tracks which decode buckets have had params attached. The
    /// `BucketedCompileCache` API doesn't expose per-bucket compile
    /// status, so we maintain it here to avoid double-loading params.
    decode_loaded_buckets: HashSet<usize>,
    /// Upper bound for dynamic HIR `SEQ` / `PAST_SEQ` (defaults to
    /// [`Llama32Config::max_position_embeddings`]).
    compile_seq_cap: Option<usize>,
    /// Resolved RoPE inverse frequencies (includes Llama 3 scaling).
    inv_freq: Vec<f64>,
    /// Tier-1 compile profile for prefill graphs.
    prefill_profile: CompileProfile,
    /// Tier-1 compile profile for decode graphs.
    decode_profile: CompileProfile,
}

impl Llama32Generator {
    /// Construct from any [`WeightLoader`] — drains it into an
    /// internal cache so the loader is free after this call.
    pub fn from_loader(
        cfg: Llama32Config,
        loader: &mut dyn WeightLoader,
        device: Device,
    ) -> Result<Self> {
        let keys = loader.remaining_keys();
        let mut weights_cache = HashMap::with_capacity(keys.len());
        for k in keys {
            let v = loader
                .take(&k)
                .with_context(|| format!("draining weight {k}"))?;
            // Normalize the cache key to the safetensors / HuggingFace
            // naming convention so subsequent builder calls that ask
            // for `model.embed_tokens.weight` (the canonical name baked
            // into the llama32 builder) hit the cache whether the
            // loader was safetensors-native or GGUF-native.
            let canonical =
                rlx_core::weight_loader::gguf_to_hf_name(&k).unwrap_or_else(|| k.clone());
            weights_cache.insert(canonical, v);
        }
        let rope_factors = weights_cache
            .get("rope_freqs.weight")
            .map(|(d, _)| d.as_slice());
        let inv_freq = resolve_inv_freq(&cfg, rope_factors);
        Ok(Self {
            cfg,
            weights_cache,
            tokens: Vec::new(),
            device,
            cache: None,
            prefill_compile_cache: None,
            prefill_dynamic_cache: None,
            decode_compile_cache: None,
            decode_dynamic_cache: None,
            decode_loaded_buckets: HashSet::new(),
            compile_seq_cap: None,
            inv_freq,
            prefill_profile: CompileProfile::llama32_prefill(),
            decode_profile: CompileProfile::llama32_decode(),
        })
    }

    fn compile_seq_cap(&self) -> usize {
        self.compile_seq_cap
            .unwrap_or(self.cfg.max_position_embeddings)
    }

    /// Cap symbolic seq / past-seq in dynamic compile paths. Use for models
    /// with very large `max_position_embeddings` (e.g. 128k) when the runner
    /// only needs a short window.
    pub fn with_compile_seq_cap(mut self, cap: usize) -> Self {
        self.compile_seq_cap = Some(cap.max(1));
        self
    }

    /// Like [`Self::from_loader`] but loads tier-1 profiles from
    /// `llama32.rlx.toml` in the weights directory when present.
    pub fn from_loader_at(
        cfg: Llama32Config,
        loader: &mut dyn WeightLoader,
        device: Device,
        weights_path: &Path,
    ) -> Result<Self> {
        let mut g = Self::from_loader(cfg, loader, device)?;
        g.prefill_profile = crate::llama32_profile_near_weights(weights_path, false);
        g.decode_profile = crate::llama32_profile_near_weights(weights_path, true);
        Ok(g)
    }

    /// Override tier-1 compile profiles explicitly.
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

    fn compile_hir_profiled(
        &self,
        session: &Session,
        hir: rlx_ir::hir::HirModule,
        decode: bool,
    ) -> Result<rlx_runtime::CompiledGraph> {
        let opts = self.profile_compile_options(decode);
        Ok(metal_decode_compile_guard(self.device, decode, || {
            session.compile_hir_with(hir, &opts)
        })?)
    }

    fn compile_graph_profiled(
        &self,
        session: &Session,
        graph: rlx_ir::Graph,
    ) -> Result<rlx_runtime::CompiledGraph> {
        let opts = self.profile_compile_options(false);
        Ok(session.compile_with(graph, &opts))
    }

    /// Enable the prefill compile cache with the given LRU capacity.
    /// Useful when the same prompt length is used across multiple
    /// generation runs — the second + Nth run skip the compile +
    /// param-attach roundtrip (~30-50ms per call on CPU).
    pub fn with_prefill_cache(mut self, capacity: usize) -> Self {
        self.prefill_compile_cache = Some(CompileCache::new(self.device, capacity));
        self.prefill_dynamic_cache = None;
        self
    }

    /// Compile prefill once with `sym::SEQ`, specialize per prompt length.
    pub fn with_dynamic_prefill_cache(mut self, capacity: usize) -> Self {
        self.prefill_dynamic_cache = Some(DynamicDimCompileCache::new(self.device, capacity));
        self.prefill_compile_cache = None;
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
    pub fn with_decode_cache(mut self, max_past: usize) -> Self {
        let cache = BucketedCompileCache::power_of_two_ladder(
            self.device,
            /*min*/ 1,
            max_past.max(1) as u64,
        );
        self.decode_compile_cache = Some(cache);
        self.decode_dynamic_cache = None;
        self.decode_loaded_buckets.clear();
        self
    }

    /// Compile decode once with `sym::PAST_SEQ`, specialize per prefix length.
    pub fn with_dynamic_decode_cache(mut self, capacity: usize) -> Self {
        self.decode_dynamic_cache = Some(DynamicDimCompileCache::new(self.device, capacity));
        self.decode_compile_cache = None;
        self.decode_loaded_buckets.clear();
        self
    }

    /// Convenience: load weights from a safetensors or GGUF path
    /// (dispatch by extension; see `rlx_core::weight_loader::load_from_path`).
    pub fn from_path(cfg: Llama32Config, path: &str, device: Device) -> Result<Self> {
        let mut loader = rlx_core::weight_loader::load_from_path(path)?;
        Self::from_loader(cfg, loader.as_mut(), device)
    }

    /// Same as [`from_path`] but with MTP-head visibility control.
    /// When `include_mtp=true` and the file is GGUF, MTP weights are
    /// drained into the generator's cache alongside the base
    /// weights. The base inference path still ignores them — they
    /// sit in cache for a future MTP-aware decoder. Non-GGUF formats
    /// silently ignore the flag (safetensors files publish all
    /// tensors uniformly; downstream code distinguishes by name).
    pub fn from_path_with_mtp(
        cfg: Llama32Config,
        path: &str,
        device: Device,
        include_mtp: bool,
    ) -> Result<Self> {
        // Branch on extension so we can flip the GGUF-specific
        // visibility knob. Safetensors has no equivalent — it
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
        let mut wm = WeightMap::from_tensors(self.weights_cache.clone());
        let (graph, params) = build_llama32_graph_sized_last_logits(
            &self.cfg, &mut wm, /*batch*/ 1, seq, /*with_kv_outputs*/ false,
        )?;
        let session = Session::new(self.device);
        let mut compiled = self.compile_graph_profiled(&session, graph)?;
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
            return Ok(tok);
        }
        let cache = self.cache.as_ref().unwrap();
        let past_seq = cache.past_seq;
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
        let input_tok = self.tokens[past_seq];

        // Branch: bucketed compile cache vs one-shot compile per step.
        let (logits, new_k, new_v) = if self.decode_dynamic_cache.is_some() {
            self.decode_step_dynamic(past_seq, input_tok)?
        } else if self.decode_compile_cache.is_some()
            && self
                .decode_compile_cache
                .as_ref()
                .unwrap()
                .bucket_for(past_seq as u64)
                .is_some()
        {
            self.decode_step_bucketed(past_seq, input_tok)?
        } else {
            self.decode_step_oneshot(past_seq, input_tok)?
        };

        let cache_mut = self.cache.as_mut().unwrap();
        cache_mut.past_seq = past_seq + 1;
        cache_mut.layers_k = new_k;
        cache_mut.layers_v = new_v;

        let vocab = self.cfg.vocab_size;
        if logits.len() != vocab {
            anyhow::bail!("decode logits length {} != vocab {}", logits.len(), vocab);
        }
        let tok = sample_token(&logits, opts) as u32;
        self.tokens.push(tok);
        Ok(tok)
    }

    /// Decode path that compiles a fresh graph for the exact `past_seq`
    /// every call. Slower but always-correct fallback.
    #[allow(clippy::type_complexity)]
    fn decode_step_oneshot(
        &mut self,
        past_seq: usize,
        input_tok: u32,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let cache = self.cache.as_ref().unwrap();

        let mut wm = WeightMap::from_tensors(self.weights_cache.clone());
        let (hir, params) =
            build_llama32_decode_hir_sized(&self.cfg, &mut wm, /*batch*/ 1, past_seq)?;
        let session = Session::new(self.device);
        let mut compiled = self.compile_hir_profiled(&session, hir, true)?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }

        let (cos, sin) = compute_rope_slice(&self.inv_freq, past_seq);
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
        self.split_decode_outputs(outputs)
    }

    #[allow(clippy::type_complexity)]
    fn decode_step_dynamic(
        &mut self,
        past_seq: usize,
        input_tok: u32,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let cache = self.cache.as_ref().unwrap();
        let binding = DimBinding::batch_past_seq(1, past_seq);
        let opts = self
            .profile_compile_options(true)
            .dim_binding(binding.clone());
        let max_past = self.compile_seq_cap();
        let cache_dyn = self
            .decode_dynamic_cache
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("dynamic decode without cache"))?;
        let needs_upload = !cache_dyn.contains(past_seq as u64);
        let cfg = self.cfg.clone();
        let weights_cache = self.weights_cache.clone();
        let device = self.device;
        let compiled = cache_dyn.get_or_specialize(
            past_seq as u64,
            &binding,
            || {
                metal_decode_compile_guard(device, true, || {
                    let mut wm = WeightMap::from_tensors(weights_cache);
                    build_llama32_decode_hir_dynamic_ext(&cfg, &mut wm, 1, max_past)
                        .expect("dynamic decode HIR")
                        .0
                })
            },
            &opts,
        )?;
        if needs_upload {
            let mut wm = WeightMap::from_tensors(self.weights_cache.clone());
            let (_, params) =
                build_llama32_decode_hir_dynamic_ext(&self.cfg, &mut wm, 1, max_past)?;
            for (name, data) in &params {
                compiled.set_param(name, data);
            }
        }

        let (cos, sin) = compute_rope_slice(&self.inv_freq, past_seq);
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
        self.split_decode_outputs(outputs)
    }

    /// Decode path using the bucketed compile cache. Compiles one graph
    /// per bucket (instead of per `past_seq`), pads `past_k`/`past_v` to
    /// the bucket's upper bound, and uses a custom mask to zero out the
    /// padded K positions in attention. After running, slices the
    /// `new_k`/`new_v` outputs back to `actual_past + 1` length so the
    /// stored cache stays compact.
    #[allow(clippy::type_complexity)]
    fn decode_step_bucketed(
        &mut self,
        past_seq: usize,
        input_tok: u32,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let cache_dec = self.decode_compile_cache.as_ref().unwrap();
        let bucket_idx = cache_dec
            .bucket_for(past_seq as u64)
            .ok_or_else(|| anyhow::anyhow!("past_seq {past_seq} outside any bucket"))?;
        let upper = cache_dec
            .buckets()
            .nth(bucket_idx)
            .map(|r| r.end - 1)
            .unwrap() as usize;

        let kv_dim = self.cfg.kv_proj_dim();
        let n_layers = self.cfg.num_hidden_layers;

        // First-time-in-bucket: build the graph + compile + attach
        // params, then mark the bucket as loaded. Subsequent calls skip
        // all of this and just .run() the cached graph.
        let needs_load = !self.decode_loaded_buckets.contains(&bucket_idx);
        if needs_load {
            let mut wm = WeightMap::from_tensors(self.weights_cache.clone());
            let (hir, params) = build_llama32_decode_hir_sized_ext(
                &self.cfg, &mut wm, /*batch*/ 1, upper, /*use_custom_mask*/ true,
            )?;
            {
                let decode_opts = self.profile_compile_options(true);
                let cache_mut = self.decode_compile_cache.as_mut().unwrap();
                metal_decode_compile_guard(self.device, true, || {
                    let (_u, compiled) = cache_mut
                        .get_or_compile_hir_with_options(
                            past_seq as u64,
                            |_upper| hir,
                            &decode_opts,
                        )
                        .expect("bucket must exist; we just looked it up");
                    for (name, data) in &params {
                        compiled.set_param(name, data);
                    }
                });
            }
            self.decode_loaded_buckets.insert(bucket_idx);
        }

        // Prepare host-side inputs.
        let (cos, sin) = compute_rope_slice(&self.inv_freq, past_seq);
        let input_ids_f32 = [input_tok as f32];

        // Mask: shape [1, upper + 1]. 1.0 at positions 0..(past_seq + 1),
        // 0.0 at (past_seq + 1)..(upper + 1). Without the mask the padded
        // zero rows would still steal softmax weight (e^0 = 1 per pad
        // position) and silently scale the output down.
        let mask_len = upper + 1;
        let mut mask = vec![0.0f32; mask_len];
        for v in mask.iter_mut().take(past_seq + 1) {
            *v = 1.0;
        }

        // Pad past_k / past_v to length `upper`.
        let padded_k: Vec<Vec<f32>> = (0..n_layers)
            .map(|i| {
                let src = &self.cache.as_ref().unwrap().layers_k[i];
                let mut out = vec![0f32; upper * kv_dim];
                out[..src.len()].copy_from_slice(src);
                out
            })
            .collect();
        let padded_v: Vec<Vec<f32>> = (0..n_layers)
            .map(|i| {
                let src = &self.cache.as_ref().unwrap().layers_v[i];
                let mut out = vec![0f32; upper * kv_dim];
                out[..src.len()].copy_from_slice(src);
                out
            })
            .collect();

        let key_strs: Vec<String> = (0..n_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        let mut inputs: Vec<(&str, &[f32])> = Vec::with_capacity(4 + 2 * n_layers);
        inputs.push(("input_ids", input_ids_f32.as_slice()));
        inputs.push(("rope_cos", cos.as_slice()));
        inputs.push(("rope_sin", sin.as_slice()));
        inputs.push(("mask", mask.as_slice()));
        for i in 0..n_layers {
            inputs.push((&key_strs[2 * i], padded_k[i].as_slice()));
            inputs.push((&key_strs[2 * i + 1], padded_v[i].as_slice()));
        }

        let cache_mut = self.decode_compile_cache.as_mut().unwrap();
        let (_u, compiled) = cache_mut
            .get_or_compile_hir(past_seq as u64, |_| {
                unreachable!("bucket was just loaded above")
            })
            .unwrap();
        let raw_outputs = compiled.run(&inputs);

        // The graph emits new_k/new_v at length `upper + 1` (padded
        // past + the new token). Slice each back to `past_seq + 1` so
        // the stored cache only holds real positions.
        let mut iter = raw_outputs.into_iter();
        let logits = iter.next().context("bucketed decode logits missing")?;
        let real_len = (past_seq + 1) * kv_dim;
        let mut new_k = Vec::with_capacity(n_layers);
        let mut new_v = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let k = iter.next().context("bucketed k missing")?;
            let v = iter.next().context("bucketed v missing")?;
            new_k.push(k[..real_len].to_vec());
            new_v.push(v[..real_len].to_vec());
        }
        Ok((logits, new_k, new_v))
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
        let compile_cap = self.compile_seq_cap();
        let dynamic_prefill = self.prefill_dynamic_cache.is_some().then(|| {
            let binding = DimBinding::batch_seq(batch, seq);
            let opts = self
                .profile_compile_options(false)
                .dim_binding(binding.clone());
            (binding, opts)
        });
        if let (Some(cache), Some((binding, opts))) = (
            self.prefill_dynamic_cache.as_mut(),
            dynamic_prefill.as_ref(),
        ) {
            let max_seq = compile_cap;
            let needs_upload = !cache.contains(seq as u64);
            let cfg = self.cfg.clone();
            let weights_cache = self.weights_cache.clone();
            let compiled = cache.get_or_specialize(
                seq as u64,
                binding,
                || {
                    let mut wm = WeightMap::from_tensors(weights_cache);
                    build_llama32_prefill_hir_dynamic_ext(&cfg, &mut wm, batch, max_seq, true)
                        .expect("dynamic prefill HIR")
                        .0
                },
                opts,
            )?;
            if needs_upload {
                let mut wm = WeightMap::from_tensors(self.weights_cache.clone());
                let (_, params) = build_llama32_prefill_hir_dynamic_ext(
                    &self.cfg, &mut wm, batch, max_seq, true,
                )?;
                for (name, data) in &params {
                    compiled.set_param(name, data);
                }
            }
            let last_idx = vec![(seq - 1) as f32];
            Ok(compiled.run(&[("input_ids", ids_f32), ("last_token_idx", &last_idx)]))
        } else if let Some(prefill_cache) = self.prefill_compile_cache.as_mut() {
            let key = ((batch as u64) << 32) | (seq as u64);
            if !prefill_cache.contains(key) {
                let mut wm = WeightMap::from_tensors(self.weights_cache.clone());
                let (graph, params) = build_llama32_graph_sized_last_logits(
                    &self.cfg, &mut wm, batch, seq, /*with_kv_outputs*/ true,
                )?;
                {
                    let compiled = prefill_cache.get_or_compile(key, || graph);
                    for (name, data) in &params {
                        compiled.set_param(name, data);
                    }
                }
            }
            let compiled =
                prefill_cache.get_or_compile(key, || unreachable!("just populated above"));
            Ok(compiled.run(&[("input_ids", ids_f32)]))
        } else {
            let mut wm = WeightMap::from_tensors(self.weights_cache.clone());
            let (graph, params) = build_llama32_graph_sized_last_logits(
                &self.cfg, &mut wm, batch, seq, /*with_kv_outputs*/ true,
            )?;
            let session = Session::new(self.device);
            let mut compiled = self.compile_graph_profiled(&session, graph)?;
            for (name, data) in &params {
                compiled.set_param(name, data);
            }
            Ok(compiled.run(&[("input_ids", ids_f32)]))
        }
    }

    /// Split raw graph outputs (logits + per-layer K + per-layer V) into
    /// (logits, layers_k, layers_v) for the one-shot decode path. The
    /// bucketed path needs slicing too, so it doesn't reuse this.
    #[allow(clippy::type_complexity)]
    fn split_decode_outputs(
        &self,
        outputs: Vec<Vec<f32>>,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let n_layers = self.cfg.num_hidden_layers;
        if outputs.len() != 1 + 2 * n_layers {
            anyhow::bail!(
                "decode graph produced {} outputs, expected {}",
                outputs.len(),
                1 + 2 * n_layers
            );
        }
        let mut iter = outputs.into_iter();
        let logits = iter.next().context("decode logits missing")?;
        let mut layers_k = Vec::with_capacity(n_layers);
        let mut layers_v = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers_k.push(iter.next().context("decode k missing")?);
            layers_v.push(iter.next().context("decode v missing")?);
        }
        Ok((logits, layers_k, layers_v))
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
        mut on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        let start = self.tokens.len();
        for _ in 0..n {
            let tok = self.step_cached(opts)?;
            on_token(tok);
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
        if outputs.len() != 1 + 2 * self.cfg.num_hidden_layers {
            anyhow::bail!(
                "prefill-with-cache produced {} outputs, expected {}",
                outputs.len(),
                1 + 2 * self.cfg.num_hidden_layers
            );
        }
        let expected_kv_len = batch * seq * kv_dim;
        let mut iter = outputs.into_iter();
        let logits = iter.next().context("prefill logits missing")?;
        let mut layers_k = Vec::with_capacity(self.cfg.num_hidden_layers);
        let mut layers_v = Vec::with_capacity(self.cfg.num_hidden_layers);
        for layer in 0..self.cfg.num_hidden_layers {
            let k = iter.next().context("prefill k missing")?;
            let v = iter.next().context("prefill v missing")?;
            if k.len() != expected_kv_len || v.len() != expected_kv_len {
                anyhow::bail!(
                    "layer {layer}: k.len={} v.len={} expected {}",
                    k.len(),
                    v.len(),
                    expected_kv_len
                );
            }
            layers_k.push(k);
            layers_v.push(v);
        }
        self.cache = Some(KvCacheState {
            past_seq: seq,
            layers_k,
            layers_v,
        });

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

    pub fn config(&self) -> &Llama32Config {
        &self.cfg
    }

    /// Low-level primitive: reset internal state, run prefill-with-cache
    /// over `context`, and return the *last position's* logits row
    /// (`P(next_token | context)`). Does NOT sample or append. The
    /// internal `tokens` buffer is set to `context` and the KV cache
    /// is populated to `past_seq = context.len()`.
    ///
    /// First row of logits after prefill-with-cache (no sampling).
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
        if outputs.len() != 1 + 2 * self.cfg.num_hidden_layers {
            anyhow::bail!(
                "prefill_get_last_logits: got {} outputs, expected {}",
                outputs.len(),
                1 + 2 * self.cfg.num_hidden_layers
            );
        }
        let expected_kv_len = batch * seq * kv_dim;
        let mut iter = outputs.into_iter();
        let logits = iter.next().context("logits missing")?;
        let mut layers_k = Vec::with_capacity(self.cfg.num_hidden_layers);
        let mut layers_v = Vec::with_capacity(self.cfg.num_hidden_layers);
        for _ in 0..self.cfg.num_hidden_layers {
            let k = iter.next().context("k missing")?;
            let v = iter.next().context("v missing")?;
            if k.len() != expected_kv_len || v.len() != expected_kv_len {
                anyhow::bail!("kv length mismatch in prefill_get_last_logits");
            }
            layers_k.push(k);
            layers_v.push(v);
        }
        self.cache = Some(KvCacheState {
            past_seq: seq,
            layers_k,
            layers_v,
        });

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
        let past_seq = cache.past_seq;

        let mut wm = WeightMap::from_tensors(self.weights_cache.clone());
        let (hir, params) =
            build_llama32_decode_hir_sized(&self.cfg, &mut wm, /*batch*/ 1, past_seq)?;
        let session = Session::new(self.device);
        let mut compiled = self.compile_hir_profiled(&session, hir, true)?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }

        let (cos, sin) = compute_rope_slice(&self.inv_freq, past_seq);
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
        if outputs.len() != 1 + 2 * self.cfg.num_hidden_layers {
            anyhow::bail!(
                "decode_get_logits: got {} outputs, expected {}",
                outputs.len(),
                1 + 2 * self.cfg.num_hidden_layers
            );
        }
        let mut iter = outputs.into_iter();
        let logits = iter.next().context("logits missing")?;
        let mut new_k = Vec::with_capacity(self.cfg.num_hidden_layers);
        let mut new_v = Vec::with_capacity(self.cfg.num_hidden_layers);
        for _ in 0..self.cfg.num_hidden_layers {
            new_k.push(iter.next().context("k missing")?);
            new_v.push(iter.next().context("v missing")?);
        }

        let cache_mut = self.cache.as_mut().unwrap();
        cache_mut.past_seq = past_seq + 1;
        cache_mut.layers_k = new_k;
        cache_mut.layers_v = new_v;
        self.tokens.push(input);

        Ok(logits)
    }
}

/// Compute the single-row (cos, sin) RoPE slice for absolute position
/// `pos`. Matches the formula in the prefill builder so cached decode
/// and recompute prefill produce the same RoPE rotation.
fn compute_rope_slice(inv_freq: &[f64], pos: usize) -> (Vec<f32>, Vec<f32>) {
    rope_slice(inv_freq, pos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Llama32Config;

    fn tiny_cfg() -> Llama32Config {
        Llama32Config {
            vocab_size: 16,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            attention_bias: false,
            head_dim: Some(8),
            rope_scaling: None,
        }
    }

    fn synthetic_weights(cfg: &Llama32Config) -> WeightMap {
        let h = cfg.hidden_size;
        let q_dim = cfg.q_proj_dim();
        let kv_dim = cfg.kv_proj_dim();
        let int_dim = cfg.intermediate_size;
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
        let mut gn = Llama32Generator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap();
        assert_eq!(wm.len(), 0, "loader should be drained");
        gn.prefill(&[1, 2, 3]);
        let t = gn.step(SampleOpts::greedy()).unwrap();
        assert!((t as usize) < cfg.vocab_size);
        assert_eq!(gn.tokens().len(), 4);
    }

    #[test]
    fn generate_n_appends_n_tokens() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let mut gn = Llama32Generator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap();
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
        let mut gn = Llama32Generator::from_loader(cfg, &mut wm, Device::Cpu).unwrap();
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
            Llama32Generator::from_loader(cfg.clone(), &mut wm_n, Device::Cpu).unwrap();
        gn_naive.prefill(&prompt);
        let naive_tokens = gn_naive.generate(steps, SampleOpts::greedy()).unwrap();

        let mut wm_c = synthetic_weights(&cfg);
        let mut gn_cached =
            Llama32Generator::from_loader(cfg.clone(), &mut wm_c, Device::Cpu).unwrap();
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
    fn cached_step_advances_cache_invariant() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let mut gn = Llama32Generator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap();
        gn.prefill(&[1, 2, 3]);
        let _ = gn.step_cached(SampleOpts::greedy()).unwrap();
        // After seed: tokens.len() == 4, cache.past_seq == 3 (cache holds prompt).
        assert_eq!(gn.tokens().len(), 4);
        assert_eq!(gn.cache.as_ref().unwrap().past_seq, 3);
        let _ = gn.step_cached(SampleOpts::greedy()).unwrap();
        // After one decode: tokens.len() == 5, cache.past_seq == 4.
        assert_eq!(gn.tokens().len(), 5);
        assert_eq!(gn.cache.as_ref().unwrap().past_seq, 4);
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
            Llama32Generator::from_loader(cfg.clone(), &mut wm_one, Device::Cpu).unwrap();
        gn_one.prefill(&prompt);
        let oneshot_tokens = gn_one.generate_cached(steps, SampleOpts::greedy()).unwrap();

        let mut wm_buc = synthetic_weights(&cfg);
        let mut gn_buc = Llama32Generator::from_loader(cfg.clone(), &mut wm_buc, Device::Cpu)
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
    fn prefill_compile_cache_does_not_change_output() {
        let cfg = tiny_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5];
        let mut wm_a = synthetic_weights(&cfg);
        let mut gn_a = Llama32Generator::from_loader(cfg.clone(), &mut wm_a, Device::Cpu).unwrap();
        gn_a.prefill(&prompt);
        let a = gn_a.generate_cached(4, SampleOpts::greedy()).unwrap();

        let mut wm_b = synthetic_weights(&cfg);
        let mut gn_b = Llama32Generator::from_loader(cfg.clone(), &mut wm_b, Device::Cpu)
            .unwrap()
            .with_prefill_cache(/*capacity*/ 4);
        gn_b.prefill(&prompt);
        let b = gn_b.generate_cached(4, SampleOpts::greedy()).unwrap();

        assert_eq!(a, b, "enabling prefill_cache must not change output");
    }

    #[test]
    fn dynamic_decode_matches_oneshot() {
        let cfg = tiny_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5];
        let steps = 6;

        let mut wm_one = synthetic_weights(&cfg);
        let mut gn_one =
            Llama32Generator::from_loader(cfg.clone(), &mut wm_one, Device::Cpu).unwrap();
        gn_one.prefill(&prompt);
        let oneshot_tokens = gn_one.generate_cached(steps, SampleOpts::greedy()).unwrap();

        let mut wm_dyn = synthetic_weights(&cfg);
        let mut gn_dyn = Llama32Generator::from_loader(cfg.clone(), &mut wm_dyn, Device::Cpu)
            .unwrap()
            .with_dynamic_decode_cache(/*capacity*/ 8);
        gn_dyn.prefill(&prompt);
        let dynamic_tokens = gn_dyn.generate_cached(steps, SampleOpts::greedy()).unwrap();

        assert_eq!(
            dynamic_tokens, oneshot_tokens,
            "dynamic past_seq decode diverged from one-shot decode"
        );
    }

    #[test]
    fn dynamic_prefill_matches_oneshot() {
        let cfg = tiny_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5];
        let steps = 4;

        let mut wm_one = synthetic_weights(&cfg);
        let mut gn_one =
            Llama32Generator::from_loader(cfg.clone(), &mut wm_one, Device::Cpu).unwrap();
        gn_one.prefill(&prompt);
        let oneshot_tokens = gn_one.generate_cached(steps, SampleOpts::greedy()).unwrap();

        let mut wm_dyn = synthetic_weights(&cfg);
        let mut gn_dyn = Llama32Generator::from_loader(cfg.clone(), &mut wm_dyn, Device::Cpu)
            .unwrap()
            .with_dynamic_prefill_cache(/*capacity*/ 8);
        gn_dyn.prefill(&prompt);
        let dynamic_tokens = gn_dyn.generate_cached(steps, SampleOpts::greedy()).unwrap();

        assert_eq!(
            dynamic_tokens, oneshot_tokens,
            "dynamic seq prefill diverged from one-shot prefill"
        );
    }

    #[test]
    fn dynamic_prefill_and_decode_matches_oneshot() {
        let cfg = tiny_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5];
        let steps = 6;

        let mut wm_one = synthetic_weights(&cfg);
        let mut gn_one =
            Llama32Generator::from_loader(cfg.clone(), &mut wm_one, Device::Cpu).unwrap();
        gn_one.prefill(&prompt);
        let oneshot_tokens = gn_one.generate_cached(steps, SampleOpts::greedy()).unwrap();

        let mut wm_dyn = synthetic_weights(&cfg);
        let mut gn_dyn = Llama32Generator::from_loader(cfg.clone(), &mut wm_dyn, Device::Cpu)
            .unwrap()
            .with_dynamic_prefill_cache(/*capacity*/ 8)
            .with_dynamic_decode_cache(/*capacity*/ 8);
        gn_dyn.prefill(&prompt);
        let dynamic_tokens = gn_dyn.generate_cached(steps, SampleOpts::greedy()).unwrap();

        assert_eq!(
            dynamic_tokens, oneshot_tokens,
            "dynamic prefill+decode diverged from one-shot path"
        );
    }

    #[test]
    fn greedy_is_deterministic_across_runs() {
        let cfg = tiny_cfg();
        let weights = synthetic_weights(&cfg);
        let mk = || {
            let mut wm = WeightMap::from_tensors(weights_as_hashmap(&weights));
            Llama32Generator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap()
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
