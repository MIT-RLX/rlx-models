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

//! Host-side generation loop for Gemma.
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
    build_gemma_decode_graph_sized, build_gemma_decode_hir_dynamic_ext,
    build_gemma_decode_hir_sized_ext, build_gemma_graph_sized_last_logits,
    build_gemma_graph_sized_last_logits_hidden, build_gemma_prefill_hidden_hir_dynamic_ext,
    build_gemma_prefill_hir_dynamic_ext,
};
use crate::config::GemmaConfig;
use crate::rope::{resolve_inv_freq, rope_slice};
use anyhow::{Context, Result};
use rlx_core::autoregressive::{
    KvCacheState, kv_from_prefill_outputs_per_layer, run_bucketed_kv_decode_hir_scratch,
    split_decode_logits_kv, split_decode_logits_kv_aux,
};
use rlx_core::flow_bridge::compile_options_from_profile;
use rlx_core::gpu_kv::{
    GpuKvBinding, device_supports_gpu_kv, run_bucketed_kv_decode_gpu_hir, sync_gpu_kv_to_host,
};
use rlx_core::weight_loader::WeightLoader;
use rlx_core::weight_map::WeightMap;
use rlx_flow::CompileProfile;
use rlx_ir::DimBinding;
use rlx_ir::logical_kernel::KernelDispatchConfig;
use rlx_qwen3::sampling::{SampleOpts, sample_token};
use rlx_runtime::compile_cache::{
    BucketedCompileCache, CacheRunInput, CompileCache, DynamicDimCompileCache,
};
use rlx_runtime::{CompileOptions, Device, Session};
use std::collections::HashMap;
use std::path::Path;

/// Decode compile profile with backend-specific fixes (Metal: optional unfused GQA).
pub fn decode_profile_for_device(device: Device) -> CompileProfile {
    metal_decode_profile(device, CompileProfile::gemma_decode())
}

/// When `RLX_GEMMA_METAL_THUNK_DECODE=1`, compile decode graphs with
/// `RLX_DISABLE_MPSGRAPH=1` (Metal thunk path + `fuse_decode_mlp`). Default:
/// MPSGraph (faster on Gemma 270M). Opt into thunks for fused MLP kernels.
fn metal_thunk_decode_requested() -> bool {
    std::env::var("RLX_GEMMA_METAL_THUNK_DECODE")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
}

pub(crate) fn metal_decode_compile_guard<R, F>(device: Device, decode: bool, f: F) -> R
where
    F: FnOnce() -> R,
{
    if decode && metal_thunk_decode_requested() {
        if device == Device::Metal {
            rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");
            let out = f();
            rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
            out
        } else {
            f()
        }
    } else {
        f()
    }
}

/// Packed prefill on Metal: default to the thunk + fused Q4 path instead of
/// whole-graph MPSGraph (often faster for small Gemma buckets on Apple Silicon).
fn metal_prefill_thunk_enabled() -> bool {
    match std::env::var("RLX_GEMMA_METAL_THUNK_PREFILL")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("no") => false,
        Some("1") | Some("true") | Some("yes") => true,
        _ => true,
    }
}

pub(crate) fn metal_prefill_compile_guard<R, F>(device: Device, f: F) -> R
where
    F: FnOnce() -> R,
{
    if device == Device::Metal && metal_prefill_thunk_enabled() {
        rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");
        let out = f();
        rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
        out
    } else {
        f()
    }
}

/// When `RLX_GEMMA_METAL_UNFUSED_DECODE=1`, disable tier-1 fusion on Metal decode
/// (MPSGraph GQA reshape workaround). Default: fusion **on** for throughput.
fn metal_unfused_decode_requested() -> bool {
    std::env::var("RLX_GEMMA_METAL_UNFUSED_DECODE")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
}

fn metal_decode_profile(device: Device, mut profile: CompileProfile) -> CompileProfile {
    if device == Device::Metal && metal_unfused_decode_requested() {
        profile.fusion.skip = true;
        profile.backend.metal.skip_fusion = true;
        profile.backend.metal.unfuse_regions = true;
    }
    profile
}

/// Stateful Gemma generation handle.
///
/// Holds the (config, weight bytes, token history) and rebuilds a
/// prefill graph on each [`step`] call. Cheap to construct after
/// initial weight load; tokens stay in-memory between calls.
pub struct GemmaGenerator {
    cfg: GemmaConfig,
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
    /// [`GemmaGenerator::with_prefill_cache`]).
    prefill_compile_cache: Option<CompileCache>,
    /// Compile prefill once with `sym::SEQ`, specialize per prompt length.
    prefill_dynamic_cache: Option<DynamicDimCompileCache>,
    /// Same as above but for fused `prefill_hidden` (multimodal).
    embed_prefill_compile_cache: Option<CompileCache>,
    embed_prefill_dynamic_cache: Option<DynamicDimCompileCache>,
    /// Bucketed compile cache for decode-mode graphs. Each bucket
    /// holds one compiled graph specialized at its upper-bound
    /// `past_seq`; the host pads `past_k`/`past_v` and supplies a
    /// per-step mask so a single bucket serves every `past_seq` in
    /// its range. Opt in via [`GemmaGenerator::with_decode_cache`].
    decode_compile_cache: Option<BucketedCompileCache>,
    decode_dynamic_cache: Option<DynamicDimCompileCache>,
    /// Resolved RoPE inverse frequencies (includes Llama 3 scaling).
    inv_freq: Vec<f64>,
    /// Tier-1 compile profile for prefill graphs.
    prefill_profile: CompileProfile,
    /// Tier-1 compile profile for decode graphs.
    decode_profile: CompileProfile,
    /// Fused prefill embeddings for the next cached prefill (multimodal).
    pending_prefill_embeds: Option<Vec<f32>>,
    pending_prefill_attn_bias: Option<Vec<f32>>,
    /// GPU-resident K/V on Metal/MLX/CUDA decode (logits-only readback).
    use_gpu_kv: bool,
    gpu_kv_binding: GpuKvBinding,
    /// Reused decode inputs (mask, padded K/V) to avoid per-step allocs.
    decode_scratch: DecodeKvScratch,
    decode_inputs: DecodeInputScratch,
    /// EAGLE3-style aux hidden-state tap: when non-empty, decode steps
    /// build a graph with `with_aux_hidden_layer_ids(...)` set,
    /// extract the per-layer hidden states from the graph outputs,
    /// and stash them in [`last_aux`] for the caller to retrieve.
    /// Forces the `decode_step_oneshot` path (the bucketed compile
    /// cache doesn't yet handle the extra outputs).
    aux_hidden_layer_ids: Vec<usize>,
    /// Aux hidden states from the most recent decode step. One
    /// `Vec<f32>` per id in [`aux_hidden_layer_ids`], each of length
    /// `batch * 1 * hidden_size`. Cleared by [`take_last_aux`].
    last_aux: Option<Vec<Vec<f32>>>,
}

/// Reusable mask + rope slices for bucketed decode (no per-step alloc).
#[derive(Default)]
struct DecodeInputScratch {
    mask: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

/// Per-session padded K/V upload buffers (resized when bucket upper changes).
#[derive(Default)]
struct DecodeKvScratch {
    padded_k: Vec<Vec<f32>>,
    padded_v: Vec<Vec<f32>>,
    bucket_upper: usize,
}

impl DecodeInputScratch {
    fn fill_mask(&mut self, past_seq: usize, upper: usize) {
        if self.mask.len() != upper + 1 {
            self.mask.resize(upper + 1, 0.0);
        }
        for (i, m) in self.mask.iter_mut().enumerate().take(upper + 1) {
            *m = if i < past_seq || i == upper { 1.0 } else { 0.0 };
        }
    }

    fn fill_rope(&mut self, inv_freq: &[f64], pos: usize) {
        let half = inv_freq.len();
        self.cos.resize(half, 0.0);
        self.sin.resize(half, 0.0);
        for (i, &freq) in inv_freq.iter().enumerate() {
            let angle = pos as f64 * freq;
            let (s, c) = angle.sin_cos();
            self.cos[i] = c as f32;
            self.sin[i] = s as f32;
        }
    }
}

impl DecodeKvScratch {
    fn ensure_bucket(&mut self, upper: usize, kv_dims: &[usize]) {
        if self.bucket_upper == upper && self.padded_k.len() == kv_dims.len() {
            return;
        }
        self.bucket_upper = upper;
        self.padded_k = kv_dims.iter().map(|&d| vec![0.0_f32; upper * d]).collect();
        self.padded_v = kv_dims.iter().map(|&d| vec![0.0_f32; upper * d]).collect();
    }
}

fn gemma_use_gpu_kv(device: Device) -> bool {
    if !device_supports_gpu_kv(device) {
        return false;
    }
    match std::env::var("RLX_GEMMA_GPU_KV").ok().as_deref() {
        Some("0") | Some("false") | Some("no") => false,
        Some("1") | Some("true") | Some("yes") => true,
        // Host readback path is faster on Gemma Metal today; Whisper/Qwen3-TTS differ.
        _ => false,
    }
}

impl GemmaGenerator {
    /// Construct from any [`WeightLoader`] — drains it into an
    /// internal cache so the loader is free after this call.
    pub fn from_loader(
        cfg: GemmaConfig,
        loader: &mut dyn WeightLoader,
        device: Device,
    ) -> Result<Self> {
        let keys = loader.remaining_keys();
        // Capture the arch up front so the cache-key normalization can
        // pick the gemma2 reverse alias (4 distinct per-layer norms)
        // over the generic Llama-flavored one (2 norms, ambiguous on
        // `ffn_norm`). Owned string so we don't hold a borrow across
        // the mutable `loader.take` calls below.
        let arch_hint: Option<String> = loader.arch_hint().map(|s| s.to_string());
        let mut weights_cache = HashMap::with_capacity(keys.len());
        for k in keys {
            let v = loader
                .take(&k)
                .with_context(|| format!("draining weight {k}"))?;
            // Normalize the cache key to the safetensors / HuggingFace
            // naming convention so subsequent builder calls that ask
            // for `model.embed_tokens.weight` (the canonical name baked
            // into the gemma builder) hit the cache whether the
            // loader was safetensors-native or GGUF-native.
            let canonical = match arch_hint.as_deref() {
                Some(a) => rlx_core::weight_loader::gguf_to_hf_name_for_arch(&k, a)
                    .unwrap_or_else(|| k.clone()),
                None => rlx_core::weight_loader::gguf_to_hf_name(&k).unwrap_or_else(|| k.clone()),
            };
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
            embed_prefill_compile_cache: None,
            embed_prefill_dynamic_cache: None,
            decode_compile_cache: None,
            decode_dynamic_cache: None,
            inv_freq,
            prefill_profile: CompileProfile::gemma_prefill(),
            decode_profile: metal_decode_profile(device, CompileProfile::gemma_decode()),
            pending_prefill_embeds: None,
            pending_prefill_attn_bias: None,
            use_gpu_kv: gemma_use_gpu_kv(device),
            gpu_kv_binding: GpuKvBinding::default(),
            decode_scratch: DecodeKvScratch::default(),
            decode_inputs: DecodeInputScratch::default(),
            aux_hidden_layer_ids: Vec::new(),
            last_aux: None,
        })
    }

    /// EAGLE3-style aux hidden state tap. When `ids` is non-empty,
    /// every subsequent decode step builds the graph with
    /// `with_aux_hidden_layer_ids(ids)` and extracts the per-layer
    /// hidden states. Call [`take_last_aux`] after each
    /// `step_cached` / `generate_cached_with` token to retrieve
    /// them. Setting this disables the bucketed-decode fast path —
    /// each step compiles via `decode_step_oneshot`.
    pub fn set_aux_hidden_layer_ids(&mut self, ids: Vec<usize>) {
        // The graph builder dedups + sorts the ids and filters OOB
        // values. Mirror that here so `self.aux_hidden_layer_ids.len()`
        // matches the number of aux outputs the graph actually emits.
        let mut v: Vec<usize> = ids
            .into_iter()
            .filter(|i| *i < self.cfg.num_hidden_layers)
            .collect();
        v.sort_unstable();
        v.dedup();
        self.aux_hidden_layer_ids = v;
        self.last_aux = None;
    }

    /// Take + clear the most recent aux hidden states.
    pub fn take_last_aux(&mut self) -> Option<Vec<Vec<f32>>> {
        self.last_aux.take()
    }

    /// True if an aux tap is currently configured.
    pub fn aux_enabled(&self) -> bool {
        !self.aux_hidden_layer_ids.is_empty()
    }

    fn reset_gpu_kv_binding(&mut self) {
        self.gpu_kv_binding = GpuKvBinding::default();
    }

    /// Like [`Self::from_loader`] but loads tier-1 profiles from
    /// `gemma.rlx.toml` in the weights directory when present.
    pub fn from_loader_at(
        cfg: GemmaConfig,
        loader: &mut dyn WeightLoader,
        device: Device,
        weights_path: &Path,
    ) -> Result<Self> {
        let mut g = Self::from_loader(cfg, loader, device)?;
        g.prefill_profile = crate::gemma_profile_near_weights(weights_path, false);
        g.decode_profile = metal_decode_profile(
            device,
            crate::gemma_profile_near_weights(weights_path, true),
        );
        Ok(g)
    }

    /// Override tier-1 compile profiles explicitly.
    pub fn with_compile_profiles(
        mut self,
        prefill: CompileProfile,
        decode: CompileProfile,
    ) -> Self {
        self.prefill_profile = prefill;
        self.decode_profile = metal_decode_profile(self.device, decode);
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

    fn compile_graph_profiled(
        &self,
        session: &Session,
        graph: rlx_ir::Graph,
    ) -> Result<rlx_runtime::CompiledGraph> {
        let opts = self.profile_compile_options(false);
        Ok(session.compile_with(graph, &opts))
    }

    fn compile_graph_profiled_decode(
        &self,
        session: &Session,
        graph: rlx_ir::Graph,
    ) -> Result<rlx_runtime::CompiledGraph> {
        Ok(metal_decode_compile_guard(self.device, true, || {
            session.compile_with(graph, &self.profile_compile_options(true))
        }))
    }

    /// Enable the prefill compile cache with the given LRU capacity.
    /// Useful when the same prompt length is used across multiple
    /// generation runs — the second + Nth run skip the compile +
    /// param-attach roundtrip (~30-50ms per call on CPU).
    pub fn with_prefill_cache(mut self, capacity: usize) -> Self {
        self.prefill_compile_cache = Some(CompileCache::new(self.device, capacity));
        self.embed_prefill_compile_cache = Some(CompileCache::new(self.device, capacity));
        self.prefill_dynamic_cache = None;
        self.embed_prefill_dynamic_cache = None;
        self
    }

    /// Compile prefill once with `sym::SEQ`, specialize per prompt length.
    pub fn with_dynamic_prefill_cache(mut self, capacity: usize) -> Self {
        self.prefill_dynamic_cache = Some(DynamicDimCompileCache::new(self.device, capacity));
        self.embed_prefill_dynamic_cache = Some(DynamicDimCompileCache::new(self.device, capacity));
        self.prefill_compile_cache = None;
        self.embed_prefill_compile_cache = None;
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
        self
    }

    /// Compile decode once with `sym::PAST_SEQ`, specialize per prefix length.
    pub fn with_dynamic_decode_cache(mut self, capacity: usize) -> Self {
        self.decode_dynamic_cache = Some(DynamicDimCompileCache::new(self.device, capacity));
        self.decode_compile_cache = None;
        self
    }

    fn inference_dynamic_decode() -> bool {
        std::env::var("RLX_GEMMA_DYNAMIC_DECODE").is_ok_and(|v| {
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
        })
    }

    /// Production inference caches: dynamic prefill (+ multimodal embed prefill)
    /// and bucketed decode. Bucketed decode avoids the per-step specialize
    /// overhead of `RLX_GEMMA_DYNAMIC_DECODE=1` (experimental; often much slower on Metal).
    pub fn with_inference_caches(mut self, max_seq: usize) -> Self {
        // Match decode buckets to the session horizon (+slack). Avoid a global
        // 256-token floor — padded bucket attention on Metal is costly for chat.
        let decode_horizon = max_seq.saturating_add(16).max(32);
        self = self.with_dynamic_prefill_cache(16);
        if Self::inference_dynamic_decode() {
            self.with_dynamic_decode_cache(32)
        } else {
            self.with_decode_cache(decode_horizon)
        }
    }

    /// Wait for in-flight Metal command buffers on all cached graphs.
    /// Call between heavy inference phases to avoid MPS lifecycle warnings.
    pub fn sync_device(&mut self) {
        if let Some(c) = &mut self.prefill_compile_cache {
            c.sync_all();
        }
        if let Some(c) = &mut self.embed_prefill_compile_cache {
            c.sync_all();
        }
        if let Some(c) = &mut self.prefill_dynamic_cache {
            c.sync_all();
        }
        if let Some(c) = &mut self.embed_prefill_dynamic_cache {
            c.sync_all();
        }
        if let Some(c) = &mut self.decode_compile_cache {
            c.sync_all();
        }
        if let Some(c) = &mut self.decode_dynamic_cache {
            c.sync_all();
        }
        rlx_runtime::device_ext::drain_device(self.device);
    }

    /// Convenience: load weights from a safetensors or GGUF path
    /// (dispatch by extension; see `rlx_core::weight_loader::load_from_path`).
    pub fn from_path(cfg: GemmaConfig, path: &str, device: Device) -> Result<Self> {
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
        cfg: GemmaConfig,
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
        self.reset_gpu_kv_binding();
    }

    /// Like [`prefill`], but the next cached prefill uses fused
    /// `inputs_embeds` (`prefill_hidden`) instead of token lookup.
    pub fn prefill_from_embeds(
        &mut self,
        prompt_ids: &[u32],
        embeds: &[f32],
        attn_bias: Option<Vec<f32>>,
    ) -> Result<()> {
        let h = self.cfg.hidden_size;
        if embeds.len() != prompt_ids.len() * h {
            anyhow::bail!(
                "prefill_from_embeds: embeds len {} != {} tokens × hidden {}",
                embeds.len(),
                prompt_ids.len(),
                h
            );
        }
        if let Some(ref bias) = attn_bias {
            let seq = prompt_ids.len();
            let nh = self.cfg.num_attention_heads;
            let expected = seq * seq * nh;
            if bias.len() != expected {
                anyhow::bail!(
                    "prefill_from_embeds: attn_bias len {} != batch×heads×seq² ({expected})",
                    bias.len()
                );
            }
        }
        self.prefill(prompt_ids);
        self.pending_prefill_embeds = Some(embeds.to_vec());
        self.pending_prefill_attn_bias = attn_bias;
        Ok(())
    }

    /// Weight table for CPU-side embedding / fusion helpers.
    pub fn weights_cache(&self) -> &HashMap<String, (Vec<f32>, Vec<usize>)> {
        &self.weights_cache
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
        let (graph, params) = build_gemma_graph_sized_last_logits(
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
        let past_seq = cache.past_len;
        if self.tokens.len() <= past_seq {
            anyhow::bail!(
                "cache invariant violated: tokens.len() {} <= past_len {}",
                self.tokens.len(),
                past_seq
            );
        }
        let input_tok = self.tokens[past_seq];

        // EAGLE3 tap forces the always-correct oneshot path — the
        // bucketed compile cache doesn't yet ingest the extra aux
        // outputs.
        let aux_active = self.aux_enabled();
        let (logits, new_k, new_v) = if aux_active {
            let (logits, k, v, aux) = self.decode_step_oneshot_with_aux(past_seq, input_tok)?;
            self.last_aux = Some(aux);
            (logits, k, v)
        } else if self.decode_dynamic_cache.is_some() {
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
        cache_mut.past_len = past_seq + 1;
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
        let (graph, params) =
            build_gemma_decode_graph_sized(&self.cfg, &mut wm, /*batch*/ 1, past_seq)?;
        let session = Session::new(self.device);
        let mut compiled = self.compile_graph_profiled_decode(&session, graph)?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }

        let input_ids_f32 = [input_tok as f32];
        let key_strs: Vec<String> = (0..self.cfg.num_hidden_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        let mut inputs: Vec<(&str, &[f32])> =
            Vec::with_capacity(1 + 2 * self.cfg.num_hidden_layers);
        inputs.push(("input_ids", input_ids_f32.as_slice()));
        for i in 0..self.cfg.num_hidden_layers {
            inputs.push((&key_strs[2 * i], cache.layers_k[i].as_slice()));
            inputs.push((&key_strs[2 * i + 1], cache.layers_v[i].as_slice()));
        }

        let outputs = compiled.run(&inputs);
        split_decode_logits_kv(outputs, self.cfg.num_hidden_layers)
    }

    /// EAGLE3 variant of [`decode_step_oneshot`]. Builds the decode
    /// graph with `aux_hidden_layer_ids` set on `GemmaDecodeOpts` so
    /// the per-layer pre-attention-norm hidden states are appended
    /// to the graph's outputs. Returns `(logits, K, V, aux)` where
    /// `aux.len() == self.aux_hidden_layer_ids.len()`.
    #[allow(clippy::type_complexity)]
    fn decode_step_oneshot_with_aux(
        &mut self,
        past_seq: usize,
        input_tok: u32,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let cache = self.cache.as_ref().unwrap();
        let aux_ids = self.aux_hidden_layer_ids.clone();
        let num_aux = aux_ids.len();

        let mut wm = WeightMap::from_tensors(self.weights_cache.clone());
        let opts = crate::flow::GemmaDecodeOpts {
            batch: 1,
            past_seq,
            dynamic_past: false,
            use_custom_mask: false,
            profile: None,
            aux_hidden_layer_ids: aux_ids,
        };
        let (graph, params) = crate::flow::build_gemma_decode_graph(&self.cfg, &mut wm, &opts)?;
        let session = Session::new(self.device);
        let mut compiled = self.compile_graph_profiled_decode(&session, graph)?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }

        let input_ids_f32 = [input_tok as f32];
        let key_strs: Vec<String> = (0..self.cfg.num_hidden_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        let mut inputs: Vec<(&str, &[f32])> =
            Vec::with_capacity(1 + 2 * self.cfg.num_hidden_layers);
        inputs.push(("input_ids", input_ids_f32.as_slice()));
        for i in 0..self.cfg.num_hidden_layers {
            inputs.push((&key_strs[2 * i], cache.layers_k[i].as_slice()));
            inputs.push((&key_strs[2 * i + 1], cache.layers_v[i].as_slice()));
        }

        let outputs = compiled.run(&inputs);
        split_decode_logits_kv_aux(outputs, self.cfg.num_hidden_layers, num_aux)
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
        let cache_dyn = self
            .decode_dynamic_cache
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("dynamic decode without cache"))?;
        let needs_upload = !cache_dyn.contains(past_seq as u64);
        let cfg = self.cfg.clone();
        let weights_cache = self.weights_cache.clone();
        let max_past = self.cfg.max_position_embeddings;
        let compiled = metal_decode_compile_guard(self.device, true, || {
            cache_dyn.get_or_specialize(
                past_seq as u64,
                &binding,
                || {
                    let mut wm = WeightMap::from_tensors(weights_cache);
                    build_gemma_decode_hir_dynamic_ext(&cfg, &mut wm, 1, max_past)
                        .expect("dynamic decode HIR")
                        .0
                },
                &opts,
            )
        })?;
        if needs_upload {
            let mut wm = WeightMap::from_tensors(self.weights_cache.clone());
            let (_, params) = build_gemma_decode_hir_dynamic_ext(&self.cfg, &mut wm, 1, max_past)?;
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
        split_decode_logits_kv(outputs, self.cfg.num_hidden_layers)
    }

    #[allow(clippy::type_complexity)]
    fn decode_step_bucketed(
        &mut self,
        past_seq: usize,
        input_tok: u32,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let kv_dims = self.per_layer_kv_dims();
        let n_layers = self.cfg.num_hidden_layers;
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

        self.decode_scratch.ensure_bucket(upper, &kv_dims);
        self.decode_inputs.fill_mask(past_seq, upper);
        self.decode_inputs.fill_rope(&self.inv_freq, past_seq);

        let input_ids_f32 = [input_tok as f32];
        let fixed = [
            CacheRunInput {
                name: "input_ids",
                data: &input_ids_f32,
                row_inner: None,
            },
            CacheRunInput {
                name: "rope_cos",
                data: &self.decode_inputs.cos,
                row_inner: None,
            },
            CacheRunInput {
                name: "rope_sin",
                data: &self.decode_inputs.sin,
                row_inner: None,
            },
            CacheRunInput {
                name: "mask",
                data: &self.decode_inputs.mask,
                row_inner: None,
            },
        ];

        if self.use_gpu_kv && self.decode_compile_cache.is_some() {
            let key = past_seq as u64;
            let upper_u = upper as u64;
            let prev_upper = self.gpu_kv_binding.upper;
            let bucket_changed = prev_upper != 0 && prev_upper != upper_u;
            let handles_live = self
                .decode_compile_cache
                .as_mut()
                .and_then(|c| c.compiled_for_key_mut(key))
                .map(|cg| cg.has_gpu_handle("past_k_0"))
                .unwrap_or(false);
            let refresh_kv = matches!(self.device, Device::Gpu | Device::Metal)
                || bucket_changed
                || !handles_live;
            let cfg = self.cfg.clone();
            let weights = self.weights_cache.clone();
            let logits = {
                let cache_dec = self.decode_compile_cache.as_mut().unwrap();
                let cache_mut = self.cache.as_mut().unwrap();
                metal_decode_compile_guard(self.device, true, || {
                    run_bucketed_kv_decode_gpu_hir(
                        cache_dec,
                        key,
                        past_seq,
                        cache_mut,
                        &mut self.gpu_kv_binding,
                        self.cfg.kv_proj_dim(),
                        n_layers,
                        &fixed,
                        move |upper| {
                            let mut wm = WeightMap::from_tensors(weights.clone());
                            build_gemma_decode_hir_sized_ext(&cfg, &mut wm, 1, upper as usize, true)
                                .expect("gemma bucketed decode HIR")
                        },
                        &decode_opts,
                        refresh_kv,
                    )
                })?
            };
            if let Some(compiled) = self
                .decode_compile_cache
                .as_mut()
                .and_then(|c| c.compiled_for_key_mut(key))
            {
                let cache_mut = self.cache.as_mut().unwrap();
                sync_gpu_kv_to_host(compiled, cache_mut, self.cfg.kv_proj_dim(), n_layers)?;
            }
            let next_key = (past_seq + 1) as u64;
            let next_upper = self
                .decode_compile_cache
                .as_ref()
                .and_then(|cache| {
                    cache
                        .bucket_for(next_key)
                        .and_then(|idx| cache.buckets().nth(idx).map(|r| (r.end - 1) as usize))
                })
                .unwrap_or(upper);
            if next_upper != upper {
                self.reset_gpu_kv_binding();
            }
            let cache_mut = self.cache.as_ref().unwrap();
            let new_k = cache_mut.layers_k.clone();
            let new_v = cache_mut.layers_v.clone();
            return Ok((logits, new_k, new_v));
        }

        let cfg = self.cfg.clone();
        let weights = self.weights_cache.clone();
        let cache_dec = self.decode_compile_cache.as_mut().unwrap();
        let kv_cache = self.cache.as_ref().unwrap();
        let DecodeKvScratch {
            padded_k, padded_v, ..
        } = &mut self.decode_scratch;
        metal_decode_compile_guard(self.device, true, || {
            run_bucketed_kv_decode_hir_scratch(
                cache_dec,
                past_seq,
                kv_cache,
                &kv_dims,
                n_layers,
                padded_k,
                padded_v,
                &fixed,
                |upper| {
                    let mut wm = WeightMap::from_tensors(weights.clone());
                    build_gemma_decode_hir_sized_ext(&cfg, &mut wm, 1, upper as usize, true)
                        .expect("gemma bucketed decode HIR")
                },
                &decode_opts,
            )
        })
    }

    /// Run prefill-with-cache and return the raw outputs. Uses the
    /// LRU `CompileCache` when enabled; otherwise compiles fresh each
    /// call. Keyed by `seq` because graph shape is seq-specialized.
    #[allow(clippy::unnecessary_unwrap)]
    fn run_prefill_with_cache(
        &mut self,
        batch: usize,
        seq: usize,
        ids_f32: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        if self.prefill_dynamic_cache.is_some() {
            let binding = DimBinding::batch_seq(batch, seq);
            let opts = compile_options_from_profile(
                &self.prefill_profile,
                self.device,
                KernelDispatchConfig::default(),
            )
            .dim_binding(binding.clone());
            let cache = self.prefill_dynamic_cache.as_mut().expect("checked");
            let needs_upload = !cache.contains(seq as u64);
            let cfg = self.cfg.clone();
            let weights_cache = self.weights_cache.clone();
            let max_seq = self.cfg.max_position_embeddings;
            let compiled = cache.get_or_specialize(
                seq as u64,
                &binding,
                || {
                    let mut wm = WeightMap::from_tensors(weights_cache);
                    build_gemma_prefill_hir_dynamic_ext(&cfg, &mut wm, batch, max_seq, true)
                        .expect("dynamic prefill HIR")
                        .0
                },
                &opts,
            )?;
            if needs_upload {
                let mut wm = WeightMap::from_tensors(self.weights_cache.clone());
                let (_, params) =
                    build_gemma_prefill_hir_dynamic_ext(&self.cfg, &mut wm, batch, max_seq, true)?;
                for (name, data) in &params {
                    compiled.set_param(name, data);
                }
            }
            let last_idx = vec![(seq - 1) as f32];
            Ok(compiled.run(&[("input_ids", ids_f32), ("last_token_idx", &last_idx)]))
        } else if self.prefill_compile_cache.is_some() {
            let key = ((batch as u64) << 32) | (seq as u64);
            let opts = self.profile_compile_options(false);
            if !self.prefill_compile_cache.as_ref().unwrap().contains(key) {
                let mut wm = WeightMap::from_tensors(self.weights_cache.clone());
                let (graph, params) = build_gemma_graph_sized_last_logits(
                    &self.cfg, &mut wm, batch, seq, /*with_kv_outputs*/ true,
                )?;
                {
                    let compiled = self
                        .prefill_compile_cache
                        .as_mut()
                        .unwrap()
                        .get_or_compile_with_options(key, || graph, &opts);
                    for (name, data) in &params {
                        compiled.set_param(name, data);
                    }
                }
            }
            let compiled = self
                .prefill_compile_cache
                .as_mut()
                .unwrap()
                .get_or_compile_with_options(key, || unreachable!("just populated above"), &opts);
            Ok(compiled.run(&[("input_ids", ids_f32)]))
        } else {
            let mut wm = WeightMap::from_tensors(self.weights_cache.clone());
            let (graph, params) = build_gemma_graph_sized_last_logits(
                &self.cfg, &mut wm, batch, seq, /*with_kv_outputs*/ true,
            )?;
            let session = Session::new(self.device);
            let opts = self.profile_compile_options(false);
            let mut compiled = session.compile_with(graph, &opts);
            for (name, data) in &params {
                compiled.set_param(name, data);
            }
            Ok(compiled.run(&[("input_ids", ids_f32)]))
        }
    }

    fn run_prefill_hidden_with_cache(
        &mut self,
        batch: usize,
        seq: usize,
        hidden: &[f32],
        attn_bias: Option<&[f32]>,
    ) -> Result<Vec<Vec<f32>>> {
        if self.cfg.use_bidirectional_vision() && attn_bias.is_none() {
            anyhow::bail!(
                "multimodal prefill requires attn_bias when use_bidirectional_attention=vision"
            );
        }
        let mut inputs: Vec<(&str, &[f32])> = vec![("prefill_hidden", hidden)];
        if let Some(bias) = attn_bias {
            inputs.push(("attn_bias", bias));
        }
        let embed_compile_opts = self.profile_compile_options(false);
        if let Some(cache) = &mut self.embed_prefill_dynamic_cache {
            let binding = DimBinding::batch_seq(batch, seq);
            let opts = compile_options_from_profile(
                &self.prefill_profile,
                self.device,
                KernelDispatchConfig::default(),
            )
            .dim_binding(binding.clone());
            let needs_upload = !cache.contains(seq as u64);
            let cfg = self.cfg.clone();
            let weights_cache = self.weights_cache.clone();
            let max_seq = self.cfg.max_position_embeddings;
            let compiled = cache.get_or_specialize(
                seq as u64,
                &binding,
                || {
                    let mut wm = WeightMap::from_tensors(weights_cache);
                    build_gemma_prefill_hidden_hir_dynamic_ext(&cfg, &mut wm, batch, max_seq, true)
                        .expect("dynamic hidden prefill HIR")
                        .0
                },
                &opts,
            )?;
            if needs_upload {
                let mut wm = WeightMap::from_tensors(self.weights_cache.clone());
                let (_, params) = build_gemma_prefill_hidden_hir_dynamic_ext(
                    &self.cfg, &mut wm, batch, max_seq, true,
                )?;
                for (name, data) in &params {
                    compiled.set_param(name, data);
                }
            }
            let last_idx = vec![(seq - 1) as f32];
            let mut dyn_inputs = inputs.clone();
            dyn_inputs.push(("last_token_idx", &last_idx));
            Ok(compiled.run(&dyn_inputs))
        } else if let Some(cache) = &mut self.embed_prefill_compile_cache {
            let key = ((batch as u64) << 32) | (seq as u64);
            let opts = &embed_compile_opts;
            if !cache.contains(key) {
                let mut wm = WeightMap::from_tensors(self.weights_cache.clone());
                let (graph, params) = build_gemma_graph_sized_last_logits_hidden(
                    &self.cfg, &mut wm, batch, seq, true,
                )?;
                {
                    let compiled = cache.get_or_compile_with_options(key, || graph, opts);
                    for (name, data) in &params {
                        compiled.set_param(name, data);
                    }
                }
            }
            let compiled = cache.get_or_compile_with_options(
                key,
                || unreachable!("just populated above"),
                opts,
            );
            Ok(compiled.run(&inputs))
        } else {
            let mut wm = WeightMap::from_tensors(self.weights_cache.clone());
            let (graph, params) =
                build_gemma_graph_sized_last_logits_hidden(&self.cfg, &mut wm, batch, seq, true)?;
            let session = Session::new(self.device);
            let opts = self.profile_compile_options(false);
            let mut compiled = session.compile_with(graph, &opts);
            for (name, data) in &params {
                compiled.set_param(name, data);
            }
            Ok(compiled.run(&inputs))
        }
    }

    /// Run `n` cached steps after [`prefill_from_embeds`].
    pub fn generate_from_embeds(
        &mut self,
        prompt_ids: &[u32],
        embeds: &[f32],
        n: usize,
        opts: SampleOpts,
    ) -> Result<Vec<u32>> {
        self.generate_from_embeds_with_bias(prompt_ids, embeds, None, n, opts)
    }

    pub fn generate_from_embeds_with_bias(
        &mut self,
        prompt_ids: &[u32],
        embeds: &[f32],
        attn_bias: Option<Vec<f32>>,
        n: usize,
        opts: SampleOpts,
    ) -> Result<Vec<u32>> {
        self.prefill_from_embeds(prompt_ids, embeds, attn_bias)?;
        self.generate_cached(n, opts)
    }

    /// Streaming variant of [`generate_from_embeds`].
    pub fn generate_from_embeds_with(
        &mut self,
        prompt_ids: &[u32],
        embeds: &[f32],
        n: usize,
        opts: SampleOpts,
        on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        self.generate_from_embeds_with_bias_and_callback(
            prompt_ids, embeds, None, n, opts, on_token,
        )
    }

    pub fn generate_from_embeds_with_bias_and_callback(
        &mut self,
        prompt_ids: &[u32],
        embeds: &[f32],
        attn_bias: Option<Vec<f32>>,
        n: usize,
        opts: SampleOpts,
        on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        self.prefill_from_embeds(prompt_ids, embeds, attn_bias)?;
        self.generate_cached_with(n, opts, on_token)
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
        let kv_dims = self.per_layer_kv_dims();

        let outputs = if let Some(embeds) = self.pending_prefill_embeds.take() {
            let bias = self.pending_prefill_attn_bias.take();
            self.run_prefill_hidden_with_cache(batch, seq, &embeds, bias.as_deref())?
        } else {
            let ids_f32: Vec<f32> = self.tokens.iter().map(|&i| i as f32).collect();
            self.run_prefill_with_cache(batch, seq, &ids_f32)?
        };
        let (logits, kv) = kv_from_prefill_outputs_per_layer(
            outputs,
            batch,
            seq,
            &kv_dims,
            self.cfg.num_hidden_layers,
        )?;
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

    pub fn config(&self) -> &GemmaConfig {
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
        self.reset_gpu_kv_binding();

        let seq = context.len();
        let batch = 1usize;
        let kv_dims = self.per_layer_kv_dims();

        let ids_f32: Vec<f32> = context.iter().map(|&i| i as f32).collect();
        let outputs = self.run_prefill_with_cache(batch, seq, &ids_f32)?;
        let (logits, kv) = kv_from_prefill_outputs_per_layer(
            outputs,
            batch,
            seq,
            &kv_dims,
            self.cfg.num_hidden_layers,
        )?;
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
        if self.cache.is_none() {
            anyhow::bail!(
                "decode_get_logits: cache not seeded; call prefill_get_last_logits first"
            );
        }
        self.tokens.push(input);
        let seq = self.tokens.len();
        let batch = 1usize;
        let kv_dims = self.per_layer_kv_dims();
        let ids_f32: Vec<f32> = self.tokens.iter().map(|&i| i as f32).collect();
        let outputs = self.run_prefill_with_cache(batch, seq, &ids_f32)?;
        let (logits, kv) = kv_from_prefill_outputs_per_layer(
            outputs,
            batch,
            seq,
            &kv_dims,
            self.cfg.num_hidden_layers,
        )?;
        self.cache = Some(kv);
        let vocab = self.cfg.vocab_size;
        Ok(logits[..vocab].to_vec())
    }

    /// Per-layer KV dimensions (`num_kv_heads * head_dim` for each
    /// layer, accounting for Gemma 4's full-attention shape divergence).
    fn per_layer_kv_dims(&self) -> Vec<usize> {
        (0..self.cfg.num_hidden_layers)
            .map(|i| self.cfg.layer_num_kv_heads(i) * self.cfg.layer_head_dim(i))
            .collect()
    }
}

impl Drop for GemmaGenerator {
    fn drop(&mut self) {
        if self.device == Device::Metal {
            self.sync_device();
        }
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
    use crate::config::GemmaConfig;
    use crate::rope::{build_rope_tables, resolve_inv_freq, rope_slice};
    use rlx_flow::CompileProfile;

    fn tiny_cfg() -> GemmaConfig {
        let mut cfg = GemmaConfig::tiny_test();
        cfg.vocab_size = 16;
        cfg.head_dim = Some(8);
        cfg
    }

    fn synthetic_tensors(cfg: &GemmaConfig) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
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
        t
    }

    fn synthetic_weights(cfg: &GemmaConfig) -> WeightMap {
        WeightMap::from_tensors(synthetic_tensors(cfg))
    }

    #[test]
    fn generator_drains_loader_and_runs_one_step() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let mut gn = GemmaGenerator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap();
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
        let mut gn = GemmaGenerator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap();
        gn.prefill(&[5, 6]);
        let new_tokens = gn.generate(3, SampleOpts::greedy()).unwrap();
        assert_eq!(new_tokens.len(), 3);
        assert_eq!(gn.tokens().len(), 5);
        for t in &new_tokens {
            assert!((*t as usize) < cfg.vocab_size);
        }
    }

    #[test]
    fn step_cached_with_aux_populates_last_aux() {
        // EAGLE3 wiring smoke test: enabling the aux tap on a small
        // synthetic config makes `step_cached` populate `last_aux`
        // with one Vec<f32> per requested layer id, each of length
        // `hidden_size`. `take_last_aux` clears it.
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let mut gn = GemmaGenerator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap();

        // Pick up to 3 distinct layer ids from the tiny config —
        // some tiny configs only have 2 layers, in which case the
        // dedupe-and-clamp in set_aux_hidden_layer_ids collapses to
        // 2 ids. Test against whatever survives.
        let n_layers = cfg.num_hidden_layers;
        let mut aux_ids: Vec<usize> = (0..n_layers).take(3).collect();
        aux_ids.sort_unstable();
        aux_ids.dedup();
        gn.set_aux_hidden_layer_ids(aux_ids.clone());
        assert!(gn.aux_enabled());

        gn.prefill(&[1, 2, 3]);
        // First step_cached runs prefill+sample (cache seed), not a
        // decode step — no aux yet. The aux tap only fires on real
        // decode-mode runs.
        let _ = gn.step_cached(SampleOpts::greedy()).unwrap();
        assert!(
            gn.take_last_aux().is_none(),
            "prefill-seed call should not populate aux"
        );
        // Second + third calls are proper decode steps.
        let _ = gn.step_cached(SampleOpts::greedy()).unwrap();
        let aux = gn.take_last_aux().expect("aux populated after decode step");
        assert_eq!(aux.len(), aux_ids.len(), "one row per aux id");
        for row in &aux {
            assert_eq!(row.len(), cfg.hidden_size, "aux row = hidden_size f32s");
            assert!(row.iter().all(|v| v.is_finite()));
        }
        assert!(
            gn.take_last_aux().is_none(),
            "take_last_aux consumes the buffer"
        );

        let _ = gn.step_cached(SampleOpts::greedy()).unwrap();
        let aux2 = gn.take_last_aux().expect("aux on next step");
        assert_eq!(aux2.len(), aux_ids.len());
    }

    #[test]
    fn aux_disabled_after_clear_ids() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let mut gn = GemmaGenerator::from_loader(cfg, &mut wm, Device::Cpu).unwrap();
        gn.set_aux_hidden_layer_ids(vec![0, 1]);
        assert!(gn.aux_enabled());
        gn.set_aux_hidden_layer_ids(Vec::new());
        assert!(!gn.aux_enabled());
        // last_aux is also cleared on set.
        assert!(gn.take_last_aux().is_none());
    }

    #[test]
    fn step_without_prefill_errors() {
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let mut gn = GemmaGenerator::from_loader(cfg, &mut wm, Device::Cpu).unwrap();
        let r = gn.step(SampleOpts::greedy());
        assert!(r.is_err());
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    #[test]
    fn prefill_logits_unchanged_with_kv_export() {
        let cfg = tiny_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5];

        let mut wm_a = synthetic_weights(&cfg);
        let mut wm_b = synthetic_weights(&cfg);
        let (graph_a, params_a) =
            build_gemma_graph_sized_last_logits(&cfg, &mut wm_a, 1, 4, false).unwrap();
        let (graph_b, params_b) =
            build_gemma_graph_sized_last_logits(&cfg, &mut wm_b, 1, 4, true).unwrap();
        let session = Session::new(Device::Cpu);
        let opts = CompileOptions::new();
        let mut ca = session.compile_with(graph_a, &opts);
        let mut cb = session.compile_with(graph_b, &opts);
        for (n, d) in &params_a {
            ca.set_param(n, d);
        }
        for (n, d) in &params_b {
            cb.set_param(n, d);
        }
        let ids: Vec<f32> = prompt.iter().map(|&i| i as f32).collect();
        let la = ca.run(&[("input_ids", &ids)])[0].clone();
        let lb = cb.run(&[("input_ids", &ids)])[0].clone();
        let d = max_abs_diff(&la, &lb);
        assert!(d < 1e-5, "kv export changed prefill logits: max_abs={d:.6}");
    }

    #[test]
    fn incremental_decode_logits_match_full_prefill() {
        let cfg = tiny_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5];

        let mut wm_a = synthetic_weights(&cfg);
        let mut gn_a = GemmaGenerator::from_loader(cfg.clone(), &mut wm_a, Device::Cpu).unwrap();
        let tok = gn_a
            .prefill_get_last_logits(&prompt)
            .map(|l| sample_token(&l, SampleOpts::greedy()) as u32)
            .unwrap();

        let mut extended = prompt.clone();
        extended.push(tok);

        let mut wm_b = synthetic_weights(&cfg);
        let mut gn_b = GemmaGenerator::from_loader(cfg.clone(), &mut wm_b, Device::Cpu).unwrap();
        let full = gn_b.prefill_get_last_logits(&extended).unwrap();

        let mut wm_c = synthetic_weights(&cfg);
        let mut gn_c = GemmaGenerator::from_loader(cfg.clone(), &mut wm_c, Device::Cpu).unwrap();
        gn_c.prefill_get_last_logits(&prompt).unwrap();
        let incremental = gn_c.decode_get_logits(tok).unwrap();

        let d = max_abs_diff(&full, &incremental);
        assert!(
            d < 1e-2,
            "decode+KV vs full prefill max_abs={d:.6} (tok={tok})"
        );
    }

    fn run_prefill_kv(
        cfg: &GemmaConfig,
        wm: &mut WeightMap,
        seq: usize,
        ids: &[u32],
    ) -> Vec<Vec<f32>> {
        run_prefill_kv_with_options(cfg, wm, seq, ids, &kv_export_compile_options(true))
    }

    fn kv_export_compile_options(prefill: bool) -> CompileOptions {
        let profile = if prefill {
            CompileProfile::gemma_prefill()
        } else {
            CompileProfile::gemma_decode()
        };
        compile_options_from_profile(&profile, Device::Cpu, KernelDispatchConfig::default())
    }

    fn run_prefill_kv_with_options(
        cfg: &GemmaConfig,
        wm: &mut WeightMap,
        seq: usize,
        ids: &[u32],
        opts: &CompileOptions,
    ) -> Vec<Vec<f32>> {
        let ids_f32: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
        let (graph, params) = build_gemma_graph_sized_last_logits(cfg, wm, 1, seq, true).unwrap();
        let session = Session::new(Device::Cpu);
        let mut compiled = session.compile_with(graph, opts);
        for (n, d) in &params {
            compiled.set_param(n, d);
        }
        let outputs = compiled.run(&[("input_ids", &ids_f32)]);
        let n_layers = cfg.num_hidden_layers;
        assert_eq!(outputs.len(), 1 + 2 * n_layers);
        let mut kv = Vec::with_capacity(2 * n_layers);
        let mut iter = outputs.into_iter().skip(1);
        for _ in 0..n_layers {
            kv.push(iter.next().unwrap());
            kv.push(iter.next().unwrap());
        }
        kv
    }

    #[test]
    fn decode_graph_bakes_rope_slice_length() {
        let cfg = tiny_cfg();
        let past_seq = 4usize;
        let half = cfg.head_dim() / 2;
        let mut wm = synthetic_weights(&cfg);
        let (_, params) = build_gemma_decode_graph_sized(&cfg, &mut wm, 1, past_seq).unwrap();
        let cos = params
            .get("decode.rope.cos")
            .expect("decode.rope.cos param");
        let sin = params
            .get("decode.rope.sin")
            .expect("decode.rope.sin param");
        assert_eq!(
            cos.len(),
            half,
            "cos param should be one row (half={half}), got {}",
            cos.len()
        );
        assert_eq!(sin.len(), half);
        for key in params.keys() {
            assert!(
                !key.starts_with("rope."),
                "decode graph must not include prefill rope table param {key}"
            );
        }
        let inv = resolve_inv_freq(&cfg, None);
        let (c_ref, s_ref) = rope_slice(&inv, past_seq);
        let d = max_abs_diff(cos, &c_ref) + max_abs_diff(sin, &s_ref);
        assert!(d < 1e-6, "baked rope mismatch: {d}");
    }

    #[test]
    fn decode_graph_all_rope_use_baked_cos() {
        use rlx_ir::Op;
        let cfg = tiny_cfg();
        let mut wm = synthetic_weights(&cfg);
        let (graph, _) = build_gemma_decode_graph_sized(&cfg, &mut wm, 1, 4).unwrap();
        for node in graph.nodes() {
            if let Op::Rope { .. } = &node.op {
                let cos_id = node.inputs[1];
                let cos_node = &graph.node(cos_id);
                match &cos_node.op {
                    Op::Param { name } => assert_eq!(
                        name, "decode.rope.cos",
                        "decode RoPE must use baked decode.rope.cos, got {name}"
                    ),
                    other => panic!("decode RoPE cos input is {other:?}, expected Param"),
                }
            }
        }
    }

    #[test]
    fn decode_graph_rope_cos_is_single_row() {
        use rlx_ir::Op;
        let cfg = tiny_cfg();
        let past_seq = 4usize;
        let half = cfg.head_dim() / 2;
        let mut wm = synthetic_weights(&cfg);
        let (graph, _) = build_gemma_decode_graph_sized(&cfg, &mut wm, 1, past_seq).unwrap();
        let mut rope_cos_lens = Vec::new();
        for node in graph.nodes() {
            if let Op::Rope { .. } = &node.op {
                let cos_shape = &graph.node(node.inputs[1]).shape;
                let rows = if cos_shape.rank() >= 2 {
                    cos_shape.dim(0).unwrap_static()
                } else {
                    1
                };
                rope_cos_lens.push(rows);
            }
        }
        assert!(!rope_cos_lens.is_empty(), "decode graph has no RoPE nodes");
        for rows in &rope_cos_lens {
            assert_eq!(
                *rows, 1,
                "decode RoPE cos must be single-row [1, half], got {rows} rows"
            );
        }
        assert_eq!(half, cfg.head_dim() / 2);
    }

    #[test]
    fn prefill_kv_matches_extended_prefix() {
        let cfg = tiny_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5];
        let tok = 6u32;
        let mut extended = prompt.clone();
        extended.push(tok);

        let mut wm_prompt = synthetic_weights(&cfg);
        let prompt_kv = run_prefill_kv(&cfg, &mut wm_prompt, 4, &prompt);
        let mut wm_ext = synthetic_weights(&cfg);
        let ext_kv = run_prefill_kv(&cfg, &mut wm_ext, 5, &extended);

        let kv_dim = cfg.kv_proj_dim();
        for layer in 0..cfg.num_hidden_layers {
            let k_prompt = &prompt_kv[2 * layer];
            let k_ext = &ext_kv[2 * layer];
            let prefix_len = 4 * kv_dim;
            assert_eq!(k_prompt.len(), prefix_len);
            assert_eq!(k_ext.len(), 5 * kv_dim);
            let d = max_abs_diff(k_prompt, &k_ext[..prefix_len]);
            assert!(
                d < 1e-4,
                "layer {layer} prefill K prefix vs extended K max_abs={d:.6}"
            );
        }
    }

    #[test]
    fn decode_rope_slice_matches_prefill_table_row() {
        let cfg = tiny_cfg();
        let inv = resolve_inv_freq(&cfg, None);
        let (cos_tab, sin_tab) = build_rope_tables(&inv, cfg.max_position_embeddings);
        let half = inv.len();
        for pos in [3usize, 4, 5] {
            let (c, s) = rope_slice(&inv, pos);
            let off = pos * half;
            let d = max_abs_diff(&c, &cos_tab[off..off + half])
                + max_abs_diff(&s, &sin_tab[off..off + half]);
            assert!(d < 1e-6, "rope_slice mismatch at pos {pos}: {d}");
        }
    }

    #[test]
    fn prefill_kv_export_correct_with_fusion() {
        let cfg = tiny_cfg();
        let tok = 6u32;
        let ids = [1u32, 2, 3, 5, tok];
        let opts = kv_export_compile_options(true);
        let mut wm_one = synthetic_weights(&cfg);
        let one_kv = run_prefill_kv_with_options(&cfg, &mut wm_one, 1, &[tok], &opts);
        let mut wm_ext = synthetic_weights(&cfg);
        let ext_kv = run_prefill_kv_with_options(&cfg, &mut wm_ext, 5, &ids, &opts);
        let kv_dim = cfg.kv_proj_dim();
        let d = max_abs_diff(&ext_kv[1][4 * kv_dim..], &one_kv[1][..kv_dim]);
        assert!(d < 1e-4, "KV export mismatch with profile fusion: {d:.6}");

        let mut wm_default = synthetic_weights(&cfg);
        let default_kv =
            run_prefill_kv_with_options(&cfg, &mut wm_default, 5, &ids, &CompileOptions::new());
        let d_default = max_abs_diff(&default_kv[1][4 * kv_dim..], &one_kv[1][..kv_dim]);
        assert!(
            d_default < 1e-4,
            "KV export mismatch with default fusion (got {d_default:.6})"
        );
    }

    #[test]
    fn decode_oneshot_kv_suffix_matches_extended() {
        let cfg = tiny_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5];
        let tok = 6u32;
        let mut extended = prompt.clone();
        extended.push(tok);

        let opts = kv_export_compile_options(false);
        let mut wm_ext = synthetic_weights(&cfg);
        let ext_kv = run_prefill_kv_with_options(&cfg, &mut wm_ext, 5, &extended, &opts);

        let mut wm = synthetic_weights(&cfg);
        let mut gn = GemmaGenerator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap();
        gn.prefill_get_last_logits(&prompt).unwrap();

        let mut wm_d = synthetic_weights(&cfg);
        let (graph, params) = build_gemma_decode_graph_sized(&cfg, &mut wm_d, 1, 4).unwrap();
        let session = Session::new(Device::Cpu);
        let mut compiled = session.compile_with(graph, &opts);
        for (n, d) in &params {
            compiled.set_param(n, d);
        }
        let cache = gn.cache.as_ref().unwrap();
        let key_strs: Vec<String> = (0..cfg.num_hidden_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        let input_ids = [tok as f32];
        let mut inputs: Vec<(&str, &[f32])> = vec![("input_ids", input_ids.as_slice())];
        for i in 0..cfg.num_hidden_layers {
            inputs.push((&key_strs[2 * i], cache.layers_k[i].as_slice()));
            inputs.push((&key_strs[2 * i + 1], cache.layers_v[i].as_slice()));
        }
        let outputs = compiled.run(&inputs);
        let kv_dim = cfg.kv_proj_dim();
        let k_dec = &outputs[1][4 * kv_dim..];

        let d = max_abs_diff(k_dec, &ext_kv[0][4 * kv_dim..]);
        assert!(
            d < 1e-3,
            "decode oneshot layer0 K suffix vs extended max_abs={d:.6}"
        );
    }

    #[test]
    fn decode_logits_match_extended_prefill_after_one_token() {
        let cfg = tiny_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5];
        let tok = 6u32;

        let mut extended = prompt.clone();
        extended.push(tok);

        let mut wm_a = synthetic_weights(&cfg);
        let mut gn_a = GemmaGenerator::from_loader(cfg.clone(), &mut wm_a, Device::Cpu).unwrap();
        let full = gn_a.prefill_get_last_logits(&extended).unwrap();

        let mut wm_b = synthetic_weights(&cfg);
        let mut gn_b = GemmaGenerator::from_loader(cfg.clone(), &mut wm_b, Device::Cpu).unwrap();
        gn_b.prefill_get_last_logits(&prompt).unwrap();
        let inc = gn_b.decode_get_logits(tok).unwrap();

        let d = max_abs_diff(&full, &inc);
        assert!(d < 1e-2, "decode vs extended prefill max_abs={d:.6}");
    }

    #[test]
    fn cached_second_token_matches_naive() {
        let cfg = tiny_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5];

        let mut wm_n = synthetic_weights(&cfg);
        let mut gn_n = GemmaGenerator::from_loader(cfg.clone(), &mut wm_n, Device::Cpu).unwrap();
        gn_n.prefill(&prompt);
        let n0 = gn_n.step(SampleOpts::greedy()).unwrap();
        let n1 = gn_n.step(SampleOpts::greedy()).unwrap();

        let mut wm_c = synthetic_weights(&cfg);
        let mut gn_c = GemmaGenerator::from_loader(cfg.clone(), &mut wm_c, Device::Cpu).unwrap();
        gn_c.prefill(&prompt);
        let c = gn_c.generate_cached(2, SampleOpts::greedy()).unwrap();

        assert_eq!(c[0], n0, "first generated token");
        assert_eq!(c[1], n1, "second generated token (decode step)");
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
            GemmaGenerator::from_loader(cfg.clone(), &mut wm_n, Device::Cpu).unwrap();
        gn_naive.prefill(&prompt);
        let naive_tokens = gn_naive.generate(steps, SampleOpts::greedy()).unwrap();

        let mut wm_c = synthetic_weights(&cfg);
        let mut gn_cached =
            GemmaGenerator::from_loader(cfg.clone(), &mut wm_c, Device::Cpu).unwrap();
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
        let mut gn = GemmaGenerator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap();
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
            GemmaGenerator::from_loader(cfg.clone(), &mut wm_one, Device::Cpu).unwrap();
        gn_one.prefill(&prompt);
        let oneshot_tokens = gn_one.generate_cached(steps, SampleOpts::greedy()).unwrap();

        let mut wm_buc = synthetic_weights(&cfg);
        let mut gn_buc = GemmaGenerator::from_loader(cfg.clone(), &mut wm_buc, Device::Cpu)
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
        let mut gn_a = GemmaGenerator::from_loader(cfg.clone(), &mut wm_a, Device::Cpu).unwrap();
        gn_a.prefill(&prompt);
        let a = gn_a.generate_cached(4, SampleOpts::greedy()).unwrap();

        let mut wm_b = synthetic_weights(&cfg);
        let mut gn_b = GemmaGenerator::from_loader(cfg.clone(), &mut wm_b, Device::Cpu)
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
            GemmaGenerator::from_loader(cfg.clone(), &mut wm_one, Device::Cpu).unwrap();
        gn_one.prefill(&prompt);
        let oneshot_tokens = gn_one.generate_cached(steps, SampleOpts::greedy()).unwrap();

        let mut wm_dyn = synthetic_weights(&cfg);
        let mut gn_dyn = GemmaGenerator::from_loader(cfg.clone(), &mut wm_dyn, Device::Cpu)
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
            GemmaGenerator::from_loader(cfg.clone(), &mut wm_one, Device::Cpu).unwrap();
        gn_one.prefill(&prompt);
        let oneshot_tokens = gn_one.generate_cached(steps, SampleOpts::greedy()).unwrap();

        let mut wm_dyn = synthetic_weights(&cfg);
        let mut gn_dyn = GemmaGenerator::from_loader(cfg.clone(), &mut wm_dyn, Device::Cpu)
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
            GemmaGenerator::from_loader(cfg.clone(), &mut wm_one, Device::Cpu).unwrap();
        gn_one.prefill(&prompt);
        let oneshot_tokens = gn_one.generate_cached(steps, SampleOpts::greedy()).unwrap();

        let mut wm_dyn = synthetic_weights(&cfg);
        let mut gn_dyn = GemmaGenerator::from_loader(cfg.clone(), &mut wm_dyn, Device::Cpu)
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
            GemmaGenerator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap()
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
