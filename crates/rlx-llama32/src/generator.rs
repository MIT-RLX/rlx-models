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
    build_llama32_decode_hir_dynamic_ext, build_llama32_decode_hir_sized_ext,
    build_llama32_graph_sized_kv_tap, build_llama32_graph_sized_last_logits,
    build_llama32_graph_sized_packed, build_llama32_prefill_hir_dynamic_ext,
    build_llama32_prefill_hir_sized_ext,
};
use crate::config::Llama32Config;
use crate::prefill_mode::MetalGgufPrefillMode;
use crate::rope::{resolve_inv_freq, rope_slice};
use anyhow::{Context, Result};
use rlx_core::flow_bridge::{
    compile_options_for_packed_gguf_prefill_with_profile, compile_options_from_profile,
    packed_gguf_compile_guard, packed_gguf_execution_device,
};
use rlx_core::weight_loader::{ArcCacheLoader, ArcF32Tensor, GgufLoader, WeightLoader};
use rlx_core::{compact_bucketed_kv_buffer, infer_prefill_kv_seq, run_packed_prefill};
use rlx_flow::CompileProfile;
use rlx_ir::DimBinding;
use rlx_ir::logical_kernel::KernelDispatchConfig;
use rlx_qwen3::sampling::{SampleOpts, sample_token, sample_token_at};
use rlx_runtime::attn_mask::bucket_decode_mask;
use rlx_runtime::compile_cache::{BucketedCompileCache, CompileCache, DynamicDimCompileCache};
use rlx_runtime::{
    CompileOptions, CompiledGraph, Device, Session, llama_decode_bucket_compile_peak_bytes,
    llama_decode_oneshot_compile_peak_bytes, would_exceed_soft_budget,
};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Metal F32 prefill on-device needs thunk lowering (no MPSGraph), fusion
/// disabled, and scalar fp32 matmul. Only applied when prefill stays on Metal.
fn metal_f32_prefill_guard<R, F>(prefill_device: Device, f: F) -> R
where
    F: FnOnce() -> R,
{
    if prefill_device != Device::Metal {
        return f();
    }
    let save_mps = rlx_ir::env::var("RLX_DISABLE_MPSGRAPH");
    let save_precise = rlx_ir::env::var("RLX_METAL_PRECISE");
    let save_fusion = rlx_ir::env::var("RLX_METAL_NO_FUSION");
    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");
    rlx_ir::env::set("RLX_METAL_PRECISE", "1");
    rlx_ir::env::set("RLX_METAL_NO_FUSION", "1");
    let out = f();
    match save_mps {
        Some(v) => rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", v),
        None => rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH"),
    }
    match save_precise {
        Some(v) => rlx_ir::env::set("RLX_METAL_PRECISE", v),
        None => rlx_ir::env::unset("RLX_METAL_PRECISE"),
    }
    match save_fusion {
        Some(v) => rlx_ir::env::set("RLX_METAL_NO_FUSION", v),
        None => rlx_ir::env::unset("RLX_METAL_NO_FUSION"),
    }
    out
}

/// Packed GGUF prefill: separate logits and KV graphs (Metal logits+KV in one
/// graph diverges like F32 `LlamaKvTap`).
struct PackedGgufPrefill {
    upper_seq: usize,
    logits: CompiledGraph,
    kv: CompiledGraph,
    exec_device: Device,
    kv_dim: usize,
    n_layers: usize,
    ids_f32: Vec<f32>,
    last_idx: [f32; 1],
}

impl PackedGgufPrefill {
    fn build(
        cfg: &Llama32Config,
        path: &Path,
        upper_seq: usize,
        device: Device,
        profile: &CompileProfile,
    ) -> Result<Self> {
        let exec_device = packed_gguf_execution_device(device);
        let path_str = path.to_str().context("non-utf8 weights path")?;
        let mut loader = GgufLoader::from_file(path_str)?;
        let mut packed = HashMap::new();
        let (logits_graph, params) = build_llama32_graph_sized_packed(
            cfg,
            &mut loader,
            1,
            upper_seq,
            true,
            true,
            false,
            &mut packed,
        )?;
        let mut loader2 = GgufLoader::from_file(path_str)?;
        let mut packed_kv = HashMap::new();
        let (kv_graph, params_kv) = build_llama32_graph_sized_packed(
            cfg,
            &mut loader2,
            1,
            upper_seq,
            false,
            false,
            true,
            &mut packed_kv,
        )?;
        let opts = compile_options_for_packed_gguf_prefill_with_profile(profile, exec_device);
        let mut logits = packed_gguf_compile_guard(exec_device, || {
            Session::new(exec_device).compile_with(logits_graph, &opts)
        });
        let mut kv = packed_gguf_compile_guard(exec_device, || {
            Session::new(exec_device).compile_with(kv_graph, &opts)
        });
        for (name, data) in &params {
            logits.set_param(name, data);
        }
        for (name, data) in &params_kv {
            kv.set_param(name, data);
        }
        for (name, (bytes, _scheme, _shape)) in &packed {
            logits.set_param_typed(name, bytes, rlx_ir::DType::U8);
            kv.set_param_typed(name, bytes, rlx_ir::DType::U8);
        }
        Ok(Self {
            upper_seq,
            logits,
            kv,
            exec_device,
            kv_dim: cfg.kv_proj_dim(),
            n_layers: cfg.num_hidden_layers,
            ids_f32: vec![0f32; upper_seq],
            last_idx: [0f32; 1],
        })
    }

    fn fill_inputs(&mut self, prompt_len: usize, ids_f32: &[f32]) -> usize {
        let n = prompt_len.min(self.upper_seq);
        for (i, &v) in ids_f32.iter().take(n).enumerate() {
            self.ids_f32[i] = v;
        }
        for i in n..self.upper_seq {
            self.ids_f32[i] = 0.0;
        }
        self.last_idx[0] = n.saturating_sub(1) as f32;
        n
    }

    fn run_logits(&mut self, prompt_len: usize, ids_f32: &[f32]) -> Result<Vec<f32>> {
        let n = self.fill_inputs(prompt_len, ids_f32);
        let outputs = run_packed_prefill(
            &mut self.logits,
            self.exec_device,
            n,
            self.upper_seq,
            &[
                ("input_ids", self.ids_f32.as_slice()),
                ("last_token_idx", self.last_idx.as_slice()),
            ],
        );
        outputs
            .into_iter()
            .next()
            .context("packed logits prefill returned no outputs")
    }

    fn run_kv_only(
        &mut self,
        prompt_len: usize,
        ids_f32: &[f32],
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let n = self.fill_inputs(prompt_len, ids_f32);
        let kv_outputs = run_packed_prefill(
            &mut self.kv,
            self.exec_device,
            n,
            self.upper_seq,
            &[("input_ids", self.ids_f32.as_slice())],
        );
        let kv_seq = infer_prefill_kv_seq(&kv_outputs, 1, &[self.kv_dim], n, self.upper_seq);
        let (mut layers_k, mut layers_v) =
            split_packed_kv_outputs(kv_outputs, 1, kv_seq, self.kv_dim, self.n_layers)?;
        if kv_seq > n {
            let keep = n * self.kv_dim;
            for i in 0..self.n_layers {
                layers_k[i].truncate(keep);
                layers_v[i].truncate(keep);
            }
        }
        Ok((layers_k, layers_v))
    }

    fn run(
        &mut self,
        prompt_len: usize,
        ids_f32: &[f32],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let n = self.fill_inputs(prompt_len, ids_f32);
        let logits = self.run_logits(prompt_len, ids_f32)?;
        let kv_outputs = run_packed_prefill(
            &mut self.kv,
            self.exec_device,
            n,
            self.upper_seq,
            &[("input_ids", self.ids_f32.as_slice())],
        );
        let kv_seq = infer_prefill_kv_seq(&kv_outputs, 1, &[self.kv_dim], n, self.upper_seq);
        let (mut layers_k, mut layers_v) =
            split_packed_kv_outputs(kv_outputs, 1, kv_seq, self.kv_dim, self.n_layers)?;
        if kv_seq > n {
            let keep = n * self.kv_dim;
            for i in 0..self.n_layers {
                layers_k[i].truncate(keep);
                layers_v[i].truncate(keep);
            }
        }
        Ok((logits, layers_k, layers_v))
    }
}

fn split_packed_kv_outputs(
    outputs: Vec<Vec<f32>>,
    batch: usize,
    seq: usize,
    kv_dim: usize,
    n_layers: usize,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    if outputs.len() != 2 * n_layers {
        anyhow::bail!(
            "packed kv prefill produced {} outputs, expected {}",
            outputs.len(),
            2 * n_layers
        );
    }
    let expected_kv_len = batch * seq * kv_dim;
    let mut iter = outputs.into_iter();
    let mut layers_k = Vec::with_capacity(n_layers);
    let mut layers_v = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        let k = iter.next().context("packed kv k missing")?;
        let v = iter.next().context("packed kv v missing")?;
        if k.len() != expected_kv_len || v.len() != expected_kv_len {
            anyhow::bail!(
                "layer {layer}: k.len={} v.len={} expected {expected_kv_len}",
                k.len(),
                v.len()
            );
        }
        layers_k.push(k);
        layers_v.push(v);
    }
    Ok((layers_k, layers_v))
}

/// Metal decode on real GGUF needs the same thunk / no-fusion / precise
/// matmul path as F32 prefill (MPSGraph decode diverges on 3B Q4).
fn metal_decode_compile_guard<R, F>(device: Device, gguf_parity: bool, decode: bool, f: F) -> R
where
    F: FnOnce() -> R,
{
    if device != Device::Metal || !decode || !gguf_parity {
        return f();
    }
    metal_f32_prefill_guard(Device::Metal, f)
}

/// GGUF on Metal/CPU: mmap weights and dequant per compile instead of draining
/// the full model to host F32 at load (~12 GiB on Orpheus 3B).
pub(crate) fn gguf_defers_f32_drain(
    device: Device,
    is_gguf: bool,
    use_packed_gguf: bool,
    prefill_mode: MetalGgufPrefillMode,
) -> bool {
    if !is_gguf {
        return false;
    }
    if device == Device::Cpu {
        return true;
    }
    // wgpu/Vulkan: mmap GGUF + CPU prefill/decode until GPU KV parity lands.
    if matches!(device, Device::Gpu | Device::Vulkan) {
        return true;
    }
    if device != Device::Metal {
        return false;
    }
    use_packed_gguf
        || matches!(
            prefill_mode.resolve(),
            MetalGgufPrefillMode::CpuF32
                | MetalGgufPrefillMode::Auto
                | MetalGgufPrefillMode::MetalF32
        )
}

/// Per-layer KV cache state for incremental decoding. Each `Vec<f32>`
/// is a flat `[batch, past_seq, kv_proj_dim]` tensor.
#[derive(Clone)]
struct KvCacheState {
    past_seq: usize,
    layers_k: Vec<Vec<f32>>,
    layers_v: Vec<Vec<f32>>,
}

/// Borrow cached F32 tensors or open a fresh GGUF loader per graph build.
enum BuildWeightLoader<'a> {
    Cached(ArcCacheLoader<'a>),
    Gguf(GgufLoader),
}

impl WeightLoader for BuildWeightLoader<'_> {
    fn format_id(&self) -> &'static str {
        match self {
            Self::Cached(l) => l.format_id(),
            Self::Gguf(l) => l.format_id(),
        }
    }
    fn len(&self) -> usize {
        match self {
            Self::Cached(l) => l.len(),
            Self::Gguf(l) => l.len(),
        }
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        match self {
            Self::Cached(l) => l.take(key),
            Self::Gguf(l) => l.take(key),
        }
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        match self {
            Self::Cached(l) => l.take_transposed(key),
            Self::Gguf(l) => l.take_transposed(key),
        }
    }
    fn remaining_keys(&self) -> Vec<String> {
        match self {
            Self::Cached(l) => l.remaining_keys(),
            Self::Gguf(l) => l.remaining_keys(),
        }
    }
}

/// Stateful LLaMA-3.2 generation handle.
///
/// Holds the (config, weight bytes, token history) and rebuilds a
/// prefill graph on each [`step`] call. Cheap to construct after
/// initial weight load; tokens stay in-memory between calls.
pub struct Llama32Generator {
    cfg: Llama32Config,
    /// Map of weight key → (shared f32 data, shape). Graph builds borrow
    /// via [`ArcCacheLoader`] so multi-GB caches are not cloned per step.
    weights_cache: HashMap<String, ArcF32Tensor>,
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
    /// Resolved RoPE inverse frequencies — currently set at
    /// construction for parity with `rlx-gemma`'s generator (which
    /// uses it via [`compute_rope_slice`] in its dynamic-decode
    /// path). The Llama 3.2 dynamic-past decode path isn't wired
    /// yet, so the field is unread today. Kept to make landing that
    /// path a one-call addition.
    #[allow(dead_code)]
    inv_freq: Vec<f64>,
    /// Tier-1 compile profile for prefill graphs.
    prefill_profile: CompileProfile,
    /// Tier-1 compile profile for decode graphs.
    decode_profile: CompileProfile,
    /// Monotonic sample index for [`step_cached`] RNG (`sample_token_at`).
    sample_step: u64,
    /// GGUF path for packed prefill / deferred F32 drain (Metal fast path).
    weights_path: Option<PathBuf>,
    /// Skip full F32 drain at construction; [`Self::ensure_weights`] loads on first decode.
    weights_deferred: bool,
    /// Lazy packed GGUF prefill session (Metal default).
    packed_gguf_prefill: Option<PackedGgufPrefill>,
    /// Metal GGUF prefill strategy ([`MetalGgufPrefillMode::Auto`] reads env).
    metal_gguf_prefill_mode: MetalGgufPrefillMode,
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
            weights_cache.insert(canonical, (std::sync::Arc::new(v.0), v.1));
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
            sample_step: 0,
            weights_path: None,
            weights_deferred: false,
            packed_gguf_prefill: None,
            metal_gguf_prefill_mode: MetalGgufPrefillMode::Auto,
        })
    }

    fn from_loader_deferred(
        cfg: Llama32Config,
        loader: &mut dyn WeightLoader,
        device: Device,
        prefill_mode: MetalGgufPrefillMode,
    ) -> Result<Self> {
        let rope_factors = loader.take("rope_freqs.weight").ok().map(|(d, _)| d);
        let inv_freq = resolve_inv_freq(&cfg, rope_factors.as_deref());
        Ok(Self {
            cfg,
            weights_cache: HashMap::new(),
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
            sample_step: 0,
            weights_path: None,
            weights_deferred: true,
            packed_gguf_prefill: None,
            metal_gguf_prefill_mode: prefill_mode,
        })
    }

    fn build_weight_loader_from<'a>(
        deferred: bool,
        path: &'a Option<PathBuf>,
        cache: &'a HashMap<String, ArcF32Tensor>,
    ) -> Result<BuildWeightLoader<'a>> {
        if deferred {
            let path = path.as_ref().context("deferred weights need gguf path")?;
            let path_str = path.to_str().context("non-utf8 weights path")?;
            Ok(BuildWeightLoader::Gguf(GgufLoader::from_file(path_str)?))
        } else {
            Ok(BuildWeightLoader::Cached(ArcCacheLoader::new(cache)))
        }
    }

    fn cached_weight_loader(&self) -> ArcCacheLoader<'_> {
        ArcCacheLoader::new(&self.weights_cache)
    }

    fn build_weight_loader(&self) -> Result<BuildWeightLoader<'_>> {
        if self.weights_deferred {
            let path = self
                .weights_path
                .as_ref()
                .context("deferred weights need gguf path")?;
            let path_str = path.to_str().context("non-utf8 weights path")?;
            Ok(BuildWeightLoader::Gguf(GgufLoader::from_file(path_str)?))
        } else {
            Ok(BuildWeightLoader::Cached(self.cached_weight_loader()))
        }
    }

    fn ensure_weights(&mut self) -> Result<()> {
        if !self.weights_deferred {
            return Ok(());
        }
        self.weights_path
            .as_ref()
            .context("deferred weights need gguf path")?;
        Ok(())
    }

    /// CPU F32 prefill-with-cache (logits + KV). Uses HIR prefill to match HIR decode.
    fn run_prefill_with_cache_cpu_f32(
        &mut self,
        batch: usize,
        seq: usize,
        ids_f32: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        self.ensure_weights()?;
        let mut loader = self.build_weight_loader()?;
        let (hir, params) = build_llama32_prefill_hir_sized_ext(
            &self.cfg,
            &mut loader,
            batch,
            seq,
            /*with_kv_outputs*/ true,
        )?;
        let session = Session::new(Device::Cpu);
        let mut compiled = self.compile_hir_profiled(&session, hir, false)?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        Ok(compiled.run(&[("input_ids", ids_f32)]))
    }

    fn packed_prefill_upper_seq(&self, prompt_len: usize) -> usize {
        let cap = self.compile_seq_cap();
        let n = prompt_len.max(1);
        if n >= cap {
            return cap;
        }
        // Power-of-two bucket (same as CPU / Llama32Runner). Forcing compile_cap on
        // Metal breaks last_token_idx + active-extent trim (argmax → token 0).
        n.next_power_of_two().min(cap)
    }

    fn ensure_packed_prefill(&mut self, seq: usize) -> Result<()> {
        let upper = self.packed_prefill_upper_seq(seq);
        let need = self
            .packed_gguf_prefill
            .as_ref()
            .is_none_or(|p| p.upper_seq != upper);
        if !need {
            return Ok(());
        }
        let path = self
            .weights_path
            .as_ref()
            .context("packed prefill needs gguf path")?;
        self.packed_gguf_prefill = Some(PackedGgufPrefill::build(
            &self.cfg,
            path,
            upper,
            self.device,
            &self.prefill_profile,
        )?);
        Ok(())
    }

    fn try_packed_gguf_logits(
        &mut self,
        prompt_len: usize,
        ids_f32: &[f32],
    ) -> Result<Option<Vec<f32>>> {
        if self.device != Device::Metal || !self.metal_gguf_prefill_mode().use_packed_gguf() {
            return Ok(None);
        }
        if self.weights_path.is_none() {
            return Ok(None);
        }
        self.ensure_packed_prefill(prompt_len)?;
        let p = self.packed_gguf_prefill.as_mut().unwrap();
        Ok(Some(p.run_logits(prompt_len, ids_f32)?))
    }

    fn try_packed_gguf_kv_only(
        &mut self,
        prompt_len: usize,
        ids_f32: &[f32],
    ) -> Result<Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)>> {
        if self.device != Device::Metal || !self.metal_gguf_prefill_mode().use_packed_gguf() {
            return Ok(None);
        }
        if self.weights_path.is_none() {
            return Ok(None);
        }
        self.ensure_packed_prefill(prompt_len)?;
        let p = self.packed_gguf_prefill.as_mut().unwrap();
        Ok(Some(p.run_kv_only(prompt_len, ids_f32)?))
    }

    /// CPU F32 last-position logits (reference). Used when Metal packed
    /// `lm_head` returns NaN on large-vocab GGUF (e.g. Orpheus 156k).
    fn run_prefill_last_logits_cpu_f32(
        &mut self,
        batch: usize,
        seq: usize,
        ids_f32: &[f32],
    ) -> Result<Vec<f32>> {
        self.ensure_weights()?;
        let prefill_opts = compile_options_from_profile(
            &self.prefill_profile,
            Device::Cpu,
            KernelDispatchConfig::default(),
        );
        // Do not use `prefill_compile_cache` here — its legacy untagged keys in
        // `run_prefill_with_cache` collide with `prefill_cache_key(_, _, 2)` for
        // some `(batch, seq)` pairs and return the wrong graph.
        let mut loader = self.build_weight_loader()?;
        let (graph, params) = build_llama32_graph_sized_last_logits(
            &self.cfg,
            &mut loader,
            batch,
            seq,
            /*with_kv_outputs*/ false,
        )?;
        let session = Session::new(Device::Cpu);
        let mut compiled = session.compile_with(graph, &prefill_opts);
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        let outputs = compiled.run(&[("input_ids", ids_f32)]);
        let logits = outputs
            .into_iter()
            .next()
            .context("cpu f32 logits prefill returned no outputs")?;
        let vocab = self.cfg.vocab_size;
        if logits.len() < vocab {
            anyhow::bail!("cpu f32 logits short: {} < {vocab}", logits.len());
        }
        Ok(logits[..vocab].to_vec())
    }

    fn drop_packed_gguf_prefill(&mut self) {
        self.packed_gguf_prefill = None;
    }

    /// Packed Metal KV + CPU F32 logits (Metal packed lm_head NaN on 156k vocab).
    fn seed_prefill_packed_kv_cpu_logits(
        &mut self,
        batch: usize,
        seq: usize,
        ids_f32: &[f32],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        // KV first while F32 weights are still deferred — avoids holding packed
        // Metal graphs and the full host F32 drain at the same time.
        let (layers_k, layers_v) = self
            .try_packed_gguf_kv_only(seq, ids_f32)?
            .context("packed gguf kv prefill")?;
        self.drop_packed_gguf_prefill();
        let logits = self.run_prefill_last_logits_cpu_f32(batch, seq, ids_f32)?;
        Ok((logits, layers_k, layers_v))
    }

    fn prefill_seed_triple(
        &mut self,
        batch: usize,
        seq: usize,
        ids_f32: &[f32],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        if self.device == Device::Metal && self.metal_gguf_prefill_mode().use_packed_gguf() {
            // Hybrid packed KV + CPU decode diverges after step 1; CPU decode needs
            // CPU F32 KV from the full prefill-with-cache graph.
            if self.decode_device() == Device::Cpu {
                self.ensure_weights()?;
                let outputs = self.run_prefill_with_cache_cpu_f32(batch, seq, ids_f32)?;
                return self.split_prefill_outputs(outputs, batch, seq);
            }
            let triple = self.seed_prefill_packed_kv_cpu_logits(batch, seq, ids_f32)?;
            self.drop_packed_gguf_prefill();
            return Ok(triple);
        }
        if let Some(triple) = self.try_packed_gguf_prefill(seq, ids_f32)? {
            return Ok(triple);
        }
        if self.gguf_cpu_host_path() {
            let outputs = self.run_prefill_with_cache_cpu_f32(batch, seq, ids_f32)?;
            return self.split_prefill_outputs(outputs, batch, seq);
        }
        if self.prefill_device() == Device::Metal && self.prefill_dynamic_cache.is_none() {
            return self.seed_prefill_metal_split(batch, seq, ids_f32);
        }
        let outputs = self.run_prefill_with_cache(batch, seq, ids_f32)?;
        self.split_prefill_outputs(outputs, batch, seq)
    }

    fn try_packed_gguf_prefill(
        &mut self,
        prompt_len: usize,
        ids_f32: &[f32],
    ) -> Result<Option<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)>> {
        if self.device != Device::Metal || !self.metal_gguf_prefill_mode().use_packed_gguf() {
            return Ok(None);
        }
        if self.weights_path.is_none() {
            return Ok(None);
        }
        self.ensure_packed_prefill(prompt_len)?;
        let p = self.packed_gguf_prefill.as_mut().unwrap();
        Ok(Some(p.run(prompt_len, ids_f32)?))
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
        Self::from_loader_at_mode(
            cfg,
            loader,
            device,
            weights_path,
            MetalGgufPrefillMode::Auto,
        )
    }

    /// Like [`Self::from_loader_at`] with an explicit Metal GGUF prefill mode.
    pub fn from_loader_at_mode(
        cfg: Llama32Config,
        loader: &mut dyn WeightLoader,
        device: Device,
        weights_path: &Path,
        prefill_mode: MetalGgufPrefillMode,
    ) -> Result<Self> {
        let is_gguf = weights_path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
        let defer_f32_drain = gguf_defers_f32_drain(
            device,
            is_gguf,
            prefill_mode.use_packed_gguf(),
            prefill_mode,
        );
        let mut g = if defer_f32_drain {
            Self::from_loader_deferred(cfg, loader, device, prefill_mode)?
        } else {
            let mut g = Self::from_loader(cfg, loader, device)?;
            g.metal_gguf_prefill_mode = prefill_mode;
            g
        };
        g.prefill_profile = crate::llama32_profile_near_weights(weights_path, false);
        g.decode_profile = crate::llama32_profile_near_weights(weights_path, true);
        if weights_path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            g.weights_path = Some(weights_path.to_path_buf());
        }
        Ok(g)
    }

    /// Override Metal GGUF prefill strategy (must be set before first prefill when
    /// not using [`Self::from_loader_at_mode`]; changing after load does not undo
    /// an eager F32 drain).
    pub fn with_metal_gguf_prefill_mode(mut self, mode: MetalGgufPrefillMode) -> Self {
        self.metal_gguf_prefill_mode = mode;
        self
    }

    pub fn metal_gguf_prefill_mode(&self) -> MetalGgufPrefillMode {
        self.metal_gguf_prefill_mode
    }

    /// `MetalGgufPrefillMode::resolve` applied to the configured
    /// mode. Used by the prefill-with-GGUF path that lands alongside
    /// the dynamic-decode rope work — kept here so callers don't
    /// have to re-`.resolve()` themselves.
    #[allow(dead_code)]
    fn resolved_metal_gguf_prefill_mode(&self) -> MetalGgufPrefillMode {
        self.metal_gguf_prefill_mode.resolve()
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

    fn prefill_device(&self) -> Device {
        self.metal_gguf_prefill_mode.prefill_device(self.device)
    }

    fn is_gguf_checkpoint(&self) -> bool {
        self.weights_path.as_ref().is_some_and(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        })
    }

    /// GGUF prefill + decode both on CPU (Metal CPU parity, wgpu/Vulkan host path).
    fn gguf_cpu_host_path(&self) -> bool {
        self.is_gguf_checkpoint()
            && self.prefill_device() == Device::Cpu
            && self.decode_device() == Device::Cpu
    }

    /// GGUF KV decode on CPU when GPU prefill/decode would diverge (Metal CPU
    /// prefill, wgpu/Vulkan until portable GPU KV parity lands).
    fn gguf_cpu_decode_required(&self) -> bool {
        if !self.is_gguf_checkpoint() {
            return false;
        }
        match self.device {
            Device::Metal => matches!(
                self.metal_gguf_prefill_mode.resolve(),
                MetalGgufPrefillMode::CpuF32
                    | MetalGgufPrefillMode::Auto
                    | MetalGgufPrefillMode::PackedGguf
            ),
            Device::Gpu | Device::Vulkan => true,
            _ => false,
        }
    }

    /// When Metal GGUF uses CPU F32 prefill, keep decode on CPU too — Metal
    /// decode with CPU-seeded KV diverges on 3B Q4_K_M.
    fn decode_device(&self) -> Device {
        if self.gguf_cpu_decode_required() {
            Device::Cpu
        } else {
            self.device
        }
    }

    fn decode_bucket_allowed(&self) -> bool {
        // Once a bucket graph is loaded this utterance, never switch to oneshot —
        // mixed paths corrupt KV and yield STOP after ~2 speech tokens.
        if self.decode_compile_cache.is_some() && !self.decode_loaded_buckets.is_empty() {
            return true;
        }
        !would_exceed_soft_budget(llama_decode_bucket_compile_peak_bytes())
    }

    fn decode_oneshot_allowed(&self) -> Result<()> {
        if would_exceed_soft_budget(llama_decode_oneshot_compile_peak_bytes()) {
            anyhow::bail!(
                "decode compile would exceed soft RAM budget (~80% of physical RAM); \
                 set RLX_SOFT_MEMORY_FRACTION or ORPHEUS_LOW_MEM=1"
            );
        }
        Ok(())
    }

    fn metal_gguf_parity(&self) -> bool {
        self.device == Device::Metal && self.is_gguf_checkpoint()
    }

    fn profile_compile_options(&self, decode: bool) -> CompileOptions {
        let mut profile = if decode {
            self.decode_profile.clone()
        } else {
            self.prefill_profile.clone()
        };
        if self.prefill_device() == Device::Metal && !decode {
            profile.fusion.skip = true;
        }
        if self.metal_gguf_parity() || self.gguf_cpu_host_path() {
            profile.fusion.skip = true;
        }
        compile_options_from_profile(
            &profile,
            if decode {
                self.decode_device()
            } else {
                self.prefill_device()
            },
            KernelDispatchConfig::default(),
        )
    }

    fn compile_hir_profiled(
        &self,
        session: &Session,
        hir: rlx_ir::hir::HirModule,
        decode: bool,
    ) -> Result<rlx_runtime::CompiledGraph> {
        let opts = self.profile_compile_options(decode);
        if decode {
            Ok(metal_decode_compile_guard(
                self.decode_device(),
                self.metal_gguf_parity(),
                true,
                || session.compile_hir_with(hir, &opts),
            )?)
        } else {
            Ok(metal_f32_prefill_guard(self.prefill_device(), || {
                session.compile_hir_with(hir, &opts)
            })?)
        }
    }

    fn compile_graph_profiled(
        &self,
        session: &Session,
        graph: rlx_ir::Graph,
    ) -> Result<rlx_runtime::CompiledGraph> {
        let opts = self.profile_compile_options(false);
        Ok(self.compile_prefill_graph(session, graph, &opts))
    }

    fn compile_prefill_graph(
        &self,
        session: &Session,
        graph: rlx_ir::Graph,
        opts: &CompileOptions,
    ) -> rlx_runtime::CompiledGraph {
        metal_f32_prefill_guard(self.prefill_device(), || session.compile_with(graph, opts))
    }

    fn prefill_cache_key(batch: usize, seq: usize, tag: u64) -> u64 {
        ((batch as u64) << 32) | ((seq as u64) << 2) | tag
    }

    /// Enable the prefill compile cache with the given LRU capacity.
    /// Useful when the same prompt length is used across multiple
    /// generation runs — the second + Nth run skip the compile +
    /// param-attach roundtrip (~30-50ms per call on CPU).
    pub fn with_prefill_cache(mut self, capacity: usize) -> Self {
        self.prefill_compile_cache = Some(CompileCache::new(self.prefill_device(), capacity));
        self.prefill_dynamic_cache = None;
        self
    }

    /// Compile prefill once with `sym::SEQ`, specialize per prompt length.
    pub fn with_dynamic_prefill_cache(mut self, capacity: usize) -> Self {
        self.prefill_dynamic_cache =
            Some(DynamicDimCompileCache::new(self.prefill_device(), capacity));
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
            self.decode_device(),
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
        self.decode_dynamic_cache =
            Some(DynamicDimCompileCache::new(self.decode_device(), capacity));
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
    /// Clears any KV cache from a prior generation and drops heavy
    /// compile caches so a new utterance stays under the soft RAM budget.
    pub fn prefill(&mut self, prompt_ids: &[u32]) {
        self.tokens.clear();
        self.tokens.extend_from_slice(prompt_ids);
        self.cache = None;
        self.sample_step = 0;
        self.release_decode_graphs();
    }

    /// Drop decode compile graphs between utterances (keep prefill cache).
    pub fn release_decode_graphs(&mut self) {
        self.decode_loaded_buckets.clear();
        if let Some(cache) = self.decode_compile_cache.as_mut() {
            cache.clear_compiled();
        }
        if let Some(cache) = self.decode_dynamic_cache.as_mut() {
            cache.clear();
        }
    }

    /// Free all compiled prefill/decode graphs (weight params).
    pub fn release_compile_graphs(&mut self) {
        self.release_decode_graphs();
        if let Some(cache) = self.prefill_compile_cache.as_mut() {
            cache.clear();
        }
        self.drop_packed_gguf_prefill();
    }

    /// Run one prefill over the current token history and sample the
    /// next token. The sampled token is appended to the history and
    /// returned. Call repeatedly to generate.
    pub fn step(&mut self, opts: SampleOpts) -> Result<u32> {
        if self.tokens.is_empty() {
            anyhow::bail!("step() called with empty token history; call prefill() first");
        }
        let seq = self.tokens.len();
        let ids_f32: Vec<f32> = self.tokens.iter().map(|&i| i as f32).collect();
        let logits = self.run_prefill_logits_graph(1, seq, &ids_f32)?;

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
        let idx = self.sample_step;
        let tok = self.step_cached_adjust(opts, idx, |_| {})?;
        self.sample_step = idx.saturating_add(1);
        Ok(tok)
    }

    /// Like [`step_cached`] but applies `adjust_logits` to the row
    /// before sampling and uses an explicit RNG step index.
    pub fn step_cached_adjust(
        &mut self,
        opts: SampleOpts,
        sample_index: u64,
        mut adjust_logits: impl FnMut(&mut [f32]),
    ) -> Result<u32> {
        if self.tokens.is_empty() {
            anyhow::bail!("step_cached() called with empty token history; call prefill() first");
        }
        if self.cache.is_none() {
            return self.seed_cache_from_prompt(opts, sample_index, adjust_logits);
        }
        let cache = self.cache.as_ref().unwrap();
        let past_seq = cache.past_seq;
        if self.tokens.len() <= past_seq {
            anyhow::bail!(
                "cache invariant violated: tokens.len() {} <= past_seq {}",
                self.tokens.len(),
                past_seq
            );
        }
        let input_tok = self.tokens[past_seq];

        let (mut logits, new_k, new_v) = if self.decode_dynamic_cache.is_some() {
            self.decode_step_dynamic(past_seq, input_tok)?
        } else if self.decode_compile_cache.is_some()
            && self.decode_bucket_allowed()
            && self
                .decode_compile_cache
                .as_ref()
                .unwrap()
                .bucket_for(past_seq as u64)
                .is_some()
        {
            self.decode_step_bucketed(past_seq, input_tok)?
        } else {
            self.decode_oneshot_allowed()?;
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
        adjust_logits(&mut logits);
        let tok = sample_token_at(&logits, opts, sample_index) as u32;
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
        self.ensure_weights()?;
        let cache = self.cache.as_ref().unwrap();

        let mut loader = self.build_weight_loader()?;
        let (hir, params) = build_llama32_decode_hir_sized_ext(
            &self.cfg,
            &mut loader,
            /*batch*/ 1,
            past_seq,
            /*use_custom_mask*/ true,
        )?;
        let session = Session::new(self.decode_device());
        let mut compiled = self.compile_hir_profiled(&session, hir, true)?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }

        let position = decode_position_input(past_seq);
        let input_ids_f32 = [input_tok as f32];
        let mask_len = past_seq + 1;
        let mask = vec![1.0f32; mask_len];
        let n_layers = self.cfg.num_hidden_layers;
        let key_strs: Vec<String> = (0..n_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        let mut inputs: Vec<(&str, &[f32])> = Vec::with_capacity(4 + 2 * n_layers);
        inputs.push(("input_ids", input_ids_f32.as_slice()));
        inputs.push(("position", position.as_slice()));
        inputs.push(("mask", mask.as_slice()));
        if past_seq > 0 {
            for i in 0..n_layers {
                inputs.push((&key_strs[2 * i], cache.layers_k[i].as_slice()));
                inputs.push((&key_strs[2 * i + 1], cache.layers_v[i].as_slice()));
            }
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
        self.ensure_weights()?;
        let cache = self.cache.as_ref().unwrap();
        let binding = DimBinding::batch_past_seq(1, past_seq);
        let opts = self
            .profile_compile_options(true)
            .dim_binding(binding.clone());
        let max_past = self.compile_seq_cap();
        let gguf_parity = self.metal_gguf_parity();
        let device = self.decode_device();
        let cache_dyn = self
            .decode_dynamic_cache
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("dynamic decode without cache"))?;
        let needs_upload = !cache_dyn.contains(past_seq as u64);
        let cfg = self.cfg.clone();
        let weights_deferred = self.weights_deferred;
        let weights_path = self.weights_path.clone();
        let weights_cache = self.weights_cache.clone();
        let compiled = cache_dyn.get_or_specialize(
            past_seq as u64,
            &binding,
            || {
                metal_decode_compile_guard(device, gguf_parity, true, || {
                    let mut loader = Self::build_weight_loader_from(
                        weights_deferred,
                        &weights_path,
                        &weights_cache,
                    )
                    .expect("decode dynamic weight loader");
                    build_llama32_decode_hir_dynamic_ext(&cfg, &mut loader, 1, max_past)
                        .expect("dynamic decode HIR")
                        .0
                })
            },
            &opts,
        )?;
        if needs_upload {
            let mut loader =
                Self::build_weight_loader_from(weights_deferred, &weights_path, &weights_cache)?;
            let (_, params) =
                build_llama32_decode_hir_dynamic_ext(&self.cfg, &mut loader, 1, max_past)?;
            for (name, data) in &params {
                compiled.set_param(name, data);
            }
        }

        let position = decode_position_input(past_seq);
        let input_ids_f32 = [input_tok as f32];
        let key_strs: Vec<String> = (0..self.cfg.num_hidden_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        let mut inputs: Vec<(&str, &[f32])> =
            Vec::with_capacity(4 + 2 * self.cfg.num_hidden_layers);
        inputs.push(("input_ids", input_ids_f32.as_slice()));
        inputs.push(("position", position.as_slice()));
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
        self.ensure_weights()?;
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
            if !self.decode_bucket_allowed() {
                anyhow::bail!(
                    "decode bucket compile would exceed soft RAM budget (~80% of physical RAM); \
                     set RLX_SOFT_MEMORY_FRACTION or ORPHEUS_DECODE_CACHE_CAP lower"
                );
            }
            if let Some(cache_mut) = self.decode_compile_cache.as_mut() {
                cache_mut.evict_except(bucket_idx);
            }
            self.decode_loaded_buckets.clear();
            let mut loader = self.build_weight_loader()?;
            let (hir, params) = build_llama32_decode_hir_sized_ext(
                &self.cfg,
                &mut loader,
                /*batch*/ 1,
                upper,
                /*use_custom_mask*/ true,
            )?;
            {
                let decode_opts = self.profile_compile_options(true);
                let gguf_parity = self.metal_gguf_parity();
                let device = self.decode_device();
                let cache_mut = self.decode_compile_cache.as_mut().unwrap();
                metal_decode_compile_guard(device, gguf_parity, true, || {
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
        let position = decode_position_input(past_seq);
        let input_ids_f32 = [input_tok as f32];
        let mask = bucket_decode_mask(past_seq, upper);

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
        inputs.push(("position", position.as_slice()));
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

        // Graph emits K/V at length `upper + 1` (padded past + new token at row
        // `upper`). Compact to dense `[0..past_seq]` + new row — not `buf[..past_seq+1]`.
        let mut iter = raw_outputs.into_iter();
        let logits = iter.next().context("bucketed decode logits missing")?;
        let past_len = past_seq + 1;
        let mut new_k = Vec::with_capacity(n_layers);
        let mut new_v = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let k = iter.next().context("bucketed k missing")?;
            let v = iter.next().context("bucketed v missing")?;
            new_k.push(compact_bucketed_kv_buffer(&k, past_len, kv_dim, 1));
            new_v.push(compact_bucketed_kv_buffer(&v, past_len, kv_dim, 1));
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
        metal_f32_prefill_guard(self.prefill_device(), || {
            if self.prefill_device() == Device::Metal && self.prefill_dynamic_cache.is_none() {
                self.run_prefill_with_cache_metal_split(batch, seq, ids_f32)
            } else {
                self.run_prefill_with_cache_inner(batch, seq, ids_f32)
            }
        })
    }

    /// Metal: fused prefill + [`LlamaKvTap`] in one graph corrupts logits on
    /// real GGUF; run logits-only and KV-tap graphs separately.
    fn run_prefill_with_cache_metal_split(
        &mut self,
        batch: usize,
        seq: usize,
        ids_f32: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        let (logits, layers_k, layers_v) = self.seed_prefill_metal_split(batch, seq, ids_f32)?;
        let mut out = vec![logits];
        for (k, v) in layers_k.into_iter().zip(layers_v) {
            out.push(k);
            out.push(v);
        }
        Ok(out)
    }

    fn run_prefill_logits_graph(
        &mut self,
        batch: usize,
        seq: usize,
        ids_f32: &[f32],
    ) -> Result<Vec<f32>> {
        metal_f32_prefill_guard(self.prefill_device(), || {
            self.run_prefill_logits_graph_inner(batch, seq, ids_f32)
        })
    }

    fn run_prefill_logits_graph_inner(
        &mut self,
        batch: usize,
        seq: usize,
        ids_f32: &[f32],
    ) -> Result<Vec<f32>> {
        if let Some(logits) = self.try_packed_gguf_logits(seq, ids_f32)? {
            return Ok(logits);
        }
        self.ensure_weights()?;
        let weights_deferred = self.weights_deferred;
        let weights_path = self.weights_path.clone();
        let weights_cache = self.weights_cache.clone();
        const TAG_LOGITS: u64 = 0;
        let prefill_opts = self.profile_compile_options(false);
        let outputs = if let Some(prefill_cache) = self.prefill_compile_cache.as_mut() {
            let key = Self::prefill_cache_key(batch, seq, TAG_LOGITS);
            if !prefill_cache.contains(key) {
                let mut loader = Self::build_weight_loader_from(
                    weights_deferred,
                    &weights_path,
                    &weights_cache,
                )?;
                let (graph, params) = build_llama32_graph_sized_last_logits(
                    &self.cfg,
                    &mut loader,
                    batch,
                    seq,
                    /*with_kv_outputs*/ false,
                )?;
                {
                    let compiled =
                        prefill_cache.get_or_compile_with_options(key, || graph, &prefill_opts);
                    for (name, data) in &params {
                        compiled.set_param(name, data);
                    }
                }
            }
            let compiled = prefill_cache.get_or_compile_with_options(
                key,
                || unreachable!("logits prefill cache populated above"),
                &prefill_opts,
            );
            compiled.run(&[("input_ids", ids_f32)])
        } else {
            let mut loader =
                Self::build_weight_loader_from(weights_deferred, &weights_path, &weights_cache)?;
            let (graph, params) = build_llama32_graph_sized_last_logits(
                &self.cfg,
                &mut loader,
                batch,
                seq,
                /*with_kv_outputs*/ false,
            )?;
            let session = Session::new(self.prefill_device());
            let mut compiled = self.compile_graph_profiled(&session, graph)?;
            for (name, data) in &params {
                compiled.set_param(name, data);
            }
            compiled.run(&[("input_ids", ids_f32)])
        };
        outputs
            .into_iter()
            .next()
            .context("logits-only prefill returned no outputs")
    }

    fn run_prefill_kv_tap_graph(
        &mut self,
        batch: usize,
        seq: usize,
        ids_f32: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        metal_f32_prefill_guard(self.prefill_device(), || {
            self.run_prefill_kv_tap_graph_inner(batch, seq, ids_f32)
        })
    }

    fn run_prefill_kv_tap_graph_inner(
        &mut self,
        batch: usize,
        seq: usize,
        ids_f32: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        self.ensure_weights()?;
        let weights_deferred = self.weights_deferred;
        let weights_path = self.weights_path.clone();
        let weights_cache = self.weights_cache.clone();
        const TAG_KV: u64 = 1;
        let n_layers = self.cfg.num_hidden_layers;
        let prefill_opts = self.profile_compile_options(false);
        let outputs = if let Some(prefill_cache) = self.prefill_compile_cache.as_mut() {
            let key = Self::prefill_cache_key(batch, seq, TAG_KV);
            if !prefill_cache.contains(key) {
                let mut loader = Self::build_weight_loader_from(
                    weights_deferred,
                    &weights_path,
                    &weights_cache,
                )?;
                let (graph, params) =
                    build_llama32_graph_sized_kv_tap(&self.cfg, &mut loader, batch, seq)?;
                {
                    let compiled =
                        prefill_cache.get_or_compile_with_options(key, || graph, &prefill_opts);
                    for (name, data) in &params {
                        compiled.set_param(name, data);
                    }
                }
            }
            let compiled = prefill_cache.get_or_compile_with_options(
                key,
                || unreachable!("kv prefill cache populated above"),
                &prefill_opts,
            );
            compiled.run(&[("input_ids", ids_f32)])
        } else {
            let mut loader =
                Self::build_weight_loader_from(weights_deferred, &weights_path, &weights_cache)?;
            let (graph, params) =
                build_llama32_graph_sized_kv_tap(&self.cfg, &mut loader, batch, seq)?;
            let session = Session::new(self.prefill_device());
            let mut compiled = self.compile_graph_profiled(&session, graph)?;
            for (name, data) in &params {
                compiled.set_param(name, data);
            }
            compiled.run(&[("input_ids", ids_f32)])
        };
        if outputs.len() != 1 + 2 * n_layers {
            anyhow::bail!(
                "kv-tap prefill produced {} outputs, expected {}",
                outputs.len(),
                1 + 2 * n_layers
            );
        }
        Ok(outputs.into_iter().skip(1).collect())
    }

    fn run_prefill_with_cache_inner(
        &mut self,
        batch: usize,
        seq: usize,
        ids_f32: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        self.ensure_weights()?;
        let weights_deferred = self.weights_deferred;
        let weights_path = self.weights_path.clone();
        let weights_cache = self.weights_cache.clone();
        let compile_cap = self.compile_seq_cap();
        let prefill_opts = self.profile_compile_options(false);
        let dynamic_prefill = self.prefill_dynamic_cache.is_some().then(|| {
            let binding = DimBinding::batch_seq(batch, seq);
            let opts = prefill_opts.clone().dim_binding(binding.clone());
            (binding, opts)
        });
        if let (Some(cache), Some((binding, opts))) = (
            self.prefill_dynamic_cache.as_mut(),
            dynamic_prefill.as_ref(),
        ) {
            let max_seq = compile_cap;
            let needs_upload = !cache.contains(seq as u64);
            let cfg = self.cfg.clone();
            let weights_path = weights_path.clone();
            let weights_cache = weights_cache.clone();
            let compiled = cache.get_or_specialize(
                seq as u64,
                binding,
                || {
                    let mut loader = Self::build_weight_loader_from(
                        weights_deferred,
                        &weights_path,
                        &weights_cache,
                    )
                    .expect("dynamic prefill weight loader");
                    build_llama32_prefill_hir_dynamic_ext(&cfg, &mut loader, batch, max_seq, true)
                        .expect("dynamic prefill HIR")
                        .0
                },
                opts,
            )?;
            if needs_upload {
                let mut loader = Self::build_weight_loader_from(
                    weights_deferred,
                    &weights_path,
                    &weights_cache,
                )?;
                let (_, params) = build_llama32_prefill_hir_dynamic_ext(
                    &self.cfg,
                    &mut loader,
                    batch,
                    max_seq,
                    true,
                )?;
                for (name, data) in &params {
                    compiled.set_param(name, data);
                }
            }
            let last_idx = vec![(seq - 1) as f32];
            Ok(compiled.run(&[("input_ids", ids_f32), ("last_token_idx", &last_idx)]))
        } else if let Some(prefill_cache) = self.prefill_compile_cache.as_mut() {
            const TAG_PREFILL_FUSED: u64 = 2;
            let key = Self::prefill_cache_key(batch, seq, TAG_PREFILL_FUSED);
            if !prefill_cache.contains(key) {
                let mut loader = Self::build_weight_loader_from(
                    weights_deferred,
                    &weights_path,
                    &weights_cache,
                )?;
                let (hir, params) = build_llama32_prefill_hir_sized_ext(
                    &self.cfg,
                    &mut loader,
                    batch,
                    seq,
                    /*with_kv_outputs*/ true,
                )?;
                {
                    let compiled =
                        prefill_cache.get_or_compile_hir_with_options(key, || hir, &prefill_opts);
                    for (name, data) in &params {
                        compiled.set_param(name, data);
                    }
                }
            }
            let compiled = prefill_cache.get_or_compile_hir_with_options(
                key,
                || unreachable!("just populated above"),
                &prefill_opts,
            );
            Ok(compiled.run(&[("input_ids", ids_f32)]))
        } else {
            let mut loader =
                Self::build_weight_loader_from(weights_deferred, &weights_path, &weights_cache)?;
            let (hir, params) = build_llama32_prefill_hir_sized_ext(
                &self.cfg,
                &mut loader,
                batch,
                seq,
                /*with_kv_outputs*/ true,
            )?;
            let session = Session::new(self.prefill_device());
            let mut compiled = self.compile_hir_profiled(&session, hir, false)?;
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

    fn seed_prefill_metal_split(
        &mut self,
        batch: usize,
        seq: usize,
        ids_f32: &[f32],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let logits = self.run_prefill_logits_graph(batch, seq, ids_f32)?;
        let kv = self.run_prefill_kv_tap_graph(batch, seq, ids_f32)?;
        let n_layers = self.cfg.num_hidden_layers;
        let kv_dim = self.cfg.kv_proj_dim();
        let expected_kv_len = batch * seq * kv_dim;
        if kv.len() != 2 * n_layers {
            anyhow::bail!(
                "kv-tap prefill returned {} tensors, expected {}",
                kv.len(),
                2 * n_layers
            );
        }
        let mut layers_k = Vec::with_capacity(n_layers);
        let mut layers_v = Vec::with_capacity(n_layers);
        for layer in 0..n_layers {
            let k = kv[2 * layer].clone();
            let v = kv[2 * layer + 1].clone();
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
        Ok((logits, layers_k, layers_v))
    }

    fn split_prefill_outputs(
        &self,
        outputs: Vec<Vec<f32>>,
        batch: usize,
        seq: usize,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let n_layers = self.cfg.num_hidden_layers;
        if outputs.len() != 1 + 2 * n_layers {
            anyhow::bail!(
                "prefill-with-cache produced {} outputs, expected {}",
                outputs.len(),
                1 + 2 * n_layers
            );
        }
        let kv_dim = self.cfg.kv_proj_dim();
        let expected_kv_len = batch * seq * kv_dim;
        let mut iter = outputs.into_iter();
        let logits = iter.next().context("prefill logits missing")?;
        let mut layers_k = Vec::with_capacity(n_layers);
        let mut layers_v = Vec::with_capacity(n_layers);
        for layer in 0..n_layers {
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
        Ok((logits, layers_k, layers_v))
    }

    /// Run prefill-with-cache on the current `self.tokens` (the
    /// prompt), populate `self.cache`, sample the next token from the
    /// last position's logits, and append it. Returns the sampled
    /// token. Invariant after: `cache.past_seq == tokens.len() - 1`.
    fn seed_cache_from_prompt(
        &mut self,
        opts: SampleOpts,
        sample_index: u64,
        mut adjust_logits: impl FnMut(&mut [f32]),
    ) -> Result<u32> {
        let seq = self.tokens.len();
        let batch = 1usize;

        let ids_f32: Vec<f32> = self.tokens.iter().map(|&i| i as f32).collect();
        let (logits, layers_k, layers_v) = self.prefill_seed_triple(batch, seq, &ids_f32)?;
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
        let mut last_row = logits[..vocab].to_vec();
        adjust_logits(&mut last_row);
        let tok = sample_token_at(&last_row, opts, sample_index) as u32;
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
        self.sample_step = 0;

        let seq = context.len();
        let batch = 1usize;

        let ids_f32: Vec<f32> = context.iter().map(|&i| i as f32).collect();
        let (logits, layers_k, layers_v) = self.prefill_seed_triple(batch, seq, &ids_f32)?;
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

        let (logits, new_k, new_v) = if self.decode_dynamic_cache.is_some() {
            self.decode_step_dynamic(past_seq, input)?
        } else if self.decode_compile_cache.is_some()
            && self.decode_bucket_allowed()
            && self
                .decode_compile_cache
                .as_ref()
                .unwrap()
                .bucket_for(past_seq as u64)
                .is_some()
        {
            self.decode_step_bucketed(past_seq, input)?
        } else {
            self.decode_oneshot_allowed()?;
            self.decode_step_oneshot(past_seq, input)?
        };

        if logits.len() != self.cfg.vocab_size {
            anyhow::bail!(
                "decode_get_logits: logits length {} != vocab {}",
                logits.len(),
                self.cfg.vocab_size
            );
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
fn decode_position_input(past_seq: usize) -> [f32; 1] {
    [past_seq as f32]
}

/// Thin wrapper over `rope::rope_slice` used by the dynamic-past
/// decode path (currently unwired — see [`Llama32Generator::inv_freq`]).
#[allow(dead_code)]
fn compute_rope_slice(inv_freq: &[f64], pos: usize) -> (Vec<f32>, Vec<f32>) {
    rope_slice(inv_freq, pos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Llama32Config;
    use rlx_core::WeightMap;

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
    fn metal_f32_gguf_defers_host_drain() {
        use crate::prefill_mode::MetalGgufPrefillMode;
        assert!(gguf_defers_f32_drain(
            Device::Metal,
            true,
            false,
            MetalGgufPrefillMode::MetalF32,
        ));
        assert!(!gguf_defers_f32_drain(
            Device::Metal,
            false,
            false,
            MetalGgufPrefillMode::MetalF32,
        ));
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

    fn llama3ish_cfg() -> Llama32Config {
        Llama32Config {
            vocab_size: 128,
            hidden_size: 512,
            intermediate_size: 1024,
            num_hidden_layers: 2,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            attention_bias: false,
            head_dim: Some(128),
            rope_scaling: None,
        }
    }

    fn synthetic_weights_map(cfg: &Llama32Config) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
        let mut wm = synthetic_weights(cfg);
        let keys: Vec<String> = wm.keys().map(|s| s.to_string()).collect();
        let mut out = HashMap::new();
        for k in keys {
            out.insert(k.clone(), wm.take(&k).unwrap());
        }
        out
    }

    #[test]
    #[cfg(feature = "metal")]
    fn metal_decode_past0_matches_prefill() {
        if !rlx_runtime::is_available(Device::Metal) {
            return;
        }
        use crate::builder::build_llama32_decode_hir_sized;

        let mut cfg = llama3ish_cfg();
        cfg.num_hidden_layers = 1;
        let tok = 42u32;
        let past_seq = 0usize;

        let weights = synthetic_weights_map(&cfg);

        let run = |device: Device| -> Vec<f32> {
            let mut wm = WeightMap::from_tensors(weights.clone());
            let (hir, params) = build_llama32_decode_hir_sized(&cfg, &mut wm, 1, past_seq).unwrap();
            let session = Session::new(device);
            let mut compiled = session.compile_hir(hir).expect("compile");
            for (name, data) in &params {
                compiled.set_param(name, data);
            }
            compiled.run(&[("input_ids", &[tok as f32][..])]).remove(0)
        };

        let cpu_decode = run(Device::Cpu);
        let metal_decode = run(Device::Metal);

        let mut gn = Llama32Generator::from_loader(
            cfg.clone(),
            &mut WeightMap::from_tensors(weights.clone()),
            Device::Cpu,
        )
        .unwrap();
        let cpu_prefill = gn.prefill_get_last_logits(&[tok]).unwrap();

        let mut gn_metal = Llama32Generator::from_loader(
            cfg.clone(),
            &mut WeightMap::from_tensors(weights.clone()),
            Device::Metal,
        )
        .unwrap();
        let metal_prefill = gn_metal.prefill_get_last_logits(&[tok]).unwrap();

        let max_metal = cpu_decode
            .iter()
            .zip(metal_decode.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_mp = metal_prefill
            .iter()
            .zip(metal_decode.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_dp = cpu_decode
            .iter()
            .zip(cpu_prefill.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "decode past0 cpu vs metal={max_metal:.6} metal prefill vs decode={max_mp:.6} cpu decode vs prefill={max_dp:.6}"
        );
        assert!(max_mp < 1e-3, "metal prefill vs decode diverged");
        assert!(max_metal < 1e-3, "metal decode vs cpu diverged");
    }

    #[test]
    #[cfg(feature = "metal")]
    fn metal_prefill_seq1_matches_cpu() {
        if !rlx_runtime::is_available(Device::Metal) {
            return;
        }
        let cfg = llama3ish_cfg();
        let prompt: Vec<u32> = vec![42];

        let mut wm_cpu = synthetic_weights(&cfg);
        let mut gn_cpu =
            Llama32Generator::from_loader(cfg.clone(), &mut wm_cpu, Device::Cpu).unwrap();
        let cpu = gn_cpu.prefill_get_last_logits(&prompt).unwrap();

        let mut wm_metal = synthetic_weights(&cfg);
        let mut gn_metal =
            Llama32Generator::from_loader(cfg.clone(), &mut wm_metal, Device::Metal).unwrap();
        let metal = gn_metal.prefill_get_last_logits(&prompt).unwrap();

        let max_abs = cpu
            .iter()
            .zip(metal.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("prefill seq1 max_abs={max_abs:.6}");
        assert!(max_abs < 1e-3);
    }

    #[test]
    #[cfg(feature = "metal")]
    fn metal_one_layer_decode_logits_match_cpu() {
        if !rlx_runtime::is_available(Device::Metal) {
            return;
        }
        let mut cfg = llama3ish_cfg();
        cfg.num_hidden_layers = 1;
        let prompt: Vec<u32> = vec![1, 2, 3, 5, 8, 13];

        let mut wm_cpu = synthetic_weights(&cfg);
        let mut gn_cpu =
            Llama32Generator::from_loader(cfg.clone(), &mut wm_cpu, Device::Cpu).unwrap();
        let pre_cpu = gn_cpu.prefill_get_last_logits(&prompt).unwrap();
        let tok = pre_cpu
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        let cpu_decode = gn_cpu.decode_get_logits(tok).unwrap();

        let mut wm_metal = synthetic_weights(&cfg);
        let mut gn_metal =
            Llama32Generator::from_loader(cfg.clone(), &mut wm_metal, Device::Metal).unwrap();
        let _ = gn_metal.prefill_get_last_logits(&prompt).unwrap();
        let metal_decode = gn_metal.decode_get_logits(tok).unwrap();

        let max_abs = cpu_decode
            .iter()
            .zip(metal_decode.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("1-layer decode logits max_abs={max_abs:.6}");
        assert!(
            max_abs < 1e-3,
            "1-layer decode diverged (max_abs={max_abs})"
        );
    }

    #[test]
    #[cfg(feature = "metal")]
    fn metal_bucketed_decode_matches_cpu() {
        if !rlx_runtime::is_available(Device::Metal) {
            return;
        }
        let cfg = llama3ish_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5, 8, 13];
        let steps = 4;

        let mut wm_cpu = synthetic_weights(&cfg);
        let mut gn_cpu = Llama32Generator::from_loader(cfg.clone(), &mut wm_cpu, Device::Cpu)
            .unwrap()
            .with_decode_cache(64);
        gn_cpu.prefill(&prompt);
        let cpu = gn_cpu.generate_cached(steps, SampleOpts::greedy()).unwrap();

        let mut wm_metal = synthetic_weights(&cfg);
        let mut gn_metal = Llama32Generator::from_loader(cfg.clone(), &mut wm_metal, Device::Metal)
            .unwrap()
            .with_decode_cache(64);
        gn_metal.prefill(&prompt);
        let metal = gn_metal
            .generate_cached(steps, SampleOpts::greedy())
            .unwrap();

        eprintln!("bucketed cpu={cpu:?} metal={metal:?}");
        assert_eq!(metal, cpu, "bucketed decode diverged");
    }

    #[test]
    #[cfg(feature = "metal")]
    fn metal_prefill_kv_matches_cpu() {
        if !rlx_runtime::is_available(Device::Metal) {
            return;
        }
        let cfg = llama3ish_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5, 8, 13];

        let mut wm_cpu = synthetic_weights(&cfg);
        let mut gn_cpu =
            Llama32Generator::from_loader(cfg.clone(), &mut wm_cpu, Device::Cpu).unwrap();
        let _ = gn_cpu.prefill_get_last_logits(&prompt).unwrap();
        let cpu_k = gn_cpu.cache.as_ref().unwrap().layers_k[0].clone();

        let mut wm_metal = synthetic_weights(&cfg);
        let mut gn_metal =
            Llama32Generator::from_loader(cfg.clone(), &mut wm_metal, Device::Metal).unwrap();
        let _ = gn_metal.prefill_get_last_logits(&prompt).unwrap();
        let metal_k = gn_metal.cache.as_ref().unwrap().layers_k[0].clone();

        let max_abs = cpu_k
            .iter()
            .zip(metal_k.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("prefill kv layer0 max_abs={max_abs:.6} len={}", cpu_k.len());
        assert!(max_abs < 1e-3, "prefill KV diverged (max_abs={max_abs})");
    }

    #[test]
    #[cfg(feature = "metal")]
    fn metal_decode_bisect_shape() {
        if !rlx_runtime::is_available(Device::Metal) {
            return;
        }
        for (label, nh, nkv, hd, layers) in [
            ("tiny", 4usize, 2, 8, 2),
            ("hd128_small", 4, 2, 128, 2),
            ("nh16_hd8", 16, 8, 8, 2),
            ("nh16_hd128", 16, 8, 128, 2),
        ] {
            let cfg = Llama32Config {
                vocab_size: 64,
                hidden_size: nh * hd,
                intermediate_size: nh * hd * 2,
                num_hidden_layers: layers,
                num_attention_heads: nh,
                num_key_value_heads: nkv,
                max_position_embeddings: 64,
                rms_norm_eps: 1e-5,
                rope_theta: 500_000.0,
                hidden_act: "silu".into(),
                tie_word_embeddings: false,
                attention_bias: false,
                head_dim: Some(hd),
                rope_scaling: None,
            };
            let prompt: Vec<u32> = vec![1, 2, 3, 5];

            let mut wm_cpu = synthetic_weights(&cfg);
            let mut gn_cpu =
                Llama32Generator::from_loader(cfg.clone(), &mut wm_cpu, Device::Cpu).unwrap();
            let _ = gn_cpu.prefill_get_last_logits(&prompt).unwrap();
            let cpu = gn_cpu.decode_get_logits(1).unwrap();

            let mut wm_metal = synthetic_weights(&cfg);
            let mut gn_metal =
                Llama32Generator::from_loader(cfg.clone(), &mut wm_metal, Device::Metal).unwrap();
            let _ = gn_metal.prefill_get_last_logits(&prompt).unwrap();
            let metal = gn_metal.decode_get_logits(1).unwrap();

            let max_abs = cpu
                .iter()
                .zip(metal.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!("{label}: decode max_abs={max_abs:.6}");
            assert!(max_abs < 1e-3, "{label} decode diverged");
        }
    }

    #[test]
    #[cfg(feature = "metal")]
    fn metal_decode_bisect_head_dim_only() {
        if !rlx_runtime::is_available(Device::Metal) {
            return;
        }
        for hd in [8usize, 32, 64, 128] {
            let nh = 16usize;
            let cfg = Llama32Config {
                vocab_size: 64,
                hidden_size: nh * hd,
                intermediate_size: nh * hd * 2,
                num_hidden_layers: 2,
                num_attention_heads: nh,
                num_key_value_heads: 8,
                max_position_embeddings: 64,
                rms_norm_eps: 1e-5,
                rope_theta: 500_000.0,
                hidden_act: "silu".into(),
                tie_word_embeddings: false,
                attention_bias: false,
                head_dim: Some(hd),
                rope_scaling: None,
            };
            let prompt: Vec<u32> = vec![1, 2, 3, 5];

            let mut wm_cpu = synthetic_weights(&cfg);
            let mut gn_cpu =
                Llama32Generator::from_loader(cfg.clone(), &mut wm_cpu, Device::Cpu).unwrap();
            let _ = gn_cpu.prefill_get_last_logits(&prompt).unwrap();
            let cpu = gn_cpu.decode_get_logits(1).unwrap();

            let mut wm_metal = synthetic_weights(&cfg);
            let mut gn_metal =
                Llama32Generator::from_loader(cfg.clone(), &mut wm_metal, Device::Metal).unwrap();
            let _ = gn_metal.prefill_get_last_logits(&prompt).unwrap();
            let metal = gn_metal.decode_get_logits(1).unwrap();

            let max_abs = cpu
                .iter()
                .zip(metal.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!("hd={hd} decode logits max_abs={max_abs:.6}");
            assert!(max_abs < 1e-3, "hd={hd} decode diverged");
        }
    }

    #[test]
    #[cfg(feature = "metal")]
    fn metal_decode_logits_match_cpu() {
        if !rlx_runtime::is_available(Device::Metal) {
            return;
        }
        let cfg = llama3ish_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5, 8, 13];

        let mut wm_cpu = synthetic_weights(&cfg);
        let mut gn_cpu =
            Llama32Generator::from_loader(cfg.clone(), &mut wm_cpu, Device::Cpu).unwrap();
        let pre_cpu = gn_cpu.prefill_get_last_logits(&prompt).unwrap();
        let tok = pre_cpu
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        let cpu_decode = gn_cpu.decode_get_logits(tok).unwrap();

        let mut wm_metal = synthetic_weights(&cfg);
        let mut gn_metal =
            Llama32Generator::from_loader(cfg.clone(), &mut wm_metal, Device::Metal).unwrap();
        let _ = gn_metal.prefill_get_last_logits(&prompt).unwrap();
        let metal_decode = gn_metal.decode_get_logits(tok).unwrap();

        let max_abs = cpu_decode
            .iter()
            .zip(metal_decode.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("decode logits max_abs={max_abs:.6} tok={tok}");
        assert!(max_abs < 1e-3, "decode logits diverged (max_abs={max_abs})");
    }

    #[test]
    #[cfg(feature = "metal")]
    fn metal_prefill_logits_match_cpu() {
        if !rlx_runtime::is_available(Device::Metal) {
            return;
        }
        let cfg = llama3ish_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5, 8, 13];

        let mut wm_cpu = synthetic_weights(&cfg);
        let mut gn_cpu =
            Llama32Generator::from_loader(cfg.clone(), &mut wm_cpu, Device::Cpu).unwrap();
        let cpu_logits = gn_cpu.prefill_get_last_logits(&prompt).unwrap();

        let mut wm_metal = synthetic_weights(&cfg);
        let mut gn_metal =
            Llama32Generator::from_loader(cfg.clone(), &mut wm_metal, Device::Metal).unwrap();
        let metal_logits = gn_metal.prefill_get_last_logits(&prompt).unwrap();

        let max_abs = cpu_logits
            .iter()
            .zip(metal_logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("prefill logits max_abs={max_abs:.6}");
        assert!(
            max_abs < 1e-3,
            "prefill logits diverged (max_abs={max_abs})"
        );
    }

    #[test]
    #[cfg(feature = "metal")]
    fn metal_decode_bisect_head_dim() {
        if !rlx_runtime::is_available(Device::Metal) {
            return;
        }
        for (label, nh, nkv, hd) in [
            ("mha_hd8", 4, 4, 8),
            ("gqa_hd8", 4, 2, 8),
            ("mha_hd128", 16, 16, 128),
            ("gqa_hd128", 16, 8, 128),
        ] {
            let cfg = Llama32Config {
                vocab_size: 64,
                hidden_size: nh * hd,
                intermediate_size: nh * hd * 2,
                num_hidden_layers: 1,
                num_attention_heads: nh,
                num_key_value_heads: nkv,
                max_position_embeddings: 64,
                rms_norm_eps: 1e-5,
                rope_theta: 500_000.0,
                hidden_act: "silu".into(),
                tie_word_embeddings: false,
                attention_bias: false,
                head_dim: Some(hd),
                rope_scaling: None,
            };
            let prompt: Vec<u32> = vec![1, 2, 3, 5];
            let steps = 3;

            let mut wm_cpu = synthetic_weights(&cfg);
            let mut gn_cpu =
                Llama32Generator::from_loader(cfg.clone(), &mut wm_cpu, Device::Cpu).unwrap();
            gn_cpu.prefill(&prompt);
            let cpu = gn_cpu.generate_cached(steps, SampleOpts::greedy()).unwrap();

            let mut wm_metal = synthetic_weights(&cfg);
            let mut gn_metal =
                Llama32Generator::from_loader(cfg.clone(), &mut wm_metal, Device::Metal).unwrap();
            gn_metal.prefill(&prompt);
            let metal = gn_metal
                .generate_cached(steps, SampleOpts::greedy())
                .unwrap();

            eprintln!("{label}: cpu={cpu:?} metal={metal:?}");
            assert_eq!(metal, cpu, "{label} diverged");
        }
    }

    #[test]
    #[cfg(feature = "metal")]
    fn cached_matches_naive_on_greedy_metal_llama3ish() {
        if !rlx_runtime::is_available(Device::Metal) {
            eprintln!("skip: Metal unavailable");
            return;
        }
        let cfg = llama3ish_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5, 8, 13];
        let steps = 4;

        let mut wm_cpu = synthetic_weights(&cfg);
        let mut gn_cpu =
            Llama32Generator::from_loader(cfg.clone(), &mut wm_cpu, Device::Cpu).unwrap();
        gn_cpu.prefill(&prompt);
        let cpu_tokens = gn_cpu.generate_cached(steps, SampleOpts::greedy()).unwrap();

        let mut wm_metal = synthetic_weights(&cfg);
        let mut gn_metal =
            Llama32Generator::from_loader(cfg.clone(), &mut wm_metal, Device::Metal).unwrap();
        gn_metal.prefill(&prompt);
        let metal_tokens = gn_metal
            .generate_cached(steps, SampleOpts::greedy())
            .unwrap();

        assert_eq!(
            metal_tokens, cpu_tokens,
            "Metal llama3-ish GQA cached decode diverged from CPU"
        );
    }

    #[test]
    #[cfg(feature = "metal")]
    fn cached_matches_naive_on_greedy_metal() {
        if !rlx_runtime::is_available(Device::Metal) {
            eprintln!("skip: Metal unavailable");
            return;
        }
        let cfg = tiny_cfg();
        let prompt: Vec<u32> = vec![1, 2, 3, 5];
        let steps = 4;

        let mut wm_cpu = synthetic_weights(&cfg);
        let mut gn_cpu =
            Llama32Generator::from_loader(cfg.clone(), &mut wm_cpu, Device::Cpu).unwrap();
        gn_cpu.prefill(&prompt);
        let cpu_tokens = gn_cpu.generate_cached(steps, SampleOpts::greedy()).unwrap();

        let mut wm_metal = synthetic_weights(&cfg);
        let mut gn_metal =
            Llama32Generator::from_loader(cfg.clone(), &mut wm_metal, Device::Metal).unwrap();
        gn_metal.prefill(&prompt);
        let metal_tokens = gn_metal
            .generate_cached(steps, SampleOpts::greedy())
            .unwrap();

        assert_eq!(
            metal_tokens, cpu_tokens,
            "Metal cached decode diverged from CPU — KV or Lq!=Lk bug"
        );
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
