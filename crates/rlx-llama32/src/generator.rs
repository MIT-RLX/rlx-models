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
    build_llama32_decode_graph_sized_packed, build_llama32_decode_graph_sized_packed_ext,
    build_llama32_decode_hir_dynamic_ext, build_llama32_decode_hir_sized_ext,
    build_llama32_graph_sized, build_llama32_graph_sized_kv_tap,
    build_llama32_graph_sized_last_logits, build_llama32_graph_sized_packed,
    build_llama32_prefill_hir_dynamic_ext, build_llama32_prefill_hir_sized_ext, gather_embed_row,
    gather_embed_rows,
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
    llama_decode_bucket_resident_bytes, llama_decode_oneshot_compile_peak_bytes,
    trim_accelerator_arena_pool, would_exceed_soft_budget,
};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

fn gguf_has_block_quant_matmul(path: &Path) -> bool {
    use rlx_gguf::{GgmlType, GgufFile};
    match GgufFile::from_path(path) {
        Ok(raw) => raw.tensors.values().any(|t| {
            matches!(
                t.dtype,
                GgmlType::Q2K
                    | GgmlType::Q3K
                    | GgmlType::Q4K
                    | GgmlType::Q5K
                    | GgmlType::Q6K
                    | GgmlType::Q8K
                    | GgmlType::Q4_0
                    | GgmlType::Q4_1
                    | GgmlType::Q5_0
                    | GgmlType::Q5_1
                    | GgmlType::Q8_0
            )
        }),
        Err(_) => false,
    }
}

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

/// Device runs packed GGUF prefill graphs on-accelerator (Metal / CUDA / ROCm / MLX).
fn uses_packed_gguf_gpu_prefill(device: Device, mode: MetalGgufPrefillMode) -> bool {
    matches!(
        device,
        Device::Metal | Device::Cuda | Device::Rocm | Device::Mlx
    ) && mode.use_packed_gguf()
}

/// Whether a GGUF checkpoint must use the packed (`builder.rs`) graph path
/// instead of the F32 `rlx-flow` path.
///
/// Two cases:
/// * **Phi 3/4** — fused `attn_qkv` / `gate_up` tensors that the split-Q/K/V
///   F32 flow can't load.
/// * **Granite** — the `embedding` / `residual` / `attention` / `logit` scalar
///   multipliers are only applied by the packed builder
///   ([`build_llama32_graph_sized_packed`] / the packed decode builder); the
///   F32 `rlx-flow` Llama blocks don't carry them, so a Granite checkpoint on
///   the F32 path would silently drop the multipliers and emit garbage. Forcing
///   the packed path (works on CPU too via `Op::DequantMatMul`) keeps Granite
///   correct on every device.
fn fused_phi_gguf_needs_packed_paths(cfg: &Llama32Config, is_gguf: bool) -> bool {
    is_gguf && (cfg.is_phi_arch() || cfg.has_granite_scalars() || cfg.needs_arch_packed_builder())
}

/// Packed GGUF decode: skip in-graph vocab matmul; host argmax on tied Q4 embed.
fn llama32_host_greedy_lm_enabled(
    cfg: &Llama32Config,
    exec_device: Device,
    weights_deferred: bool,
    is_gguf: bool,
) -> bool {
    if !is_gguf || !weights_deferred {
        return false;
    }
    if rlx_ir::env::flag("RLX_LLAMA32_GRAPH_LM_HEAD") {
        return false;
    }
    if !(cfg.tie_word_embeddings || cfg.is_phi_arch()) {
        return false;
    }
    matches!(
        exec_device,
        Device::Cpu | Device::Ane | Device::Metal | Device::Cuda | Device::Rocm
    )
}

/// CUDA / ROCm: GPU-packed KV + CPU F32 logits (packed lm_head diverges on device).
/// Opt out with `ORPHEUS_CUDA_PACKED_KV=0`.
fn cuda_packed_kv_cpu_logits_prefill_enabled() -> bool {
    !matches!(
        std::env::var("ORPHEUS_CUDA_PACKED_KV").ok().as_deref(),
        Some("0") | Some("false") | Some("FALSE")
    )
}

/// Experimental: seed KV from the device packed graph. Opt in with
/// `ORPHEUS_CUDA_GPU_KV=1` (hybrid: device KV + CPU logits only).
fn cuda_gpu_kv_prefill_enabled() -> bool {
    std::env::var("ORPHEUS_CUDA_GPU_KV").ok().as_deref() == Some("1")
}

/// Force slow CPU F32 GGUF prefill on CUDA/ROCm (parity baseline).
fn cuda_f32_prefill_forced() -> bool {
    std::env::var("ORPHEUS_CUDA_F32_PREFILL").ok().as_deref() == Some("1")
        || std::env::var("ORPHEUS_CUDA_PREFILL").ok().as_deref() == Some("cpu")
}

/// CUDA / ROCm resident decode: keep KV on device between steps and bulk-flush
/// only when rebinding a wider bucket. Opt out with `ORPHEUS_CUDA_LAZY_KV=0`.
fn cuda_lazy_kv_enabled(device: Device) -> bool {
    if !matches!(device, Device::Cuda | Device::Rocm) {
        return false;
    }
    !matches!(
        std::env::var("ORPHEUS_CUDA_LAZY_KV").ok().as_deref(),
        Some("0") | Some("false") | Some("FALSE")
    )
}

/// CUDA / ROCm bucket rollover: defer evicting the outgoing bucket until after
/// the wider bucket is compiled and resident K/V is rebound. Parity-safe today
/// (flush + host-cache H2D bind, same as the default path). Opt in with
/// `ORPHEUS_CUDA_KV_DEVICE_REBIND=1`. Pure D2D seed helpers live in `rlx-cuda`
/// (`copy_resident_kv_rows_from`) for a future fast path.
fn cuda_device_kv_rebind_enabled(device: Device) -> bool {
    cuda_lazy_kv_enabled(device)
        && std::env::var("ORPHEUS_CUDA_KV_DEVICE_REBIND")
            .ok()
            .as_deref()
            == Some("1")
        && !matches!(
            std::env::var("ORPHEUS_CUDA_KV_HOST_REBIND").ok().as_deref(),
            Some("1") | Some("true") | Some("TRUE")
        )
}

/// Metadata for pulling missing K/V rows off the outgoing decode bucket before
/// rebinding a wider bucket.
struct ResidentBucketFlushPlan {
    prev_key: u64,
    bucket_start: usize,
    flush_upper: usize,
}

fn resident_bucket_flush_plan(
    cache_dec: &BucketedCompileCache,
    past_seq: usize,
) -> ResidentBucketFlushPlan {
    let prev_key = past_seq.saturating_sub(1) as u64;
    let prev_bucket_idx = cache_dec.bucket_for(prev_key).unwrap_or(0);
    let bucket_start = cache_dec
        .buckets()
        .nth(prev_bucket_idx)
        .map(|r| r.start as usize)
        .unwrap_or(0);
    let flush_upper = cache_dec
        .buckets()
        .nth(prev_bucket_idx)
        .map(|r| r.end - 1)
        .unwrap_or(0) as usize;
    ResidentBucketFlushPlan {
        prev_key,
        bucket_start,
        flush_upper,
    }
}

fn maybe_flush_resident_kv_before_bucket(
    compiled: &mut CompiledGraph,
    cache: &mut KvCacheState,
    plan: &ResidentBucketFlushPlan,
    prefix_tokens: usize,
    kv_dim: usize,
    n_layers: usize,
) -> Result<()> {
    flush_missing_resident_kv_to_cache(
        compiled,
        cache,
        prefix_tokens,
        plan.bucket_start,
        plan.flush_upper,
        kv_dim,
        n_layers,
    )
}

/// Pad host K/V to bucket width and bind as resident GPU handles (one-time per bucket).
fn bind_resident_kv_from_host_cache(
    compiled: &mut CompiledGraph,
    cache: &KvCacheState,
    upper: usize,
    kv_dim: usize,
    n_layers: usize,
) {
    for i in 0..n_layers {
        let kc = &cache.layers_k[i];
        let vc = &cache.layers_v[i];
        let mut kp = vec![0f32; upper * kv_dim];
        let mut vp = vec![0f32; upper * kv_dim];
        let nk = kc.len().min(kp.len());
        let nv = vc.len().min(vp.len());
        kp[..nk].copy_from_slice(&kc[..nk]);
        vp[..nv].copy_from_slice(&vc[..nv]);
        let k_name = format!("past_k_{i}");
        let v_name = format!("past_v_{i}");
        compiled.bind_gpu_handle(&k_name, &kp);
        compiled.bind_gpu_handle(&v_name, &vp);
        compiled.register_kv_row_feed(&k_name, 1 + 2 * i);
        compiled.register_kv_row_feed(&v_name, 2 + 2 * i);
    }
}

/// Host-side prefill inputs kept separate from compiled graphs so `run()`
/// can borrow feed buffers and `CompiledGraph` mutably at the same time.
struct PackedGgufPrefillFeed {
    upper_seq: usize,
    hidden: usize,
    ids_f32: Vec<f32>,
    last_idx: [f32; 1],
    embed_lazy: Option<(Vec<u8>, rlx_ir::quant::QuantScheme)>,
    embed_scratch: Vec<f32>,
    /// One-shot multimodal splice: `(start_row, [rows * hidden] f32)`. Applied
    /// over the host-gathered text embeds in `fill_embed_inputs`, so vision soft
    /// tokens replace their placeholder rows in `input_embeddings` — the packed
    /// (K-quant) prefill analog of the F32 `inputs_embeds` path. Consumed by the
    /// generator each prefill (set from `pending_embed_override`).
    embed_override: Option<(usize, Vec<f32>)>,
}

impl PackedGgufPrefillFeed {
    fn new(
        upper_seq: usize,
        hidden: usize,
        embed_lazy: Option<(Vec<u8>, rlx_ir::quant::QuantScheme)>,
    ) -> Self {
        Self {
            upper_seq,
            hidden,
            ids_f32: vec![0f32; upper_seq],
            last_idx: [0f32; 1],
            embed_lazy,
            embed_scratch: vec![0f32; upper_seq * hidden],
            embed_override: None,
        }
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

    fn fill_embed_inputs(&mut self, n: usize) -> Result<()> {
        if let Some((bytes, scheme)) = &self.embed_lazy {
            gather_embed_rows(
                bytes,
                *scheme,
                self.hidden,
                &self.ids_f32[..n],
                &mut self.embed_scratch[..n * self.hidden],
            )?;
        }
        // Multimodal splice: overwrite the placeholder rows with vision soft
        // tokens. Requires embed mode (`input_embeddings`); a GGUF whose embed
        // table is stored uncompressed (F16/F32) takes the `input_ids` graph
        // where `embed_scratch` is unused, so a splice would be silently lost —
        // hence the explicit guard.
        if let Some((start, data)) = &self.embed_override {
            anyhow::ensure!(
                self.embed_lazy.is_some(),
                "multimodal embed splice needs a block-quantized embed table \
                 (input_embeddings mode); this GGUF's `token_embd` is not packed"
            );
            let rows = data.len() / self.hidden;
            anyhow::ensure!(
                data.len() % self.hidden == 0 && start + rows <= n,
                "embed splice [{start}..{}] (hidden {}) exceeds prefill len {n}",
                start + rows,
                self.hidden
            );
            let off = start * self.hidden;
            self.embed_scratch[off..off + data.len()].copy_from_slice(data);
        }
        Ok(())
    }

    fn token_input(&self) -> (&str, &[f32]) {
        if self.embed_lazy.is_some() {
            ("input_embeddings", self.embed_scratch.as_slice())
        } else {
            ("input_ids", self.ids_f32.as_slice())
        }
    }

    fn kv_run_inputs(&self) -> [(&str, &[f32]); 1] {
        [self.token_input()]
    }

    fn logits_run_inputs(&self) -> [(&str, &[f32]); 2] {
        [
            self.token_input(),
            ("last_token_idx", self.last_idx.as_slice()),
        ]
    }
}

/// Packed GGUF prefill: separate logits and KV graphs (Metal logits+KV in one
/// graph diverges like F32 `LlamaKvTap`). When `host_greedy_prefill`, one graph
/// emits last-token hidden + KV (no vocab matmul).
struct PackedGgufPrefill {
    feed: PackedGgufPrefillFeed,
    logits: Option<CompiledGraph>,
    kv: CompiledGraph,
    exec_device: Device,
    kv_dim: usize,
    n_layers: usize,
    host_greedy_prefill: bool,
    /// Deferred logits compile (CUDA / ROCm — one graph at a time in VRAM).
    logits_plan: Option<(Llama32Config, PathBuf, CompileProfile)>,
}

impl PackedGgufPrefill {
    fn compile_logits(&mut self) -> Result<()> {
        if self.host_greedy_prefill {
            return Ok(());
        }
        if self.logits.is_some() {
            return Ok(());
        }
        let (cfg, path, profile) = self
            .logits_plan
            .as_ref()
            .context("packed logits compile plan missing")?;
        let path_str = path.to_str().context("non-utf8 weights path")?;
        let mut loader = GgufLoader::from_file(path_str)?;
        let mut packed = HashMap::new();
        let mut embed_host = None;
        let (logits_graph, params) = build_llama32_graph_sized_packed(
            cfg,
            &mut loader,
            1,
            self.feed.upper_seq,
            true,
            true,
            false,
            &mut packed,
            &mut embed_host,
        )?;
        let opts = compile_options_for_packed_gguf_prefill_with_profile(profile, self.exec_device);
        let mut logits = packed_gguf_compile_guard(self.exec_device, || {
            Session::new(self.exec_device).compile_with(logits_graph, &opts)
        });
        attach_f32_params(&mut logits, params);
        upload_packed_borrowed(&mut logits, &packed, &loader)?;
        if self.feed.embed_lazy.is_none() {
            if let Some(host) = embed_host {
                self.feed.embed_lazy = Some(host);
                self.feed
                    .embed_scratch
                    .resize(self.feed.upper_seq * self.feed.hidden, 0.0);
            }
        }
        self.logits = Some(logits);
        Ok(())
    }

    fn build_hidden_with_kv(
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
        let mut embed_host = None;
        let (graph, params) = build_llama32_graph_sized_packed(
            cfg,
            &mut loader,
            1,
            upper_seq,
            false,
            true,
            true,
            &mut packed,
            &mut embed_host,
        )?;
        let opts = compile_options_for_packed_gguf_prefill_with_profile(profile, exec_device);
        let mut kv = packed_gguf_compile_guard(exec_device, || {
            Session::new(exec_device).compile_with(graph, &opts)
        });
        attach_f32_params(&mut kv, params);
        upload_packed_borrowed(&mut kv, &packed, &loader)?;
        Ok(Self {
            feed: PackedGgufPrefillFeed::new(upper_seq, cfg.hidden_size, embed_host),
            logits: None,
            kv,
            exec_device,
            kv_dim: cfg.kv_proj_dim(),
            n_layers: cfg.kv_layers(),
            host_greedy_prefill: true,
            logits_plan: None,
        })
    }

    fn build(
        cfg: &Llama32Config,
        path: &Path,
        upper_seq: usize,
        device: Device,
        profile: &CompileProfile,
    ) -> Result<Self> {
        let mut out = Self::build_kv_only(cfg, path, upper_seq, device, profile)?;
        if matches!(device, Device::Metal) {
            out.compile_logits()?;
        }
        Ok(out)
    }

    /// KV graph only — half the compile RAM of [`Self::build`] (16 GiB CUDA).
    fn build_kv_only(
        cfg: &Llama32Config,
        path: &Path,
        upper_seq: usize,
        device: Device,
        profile: &CompileProfile,
    ) -> Result<Self> {
        let exec_device = packed_gguf_execution_device(device);
        let path_str = path.to_str().context("non-utf8 weights path")?;
        let mut loader = GgufLoader::from_file(path_str)?;
        let mut packed_kv = HashMap::new();
        let mut embed_host = None;
        let (kv_graph, params_kv) = build_llama32_graph_sized_packed(
            cfg,
            &mut loader,
            1,
            upper_seq,
            false,
            false,
            true,
            &mut packed_kv,
            &mut embed_host,
        )?;
        let opts = compile_options_for_packed_gguf_prefill_with_profile(profile, exec_device);
        let mut kv = packed_gguf_compile_guard(exec_device, || {
            Session::new(exec_device).compile_with(kv_graph, &opts)
        });
        attach_f32_params(&mut kv, params_kv);
        upload_packed_borrowed(&mut kv, &packed_kv, &loader)?;
        Ok(Self {
            feed: PackedGgufPrefillFeed::new(upper_seq, cfg.hidden_size, embed_host),
            logits: None,
            kv,
            exec_device,
            kv_dim: cfg.kv_proj_dim(),
            n_layers: cfg.kv_layers(),
            host_greedy_prefill: false,
            logits_plan: Some((cfg.clone(), path.to_path_buf(), profile.clone())),
        })
    }

    fn fill_inputs(&mut self, prompt_len: usize, ids_f32: &[f32]) -> usize {
        self.feed.fill_inputs(prompt_len, ids_f32)
    }

    fn fill_embed_inputs(&mut self, n: usize) -> Result<()> {
        self.feed.fill_embed_inputs(n)
    }

    fn run_logits(&mut self, prompt_len: usize, ids_f32: &[f32]) -> Result<Vec<f32>> {
        self.compile_logits()?;
        let n = self.fill_inputs(prompt_len, ids_f32);
        self.fill_embed_inputs(n)?;
        let logits = self
            .logits
            .as_mut()
            .context("packed logits prefill not compiled")?;
        let run_inputs = self.feed.logits_run_inputs();
        let outputs = run_packed_prefill(
            logits,
            self.exec_device,
            n,
            self.feed.upper_seq,
            &run_inputs,
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
        self.fill_embed_inputs(n)?;
        let run_inputs = self.feed.kv_run_inputs();
        let kv_outputs = run_packed_prefill(
            &mut self.kv,
            self.exec_device,
            n,
            self.feed.upper_seq,
            &run_inputs,
        );
        let kv_seq = infer_prefill_kv_seq(&kv_outputs, 1, &[self.kv_dim], n, self.feed.upper_seq);
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

    fn run_hidden_with_kv(
        &mut self,
        prompt_len: usize,
        ids_f32: &[f32],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let n = self.fill_inputs(prompt_len, ids_f32);
        self.fill_embed_inputs(n)?;
        let run_inputs = self.feed.logits_run_inputs();
        let outputs = run_packed_prefill(
            &mut self.kv,
            self.exec_device,
            n,
            self.feed.upper_seq,
            &run_inputs,
        );
        if outputs.len() != 1 + 2 * self.n_layers {
            anyhow::bail!(
                "packed hidden prefill produced {} outputs, expected {}",
                outputs.len(),
                1 + 2 * self.n_layers
            );
        }
        let hidden = packed_prefill_last_hidden(&outputs[0], n, self.feed.hidden)?;
        let kv_seq = infer_prefill_kv_seq(&outputs[1..], 1, &[self.kv_dim], n, self.feed.upper_seq);
        let (mut layers_k, mut layers_v) =
            split_packed_kv_outputs(outputs[1..].to_vec(), 1, kv_seq, self.kv_dim, self.n_layers)?;
        if kv_seq > n {
            let keep = n * self.kv_dim;
            for i in 0..self.n_layers {
                layers_k[i].truncate(keep);
                layers_v[i].truncate(keep);
            }
        }
        Ok((hidden, layers_k, layers_v))
    }

    fn run(
        &mut self,
        prompt_len: usize,
        ids_f32: &[f32],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        if self.host_greedy_prefill {
            return self.run_hidden_with_kv(prompt_len, ids_f32);
        }
        let n = self.fill_inputs(prompt_len, ids_f32);
        let logits = self.run_logits(prompt_len, ids_f32)?;
        if matches!(self.exec_device, Device::Cuda | Device::Rocm) {
            self.logits = None;
            trim_accelerator_arena_pool(self.exec_device);
        }
        self.fill_embed_inputs(n)?;
        let run_inputs = self.feed.kv_run_inputs();
        let kv_outputs = run_packed_prefill(
            &mut self.kv,
            self.exec_device,
            n,
            self.feed.upper_seq,
            &run_inputs,
        );
        let kv_seq = infer_prefill_kv_seq(&kv_outputs, 1, &[self.kv_dim], n, self.feed.upper_seq);
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

/// Slice the last prompt position from a packed prefill hidden tensor. CUDA may
/// return either `[hidden]` (gathered) or `[seq, hidden]` (full sequence).
fn packed_prefill_last_hidden(raw: &[f32], prompt_len: usize, hidden: usize) -> Result<Vec<f32>> {
    if raw.len() == hidden {
        return Ok(raw.to_vec());
    }
    let n = prompt_len.max(1);
    let need = n * hidden;
    if raw.len() >= need {
        let start = (n - 1) * hidden;
        return Ok(raw[start..start + hidden].to_vec());
    }
    anyhow::bail!(
        "packed prefill hidden len {} (prompt_len={n}, hidden={hidden})",
        raw.len()
    );
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

fn push_packed_decode_token_input<'a>(
    cfg: &Llama32Config,
    input_tok: u32,
    lazy_embed: Option<(&[u8], rlx_ir::quant::QuantScheme)>,
    embed_scratch: &'a mut Vec<f32>,
    inputs: &mut Vec<(&str, &'a [f32])>,
    input_ids_f32: &'a [f32; 1],
) -> Result<()> {
    if let Some((bytes, scheme)) = lazy_embed {
        embed_scratch.resize(cfg.hidden_size, 0.0);
        gather_embed_row(
            bytes,
            scheme,
            cfg.hidden_size,
            input_tok as usize,
            embed_scratch,
        )?;
        inputs.push(("input_embeddings", embed_scratch.as_slice()));
    } else {
        inputs.push(("input_ids", input_ids_f32.as_slice()));
    }
    Ok(())
}

fn packed_graph_uses_lazy_embed(
    packed: &HashMap<String, (rlx_ir::quant::QuantScheme, Vec<usize>)>,
    params: &HashMap<String, Vec<f32>>,
) -> bool {
    packed.contains_key("model.embed_tokens.weight")
        && !params.contains_key("model.embed_tokens.weight")
}

/// Bucketed packed decode: prefer logits-only + row readback (Metal/Vulkan/CUDA).
/// Fall back to full K/V D2H when row readback is disabled or resident KV is off.
fn packed_bucketed_cuda_full_kv_readback(device: Device) -> bool {
    if !matches!(device, Device::Cuda | Device::Rocm) {
        return false;
    }
    std::env::var("ORPHEUS_RESIDENT_KV").ok().as_deref() == Some("0")
        || std::env::var("ORPHEUS_VULKAN_RESIDENT_KV").ok().as_deref() == Some("0")
        || std::env::var("RLX_CUDA_FULL_KV_READBACK").ok().as_deref() == Some("1")
}

/// Bucketed packed decode: full K/V tensor readback instead of per-row
/// `read_output_row` (CPU and CUDA paths).
fn packed_bucketed_full_kv_readback(device: Device) -> bool {
    packed_bucketed_cuda_full_kv_readback(device)
        || matches!(device, Device::Cpu | Device::Mlx | Device::Ane)
}

/// Pull only the host-missing KV rows from the outgoing bucket's resident GPU
/// state before rebinding a wider bucket. In-bucket steps keep KV on device.
fn flush_missing_resident_kv_to_cache(
    compiled: &rlx_runtime::CompiledGraph,
    cache: &mut KvCacheState,
    prefix_tokens: usize,
    _bucket_start: usize,
    outgoing_upper: usize,
    kv_dim: usize,
    n_layers: usize,
) -> Result<()> {
    let host_rows = cache
        .layers_k
        .first()
        .map(|k| k.len() / kv_dim.max(1))
        .unwrap_or(0);
    if host_rows >= prefix_tokens {
        return Ok(());
    }
    let top_global = outgoing_upper;
    for g in host_rows..prefix_tokens {
        let from_output = g == top_global;
        for i in 0..n_layers {
            let nk = if from_output {
                compiled
                    .read_output_row(1 + 2 * i, outgoing_upper, kv_dim)
                    .with_context(|| format!("resident flush K output row layer {i}"))?
            } else {
                compiled
                    .read_gpu_handle_row(&format!("past_k_{i}"), g, kv_dim)
                    .with_context(|| format!("resident flush K handle row layer {i} row {g}"))?
            };
            let nv = if from_output {
                compiled
                    .read_output_row(2 + 2 * i, outgoing_upper, kv_dim)
                    .with_context(|| format!("resident flush V output row layer {i}"))?
            } else {
                compiled
                    .read_gpu_handle_row(&format!("past_v_{i}"), g, kv_dim)
                    .with_context(|| format!("resident flush V handle row layer {i} row {g}"))?
            };
            cache.layers_k[i].extend_from_slice(&nk);
            cache.layers_v[i].extend_from_slice(&nv);
        }
    }
    Ok(())
}

/// Append one new K/V row per layer from bucketed packed decode outputs.
fn append_packed_decode_kv_rows(
    compiled: &CompiledGraph,
    cache: &KvCacheState,
    upper: usize,
    kv_dim: usize,
    n_layers: usize,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let mut new_k = Vec::with_capacity(n_layers);
    let mut new_v = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let row_k = compiled
            .read_output_row(1 + 2 * i, upper, kv_dim)
            .with_context(|| format!("decode K row read layer {i}"))?;
        let row_v = compiled
            .read_output_row(2 + 2 * i, upper, kv_dim)
            .with_context(|| format!("decode V row read layer {i}"))?;
        let mut k_out = cache.layers_k[i].clone();
        let mut v_out = cache.layers_v[i].clone();
        k_out.extend_from_slice(&row_k);
        v_out.extend_from_slice(&row_v);
        new_k.push(k_out);
        new_v.push(v_out);
    }
    Ok((new_k, new_v))
}

/// Metal packed-GGUF decode: thunk lowering (MPSGraph diverges on Q4 bucketed
/// decode) while keeping tier-1 / `fuse_decode_mlp` enabled.
fn metal_decode_compile_guard<R, F>(device: Device, gguf_parity: bool, decode: bool, f: F) -> R
where
    F: FnOnce() -> R,
{
    if device != Device::Metal || !decode || !gguf_parity {
        return f();
    }
    let save_mps = rlx_ir::env::var("RLX_DISABLE_MPSGRAPH");
    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");
    let out = f();
    match save_mps {
        Some(v) => rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", v),
        None => rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH"),
    }
    out
}

/// GGUF on Metal/CPU/CUDA packed: mmap weights and dequant per compile instead of
/// draining the full model to host F32 at load (~12 GiB on Orpheus 3B).
pub(crate) fn gguf_defers_f32_drain(
    device: Device,
    is_gguf: bool,
    use_packed_gguf: bool,
    prefill_mode: MetalGgufPrefillMode,
) -> bool {
    if !is_gguf {
        return false;
    }
    if matches!(device, Device::Cpu | Device::Mlx | Device::Ane) {
        return true;
    }
    // wgpu/Vulkan: mmap GGUF + CPU prefill/decode until GPU KV parity lands.
    if matches!(device, Device::Gpu | Device::Vulkan) {
        return true;
    }
    // CUDA / ROCm: defer the ~12 GiB F32 drain when running packed GGUF on device.
    if matches!(device, Device::Cuda | Device::Rocm) {
        return use_packed_gguf;
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

/// Dense HF safetensors (BF16→F32 on take): skip the eager host drain that
/// otherwise doubles RSS with the device copy (Nanbeige 3B ≈ 15 GiB × 2).
/// Override with `RLX_LLAMA32_DEFER_SAFETENSORS=0|1`.
pub(crate) fn safetensors_defers_f32_drain(device: Device, is_safetensors: bool) -> bool {
    if !is_safetensors {
        return false;
    }
    match rlx_ir::env::var("RLX_LLAMA32_DEFER_SAFETENSORS").as_deref() {
        Some("0") | Some("false") | Some("off") => false,
        Some("1") | Some("true") | Some("on") => true,
        _ => matches!(
            device,
            Device::Metal | Device::Mlx | Device::Cuda | Device::Rocm
        ),
    }
}

/// Upload F32 compile params then drop each host `Vec` immediately so peak RSS
/// stays near one model copy + the growing device arena (not params∪device).
fn attach_f32_params(compiled: &mut CompiledGraph, mut params: HashMap<String, Vec<f32>>) {
    for (name, data) in params.drain() {
        compiled.set_param(&name, &data);
    }
}

/// Upload packed K-quant weights **zero-copy**: borrow each tensor's bytes
/// straight from the (mmap'd) loader and hand them to the arena via
/// `set_param_typed`. The `packed` map now carries only `(scheme, shape)`
/// metadata, so the quantized bytes are never materialized into an owned
/// buffer nor cached a second time — the model isn't duplicated in RSS. The
/// loader must outlive this call (it does at every call site: a local
/// `GgufLoader` for prefill, `self.decode_weights_cache` for decode).
fn upload_packed_borrowed(
    compiled: &mut CompiledGraph,
    packed: &HashMap<String, (rlx_ir::quant::QuantScheme, Vec<usize>)>,
    loader: &dyn WeightLoader,
) -> Result<()> {
    for name in packed.keys() {
        let bytes = loader
            .tensor_bytes_borrowed(name)
            .with_context(|| format!("packed upload: bytes unavailable for {name}"))?;
        compiled.set_param_typed(name, bytes, rlx_ir::DType::U8);
    }
    Ok(())
}

/// Per-layer KV cache state for incremental decoding. Each `Vec<f32>`
/// is a flat `[batch, past_seq, kv_proj_dim]` tensor.
#[derive(Clone)]
struct KvCacheState {
    past_seq: usize,
    layers_k: Vec<Vec<f32>>,
    layers_v: Vec<Vec<f32>>,
}

/// Per-session padded K/V upload buffers (resized when bucket upper changes).
#[derive(Default)]
struct DecodeKvScratch {
    padded_k: Vec<Vec<f32>>,
    padded_v: Vec<Vec<f32>>,
}

impl DecodeKvScratch {
    fn ensure_bucket(&mut self, upper: usize, kv_dims: &[usize]) {
        let n = kv_dims.len();
        if self.padded_k.len() != n {
            self.padded_k.resize_with(n, Vec::new);
            self.padded_v.resize_with(n, Vec::new);
        }
        for (i, &kd) in kv_dims.iter().enumerate() {
            let need = upper * kd;
            if self.padded_k[i].len() != need {
                self.padded_k[i].resize(need, 0.0);
                self.padded_v[i].resize(need, 0.0);
            }
        }
    }
}

/// Borrow cached F32 tensors, open a fresh GGUF loader, or mmap safetensors
/// (deferred dense HF path — F32 only on `take`).
enum BuildWeightLoader<'a> {
    Cached(ArcCacheLoader<'a>),
    Gguf(GgufLoader),
    // Owned, file-backed loader from `load_from_path` — always `'static`, so it
    // must NOT borrow the enum's `'a` (that would force `'a: 'static` at every
    // trait-method dispatch below → E0521).
    Live(Box<dyn WeightLoader>),
}

impl WeightLoader for BuildWeightLoader<'_> {
    fn format_id(&self) -> &'static str {
        match self {
            Self::Cached(l) => l.format_id(),
            Self::Gguf(l) => l.format_id(),
            Self::Live(l) => l.format_id(),
        }
    }
    fn len(&self) -> usize {
        match self {
            Self::Cached(l) => l.len(),
            Self::Gguf(l) => l.len(),
            Self::Live(l) => l.len(),
        }
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        match self {
            Self::Cached(l) => l.take(key),
            Self::Gguf(l) => l.take(key),
            Self::Live(l) => l.take(key),
        }
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        match self {
            Self::Cached(l) => l.take_transposed(key),
            Self::Gguf(l) => l.take_transposed(key),
            Self::Live(l) => l.take_transposed(key),
        }
    }
    fn remaining_keys(&self) -> Vec<String> {
        match self {
            Self::Cached(l) => l.remaining_keys(),
            Self::Gguf(l) => l.remaining_keys(),
            Self::Live(l) => l.remaining_keys(),
        }
    }
    fn tensor_bytes_borrowed(&self, key: &str) -> Option<&[u8]> {
        match self {
            Self::Cached(l) => l.tensor_bytes_borrowed(key),
            Self::Gguf(l) => l.tensor_bytes_borrowed(key),
            Self::Live(l) => l.tensor_bytes_borrowed(key),
        }
    }
    fn packed_meta(&self, key: &str) -> Option<(rlx_ir::quant::QuantScheme, Vec<usize>)> {
        match self {
            Self::Cached(l) => l.packed_meta(key),
            Self::Gguf(l) => l.packed_meta(key),
            Self::Live(l) => l.packed_meta(key),
        }
    }
}

/// Memoizing GGUF loader for the packed decode path.
///
/// Bucketed/dynamic packed decode rebuilds the decode graph for each bucket
/// bound (and per token in the oneshot path), and every build `take`s the model
/// weights — re-reading and re-dequantizing the multi-GB GGUF each time
/// (`take` is destructive, so the old code opened a fresh `GgufLoader` per
/// build). This wrapper extracts each weight from the underlying loader exactly
/// once, caches it behind an `Arc`, and serves cheap clones thereafter — so the
/// GGUF is parsed/dequantized once per generator lifetime instead of once per
/// bucket × per utterance. Held on [`Llama32Generator`] and reused across all
/// packed decode builds.
struct CachedGgufWeights {
    inner: GgufLoader,
    f32_take: HashMap<String, (std::sync::Arc<Vec<f32>>, Vec<usize>)>,
    f32_take_t: HashMap<String, (std::sync::Arc<Vec<f32>>, Vec<usize>)>,
    packed: HashMap<
        String,
        (
            std::sync::Arc<Vec<u8>>,
            rlx_ir::quant::QuantScheme,
            Vec<usize>,
        ),
    >,
}

impl CachedGgufWeights {
    fn from_file(path: &str) -> Result<Self> {
        Ok(Self {
            inner: GgufLoader::from_file(path)?,
            f32_take: HashMap::new(),
            f32_take_t: HashMap::new(),
            packed: HashMap::new(),
        })
    }

    fn tied_packed_embed_bytes(&mut self) -> Result<(Vec<u8>, rlx_ir::quant::QuantScheme)> {
        const KEY: &str = "model.embed_tokens.weight";
        if let Some((b, sc, _)) = self.packed.get(KEY) {
            return Ok((b.to_vec(), *sc));
        }
        match self.inner.take_packed(KEY)? {
            Some((bytes, scheme, shape)) => {
                let arc = std::sync::Arc::new(bytes);
                self.packed
                    .insert(KEY.to_string(), (arc.clone(), scheme, shape));
                Ok(((*arc).clone(), scheme))
            }
            None => anyhow::bail!("host greedy lm_head: missing packed tied embed"),
        }
    }
}

impl WeightLoader for CachedGgufWeights {
    fn format_id(&self) -> &'static str {
        self.inner.format_id()
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn arch_hint(&self) -> Option<&str> {
        self.inner.arch_hint()
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.inner.remaining_keys()
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        if let Some((d, s)) = self.f32_take.get(key) {
            return Ok(((**d).clone(), s.clone()));
        }
        let (d, s) = self.inner.take(key)?;
        let arc = std::sync::Arc::new(d);
        self.f32_take
            .insert(key.to_string(), (arc.clone(), s.clone()));
        Ok(((*arc).clone(), s))
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        if let Some((d, s)) = self.f32_take_t.get(key) {
            return Ok(((**d).clone(), s.clone()));
        }
        let (d, s) = self.inner.take_transposed(key)?;
        let arc = std::sync::Arc::new(d);
        self.f32_take_t
            .insert(key.to_string(), (arc.clone(), s.clone()));
        Ok(((*arc).clone(), s))
    }
    fn take_packed(
        &mut self,
        key: &str,
    ) -> Result<Option<rlx_core::weight_map::PackedWeightTensor>> {
        if let Some((b, sc, sh)) = self.packed.get(key) {
            return Ok(Some(((**b).clone(), *sc, sh.clone())));
        }
        match self.inner.take_packed(key)? {
            Some((bytes, scheme, shape)) => {
                let arc = std::sync::Arc::new(bytes);
                self.packed
                    .insert(key.to_string(), (arc.clone(), scheme, shape.clone()));
                Ok(Some(((*arc).clone(), scheme, shape)))
            }
            None => Ok(None),
        }
    }

    fn tensor_bytes_borrowed(&self, key: &str) -> Option<&[u8]> {
        self.inner.tensor_bytes_borrowed(key)
    }
    fn packed_meta(&self, key: &str) -> Option<(rlx_ir::quant::QuantScheme, Vec<usize>)> {
        // Non-destructive: read straight from the inner GGUF header on every
        // (bucketed) rebuild. Bytes are served zero-copy via
        // `tensor_bytes_borrowed`, so we no longer cache packed bytes here.
        self.inner.packed_meta(key)
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
    /// Persists across utterances (compiled weights are model-constant)
    /// so a warm utterance reuses buckets instead of recompiling them —
    /// see [`Self::soft_release_decode_kv_bindings`].
    decode_loaded_buckets: HashSet<usize>,
    /// Decode buckets whose resident GPU K/V handles hold the *current*
    /// utterance's prefix (main resident path). Separate from
    /// `decode_loaded_buckets` because K/V is per-utterance while the
    /// compiled graph is not: cleared each `prefill` to force a fresh
    /// [`bind_resident_kv_from_host_cache`] without discarding the graph.
    decode_resident_bound: HashSet<usize>,
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
    /// Memoizing GGUF loader for packed decode builds — parses/dequantizes the
    /// GGUF once and serves cached clones, so per-bucket / per-token graph
    /// rebuilds don't re-read the multi-GB file. Lazily created on first use.
    decode_weights_cache: Option<CachedGgufWeights>,
    /// Lazy packed GGUF prefill session (Metal default).
    packed_gguf_prefill: Option<PackedGgufPrefill>,
    /// Metal GGUF prefill strategy ([`MetalGgufPrefillMode::Auto`] reads env).
    metal_gguf_prefill_mode: MetalGgufPrefillMode,
    /// GGUF block-quant (Q4/Q8/K) vs dense F16/F32 — packed decode is quant-only.
    gguf_block_quant: bool,
    /// CPU packed greedy: host tied-lm_head argmax (skip in-graph vocab matmul).
    host_greedy_lm: bool,
    /// Hidden-only decode graphs for [`Self::host_greedy_lm`].
    decode_compile_cache_hidden: Option<BucketedCompileCache>,
    decode_loaded_buckets_hidden: HashSet<usize>,
    /// Hidden decode buckets with resident GPU K/V handles bound.
    decode_resident_hidden_bound: HashSet<usize>,
    decode_kv_scratch: DecodeKvScratch,
    decode_embed_scratch: Vec<f32>,
    /// Pending one-shot multimodal embed splice for the next packed prefill.
    /// Set via [`Self::set_multimodal_embed_override`]; moved into the packed
    /// prefill feed (and thereby consumed) at each prefill run site.
    pending_embed_override: Option<(usize, Vec<f32>)>,
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
            decode_resident_bound: HashSet::new(),
            compile_seq_cap: None,
            inv_freq,
            prefill_profile: CompileProfile::llama32_prefill(),
            decode_profile: CompileProfile::llama32_decode(),
            sample_step: 0,
            weights_path: None,
            weights_deferred: false,
            decode_weights_cache: None,
            packed_gguf_prefill: None,
            metal_gguf_prefill_mode: MetalGgufPrefillMode::Auto,
            gguf_block_quant: false,
            host_greedy_lm: false,
            decode_compile_cache_hidden: None,
            decode_loaded_buckets_hidden: HashSet::new(),
            decode_resident_hidden_bound: HashSet::new(),
            decode_kv_scratch: DecodeKvScratch::default(),
            decode_embed_scratch: Vec::new(),
            pending_embed_override: None,
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
            decode_resident_bound: HashSet::new(),
            compile_seq_cap: None,
            inv_freq,
            prefill_profile: CompileProfile::llama32_prefill(),
            decode_profile: CompileProfile::llama32_decode(),
            sample_step: 0,
            weights_path: None,
            weights_deferred: true,
            decode_weights_cache: None,
            packed_gguf_prefill: None,
            metal_gguf_prefill_mode: prefill_mode,
            gguf_block_quant: false,
            host_greedy_lm: false,
            decode_compile_cache_hidden: None,
            decode_loaded_buckets_hidden: HashSet::new(),
            decode_resident_hidden_bound: HashSet::new(),
            decode_kv_scratch: DecodeKvScratch::default(),
            decode_embed_scratch: Vec::new(),
            pending_embed_override: None,
        })
    }

    fn build_weight_loader_from<'a>(
        deferred: bool,
        path: &'a Option<PathBuf>,
        cache: &'a HashMap<String, ArcF32Tensor>,
    ) -> Result<BuildWeightLoader<'a>> {
        if deferred {
            let path = path
                .as_ref()
                .context("deferred weights need weights path")?;
            let path_str = path.to_str().context("non-utf8 weights path")?;
            let is_gguf = path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
            if is_gguf {
                Ok(BuildWeightLoader::Gguf(GgufLoader::from_file(path_str)?))
            } else {
                Ok(BuildWeightLoader::Live(
                    rlx_core::weight_loader::load_from_path(path_str)?,
                ))
            }
        } else {
            Ok(BuildWeightLoader::Cached(ArcCacheLoader::new(cache)))
        }
    }

    fn build_weight_loader(&self) -> Result<BuildWeightLoader<'_>> {
        Self::build_weight_loader_from(
            self.weights_deferred,
            &self.weights_path,
            &self.weights_cache,
        )
    }

    fn ensure_weights(&mut self) -> Result<()> {
        if !self.weights_deferred {
            return Ok(());
        }
        self.weights_path
            .as_ref()
            .context("deferred weights need weights path")?;
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
        attach_f32_params(&mut compiled, params);
        Ok(compiled.run(&[("input_ids", ids_f32)]))
    }

    fn packed_prefill_upper_seq(&self, prompt_len: usize) -> usize {
        let cap = self.compile_seq_cap();
        let n = prompt_len.max(1);
        if n >= cap {
            return cap;
        }
        // CUDA / ROCm lack active-extent on DequantMatMul + attention, so a
        // power-of-two bucket runs padded zero rows and corrupts logits/KV.
        if matches!(self.device, Device::Cuda | Device::Rocm) {
            return n;
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
            .is_none_or(|p| p.feed.upper_seq != upper);
        if !need {
            return Ok(());
        }
        let path = self
            .weights_path
            .as_ref()
            .context("packed prefill needs gguf path")?;
        self.packed_gguf_prefill = Some(if self.host_greedy_lm {
            PackedGgufPrefill::build_hidden_with_kv(
                &self.cfg,
                path,
                upper,
                self.device,
                &self.prefill_profile,
            )?
        } else {
            PackedGgufPrefill::build(&self.cfg, path, upper, self.device, &self.prefill_profile)?
        });
        Ok(())
    }

    fn ensure_packed_kv_prefill(&mut self, seq: usize) -> Result<()> {
        let upper = self.packed_prefill_upper_seq(seq);
        let need = self
            .packed_gguf_prefill
            .as_ref()
            .is_none_or(|p| p.feed.upper_seq != upper || p.logits.is_some());
        if !need {
            return Ok(());
        }
        let path = self
            .weights_path
            .as_ref()
            .context("packed kv prefill needs gguf path")?;
        self.packed_gguf_prefill = Some(PackedGgufPrefill::build_kv_only(
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
        if !uses_packed_gguf_gpu_prefill(self.device, self.metal_gguf_prefill_mode()) {
            return Ok(None);
        }
        if matches!(self.device, Device::Cuda | Device::Rocm)
            && cuda_packed_kv_cpu_logits_prefill_enabled()
            && !cuda_gpu_kv_prefill_enabled()
        {
            return Ok(None);
        }
        if self.weights_path.is_none() {
            return Ok(None);
        }
        self.ensure_packed_prefill(prompt_len)?;
        let p = self.packed_gguf_prefill.as_mut().unwrap();
        Ok(Some(p.run_logits(prompt_len, ids_f32)?))
    }

    #[allow(dead_code)]
    fn try_packed_gguf_kv_only(
        &mut self,
        prompt_len: usize,
        ids_f32: &[f32],
    ) -> Result<Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)>> {
        if !uses_packed_gguf_gpu_prefill(self.device, self.metal_gguf_prefill_mode()) {
            return Ok(None);
        }
        if self.weights_path.is_none() {
            return Ok(None);
        }
        self.ensure_packed_prefill(prompt_len)?;
        let p = self.packed_gguf_prefill.as_mut().unwrap();
        Ok(Some(p.run_kv_only(prompt_len, ids_f32)?))
    }

    /// CPU F32 last-position hidden (reference). Used by CUDA host-greedy native
    /// prefill when device packed hidden is not parity-safe.
    fn run_prefill_last_hidden_cpu_f32(
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
        let mut loader = self.build_weight_loader()?;
        let (graph, params) = build_llama32_graph_sized(
            &self.cfg,
            &mut loader,
            batch,
            seq,
            /*with_lm_head*/ false,
            /*with_kv_outputs*/ false,
        )?;
        let session = Session::new(Device::Cpu);
        let mut compiled = session.compile_with(graph, &prefill_opts);
        attach_f32_params(&mut compiled, params);
        let outputs = compiled.run(&[("input_ids", ids_f32)]);
        let hidden = outputs
            .into_iter()
            .next()
            .context("cpu f32 hidden prefill returned no outputs")?;
        packed_prefill_last_hidden(&hidden, seq, self.cfg.hidden_size)
    }

    /// CUDA / ROCm host-greedy prefill with CPU F32 reference hidden + KV.
    ///
    /// Device packed prefill hidden/KV still diverge on Orpheus-scale GGUF; this
    /// path keeps mmap'd weights (no ~12 GiB host F32 drain) while matching CPU
    /// greedy parity. Used when `MetalGgufPrefillMode::PackedGguf` is selected
    /// (`ORPHEUS_CUDA_NATIVE_PREFILL=1` in TTS).
    fn seed_cuda_host_greedy_reference_prefill(
        &mut self,
        seq: usize,
        ids_f32: &[f32],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let hidden = self.run_prefill_last_hidden_cpu_f32(1, seq, ids_f32)?;
        let outputs = self.run_prefill_with_cache_cpu_f32(1, seq, ids_f32)?;
        let (_, layers_k, layers_v) = self.split_prefill_outputs(outputs, 1, seq)?;
        Ok((hidden, layers_k, layers_v))
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
        attach_f32_params(&mut compiled, params);
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
        if self.packed_gguf_prefill.take().is_some() {
            trim_accelerator_arena_pool(self.device);
        }
    }

    /// Packed GPU KV + CPU F32 logits — seeds device KV for native CUDA/ROCm
    /// decode without compiling the packed logits graph (16 GiB-safe).
    fn seed_prefill_packed_kv_cpu_logits(
        &mut self,
        batch: usize,
        seq: usize,
        ids_f32: &[f32],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        self.ensure_packed_kv_prefill(seq)?;
        let (layers_k, layers_v) = self
            .packed_gguf_prefill
            .as_mut()
            .context("packed kv prefill")?
            .run_kv_only(seq, ids_f32)?;
        self.drop_packed_gguf_prefill();
        let logits = self.run_prefill_last_logits_cpu_f32(batch, seq, ids_f32)?;
        Ok((logits, layers_k, layers_v))
    }

    /// Full GPU packed GGUF prefill on Metal: a single packed `DequantMatMul`
    /// forward yields the last-token logits **and** per-layer KV directly on the
    /// device, so the ~12 GiB CPU F32 model dequant is never materialized.
    ///
    /// Only if the packed lm_head emits non-finite logits (historically a risk
    /// on the 156k-vocab Orpheus head) do we recompute the last-position logits
    /// on CPU F32 — the GPU-packed KV stays valid for decode seeding.
    fn seed_prefill_packed_gpu(
        &mut self,
        batch: usize,
        seq: usize,
        ids_f32: &[f32],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        self.ensure_packed_prefill(seq)?;
        let ov = self.pending_embed_override.take();
        let p = self.packed_gguf_prefill.as_mut().unwrap();
        p.feed.embed_override = ov;
        let (logits, layers_k, layers_v) = p.run(seq, ids_f32)?;
        if logits.iter().all(|x| x.is_finite()) {
            eprintln!(
                "[llama32] packed GGUF prefill on {:?} (logits+KV on device, no host F32 dequant)",
                self.device
            );
            return Ok((logits, layers_k, layers_v));
        }
        eprintln!(
            "[llama32] packed GGUF prefill lm_head non-finite on {:?}; \
             recomputing last-position logits on CPU F32 (KV stays on device)",
            self.device
        );
        self.drop_packed_gguf_prefill();
        let cpu_logits = self.run_prefill_last_logits_cpu_f32(batch, seq, ids_f32)?;
        Ok((cpu_logits, layers_k, layers_v))
    }

    fn prefill_seed_triple(
        &mut self,
        batch: usize,
        seq: usize,
        ids_f32: &[f32],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        if uses_packed_gguf_gpu_prefill(self.device, self.metal_gguf_prefill_mode()) {
            // Hybrid packed KV + CPU decode diverges after step 1; CPU decode needs
            // CPU F32 KV from the full prefill-with-cache graph.
            if self.decode_device() == Device::Cpu {
                self.ensure_weights()?;
                let outputs = self.run_prefill_with_cache_cpu_f32(batch, seq, ids_f32)?;
                return self.split_prefill_outputs(outputs, batch, seq);
            }
            // CUDA / ROCm native decode: CPU F32 reference KV unless forced otherwise.
            // Host-greedy uses [`Self::seed_cuda_host_greedy_reference_prefill`];
            // non-greedy may use packed GPU logits+KV (`seed_prefill_packed_gpu`).
            if matches!(self.device, Device::Cuda | Device::Rocm)
                && self.decode_device() == self.device
                && cuda_packed_kv_cpu_logits_prefill_enabled()
            {
                if cuda_gpu_kv_prefill_enabled() {
                    eprintln!(
                        "[llama32] packed GPU KV + CPU logits prefill on {:?} (ORPHEUS_CUDA_GPU_KV=1)",
                        self.device
                    );
                    let triple = self.seed_prefill_packed_kv_cpu_logits(batch, seq, ids_f32)?;
                    if std::env::var("ORPHEUS_PREFILL_PERSIST").ok().as_deref() != Some("1") {
                        self.drop_packed_gguf_prefill();
                    }
                    return Ok(triple);
                }
                if !cuda_f32_prefill_forced()
                    && std::env::var("ORPHEUS_CUDA_PACKED_KV_PREFILL")
                        .ok()
                        .as_deref()
                        == Some("1")
                {
                    eprintln!(
                        "[llama32] packed GPU KV + CPU logits prefill on {:?}",
                        self.device
                    );
                    let triple = self.seed_prefill_packed_kv_cpu_logits(batch, seq, ids_f32)?;
                    if std::env::var("ORPHEUS_PREFILL_PERSIST").ok().as_deref() != Some("1") {
                        self.drop_packed_gguf_prefill();
                    }
                    return Ok(triple);
                }
                if !cuda_f32_prefill_forced()
                    && std::env::var("ORPHEUS_CUDA_PACKED_PREFILL").ok().as_deref() != Some("0")
                {
                    if self.host_greedy_lm {
                        let triple = self.seed_cuda_host_greedy_reference_prefill(seq, ids_f32)?;
                        if std::env::var("ORPHEUS_PREFILL_PERSIST").ok().as_deref() != Some("1") {
                            self.drop_packed_gguf_prefill();
                        }
                        return Ok(triple);
                    }
                    let triple = self.seed_prefill_packed_gpu(batch, seq, ids_f32)?;
                    if std::env::var("ORPHEUS_PREFILL_PERSIST").ok().as_deref() != Some("1") {
                        self.drop_packed_gguf_prefill();
                    }
                    return Ok(triple);
                }
                eprintln!(
                    "[llama32] CPU F32 prefill-with-cache on {:?} (reference KV for native decode)",
                    self.device
                );
                self.ensure_weights()?;
                let outputs = self.run_prefill_with_cache_cpu_f32(batch, seq, ids_f32)?;
                return self.split_prefill_outputs(outputs, batch, seq);
            }
            // Native decode (Metal): run the whole prefill forward on the GPU via
            // the packed Q4 graph — logits and KV both — instead of dequantizing
            // the full 3B model to ~12 GiB host F32.
            let triple = self.seed_prefill_packed_gpu(batch, seq, ids_f32)?;
            // Drop the packed prefill graph before decode unless explicitly retained
            // (`ORPHEUS_PREFILL_PERSIST=1`). Keeping both prefill + decode compiled
            // graphs resident peaks past ~50 GiB on Orpheus 3B Q4 Metal.
            if std::env::var("ORPHEUS_PREFILL_PERSIST").ok().as_deref() != Some("1") {
                self.drop_packed_gguf_prefill();
            }
            return Ok(triple);
        }
        // CUDA/ROCm: same hybrid rule as Metal — never compile device decode graphs
        // when `decode_device` fell back to CPU (16 GiB consumer GPUs OOM otherwise).
        if matches!(self.device, Device::Cuda | Device::Rocm) && self.decode_device() == Device::Cpu
        {
            self.ensure_weights()?;
            let outputs = self.run_prefill_with_cache_cpu_f32(batch, seq, ids_f32)?;
            return self.split_prefill_outputs(outputs, batch, seq);
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
        if !uses_packed_gguf_gpu_prefill(self.device, self.metal_gguf_prefill_mode())
            && !self.uses_packed_gguf_cpu_prefill()
            && !self.uses_packed_gguf_mlx_prefill()
        {
            return Ok(None);
        }
        if self.weights_path.is_none() {
            return Ok(None);
        }
        self.ensure_packed_prefill(prompt_len)?;
        let ov = self.pending_embed_override.take();
        let p = self.packed_gguf_prefill.as_mut().unwrap();
        p.feed.embed_override = ov;
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

    /// Register a one-shot multimodal embed splice for the **next** prefill:
    /// `embeds` is `[rows * hidden]` f32 that overwrites sequence positions
    /// `start .. start + rows` of `input_embeddings` (the vision soft tokens),
    /// leaving the surrounding text tokens gathered from the packed embed table.
    ///
    /// This is the packed (K-quant, on-device) analog of the F32 `inputs_embeds`
    /// prefill: it keeps the LM weights packed (no ~4× F32 dequant) so a 24B VL
    /// model fits in unified memory. Only effective on the packed GGUF prefill
    /// path (Metal / MLX / CUDA / ROCm with a block-quantized `token_embd`);
    /// `fill_embed_inputs` errors if the embed table is not packed.
    pub fn set_multimodal_embed_override(&mut self, start: usize, embeds: Vec<f32>) {
        self.pending_embed_override = Some((start, embeds));
    }

    /// Drop any pending multimodal splice that a prefill has not consumed.
    /// Callers should invoke this after a `generate` that may have failed before
    /// the packed prefill ran, so a stale splice can never leak into a later
    /// (possibly text-only) generation on the same runner.
    pub fn clear_multimodal_embed_override(&mut self) {
        self.pending_embed_override = None;
    }

    /// True while a multimodal splice is still pending — i.e. no packed prefill
    /// has consumed it. After a `generate`/`prefill`, a lingering `true` means
    /// the packed GGUF path was **not** taken (e.g. CPU F32 fallback), so the
    /// vision tokens were dropped; callers should treat that as an error.
    pub fn multimodal_override_pending(&self) -> bool {
        self.pending_embed_override.is_some()
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
        let is_safetensors = !is_gguf
            && (weights_path.is_dir()
                || weights_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("safetensors")));
        let defer_f32_drain = gguf_defers_f32_drain(
            device,
            is_gguf,
            prefill_mode.use_packed_gguf(),
            prefill_mode,
        ) || safetensors_defers_f32_drain(device, is_safetensors);
        let mut g = if defer_f32_drain {
            Self::from_loader_deferred(cfg, loader, device, prefill_mode)?
        } else {
            let mut g = Self::from_loader(cfg, loader, device)?;
            g.metal_gguf_prefill_mode = prefill_mode;
            g
        };
        g.prefill_profile = crate::llama32_profile_near_weights(weights_path, false);
        g.decode_profile = crate::llama32_profile_near_weights(weights_path, true);
        // Always retain the path for deferred mmap reloads (GGUF or safetensors).
        g.weights_path = Some(weights_path.to_path_buf());
        if is_gguf {
            g.gguf_block_quant = gguf_has_block_quant_matmul(weights_path);
            let exec = packed_gguf_execution_device(device);
            g.host_greedy_lm =
                llama32_host_greedy_lm_enabled(&g.cfg, exec, g.weights_deferred, is_gguf);
            if g.host_greedy_lm {
                eprintln!(
                    "[llama32-runner] greedy decode: host tied-lm_head argmax (skip in-graph vocab matmul)"
                );
            }
        } else if g.weights_deferred {
            eprintln!(
                "[llama32-runner] safetensors mmap-on-take (no eager F32 drain) on {device:?}"
            );
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
            && !fused_phi_gguf_needs_packed_paths(&self.cfg, self.is_gguf_checkpoint())
    }

    fn uses_packed_gguf_cpu_prefill(&self) -> bool {
        self.prefill_device() == Device::Cpu
            && self.weights_deferred
            && fused_phi_gguf_needs_packed_paths(&self.cfg, self.is_gguf_checkpoint())
    }

    /// Phi fused GGUF on MLX when packed GPU prefill is off (CpuF32 mode).
    fn uses_packed_gguf_mlx_prefill(&self) -> bool {
        self.device == Device::Mlx
            && self.weights_deferred
            && fused_phi_gguf_needs_packed_paths(&self.cfg, self.is_gguf_checkpoint())
    }

    /// GGUF KV decode on CPU when GPU prefill/decode would diverge (Metal CPU
    /// prefill, wgpu/Vulkan until portable GPU KV parity lands).
    fn gguf_cpu_decode_required(&self) -> bool {
        if !self.is_gguf_checkpoint() {
            return false;
        }
        match self.device {
            // Native Metal decode only when prefill also ran on Metal (packed/F32).
            // CPU F32 prefill seeds KV that diverges if decode stays on Metal
            // (Orpheus 3B Q4_K_M / Q8_0 — see `decode_device` below).
            Device::Metal => {
                if std::env::var("ORPHEUS_METAL_NATIVE_DECODE").ok().as_deref() == Some("1") {
                    false
                } else {
                    self.prefill_device() == Device::Cpu
                }
            }
            Device::Ane => self.prefill_device() == Device::Cpu,
            // CUDA / ROCm: native packed decode on by default (m=1 uses host GGUF
            // dequant in rlx-cuda for parity). Opt out with ORPHEUS_CUDA_NATIVE_DECODE=0.
            Device::Cuda => {
                std::env::var("ORPHEUS_CUDA_NATIVE_DECODE").ok().as_deref() == Some("0")
            }
            Device::Rocm => {
                std::env::var("ORPHEUS_ROCM_NATIVE_DECODE").ok().as_deref() == Some("0")
            }
            // wgpu (`Device::Gpu`): RoPE-GptJ + native GGUF Q4 GEMV kernels are
            // implemented and parity-verified, but native decode is blocked by a
            // pre-existing wgpu >4 GB arena windowing bug (the ~10 GB Orpheus
            // arena: F32 embed + tied-lm_head copy + 4x-inflated U8 weights).
            // Decode produces the correct first token then diverges. Default to
            // the CPU host path; opt into native wgpu decode (for testing the
            // kernels / once the arena fix lands) via `ORPHEUS_WGPU_NATIVE=1`.
            Device::Gpu => std::env::var("ORPHEUS_WGPU_NATIVE").ok().as_deref() != Some("1"),
            // Vulkan (MoltenVK on Apple, native Vulkan on Linux): RoPE-GptJ
            // (`rope.comp` style 1) + native GGUF Q4_K/Q6_K decode GEMV
            // (`dequant_matmul.comp`, `m == 1`) kernels are implemented; prefill
            // stays on CPU F32 (see `prefill_device`). Native on-device decode is
            // opt-in until broadly validated — set `ORPHEUS_VULKAN_NATIVE=1`.
            Device::Vulkan => std::env::var("ORPHEUS_VULKAN_NATIVE").ok().as_deref() != Some("1"),
            // MLX decodes GGUF Llama natively: its RoPE kernel now honors the
            // interleaved/GPT-J flavor that GGUF weights need (see rlx-mlx
            // `Op::Rope` lowering), so no CPU host fallback is required.
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

    /// Native-quantized decode gate: run the packed (`Op::DequantMatMul`)
    /// decode graph instead of dequantizing the full model to F32. Enabled for
    /// GGUF checkpoints decoding on CPU, MLX, Metal, wgpu (`Device::Gpu`), or
    /// Vulkan — all have native DequantMatMul kernels.
    fn use_packed_decode(&self) -> bool {
        if !self.is_gguf_checkpoint() || self.weights_path.is_none() {
            return false;
        }
        if !self.gguf_block_quant {
            return false;
        }
        // A/B escape hatch for benchmarking Q4 vs F32 decode on the same binary.
        if std::env::var("ORPHEUS_NO_PACKED").ok().as_deref() == Some("1") {
            return false;
        }
        match self.decode_device() {
            // GPU Q4 decode is memory-bandwidth-bound and faster than F32 here
            // (Metal measured ~1.9x vs CPU); MLX uses its native quantized matmul.
            // `Device::Gpu`/`Device::Vulkan` only reach here when `ORPHEUS_WGPU_NATIVE`/
            // `ORPHEUS_VULKAN_NATIVE=1` made `gguf_cpu_decode_required` return false
            // (else decode_device is Cpu and the F32 host path is used).
            Device::Metal
            | Device::Mlx
            | Device::Gpu
            | Device::Vulkan
            | Device::Cuda
            | Device::Rocm => true,
            // CPU: F32 via Accelerate AMX is faster than on-the-fly Q4 dequant
            // (measured 104s vs 127s/100 tok), so default to the F32 flow path.
            // Opt into packed Q4 only when memory-constrained (avoids the ~12 GB
            // F32 model footprint) via `ORPHEUS_PACKED_DECODE=1`, or when Phi
            // fused GGUF layout cannot use the split-projection F32 flow.
            Device::Cpu => {
                std::env::var("ORPHEUS_PACKED_DECODE").ok().as_deref() == Some("1")
                    || (self.weights_deferred
                        && fused_phi_gguf_needs_packed_paths(&self.cfg, self.is_gguf_checkpoint()))
            }
            _ => false,
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

    /// Bail if compiling another decode bucket would exceed the soft RAM
    /// budget. Shared gate for the packed-decode compile sites.
    fn ensure_decode_bucket_budget(&self) -> Result<()> {
        if !self.decode_bucket_allowed() {
            anyhow::bail!(
                "decode bucket compile would exceed soft RAM budget (~80% of physical RAM); \
                 set RLX_SOFT_MEMORY_FRACTION or ORPHEUS_DECODE_CACHE_CAP lower"
            );
        }
        Ok(())
    }

    /// Whether the bucketed compile cache can serve `past_seq` (used by both
    /// the F32 and packed-GGUF decode dispatch to pick the compile-once
    /// bucketed path over the slow per-token oneshot path).
    fn bucket_decode_eligible(&self, past_seq: usize) -> bool {
        self.decode_compile_cache.is_some()
            && self.decode_bucket_allowed()
            && self
                .decode_compile_cache
                .as_ref()
                .unwrap()
                .bucket_for(past_seq as u64)
                .is_some()
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
        if self.host_greedy_lm {
            self.decode_compile_cache_hidden = Some(BucketedCompileCache::power_of_two_ladder(
                packed_gguf_execution_device(self.decode_device()),
                /*min*/ 1,
                max_past.max(1) as u64,
            ));
            self.decode_loaded_buckets_hidden.clear();
            self.decode_resident_hidden_bound.clear();
        }
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
    /// model — the next [`Self::step`] call processes the full sequence.
    /// Clears the KV cache from a prior generation. By default it **keeps**
    /// the compiled decode-bucket graphs resident across utterances (dropping
    /// only the per-utterance K/V bindings) so a warm utterance skips the
    /// recompile + weight upload — see [`Self::soft_release_decode_kv_bindings`].
    /// The soft RAM budget trims buckets on constrained hardware; opt out of
    /// reuse entirely with `ORPHEUS_KEEP_DECODE_GRAPHS=0` (full release, the old
    /// behavior). CUDA/ROCm D2D-K/V modes always take the full release.
    pub fn prefill(&mut self, prompt_ids: &[u32]) {
        self.tokens.clear();
        self.tokens.extend_from_slice(prompt_ids);
        self.cache = None;
        self.sample_step = 0;
        // Default: keep compiled decode-bucket graphs resident across
        // utterances (drop only the per-utterance K/V bindings) so a warm
        // utterance skips the multi-second recompile + weight upload. The
        // soft memory budget trims buckets on constrained hardware. Opt out
        // to the old full release with `ORPHEUS_KEEP_DECODE_GRAPHS=0`, and
        // CUDA/ROCm D2D K/V modes also take the full release.
        if self.cross_utterance_decode_reuse_enabled() {
            self.soft_release_decode_kv_bindings();
        } else {
            self.release_decode_graphs();
        }
    }

    /// Drop decode compile graphs between utterances (keep prefill cache).
    pub fn release_decode_graphs(&mut self) {
        self.decode_loaded_buckets.clear();
        self.decode_resident_bound.clear();
        self.decode_loaded_buckets_hidden.clear();
        self.decode_resident_hidden_bound.clear();
        if let Some(cache) = self.decode_compile_cache.as_mut() {
            cache.clear_compiled();
        }
        if let Some(cache) = self.decode_compile_cache_hidden.as_mut() {
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

    /// Between-utterance release that KEEPS the compiled decode-bucket graphs
    /// (and their uploaded weights) resident so the next utterance reuses
    /// them — the expensive per-bucket recompile + multi-GB weight upload
    /// then happens only on the first utterance. Only the per-utterance K/V
    /// *bindings* are dropped (`decode_resident_bound`), forcing a fresh
    /// [`bind_resident_kv_from_host_cache`] from the new prompt's host cache.
    ///
    /// RAM-adaptive: buckets are trimmed to the soft memory budget
    /// (highest-index first) so a memory-constrained machine does not
    /// accumulate the whole ladder across utterances, while a roomy machine
    /// keeps it all (zero cross-utterance recompiles). See
    /// [`llama_decode_bucket_resident_bytes`] and `RLX_SOFT_MEMORY_FRACTION`.
    ///
    /// The host-greedy `*_hidden` decode path is not wired for
    /// cross-utterance reuse, so its graphs are fully released here (its
    /// original between-utterance behavior); it is unused when the resident
    /// sampling path is active (the Orpheus default).
    pub fn soft_release_decode_kv_bindings(&mut self) {
        // K/V is per-utterance; force a fresh bind next utterance while the
        // compiled graphs (and set params) stay put.
        self.decode_resident_bound.clear();
        self.trim_decode_buckets_to_budget(usize::MAX);

        // Host-greedy hidden path: preserve full-release behavior.
        self.decode_loaded_buckets_hidden.clear();
        self.decode_resident_hidden_bound.clear();
        if let Some(cache) = self.decode_compile_cache_hidden.as_mut() {
            cache.clear_compiled();
        }
        if let Some(cache) = self.decode_dynamic_cache.as_mut() {
            cache.clear();
        }
    }

    /// Whether compiled decode-bucket graphs may be kept resident across
    /// utterances (skip recompile + weight re-upload on warm utterances).
    /// Only the plain resident path is wired for this; CUDA/ROCm D2D K/V
    /// rebind modes keep their existing per-utterance release + single
    /// resident bucket. Opt out entirely with `ORPHEUS_KEEP_DECODE_GRAPHS=0`.
    fn cross_utterance_decode_reuse_enabled(&self) -> bool {
        if std::env::var("ORPHEUS_KEEP_DECODE_GRAPHS").ok().as_deref() == Some("0") {
            return false;
        }
        // Scope to the packed (Q4) decode paths that were audited for this:
        // the resident and per-step-K/V bucketed paths. Non-packed (F32)
        // bucketed decode keeps its original per-utterance release.
        if !self.use_packed_decode() {
            return false;
        }
        let exec = packed_gguf_execution_device(self.decode_device());
        !(cuda_device_kv_rebind_enabled(exec) || cuda_lazy_kv_enabled(exec))
    }

    /// Evict resident decode buckets (highest index first, never `protect`)
    /// while keeping one more would exceed the soft memory budget. Keeps the
    /// `decode_loaded_buckets`/`decode_resident_bound` bookkeeping in sync so
    /// an evicted bucket is recompiled + re-bound on next use. No-op on a
    /// machine with headroom for the whole ladder.
    fn trim_decode_buckets_to_budget(&mut self, protect: usize) {
        let resident_bytes = llama_decode_bucket_resident_bytes();
        loop {
            let victim = {
                let Some(cache) = self.decode_compile_cache.as_mut() else {
                    return;
                };
                if cache.compiled_count() == 0 || !would_exceed_soft_budget(resident_bytes) {
                    return;
                }
                cache.evict_one_except(protect)
            };
            match victim {
                Some(v) => {
                    self.decode_loaded_buckets.remove(&v);
                    self.decode_resident_bound.remove(&v);
                }
                None => return,
            }
        }
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

        if self.host_greedy_lm_active(opts) {
            let tok = self.decode_step_greedy_host(past_seq, input_tok, &mut adjust_logits)?;
            self.tokens.push(tok);
            return Ok(tok);
        }

        // GPU-resident KV decode (Vulkan): K/V live in the device arena across
        // steps, fed in-place from the decode output — no per-step host K/V
        // upload/readback, logits-only readback. Returns logits only; the host
        // `cache.layers_k/v` is left stale (synced back only on bucket change).
        let resident = self.resident_kv_decode_enabled() && self.bucket_decode_eligible(past_seq);

        let (mut logits, new_k, new_v) = if resident {
            (
                self.decode_step_bucketed_packed_resident(past_seq, input_tok)?,
                Vec::new(),
                Vec::new(),
            )
        } else if self.use_packed_decode() {
            // Packed (Q4) decode: reuse one compiled bucket graph across tokens
            // when a bucket cache is available; else the slow per-token rebuild.
            if self.bucket_decode_eligible(past_seq) {
                self.decode_step_bucketed_packed(past_seq, input_tok)?
            } else {
                self.decode_step_packed(past_seq, input_tok)?
            }
        } else if self.decode_dynamic_cache.is_some() {
            self.decode_step_dynamic(past_seq, input_tok)?
        } else if self.bucket_decode_eligible(past_seq) {
            self.decode_step_bucketed(past_seq, input_tok)?
        } else {
            self.decode_oneshot_allowed()?;
            self.decode_step_oneshot(past_seq, input_tok)?
        };

        let cache_mut = self.cache.as_mut().unwrap();
        cache_mut.past_seq = past_seq + 1;
        if !resident {
            // Resident path keeps K/V in the device arena; host cache untouched.
            cache_mut.layers_k = new_k;
            cache_mut.layers_v = new_v;
        }

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
        attach_f32_params(&mut compiled, params);

        let position = decode_position_input(past_seq);
        let input_ids_f32 = [input_tok as f32];
        let mask_len = past_seq + 1;
        let mask = vec![1.0f32; mask_len];
        let n_layers = self.cfg.kv_layers();
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

    /// Native-quantized decode step. Builds a packed
    /// (`Op::DequantMatMul`) decode graph for the exact `past_seq` from
    /// the GGUF K-quant bytes — no full-model F32 drain — compiles it on
    /// the decode device, attaches F32 params (`set_param`) + packed U8
    /// params (`set_param_typed`), and runs it with the same input/output
    /// contract as the F32 decode steps (`input_ids` + per-layer
    /// `past_k`/`past_v`; outputs `logits` + full new KV). The KV-cache
    /// update in the caller is therefore unchanged.
    #[allow(clippy::type_complexity)]
    fn decode_step_packed(
        &mut self,
        past_seq: usize,
        input_tok: u32,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let path_str = {
            let path = self
                .weights_path
                .as_ref()
                .context("packed decode needs gguf path")?;
            path.to_str().context("non-utf8 weights path")?.to_string()
        };
        if self.decode_weights_cache.is_none() {
            self.decode_weights_cache = Some(CachedGgufWeights::from_file(&path_str)?);
        }
        let cfg = self.cfg.clone();
        let mut packed: HashMap<String, (rlx_ir::quant::QuantScheme, Vec<usize>)> = HashMap::new();
        let (graph, params) = build_llama32_decode_graph_sized_packed(
            &cfg,
            self.decode_weights_cache.as_mut().unwrap(),
            /*batch*/ 1,
            past_seq,
            /*use_custom_mask*/ false,
            &mut packed,
        )?;

        let exec_device = packed_gguf_execution_device(self.decode_device());
        let opts =
            compile_options_for_packed_gguf_prefill_with_profile(&self.decode_profile, exec_device);
        trim_accelerator_arena_pool(exec_device);
        let mut compiled = packed_gguf_compile_guard(exec_device, || {
            Session::new(exec_device).compile_with(graph, &opts)
        });
        let lazy_embed = packed_graph_uses_lazy_embed(&packed, &params);
        attach_f32_params(&mut compiled, params);
        upload_packed_borrowed(
            &mut compiled,
            &packed,
            self.decode_weights_cache.as_ref().unwrap(),
        )?;

        let cache = self.cache.as_ref().context("packed decode without cache")?;
        let input_ids_f32 = [input_tok as f32];
        let n_layers = self.cfg.kv_layers();
        let key_strs: Vec<String> = (0..n_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        let mut inputs: Vec<(&str, &[f32])> = Vec::with_capacity(1 + 2 * n_layers);
        let mut embed_scratch = Vec::new();
        push_packed_decode_token_input(
            &self.cfg,
            input_tok,
            if lazy_embed {
                // Metadata from the map; embed bytes borrowed zero-copy from the
                // live decode loader (mmap) for this token's host-side gather.
                let scheme = packed
                    .get("model.embed_tokens.weight")
                    .map(|(s, _)| *s)
                    .expect("lazy decode embed meta");
                let bytes = self
                    .decode_weights_cache
                    .as_ref()
                    .unwrap()
                    .tensor_bytes_borrowed("model.embed_tokens.weight")
                    .expect("lazy decode embed bytes");
                Some((bytes, scheme))
            } else {
                None
            },
            &mut embed_scratch,
            &mut inputs,
            &input_ids_f32,
        )?;
        for i in 0..n_layers {
            inputs.push((&key_strs[2 * i], cache.layers_k[i].as_slice()));
            inputs.push((&key_strs[2 * i + 1], cache.layers_v[i].as_slice()));
        }
        let outputs = compiled.run(&inputs);
        self.split_decode_outputs(outputs)
    }

    /// Native-quantized decode using the **bucketed** compile cache.
    /// Same fast-path idea as [`decode_step_bucketed`] (compile one packed
    /// graph per bucket, pad `past_k`/`past_v` to the bucket upper bound,
    /// drive a custom keep-mask), but the graph keeps K-quant matmuls in the
    /// arena via `Op::DequantMatMul`. cos/sin rows are fed as runtime inputs
    /// so a single compiled graph serves every position in its bucket — no
    /// per-token GGUF reload / rebuild / recompile (the slow path that
    /// [`decode_step_packed`] takes). Numerically identical to
    /// `decode_step_packed`: same inv_freq-derived RoPE row, same mask
    /// semantics, same packed weights.
    #[allow(clippy::type_complexity)]
    fn decode_step_bucketed_packed(
        &mut self,
        past_seq: usize,
        input_tok: u32,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let path = self
            .weights_path
            .as_ref()
            .context("packed decode needs gguf path")?;
        let path_str = path.to_str().context("non-utf8 weights path")?.to_string();

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
        let n_layers = self.cfg.kv_layers();
        let exec_device = packed_gguf_execution_device(self.decode_device());

        // First-time-in-bucket: build the packed graph for `upper`, compile it
        // through the bucketed cache, attach F32 + U8 packed params once. Later
        // calls in the same bucket skip all of this and just `.run()`.
        let needs_load = !self.decode_loaded_buckets.contains(&bucket_idx);
        if needs_load {
            trim_accelerator_arena_pool(exec_device);
            // K/V is re-uploaded per step on this path, so keeping other
            // buckets compiled across utterances is safe (no bind coupling).
            if self.cross_utterance_decode_reuse_enabled() {
                self.trim_decode_buckets_to_budget(bucket_idx);
                self.ensure_decode_bucket_budget()?;
            } else {
                self.ensure_decode_bucket_budget()?;
                if let Some(cache_mut) = self.decode_compile_cache.as_mut() {
                    cache_mut.evict_except(bucket_idx);
                }
                self.decode_loaded_buckets.clear();
            }

            if self.decode_weights_cache.is_none() {
                self.decode_weights_cache = Some(CachedGgufWeights::from_file(&path_str)?);
            }
            let cfg = self.cfg.clone();
            let mut packed: HashMap<String, (rlx_ir::quant::QuantScheme, Vec<usize>)> =
                HashMap::new();
            let (graph, params) = build_llama32_decode_graph_sized_packed(
                &cfg,
                self.decode_weights_cache.as_mut().unwrap(),
                /*batch*/ 1,
                upper,
                /*use_custom_mask*/ true,
                &mut packed,
            )?;

            let opts = compile_options_for_packed_gguf_prefill_with_profile(
                &self.decode_profile,
                exec_device,
            );
            let cache_mut = self.decode_compile_cache.as_mut().unwrap();
            let loader_ref: &dyn WeightLoader = self.decode_weights_cache.as_ref().unwrap();
            packed_gguf_compile_guard(exec_device, || {
                let (_u, compiled) = cache_mut
                    .get_or_compile_with_options(past_seq as u64, |_upper| graph, &opts)
                    .expect("bucket must exist; we just looked it up");
                attach_f32_params(compiled, params);
                upload_packed_borrowed(compiled, &packed, loader_ref)
                    .expect("packed decode: zero-copy weight upload");
            });
            self.decode_loaded_buckets.insert(bucket_idx);
        }

        // Per-token host inputs. cos/sin come from the same inv_freq the baked
        // builder uses, so this is bit-for-bit the oneshot packed RoPE row.
        let (cos_row, sin_row) = rope_slice(&self.inv_freq, past_seq);
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
        let mut embed_scratch = Vec::new();
        let lazy_embed = self.decode_weights_cache.as_ref().and_then(|cache| {
            // Host-side lazy gather iff the embed is stored K-quant (packed)
            // and was NOT dequantized to F32. Bytes are borrowed zero-copy from
            // the loader's mmap — no longer sourced from a resident byte cache.
            let key = "model.embed_tokens.weight";
            if cache.f32_take.contains_key(key) {
                return None;
            }
            let (scheme, _) = cache.packed_meta(key)?;
            let bytes = cache.tensor_bytes_borrowed(key)?;
            Some((bytes, scheme))
        });
        push_packed_decode_token_input(
            &self.cfg,
            input_tok,
            lazy_embed,
            &mut embed_scratch,
            &mut inputs,
            &input_ids_f32,
        )?;
        inputs.push(("cos", cos_row.as_slice()));
        inputs.push(("sin", sin_row.as_slice()));
        inputs.push(("mask", mask.as_slice()));
        for i in 0..n_layers {
            inputs.push((&key_strs[2 * i], padded_k[i].as_slice()));
            inputs.push((&key_strs[2 * i + 1], padded_v[i].as_slice()));
        }

        let cache_mut = self.decode_compile_cache.as_mut().unwrap();
        let compiled = cache_mut
            .compiled_for_key_mut(past_seq as u64)
            .expect("bucket was just loaded above");
        // CUDA / ROCm: per-row `read_output_row` after logits-only readback is
        // unreliable today (bounds / stream visibility). Full K/V D2H matches the
        // F32 bucketed path and is correct; row-only readback stays on Metal/Vulkan.
        if packed_bucketed_full_kv_readback(exec_device) {
            let raw_outputs = compiled.run(&inputs);
            let mut iter = raw_outputs.into_iter();
            let logits = iter
                .next()
                .context("bucketed packed decode logits missing")?;
            let past_len = past_seq + 1;
            let mut new_k = Vec::with_capacity(n_layers);
            let mut new_v = Vec::with_capacity(n_layers);
            for _ in 0..n_layers {
                let k = iter.next().context("bucketed packed k missing")?;
                let v = iter.next().context("bucketed packed v missing")?;
                new_k.push(compact_bucketed_kv_buffer(&k, past_len, kv_dim, 1));
                new_v.push(compact_bucketed_kv_buffer(&v, past_len, kv_dim, 1));
            }
            return Ok((logits, new_k, new_v));
        }
        let cache_ref = self.cache.as_ref().context("packed decode without cache")?;
        let logits = compiled
            .run_read_outputs(&inputs, Some(&[0]))
            .into_iter()
            .next()
            .context("bucketed packed decode logits missing")?;
        let (new_k, new_v) =
            append_packed_decode_kv_rows(compiled, cache_ref, upper, kv_dim, n_layers)?;
        Ok((logits, new_k, new_v))
    }

    fn host_greedy_prefill_eligible(&self, opts: SampleOpts) -> bool {
        self.host_greedy_lm && opts.greedy && opts.is_classic()
    }

    /// Host-greedy decode (packed Q4 graph + CPU tied-lm_head argmax).
    fn host_greedy_lm_active(&self, opts: SampleOpts) -> bool {
        self.host_greedy_prefill_eligible(opts) && self.use_packed_decode()
    }

    fn ensure_decode_weights_cache(&mut self) -> Result<()> {
        if self.decode_weights_cache.is_some() {
            return Ok(());
        }
        let path = self
            .weights_path
            .as_ref()
            .context("packed decode needs gguf path")?;
        let path_str = path.to_str().context("non-utf8 weights path")?;
        self.decode_weights_cache = Some(CachedGgufWeights::from_file(path_str)?);
        Ok(())
    }

    /// Host-side tied-`lm_head` argmax over the decode `hidden`, applying the
    /// caller's logit adjustment (`adjust`) first. The full logit row is
    /// materialized via the same GGUF matmul the in-graph `lm_head` lowers to
    /// (`gguf_matmul_bt`) so `adjust` — e.g. Orpheus's SNAC-slot mask +
    /// repetition penalty — can gate tokens *before* the argmax. This is
    /// essential for correctness: a bare streaming argmax that skipped `adjust`
    /// picks masked / repeated tokens and derails structured (audio-codebook)
    /// decode.
    fn host_greedy_argmax(
        &mut self,
        hidden: &[f32],
        adjust: &mut dyn FnMut(&mut [f32]),
    ) -> Result<u32> {
        self.ensure_decode_weights_cache()?;
        let h = self.cfg.hidden_size;
        if hidden.len() < h {
            anyhow::bail!("decode hidden short: {} < {h}", hidden.len());
        }
        let vocab = self.cfg.vocab_size;
        let (bytes, scheme) = self
            .decode_weights_cache
            .as_mut()
            .unwrap()
            .tied_packed_embed_bytes()?;
        let mut logits = vec![0f32; vocab];
        rlx_cpu::gguf_matmul::gguf_matmul_bt_parallel(
            &hidden[..h],
            &bytes,
            &mut logits,
            1,
            h,
            vocab,
            scheme,
        );
        adjust(&mut logits);
        let mut best_idx = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &l) in logits.iter().enumerate() {
            if l > best_val {
                best_val = l;
                best_idx = i as u32;
            }
        }
        Ok(best_idx)
    }

    /// Host-greedy decode: run a hidden-only bucketed packed graph and pick the
    /// next token with a host-side tied-`lm_head` argmax ([`Self::host_greedy_argmax`]),
    /// skipping the in-graph vocab matmul. Delegates to
    /// [`Self::decode_step_greedy_resident`] when resident-KV is enabled.
    ///
    /// Uses the same bucket conventions as [`Self::decode_step_bucketed_packed`]:
    /// the binary keep-mask is [`bucket_decode_mask`] (real past `[0,past_seq)`
    /// plus the newly-rope'd key at bucket-top `upper`), and the new K/V row is
    /// gathered with [`compact_bucketed_kv_buffer`] (which takes the row at
    /// `upper`, not a naive truncate that would drop it).
    ///
    /// `adjust` is the caller's per-step logit adjustment (e.g. Orpheus's
    /// SNAC-slot mask + repetition penalty). It is applied to the full logit row
    /// inside [`Self::host_greedy_argmax`] before argmax — the sampling path
    /// applies the same closure, so host-greedy must too or it selects
    /// masked/repeated tokens and derails structured decode.
    fn decode_step_greedy_host(
        &mut self,
        past_seq: usize,
        input_tok: u32,
        adjust: &mut dyn FnMut(&mut [f32]),
    ) -> Result<u32> {
        if self.resident_kv_decode_enabled() && self.bucket_decode_eligible(past_seq) {
            return self.decode_step_greedy_resident(past_seq, input_tok, adjust);
        }
        let decode_cache = self
            .decode_compile_cache_hidden
            .as_mut()
            .context("host greedy lm_head requires decode_compile_cache_hidden")?;
        let bucket_idx = decode_cache
            .bucket_for(past_seq as u64)
            .ok_or_else(|| anyhow::anyhow!("past_seq {past_seq} outside any bucket"))?;
        let upper = decode_cache
            .buckets()
            .nth(bucket_idx)
            .map(|r| r.end - 1)
            .unwrap() as usize;

        let kv_dim = self.cfg.kv_proj_dim();
        let n_layers = self.cfg.kv_layers();
        let exec_device = packed_gguf_execution_device(self.decode_device());

        let needs_load = !self.decode_loaded_buckets_hidden.contains(&bucket_idx);
        if needs_load {
            trim_accelerator_arena_pool(exec_device);
            if !self.decode_bucket_allowed() {
                anyhow::bail!(
                    "decode bucket compile would exceed soft RAM budget (~80% of physical RAM); \
                     set RLX_SOFT_MEMORY_FRACTION or ORPHEUS_DECODE_CACHE_CAP lower"
                );
            }
            if let Some(cache_mut) = self.decode_compile_cache_hidden.as_mut() {
                cache_mut.evict_except(bucket_idx);
            }
            self.decode_loaded_buckets_hidden.clear();

            self.ensure_decode_weights_cache()?;
            let cfg = self.cfg.clone();
            let mut packed: HashMap<String, (rlx_ir::quant::QuantScheme, Vec<usize>)> =
                HashMap::new();
            let (graph, params) = build_llama32_decode_graph_sized_packed_ext(
                &cfg,
                self.decode_weights_cache.as_mut().unwrap(),
                /*batch*/ 1,
                upper,
                /*use_custom_mask*/ true,
                /*with_lm_head*/ false,
                &mut packed,
            )?;

            let opts = compile_options_for_packed_gguf_prefill_with_profile(
                &self.decode_profile,
                exec_device,
            );
            let cache_mut = self.decode_compile_cache_hidden.as_mut().unwrap();
            let loader_ref: &dyn WeightLoader = self.decode_weights_cache.as_ref().unwrap();
            packed_gguf_compile_guard(exec_device, || {
                let (_u, compiled) = cache_mut
                    .get_or_compile_with_options(past_seq as u64, |_upper| graph, &opts)
                    .expect("bucket must exist; we just looked it up");
                attach_f32_params(compiled, params);
                upload_packed_borrowed(compiled, &packed, loader_ref)
                    .expect("packed decode: zero-copy weight upload");
            });
            self.decode_loaded_buckets_hidden.insert(bucket_idx);
        }

        self.decode_kv_scratch
            .ensure_bucket(upper, &vec![kv_dim; n_layers]);

        let (cos_row, sin_row) = rope_slice(&self.inv_freq, past_seq);
        // Same binary keep-mask the graph expects: real past keys `[0, past_seq)`
        // plus the newly-rope'd key at bucket-top index `upper`. `fill_mask` here
        // instead kept `past_seq` (zero padding) and masked `upper` (the token's
        // own key) — inverted, which collapsed host-greedy decode to garbage.
        let mask = bucket_decode_mask(past_seq, upper);
        let input_ids_f32 = [input_tok as f32];
        let cache = self.cache.as_ref().context("packed decode without cache")?;
        let past_bytes = past_seq.saturating_mul(kv_dim);
        for i in 0..n_layers {
            let src_k = &cache.layers_k[i];
            let src_v = &cache.layers_v[i];
            let copy_k = past_bytes
                .min(src_k.len())
                .min(upper.saturating_mul(kv_dim));
            let copy_v = past_bytes
                .min(src_v.len())
                .min(upper.saturating_mul(kv_dim));
            self.decode_kv_scratch.padded_k[i].fill(0.0);
            self.decode_kv_scratch.padded_v[i].fill(0.0);
            self.decode_kv_scratch.padded_k[i][..copy_k].copy_from_slice(&src_k[..copy_k]);
            self.decode_kv_scratch.padded_v[i][..copy_v].copy_from_slice(&src_v[..copy_v]);
        }

        let key_strs: Vec<String> = (0..n_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        let mut inputs: Vec<(&str, &[f32])> = Vec::with_capacity(4 + 2 * n_layers);
        let lazy_embed = self.decode_weights_cache.as_ref().and_then(|c| {
            // Host-side lazy gather iff the embed is K-quant and not F32; bytes
            // borrowed zero-copy from the loader mmap (see the mirror in
            // `decode_step_packed_bucketed`).
            let key = "model.embed_tokens.weight";
            if c.f32_take.contains_key(key) {
                return None;
            }
            let (scheme, _) = c.packed_meta(key)?;
            let bytes = c.tensor_bytes_borrowed(key)?;
            Some((bytes, scheme))
        });
        push_packed_decode_token_input(
            &self.cfg,
            input_tok,
            lazy_embed,
            &mut self.decode_embed_scratch,
            &mut inputs,
            &input_ids_f32,
        )?;
        inputs.push(("cos", cos_row.as_slice()));
        inputs.push(("sin", sin_row.as_slice()));
        inputs.push(("mask", mask.as_slice()));
        for i in 0..n_layers {
            inputs.push((
                &key_strs[2 * i],
                self.decode_kv_scratch.padded_k[i].as_slice(),
            ));
            inputs.push((
                &key_strs[2 * i + 1],
                self.decode_kv_scratch.padded_v[i].as_slice(),
            ));
        }

        let cache_mut = self.decode_compile_cache_hidden.as_mut().unwrap();
        let compiled = cache_mut
            .compiled_for_key_mut(past_seq as u64)
            .expect("hidden decode bucket was just loaded");
        let past_len = past_seq + 1;
        let (hidden, new_k, new_v) = if packed_bucketed_full_kv_readback(exec_device) {
            // Bucket layout: new K/V is at the bucket-top row (`upper`), not row
            // `past_seq`. `compact_bucketed_kv_buffer` gathers real past
            // `[0, past_seq)` + that new row into dense `past_len` rows — the
            // same extraction `decode_step_bucketed_packed` uses. A naive
            // truncate-to-`past_len` would drop the new K and keep a zero-pad
            // row, collapsing decode to garbage.
            let raw = compiled.run(&inputs);
            let mut iter = raw.into_iter();
            let hidden = iter.next().context("bucketed greedy hidden missing")?;
            let mut new_k = Vec::with_capacity(n_layers);
            let mut new_v = Vec::with_capacity(n_layers);
            for _ in 0..n_layers {
                let k = iter.next().context("bucketed greedy k missing")?;
                let v = iter.next().context("bucketed greedy v missing")?;
                new_k.push(compact_bucketed_kv_buffer(&k, past_len, kv_dim, 1));
                new_v.push(compact_bucketed_kv_buffer(&v, past_len, kv_dim, 1));
            }
            (hidden, new_k, new_v)
        } else {
            let hidden = compiled
                .run_read_outputs(&inputs, Some(&[0]))
                .into_iter()
                .next()
                .context("bucketed packed decode hidden missing")?;
            let (new_k, new_v) =
                append_packed_decode_kv_rows(compiled, cache, upper, kv_dim, n_layers)?;
            (hidden, new_k, new_v)
        };

        let cache_mut = self.cache.as_mut().unwrap();
        cache_mut.past_seq = past_seq + 1;
        cache_mut.layers_k = new_k;
        cache_mut.layers_v = new_v;

        self.host_greedy_argmax(&hidden, adjust)
    }

    /// GPU-resident K/V + hidden-only graph: no padded K/V upload, no vocab logits D2H.
    fn decode_step_greedy_resident(
        &mut self,
        past_seq: usize,
        input_tok: u32,
        adjust: &mut dyn FnMut(&mut [f32]),
    ) -> Result<u32> {
        let path = self
            .weights_path
            .as_ref()
            .context("packed decode needs gguf path")?;
        let path_str = path.to_str().context("non-utf8 weights path")?.to_string();

        let (bucket_idx, upper) = {
            let cache_dec = self.decode_compile_cache_hidden.as_ref().unwrap();
            let bucket_idx = cache_dec
                .bucket_for(past_seq as u64)
                .ok_or_else(|| anyhow::anyhow!("past_seq {past_seq} outside any bucket"))?;
            let upper = cache_dec
                .buckets()
                .nth(bucket_idx)
                .map(|r| r.end - 1)
                .unwrap() as usize;
            (bucket_idx, upper)
        };

        let kv_dim = self.cfg.kv_proj_dim();
        let n_layers = self.cfg.kv_layers();
        let exec_device = packed_gguf_execution_device(self.decode_device());

        let needs_bind = !self.decode_resident_hidden_bound.contains(&bucket_idx);
        if needs_bind {
            if !self.decode_bucket_allowed() {
                anyhow::bail!(
                    "decode bucket compile would exceed soft RAM budget (~80% of physical RAM); \
                     set RLX_SOFT_MEMORY_FRACTION or ORPHEUS_DECODE_CACHE_CAP lower"
                );
            }
            let device_kv_rebind = cuda_device_kv_rebind_enabled(exec_device) && bucket_idx > 0;
            let flush_plan = resident_bucket_flush_plan(
                self.decode_compile_cache_hidden.as_ref().unwrap(),
                past_seq,
            );
            if cuda_lazy_kv_enabled(exec_device) && bucket_idx > 0 {
                if let Some(compiled) = self
                    .decode_compile_cache_hidden
                    .as_mut()
                    .and_then(|c| c.compiled_for_key_mut(flush_plan.prev_key))
                {
                    if let Some(cache) = self.cache.as_mut() {
                        maybe_flush_resident_kv_before_bucket(
                            compiled,
                            cache,
                            &flush_plan,
                            past_seq,
                            kv_dim,
                            n_layers,
                        )?;
                    }
                }
            }
            if !device_kv_rebind {
                if let Some(cache_mut) = self.decode_compile_cache_hidden.as_mut() {
                    cache_mut.evict_except(bucket_idx);
                }
                self.decode_resident_hidden_bound.clear();
            }

            let needs_compile = !self.decode_loaded_buckets_hidden.contains(&bucket_idx);
            if needs_compile {
                self.ensure_decode_weights_cache()?;
                let cfg = self.cfg.clone();
                let mut packed: HashMap<String, (rlx_ir::quant::QuantScheme, Vec<usize>)> =
                    HashMap::new();
                let (graph, params) = build_llama32_decode_graph_sized_packed_ext(
                    &cfg,
                    self.decode_weights_cache.as_mut().unwrap(),
                    /*batch*/ 1,
                    upper,
                    /*use_custom_mask*/ true,
                    /*with_lm_head*/ false,
                    &mut packed,
                )?;

                let opts = compile_options_for_packed_gguf_prefill_with_profile(
                    &self.decode_profile,
                    exec_device,
                );
                let cache_mut = self.decode_compile_cache_hidden.as_mut().unwrap();
                let loader_ref: &dyn WeightLoader = self.decode_weights_cache.as_ref().unwrap();
                packed_gguf_compile_guard(exec_device, || {
                    let (_u, compiled) = cache_mut
                        .get_or_compile_with_options(past_seq as u64, |_upper| graph, &opts)
                        .expect("bucket must exist; we just looked it up");
                    let f32_keys: HashSet<String> = params.keys().cloned().collect();
                    attach_f32_params(compiled, params);
                    // Zero-copy upload, but skip any key already attached as an
                    // F32 param (packed/F32 keys are disjoint by construction;
                    // the guard is kept defensively).
                    for name in packed.keys() {
                        if f32_keys.contains(name.as_str()) {
                            continue;
                        }
                        let bytes = loader_ref
                            .tensor_bytes_borrowed(name)
                            .expect("packed decode: zero-copy weight bytes");
                        compiled.set_param_typed(name, bytes, rlx_ir::DType::U8);
                    }
                });
                self.decode_loaded_buckets_hidden.insert(bucket_idx);
            } else if self.decode_weights_cache.is_none() {
                self.decode_weights_cache = Some(CachedGgufWeights::from_file(&path_str)?);
            }

            if device_kv_rebind {
                if let Some(cache_mut) = self.decode_compile_cache_hidden.as_mut() {
                    cache_mut.evict_except(bucket_idx);
                }
                self.decode_resident_hidden_bound.clear();
            }
            let cache_ref = self
                .cache
                .as_ref()
                .context("resident decode without cache")?;
            let cache_mut = self.decode_compile_cache_hidden.as_mut().unwrap();
            packed_gguf_compile_guard(exec_device, || {
                let compiled = cache_mut
                    .compiled_for_key_mut(past_seq as u64)
                    .expect("hidden decode bucket must exist for resident bind");
                bind_resident_kv_from_host_cache(compiled, cache_ref, upper, kv_dim, n_layers);
            });
            self.decode_resident_hidden_bound.insert(bucket_idx);
        }

        let (cos_row, sin_row) = rope_slice(&self.inv_freq, past_seq);
        let input_ids_f32 = [input_tok as f32];
        let mask = bucket_decode_mask(past_seq, upper);

        let lazy_embed = self.decode_weights_cache.as_ref().and_then(|cache| {
            // Host-side lazy gather iff the embed is K-quant and not F32; bytes
            // borrowed zero-copy from the loader mmap.
            let key = "model.embed_tokens.weight";
            if cache.f32_take.contains_key(key) {
                return None;
            }
            let (scheme, _) = cache.packed_meta(key)?;
            let bytes = cache.tensor_bytes_borrowed(key)?;
            Some((bytes, scheme))
        });
        let mut embed_scratch = Vec::new();
        let mut run_inputs: Vec<(&str, &[f32])> = Vec::with_capacity(4);
        push_packed_decode_token_input(
            &self.cfg,
            input_tok,
            lazy_embed,
            &mut embed_scratch,
            &mut run_inputs,
            &input_ids_f32,
        )?;
        run_inputs.push(("cos", cos_row.as_slice()));
        run_inputs.push(("sin", sin_row.as_slice()));
        run_inputs.push(("mask", mask.as_slice()));

        let cache_mut = self.decode_compile_cache_hidden.as_mut().unwrap();
        let compiled = cache_mut
            .compiled_for_key_mut(past_seq as u64)
            .expect("hidden resident bucket was just loaded");
        let mut outs = compiled.run_read_outputs(&run_inputs, Some(&[0]));
        compiled.feed_kv_row(upper, past_seq, kv_dim);

        let lazy_kv = cuda_lazy_kv_enabled(exec_device);
        let sync_host = !lazy_kv;
        let mut new_rows: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(n_layers);
        if sync_host {
            for i in 0..n_layers {
                let nk = compiled
                    .read_output_row(1 + 2 * i, upper, kv_dim)
                    .with_context(|| format!("resident greedy decode K row layer {i}"))?;
                let nv = compiled
                    .read_output_row(2 + 2 * i, upper, kv_dim)
                    .with_context(|| format!("resident greedy decode V row layer {i}"))?;
                new_rows.push((nk, nv));
            }
        }
        let hidden = outs
            .drain(..)
            .next()
            .context("resident greedy decode hidden missing")?;

        if sync_host {
            if let Some(cache) = self.cache.as_mut() {
                cache.past_seq = past_seq + 1;
                for (i, (nk, nv)) in new_rows.into_iter().enumerate() {
                    cache.layers_k[i].extend_from_slice(&nk);
                    cache.layers_v[i].extend_from_slice(&nv);
                }
            }
        } else if let Some(cache) = self.cache.as_mut() {
            cache.past_seq = past_seq + 1;
        }

        self.host_greedy_argmax(&hidden, adjust)
    }

    /// GPU-resident KV bucketed packed decode (Metal/Vulkan/CUDA). Same compiled
    /// graph and numerics as [`Self::decode_step_bucketed_packed`], but the
    /// per-layer `past_k_*`/`past_v_*` are bound as resident device handles once
    /// per bucket and the decode output's new-token row is folded back in-place
    /// on the arena (`feed_kv_row`) — so no padded K/V is uploaded and only
    /// logits are read back each step. The host `cache.layers_k/v` is synced
    /// from the device only when a bucket change forces a rebuild.
    ///
    /// Two per-bucket concerns are tracked separately so a warm utterance can
    /// reuse a compiled graph without replaying stale K/V (see
    /// [`Self::soft_release_decode_kv_bindings`]):
    /// - `needs_compile` (`decode_loaded_buckets`) — graph built + params
    ///   uploaded; model-constant, so it **persists across utterances** and the
    ///   expensive recompile + weight upload is skipped on warm runs.
    /// - `needs_bind` (`decode_resident_bound`) — resident K/V holds *this*
    ///   utterance's prefix; per-utterance, so `prefill` clears it to force a
    ///   fresh [`bind_resident_kv_from_host_cache`]. Conflating the two (the
    ///   original single `needs_load`) would skip the re-bind and decode stale
    ///   K/V → STOP after ~2 tokens. On CUDA/ROCm the two sets move in lockstep
    ///   (reuse disabled), reproducing the original behavior.
    fn decode_step_bucketed_packed_resident(
        &mut self,
        past_seq: usize,
        input_tok: u32,
    ) -> Result<Vec<f32>> {
        let path = self
            .weights_path
            .as_ref()
            .context("packed decode needs gguf path")?;
        let path_str = path.to_str().context("non-utf8 weights path")?.to_string();

        let (bucket_idx, upper) = {
            let cache_dec = self.decode_compile_cache.as_ref().unwrap();
            let bucket_idx = cache_dec
                .bucket_for(past_seq as u64)
                .ok_or_else(|| anyhow::anyhow!("past_seq {past_seq} outside any bucket"))?;
            let upper = cache_dec
                .buckets()
                .nth(bucket_idx)
                .map(|r| r.end - 1)
                .unwrap() as usize;
            (bucket_idx, upper)
        };

        let kv_dim = self.cfg.kv_proj_dim();
        let n_layers = self.cfg.kv_layers();
        let exec_device = packed_gguf_execution_device(self.decode_device());

        // Two independent per-bucket concerns, tracked separately so a warm
        // utterance can reuse a compiled graph without re-binding stale K/V:
        //   • `needs_compile` — graph built + params uploaded? (model-constant,
        //     persists across utterances → the expensive part is skipped warm)
        //   • `needs_bind`    — resident K/V holds *this* utterance's prefix?
        //     (per-utterance; `prefill` clears `decode_resident_bound`)
        // On CUDA/ROCm both sets move in lockstep (reuse disabled), reproducing
        // the original single-`needs_load` behavior.
        let needs_bind = !self.decode_resident_bound.contains(&bucket_idx);
        if needs_bind {
            let reuse = self.cross_utterance_decode_reuse_enabled();
            let needs_compile = !self.decode_loaded_buckets.contains(&bucket_idx);
            let device_kv_rebind = cuda_device_kv_rebind_enabled(exec_device) && bucket_idx > 0;
            let flush_plan =
                resident_bucket_flush_plan(self.decode_compile_cache.as_ref().unwrap(), past_seq);
            if cuda_lazy_kv_enabled(exec_device) && bucket_idx > 0 {
                if let Some(compiled) = self
                    .decode_compile_cache
                    .as_mut()
                    .and_then(|c| c.compiled_for_key_mut(flush_plan.prev_key))
                {
                    if let Some(cache) = self.cache.as_mut() {
                        maybe_flush_resident_kv_before_bucket(
                            compiled,
                            cache,
                            &flush_plan,
                            past_seq,
                            kv_dim,
                            n_layers,
                        )?;
                    }
                }
            }

            if needs_compile {
                if reuse {
                    // Free room for the new bucket BEFORE the budget gate so a
                    // full resident ladder doesn't spuriously trip it; trim only
                    // to fit so the rest of the ladder stays cached.
                    if !device_kv_rebind {
                        self.trim_decode_buckets_to_budget(bucket_idx);
                    }
                    self.ensure_decode_bucket_budget()?;
                } else {
                    // Original order: gate first, then collapse to the single
                    // active bucket (the `!decode_loaded_buckets.is_empty()`
                    // short-circuit keeps mid-utterance climbs allowed).
                    self.ensure_decode_bucket_budget()?;
                    if !device_kv_rebind {
                        if let Some(cache_mut) = self.decode_compile_cache.as_mut() {
                            cache_mut.evict_except(bucket_idx);
                        }
                        self.decode_loaded_buckets.clear();
                        self.decode_resident_bound.clear();
                    }
                }

                if self.decode_weights_cache.is_none() {
                    self.decode_weights_cache = Some(CachedGgufWeights::from_file(&path_str)?);
                }
                let cfg = self.cfg.clone();
                let mut packed: HashMap<String, (rlx_ir::quant::QuantScheme, Vec<usize>)> =
                    HashMap::new();
                let (graph, params) = build_llama32_decode_graph_sized_packed(
                    &cfg,
                    self.decode_weights_cache.as_mut().unwrap(),
                    /*batch*/ 1,
                    upper,
                    /*use_custom_mask*/ true,
                    &mut packed,
                )?;

                let opts = compile_options_for_packed_gguf_prefill_with_profile(
                    &self.decode_profile,
                    exec_device,
                );
                let cache_mut = self.decode_compile_cache.as_mut().unwrap();
                let loader_ref: &dyn WeightLoader = self.decode_weights_cache.as_ref().unwrap();
                packed_gguf_compile_guard(exec_device, || {
                    let (_u, compiled) = cache_mut
                        .get_or_compile_with_options(past_seq as u64, |_upper| graph, &opts)
                        .expect("bucket must exist; we just looked it up");
                    attach_f32_params(compiled, params);
                    upload_packed_borrowed(compiled, &packed, loader_ref)
                        .expect("packed decode: zero-copy weight upload");
                });
                self.decode_loaded_buckets.insert(bucket_idx);
            } else if self.decode_weights_cache.is_none() {
                // Reused compiled bucket: the graph + params are already
                // resident, but the lazy-embed gather below still needs the
                // weights cache.
                self.decode_weights_cache = Some(CachedGgufWeights::from_file(&path_str)?);
            }

            if device_kv_rebind {
                // CUDA D2D rebind ordering: collapse to the single active
                // bucket after (re)compile, then bind K/V.
                if let Some(cache_mut) = self.decode_compile_cache.as_mut() {
                    cache_mut.evict_except(bucket_idx);
                }
                self.decode_loaded_buckets.clear();
                self.decode_loaded_buckets.insert(bucket_idx);
                self.decode_resident_bound.clear();
            }
            let cache_ref = self
                .cache
                .as_ref()
                .context("resident decode without cache")?;
            packed_gguf_compile_guard(exec_device, || {
                let cache_mut = self.decode_compile_cache.as_mut().unwrap();
                let compiled = cache_mut
                    .compiled_for_key_mut(past_seq as u64)
                    .expect("bucket must exist after compile");
                bind_resident_kv_from_host_cache(compiled, cache_ref, upper, kv_dim, n_layers);
            });
            self.decode_resident_bound.insert(bucket_idx);
        }

        let (cos_row, sin_row) = rope_slice(&self.inv_freq, past_seq);
        let input_ids_f32 = [input_tok as f32];
        let mask = bucket_decode_mask(past_seq, upper);

        let lazy_embed = self.decode_weights_cache.as_ref().and_then(|cache| {
            // Host-side lazy gather iff the embed is K-quant and not F32; bytes
            // borrowed zero-copy from the loader mmap.
            let key = "model.embed_tokens.weight";
            if cache.f32_take.contains_key(key) {
                return None;
            }
            let (scheme, _) = cache.packed_meta(key)?;
            let bytes = cache.tensor_bytes_borrowed(key)?;
            Some((bytes, scheme))
        });
        let mut embed_scratch = Vec::new();
        let mut run_inputs: Vec<(&str, &[f32])> = Vec::with_capacity(4);
        push_packed_decode_token_input(
            &self.cfg,
            input_tok,
            lazy_embed,
            &mut embed_scratch,
            &mut run_inputs,
            &input_ids_f32,
        )?;
        run_inputs.push(("cos", cos_row.as_slice()));
        run_inputs.push(("sin", sin_row.as_slice()));
        run_inputs.push(("mask", mask.as_slice()));

        let cache_mut = self.decode_compile_cache.as_mut().unwrap();
        let compiled = cache_mut
            .compiled_for_key_mut(past_seq as u64)
            .expect("bucket was just loaded above");
        // Resident K/V: only the small per-token inputs are uploaded, only
        // logits (output 0) are read back; K/V never leaves the arena.
        let mut outs = compiled.run_read_outputs(&run_inputs, Some(&[0]));
        // In-bucket residency: fold the new token row (output row `upper`) into
        // the resident `past_*` slot at the active position, in-place on the
        // arena — so the next in-bucket step needs no K/V re-upload. This is a
        // no-op clamp when `past_seq == upper` (top of bucket); that boundary
        // token is still carried forward via the host-cache append below.
        compiled.feed_kv_row(upper, past_seq, kv_dim);

        // Pull just the new-token K/V row (output row `upper`) back to host so
        // `cache.layers_k/v` always reflects the full prefix — the bucket-change
        // rebind seeds the next (wider) bucket from it, and it captures the
        // top-of-bucket token the in-slot feed cannot store. One row per layer
        // (~kv_dim) vs the legacy full-K/V upload+readback per step.
        // CUDA lazy KV skips in-bucket row D2H and bulk-flushes on bucket change.
        let lazy_kv = cuda_lazy_kv_enabled(exec_device);
        let sync_host = !lazy_kv;
        let mut new_rows: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(n_layers);
        if sync_host {
            for i in 0..n_layers {
                let nk = compiled
                    .read_output_row(1 + 2 * i, upper, kv_dim)
                    .with_context(|| format!("resident decode K row read layer {i}"))?;
                let nv = compiled
                    .read_output_row(2 + 2 * i, upper, kv_dim)
                    .with_context(|| format!("resident decode V row read layer {i}"))?;
                new_rows.push((nk, nv));
            }
        }
        let logits = outs
            .drain(..)
            .next()
            .context("resident decode logits missing")?;

        if sync_host {
            if let Some(cache) = self.cache.as_mut() {
                for (i, (nk, nv)) in new_rows.into_iter().enumerate() {
                    cache.layers_k[i].extend_from_slice(&nk);
                    cache.layers_v[i].extend_from_slice(&nv);
                }
            }
        }
        Ok(logits)
    }

    /// Resident GPU-KV decode: K/V live in the device arena across steps, fed
    /// in-place from the decode output (no per-step host K/V upload/readback).
    /// Enabled for native GGUF packed bucket decode on backends with a
    /// host-accessible arena + row-feed support (Vulkan, Metal). Opt out with
    /// `ORPHEUS_RESIDENT_KV=0` (legacy `ORPHEUS_VULKAN_RESIDENT_KV=0` honored).
    fn resident_kv_decode_enabled(&self) -> bool {
        // CUDA / ROCm: resident KV via D2D `feed_kv_row` + logits-only readback.
        let supported = matches!(
            self.decode_device(),
            Device::Vulkan | Device::Metal | Device::Cuda | Device::Rocm
        );
        let off = std::env::var("ORPHEUS_RESIDENT_KV").ok().as_deref() == Some("0")
            || std::env::var("ORPHEUS_VULKAN_RESIDENT_KV").ok().as_deref() == Some("0");
        supported && !off && self.use_packed_decode() && self.decode_compile_cache.is_some()
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
            attach_f32_params(compiled, params);
        }

        let position = decode_position_input(past_seq);
        let input_ids_f32 = [input_tok as f32];
        let key_strs: Vec<String> = (0..self.cfg.kv_layers())
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        let mut inputs: Vec<(&str, &[f32])> = Vec::with_capacity(4 + 2 * self.cfg.kv_layers());
        inputs.push(("input_ids", input_ids_f32.as_slice()));
        inputs.push(("position", position.as_slice()));
        for i in 0..self.cfg.kv_layers() {
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
        let n_layers = self.cfg.kv_layers();

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
            let (hir, params) = {
                let mut loader = self.build_weight_loader()?;
                build_llama32_decode_hir_sized_ext(
                    &self.cfg,
                    &mut loader,
                    /*batch*/ 1,
                    upper,
                    /*use_custom_mask*/ true,
                )?
            };
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
                    attach_f32_params(compiled, params);
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
                    attach_f32_params(compiled, params);
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
            attach_f32_params(&mut compiled, params);
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
        let n_layers = self.cfg.kv_layers();
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
                    attach_f32_params(compiled, params);
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
            attach_f32_params(&mut compiled, params);
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
                attach_f32_params(compiled, params);
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
                    attach_f32_params(compiled, params);
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
            attach_f32_params(&mut compiled, params);
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
        let n_layers = self.cfg.kv_layers();
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

    /// Like [`generate_cached_with`](Self::generate_cached_with) but the
    /// callback returns whether to keep going. Returning `false` stops the
    /// loop *after* the just-sampled token has been observed, so callers
    /// can halt on an end-of-sequence id (or any other stop condition)
    /// without wasting decode steps past it — the freshly sampled token is
    /// still included in the returned slice.
    pub fn generate_cached_until(
        &mut self,
        n: usize,
        opts: SampleOpts,
        mut keep_going: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        let start = self.tokens.len();
        for _ in 0..n {
            let tok = self.step_cached(opts)?;
            if !keep_going(tok) {
                break;
            }
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
        let n_layers = self.cfg.kv_layers();
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
        let n_layers = self.cfg.kv_layers();
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
        if self.host_greedy_prefill_eligible(opts) {
            return self.seed_cache_from_prompt_host_greedy(&mut adjust_logits);
        }
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

    fn seed_cache_from_prompt_host_greedy(
        &mut self,
        adjust: &mut dyn FnMut(&mut [f32]),
    ) -> Result<u32> {
        let seq = self.tokens.len();
        let ids_f32: Vec<f32> = self.tokens.iter().map(|&i| i as f32).collect();
        let (hidden, layers_k, layers_v) = if self.uses_packed_gguf_cpu_prefill()
            || self.uses_packed_gguf_mlx_prefill()
        {
            self.ensure_packed_prefill(seq)?;
            self.packed_gguf_prefill
                .as_mut()
                .unwrap()
                .run_hidden_with_kv(seq, &ids_f32)?
        } else if uses_packed_gguf_gpu_prefill(self.device, self.metal_gguf_prefill_mode()) {
            if matches!(self.device, Device::Cuda | Device::Rocm) && !cuda_f32_prefill_forced() {
                self.seed_cuda_host_greedy_reference_prefill(seq, &ids_f32)?
            } else {
                self.ensure_packed_prefill(seq)?;
                self.packed_gguf_prefill
                    .as_mut()
                    .unwrap()
                    .run_hidden_with_kv(seq, &ids_f32)?
            }
        } else if self.host_greedy_lm {
            let hidden = self.run_prefill_last_hidden_cpu_f32(1, seq, &ids_f32)?;
            let outputs = self.run_prefill_with_cache_cpu_f32(1, seq, &ids_f32)?;
            let (_, layers_k, layers_v) = self.split_prefill_outputs(outputs, 1, seq)?;
            (hidden, layers_k, layers_v)
        } else if let Some(triple) = self.try_packed_gguf_prefill(seq, &ids_f32)? {
            let (logits, k, v) = triple;
            let h = self.cfg.hidden_size;
            if logits.len() >= h && logits.len() < self.cfg.vocab_size {
                (logits[..h].to_vec(), k, v)
            } else {
                anyhow::bail!(
                    "host greedy prefill expected hidden vector, got len {}",
                    logits.len()
                );
            }
        } else {
            anyhow::bail!("host greedy lm_head requires packed GGUF prefill");
        };
        self.cache = Some(KvCacheState {
            past_seq: seq,
            layers_k,
            layers_v,
        });
        let kv_dim = self.cfg.kv_proj_dim();
        let keep = seq.saturating_mul(kv_dim);
        if let Some(cache) = self.cache.as_mut() {
            for i in 0..cache.layers_k.len() {
                cache.layers_k[i].truncate(keep);
                cache.layers_v[i].truncate(keep);
            }
        }
        let tok = self.host_greedy_argmax(&hidden, adjust)?;
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

        let (logits, new_k, new_v) = if self.use_packed_decode() {
            if self.bucket_decode_eligible(past_seq) {
                self.decode_step_bucketed_packed(past_seq, input)?
            } else {
                self.decode_step_packed(past_seq, input)?
            }
        } else if self.decode_dynamic_cache.is_some() {
            self.decode_step_dynamic(past_seq, input)?
        } else if self.bucket_decode_eligible(past_seq) {
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
            embedding_scale: None,
            residual_scale: None,
            attention_scale: None,
            logit_scale: None,
            num_loops: 1,
            skip_loop_final_norm: false,
            rope_style: rlx_ir::RopeStyle::NeoX,
            gguf_arch: None,
            rope_dim: None,
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
    fn cuda_packed_gguf_defers_host_drain() {
        use crate::prefill_mode::MetalGgufPrefillMode;
        assert!(gguf_defers_f32_drain(
            Device::Cuda,
            true,
            true,
            MetalGgufPrefillMode::PackedGguf,
        ));
        assert!(!gguf_defers_f32_drain(
            Device::Cuda,
            true,
            false,
            MetalGgufPrefillMode::CpuF32,
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

    // Used only by feature-gated (`metal`/etc.) tests; dead under default features.
    #[allow(dead_code)]
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
            embedding_scale: None,
            residual_scale: None,
            attention_scale: None,
            logit_scale: None,
            num_loops: 1,
            skip_loop_final_norm: false,
            rope_style: rlx_ir::RopeStyle::NeoX,
            gguf_arch: None,
            rope_dim: None,
        }
    }

    #[allow(dead_code)]
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
            attach_f32_params(&mut compiled, params);
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
                embedding_scale: None,
                residual_scale: None,
                attention_scale: None,
                logit_scale: None,
                num_loops: 1,
                skip_loop_final_norm: false,
                rope_style: rlx_ir::RopeStyle::NeoX,
                gguf_arch: None,
                rope_dim: None,
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
                embedding_scale: None,
                residual_scale: None,
                attention_scale: None,
                logit_scale: None,
                num_loops: 1,
                skip_loop_final_norm: false,
                rope_style: rlx_ir::RopeStyle::NeoX,
                gguf_arch: None,
                rope_dim: None,
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
                embedding_scale: None,
                residual_scale: None,
                attention_scale: None,
                logit_scale: None,
                num_loops: 1,
                skip_loop_final_norm: false,
                rope_style: rlx_ir::RopeStyle::NeoX,
                gguf_arch: None,
                rope_dim: None,
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
