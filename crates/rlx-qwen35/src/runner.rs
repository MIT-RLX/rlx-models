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

//! High-level runner for Qwen3.5 / Qwen3.6 (qwen35 architecture).

use crate::cache::{
    Qwen35DecodeCache, advance_cache_from_decode_outputs, decode_step_feeds, last_token_indices,
    pack_input_ids, seed_cache_from_outputs, zero_prompt_padding_kv, zero_recurrent_inputs,
};
use crate::capabilities::validate_device;
use crate::config::Qwen35Config;
use crate::encode_prompt_auto;
use crate::lm_head::{
    greedy_lm_head_argmax, lm_head_logits_row, sample_lm_cap, sample_lm_head_from_hidden,
};
use crate::moe_offload::{MoeOffloadState, build_moe_offload};
use crate::moe_store::{build_moe_expert_store, moe_host_bind_from_store};
use crate::rope::{mrope_prefill_feeds, mrope_row_for_sections, text_section_pos};
use crate::vision::{
    MmProjConfig, MmProjWeights, MultimodalPrefill, MultimodalPrompt, Qwen35VisionEncoder,
    load_vision_encoder,
};
use crate::weights::Qwen35Weights;
use crate::{
    PackedParams, build_qwen35_decode_hir_dynamic_ext, build_qwen35_decode_hir_ext,
    build_qwen35_hir_sized_ext, build_qwen35_prefill_cache_hir_dynamic_ext,
    build_qwen35_prefill_cache_hir_ext, build_qwen35_prefill_hidden_cache_hir_dynamic_ext,
    build_qwen35_prefill_hidden_cache_hir_ext,
};
use rlx_core::WeightLoader;
use rlx_runtime::MoeExpertStore;

fn push_moe_residency(compiled: &mut rlx_runtime::CompiledGraph, layers: &[Vec<bool>]) {
    let refs: Vec<&[bool]> = layers.iter().map(|m| m.as_slice()).collect();
    compiled.set_moe_resident_experts_per_layer(&refs);
}

/// Minimal NumPy `.npy` writer for F32 row-major dumps (layer parity).
fn write_npy_f32_row_major(path: &Path, rows: usize, cols: usize, data: &[f32]) -> Result<()> {
    use std::io::Write;
    anyhow::ensure!(
        data.len() == rows * cols,
        "npy len {} != {rows}*{cols}",
        data.len()
    );
    let header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    let mut header_bytes = header.into_bytes();
    while (10 + header_bytes.len() + 1) % 16 != 0 {
        header_bytes.push(b' ');
    }
    header_bytes.push(b'\n');
    let mut file = std::fs::File::create(path)?;
    file.write_all(b"\x93NUMPY\x01\x00")?;
    file.write_all(&(header_bytes.len() as u16).to_le_bytes())?;
    file.write_all(&header_bytes)?;
    for &x in data {
        file.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn refresh_moe_from_capture(
    mo: &mut MoeOffloadState,
    store: Option<&MoeExpertStore>,
    compiled: &mut rlx_runtime::CompiledGraph,
    layer_indices: &[Vec<u32>],
    denoise_step: usize,
    is_prefill_block: bool,
) -> bool {
    let refreshed = if let Some(store) = store {
        mo.refresh_from_capture_with_store(store, layer_indices, denoise_step, is_prefill_block)
    } else {
        mo.refresh_from_capture(layer_indices, denoise_step, is_prefill_block)
    };
    if refreshed {
        push_moe_residency(compiled, &mo.per_layer_resident_masks());
    }
    refreshed
}
use crate::execution::{
    Qwen35CompileCache, decode_config, get_or_specialize_hir_with_options, hidden_prefill_config,
    prefill_config,
};
use crate::flow::{Qwen35PrefillCacheOpts, build_qwen35_prefill_cache_built};
use crate::profile::{qwen35_profile_default, qwen35_profile_near_weights};
use anyhow::{Context, Result, anyhow, bail};
use rlx_core::flow_bridge::{
    compile_options_for_packed_gguf_prefill_with_profile, compile_options_from_profile,
};
use rlx_core::gguf_support::{GgufModelFamily, assert_gguf_family, resolve_weights_file};
use rlx_core::weight_loader::GgufLoader;
use rlx_core::{packed_prefill_active_extent_enabled, run_packed_prefill};
use rlx_flow::ModelExecutionConfig;
use rlx_flow::{CompileProfile, ExecutionPreset};
use rlx_ir::CompilationMode;
use rlx_ir::logical_kernel::KernelDispatchConfig;
use rlx_qwen3::sampling::{SampleOpts, sample_token};
use rlx_runtime::compile_cache::BucketedCompileCache;
use rlx_runtime::{AotCache, CompileOptions, Device, Session, trim_accelerator_arena_pool};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Source for the Qwen3.5 / 3.6 config. Mirrors `Qwen3ConfigSource`
/// so callers using `Qwen35RunnerBuilder` have the same shape as
/// `Qwen3RunnerBuilder` (PLAN.md M1).
///
/// `JsonFile` and `Explicit` are wired into the future safetensors
/// load path (catalog rows: `qwen35-{4b,9b,27b}-hauhau-aggressive`).
/// Until the safetensors load lands, `build()` errors with a clear
/// M1 follow-up message when these variants are used.
/// Alias of `rlx_runtime::ConfigSource<Qwen35Config>`. `JsonFile` /
/// `Explicit` variants are wired into the future safetensors load path
/// (catalog rows: `qwen35-{4b,9b,27b}-hauhau-aggressive`). Until that
/// path lands, `build()` errors with a clear M1 follow-up message when
/// these variants are used.
pub type Qwen35ConfigSource = rlx_runtime::ConfigSource<Qwen35Config>;

#[derive(Default, Debug)]
pub struct Qwen35RunnerBuilder {
    weights: Option<PathBuf>,
    config: Option<Qwen35ConfigSource>,
    device: Option<Device>,
    max_seq: Option<usize>,
    /// Compile static prefill at this seq (≤ max_seq). `--fast` sets this to
    /// the prompt length so prefill GEMMs are not padded to decode capacity.
    prefill_seq: Option<usize>,
    enable_mtp: bool,
    last_logits_only: bool,
    /// `None` = auto-detect (packed when GGUF ≥ 256 MB to avoid the
    /// F32-dequant memory explosion — a 4B Q3_K_S file is ~2 GB on
    /// disk but ~16 GB dense-F32). `Some(true)` / `Some(false)` are
    /// explicit user overrides.
    packed_weights: Option<bool>,
    runtime_mrope: bool,
    mrope_section_positions: Option<Vec<[usize; 4]>>,
    batch: Option<usize>,
    bucketed_decode: Option<bool>,
    /// Emit/consume MTP logits on the prefill-cache + decode path (draft speculator).
    mtp_logits_path: bool,
    fast_mtp: bool,
    /// Skip LM head in decode graphs; argmax on host (default: true).
    fast_greedy_lm_head: Option<bool>,
    /// Persist optimized LIR under this directory (warm-start / AOT).
    aot_cache_dir: Option<PathBuf>,
    /// Compile prefill once with `sym::SEQ`, specialize per prompt length.
    dynamic_prefill: bool,
    /// Compile decode once with `sym::PAST_SEQ`, specialize per prefix length.
    dynamic_decode: bool,
    inline_weights: Option<(Qwen35Config, Qwen35Weights)>,
    /// Optional mmproj GGUF for VLM vision encoding.
    mmproj: Option<PathBuf>,
    /// Inline mmproj weights (tests; mutually exclusive with [`Self::mmproj`]).
    inline_mmproj: Option<(crate::vision::MmProjConfig, crate::vision::MmProjWeights)>,
    /// Skip auto-loading HF `model.visual.*` (text-only / low-mem probes).
    skip_auto_mmproj: bool,
    /// Override tier-1 prefill profile (else `qwen35.rlx.toml` or defaults).
    prefill_profile: Option<CompileProfile>,
    /// Override tier-1 decode profile.
    decode_profile: Option<CompileProfile>,
    /// TIDE-style cap on GPU-resident experts per MoE layer (`max_gpu_experts_per_layer`).
    max_gpu_experts_per_layer: Option<usize>,
    /// Unified RAM / VRAM budget for auto expert cap (optional).
    moe_memory_budget_bytes: Option<usize>,
    /// Refresh expert placement every N decode/denoise steps (TIDE `jump_steps` τ).
    expert_refresh_every_decode_steps: Option<usize>,
    /// TIDE `jump_steps` alias (preferred name).
    jump_steps: Option<usize>,
    /// TIDE `reserve_vram_gb` (default 1.5).
    reserve_vram_gb: Option<f64>,
    /// TIDE `collect_stats` on MoE forwards.
    moe_collect_stats: bool,
    /// Build hidden-state prefill graphs without loading mmproj (Gepard TTS).
    hidden_prefill: bool,
    /// Force host-gathered `inputs_embeds` on decode (custom per-step embeddings).
    force_host_embed: bool,
    /// Skip eager decode/predict warm (Gepard / low-mem).
    skip_warm: bool,
}

impl Qwen35RunnerBuilder {
    pub fn weights(mut self, path: impl Into<PathBuf>) -> Self {
        self.weights = Some(path.into());
        self
    }

    /// Source for the Qwen3.5 / 3.6 config. Default
    /// `Qwen35ConfigSource::Embedded` (GGUF metadata). PLAN.md M1
    /// — `JsonFile` / `Explicit` reserve the API shape for the
    /// safetensors load path; today they error in `build()` with a
    /// follow-up message.
    pub fn config(mut self, src: Qwen35ConfigSource) -> Self {
        self.config = Some(src);
        self
    }

    /// Convenience: explicit `Qwen35Config` (shorthand for
    /// `.config(Qwen35ConfigSource::Explicit(cfg))`).
    pub fn config_value(self, cfg: Qwen35Config) -> Self {
        self.config(Qwen35ConfigSource::Explicit(cfg))
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }
    pub fn max_seq(mut self, n: usize) -> Self {
        self.max_seq = Some(n);
        self
    }

    /// Compile the static prefill graph at `n` (must be ≤ [`Self::max_seq`]).
    /// Decode capacity stays at `max_seq`. Used by `--fast` so a short prompt
    /// does not pay full-`max_seq` packed GEMMs.
    pub fn prefill_seq(mut self, n: usize) -> Self {
        self.prefill_seq = Some(n);
        self
    }

    pub fn enable_mtp(mut self, on: bool) -> Self {
        self.enable_mtp = on;
        self
    }
    pub fn last_logits_only(mut self, on: bool) -> Self {
        self.last_logits_only = on;
        self
    }
    pub fn packed_weights(mut self, on: bool) -> Self {
        self.packed_weights = Some(on);
        self
    }
    /// Use runtime MRoPE cos/sin inputs instead of a baked table. Required
    /// for multimodal prompts where section positions differ from `[p,p,p,0]`.
    pub fn runtime_mrope(mut self, on: bool) -> Self {
        self.runtime_mrope = on;
        self
    }
    /// Per-token MRoPE section positions `[t,h,w,extra]` (length = prompt seq).
    pub fn mrope_section_positions(mut self, positions: Vec<[usize; 4]>) -> Self {
        self.mrope_section_positions = Some(positions);
        self
    }
    /// Batch size for compiled graphs (default 1).
    pub fn batch(mut self, n: usize) -> Self {
        self.batch = Some(n);
        self
    }
    /// Use power-of-two bucketed decode compile cache (default: true).
    pub fn bucketed_decode(mut self, on: bool) -> Self {
        self.bucketed_decode = Some(on);
        self
    }
    /// Use MTP head logits on prefill-cache seeding and decode steps.
    pub fn mtp_logits_path(mut self, on: bool) -> Self {
        self.mtp_logits_path = on;
        self
    }
    /// Trim MTP LM head to 32K vocab (llama.cpp FastMTP draft path).
    pub fn fast_mtp(mut self, on: bool) -> Self {
        self.fast_mtp = on;
        self
    }
    /// Decode without graph LM head — host argmax over tied embedding.
    ///
    /// Default is **on** for CPU / small models. For packed GGUF on discrete
    /// GPUs the host Q1/K-quant vocab scan is minute-scale (Bonsai-27B), so
    /// the builder defaults to **off** and keeps the LM head in the decode
    /// graph (fused on-device GEMV).
    pub fn fast_greedy_lm_head(mut self, on: bool) -> Self {
        self.fast_greedy_lm_head = Some(on);
        self
    }
    /// Cache optimized LIR on disk for faster subsequent runs.
    pub fn aot_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.aot_cache_dir = Some(path.into());
        self
    }
    /// Use dynamic prefill specialization (symbolic seq; batch=1).
    pub fn dynamic_prefill(mut self, on: bool) -> Self {
        self.dynamic_prefill = on;
        self
    }
    /// Use dynamic decode specialization (symbolic past_seq; batch=1).
    pub fn dynamic_decode(mut self, on: bool) -> Self {
        self.dynamic_decode = on;
        self
    }
    /// Supply config + weights directly (tests/benches; no GGUF on disk).
    pub fn inline_weights(mut self, cfg: Qwen35Config, weights: Qwen35Weights) -> Self {
        self.inline_weights = Some((cfg, weights));
        self
    }

    /// Compile hidden-state prefill graphs without mmproj (Gepard / custom embed prefill).
    pub fn hidden_prefill(mut self, on: bool) -> Self {
        self.hidden_prefill = on;
        self
    }

    /// Always feed decode `inputs_embeds` from the host (required for custom embeddings).
    pub fn force_host_embed(mut self, on: bool) -> Self {
        self.force_host_embed = on;
        self
    }

    /// Skip warming decode buckets / predict graph at build time.
    pub fn skip_warm(mut self, on: bool) -> Self {
        self.skip_warm = on;
        self
    }

    /// Load vision encoder weights from an mmproj GGUF (enables multimodal prefill).
    pub fn mmproj(mut self, path: impl Into<PathBuf>) -> Self {
        self.mmproj = Some(path.into());
        self
    }

    /// Do not auto-attach HF vision weights from the model dir (text-only).
    pub fn skip_auto_mmproj(mut self, on: bool) -> Self {
        self.skip_auto_mmproj = on;
        self
    }

    /// Supply mmproj config + weights directly (tests; no mmproj GGUF on disk).
    pub fn inline_mmproj(
        mut self,
        cfg: crate::vision::MmProjConfig,
        weights: crate::vision::MmProjWeights,
    ) -> Self {
        self.inline_mmproj = Some((cfg, weights));
        self
    }

    /// Override tier-1 compile profiles (skips `qwen35.rlx.toml` discovery).
    pub fn with_compile_profiles(
        mut self,
        prefill: CompileProfile,
        decode: CompileProfile,
    ) -> Self {
        self.prefill_profile = Some(prefill);
        self.decode_profile = Some(decode);
        self
    }

    /// Enable MoE expert offload (TIDE). Caps GPU-resident experts per layer; remainder on host.
    pub fn max_gpu_experts_per_layer(mut self, n: usize) -> Self {
        self.max_gpu_experts_per_layer = Some(n);
        self
    }

    /// Memory budget for automatic expert cap (macOS: unified RAM when unset).
    pub fn moe_memory_budget_bytes(mut self, bytes: usize) -> Self {
        self.moe_memory_budget_bytes = Some(bytes);
        self
    }

    /// Refresh expert GPU set every N decode steps (TIDE `jump_steps` for AR).
    pub fn expert_refresh_every_decode_steps(mut self, n: usize) -> Self {
        self.expert_refresh_every_decode_steps = Some(n);
        self.jump_steps = Some(n);
        self
    }

    /// TIDE `jump_steps` (τ): refresh expert placement every N denoise/decode steps.
    pub fn jump_steps(mut self, n: usize) -> Self {
        self.jump_steps = Some(n);
        self.expert_refresh_every_decode_steps = Some(n);
        self
    }

    /// TIDE `reserve_vram_gb` for GPU expert budget sizing (default 1.5).
    pub fn reserve_vram_gb(mut self, gb: f64) -> Self {
        self.reserve_vram_gb = Some(gb);
        self
    }

    /// TIDE `collect_stats` — aggregate token/compute counters per forward.
    pub fn moe_collect_stats(mut self, on: bool) -> Self {
        self.moe_collect_stats = on;
        self
    }

    /// TIDE `enable_predictive_expert_offload(max_gpu_experts_per_layer=…)`.
    pub fn enable_predictive_expert_offload(mut self, max_gpu_experts_per_layer: usize) -> Self {
        self.max_gpu_experts_per_layer = Some(max_gpu_experts_per_layer);
        self
    }

    pub fn build(self) -> Result<Qwen35Runner> {
        let device = self.device.unwrap_or(Device::Cpu);
        let max_seq = self.max_seq.unwrap_or(128);
        let prefill_seq = self.prefill_seq.unwrap_or(max_seq).clamp(1, max_seq);
        if self.prefill_seq.is_some() && prefill_seq < max_seq {
            eprintln!("[qwen35] prefill_seq={prefill_seq} (decode max_seq={max_seq})");
        }
        let batch = self.batch.unwrap_or(1);
        if batch == 0 {
            bail!("qwen35: batch must be >= 1");
        }

        // PLAN.md M1: safetensors load path for HauhauCS catalog rows
        // (and any other Qwen3.5 / 3.6 HF safetensors checkpoint,
        // including nested multimodal Fara / Qwen3.5-VL model dirs).
        // When the caller picks `JsonFile(p)` or `Explicit(cfg)`, we
        // detect a safetensors weights path, parse the HF config (or
        // use the explicit one), open the file via the weight registry,
        // wrap it in `HfTranslatingLoader` so GGUF-named lookups
        // succeed against HF tensor names, and run the standard
        // `Qwen35Weights::from_loader` drain.
        if let Some(src) = self.config.as_ref()
            && !matches!(src, Qwen35ConfigSource::Embedded)
        {
            let weights_path = self
                .weights
                .as_ref()
                .ok_or_else(|| anyhow!("weights path required (call .weights(...))"))?
                .clone();
            if self.packed_weights == Some(true) {
                bail!("qwen35: packed_weights requires GGUF; safetensors path is dequant-only");
            }
            let (model_dir, resolved, mut mmap_loader) = load_hf_mmap_loader(&weights_path)?;
            let cfg = match src {
                Qwen35ConfigSource::Embedded => unreachable!(),
                Qwen35ConfigSource::JsonFile(p) => Qwen35Config::from_hf_config_json(p)
                    .with_context(|| format!("qwen35: parse HF config {p:?}"))?,
                Qwen35ConfigSource::Explicit(cfg) => cfg.clone(),
            };
            if self.enable_mtp && cfg.nextn_predict_layers == 0 {
                bail!(
                    "qwen35: enable_mtp(true) but config has \
                     nextn_predict_layers=0 (no MTP heads to wire)"
                );
            }
            validate_device(&cfg, device, false)?;

            // Auto-attach HF vision tower when present and the caller
            // did not already supply mmproj / inline_mmproj.
            let mut inline_mmproj = self.inline_mmproj.clone();
            let mmproj_path = self.mmproj.clone();
            if !self.skip_auto_mmproj && mmproj_path.is_none() && inline_mmproj.is_none() {
                let cfg_path = match src {
                    Qwen35ConfigSource::JsonFile(p) => Some(p.clone()),
                    _ => {
                        let p = model_dir.join("config.json");
                        p.is_file().then_some(p)
                    }
                };
                if let Some(cp) = cfg_path.as_ref()
                    && let Ok(vcfg) = crate::vision::MmProjConfig::from_hf_config_json(cp)
                {
                    let has_visual = mmap_loader
                        .remaining_keys()
                        .iter()
                        .any(|k| k.starts_with("model.visual."));
                    if has_visual {
                        let t_vis = std::time::Instant::now();
                        let vweights =
                            crate::vision::MmProjWeights::from_hf_visual(&vcfg, &mut *mmap_loader)
                                .with_context(|| {
                                    format!("qwen35: load HF vision from {}", model_dir.display())
                                })?;
                        eprintln!(
                            "[qwen35] read HF vision weights in {:.2?} \
                             (layers={}, n_embd={}, llm_hidden={})",
                            t_vis.elapsed(),
                            vcfg.n_layer,
                            vcfg.n_embd,
                            vcfg.llm_hidden_size,
                        );
                        inline_mmproj = Some((vcfg, vweights));
                    }
                }
            }

            let mut loader = rlx_core::HfTranslatingLoader::new(mmap_loader);
            let t = std::time::Instant::now();
            let weights = Qwen35Weights::from_loader(&mut loader, &cfg)?;
            eprintln!(
                "[qwen35] read safetensors weights in {:.2?} \
                 (layers={}, hidden={}) [mmap-on-take]",
                t.elapsed(),
                cfg.num_hidden_layers,
                cfg.hidden_size,
            );
            return finish_build(
                cfg,
                weights,
                resolved,
                None,
                device,
                max_seq,
                prefill_seq,
                batch,
                self.enable_mtp,
                self.last_logits_only,
                self.runtime_mrope,
                self.mrope_section_positions,
                self.bucketed_decode,
                self.mtp_logits_path,
                self.fast_mtp,
                self.fast_greedy_lm_head.unwrap_or(true),
                self.aot_cache_dir.clone(),
                self.dynamic_prefill,
                self.dynamic_decode,
                mmproj_path,
                inline_mmproj,
                self.prefill_profile,
                self.decode_profile,
                self.max_gpu_experts_per_layer,
                self.moe_memory_budget_bytes,
                self.jump_steps.or(self.expert_refresh_every_decode_steps),
                self.reserve_vram_gb.unwrap_or(1.5),
                self.moe_collect_stats,
                self.hidden_prefill,
                self.force_host_embed,
                self.skip_warm,
            );
        }

        if let Some((cfg, weights)) = self.inline_weights {
            if self.packed_weights == Some(true) {
                bail!("qwen35: inline_weights and packed_weights are mutually exclusive");
            }
            if self.enable_mtp && cfg.nextn_predict_layers == 0 {
                bail!(
                    "qwen35: enable_mtp(true) but config has \
                     nextn_predict_layers=0 (no MTP heads to wire)"
                );
            }
            if self.mmproj.is_some() && self.inline_mmproj.is_some() {
                bail!("qwen35: mmproj and inline_mmproj are mutually exclusive");
            }
            validate_device(&cfg, device, false)?;
            return finish_build(
                cfg,
                weights,
                PathBuf::new(),
                None,
                device,
                max_seq,
                prefill_seq,
                batch,
                self.enable_mtp,
                self.last_logits_only,
                self.runtime_mrope,
                self.mrope_section_positions,
                self.bucketed_decode,
                self.mtp_logits_path,
                self.fast_mtp,
                self.fast_greedy_lm_head.unwrap_or(true),
                self.aot_cache_dir.clone(),
                self.dynamic_prefill,
                self.dynamic_decode,
                self.mmproj.clone(),
                self.inline_mmproj,
                self.prefill_profile,
                self.decode_profile,
                self.max_gpu_experts_per_layer,
                self.moe_memory_budget_bytes,
                self.jump_steps.or(self.expert_refresh_every_decode_steps),
                self.reserve_vram_gb.unwrap_or(1.5),
                self.moe_collect_stats,
                self.hidden_prefill,
                self.force_host_embed,
                self.skip_warm,
            );
        }

        let weights_path = resolve_weights_file(
            &self
                .weights
                .ok_or_else(|| anyhow!("weights path required (call .weights(...))"))?,
        )?;
        let _t_total = Instant::now();
        let t = Instant::now();
        let raw = assert_gguf_family(&weights_path, GgufModelFamily::Qwen35)?;
        let mut loader = GgufLoader::from_file(
            weights_path
                .to_str()
                .ok_or_else(|| anyhow!("non-utf8 weights path"))?,
        )?;
        loader.include_mtp(true);
        let cfg = Qwen35Config::from_gguf(&raw)?;
        eprintln!(
            "[qwen35] loaded GGUF metadata in {:.2?} \
             (layers={}, hidden={}, ssm_state={})",
            t.elapsed(),
            cfg.num_hidden_layers,
            cfg.hidden_size,
            cfg.ssm_state_size,
        );

        if self.enable_mtp && cfg.nextn_predict_layers == 0 {
            bail!(
                "qwen35: enable_mtp(true) but the file has \
                 nextn_predict_layers=0 (no MTP heads to wire)"
            );
        }
        // Resolve auto-default. Llama.cpp keeps K-quant tensors packed
        // in memory and dequantises per block inside the matmul kernel —
        // it never materialises a dense F32 weight matrix. Mirror that:
        // when *any* tensor in the GGUF is a K-quant block format
        // (Q2_K..Q8_K), force the packed path. Otherwise fall back to
        // the size heuristic (≥ 256 MB → packed) for legacy quant
        // formats. Explicit `.packed_weights(_)` overrides.
        let packed = self.packed_weights.unwrap_or_else(|| {
            if raw.tensors.values().any(|t| {
                matches!(
                    t.dtype,
                    rlx_gguf::GgmlType::Q2K
                        | rlx_gguf::GgmlType::Q3K
                        | rlx_gguf::GgmlType::Q4K
                        | rlx_gguf::GgmlType::Q5K
                        | rlx_gguf::GgmlType::Q6K
                        | rlx_gguf::GgmlType::Q8K
                        | rlx_gguf::GgmlType::Q1_0
                        | rlx_gguf::GgmlType::Q2_0
                )
            }) {
                return true;
            }
            std::fs::metadata(&weights_path)
                .ok()
                .map(|m| m.len() >= 256 * 1024 * 1024)
                .unwrap_or(false)
        });
        // Packed GGUF stays on the requested device by default (see
        // `packed_gguf_execution_device`). Opt into CPU with
        // `RLX_PACKED_GGUF_{WGPU,VULKAN,COREML}_HOST=1`.
        let device = if packed {
            let redirected = rlx_core::flow_bridge::packed_gguf_execution_device(device);
            if redirected != device {
                eprintln!(
                    "[qwen35] packed GGUF: redirecting device {device:?} → {redirected:?} \
                     (host override via RLX_PACKED_GGUF_*_HOST=1)"
                );
            }
            redirected
        } else {
            device
        };
        validate_device(&cfg, device, packed)?;

        let t = Instant::now();
        let weights = if packed {
            Qwen35Weights::from_loader_packed(&mut loader, &cfg)?
        } else {
            Qwen35Weights::from_loader(&mut loader, &cfg)?
        };
        eprintln!(
            "[qwen35] read weights ({}) in {:.2?}",
            if packed { "packed" } else { "F32" },
            t.elapsed(),
        );

        finish_build(
            cfg,
            weights,
            weights_path,
            Some(loader),
            device,
            max_seq,
            prefill_seq,
            batch,
            self.enable_mtp,
            self.last_logits_only,
            self.runtime_mrope,
            self.mrope_section_positions,
            self.bucketed_decode,
            self.mtp_logits_path,
            self.fast_mtp,
            self.fast_greedy_lm_head.unwrap_or_else(|| {
                // Env override for Metal↔CUDA tap compares.
                match rlx_ir::env::var("RLX_QWEN35_FAST_GREEDY_LM").as_deref() {
                    Some("0") | Some("false") | Some("off") => return false,
                    Some("1") | Some("true") | Some("on") => return true,
                    _ => {}
                }
                // Packed GPU/accelerator: host tied-head Q1 vocab scan is far
                // slower than a fused on-device DequantMatMul + argmax (Metal
                // host path was ~1–2 tok/s vs Prism ~26 on M4 Pro). Keep host
                // fast-greedy for CPU / CoreML / MLX.
                //
                // MLX: in-graph packed LM head (tied Q4_K embd ≈ vocab×hidden)
                // currently yields all-zero logits on large GGUFs (trunk layers
                // match; only the final DequantMatMul collapses). Use host
                // tied-head until that path is fixed.
                !(packed
                    && matches!(
                        device,
                        Device::Cuda | Device::Rocm | Device::Gpu | Device::Vulkan | Device::Metal
                    ))
            }),
            self.aot_cache_dir.clone(),
            self.dynamic_prefill,
            self.dynamic_decode,
            self.mmproj.clone(),
            self.inline_mmproj,
            self.prefill_profile,
            self.decode_profile,
            self.max_gpu_experts_per_layer,
            self.moe_memory_budget_bytes,
            self.jump_steps.or(self.expert_refresh_every_decode_steps),
            self.reserve_vram_gb.unwrap_or(1.5),
            self.moe_collect_stats,
            self.hidden_prefill,
            self.force_host_embed,
            self.skip_warm,
        )
    }
}

fn make_qwen35_dyn_cache(
    device: Device,
    capacity: usize,
    aot_cache_dir: Option<&std::path::Path>,
) -> Qwen35CompileCache {
    if let Some(dir) = aot_cache_dir {
        Qwen35CompileCache::with_aot(device, capacity, dir)
    } else {
        Qwen35CompileCache::new(device, capacity)
    }
}

fn trunk_has_dense_f32_mats(weights: &Qwen35Weights) -> bool {
    weights.has_dense_f32_projections()
}

/// Drop host F32 projections after param extract / Metal upload (default on
/// for **dense HF** checkpoints on Metal/MLX that can remmap from a model dir).
/// Packed GGUF keeps a few host-F32 side projections (`ssm_alpha`, …) that later
/// HIR rebuilds still need — auto-release is off when any packed mat remains.
/// Override with `RLX_QWEN35_RELEASE_HOST_WEIGHTS=0|1`.
fn release_host_dense_enabled(device: Device, weights: &Qwen35Weights) -> bool {
    match rlx_ir::env::var("RLX_QWEN35_RELEASE_HOST_WEIGHTS").as_deref() {
        Some("0") | Some("false") | Some("off") => false,
        Some("1") | Some("true") | Some("on") => true,
        _ => {
            weights.has_dense_f32_projections()
                && !weights.has_packed_projections()
                && matches!(device, Device::Metal | Device::Mlx)
        }
    }
}

/// Static prefill-cache compile via tier-0 [`BuiltModel`] + [`Qwen35CompileCache`].
fn compile_static_prefill_cache(
    cfg: &Qwen35Config,
    weights: Arc<Qwen35Weights>,
    batch: usize,
    max_seq: usize,
    device: Device,
    prefill_profile: &CompileProfile,
    runtime_mrope: bool,
    enable_mtp_head: bool,
    fast_mtp: bool,
    fast_greedy_lm_head: bool,
    aot_cache_dir: Option<&std::path::Path>,
) -> Result<(
    rlx_runtime::CompiledGraph,
    HashMap<String, Vec<f32>>,
    PackedParams,
)> {
    let mut flow_opts = Qwen35PrefillCacheOpts::static_cache(batch, max_seq);
    flow_opts.with_lm_head = !fast_greedy_lm_head;
    flow_opts.runtime_mrope = runtime_mrope;
    flow_opts.enable_mtp_head = enable_mtp_head;
    flow_opts.fast_mtp = fast_mtp;
    flow_opts.fast_greedy_lm_head = fast_greedy_lm_head;
    flow_opts.profile = Some(prefill_profile.clone());

    let t_build = Instant::now();
    let (built, packed) = build_qwen35_prefill_cache_built(cfg, weights, &flow_opts)?;
    let build_s = t_build.elapsed();
    // (c)+(d) low-mem: the static compiled arena already embeds the F32
    // params (via `compile_built` below); the returned copy is only used
    // for *dynamic* prefill re-specialization, which the static path never
    // takes. Skipping the clone saves ~5 GB at compile AND avoids retaining
    // a redundant embed copy for the runner's lifetime.
    let params = if rlx_core::gguf_support::low_mem_compile() {
        HashMap::new()
    } else {
        built.params().clone()
    };
    let config = prefill_config(batch, max_seq);
    // Packed GGUF: skip fusion (same as llama32/qwen3) — fusion into F32 RMS
    // assumptions skews K-quant and burns minutes of CUDA compile time.
    let compile_opts = if !packed.is_empty() {
        compile_options_for_packed_gguf_prefill_with_profile(prefill_profile, device)
    } else {
        compile_options_from_profile(prefill_profile, device, KernelDispatchConfig::default())
    };

    let mut cache = match aot_cache_dir {
        Some(dir) => Qwen35CompileCache::with_aot(device, 1, dir),
        None => Qwen35CompileCache::new(device, 1),
    };
    let mut config = config;
    if aot_cache_dir.is_some() {
        config = config.with_compilation_mode(CompilationMode::Aot);
    }
    let built = built.with_execution_config(&config);
    let t_compile = Instant::now();
    let compiled = cache.compile_built(built, &config, &compile_opts)?;
    let compile_s = t_compile.elapsed();
    eprintln!(
        "[qwen35] prefill-cache stages: flow_build={build_s:.2?}, hir+backend={compile_s:.2?} (params={}, packed={})",
        params.len(),
        packed.len()
    );
    Ok((compiled, params, packed))
}

/// Open a checkpoint directory as either an mlx-community affine-quantized
/// loader (`config.json` `quantization` block → [`rlx_core::weight_loader::MlxLoader`],
/// which dequantizes packed 4/8-bit affine weights → F32 on take) or a plain
/// HF safetensors mmap loader. Both satisfy [`WeightLoader`], so downstream
/// `HfTranslatingLoader` + `Qwen35Weights::from_loader` are unchanged.
fn open_dir_loader(dir: &Path) -> Result<Box<dyn WeightLoader>> {
    if rlx_core::weight_registry::dir_is_mlx_quant(dir) {
        let path = dir
            .to_str()
            .ok_or_else(|| anyhow!("qwen35: non-UTF8 mlx weights dir {dir:?}"))?;
        let loader = rlx_core::weight_loader::MlxLoader::open(path)
            .with_context(|| format!("qwen35: open mlx-affine dir {dir:?}"))?;
        return Ok(Box::new(loader));
    }
    let loader = rlx_core::SafetensorsMmapLoader::open(dir)
        .with_context(|| format!("qwen35: open safetensors dir {dir:?}"))?;
    Ok(Box::new(loader))
}

#[allow(clippy::too_many_arguments)]
/// Open a HuggingFace safetensors or mlx-community checkpoint via mmap
/// (F32 on take only; mlx-affine weights are dequantized on take).
fn load_hf_mmap_loader(weights_path: &Path) -> Result<(PathBuf, PathBuf, Box<dyn WeightLoader>)> {
    if weights_path.is_dir() {
        let loader = open_dir_loader(weights_path)?;
        return Ok((
            weights_path.to_path_buf(),
            weights_path.to_path_buf(),
            loader,
        ));
    }
    let resolved = resolve_weights_file(weights_path)?;
    let ext = resolved
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "gguf" {
        bail!(
            "qwen35: non-Embedded config source supplied with a GGUF weights file at \
             {resolved:?} — drop the config source (use the default Embedded) so the GGUF \
             metadata is the source of truth"
        );
    }
    let parent = resolved
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    // Sharded sibling layout: prefer loading the whole directory when an
    // index is present next to the resolved shard.
    if parent.join("model.safetensors.index.json").is_file() || weights_path.is_file() {
        let dir = if parent.join("model.safetensors.index.json").is_file() {
            parent.clone()
        } else {
            // Single-file: open the parent if it only has this shard, else
            // fall back to a temp dir view via index_from_dir on parent.
            parent.clone()
        };
        let loader = open_dir_loader(&dir)?;
        return Ok((dir, resolved, loader));
    }
    let loader = open_dir_loader(&parent)?;
    Ok((parent, resolved, loader))
}

fn finish_build(
    cfg: Qwen35Config,
    weights: Qwen35Weights,
    weights_path: PathBuf,
    gguf_loader: Option<GgufLoader>,
    device: Device,
    max_seq: usize,
    prefill_seq: usize,
    batch: usize,
    enable_mtp: bool,
    last_logits_only: bool,
    runtime_mrope: bool,
    mrope_section_positions: Option<Vec<[usize; 4]>>,
    bucketed_decode: Option<bool>,
    mtp_logits_path: bool,
    fast_mtp: bool,
    fast_greedy_lm_head: bool,
    aot_cache_dir: Option<PathBuf>,
    dynamic_prefill: bool,
    dynamic_decode: bool,
    mmproj_path: Option<PathBuf>,
    inline_mmproj: Option<(MmProjConfig, MmProjWeights)>,
    prefill_profile_override: Option<CompileProfile>,
    decode_profile_override: Option<CompileProfile>,
    max_gpu_experts_per_layer: Option<usize>,
    moe_memory_budget_bytes: Option<usize>,
    jump_steps: Option<usize>,
    reserve_vram_gb: f64,
    moe_collect_stats: bool,
    hidden_prefill: bool,
    force_host_embed: bool,
    skip_warm: bool,
) -> Result<Qwen35Runner> {
    let weights = Arc::new(weights);
    if force_host_embed {
        // SAFETY: single-threaded runner build; env read at decode HIR compile time.
        unsafe { std::env::set_var("RLX_QWEN35_HOST_EMBED", "1") };
    }
    // Auto-enable the low-mem compile/upload path for large models: it skips the
    // redundant F32 param clone, streams packed weights straight from the mmap
    // instead of a multi-GB owned cache copy, and drops the decode-bucket warm —
    // roughly halving peak RSS (Qwen3.6-27B: ~33 GB → ~18–24 GB) so the model
    // survives concurrent memory pressure. Explicit `RLX_LOW_MEM_COMPILE` wins.
    if cfg.hidden_size >= 4096 && std::env::var("RLX_LOW_MEM_COMPILE").is_err() {
        // SAFETY: single-threaded runner build; env read at compile time below.
        unsafe { std::env::set_var("RLX_LOW_MEM_COMPILE", "1") };
        eprintln!(
            "[qwen35] large model (hidden={}): auto-enabling low-mem compile \
             (lower peak RSS; set RLX_LOW_MEM_COMPILE=0 to disable)",
            cfg.hidden_size
        );
    }
    let host_embed =
        force_host_embed || crate::flow::host_embed_enabled_for_bytes(weights.token_embd.len() * 4);
    let prefill_profile = prefill_profile_override.unwrap_or_else(|| {
        if weights_path.as_os_str().is_empty() {
            qwen35_profile_default(false)
        } else {
            qwen35_profile_near_weights(&weights_path, false)
        }
    });
    let decode_profile = decode_profile_override.unwrap_or_else(|| {
        if weights_path.as_os_str().is_empty() {
            qwen35_profile_default(true)
        } else {
            qwen35_profile_near_weights(&weights_path, true)
        }
    });

    if fast_mtp && !mtp_logits_path && !enable_mtp {
        bail!("qwen35: fast_mtp requires enable_mtp(true) or mtp_logits_path(true)");
    }
    if mtp_logits_path && !enable_mtp {
        bail!("qwen35: mtp_logits_path requires enable_mtp(true)");
    }

    if dynamic_prefill && batch != 1 {
        bail!("qwen35: dynamic_prefill requires batch=1");
    }
    if dynamic_decode && batch != 1 {
        bail!("qwen35: dynamic_decode requires batch=1");
    }
    if dynamic_decode && bucketed_decode.unwrap_or(true) {
        eprintln!("[qwen35] dynamic_decode enabled — disabling bucketed decode cache");
    }
    let bucketed_decode = if dynamic_decode {
        false
    } else {
        match rlx_ir::env::var("RLX_QWEN35_BUCKETED_DECODE").as_deref() {
            Some("0") | Some("false") | Some("off") => false,
            Some("1") | Some("true") | Some("on") => true,
            _ => bucketed_decode.unwrap_or(true),
        }
    };

    let vision_encoder = if let Some(ref path) = mmproj_path {
        Some(load_vision_encoder(
            path.to_str()
                .ok_or_else(|| anyhow!("non-utf8 mmproj path"))?,
            224,
            224,
        )?)
    } else if let Some((vcfg, vweights)) = inline_mmproj {
        // Placeholder compile size — `encode_rgb` recompiles for the real
        // smart-resized dims. Must satisfy `patch_size * 2` (HF Fara uses 16→32).
        let step = vcfg.patch_size.saturating_mul(2).max(2);
        let side = 224usize.next_multiple_of(step).max(step);
        Some(Qwen35VisionEncoder::from_parts(vcfg, vweights, side, side)?)
    } else {
        None
    };
    let runtime_mrope = runtime_mrope || vision_encoder.is_some();
    if vision_encoder.is_some() && batch != 1 {
        bail!("qwen35: VLM (mmproj) requires batch=1");
    }
    if vision_encoder.is_some() && !dynamic_prefill {
        eprintln!("[qwen35] mmproj loaded — enabling dynamic prefill for variable multimodal seq");
    }
    let dynamic_prefill = dynamic_prefill || vision_encoder.is_some();

    let t = Instant::now();
    let aot = aot_cache_dir.as_ref().map(AotCache::new);
    let (cache_params, cache_packed, mut prefill_cache, prefill_dynamic_cache) = if dynamic_prefill
    {
        // Dense F32 (HF BF16→F32) already lives in `weights`. Eagerly
        // extracting the same tensors into `cache_params` doubles host RAM
        // (~17 GiB extra for Fara-4B). Defer extract to first specialize.
        let dense_f32 = trunk_has_dense_f32_mats(&weights);
        let (cache_params, cache_packed) = if dense_f32 {
            eprintln!(
                "[qwen35] dynamic prefill: deferring F32 param extract \
                 (avoids duplicating host weights)"
            );
            (HashMap::new(), HashMap::new())
        } else {
            let (_cache_hir, cache_params, cache_packed) =
                build_qwen35_prefill_cache_hir_dynamic_ext(
                    &cfg,
                    weights.clone(),
                    batch,
                    max_seq,
                    runtime_mrope,
                    mtp_logits_path,
                    fast_mtp,
                    fast_greedy_lm_head,
                )?;
            eprintln!(
                "[qwen35] built prefill-cache IR in {:.2?} (params={}, packed={})",
                t.elapsed(),
                cache_params.len(),
                cache_packed.len(),
            );
            (cache_params, cache_packed)
        };
        eprintln!("[qwen35] dynamic prefill template ready (compile on first prompt)");
        (
            cache_params,
            cache_packed,
            None,
            Some(make_qwen35_dyn_cache(device, 32, aot_cache_dir.as_deref())),
        )
    } else {
        let (compiled, cache_params, cache_packed) = compile_static_prefill_cache(
            &cfg,
            weights.clone(),
            batch,
            prefill_seq,
            device,
            &prefill_profile,
            runtime_mrope,
            mtp_logits_path,
            fast_mtp,
            fast_greedy_lm_head,
            aot_cache_dir.as_deref(),
        )?;
        eprintln!(
            "[qwen35] compiled prefill-cache via BuiltModel in {:.2?} (params={}, packed={}, seq={prefill_seq})",
            t.elapsed(),
            cache_params.len(),
            cache_packed.len(),
        );
        (cache_params, cache_packed, Some(compiled), None)
    };

    let (prefill_hidden_dynamic_cache, prefill_hidden_cache_params, prefill_hidden_cache_packed) =
        if vision_encoder.is_some() || hidden_prefill {
            let dense_f32 = trunk_has_dense_f32_mats(&weights);
            let (hidden_params, hidden_packed) = if dense_f32 {
                eprintln!(
                    "[qwen35] hidden prefill: deferring F32 param extract \
                     (avoids duplicating host weights)"
                );
                (HashMap::new(), HashMap::new())
            } else {
                let (hidden_hir, hidden_params, hidden_packed) =
                    build_qwen35_prefill_hidden_cache_hir_dynamic_ext(
                        &cfg,
                        weights.clone(),
                        batch,
                        max_seq,
                        runtime_mrope,
                        mtp_logits_path,
                        fast_mtp,
                        fast_greedy_lm_head,
                    )?;
                let _ = hidden_hir;
                (hidden_params, hidden_packed)
            };
            (
                Some(make_qwen35_dyn_cache(device, 32, aot_cache_dir.as_deref())),
                hidden_params,
                hidden_packed,
            )
        } else {
            (None, HashMap::new(), HashMap::new())
        };

    let t = Instant::now();
    if let Some(ref mut compiled) = prefill_cache {
        for (name, data) in &cache_params {
            compiled.set_param(name, data);
        }
    }

    let decode_compile_cache = if bucketed_decode {
        // Single bucket at exact `max_seq`. Decode pins to that upper bound
        // (avoids mid-stream recompiles under low-mem); a power-of-two ladder
        // would only pad further (e.g. 39→64) with no benefit under pinning.
        let max = max_seq.max(1) as u64;
        // One exact-`max_seq` bucket (not a collected integer range).
        #[allow(clippy::single_range_in_vec_init)]
        Some(BucketedCompileCache::new(device, vec![1..(max + 1)]))
    } else {
        None
    };
    let decode_dynamic_cache = if dynamic_decode {
        Some(make_qwen35_dyn_cache(device, 32, aot_cache_dir.as_deref()))
    } else {
        None
    };
    let (decode_dynamic_params, decode_dynamic_packed) = if dynamic_decode {
        let (_, p, packed) = build_qwen35_decode_hir_dynamic_ext(
            &cfg,
            weights.clone(),
            batch,
            max_seq,
            mtp_logits_path,
            fast_mtp,
            fast_greedy_lm_head,
        )?;
        (p, packed)
    } else {
        (HashMap::new(), HashMap::new())
    };

    if dynamic_decode {
        eprintln!("[qwen35] dynamic decode template ready (compile on first step)");
    }

    let moe_offload = build_moe_offload(
        &cfg,
        &weights,
        max_gpu_experts_per_layer,
        moe_memory_budget_bytes,
        jump_steps,
        reserve_vram_gb,
        moe_collect_stats,
    );
    let moe_store = if moe_offload.is_some() {
        build_moe_expert_store(&cfg, &weights).ok()
    } else {
        None
    };
    if let Some(ref mo) = moe_offload {
        eprintln!(
            "[qwen35] TIDE MoE offload: layers={} gpu_budget={}/{} jump_steps={} reserve_bytes={}",
            mo.num_layers(),
            mo.info.gpu_expert_budget_per_layer,
            mo.pools[0].num_experts(),
            mo.jump_steps,
            mo.info.reserve_bytes,
        );
    }

    let mut runner = Qwen35Runner {
        compiled: None,
        prefill_cache,
        prefill_dynamic_cache,
        prefill_hidden_dynamic_cache,
        prefill_cache_params: cache_params,
        prefill_cache_packed: cache_packed,
        _prefill_hidden_cache_params: prefill_hidden_cache_params,
        _prefill_hidden_cache_packed: prefill_hidden_cache_packed,
        decode_graphs: HashMap::new(),
        decode_compile_cache,
        decode_dynamic_cache,
        predict_hir_cache: None,
        decode_dynamic_params,
        decode_dynamic_packed,
        packed_bytes_cache: HashMap::new(),
        cfg,
        device,
        batch,
        max_seq,
        prefill_seq,
        last_logits_only,
        enable_mtp,
        mtp_logits_path,
        fast_mtp,
        fast_greedy_lm_head,
        host_embed,
        weights,
        weights_path,
        gguf_loader,
        decode_cache: None,
        runtime_mrope,
        mrope_section_positions,
        aot_cache: aot,
        dynamic_prefill,
        dynamic_decode,
        vision_encoder,
        mmproj_path,
        prefill_profile,
        decode_profile,
        moe_offload,
        moe_store,
        moe_refresh_step: 0,
    };

    if let Some(ref mut compiled) = runner.prefill_cache {
        upload_packed_opt(
            compiled,
            runner.gguf_loader.as_mut(),
            &runner.prefill_cache_packed,
            &mut runner.packed_bytes_cache,
        )?;
    }
    eprintln!(
        "[qwen35] uploaded prefill-cache {} F32 + {} packed params in {:.2?}",
        runner.prefill_cache_params.len(),
        runner.prefill_cache_packed.len(),
        t.elapsed(),
    );

    // (e) Decode warm policy for packed GGUF:
    // - Default: skip eager warm on low-mem / discrete-GPU packed builds
    //   (prefill already pins a large arena).
    // - Short max_seq (≤128) on Metal / MLX / CUDA: keep prefill and warm one
    //   decode bucket (~1s) so the first generate is not cold.
    // - `RLX_QWEN35_WARM_DECODE=1`: force warm; drops prefill first unless the
    //   short-context keep-prefill rule applies.
    let force_warm = matches!(std::env::var("RLX_QWEN35_WARM_DECODE").as_deref(), Ok("1"));
    let short_ctx = runner.max_seq <= 128
        && matches!(runner.device, Device::Metal | Device::Mlx | Device::Cuda);
    let skip_warm = skip_warm
        || rlx_core::gguf_support::low_mem_compile()
        || (runner.gguf_loader.is_some()
            && matches!(
                runner.device,
                Device::Cuda | Device::Rocm | Device::Gpu | Device::Vulkan
            ));
    let skip_warm = skip_warm && !short_ctx && !force_warm;
    if force_warm || short_ctx {
        let keep_prefill = short_ctx;
        if force_warm && !keep_prefill {
            runner.drop_prefill_cache();
        }
        runner.warm_decode_graphs()?;
        if rlx_ir::env::flag("RLX_QWEN35_WARM_PREDICT") {
            runner.warm_predict_graph()?;
        }
    } else if !skip_warm {
        runner.warm_decode_graphs()?;
        // Prefill-cache generate never touches the predict graph; warming it
        // is wasted work unless explicitly requested.
        if runner.prefill_cache.is_none() || rlx_ir::env::flag("RLX_QWEN35_WARM_PREDICT") {
            runner.warm_predict_graph()?;
        }
    }
    Ok(runner)
}

fn ensure_packed_cache(
    loader: &mut GgufLoader,
    packed: &PackedParams,
    cache: &mut HashMap<String, Arc<[u8]>>,
) -> Result<()> {
    for (loader_key, _, _) in packed.values() {
        if cache.contains_key(loader_key) {
            continue;
        }
        let bytes = loader
            .tensor_bytes_borrowed(loader_key)
            .ok_or_else(|| anyhow!("packed cache: {loader_key} bytes missing"))?;
        cache.insert(loader_key.clone(), Arc::from(bytes));
    }
    Ok(())
}

fn upload_packed_opt(
    compiled: &mut rlx_runtime::CompiledGraph,
    loader: Option<&mut GgufLoader>,
    packed: &PackedParams,
    cache: &mut HashMap<String, Arc<[u8]>>,
) -> Result<usize> {
    if packed.is_empty() {
        return Ok(0);
    }
    let loader = loader
        .ok_or_else(|| anyhow!("packed params require a GGUF loader (missing weights path)"))?;
    let mut total = 0usize;
    // (b) low-mem: stream each packed tensor straight from the (mmap'd)
    // loader into the arena and never retain a 3.5 GB owned copy in
    // `cache`. Decode-bucket re-uploads re-borrow from the same mmap
    // (page-cache cheap). Default path keeps the cache for upload speed.
    if rlx_core::gguf_support::low_mem_compile() {
        for (param_name, (loader_key, _scheme, _shape)) in packed {
            let bytes = loader
                .tensor_bytes_borrowed(loader_key)
                .ok_or_else(|| anyhow!("packed upload: bytes missing for {loader_key}"))?;
            total = total.saturating_add(bytes.len());
            compiled.set_param_typed(param_name, bytes, rlx_ir::DType::U8);
        }
        return Ok(total);
    }
    ensure_packed_cache(loader, packed, cache)?;
    for (param_name, (loader_key, _scheme, _shape)) in packed {
        let bytes = cache
            .get(loader_key)
            .ok_or_else(|| anyhow!("packed upload: cache miss for {loader_key}"))?;
        total = total.saturating_add(bytes.len());
        compiled.set_param_typed(param_name, bytes, rlx_ir::DType::U8);
    }
    Ok(total)
}

/// Byte size of packed tensors (for resident accounting after weight-buffer share).
fn packed_param_bytes(loader: Option<&GgufLoader>, packed: &PackedParams) -> usize {
    let Some(loader) = loader else {
        return 0;
    };
    packed
        .values()
        .filter_map(|(loader_key, _, _)| loader.tensor_bytes_borrowed(loader_key))
        .map(|b| b.len())
        .sum()
}

#[allow(dead_code)]
fn upload_decode_packed(
    weights_path: &std::path::Path,
    compiled: &mut rlx_runtime::CompiledGraph,
    packed: &PackedParams,
) -> Result<()> {
    if packed.is_empty() {
        return Ok(());
    }
    let path = weights_path
        .to_str()
        .filter(|p| !p.is_empty())
        .ok_or_else(|| anyhow!("packed decode params require a GGUF weights path"))?;
    let mut loader = GgufLoader::from_file(path)?;
    loader.include_mtp(true);
    upload_packed_opt(compiled, Some(&mut loader), packed, &mut HashMap::new()).map(|_| ())
}

pub struct Qwen35Runner {
    compiled: Option<rlx_runtime::CompiledGraph>,
    prefill_cache: Option<rlx_runtime::CompiledGraph>,
    prefill_dynamic_cache: Option<Qwen35CompileCache>,
    prefill_hidden_dynamic_cache: Option<Qwen35CompileCache>,
    prefill_cache_params: HashMap<String, Vec<f32>>,
    prefill_cache_packed: PackedParams,
    _prefill_hidden_cache_params: HashMap<String, Vec<f32>>,
    _prefill_hidden_cache_packed: PackedParams,
    decode_graphs: HashMap<usize, rlx_runtime::CompiledGraph>,
    decode_compile_cache: Option<BucketedCompileCache>,
    decode_dynamic_cache: Option<Qwen35CompileCache>,
    /// Predict / reprefill HIR (must not share a template with decode graphs).
    predict_hir_cache: Option<Qwen35CompileCache>,
    decode_dynamic_params: HashMap<String, Vec<f32>>,
    decode_dynamic_packed: PackedParams,
    packed_bytes_cache: HashMap<String, Arc<[u8]>>,
    cfg: Qwen35Config,
    device: Device,
    batch: usize,
    max_seq: usize,
    /// Compiled static prefill sequence length (≤ max_seq).
    prefill_seq: usize,
    last_logits_only: bool,
    enable_mtp: bool,
    mtp_logits_path: bool,
    fast_mtp: bool,
    fast_greedy_lm_head: bool,
    host_embed: bool,
    weights: Arc<Qwen35Weights>,
    weights_path: PathBuf,
    gguf_loader: Option<GgufLoader>,
    decode_cache: Option<Qwen35DecodeCache>,
    runtime_mrope: bool,
    mrope_section_positions: Option<Vec<[usize; 4]>>,
    aot_cache: Option<AotCache>,
    dynamic_prefill: bool,
    dynamic_decode: bool,
    vision_encoder: Option<Qwen35VisionEncoder>,
    mmproj_path: Option<PathBuf>,
    prefill_profile: CompileProfile,
    decode_profile: CompileProfile,
    /// TIDE-style per-layer expert offload.
    moe_offload: Option<MoeOffloadState>,
    /// Host expert stacks (migration source; F32 MoE only).
    moe_store: Option<MoeExpertStore>,
    /// Decode-step counter for MoE refresh scheduling (TIDE τ).
    moe_refresh_step: usize,
}

#[derive(Debug, Clone)]
pub struct Qwen35PrefillSeed {
    pub trunk_logits: Vec<f32>,
    pub mtp_logits: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct Qwen35PrefillOutput {
    pub logits: Vec<f32>,
    pub mtp_logits: Option<Vec<f32>>,
    pub vocab_size: usize,
}

impl Qwen35Runner {
    pub fn builder() -> Qwen35RunnerBuilder {
        Qwen35RunnerBuilder::default()
    }

    /// Drop the static prefill compiled graph to free its device arena.
    /// Free prefill device arena before decode when both arenas may not fit.
    ///
    /// Packed Bonsai-27B prefill is ~4 GiB; keeping it while compiling a
    /// decode bucket OOMs 16 GB CUDA cards. Dense HF F32 (Fara-4B ~17 GiB)
    /// has the same problem on unified memory. Prefill is rebuilt lazily on
    /// the next [`Self::prefill_seed_decode_cache`] call.
    pub fn drop_prefill_cache(&mut self) {
        let mut dropped = false;
        if self.prefill_cache.take().is_some() {
            dropped = true;
        }
        if let Some(cache) = self.prefill_dynamic_cache.as_mut() {
            let device = cache.device();
            *cache = Qwen35CompileCache::new(device, 32);
            dropped = true;
        }
        // VLM path specializes into this cache; leaving it live with decode
        // doubles dense F32 residency on unified memory (Fara-4B ≈ +17 GiB).
        if let Some(cache) = self.prefill_hidden_dynamic_cache.as_mut() {
            let device = cache.device();
            *cache = Qwen35CompileCache::new(device, 32);
            dropped = true;
        }
        if dropped {
            trim_accelerator_arena_pool(self.device);
            eprintln!("[qwen35] dropped prefill cache (free VRAM for decode)");
        }
    }

    /// Reload dense F32 projections when they were released after a prior
    /// Metal/MLX upload. Supports safetensors model dirs and packed GGUF files.
    fn ensure_dense_projections_resident(&mut self) -> Result<()> {
        if self.weights.has_dense_f32_projections() {
            return Ok(());
        }
        if !self.weights.has_cleared_f32_projection_shells() {
            return Ok(());
        }
        if self.weights_path.as_os_str().is_empty() {
            bail!(
                "qwen35: host F32 projections were released and cannot be \
                 reloaded (no weights path)"
            );
        }
        let emb = std::sync::Arc::clone(&self.weights.token_embd);
        let t = Instant::now();
        if self.weights_path.is_file()
            && self
                .weights_path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            let mut loader = GgufLoader::from_file(
                self.weights_path
                    .to_str()
                    .ok_or_else(|| anyhow!("non-utf8 weights path"))?,
            )?;
            loader.include_mtp(true);
            let mut fresh = Qwen35Weights::from_loader_packed(&mut loader, &self.cfg)?;
            fresh.token_embd = emb;
            self.weights = std::sync::Arc::new(fresh);
            // Keep the live GGUF loader in sync for packed byte uploads.
            self.gguf_loader = Some(loader);
            eprintln!(
                "[qwen35] reloaded host F32 projections from GGUF {} in {:.2?}",
                self.weights_path.display(),
                t.elapsed()
            );
            return Ok(());
        }
        if !self.weights_path.is_dir() {
            bail!(
                "qwen35: host F32 projections were released and cannot be \
                 reloaded (need a safetensors model dir or .gguf file)"
            );
        }
        let (_model_dir, _resolved, mmap_loader) = load_hf_mmap_loader(&self.weights_path)?;
        let mut loader = rlx_core::HfTranslatingLoader::new(mmap_loader);
        let mut fresh = Qwen35Weights::from_loader(&mut loader, &self.cfg)?;
        // Keep the live embedding table (host-embed / multimodal splice).
        fresh.token_embd = emb;
        self.weights = std::sync::Arc::new(fresh);
        eprintln!(
            "[qwen35] reloaded host F32 projections from {} in {:.2?}",
            self.weights_path.display(),
            t.elapsed()
        );
        Ok(())
    }

    fn clear_host_dense_projections(&mut self) {
        if !self.weights.has_dense_f32_projections() {
            return;
        }
        // Inline weights (`.inline_weights(...)`, e.g. Gepard) have no on-disk
        // source, so once released `ensure_dense_projections_resident` bails
        // ("host F32 projections were released and cannot be reloaded — no
        // weights path") and Metal/MLX decode crashes. Only release when there
        // IS a weights_path to remap from; otherwise keep them resident.
        if self.weights_path.as_os_str().is_empty() {
            return;
        }
        let Some(w) = std::sync::Arc::get_mut(&mut self.weights) else {
            eprintln!(
                "[qwen35] skip host F32 release (weights Arc still shared; \
                 avoid weights.clone() across extract)"
            );
            return;
        };
        w.clear_dense_f32_projections();
        eprintln!(
            "[qwen35] released host F32 projections (device keeps uploaded copy; \
             set RLX_QWEN35_RELEASE_HOST_WEIGHTS=0 to keep host)"
        );
    }

    fn ensure_static_prefill_cache(&mut self) -> Result<()> {
        if self.prefill_cache.is_some() || self.dynamic_prefill {
            return Ok(());
        }
        let aot_dir = self.aot_cache.as_ref().map(|c| c.root().to_path_buf());
        let (compiled, params, packed) = compile_static_prefill_cache(
            &self.cfg,
            self.weights.clone(),
            self.batch,
            self.prefill_seq,
            self.device,
            &self.prefill_profile,
            self.runtime_mrope,
            self.enable_mtp || self.mtp_logits_path,
            self.fast_mtp,
            self.fast_greedy_lm_head,
            aot_dir.as_deref(),
        )?;
        self.prefill_cache = Some(compiled);
        if !params.is_empty() {
            self.prefill_cache_params = params;
        }
        if !packed.is_empty() {
            self.prefill_cache_packed = packed;
        }
        if let Some(ref mut compiled) = self.prefill_cache {
            upload_packed_opt(
                compiled,
                self.gguf_loader.as_mut(),
                &self.prefill_cache_packed,
                &mut self.packed_bytes_cache,
            )?;
        }
        eprintln!("[qwen35] rebuilt static prefill cache");
        Ok(())
    }

    /// Whether an mmproj vision encoder (or its weights) is wired up,
    /// allowing [`Self::generate_multimodal`] to splice image embeddings
    /// into the prefill. Backs the `LmRunner::supports_multimodal` hook.
    pub fn has_mmproj(&self) -> bool {
        self.mmproj_path.is_some() || self.vision_encoder.is_some()
    }

    /// Apply runner AOT settings and build compile options for a dynamic specialize path.
    fn execution_config(&self, config: ModelExecutionConfig) -> ModelExecutionConfig {
        if self.aot_cache.is_some() {
            config.with_compilation_mode(CompilationMode::Aot)
        } else {
            config
        }
    }

    pub fn prefill_profile(&self) -> &CompileProfile {
        &self.prefill_profile
    }

    pub fn decode_profile(&self) -> &CompileProfile {
        &self.decode_profile
    }

    /// MoE offload state when enabled at build time.
    pub fn moe_offload(&self) -> Option<&MoeOffloadState> {
        self.moe_offload.as_ref()
    }

    /// TIDE `enable_predictive_expert_offload` return payload (when offload is active).
    pub fn predictive_offload_info(&self) -> Option<&rlx_llada2::tide::PredictiveOffloadInfo> {
        self.moe_offload.as_ref().map(|m| &m.info)
    }

    /// TIDE `get_offload_stats()` — pool promotions/demotions + last-forward residency (CPU).
    pub fn get_offload_stats(
        &self,
        residency: Option<&rlx_runtime::MoeResidencyStats>,
    ) -> rlx_llada2::tide::TideOffloadStats {
        self.moe_offload
            .as_ref()
            .map(|m| m.tide_offload_stats(residency))
            .unwrap_or_default()
    }

    pub fn jump_steps(&self) -> usize {
        self.moe_offload.as_ref().map(|m| m.jump_steps).unwrap_or(1)
    }

    pub fn predictive_offload_enabled(&self) -> bool {
        self.moe_offload
            .as_ref()
            .is_some_and(|m| m.predictive_enabled)
    }

    pub fn moe_offload_mut(&mut self) -> Option<&mut MoeOffloadState> {
        self.moe_offload.as_mut()
    }

    /// MoE refresh step index (TIDE `step` in block denoise / decode loop).
    pub fn moe_refresh_step(&self) -> usize {
        self.moe_refresh_step
    }

    /// Enable TopK capture on a compiled graph (CPU; call once after compile).
    pub fn enable_moe_topk_on(&self, compiled: &mut rlx_runtime::CompiledGraph) {
        if self.moe_offload.is_some() {
            compiled.enable_moe_topk_capture(self.cfg.num_experts);
        }
    }

    /// Push per-layer TIDE residency masks into the compiled graph.
    pub fn sync_moe_residency(&self, compiled: &mut rlx_runtime::CompiledGraph) {
        if let Some(mo) = &self.moe_offload {
            push_moe_residency(compiled, &mo.per_layer_resident_masks());
        }
    }

    #[allow(dead_code)]
    fn moe_prepare_forward(&self, compiled: &mut rlx_runtime::CompiledGraph) {
        self.bind_moe_host_weights();
        if self.moe_offload.is_some() {
            compiled.enable_moe_topk_capture(self.cfg.num_experts);
            self.sync_moe_residency(compiled);
        }
    }

    #[allow(dead_code)]
    fn moe_finish_forward(
        &mut self,
        compiled: &mut rlx_runtime::CompiledGraph,
        denoise_step: usize,
        is_prefill_block: bool,
    ) -> bool {
        let Some(layers) = compiled.take_moe_topk_capture() else {
            return false;
        };
        let store = self.moe_store.clone();
        let Some(mo) = self.moe_offload.as_mut() else {
            return false;
        };
        let refreshed = if let Some(store) = store.as_ref() {
            mo.refresh_from_capture_with_store(store, &layers, denoise_step, is_prefill_block)
        } else {
            mo.refresh_from_capture(&layers, denoise_step, is_prefill_block)
        };
        if refreshed {
            push_moe_residency(compiled, &mo.per_layer_resident_masks());
        }
        refreshed
    }

    /// Install per-expert host pointers for CPU GroupedMatMul fallback (TIDE).
    fn bind_moe_host_weights(&self) {
        if self.moe_offload.is_none() {
            rlx_cpu::moe_residency::bind_host_weights(None);
            return;
        }
        if let Some(store) = &self.moe_store {
            rlx_cpu::moe_residency::bind_host_weights(Some(moe_host_bind_from_store(store)));
        } else {
            rlx_cpu::moe_residency::bind_host_weights(None);
        }
    }

    /// After forward: refresh pools from captured TopK and update graph mask.
    pub fn moe_offload_after_forward(&mut self, compiled: &mut rlx_runtime::CompiledGraph) -> bool {
        let Some(mo) = self.moe_offload.as_mut() else {
            return false;
        };
        let Some(layers) = compiled.take_moe_topk_capture() else {
            return false;
        };
        let refreshed = mo.refresh_from_capture(&layers, self.moe_refresh_step, false);
        if refreshed {
            self.sync_moe_residency(compiled);
        }
        self.moe_refresh_step = self.moe_refresh_step.saturating_add(1);
        refreshed
    }

    /// Manual refresh from flat expert indices (single shared indices for all layers).
    pub fn moe_refresh_after_forward(&mut self, expert_idx: &[u32]) -> bool {
        let Some(mo) = self.moe_offload.as_mut() else {
            return false;
        };
        let refresh = mo.pools[0].should_refresh(
            rlx_runtime::MoEExecMode::Reuse,
            self.moe_refresh_step,
            false,
        );
        if refresh {
            for pool in &mut mo.pools {
                pool.refresh_from_indices(expert_idx);
            }
        }
        self.moe_refresh_step = self.moe_refresh_step.saturating_add(1);
        refresh
    }

    /// Override tier-1 profiles after build (e.g. tests).
    pub fn with_compile_profiles(
        mut self,
        prefill: CompileProfile,
        decode: CompileProfile,
    ) -> Self {
        self.prefill_profile = prefill;
        self.decode_profile = decode;
        self
    }

    fn profile_compile_options(&self, decode: bool) -> CompileOptions {
        let profile = if decode {
            &self.decode_profile
        } else {
            &self.prefill_profile
        };
        // Packed Q1_0 / K-quant: disable fusion for both prefill and decode.
        if self.gguf_loader.is_some() {
            compile_options_for_packed_gguf_prefill_with_profile(profile, self.device)
        } else {
            compile_options_from_profile(profile, self.device, KernelDispatchConfig::default())
        }
    }

    fn dyn_compile_options(&self, config: &ModelExecutionConfig) -> CompileOptions {
        let decode = matches!(config.preset, ExecutionPreset::Qwen35Decode);
        let mut opts = self.profile_compile_options(decode);
        opts.kernel_dispatch = config.component().kernel_dispatch;
        opts.dim_binding(config.dim_binding())
    }

    fn bucketed_decode_compile_options(&self) -> CompileOptions {
        self.profile_compile_options(true)
    }

    /// Compile a tier-0 prefill [`rlx_flow::BuiltModel`] through [`Qwen35CompileCache`].
    pub fn compile_prefill_built(
        &self,
        cache: &mut Qwen35CompileCache,
        built: rlx_flow::BuiltModel,
        batch: usize,
        seq: usize,
    ) -> Result<rlx_runtime::CompiledGraph> {
        let config = self.execution_config(prefill_config(batch, seq));
        let opts = self.dyn_compile_options(&config);
        cache.compile_built(built, &config, &opts)
    }

    pub fn cfg(&self) -> &Qwen35Config {
        &self.cfg
    }
    pub fn device(&self) -> Device {
        self.device
    }
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }
    pub fn lm_vocab_size(&self) -> usize {
        self.weights.lm_vocab_size(&self.cfg)
    }

    /// True when an mmproj vision encoder was loaded at build time.
    pub fn has_vision(&self) -> bool {
        self.vision_encoder.is_some()
    }

    /// Optional path to the mmproj GGUF (if configured).
    pub fn mmproj_path(&self) -> Option<&std::path::Path> {
        self.mmproj_path.as_deref()
    }

    fn effective_vocab(&self, graph_vocab: usize) -> usize {
        self.lm_vocab_size().min(graph_vocab)
    }

    fn compile_hir_for_config(
        &mut self,
        config: ModelExecutionConfig,
        aot_disk_key: &str,
        hir: rlx_ir::hir::HirModule,
    ) -> Result<rlx_runtime::CompiledGraph> {
        let config = self.execution_config(config);
        let opts = self.dyn_compile_options(&config);
        if let Some(aot) = self.aot_cache.as_ref() {
            return Ok(aot.compile_hir_cached(aot_disk_key, self.device, hir, &opts)?);
        }
        // Per-`past_seq` decode HIR is concrete (not symbolic). Sharing one
        // `ModelCompilePipeline` template across variants reuses the wrong graph.
        if config.preset == ExecutionPreset::Qwen35Decode {
            return Ok(Session::new(self.device).compile_hir_with(hir, &opts)?);
        }
        let cache = self
            .predict_hir_cache
            .get_or_insert_with(|| make_qwen35_dyn_cache(self.device, 64, None));
        let hir = hir;
        get_or_specialize_hir_with_options(cache, &config, || hir.clone(), &opts, |_| Ok(()))?;
        if self.device == Device::Cpu {
            let compiled = get_or_specialize_hir_with_options(
                cache,
                &config,
                || hir.clone(),
                &opts,
                |_| Ok(()),
            )?;
            return Ok(compiled.clone());
        }
        Ok(Session::new(self.device).compile_hir_with(hir, &opts)?)
    }

    fn lm_loader(&self) -> Option<&GgufLoader> {
        self.gguf_loader.as_ref()
    }

    fn argmax_batch_from_hidden(&self, hidden: &[f32]) -> Result<Vec<u32>> {
        crate::trace::log_lm_head_path("host_parallel_or_serial");
        let n_embd = self.cfg.hidden_size;
        let mut toks = Vec::with_capacity(self.batch);
        for b in 0..self.batch {
            let h = &hidden[b * n_embd..(b + 1) * n_embd];
            let (idx, _) = greedy_lm_head_argmax(&self.weights, &self.cfg, h, self.lm_loader())?;
            toks.push(idx);
        }
        Ok(toks)
    }

    fn sample_batch_from_hidden(&self, hidden: &[f32], opts: SampleOpts) -> Result<Vec<u32>> {
        let n_embd = self.cfg.hidden_size;
        let mut toks = Vec::with_capacity(self.batch);
        for b in 0..self.batch {
            let h = &hidden[b * n_embd..(b + 1) * n_embd];
            toks.push(sample_lm_head_from_hidden(
                &self.weights,
                &self.cfg,
                h,
                self.lm_loader(),
                opts,
            )?);
        }
        Ok(toks)
    }

    fn decode_step_trunk_raw(
        &mut self,
        cache: &mut Qwen35DecodeCache,
        tokens: &[u32],
        generated_per_row: &[usize],
        custom_embed: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        if self.dynamic_decode {
            return self.decode_step_dynamic_raw(cache, tokens, generated_per_row, custom_embed);
        }
        let past_seq = cache.past_seq;
        let (cos, sin) = self.mrope_decode_rope_at_past(past_seq);
        let use_bucket = self
            .decode_compile_cache
            .as_ref()
            .and_then(|c| c.bucket_for(past_seq as u64))
            .is_some();
        if use_bucket {
            self.decode_step_bucketed_raw(
                cache,
                tokens,
                generated_per_row,
                &cos,
                &sin,
                custom_embed,
            )
        } else {
            let feeds_owned = decode_step_feeds(
                &self.cfg,
                cache,
                tokens,
                &cos,
                &sin,
                None,
                generated_per_row,
                self.host_embed.then(|| self.weights.token_embd.as_ref()),
                custom_embed,
            )?;
            let feeds: Vec<(&str, &[f32])> = feeds_owned
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_slice()))
                .collect();
            if !self.decode_graphs.contains_key(&past_seq) {
                let (hir, params, packed) = build_qwen35_decode_hir_ext(
                    &self.cfg,
                    self.weights.clone(),
                    self.batch,
                    past_seq,
                    false,
                    self.mtp_logits_path,
                    self.fast_mtp,
                    self.fast_greedy_lm_head,
                )?;
                let mut compiled = self.compile_hir_for_config(
                    decode_config(self.batch, past_seq),
                    &format!("decode_{past_seq}"),
                    hir,
                )?;
                for (name, data) in &params {
                    compiled.set_param(name, data);
                }
                upload_packed_opt(
                    &mut compiled,
                    self.gguf_loader.as_mut(),
                    &packed,
                    &mut self.packed_bytes_cache,
                )?;
                self.decode_graphs.insert(past_seq, compiled);
            }
            let step = self.moe_refresh_step;
            let has_moe = self.moe_offload.is_some();
            let num_experts = self.cfg.num_experts;
            let moe_masks = self
                .moe_offload
                .as_ref()
                .map(|m| m.per_layer_resident_masks());
            self.bind_moe_host_weights();
            let outs = {
                let compiled = self.decode_graphs.get_mut(&past_seq).unwrap();
                if has_moe {
                    compiled.enable_moe_topk_capture(num_experts);
                    if let Some(layers) = &moe_masks {
                        push_moe_residency(compiled, layers);
                    }
                }
                compiled.run(&feeds)
            };
            if has_moe {
                let layers = {
                    let compiled = self.decode_graphs.get_mut(&past_seq).unwrap();
                    compiled.take_moe_topk_capture()
                };
                if let (Some(mo), Some(layers)) = (self.moe_offload.as_mut(), layers) {
                    let store = self.moe_store.as_ref();
                    let compiled = self.decode_graphs.get_mut(&past_seq).unwrap();
                    if refresh_moe_from_capture(mo, store, compiled, &layers, step, false) {
                        if let Some(store) = self.moe_store.as_ref() {
                            store.apply_to_compiled(compiled);
                        }
                    }
                }
            }
            self.moe_refresh_step = step.saturating_add(1);
            advance_cache_from_decode_outputs(
                &self.cfg,
                cache,
                outs,
                None,
                self.mtp_logits_path,
                false,
                self.fast_greedy_lm_head,
            )
        }
    }

    fn decode_step_dynamic_raw(
        &mut self,
        cache: &mut Qwen35DecodeCache,
        tokens: &[u32],
        generated_per_row: &[usize],
        custom_embed: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        let past_seq = cache.past_seq;
        let (cos, sin) = self.mrope_decode_rope_at_past(past_seq);
        let feeds_owned = decode_step_feeds(
            &self.cfg,
            cache,
            tokens,
            &cos,
            &sin,
            None,
            generated_per_row,
            self.host_embed.then(|| self.weights.token_embd.as_ref()),
            custom_embed,
        )?;
        let feeds: Vec<(&str, &[f32])> = feeds_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_slice()))
            .collect();

        let config = self.execution_config(decode_config(self.batch, past_seq));
        let compile_opts = self.dyn_compile_options(&config);
        let dyn_cache = self
            .decode_dynamic_cache
            .as_mut()
            .ok_or_else(|| anyhow!("dynamic decode without cache"))?;
        let cfg = self.cfg.clone();
        let weights = self.weights.clone();
        let max_seq = self.max_seq;
        let mtp_logits_path = self.mtp_logits_path;
        let fast_mtp = self.fast_mtp;
        let fast_greedy = self.fast_greedy_lm_head;
        let batch = self.batch;
        let decode_params = &self.decode_dynamic_params;
        let decode_packed = &self.decode_dynamic_packed;
        let gguf_loader = &mut self.gguf_loader;
        let packed_bytes_cache = &mut self.packed_bytes_cache;
        let compiled = get_or_specialize_hir_with_options(
            dyn_cache,
            &config,
            || {
                build_qwen35_decode_hir_dynamic_ext(
                    &cfg,
                    weights,
                    batch,
                    max_seq,
                    mtp_logits_path,
                    fast_mtp,
                    fast_greedy,
                )
                .expect("dynamic decode HIR")
                .0
            },
            &compile_opts,
            |c| {
                for (name, data) in decode_params {
                    c.set_param(name, data);
                }
                upload_packed_opt(c, gguf_loader.as_mut(), decode_packed, packed_bytes_cache)
                    .map(|_| ())
            },
        )?;
        let outs = compiled.run(&feeds);
        advance_cache_from_decode_outputs(
            &self.cfg,
            cache,
            outs,
            None,
            self.mtp_logits_path,
            false,
            self.fast_greedy_lm_head,
        )
    }

    fn trunk_to_logits(&self, trunk: Vec<f32>, is_hidden: bool) -> Result<Vec<f32>> {
        if !is_hidden {
            return Ok(trunk);
        }
        let n_embd = self.cfg.hidden_size;
        let vocab = self.lm_vocab_size();
        let mut logits = Vec::with_capacity(self.batch * vocab);
        for b in 0..self.batch {
            let h = &trunk[b * n_embd..(b + 1) * n_embd];
            logits.extend(lm_head_logits_row(
                &self.weights,
                &self.cfg,
                h,
                self.lm_loader(),
            )?);
        }
        Ok(logits)
    }

    /// Bucket lookup key for monotonic decode: prefer the largest rung.
    ///
    /// With `RLX_LOW_MEM_COMPILE` we skip eager warm, so using `past_seq` as
    /// the key climbed every power-of-two rung (multi-second Metal recompiles
    /// mid-stream). Pinning to the max bucket compiles once and pads masks to
    /// `upper` (already supported by `decode_step_feeds`).
    fn decode_bucket_key(&self, past_seq: u64) -> u64 {
        let max_upper = self
            .decode_compile_cache
            .as_ref()
            .and_then(|c| c.buckets().map(|r| r.end.saturating_sub(1)).max());
        match max_upper {
            Some(u) if u >= past_seq => u,
            _ => past_seq,
        }
    }

    /// Compile decode HIR for `key`'s bucket (if needed) and upload packed GGUF params once.
    ///
    /// After the first bucket upload, later rungs
    /// [`BucketedCompileCache::try_share_params_from_donor`] the Metal/CUDA
    /// weight buffer (no second ~3.9 GB Q1_0 copy). With
    /// `RLX_KV_CACHE_MAX_RESIDENT=1`, only one decode arena stays resident —
    /// share runs before peer eviction so weights are not re-uploaded.
    fn ensure_decode_bucket_compiled(&mut self, key: u64) -> Result<usize> {
        let key = self.decode_bucket_key(key);
        let upper = {
            let cache = self
                .decode_compile_cache
                .as_ref()
                .ok_or_else(|| anyhow!("bucketed decode without cache"))?;
            cache
                .bucket_upper_for_key(key)
                .ok_or_else(|| anyhow!("past_seq {key} outside decode buckets"))?
        };
        if self
            .decode_compile_cache
            .as_ref()
            .and_then(|c| c.compiled_for_upper(upper))
            .is_some()
        {
            return Ok(upper as usize);
        }

        self.ensure_dense_projections_resident()?;
        let release = release_host_dense_enabled(self.device, &self.weights);
        let decode_opts = self.bucketed_decode_compile_options();
        let cfg = self.cfg.clone();
        let batch = self.batch;
        let mtp_logits_path = self.mtp_logits_path;
        let fast_mtp = self.fast_mtp;
        let fast_greedy = self.fast_greedy_lm_head;
        let (hir, params, packed) = build_qwen35_decode_hir_ext(
            &cfg,
            std::sync::Arc::clone(&self.weights),
            batch,
            upper as usize,
            true,
            mtp_logits_path,
            fast_mtp,
            fast_greedy,
        )
        .context("qwen35 decode HIR")?;
        if release {
            self.clear_host_dense_projections();
        }
        let hir_cell = RefCell::new(Some(hir));
        let params_cell = RefCell::new(Some(params));
        let packed_slot = RefCell::new(Some(packed));
        let mut packed_nbytes = 0usize;
        let upper = {
            let cache_mut = self
                .decode_compile_cache
                .as_mut()
                .ok_or_else(|| anyhow!("bucketed decode without cache"))?;
            let (upper, _) = cache_mut
                .ensure_hir_with_params(
                    key,
                    |_upper| {
                        (
                            hir_cell
                                .borrow_mut()
                                .take()
                                .expect("decode HIR already taken"),
                            params_cell
                                .borrow_mut()
                                .take()
                                .expect("decode params already taken"),
                        )
                    },
                    &decode_opts,
                )
                .ok_or_else(|| anyhow!("past_seq {key} outside decode buckets"))?;
            if let Some(packed) = packed_slot.take() {
                if !packed.is_empty() {
                    if cache_mut.try_share_params_from_donor(upper) {
                        packed_nbytes = packed_param_bytes(self.gguf_loader.as_ref(), &packed);
                    } else {
                        let compiled = cache_mut
                            .compiled_for_key_mut(key)
                            .ok_or_else(|| anyhow!("decode bucket missing after ensure"))?;
                        packed_nbytes = upload_packed_opt(
                            compiled,
                            self.gguf_loader.as_mut(),
                            &packed,
                            &mut self.packed_bytes_cache,
                        )?;
                        if packed_nbytes > 0 {
                            cache_mut.set_weight_donor(upper);
                        }
                    }
                }
            }
            upper
        };
        if packed_nbytes > 0 {
            if let Some(cache) = self.decode_compile_cache.as_mut() {
                cache.note_resident_bytes(key, packed_nbytes);
            }
        }
        Ok(upper as usize)
    }

    /// Pre-compile the largest decode bucket (covers growth up to `max_seq`).
    ///
    /// Warming the full power-of-two ladder used to cost N× multi-minute CUDA
    /// compiles and OOM 16 GB cards. Monotonic decode only needs the rung that
    /// contains the current `past_seq`; the top rung covers the whole run once
    /// the prompt is longer than the previous power-of-two boundary.
    fn warm_decode_graphs(&mut self) -> Result<()> {
        let upper = match self.decode_compile_cache.as_ref() {
            Some(cache) => cache
                .buckets()
                .map(|r| (r.end - 1) as usize)
                .max()
                .unwrap_or(0),
            None => return Ok(()),
        };
        if upper == 0 {
            return Ok(());
        }
        let t = Instant::now();
        self.ensure_decode_bucket_compiled(upper as u64)?;
        eprintln!(
            "[qwen35] warmed decode bucket upper={upper} in {:.2?}",
            t.elapsed()
        );
        Ok(())
    }

    /// Pre-compile the predict (prefill logits) graph at build time.
    fn warm_predict_graph(&mut self) -> Result<()> {
        if self.compiled.is_some() {
            return Ok(());
        }
        let t = Instant::now();
        self.ensure_predict_compiled()?;
        eprintln!("[qwen35] warmed predict graph in {:.2?}", t.elapsed());
        Ok(())
    }

    /// Clear the decode KV / recurrent cache (e.g. before a fresh prefill in spec decode).
    pub fn reset_decode_cache(&mut self) {
        self.decode_cache = None;
    }

    /// Snapshot the current decode cache (for two-phase MTP draft propose).
    pub fn decode_cache_checkpoint(&self) -> Option<Qwen35DecodeCache> {
        self.decode_cache.clone()
    }

    /// Restore a decode cache snapshot (discards uncommitted draft steps).
    pub fn restore_decode_cache(&mut self, cache: Option<Qwen35DecodeCache>) {
        self.decode_cache = cache;
    }

    /// Advance the decode cache by `tokens` without returning logits (MTP commit path).
    pub fn commit_decode_tokens(&mut self, tokens: &[u32]) -> Result<()> {
        for &tok in tokens {
            let _ = self.decode_get_logits(tok)?;
        }
        Ok(())
    }

    fn ensure_predict_compiled(&mut self) -> Result<()> {
        if self.compiled.is_some() {
            return Ok(());
        }
        // RLX_QWEN35_DEBUG_LAYERS=1 makes the predict graph emit every
        // trunk layer's hidden state as an extra output (gathered at
        // last_token_idx → shape `[batch, n_embd]` per layer). Combined
        // with the per-output stats dump in `predict_logits_batch` this
        // is the bisection harness for "all-zero logits" symptoms: the
        // log shows which layer first emits zero/NaN. Requires
        // `last_logits_only=true` (the assertion is in the builder).
        let debug_layers = std::env::var("RLX_QWEN35_DEBUG_LAYERS")
            .map(|v| v == "1")
            .unwrap_or(false);
        self.ensure_dense_projections_resident()?;
        let t = Instant::now();
        // Predict graph must match the runner's LM-head mode: when
        // `fast_greedy_lm_head` is on, emit normed hidden and score on host
        // (see generate path). Hard-coding `with_lm_head=true` here made
        // `RLX_QWEN35_FAST_GREEDY_LM=1` a no-op for `predict_logits`.
        let with_lm_head = !self.fast_greedy_lm_head;
        let (hir, params, packed) = build_qwen35_hir_sized_ext(
            &self.cfg,
            (*self.weights).clone(),
            self.batch,
            self.max_seq,
            with_lm_head,
            self.last_logits_only,
            self.enable_mtp,
            false,
            None,
            self.runtime_mrope,
            self.fast_mtp,
            self.fast_greedy_lm_head,
            debug_layers,
        )?;
        eprintln!(
            "[qwen35] built predict IR (lazy) in {:.2?} (params={}, packed={})",
            t.elapsed(),
            params.len(),
            packed.len(),
        );
        let t = Instant::now();
        let mut compiled = self.compile_hir_for_config(
            prefill_config(self.batch, self.max_seq),
            "predict_logits",
            hir,
        )?;
        eprintln!(
            "[qwen35] compiled predict graph (lazy) in {:.2?}",
            t.elapsed()
        );
        let t = Instant::now();
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        if !packed.is_empty() {
            upload_packed_opt(
                &mut compiled,
                self.gguf_loader.as_mut(),
                &packed,
                &mut self.packed_bytes_cache,
            )?;
        }
        eprintln!(
            "[qwen35] uploaded predict {} F32 + {} packed params in {:.2?}",
            params.len(),
            packed.len(),
            t.elapsed(),
        );
        self.compiled = Some(compiled);
        Ok(())
    }

    pub fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Qwen35PrefillOutput> {
        let out = self
            .predict_logits_batch(&[prompt_ids.to_vec()])
            .map(|v| v.into_iter().next().unwrap())?;
        // Preflight guard: a degenerate all-zero (or all-equal) logits
        // tensor is the signature of a broken forward pass — a buggy
        // op writing zeros, a packed-K-quant dispatch mis-routed, an
        // arena slot being read from the wrong offset, etc. Fail fast
        // with a clear error rather than silently returning argmax =
        // vocab_size − 1.
        if !out.logits.is_empty() {
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for &v in &out.logits {
                if v.is_finite() {
                    if v < min {
                        min = v;
                    }
                    if v > max {
                        max = v;
                    }
                }
            }
            if (max - min).abs() < 1e-6 {
                bail!(
                    "qwen35: predict_logits returned degenerate output \
                     (min={min}, max={max}) — the forward pass produced \
                     all-equal logits, which indicates a broken op or a \
                     mis-routed weight tensor in the trunk. Re-run with \
                     RUST_LOG=debug to capture the offending layer."
                );
            }
        }
        Ok(out)
    }

    /// Prefill forward for `batch` prompts (must equal `self.batch()`).
    /// Each row may have a different length; all are zero-padded to
    /// `max_seq` in the compiled graph.
    pub fn predict_logits_batch(
        &mut self,
        batch_prompts: &[Vec<u32>],
    ) -> Result<Vec<Qwen35PrefillOutput>> {
        if batch_prompts.len() != self.batch {
            bail!(
                "qwen35: expected {} prompts (batch={}), got {}",
                self.batch,
                self.batch,
                batch_prompts.len()
            );
        }
        let max_prompt = batch_prompts.iter().map(|p| p.len()).max().unwrap_or(0);
        if max_prompt > self.max_seq {
            bail!(
                "qwen35: prompt length {max_prompt} exceeds compiled max_seq={}",
                self.max_seq
            );
        }
        let padded = pack_input_ids(batch_prompts, self.max_seq)?;
        let prompt_lens: Vec<usize> = batch_prompts.iter().map(|p| p.len()).collect();
        let last_idx = last_token_indices(&prompt_lens);

        // Match decode/prefill: when the predict graph uses host embed
        // (Fara BF16 / large vocab), gather rows here — do not leave
        // `inputs_embeds` unset (that yields all-zero logits).
        let n_embd = self.cfg.hidden_size;
        let host_embeds: Vec<f32> = if self.host_embed && !self.weights.token_embd.is_empty() {
            let tbl = &self.weights.token_embd;
            let mut v = vec![0f32; padded.len() * n_embd];
            for (pos, &id_f) in padded.iter().enumerate() {
                let src = (id_f as usize) * n_embd;
                if src + n_embd <= tbl.len() {
                    v[pos * n_embd..pos * n_embd + n_embd].copy_from_slice(&tbl[src..src + n_embd]);
                }
            }
            v
        } else {
            Vec::new()
        };

        let mut feeds: Vec<(&str, &[f32])> = vec![("input_ids", padded.as_slice())];
        if self.last_logits_only {
            feeds.push(("last_token_idx", last_idx.as_slice()));
        }
        if !host_embeds.is_empty() {
            feeds.push(("inputs_embeds", host_embeds.as_slice()));
        }
        let zero_in = zero_recurrent_inputs(&self.cfg, self.batch);
        for (name, data) in &zero_in {
            feeds.push((name, data.as_slice()));
        }
        // Prefill RoPE table must cover the compiled sequence extent
        // (`max_seq`), not only the unpadded prompt length.
        let rope_owned = self.mrope_prefill_rope_feeds(self.max_seq);
        for (name, data) in &rope_owned {
            feeds.push((name.as_str(), data.as_slice()));
        }
        self.ensure_predict_compiled()?;
        let outs = self.compiled.as_mut().unwrap().run(&feeds);
        if outs.is_empty() {
            bail!("qwen35: forward produced no outputs");
        }
        // RLX_QWEN35_DEBUG_LAYERS=1 enabled extra per-layer outputs in
        // the predict graph (see `ensure_predict_compiled`). Dump stats
        // here so the next debugger can locate which layer first emits
        // zero/NaN values without re-running the entire 18-min cycle.
        if std::env::var("RLX_QWEN35_DEBUG_LAYERS").as_deref() == Ok("1") {
            // When `export_trunk_layer_hiddens` is on, builder emits:
            //   outs[0] = post-embed last token [batch, n_embd]
            //   outs[1..n_layers] = after trunk layers 0..n_layers-1
            //   outs[n_layers+1] = logits [batch, vocab]
            // (layer hiddens are pushed *before* the LM head — see builder).
            let n_layers = self.cfg.num_hidden_layers - self.cfg.nextn_predict_layers;
            let n_embd = self.cfg.hidden_size;
            let dump_dir = std::env::var("RLX_QWEN35_DUMP_LAYERS")
                .ok()
                .map(PathBuf::from);
            if let Some(ref dir) = dump_dir {
                std::fs::create_dir_all(dir).with_context(|| {
                    format!("create RLX_QWEN35_DUMP_LAYERS dir {}", dir.display())
                })?;
            }
            let n_hidden_outs = n_layers + 1; // embed + one per layer
            for i in 0..outs.len() {
                let v = &outs[i];
                let mut min = f32::INFINITY;
                let mut max = f32::NEG_INFINITY;
                let mut sum = 0.0f64;
                let mut nan = 0usize;
                let mut nnz = 0usize;
                for &x in v {
                    if x.is_nan() {
                        nan += 1;
                        continue;
                    }
                    sum += x as f64;
                    if x < min {
                        min = x;
                    }
                    if x > max {
                        max = x;
                    }
                    if x != 0.0 {
                        nnz += 1;
                    }
                }
                let mean = sum / v.len().max(1) as f64;
                let is_logits = i >= n_hidden_outs;
                let label = if is_logits {
                    if i == n_hidden_outs {
                        "logits".to_string()
                    } else {
                        format!("extra_{:02}", i - n_hidden_outs)
                    }
                } else if i == 0 {
                    "embed".to_string()
                } else {
                    format!("layer_{:02}", i - 1)
                };
                eprintln!(
                    "[qwen35][debug-layers] {label}: len={} nnz={} nan={} min={} max={} mean={:.6}",
                    v.len(),
                    nnz,
                    nan,
                    min,
                    max,
                    mean
                );
                if let Some(ref dir) = dump_dir {
                    let path = dir.join(format!("{label}.npy"));
                    let (rows, cols) = if is_logits {
                        (self.batch, v.len() / self.batch.max(1))
                    } else {
                        (self.batch, n_embd)
                    };
                    write_npy_f32_row_major(&path, rows, cols, v)
                        .with_context(|| format!("write layer dump {}", path.display()))?;
                }
            }
            if let Some(ref dir) = dump_dir {
                eprintln!(
                    "[qwen35][debug-layers] wrote {} tensors under {}",
                    outs.len(),
                    dir.display()
                );
            }
        }
        let n_layers = self.cfg.num_hidden_layers - self.cfg.nextn_predict_layers;
        // With DEBUG_LAYERS, layer hiddens precede logits in `outs`.
        let logits_idx = if std::env::var("RLX_QWEN35_DEBUG_LAYERS").as_deref() == Ok("1") {
            n_layers + 1
        } else {
            0
        };
        anyhow::ensure!(
            logits_idx < outs.len(),
            "qwen35: missing logits output (idx={logits_idx}, outs={})",
            outs.len()
        );

        // Host tied-head: graph emitted normed hidden; score against the
        // F32 embedding table (or packed GGUF LM head via lm_loader).
        if self.fast_greedy_lm_head {
            let n_embd = self.cfg.hidden_size;
            let hidden = &outs[logits_idx];
            anyhow::ensure!(
                hidden.len() == self.batch * n_embd,
                "qwen35: fast_greedy predict expected hidden [{}, {}], got len={}",
                self.batch,
                n_embd,
                hidden.len()
            );
            let mut per_batch = Vec::with_capacity(self.batch);
            for b in 0..self.batch {
                let h = &hidden[b * n_embd..(b + 1) * n_embd];
                let row = lm_head_logits_row(&self.weights, &self.cfg, h, self.lm_loader())?;
                let sample_vocab = self.effective_vocab(row.len());
                let mut row = row;
                row.truncate(sample_vocab);
                per_batch.push(Qwen35PrefillOutput {
                    logits: row,
                    mtp_logits: None,
                    vocab_size: sample_vocab,
                });
            }
            return Ok(per_batch);
        }

        let vocab_size = if self.last_logits_only {
            outs[logits_idx].len() / self.batch
        } else {
            outs[logits_idx].len() / (self.batch * self.max_seq)
        };
        let sample_vocab = self.effective_vocab(vocab_size);
        let mtp_logits = if self.enable_mtp && outs.len() >= 2 && logits_idx == 0 {
            Some(outs[1].clone())
        } else {
            None
        };
        let mut per_batch = Vec::with_capacity(self.batch);
        for b in 0..self.batch {
            let start = b * vocab_size;
            let mut row = outs[logits_idx][start..start + vocab_size].to_vec();
            row.truncate(sample_vocab);
            per_batch.push(Qwen35PrefillOutput {
                logits: row,
                mtp_logits: mtp_logits.as_ref().map(|m| {
                    let m_vocab = m.len() / self.batch.max(1);
                    let mut mv = m[b * m_vocab..(b + 1) * m_vocab].to_vec();
                    mv.truncate(sample_vocab);
                    mv
                }),
                vocab_size: sample_vocab,
            });
        }
        Ok(per_batch)
    }

    /// Greedy autoregressive generation with decode-state caching (batch=1).
    pub fn generate<F>(&mut self, prompt_ids: &[u32], n_new: usize, on_token: F) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> bool,
    {
        self.generate_with_opts(prompt_ids, n_new, SampleOpts::greedy(), on_token)
    }

    /// Autoregressive generation with sampling options (batch=1).
    pub fn generate_with_opts<F>(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        opts: SampleOpts,
        mut on_token: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> bool,
    {
        if self.batch != 1 {
            bail!(
                "qwen35::generate: runner batch={} — use generate_batch() instead",
                self.batch
            );
        }
        let generated = self
            .generate_batch_with_opts(&[prompt_ids.to_vec()], n_new, None, opts, |_, tok| {
                on_token(tok)
            })?
            .into_iter()
            .next()
            .unwrap_or_default();
        Ok(generated)
    }

    /// Batched greedy generation. `prompts.len()` must equal `self.batch()`.
    pub fn generate_batch<F>(
        &mut self,
        prompts: &[Vec<u32>],
        n_new: usize,
        on_token: F,
    ) -> Result<Vec<Vec<u32>>>
    where
        F: FnMut(usize, u32) -> bool,
    {
        self.generate_batch_with_opts(prompts, n_new, None, SampleOpts::greedy(), on_token)
    }

    /// Batched generation with per-row token limits and sampling.
    ///
    /// `n_new_per_row`: optional per-row max new tokens (defaults to `n_new`).
    pub fn generate_batch_with_opts<F>(
        &mut self,
        prompts: &[Vec<u32>],
        n_new: usize,
        n_new_per_row: Option<&[usize]>,
        opts: SampleOpts,
        mut on_token: F,
    ) -> Result<Vec<Vec<u32>>>
    where
        F: FnMut(usize, u32) -> bool,
    {
        if prompts.is_empty() {
            bail!("qwen35::generate_batch: prompts must be non-empty");
        }
        if prompts.len() != self.batch {
            bail!(
                "qwen35::generate_batch: expected {} prompts, got {}",
                self.batch,
                prompts.len()
            );
        }
        if let Some(limits) = n_new_per_row {
            if limits.len() != self.batch {
                bail!(
                    "qwen35::generate_batch: n_new_per_row len {} != batch {}",
                    limits.len(),
                    self.batch
                );
            }
        }
        for (i, p) in prompts.iter().enumerate() {
            if p.is_empty() {
                bail!("qwen35::generate_batch: prompt row {i} is empty");
            }
        }

        self.decode_cache = None;

        let _prompt_lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
        let row_limits: Vec<usize> = if let Some(limits) = n_new_per_row {
            limits.to_vec()
        } else {
            vec![n_new; self.batch]
        };

        crate::trace::reset_tap_step();
        crate::trace::log_generate_header(
            self.device,
            self.fast_greedy_lm_head,
            crate::flow::host_embed_enabled_for_bytes(self.weights.token_embd.len() * 4),
            self.gguf_loader.is_some(),
            self.max_seq,
            self.lm_vocab_size(),
            n_new,
        );

        let bench = rlx_ir::env::flag("RLX_QWEN35_BENCH");
        let t_all = std::time::Instant::now();
        let t_prefill = std::time::Instant::now();
        let (trunk, mut cache, _) = self.prefill_seed_decode_cache(prompts)?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;

        if crate::trace::tap_enabled() {
            let n_embd = self.cfg.hidden_size;
            let row = if trunk.len() >= n_embd {
                &trunk[..n_embd]
            } else {
                trunk.as_slice()
            };
            // Prefill trunk is hidden when fast_greedy, else logits.
            let kind = if self.fast_greedy_lm_head {
                "hidden"
            } else {
                "logits"
            };
            let fp = crate::trace::fingerprint(row, 16);
            crate::trace::emit_tap("prefill", Some(0), None, kind, &fp);
        }

        let mut generated: Vec<Vec<u32>> = vec![Vec::new(); self.batch];
        let mut active = vec![true; self.batch];
        let mut row_gen_count = vec![0usize; self.batch];

        let t_decode = std::time::Instant::now();
        let mut step_timer = crate::trace::StepTimer::start();
        let mut next_tokens = if self.fast_greedy_lm_head && opts.greedy {
            crate::trace::log_lm_head_path("host_greedy");
            step_timer.time_lm(|| self.argmax_batch_from_hidden(&trunk))?
        } else if self.fast_greedy_lm_head
            && sample_lm_cap(&opts, self.lm_vocab_size()) < self.lm_vocab_size()
        {
            crate::trace::log_lm_head_path("host_sample");
            step_timer.time_lm(|| self.sample_batch_from_hidden(&trunk, opts))?
        } else {
            crate::trace::log_lm_head_path("graph_or_logits");
            let logits =
                step_timer.time_lm(|| self.trunk_to_logits(trunk, self.fast_greedy_lm_head))?;
            if crate::trace::tap_enabled() && !logits.is_empty() {
                let row = &logits[..logits.len().min(self.lm_vocab_size())];
                let fp = crate::trace::fingerprint(row, 16);
                crate::trace::emit_tap("prefill", Some(0), None, "logits", &fp);
            }
            sample_logits_batch(&logits, self.lm_vocab_size(), self.batch, opts)
        };
        if n_new > 0 {
            for b in 0..self.batch {
                if row_gen_count[b] >= row_limits[b] {
                    active[b] = false;
                    continue;
                }
                let tok = next_tokens[b];
                generated[b].push(tok);
                row_gen_count[b] += 1;
                active[b] = on_token(b, tok) && row_gen_count[b] < row_limits[b];
            }
            if self.batch == 1 {
                step_timer.finish(0, next_tokens[0]);
                if crate::trace::tap_enabled() {
                    crate::trace::emit_tap(
                        "token",
                        Some(0),
                        Some(next_tokens[0]),
                        "chosen",
                        &crate::trace::fingerprint(&[next_tokens[0] as f32], 1),
                    );
                }
            }
        }

        for step in 1..n_new {
            if !active.iter().any(|&a| a) {
                break;
            }
            if cache.past_seq >= self.max_seq - 1 {
                bail!("qwen35: decode cache reached max_seq={}", self.max_seq);
            }
            let mut step_timer = crate::trace::StepTimer::start();
            next_tokens = step_timer
                .time_run(|| self.decode_step(&mut cache, &next_tokens, &row_gen_count, opts))?;
            step_timer.time_cache(|| {
                self.decode_cache = Some(cache.clone());
            });
            for b in 0..self.batch {
                if !active[b] || row_gen_count[b] >= row_limits[b] {
                    active[b] = false;
                    continue;
                }
                let tok = next_tokens[b];
                generated[b].push(tok);
                row_gen_count[b] += 1;
                active[b] = on_token(b, tok) && row_gen_count[b] < row_limits[b];
            }
            if self.batch == 1 {
                step_timer.finish(step, next_tokens[0]);
                if crate::trace::tap_enabled() {
                    crate::trace::emit_tap(
                        "token",
                        Some(step),
                        Some(next_tokens[0]),
                        "chosen",
                        &crate::trace::fingerprint(&[next_tokens[0] as f32], 1),
                    );
                }
            }
        }
        let decode_ms = t_decode.elapsed().as_secs_f64() * 1e3;
        let total_ms = t_all.elapsed().as_secs_f64() * 1e3;
        if bench {
            let prompt_tokens: usize = prompts.iter().map(|p| p.len()).sum();
            let new_tokens: usize = generated.iter().map(|g| g.len()).sum();
            let ms_per_tok = if new_tokens > 0 {
                decode_ms / new_tokens as f64
            } else {
                0.0
            };
            let tok_s = if decode_ms > 0.0 {
                new_tokens as f64 / (decode_ms / 1e3)
            } else {
                0.0
            };
            eprintln!(
                "[qwen35][bench] device={:?} prompt_tokens={prompt_tokens} new_tokens={new_tokens} \
                 prefill_ms={prefill_ms:.1} decode_ms={decode_ms:.1} total_ms={total_ms:.1} \
                 ms/tok={ms_per_tok:.1} tok/s={tok_s:.3}",
                self.device
            );
        }
        Ok(generated)
    }

    /// Prefill and return trunk logits for the last prompt position.
    /// Seeds the decode cache so subsequent [`Self::decode_get_logits`] calls work.
    pub fn prefill_get_last_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        Ok(self.prefill_seed_for_decode(prompt_ids)?.trunk_logits)
    }

    /// Prefill-cache seed + decode cache for spec decode / MTP draft paths.
    pub fn prefill_seed_for_decode(&mut self, prompt_ids: &[u32]) -> Result<Qwen35PrefillSeed> {
        if self.batch != 1 {
            bail!(
                "qwen35: prefill_seed_for_decode requires batch=1 (runner batch={})",
                self.batch
            );
        }
        let (trunk, _, mtp_logits) = self.prefill_seed_decode_cache(&[prompt_ids.to_vec()])?;
        Ok(Qwen35PrefillSeed {
            trunk_logits: self.trunk_to_logits(trunk, self.fast_greedy_lm_head)?,
            mtp_logits,
        })
    }

    /// Multimodal prefill: vision-encode `rgb`, splice into `prompt` at
    /// [`MEDIA_MARKER`](crate::MEDIA_MARKER), seed decode cache.
    pub fn prefill_multimodal(
        &mut self,
        prompt: &str,
        rgb: &[u8],
        img_w: usize,
        img_h: usize,
        tokenizer: Option<&std::path::Path>,
    ) -> Result<Qwen35PrefillSeed> {
        let (trunk, mtp_logits) =
            self.prefill_multimodal_trunk(prompt, rgb, img_w, img_h, tokenizer)?;
        Ok(Qwen35PrefillSeed {
            trunk_logits: self.trunk_to_logits(trunk, self.fast_greedy_lm_head)?,
            mtp_logits,
        })
    }

    /// Prefill from an already-assembled multimodal payload (tests / custom tokenizers).
    pub fn prefill_from_assembled(
        &mut self,
        prefill: MultimodalPrefill,
    ) -> Result<Qwen35PrefillSeed> {
        if self.batch != 1 {
            bail!(
                "qwen35: prefill_from_assembled requires batch=1 (runner batch={})",
                self.batch
            );
        }
        self.mrope_section_positions = Some(prefill.mrope_sections.clone());
        let (trunk, _, mtp_logits) = self.prefill_seed_from_hidden(prefill)?;
        Ok(Qwen35PrefillSeed {
            trunk_logits: self.trunk_to_logits(trunk, self.fast_greedy_lm_head)?,
            mtp_logits,
        })
    }

    fn prefill_multimodal_trunk(
        &mut self,
        prompt: &str,
        rgb: &[u8],
        img_w: usize,
        img_h: usize,
        tokenizer: Option<&std::path::Path>,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        if self.batch != 1 {
            bail!(
                "qwen35: prefill_multimodal requires batch=1 (runner batch={})",
                self.batch
            );
        }
        let vision = {
            let enc = self
                .vision_encoder
                .as_mut()
                .ok_or_else(|| anyhow!("qwen35: prefill_multimodal requires .mmproj(...)"))?;
            enc.encode_rgb(rgb, img_w, img_h)?
        };
        eprintln!(
            "[qwen35] vision encode: grid={}x{} tokens={} (src={}x{})",
            vision.grid_x, vision.grid_y, vision.n_tokens, img_w, img_h
        );
        // Vision tower is only needed for encode; drop it before the dense
        // LLM graphs so its F32 param clone is not resident with prefill.
        if release_host_dense_enabled(self.device, &self.weights) {
            self.vision_encoder = None;
            eprintln!("[qwen35] dropped vision encoder after encode (low-mem)");
        }
        if self.weights.token_embd.is_empty() {
            bail!("qwen35: multimodal prefill requires token_embd weights");
        }
        let weights_path = self.weights_path.as_path();
        if weights_path.as_os_str().is_empty() {
            bail!("qwen35: multimodal prefill requires a GGUF weights path (for tokenizer)");
        }
        let n_embd = self.cfg.hidden_size;
        let mm = MultimodalPrompt {
            prompt,
            vision: &vision,
        };
        let prefill = mm.assemble(
            |text| encode_prompt_auto(weights_path, tokenizer, text),
            &self.weights.token_embd,
            n_embd,
            0,
        )?;
        eprintln!(
            "[qwen35] multimodal seq={} (vision_tokens={})",
            prefill.seq.len(),
            vision.n_tokens
        );
        self.mrope_section_positions = Some(prefill.mrope_sections.clone());
        let (trunk, _, mtp_logits) = self.prefill_seed_from_hidden(prefill)?;
        Ok((trunk, mtp_logits))
    }

    /// Autoregressive generation from a multimodal prompt (batch=1).
    pub fn generate_multimodal_with_opts<F>(
        &mut self,
        prompt: &str,
        rgb: &[u8],
        img_w: usize,
        img_h: usize,
        tokenizer: Option<&std::path::Path>,
        n_new: usize,
        opts: SampleOpts,
        mut on_token: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> bool,
    {
        if self.batch != 1 {
            bail!(
                "qwen35: generate_multimodal requires batch=1 (runner batch={})",
                self.batch
            );
        }
        self.decode_cache = None;
        let (trunk, _) = self.prefill_multimodal_trunk(prompt, rgb, img_w, img_h, tokenizer)?;
        let mut cache = self
            .decode_cache
            .take()
            .ok_or_else(|| anyhow!("qwen35: multimodal prefill did not seed decode cache"))?;
        if rlx_ir::env::flag("RLX_QWEN35_DEBUG_LOGITS") {
            let n_embd = self.cfg.hidden_size;
            let h = &trunk[..n_embd.min(trunk.len())];
            let (norm_sq, abs_max) = h
                .iter()
                .fold((0f32, 0f32), |(s, m), &x| (s + x * x, m.max(x.abs())));
            eprintln!(
                "[qwen35][debug] trunk_l2={:.4} absmax={:.4} n_embd={} mrope_interleaved={}",
                norm_sq.sqrt(),
                abs_max,
                n_embd,
                self.cfg.mrope_interleaved
            );
            if let Ok((idx, score)) =
                greedy_lm_head_argmax(&self.weights, &self.cfg, h, self.lm_loader())
            {
                eprintln!("[qwen35][debug] lm_argmax id={idx} score={score:.4}");
            }
        }
        let mut next_tokens = if self.fast_greedy_lm_head && opts.greedy {
            self.argmax_batch_from_hidden(&trunk)?
        } else if self.fast_greedy_lm_head
            && sample_lm_cap(&opts, self.lm_vocab_size()) < self.lm_vocab_size()
        {
            self.sample_batch_from_hidden(&trunk, opts)?
        } else {
            let logits = self.trunk_to_logits(trunk, self.fast_greedy_lm_head)?;
            sample_logits_batch(&logits, self.lm_vocab_size(), 1, opts)
        };
        let mut generated = Vec::new();
        if n_new > 0 {
            let tok = next_tokens[0];
            generated.push(tok);
            if !on_token(tok) {
                return Ok(generated);
            }
        }
        for _ in 1..n_new {
            if cache.past_seq >= self.max_seq - 1 {
                bail!("qwen35: decode cache reached max_seq={}", self.max_seq);
            }
            // Dense batch=1: derive generated count from cache so bucketed masks
            // keep every prior token visible (do not freeze at 0).
            let row_gen = dense_generated_per_row(&cache);
            next_tokens = self.decode_step(&mut cache, &next_tokens, &row_gen, opts)?;
            let tok = next_tokens[0];
            generated.push(tok);
            self.decode_cache = Some(cache.clone());
            if !on_token(tok) {
                break;
            }
        }
        Ok(generated)
    }

    /// Greedy multimodal generation (batch=1).
    pub fn generate_multimodal<F>(
        &mut self,
        prompt: &str,
        rgb: &[u8],
        img_w: usize,
        img_h: usize,
        tokenizer: Option<&std::path::Path>,
        n_new: usize,
        on_token: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> bool,
    {
        self.generate_multimodal_with_opts(
            prompt,
            rgb,
            img_w,
            img_h,
            tokenizer,
            n_new,
            SampleOpts::greedy(),
            on_token,
        )
    }

    /// Run the prefill-cache graph, seed decode state, return flattened batch logits.
    fn prefill_seed_decode_cache(
        &mut self,
        prompts: &[Vec<u32>],
    ) -> Result<(Vec<f32>, Qwen35DecodeCache, Option<Vec<f32>>)> {
        if prompts.len() != self.batch {
            bail!(
                "qwen35: expected {} prompts (batch={}), got {}",
                self.batch,
                self.batch,
                prompts.len()
            );
        }
        for (i, p) in prompts.iter().enumerate() {
            if p.is_empty() {
                bail!("qwen35: prompt row {i} is empty");
            }
        }

        self.ensure_static_prefill_cache()?;

        let prompt_lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
        let seq = prompt_lens.iter().copied().max().unwrap();
        if seq > self.max_seq {
            bail!(
                "qwen35: prompt length {seq} exceeds compiled max_seq={}",
                self.max_seq
            );
        }
        let prefill_pad = if self.dynamic_prefill {
            seq
        } else {
            if seq > self.prefill_seq {
                bail!(
                    "qwen35: prompt length {seq} exceeds compiled prefill_seq={}",
                    self.prefill_seq
                );
            }
            self.prefill_seq
        };

        let input_ids = pack_input_ids(prompts, prefill_pad)?;
        let last_idx = last_token_indices(&prompt_lens);

        // Host-gathered token embeddings (RLX_QWEN35_HOST_EMBED): the graph
        // takes an `inputs_embeds` input instead of a resident `[vocab,hidden]`
        // F32 table param. We index `token_embd` (already an F32 host `Vec`) by
        // the SAME packed `input_ids` the graph gather would use — so padding
        // rows (token 0) match bit-for-bit. Keeps the 4.7 GiB Bonsai table off
        // the accelerator.
        let host_embeds: Vec<f32> =
            if crate::flow::host_embed_enabled_for_bytes(self.weights.token_embd.len() * 4)
                && !self.weights.token_embd.is_empty()
            {
                let n_embd = self.cfg.hidden_size;
                let tbl = &self.weights.token_embd;
                let mut v = vec![0f32; input_ids.len() * n_embd];
                for (pos, &id_f) in input_ids.iter().enumerate() {
                    let src = (id_f as usize) * n_embd;
                    if src + n_embd <= tbl.len() {
                        v[pos * n_embd..pos * n_embd + n_embd]
                            .copy_from_slice(&tbl[src..src + n_embd]);
                    }
                }
                v
            } else {
                Vec::new()
            };

        let mut feeds: Vec<(&str, &[f32])> = vec![("input_ids", input_ids.as_slice())];
        feeds.push(("last_token_idx", last_idx.as_slice()));
        if !host_embeds.is_empty() {
            feeds.push(("inputs_embeds", host_embeds.as_slice()));
        }
        let zero_in = zero_recurrent_inputs(&self.cfg, self.batch);
        for (name, data) in &zero_in {
            feeds.push((name, data.as_slice()));
        }
        let rope_owned = self.mrope_prefill_rope_feeds(seq);
        for (name, data) in &rope_owned {
            feeds.push((name.as_str(), data.as_slice()));
        }

        let has_moe = self.moe_offload.is_some();
        let num_experts = self.cfg.num_experts;
        let moe_masks = self
            .moe_offload
            .as_ref()
            .map(|m| m.per_layer_resident_masks());
        self.bind_moe_host_weights();

        let outs = if self.dynamic_prefill {
            let config = self.execution_config(prefill_config(self.batch, seq));
            let compile_opts = self.dyn_compile_options(&config);
            let compiled = {
                let cache = self
                    .prefill_dynamic_cache
                    .as_mut()
                    .expect("dynamic prefill cache");
                let cfg = self.cfg.clone();
                let weights = self.weights.clone();
                let runtime_mrope = self.runtime_mrope;
                let mtp_logits_path = self.mtp_logits_path;
                let fast_mtp = self.fast_mtp;
                let fast_greedy = self.fast_greedy_lm_head;
                let need_extract =
                    self.prefill_cache_params.is_empty() && self.prefill_cache_packed.is_empty();
                let captured =
                    std::cell::RefCell::new(None::<(HashMap<String, Vec<f32>>, PackedParams)>);
                let cache_params = &self.prefill_cache_params;
                let cache_packed = &self.prefill_cache_packed;
                let gguf_loader = &mut self.gguf_loader;
                let packed_bytes_cache = &mut self.packed_bytes_cache;
                get_or_specialize_hir_with_options(
                    cache,
                    &config,
                    || {
                        // Concrete-seq static HIR: avoids Dynamic(SEQ)+Static(k-1)
                        // causal-pad concat, which MPSGraph mis-sizes.
                        let (hir, params, packed) = build_qwen35_prefill_cache_hir_ext(
                            &cfg,
                            weights.clone(),
                            1,
                            seq,
                            runtime_mrope,
                            mtp_logits_path,
                            fast_mtp,
                            fast_greedy,
                        )
                        .expect("static prefill HIR at concrete seq");
                        if need_extract {
                            *captured.borrow_mut() = Some((params, packed));
                        }
                        hir
                    },
                    &compile_opts,
                    |c| {
                        if let Some((params, packed)) = captured.borrow_mut().take() {
                            for (name, data) in &params {
                                c.set_param(name, data);
                            }
                            upload_packed_opt(c, gguf_loader.as_mut(), &packed, packed_bytes_cache)
                                .map(|_| ())
                        } else {
                            for (name, data) in cache_params {
                                c.set_param(name, data);
                            }
                            upload_packed_opt(
                                c,
                                gguf_loader.as_mut(),
                                cache_packed,
                                packed_bytes_cache,
                            )
                            .map(|_| ())
                        }
                    },
                )?
            };
            if has_moe {
                compiled.enable_moe_topk_capture(num_experts);
                if let Some(layers) = &moe_masks {
                    push_moe_residency(compiled, layers);
                }
            }
            let outs = compiled.run(&feeds);
            if let Some(layers) = compiled.take_moe_topk_capture() {
                if let Some(mo) = self.moe_offload.as_mut() {
                    let store = self.moe_store.as_ref();
                    if refresh_moe_from_capture(mo, store, compiled, &layers, 0, true) {
                        if let Some(store) = self.moe_store.as_ref() {
                            store.apply_to_compiled(compiled);
                        }
                    }
                }
            }
            outs
        } else {
            let compiled = self.prefill_cache.as_mut().expect("static prefill cache");
            if has_moe {
                compiled.enable_moe_topk_capture(num_experts);
                if let Some(layers) = &moe_masks {
                    push_moe_residency(compiled, layers);
                }
            }
            let outs = if packed_prefill_active_extent_enabled(self.device)
                && seq > 0
                && seq < self.prefill_seq
            {
                run_packed_prefill(compiled, self.device, seq, self.prefill_seq, &feeds)
            } else {
                compiled.run(&feeds)
            };
            let layers = if has_moe {
                compiled.take_moe_topk_capture()
            } else {
                None
            };
            if let (Some(mo), Some(layers)) = (self.moe_offload.as_mut(), layers) {
                let store = self.moe_store.as_ref();
                if refresh_moe_from_capture(mo, store, compiled, &layers, 0, true) {
                    if let Some(store) = self.moe_store.as_ref() {
                        store.apply_to_compiled(compiled);
                    }
                }
            }
            outs
        };
        let (trunk, mut cache, mtp_logits) = seed_cache_from_outputs(
            &self.cfg,
            self.batch,
            seq,
            &prompt_lens,
            outs,
            self.mtp_logits_path,
            self.fast_greedy_lm_head,
        )?;
        zero_prompt_padding_kv(&self.cfg, &mut cache, seq);
        crate::trace::emit_cache_tap(&cache);
        self.decode_cache = Some(cache.clone());
        // Free prefill device arena before decode on discrete GPUs when both
        // arenas may not fit. Short Metal/MLX/CUDA contexts keep prefill —
        // avoids a multi-second rebuild on the next turn / after warm.
        // Override: RLX_QWEN35_KEEP_PREFILL=1|0.
        let keep_prefill = match rlx_ir::env::var("RLX_QWEN35_KEEP_PREFILL").as_deref() {
            Some("1") | Some("true") | Some("on") => true,
            Some("0") | Some("false") | Some("off") => false,
            _ => {
                self.max_seq <= 128
                    && matches!(self.device, Device::Metal | Device::Mlx | Device::Cuda)
            }
        };
        if !keep_prefill
            && matches!(
                self.device,
                Device::Cuda
                    | Device::Rocm
                    | Device::Gpu
                    | Device::Vulkan
                    | Device::Metal
                    | Device::Mlx
            )
        {
            self.drop_prefill_cache();
        }
        Ok((trunk, cache, mtp_logits))
    }

    /// VLM prefill-cache path: host-spliced hidden states + runtime MRoPE sections.
    fn prefill_seed_from_hidden(
        &mut self,
        prefill: MultimodalPrefill,
    ) -> Result<(Vec<f32>, Qwen35DecodeCache, Option<Vec<f32>>)> {
        let seq = prefill.seq.len();
        if seq == 0 {
            bail!("qwen35: multimodal prefill seq is empty");
        }
        if seq > self.max_seq {
            bail!(
                "qwen35: multimodal seq {seq} exceeds compiled max_seq={}",
                self.max_seq
            );
        }
        let n_embd = self.cfg.hidden_size;
        if prefill.hidden.len() != seq * n_embd {
            bail!(
                "qwen35: prefill hidden len {} != seq*n_embd {}*{}",
                prefill.hidden.len(),
                seq,
                n_embd
            );
        }

        let last_idx = vec![prefill.last_token_idx as f32];
        let zero_in = zero_recurrent_inputs(&self.cfg, self.batch);
        let input_ids = if self.mtp_logits_path || self.enable_mtp {
            Some(pack_input_ids(std::slice::from_ref(&prefill.seq), seq)?)
        } else {
            None
        };
        let mut feeds: Vec<(&str, &[f32])> = vec![("prefill_hidden", prefill.hidden.as_slice())];
        feeds.push(("last_token_idx", last_idx.as_slice()));
        for (name, data) in &zero_in {
            feeds.push((name, data.as_slice()));
        }
        if let Some(ref ids) = input_ids {
            feeds.push(("input_ids", ids.as_slice()));
        }
        let rope_owned = self.mrope_prefill_rope_feeds(seq);
        for (name, data) in &rope_owned {
            feeds.push((name.as_str(), data.as_slice()));
        }

        let config = self.execution_config(hidden_prefill_config(self.batch, seq));
        let compile_opts = self.dyn_compile_options(&config);
        let already = self
            .prefill_hidden_dynamic_cache
            .as_ref()
            .is_some_and(|c| c.contains(&config));
        if !already {
            self.ensure_dense_projections_resident()?;
            let release = release_host_dense_enabled(self.device, &self.weights);
            let (hir, params, packed) = build_qwen35_prefill_hidden_cache_hir_ext(
                &self.cfg,
                std::sync::Arc::clone(&self.weights),
                1,
                seq,
                self.runtime_mrope,
                self.mtp_logits_path,
                self.fast_mtp,
                self.fast_greedy_lm_head,
            )
            .context("static hidden prefill HIR at concrete seq")?;
            // Free host before Metal compile; decode reloads from mmap after
            // prefill arena is dropped.
            if release {
                self.clear_host_dense_projections();
            }
            let hir_cell = std::cell::RefCell::new(Some(hir));
            let capt = std::cell::RefCell::new(Some((params, packed)));
            let gguf_loader = &mut self.gguf_loader;
            let packed_bytes_cache = &mut self.packed_bytes_cache;
            let cache = self.prefill_hidden_dynamic_cache.as_mut().ok_or_else(|| {
                anyhow!("qwen35: hidden prefill cache missing (mmproj not loaded?)")
            })?;
            get_or_specialize_hir_with_options(
                cache,
                &config,
                || {
                    hir_cell
                        .borrow_mut()
                        .take()
                        .expect("hidden prefill HIR already taken")
                },
                &compile_opts,
                |c| {
                    let (params, packed) = capt
                        .borrow_mut()
                        .take()
                        .expect("hidden prefill params already taken");
                    for (name, data) in params {
                        c.set_param(&name, &data);
                    }
                    upload_packed_opt(c, gguf_loader.as_mut(), &packed, packed_bytes_cache)
                        .map(|_| ())
                },
            )?;
        }
        let cache = self
            .prefill_hidden_dynamic_cache
            .as_mut()
            .ok_or_else(|| anyhow!("qwen35: hidden prefill cache missing (mmproj not loaded?)"))?;
        let compiled = get_or_specialize_hir_with_options(
            cache,
            &config,
            || panic!("qwen35: hidden prefill cache miss after low-mem specialize"),
            &compile_opts,
            |_| Ok(()),
        )?;
        let outs = compiled.run(&feeds);
        let prompt_lens = vec![seq];
        let (trunk, mut cache, mtp_logits) = seed_cache_from_outputs(
            &self.cfg,
            self.batch,
            seq,
            &prompt_lens,
            outs,
            self.mtp_logits_path,
            self.fast_greedy_lm_head,
        )?;
        zero_prompt_padding_kv(&self.cfg, &mut cache, seq);
        self.decode_cache = Some(cache.clone());
        // Same policy as token-id prefill: free the specialized hidden-prefill
        // arena before decode so dense F32 Metal/MLX builds fit in RAM.
        // Note: host projections may already be cleared — do not gate on
        // `has_dense_f32_projections()` here.
        let keep_prefill = match rlx_ir::env::var("RLX_QWEN35_KEEP_PREFILL").as_deref() {
            Some("1") | Some("true") | Some("on") => true,
            Some("0") | Some("false") | Some("off") => false,
            _ => {
                self.max_seq <= 128
                    && matches!(self.device, Device::Metal | Device::Mlx | Device::Cuda)
            }
        };
        if !keep_prefill
            && matches!(
                self.device,
                Device::Cuda
                    | Device::Rocm
                    | Device::Gpu
                    | Device::Vulkan
                    | Device::Metal
                    | Device::Mlx
            )
        {
            self.drop_prefill_cache();
        }
        Ok((trunk, cache, mtp_logits))
    }

    /// Single cached decode step returning trunk logits for `token`.
    pub fn decode_get_logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.decode_forward_logits(token, false)
    }

    /// Single cached decode step returning MTP head logits for `token`.
    pub fn decode_get_mtp_logits(&mut self, token: u32) -> Result<Vec<f32>> {
        if !self.mtp_logits_path {
            bail!("qwen35: decode_get_mtp_logits requires mtp_logits_path(true)");
        }
        self.decode_forward_logits(token, true)
    }

    /// Prefill and return optional MTP logits (requires `enable_mtp`).
    pub fn prefill_get_mtp_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        self.predict_logits(prompt_ids)?
            .mtp_logits
            .ok_or_else(|| anyhow!("qwen35: MTP logits unavailable (enable_mtp?)"))
    }

    /// Prefill from host-assembled hidden states. Returns last-token hidden + decode cache.
    pub fn prefill_hidden_state(
        &mut self,
        hidden: &[f32],
        seq_len: usize,
    ) -> Result<(Vec<f32>, Qwen35DecodeCache)> {
        let n_embd = self.cfg.hidden_size;
        if hidden.len() != seq_len * n_embd {
            bail!(
                "qwen35: prefill hidden len {} != seq*n_embd {}*{}",
                hidden.len(),
                seq_len,
                n_embd
            );
        }
        let prefill = MultimodalPrefill {
            hidden: hidden.to_vec(),
            mrope_sections: (0..seq_len).map(crate::rope::text_section_pos).collect(),
            last_token_idx: seq_len.saturating_sub(1),
            seq: vec![0u32; seq_len],
        };
        let (trunk, cache, _) = self.prefill_seed_from_hidden(prefill)?;
        Ok((trunk, cache))
    }

    /// One decode step from a custom embedding vector (Gepard audio-frame path).
    pub fn decode_hidden_state(
        &mut self,
        cache: &mut Qwen35DecodeCache,
        embed: &[f32],
    ) -> Result<Vec<f32>> {
        if embed.len() != self.cfg.hidden_size {
            bail!(
                "qwen35: decode embed len {} != hidden {}",
                embed.len(),
                self.cfg.hidden_size
            );
        }
        if !self.host_embed {
            bail!("qwen35: decode_hidden_state requires force_host_embed(true) on the builder");
        }
        let tokens = vec![0u32];
        // Bucketed custom-mask decode needs `prompt_len + generated == past_seq`
        // so every cached row stays visible (not just the original prompt).
        let row_gen = dense_generated_per_row(cache);
        let (trunk, _) = self.decode_step_trunk_raw(cache, &tokens, &row_gen, Some(embed))?;
        Ok(trunk)
    }

    fn decode_step(
        &mut self,
        cache: &mut Qwen35DecodeCache,
        tokens: &[u32],
        generated_per_row: &[usize],
        opts: SampleOpts,
    ) -> Result<Vec<u32>> {
        if self.fast_greedy_lm_head {
            let vocab = self.lm_vocab_size();
            let (trunk, _mtp) =
                self.decode_step_trunk_raw(cache, tokens, generated_per_row, None)?;
            if opts.greedy {
                return self.argmax_batch_from_hidden(&trunk);
            }
            if sample_lm_cap(&opts, vocab) < vocab {
                return self.sample_batch_from_hidden(&trunk, opts);
            }
        }
        let logits = self.decode_forward_logits_batch(cache, tokens, generated_per_row, false)?;
        if crate::trace::tap_enabled() && self.batch == 1 && !logits.is_empty() {
            let row = &logits[..logits.len().min(self.lm_vocab_size())];
            let fp = crate::trace::fingerprint(row, 16);
            crate::trace::emit_tap("decode", None, None, "logits", &fp);
        }
        Ok(sample_logits_batch(
            &logits,
            self.lm_vocab_size(),
            self.batch,
            opts,
        ))
    }

    fn decode_forward_logits(&mut self, token: u32, want_mtp: bool) -> Result<Vec<f32>> {
        let mut cache = self
            .decode_cache
            .take()
            .ok_or_else(|| anyhow!("qwen35: decode requires seeded cache"))?;
        let row_gen = dense_generated_per_row(&cache);
        let logits = self.decode_forward_logits_batch(&mut cache, &[token], &row_gen, want_mtp)?;
        self.decode_cache = Some(cache);
        Ok(logits)
    }

    fn decode_forward_logits_batch(
        &mut self,
        cache: &mut Qwen35DecodeCache,
        tokens: &[u32],
        generated_per_row: &[usize],
        want_mtp: bool,
    ) -> Result<Vec<f32>> {
        let past_seq = cache.past_seq;
        let (cos, sin) = self.mrope_decode_rope_at_past(past_seq);

        let use_bucket = self
            .decode_compile_cache
            .as_ref()
            .and_then(|c| c.bucket_for(past_seq as u64))
            .is_some();

        if use_bucket {
            let (logits, mtp_logits) =
                self.decode_step_bucketed(cache, tokens, generated_per_row, &cos, &sin)?;
            if want_mtp {
                mtp_logits.ok_or_else(|| anyhow!("mtp decode logits missing from bucketed graph"))
            } else {
                Ok(logits)
            }
        } else {
            let feeds_owned = decode_step_feeds(
                &self.cfg,
                cache,
                tokens,
                &cos,
                &sin,
                None,
                generated_per_row,
                self.host_embed.then(|| self.weights.token_embd.as_ref()),
                None,
            )?;
            let feeds: Vec<(&str, &[f32])> = feeds_owned
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_slice()))
                .collect();
            if !self.decode_graphs.contains_key(&past_seq) {
                let (hir, params, packed) = build_qwen35_decode_hir_ext(
                    &self.cfg,
                    self.weights.clone(),
                    self.batch,
                    past_seq,
                    false,
                    self.mtp_logits_path,
                    self.fast_mtp,
                    self.fast_greedy_lm_head,
                )?;
                let mut compiled = self.compile_hir_for_config(
                    decode_config(self.batch, past_seq),
                    &format!("decode_{past_seq}"),
                    hir,
                )?;
                for (name, data) in &params {
                    compiled.set_param(name, data);
                }
                upload_packed_opt(
                    &mut compiled,
                    self.gguf_loader.as_mut(),
                    &packed,
                    &mut self.packed_bytes_cache,
                )?;
                self.decode_graphs.insert(past_seq, compiled);
            }
            let step = self.moe_refresh_step;
            let has_moe = self.moe_offload.is_some();
            let num_experts = self.cfg.num_experts;
            let moe_masks = self
                .moe_offload
                .as_ref()
                .map(|m| m.per_layer_resident_masks());
            self.bind_moe_host_weights();
            let outs = {
                let compiled = self.decode_graphs.get_mut(&past_seq).unwrap();
                if has_moe {
                    compiled.enable_moe_topk_capture(num_experts);
                    if let Some(layers) = &moe_masks {
                        push_moe_residency(compiled, layers);
                    }
                }
                compiled.run(&feeds)
            };
            if has_moe {
                let layers = {
                    let compiled = self.decode_graphs.get_mut(&past_seq).unwrap();
                    compiled.take_moe_topk_capture()
                };
                if let (Some(mo), Some(layers)) = (self.moe_offload.as_mut(), layers) {
                    let store = self.moe_store.as_ref();
                    let compiled = self.decode_graphs.get_mut(&past_seq).unwrap();
                    if refresh_moe_from_capture(mo, store, compiled, &layers, step, false) {
                        if let Some(store) = self.moe_store.as_ref() {
                            store.apply_to_compiled(compiled);
                        }
                    }
                }
            }
            self.moe_refresh_step = step.saturating_add(1);
            let (trunk, mtp_logits) = advance_cache_from_decode_outputs(
                &self.cfg,
                cache,
                outs,
                None,
                self.mtp_logits_path,
                want_mtp,
                self.fast_greedy_lm_head,
            )?;
            let logits = self.trunk_to_logits(trunk, self.fast_greedy_lm_head)?;
            if want_mtp {
                mtp_logits.ok_or_else(|| anyhow!("mtp decode logits missing from decode graph"))
            } else {
                Ok(logits)
            }
        }
    }

    fn decode_step_bucketed(
        &mut self,
        cache: &mut Qwen35DecodeCache,
        tokens: &[u32],
        generated_per_row: &[usize],
        cos: &[f32],
        sin: &[f32],
    ) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        let (trunk, mtp) =
            self.decode_step_bucketed_raw(cache, tokens, generated_per_row, cos, sin, None)?;
        let logits = self.trunk_to_logits(trunk, self.fast_greedy_lm_head)?;
        Ok((logits, mtp))
    }

    fn decode_step_bucketed_raw(
        &mut self,
        cache: &mut Qwen35DecodeCache,
        tokens: &[u32],
        generated_per_row: &[usize],
        cos: &[f32],
        sin: &[f32],
        custom_embed: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        let past_seq = cache.past_seq;
        let bucket_key = self.decode_bucket_key(past_seq as u64);
        let upper = self.ensure_decode_bucket_compiled(past_seq as u64)?;

        let feeds_owned = decode_step_feeds(
            &self.cfg,
            cache,
            tokens,
            cos,
            sin,
            Some(upper),
            generated_per_row,
            self.host_embed.then(|| self.weights.token_embd.as_ref()),
            custom_embed,
        )?;
        let feeds: Vec<(&str, &[f32])> = feeds_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_slice()))
            .collect();

        let decode_opts = self.bucketed_decode_compile_options();
        let cache_mut = self.decode_compile_cache.as_mut().unwrap();
        let (_u, compiled) = cache_mut
            .ensure_hir_with_params(
                bucket_key,
                |_| panic!("decode bucket must be compiled"),
                &decode_opts,
            )
            .expect("decode bucket missing after ensure");
        // Custom-mask bucketed decode places the new K/V row at index `upper`,
        // not at `past_seq`. Scaling launch dims by `(past_seq+1)/(upper+1)`
        // truncates hidden-sized elementwise ops mid-vector on CUDA. The binary
        // mask already zeros padding slots — leave full compiled extent.
        let outs = compiled.run(&feeds);
        advance_cache_from_decode_outputs(
            &self.cfg,
            cache,
            outs,
            Some(upper),
            self.mtp_logits_path,
            self.mtp_logits_path,
            self.fast_greedy_lm_head,
        )
    }

    fn mrope_prefill_rope_feeds(&self, seq: usize) -> Vec<(String, Vec<f32>)> {
        if !self.runtime_mrope {
            return Vec::new();
        }
        let head_half = self.cfg.key_length / 2;
        let sections = self.mrope_section_positions.as_deref();
        let (cos, sin) = mrope_prefill_feeds(&self.cfg, seq, sections, head_half);
        vec![("rope_cos".into(), cos), ("rope_sin".into(), sin)]
    }

    /// Decode-step MRoPE at the next absolute position after the multimodal
    /// (or text) prefix. Vision tokens compress positions (`max(nx,ny)` for
    /// `nx*ny` tokens); using raw `past_seq` desyncs queries from cached K.
    fn mrope_decode_rope_at_past(&self, past_seq: usize) -> (Vec<f32>, Vec<f32>) {
        let head_half = self.cfg.key_length / 2;
        let next_sec = if let Some(sections) = self.mrope_section_positions.as_deref() {
            if past_seq == 0 {
                text_section_pos(0)
            } else if let Some(last) = sections.get(past_seq.saturating_sub(1)) {
                // Continue from the last prefill/decode section's temporal pos.
                let t = last[0] + 1;
                [t, t, t, 0]
            } else {
                // Past the assembled prefix: step from the last known section.
                let last = sections.last().copied().unwrap_or(text_section_pos(0));
                let extra = past_seq.saturating_sub(sections.len()) + 1;
                let t = last[0] + extra;
                [t, t, t, 0]
            }
        } else {
            text_section_pos(past_seq)
        };
        mrope_row_for_sections(&self.cfg, next_sec, head_half)
    }
}

impl rlx_cli::LmRunner for Qwen35Runner {
    fn family(&self) -> &'static str {
        "qwen35"
    }
    fn vocab_size(&self) -> usize {
        self.lm_vocab_size()
    }
    fn predict_logits(&mut self, prompt_ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        let out = Qwen35Runner::predict_logits(self, prompt_ids)?;
        Ok(out.logits)
    }
    fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> anyhow::Result<Vec<u32>> {
        // Qwen35Runner::generate takes a bool-returning callback
        // (false = stop). The trait callback now has the same shape,
        // so just forward.
        Qwen35Runner::generate(self, prompt_ids, n_new, on_token)
    }

    fn supports_multimodal(&self) -> bool {
        // True when an mmproj path or inline mmproj weights were
        // attached at builder time. The encoder is lazy-loaded inside
        // `generate_multimodal_with_opts`.
        self.has_mmproj()
    }

    fn generate_multimodal(
        &mut self,
        prompt: &str,
        rgb: &[u8],
        img_w: usize,
        img_h: usize,
        tokenizer: Option<&std::path::Path>,
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> anyhow::Result<Vec<u32>> {
        Qwen35Runner::generate_multimodal(
            self, prompt, rgb, img_w, img_h, tokenizer, n_new, on_token,
        )
    }
}

fn sample_logits_batch(logits: &[f32], vocab: usize, batch: usize, opts: SampleOpts) -> Vec<u32> {
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let row = &logits[b * vocab..(b + 1) * vocab];
        out.push(sample_token(row, opts) as u32);
    }
    out
}

/// Tokens past each row's prompt for dense (non-varlen-pad) KV caches.
///
/// Bucketed decode masks with `prompt_lens[b] + generated[b]`; for Gepard-style
/// sequential prefill/`decode_hidden_state` the host cache is dense up to
/// `past_seq`, so generated must be `past_seq - prompt_lens[b]` (not a frozen 0).
fn dense_generated_per_row(cache: &Qwen35DecodeCache) -> Vec<usize> {
    (0..cache.batch)
        .map(|b| {
            let pl = cache.prompt_lens.get(b).copied().unwrap_or(cache.past_seq);
            cache.past_seq.saturating_sub(pl)
        })
        .collect()
}
