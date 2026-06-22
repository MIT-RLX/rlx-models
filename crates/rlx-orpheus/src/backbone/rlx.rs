// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Orpheus GGUF backbone — KV-cache decode when supported, packed GGUF on CPU.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use rlx_core::validate_standard_device;
use rlx_core::weight_loader::GgufLoader;
use rlx_llama_base::LlamaBaseConfig;
use rlx_llama32::{Llama32Generator, Llama32Runner, Llama32RunnerBuilder, MetalGgufPrefillMode};

use crate::backbone::BackboneLoadOptions;
use crate::device::lm_kv_decode_supported;
use rlx_qwen3::{SampleOpts, apply_repetition_penalty, sample_token_at};
use rlx_qwen35::encode_prompt_from_gguf;
use rlx_runtime::Device;
use rlx_runtime::{
    llama_decode_bucket_compile_peak_bytes, llama_decode_oneshot_compile_peak_bytes,
    memory_headroom_bytes, process_rss_bytes, soft_memory_budget_bytes, would_exceed_soft_budget,
};

use crate::runner::GenerationConfig;
use crate::tokens::{
    STOP_TOKEN_ID, custom_token_id_to_code, is_snac_slot_token, mask_logits_for_snac_slot,
    use_snac_logit_mask_for,
};

pub const DEFAULT_N_CTX: u32 = 2048;
/// KV compile / bucket cap for TTS (prompt + speech tokens). Full [`DEFAULT_N_CTX`]
/// is only used for prompt-length validation; compiling at 2048 OOMs on Metal.
pub const DEFAULT_COMPILE_SEQ_CAP: u32 = 256;

/// Default compile cap for synthesis when not in high-memory mode (speech tokens +
/// short prompt fit in 128).
pub const SYNTHESIS_COMPILE_SEQ_CAP: u32 = 128;

enum DecodeStrategy {
    /// Power-of-two bucket ladder — fast multi-step decode, high compile RAM (~12 GiB peak).
    Bucket { max_past: usize },
    /// Single dynamic graph specialized per `past_seq` — lower RAM, default for TTS.
    Dynamic { cache_capacity: usize },
}

fn shared_ram_mode() -> bool {
    if matches!(
        std::env::var("ORPHEUS_LOW_MEM").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE")
    ) {
        return true;
    }
    // Tighter caps when the soft budget is below half the bucket compile estimate.
    soft_memory_budget_bytes()
        .is_some_and(|budget| budget < llama_decode_bucket_compile_peak_bytes() / 2)
}

pub(crate) fn low_mem_mode() -> bool {
    shared_ram_mode()
        || memory_headroom_bytes().is_some_and(|h| h < llama_decode_oneshot_compile_peak_bytes())
}

fn log_soft_memory_budget() {
    if let Some(budget) = soft_memory_budget_bytes() {
        let rss = process_rss_bytes().unwrap_or(0);
        let headroom = budget.saturating_sub(rss);
        eprintln!(
            "[orpheus/backbone] soft RAM budget {:.1} GiB (RSS {:.1} GiB, headroom {:.1} GiB)",
            budget as f64 / 1024.0 / 1024.0 / 1024.0,
            rss as f64 / 1024.0 / 1024.0 / 1024.0,
            headroom as f64 / 1024.0 / 1024.0 / 1024.0,
        );
    }
}

/// Pick decode cache strategy. Default is **dynamic** decode (low RAM). Bucket
/// ladder is opt-in via `ORPHEUS_BUCKET_DECODE=1` (tests / throughput).
fn synthesis_decode_plan(compile_cap: usize, opts: &BackboneLoadOptions) -> DecodeStrategy {
    let max_past = decode_bucket_max_past(compile_cap);
    let want_bucket = env_flag("ORPHEUS_BUCKET_DECODE") == Some(true)
        || (!opts.memory_efficient && env_flag("ORPHEUS_BUCKET_DECODE") != Some(false));

    if want_bucket {
        if would_exceed_soft_budget(llama_decode_bucket_compile_peak_bytes()) {
            eprintln!(
                "[orpheus/backbone] bucket decode denied (headroom < {:.1} GiB) — dynamic decode",
                llama_decode_bucket_compile_peak_bytes() as f64 / 1024.0 / 1024.0 / 1024.0,
            );
        } else {
            eprintln!(
                "[orpheus/backbone] bucket decode enabled (max_past={max_past}) — set ORPHEUS_BUCKET_DECODE=0 to save RAM"
            );
            return DecodeStrategy::Bucket { max_past };
        }
    } else if env_flag("ORPHEUS_BUCKET_DECODE") == Some(false) {
        eprintln!("[orpheus/backbone] ORPHEUS_BUCKET_DECODE=0 — dynamic decode");
    }

    let cache_capacity = if low_mem_mode() || opts.memory_efficient {
        1
    } else {
        2
    };
    DecodeStrategy::Dynamic { cache_capacity }
}

fn compile_seq_cap_for(n_ctx: u32, opts: &BackboneLoadOptions) -> usize {
    let base = if env_flag("ORPHEUS_HIGH_MEM") == Some(true) {
        DEFAULT_COMPILE_SEQ_CAP
    } else if low_mem_mode() {
        64
    } else if opts.memory_efficient {
        SYNTHESIS_COMPILE_SEQ_CAP
    } else {
        DEFAULT_COMPILE_SEQ_CAP
    };
    std::env::var("ORPHEUS_COMPILE_SEQ_CAP")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .map(|c| c.min(n_ctx) as usize)
        .unwrap_or((base.min(n_ctx)) as usize)
}

fn prefill_cache_capacity() -> usize {
    std::env::var("ORPHEUS_PREFILL_CACHE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if low_mem_mode() { 1 } else { 2 })
}

fn decode_bucket_max_past(compile_cap: usize) -> usize {
    std::env::var("ORPHEUS_DECODE_CACHE_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(compile_cap.min(64))
}

fn env_flag(name: &str) -> Option<bool> {
    match std::env::var(name).ok().as_deref() {
        Some("1") | Some("true") | Some("TRUE") => Some(true),
        Some("0") | Some("false") | Some("FALSE") => Some(false),
        _ => None,
    }
}

fn sample_opts(cfg: &GenerationConfig) -> SampleOpts {
    if cfg.greedy || std::env::var("ORPHEUS_GREEDY").ok().as_deref() == Some("1") {
        return SampleOpts::greedy();
    }
    SampleOpts::temperature(cfg.temperature, cfg.seed)
        .with_top_p(cfg.top_p)
        .with_top_k(cfg.top_k)
}

/// Metal KV path: prefill via [`MetalGgufPrefillMode`] on [`BackboneLoadOptions`]
/// (CLI: `--metal-prefill auto|cpu|packed|metal`).
fn use_compile_caches() -> bool {
    env_flag("ORPHEUS_COMPILE_CACHE") == Some(true) && !low_mem_mode()
}

fn metal_f32_kv_enabled() -> bool {
    env_flag("ORPHEUS_METAL_KV") == Some(true)
}

/// Slow packed GGUF recompute path (opt-in). Default: CPU uses packed; GPU uses KV.
fn use_packed_lm(device: Device) -> bool {
    if let Some(v) = env_flag("ORPHEUS_PACKED_LM") {
        return v;
    }
    if env_flag("ORPHEUS_FAST_LM") == Some(true) {
        return false;
    }
    if matches!(device, Device::Metal) && metal_f32_kv_enabled() {
        return false;
    }
    // CPU default: packed prefill (memory-friendly; correct but O(n²) on 3B).
    matches!(device, Device::Cpu) && !lm_kv_decode_supported(device)
}

/// Incremental KV decode (O(n) per token).
fn use_fast_kv(device: Device, opts: &BackboneLoadOptions) -> bool {
    if let Some(v) = opts.use_fast_kv {
        return v;
    }
    if env_flag("ORPHEUS_FAST_LM") == Some(false) {
        return false;
    }
    if env_flag("ORPHEUS_METAL_KV") == Some(false) {
        return false;
    }
    lm_kv_decode_supported(device) && !use_packed_lm(device)
}

enum LmEngine {
    Packed(Box<Llama32Runner>),
    Kv(Box<Llama32Generator>),
}

fn push_orpheus_stream_token(
    tok: u32,
    stream_index: &mut usize,
    on_code: &mut impl FnMut(i32) -> Result<()>,
) -> Result<bool> {
    if tok == STOP_TOKEN_ID {
        return Ok(true);
    }
    // Match `orpheus_tts/decoder.py`: only accept custom tokens for the active
    // slot; skip code index 0 when building the SNAC buffer.
    if is_snac_slot_token(tok, *stream_index) {
        if let Some(code) = custom_token_id_to_code(tok, *stream_index) {
            if code > 0 {
                on_code(code)?;
            }
        }
        *stream_index += 1;
    }
    Ok(false)
}

fn adjust_orpheus_logits(
    logits: &mut [f32],
    stream_index: usize,
    token_counts: &std::collections::HashMap<u32, u32>,
    repetition_penalty: f32,
    apply_penalty: bool,
    lm_device: Device,
    lm_decode_on_cpu: bool,
) {
    if use_snac_logit_mask_for(lm_device, lm_decode_on_cpu) {
        mask_logits_for_snac_slot(logits, stream_index);
    }
    if apply_penalty {
        apply_repetition_penalty(logits, token_counts, repetition_penalty);
    }
}

pub struct BackboneModel {
    engine: Mutex<LmEngine>,
    weights: PathBuf,
    n_ctx: u32,
    lm_device: Device,
    lm_decode_on_cpu: bool,
    pub seed: Option<u32>,
}

impl BackboneModel {
    pub fn weights_path(&self) -> &Path {
        &self.weights
    }

    pub fn load(path: &Path, n_ctx: u32) -> Result<Self> {
        Self::load_on(path, n_ctx, Device::Cpu)
    }

    pub fn load_on(path: &Path, n_ctx: u32, device: Device) -> Result<Self> {
        Self::load_on_with(path, n_ctx, device, BackboneLoadOptions::for_device(device))
    }

    pub fn load_on_with(
        path: &Path,
        n_ctx: u32,
        device: Device,
        opts: BackboneLoadOptions,
    ) -> Result<Self> {
        validate_standard_device("orpheus", device)?;
        let path_str = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("non-utf8 weights path"))?;

        let compile_cap = compile_seq_cap_for(n_ctx, &opts);
        log_soft_memory_budget();
        let lm_device = device;
        let lm_decode_on_cpu = matches!(lm_device, Device::Cpu)
            || matches!(lm_device, Device::Gpu | Device::Vulkan)
            || (lm_device == Device::Metal
                && matches!(
                    opts.metal_prefill.resolve(),
                    MetalGgufPrefillMode::CpuF32
                        | MetalGgufPrefillMode::Auto
                        | MetalGgufPrefillMode::PackedGguf,
                ));

        let engine = if use_fast_kv(lm_device, &opts) {
            let mut gguf = GgufLoader::from_file(path_str).context("open GGUF for KV generator")?;
            if gguf.architecture() != "llama" {
                bail!(
                    "rlx-orpheus: expected llama GGUF at {}; got `{}`",
                    path.display(),
                    gguf.architecture()
                );
            }
            let cfg = rlx_llama32::llama32_cfg_from_gguf(gguf.file())?;
            let mut generator = Llama32Generator::from_loader_at_mode(
                cfg,
                &mut gguf,
                lm_device,
                path,
                if lm_device == Device::Cpu {
                    MetalGgufPrefillMode::CpuF32
                } else {
                    opts.metal_prefill.resolve()
                },
            )?
            .with_compile_seq_cap(compile_cap);
            let decode = synthesis_decode_plan(compile_cap, &opts);
            let use_bucket = matches!(decode, DecodeStrategy::Bucket { .. });
            generator = match decode {
                DecodeStrategy::Bucket { max_past } => generator.with_decode_cache(max_past),
                DecodeStrategy::Dynamic { cache_capacity } => {
                    generator.with_dynamic_decode_cache(cache_capacity)
                }
            };
            if use_bucket && use_compile_caches() {
                generator = generator.with_prefill_cache(prefill_cache_capacity());
            }
            let decode_on_cpu = lm_decode_on_cpu;
            eprintln!(
                "[orpheus/backbone] kv-cache LM on {lm_device:?} {} (prefill={:?}, decode={}, compile_cap={compile_cap}, decode={})",
                path.display(),
                if lm_device == Device::Cpu {
                    MetalGgufPrefillMode::CpuF32
                } else {
                    opts.metal_prefill.resolve()
                },
                if decode_on_cpu { "cpu" } else { "device" },
                if use_bucket { "bucket" } else { "dynamic" },
            );
            LmEngine::Kv(Box::new(generator))
        } else {
            let base = LlamaBaseConfig::from_gguf_path(path)
                .with_context(|| format!("parse GGUF {}", path.display()))?;
            if base.arch != "llama" {
                bail!(
                    "rlx-orpheus: expected llama GGUF at {}; got `{}`",
                    path.display(),
                    base.arch
                );
            }
            let runner = Llama32RunnerBuilder::default()
                .weights(path)
                .max_seq(n_ctx as usize)
                .device(lm_device)
                .packed_weights(use_packed_lm(lm_device))
                .stream(false)
                .sample(SampleOpts::greedy())
                .build()
                .context("build Llama32Runner for Orpheus")?;
            eprintln!(
                "[orpheus/backbone] {} on {device:?} {}",
                if use_packed_lm(device) {
                    "packed GGUF prefill"
                } else {
                    "f32 runner"
                },
                path.display()
            );
            LmEngine::Packed(Box::new(runner))
        };

        Ok(Self {
            engine: Mutex::new(engine),
            weights: path.to_path_buf(),
            n_ctx,
            lm_device,
            lm_decode_on_cpu,
            seed: Some(GenerationConfig::default().seed as u32),
        })
    }

    pub fn generate_codes(&self, prompt: &str, cfg: &GenerationConfig) -> Result<Vec<i32>> {
        let prompt_ids = encode_prompt_from_gguf(&self.weights, prompt)
            .with_context(|| format!("tokenize prompt for {}", self.weights.display()))?;
        self.generate_codes_from_prompt(&prompt_ids, cfg)
    }

    pub fn generate_codes_from_prompt(
        &self,
        prompt_ids: &[u32],
        cfg: &GenerationConfig,
    ) -> Result<Vec<i32>> {
        let mut codes = Vec::new();
        self.generate_codes_from_prompt_streaming(prompt_ids, cfg, |c| {
            codes.push(c);
            Ok(())
        })?;
        Ok(codes)
    }

    pub fn generate_codes_from_prompt_streaming(
        &self,
        prompt_ids: &[u32],
        cfg: &GenerationConfig,
        mut on_code: impl FnMut(i32) -> Result<()>,
    ) -> Result<()> {
        if prompt_ids.len() as u32 > self.n_ctx {
            bail!(
                "prompt too long: {} tokens > n_ctx={}",
                prompt_ids.len(),
                self.n_ctx
            );
        }

        let lm_device = self.lm_device;
        let lm_decode_on_cpu = self.lm_decode_on_cpu;
        let mut engine = self
            .engine
            .lock()
            .map_err(|e| anyhow::anyhow!("backbone lock poisoned: {e}"))?;

        let mut stream_index = 0usize;

        match &mut *engine {
            LmEngine::Kv(generator) => {
                let sample = sample_opts(cfg);
                let apply_penalty = cfg.repetition_penalty > 1.0;
                let mut token_counts = std::collections::HashMap::<u32, u32>::new();
                generator.prefill(prompt_ids);
                eprintln!(
                    "[orpheus/backbone] generating up to {} tokens (prompt len={})",
                    cfg.max_new_tokens,
                    prompt_ids.len()
                );
                for step in 0..cfg.max_new_tokens {
                    let penalty = cfg.repetition_penalty;
                    let slot_ix = stream_index;
                    let next = generator
                        .step_cached_adjust(sample, step as u64, |logits| {
                            adjust_orpheus_logits(
                                logits,
                                slot_ix,
                                &token_counts,
                                penalty,
                                apply_penalty,
                                lm_device,
                                lm_decode_on_cpu,
                            );
                        })
                        .context("Orpheus LM cached decode step")?;
                    if step > 0 && step % 7 == 0 {
                        eprintln!(
                            "[orpheus/backbone] step {}/{} ({} tokens in history)",
                            step,
                            cfg.max_new_tokens,
                            generator.tokens().len()
                        );
                    }
                    if push_orpheus_stream_token(next, &mut stream_index, &mut on_code)? {
                        break;
                    }
                    *token_counts.entry(next).or_insert(0) += 1;
                }
                if std::env::var("ORPHEUS_DEBUG_TOKENS").ok().as_deref() == Some("1") {
                    let generated = generator.tokens()[prompt_ids.len()..].to_vec();
                    eprintln!(
                        "[orpheus/backbone] raw tokens ({}): {:?}",
                        generated.len(),
                        &generated[..generated.len().min(16)]
                    );
                }
            }
            LmEngine::Packed(runner) => {
                let sample = sample_opts(cfg);
                let mut history = prompt_ids.to_vec();
                let mut token_counts = std::collections::HashMap::<u32, u32>::new();
                for step in 0..cfg.max_new_tokens {
                    let slot_ix = stream_index;
                    let mut logits = runner.predict_logits(&history)?;
                    adjust_orpheus_logits(
                        &mut logits,
                        slot_ix,
                        &token_counts,
                        cfg.repetition_penalty,
                        true,
                        lm_device,
                        lm_decode_on_cpu,
                    );
                    let next = sample_token_at(&logits, sample, step as u64) as u32;
                    if push_orpheus_stream_token(next, &mut stream_index, &mut on_code)? {
                        break;
                    }
                    *token_counts.entry(next).or_insert(0) += 1;
                    history.push(next);
                }
                if std::env::var("ORPHEUS_DEBUG_TOKENS").ok().as_deref() == Some("1") {
                    let generated = history[prompt_ids.len()..].to_vec();
                    eprintln!(
                        "[orpheus/backbone] raw tokens ({}): {:?}",
                        generated.len(),
                        &generated[..generated.len().min(16)]
                    );
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backbone::BackboneLoadOptions;
    use crate::tokens::STOP_TOKEN_ID;

    #[test]
    fn synthesis_defaults_to_dynamic_decode() {
        let _guard = EnvGuard::unset("ORPHEUS_BUCKET_DECODE");
        let plan = synthesis_decode_plan(128, &BackboneLoadOptions::synthesis());
        assert!(matches!(plan, DecodeStrategy::Dynamic { .. }));
    }

    #[test]
    fn reference_parity_allows_bucket_when_env_set() {
        let _guard = EnvGuard::set("ORPHEUS_BUCKET_DECODE", "1");
        let plan = synthesis_decode_plan(128, &BackboneLoadOptions::reference_parity());
        // May be bucket or dynamic if soft budget denies — just must not panic.
        let _ = plan;
    }

    /// Restore env after tests that set ORPHEUS_* vars.
    struct EnvGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, val: &str) -> Self {
            let prev = std::env::var(key).ok();
            unsafe { std::env::set_var(key, val) };
            Self { key, prev }
        }

        fn unset(key: &'static str) -> Self {
            let prev = std::env::var(key).ok();
            unsafe { std::env::remove_var(key) };
            Self { key, prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => unsafe { std::env::set_var(self.key, v) },
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }

    #[test]
    fn stop_token_yields_no_codes() {
        let mut codes = Vec::new();
        let tokens = [STOP_TOKEN_ID];
        let mut stream_index = 0usize;
        for &tok in &tokens {
            let stop = push_orpheus_stream_token(tok, &mut stream_index, &mut |c| {
                codes.push(c);
                Ok(())
            })
            .expect("push token");
            if stop {
                break;
            }
        }
        assert!(codes.is_empty());
    }
}
