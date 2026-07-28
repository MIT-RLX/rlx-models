// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Per-backend sizing for Nanbeige Looped Transformers.
// `num_loops > 1` multiplies KV slots (`kv_layers = physical * loops`) while
// weights stay shared — OOM risk is mostly KV + compile arenas, not params.

use anyhow::{Result, bail};
use rlx_llama32::Llama32Config;
use rlx_runtime::Device;

/// Bytes for past_k + past_v across all loop-unrolled KV layers at `seq`.
pub fn kv_cache_bytes(cfg: &Llama32Config, seq: usize) -> usize {
    let kv_dim = cfg.kv_proj_dim();
    cfg.kv_layers()
        .saturating_mul(2)
        .saturating_mul(seq)
        .saturating_mul(kv_dim)
        .saturating_mul(4)
}

/// Rough F32 parameter bytes for the physical stack (shared across loops).
pub fn approx_param_bytes_f32(cfg: &Llama32Config) -> usize {
    let h = cfg.hidden_size;
    let q = cfg.q_proj_dim();
    let kv = cfg.kv_proj_dim();
    let ff = cfg.intermediate_size;
    let v = cfg.vocab_size;
    let per_layer = (q * h) + (2 * kv * h) + (h * q) + (3 * ff * h) + (2 * h);
    let embed = v * h;
    let head = if cfg.tie_word_embeddings { 0 } else { v * h };
    let norm = h;
    (embed + head + norm + cfg.physical_layers() * per_layer) * 4
}

/// Working-set estimate: params + KV at `max_seq` + small activation slack.
pub fn working_set_bytes(cfg: &Llama32Config, max_seq: usize) -> usize {
    let act = cfg.hidden_size.saturating_mul(max_seq).saturating_mul(4) * 8;
    approx_param_bytes_f32(cfg)
        .saturating_add(kv_cache_bytes(cfg, max_seq))
        .saturating_add(act)
}

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok()?.parse().ok()
}

fn device_mem_budget_bytes(device: Device) -> Option<usize> {
    if let Ok(b) = std::env::var("RLX_NANBEIGE_MEM_BUDGET_BYTES") {
        return b.parse().ok();
    }
    // Looped Transformer: weights shared, but KV + compile arenas scale with
    // `num_loops`. Stay well below physical RAM so Metal/MLX don't thrash.
    let frac = match device {
        Device::Cpu => 0.25,
        Device::Metal | Device::Mlx => 0.40,
        Device::Cuda | Device::Rocm => 0.45,
        Device::Gpu | Device::Vulkan => 0.08,
        Device::Ane => 0.10,
        _ => 0.20,
    };
    rlx_runtime::memory_estimate::available_unified_memory()
        .map(|t| ((t as f64) * frac).floor() as usize)
        .or(match device {
            Device::Cpu => Some(8usize << 30),
            Device::Cuda | Device::Rocm => Some(10usize << 30),
            Device::Gpu | Device::Vulkan => Some(512usize << 20),
            _ => None,
        })
}

/// Clamp `want` so estimated working set fits the device budget.
pub fn clamp_max_seq(cfg: &Llama32Config, device: Device, want: usize) -> usize {
    let want = want.max(16);
    let Some(budget) = device_mem_budget_bytes(device) else {
        return want.min(256);
    };
    let params = approx_param_bytes_f32(cfg);
    if params >= budget {
        // Params alone exceed budget — only tiny synthetic graphs are safe.
        return 32.min(want);
    }
    let mut lo = 16usize;
    let mut hi = want;
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if working_set_bytes(cfg, mid) <= budget {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// Per-backend inference plan for Nanbeige (synth or full 3B).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendPlan {
    pub max_seq: usize,
    pub prompt_len: usize,
    pub decode_tokens: usize,
    pub bucketed_decode: bool,
    pub prefill_cache: usize,
    /// Full BF16/F32 3B checkpoint is expected to fit.
    pub allow_full_f32: bool,
    pub note: &'static str,
}

impl BackendPlan {
    pub fn for_device(cfg: &Llama32Config, device: Device) -> Self {
        let base = match device {
            Device::Mlx => Self {
                // Full 3B F32 ≈15 GiB; keep seq short so KV+compile arenas fit.
                max_seq: 128,
                prompt_len: 32,
                decode_tokens: 16,
                // Bucketed decode keeps one resident F32 graph (oneshot re-uploads
                // ~15 GiB per token and trips the soft RAM gate after Compiled prefill).
                bucketed_decode: true,
                prefill_cache: 4,
                allow_full_f32: true,
                note: "MLX preferred on Apple Silicon (bucketed decode for dense F32)",
            },
            Device::Metal => Self {
                max_seq: 96,
                prompt_len: 24,
                decode_tokens: 12,
                bucketed_decode: true,
                prefill_cache: 4,
                allow_full_f32: true,
                note: "Metal: short seq + bucketed decode; prefer MLX",
            },
            Device::Cuda | Device::Rocm => Self {
                max_seq: 128,
                prompt_len: 24,
                decode_tokens: 12,
                bucketed_decode: true,
                prefill_cache: 4,
                allow_full_f32: true,
                note: "discrete GPU — short seq for 2× KV",
            },
            Device::Gpu => Self {
                max_seq: 64,
                prompt_len: 16,
                decode_tokens: 8,
                bucketed_decode: false,
                prefill_cache: 2,
                allow_full_f32: false,
                note: "wgpu: skip full 3B F32 (storage bind); synth only",
            },
            Device::Vulkan => Self {
                max_seq: 96,
                prompt_len: 16,
                decode_tokens: 8,
                bucketed_decode: false,
                prefill_cache: 2,
                allow_full_f32: false,
                note: "Vulkan: skip full 3B F32; synth / GGUF only",
            },
            Device::Cpu => Self {
                max_seq: 64,
                prompt_len: 16,
                decode_tokens: 8,
                bucketed_decode: true,
                prefill_cache: 2,
                allow_full_f32: false,
                note: "CPU: synth/GGUF only for full 3B (F32 thrash risk)",
            },
            Device::Ane => Self {
                max_seq: 64,
                prompt_len: 16,
                decode_tokens: 4,
                bucketed_decode: false,
                prefill_cache: 1,
                allow_full_f32: false,
                note: "CoreML validates; full LM not targeted",
            },
            _ => Self {
                max_seq: 64,
                prompt_len: 16,
                decode_tokens: 8,
                bucketed_decode: false,
                prefill_cache: 2,
                allow_full_f32: false,
                note: "conservative defaults",
            },
        };

        let max_seq = env_usize("RLX_NANBEIGE_MAX_SEQ")
            .unwrap_or_else(|| clamp_max_seq(cfg, device, base.max_seq));
        let prompt_len = env_usize("RLX_NANBEIGE_PROMPT_LEN")
            .unwrap_or(base.prompt_len)
            .min(max_seq.saturating_sub(base.decode_tokens.max(1)))
            .max(1);
        let decode_tokens = env_usize("RLX_NANBEIGE_DECODE_TOKENS")
            .unwrap_or(base.decode_tokens)
            .min(max_seq.saturating_sub(prompt_len).max(1));

        Self {
            max_seq,
            prompt_len,
            decode_tokens,
            ..base
        }
    }
}

/// Fail fast when a full F32/BF16 3B run would exceed the device budget.
///
/// With mmap-on-take + releasing host params after upload, accelerators keep
/// about **one** F32 model copy on device (plus KV). Budget uses `1.25 × params + KV`.
pub fn assert_full_model_fits(cfg: &Llama32Config, device: Device, max_seq: usize) -> Result<()> {
    let plan = BackendPlan::for_device(cfg, device);
    if !plan.allow_full_f32 {
        bail!(
            "nanbeige: device {device:?} is not configured for full F32/BF16 3B \
             ({}); use GGUF packed weights or a synth bench",
            plan.note
        );
    }
    let params = approx_param_bytes_f32(cfg);
    let kv = kv_cache_bytes(cfg, max_seq);
    let need = match device {
        Device::Cpu => params.saturating_add(kv),
        _ => {
            // Device-resident F32 + compile slack (host copies released after upload).
            ((params as f64) * 1.25) as usize + kv
        }
    };
    if let Some(budget) = device_mem_budget_bytes(device) {
        if need > budget {
            bail!(
                "nanbeige: estimated working set {:.2} GiB exceeds {:.2} GiB budget on {device:?} \
                 (max_seq={max_seq}, kv_layers={}, loops={}) — use GGUF packed or more RAM",
                need as f64 / (1024.0 * 1024.0 * 1024.0),
                budget as f64 / (1024.0 * 1024.0 * 1024.0),
                cfg.kv_layers(),
                cfg.num_loops
            );
        }
    }
    Ok(())
}

/// Apply process-wide defaults that help Nanbeige on `device` (idempotent-ish).
///
/// Call **before** the first MLX graph compile. Nanbeige looped prefill/decode
/// graphs are ~2k nodes; the MLX default compile cap (1536) forces Lazy and
/// makes real 3B F32 impractically slow.
pub fn prepare(device: Device) {
    // SAFETY: single-threaded prepare before concurrent runtime use.
    unsafe {
        if matches!(device, Device::Mlx) && std::env::var_os("RLX_MLX_COMPILE_MAX_NODES").is_none()
        {
            // Prefill ≈2013 nodes, decode ≈1930 for Nanbeige4.2-3B (22×2 loops).
            std::env::set_var("RLX_MLX_COMPILE_MAX_NODES", "4096");
        }
        if matches!(device, Device::Mlx | Device::Metal) {
            // Dense F32 3B leaves high RSS after Compiled prefill; default 80%
            // soft cap then rejects decode (+4–12 GiB peak estimate).
            if std::env::var_os("RLX_SOFT_MEMORY_FRACTION").is_none() {
                std::env::set_var("RLX_SOFT_MEMORY_FRACTION", "0.95");
            }
            // Orpheus Q4 bucket peaks (~12 GiB) overstate mmap F32 IR/temps.
            if std::env::var_os("RLX_DECODE_BUCKET_PEAK_BYTES").is_none() {
                std::env::set_var(
                    "RLX_DECODE_BUCKET_PEAK_BYTES",
                    (2usize * 1024 * 1024 * 1024).to_string(),
                );
            }
            if std::env::var_os("RLX_DECODE_ONESHOT_PEAK_BYTES").is_none() {
                std::env::set_var(
                    "RLX_DECODE_ONESHOT_PEAK_BYTES",
                    (1024usize * 1024 * 1024).to_string(),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nanbeige42_3b_preset;

    #[test]
    fn kv_scales_with_loops() {
        let mut cfg = nanbeige42_3b_preset();
        let once = kv_cache_bytes(&cfg, 128);
        cfg.num_loops = 1;
        let half = kv_cache_bytes(&cfg, 128);
        assert_eq!(once, half * 2);
    }

    #[test]
    fn wgpu_rejects_full_f32() {
        let cfg = nanbeige42_3b_preset();
        let plan = BackendPlan::for_device(&cfg, Device::Gpu);
        assert!(!plan.allow_full_f32);
        assert!(assert_full_model_fits(&cfg, Device::Gpu, 64).is_err());
    }

    #[test]
    fn metal_allows_full_when_budget_ok() {
        let cfg = nanbeige42_3b_preset();
        let plan = BackendPlan::for_device(&cfg, Device::Metal);
        assert!(plan.allow_full_f32);
        assert!(plan.bucketed_decode);
        assert!(plan.max_seq >= 64);
        // On a small-RAM host assert_full_model_fits may still reject;
        // plan itself stays optimistic for Metal.
    }

    #[test]
    fn mlx_enables_bucketed_decode_for_dense_f32() {
        let cfg = nanbeige42_3b_preset();
        let plan = BackendPlan::for_device(&cfg, Device::Mlx);
        assert!(plan.bucketed_decode);
        assert!(plan.allow_full_f32);
    }

    #[test]
    fn clamp_shrinks_when_budget_tiny() {
        let cfg = nanbeige42_3b_preset();
        // Safety: clamp never returns below 16.
        let s = clamp_max_seq(&cfg, Device::Gpu, 512);
        assert!(s <= 96);
        assert!(s >= 16);
    }
}
