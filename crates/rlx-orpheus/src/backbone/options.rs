// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Load-time options for the Orpheus GGUF backbone.

use rlx_llama32::MetalGgufPrefillMode;
use rlx_runtime::Device;

/// Options when constructing [`crate::backbone::BackboneModel`].
#[derive(Debug, Clone)]
pub struct BackboneLoadOptions {
    /// Metal GGUF prefill strategy for the KV generator path.
    pub metal_prefill: MetalGgufPrefillMode,
    /// Force incremental KV decode vs packed runner. `None` uses env
    /// (`ORPHEUS_METAL_KV`, `ORPHEUS_PACKED_LM`, …).
    pub use_fast_kv: Option<bool>,
    /// When true, prefer dynamic per-`past_seq` decode (lower peak compile RAM).
    /// When false (default for [`Self::for_tts`]), bucket decode is used when the
    /// soft RAM budget allows — avoids unbounded Metal graph growth during TTS.
    pub memory_efficient: bool,
}

impl Default for BackboneLoadOptions {
    fn default() -> Self {
        Self::synthesis()
    }
}

impl BackboneLoadOptions {
    /// Low-RAM TTS: CPU F32 GGUF prefill + CPU decode on Metal hosts (reference path).
    pub fn synthesis() -> Self {
        Self {
            metal_prefill: MetalGgufPrefillMode::CpuF32,
            use_fast_kv: Some(true),
            memory_efficient: true,
        }
    }

    /// GPU TTS on Metal / CUDA / ROCm: on-device prefill + decode (mmap GGUF).
    /// wgpu / Vulkan use CPU GGUF prefill + decode (parity) with SNAC logit mask.
    pub fn gpu(device: Device) -> Self {
        Self {
            metal_prefill: metal_prefill_for_device(device),
            use_fast_kv: Some(true),
            memory_efficient: true,
        }
    }

    /// Pick GPU path when an accelerator is available and low-memory mode is off.
    pub fn for_device(device: Device) -> Self {
        Self::for_tts(device)
    }

    /// TTS-optimized load path for every RLX backend (Metal, CUDA, ROCm, wgpu, CPU).
    ///
    /// * CPU / low-RAM: [`Self::synthesis`] — CpuF32 prefill, incremental KV decode.
    /// * Metal: CPU GGUF prefill + bucket decode (Q8_0 Metal HIR decode diverges today).
    /// * CUDA / ROCm: on-device prefill + bucket decode (bounded compile RAM).
    /// * wgpu / Vulkan: CPU GGUF prefill + portable GPU KV decode.
    pub fn for_tts(device: Device) -> Self {
        if super::rlx::low_mem_mode() {
            return Self::synthesis();
        }
        match device {
            Device::Cpu => Self::synthesis(),
            Device::Metal | Device::Cuda | Device::Rocm => Self {
                metal_prefill: metal_prefill_for_device(device),
                use_fast_kv: Some(true),
                memory_efficient: false,
            },
            Device::Gpu | Device::Vulkan => Self {
                metal_prefill: MetalGgufPrefillMode::CpuF32,
                use_fast_kv: Some(true),
                memory_efficient: false,
            },
            _ => Self::synthesis(),
        }
    }

    /// Reference parity / throughput: bucket decode allowed when RAM budget permits.
    pub fn reference_parity() -> Self {
        Self {
            metal_prefill: MetalGgufPrefillMode::CpuF32,
            use_fast_kv: Some(true),
            memory_efficient: false,
        }
    }

    /// Experimental: packed Metal KV + CPU F32 logits.
    pub fn fast_prefill() -> Self {
        Self {
            metal_prefill: MetalGgufPrefillMode::PackedGguf,
            use_fast_kv: Some(true),
            memory_efficient: true,
        }
    }

    pub fn with_memory_efficient(mut self, enabled: bool) -> Self {
        self.memory_efficient = enabled;
        self
    }

    pub fn with_metal_prefill(mut self, mode: MetalGgufPrefillMode) -> Self {
        self.metal_prefill = mode;
        self
    }

    pub fn with_fast_kv(mut self, enabled: bool) -> Self {
        self.use_fast_kv = Some(enabled);
        self
    }
}

fn metal_prefill_for_device(device: Device) -> MetalGgufPrefillMode {
    if let Ok(s) = std::env::var("ORPHEUS_METAL_PREFILL") {
        if let Some(m) = MetalGgufPrefillMode::parse(&s) {
            return m;
        }
    }
    match device {
        // Default reference path: Llama32Generator routes GGUF to CPU HIR on Metal.
        Device::Metal | Device::Gpu | Device::Vulkan => MetalGgufPrefillMode::CpuF32,
        _ => MetalGgufPrefillMode::Auto,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_on_metal_uses_cpu_gguf_prefill() {
        let opts = BackboneLoadOptions::gpu(Device::Metal);
        assert_eq!(opts.metal_prefill, MetalGgufPrefillMode::CpuF32);
    }

    #[test]
    fn for_device_wgpu_uses_cpu_gguf_path() {
        let opts = BackboneLoadOptions::for_device(Device::Gpu);
        assert_eq!(opts.metal_prefill, MetalGgufPrefillMode::CpuF32);
        assert_eq!(opts.use_fast_kv, Some(true));
    }

    #[test]
    fn for_device_matches_for_tts() {
        let opts = BackboneLoadOptions::for_device(Device::Metal);
        let tts = BackboneLoadOptions::for_tts(Device::Metal);
        assert_eq!(opts.metal_prefill, tts.metal_prefill);
        assert_eq!(opts.memory_efficient, tts.memory_efficient);
    }

    #[test]
    fn for_tts_enables_bucket_decode_budget_on_metal() {
        let opts = BackboneLoadOptions::for_tts(Device::Metal);
        assert_eq!(opts.metal_prefill, MetalGgufPrefillMode::CpuF32);
        assert!(!opts.memory_efficient);
    }

    #[test]
    fn for_tts_cpu_uses_synthesis() {
        let opts = BackboneLoadOptions::for_tts(Device::Cpu);
        assert!(opts.memory_efficient);
        assert_eq!(opts.metal_prefill, MetalGgufPrefillMode::CpuF32);
    }
}
