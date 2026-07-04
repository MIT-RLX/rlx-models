// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Metal + GGUF prefill strategy for [`crate::Llama32Generator`].

use rlx_runtime::Device;

/// How Metal runs GGUF prefill in [`crate::Llama32Generator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MetalGgufPrefillMode {
    /// Read `RLX_METAL_PACKED_PREFILL` / `RLX_METAL_F32_PREFILL_CPU` (default: CPU F32).
    #[default]
    Auto,
    /// CPU F32 host-dequant prefill (reference parity).
    CpuF32,
    /// Packed GGUF on Metal: CPU F32 logits + packed KV (Metal packed lm_head NaN on 156k vocab).
    PackedGguf,
    /// On-device Metal F32 prefill (thunk path). Weights stay mmap'd via GGUF
    /// loader until first compile — no eager host F32 drain.
    MetalF32,
}

impl MetalGgufPrefillMode {
    /// Parse CLI / config strings (`auto`, `cpu`, `packed`, `metal`, …).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "cpu" | "cpu_f32" | "f32" => Some(Self::CpuF32),
            "packed" | "packed_gguf" | "gguf" => Some(Self::PackedGguf),
            "metal" | "metal_f32" => Some(Self::MetalF32),
            _ => None,
        }
    }

    /// Apply explicit mode or env overrides when [`Self::Auto`].
    pub fn resolve(self) -> Self {
        if self != Self::Auto {
            return self;
        }
        if matches!(
            rlx_ir::env::var("RLX_METAL_F32_PREFILL_CPU").as_deref(),
            Some("1") | Some("true") | Some("TRUE")
        ) {
            return Self::CpuF32;
        }
        if matches!(
            rlx_ir::env::var("RLX_METAL_PACKED_PREFILL").as_deref(),
            Some("1") | Some("true") | Some("TRUE")
        ) {
            return Self::PackedGguf;
        }
        if matches!(
            rlx_ir::env::var("RLX_METAL_F32_PREFILL_CPU").as_deref(),
            Some("0") | Some("false") | Some("FALSE")
        ) {
            return Self::MetalF32;
        }
        Self::CpuF32
    }

    pub fn use_packed_gguf(self) -> bool {
        matches!(self.resolve(), Self::PackedGguf)
    }

    pub fn prefill_device(self, inference_device: Device) -> Device {
        let mode = self.resolve();
        match inference_device {
            Device::Metal => match mode {
                Self::PackedGguf | Self::MetalF32 => Device::Metal,
                Self::CpuF32 | Self::Auto => Device::Cpu,
            },
            // GGUF Q4 prefill on wgpu/Vulkan lacks parity — always seed KV on CPU.
            Device::Gpu | Device::Vulkan => Device::Cpu,
            // CUDA/ROCm: default Auto/CpuF32 → host prefill (avoids 16 GiB OOM on
            // Orpheus 3B); opt into device prefill via PackedGguf / MetalF32 mode.
            Device::Cuda | Device::Rocm => match mode {
                Self::PackedGguf | Self::MetalF32 => inference_device,
                Self::CpuF32 | Self::Auto => Device::Cpu,
            },
            Device::Ane => Device::Cpu,
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_auto_prefill_routes_to_cpu() {
        assert_eq!(
            MetalGgufPrefillMode::Auto.prefill_device(Device::Cuda),
            Device::Cpu
        );
        assert_eq!(
            MetalGgufPrefillMode::CpuF32.prefill_device(Device::Cuda),
            Device::Cpu
        );
    }

    #[test]
    fn cuda_packed_prefill_stays_on_device() {
        assert_eq!(
            MetalGgufPrefillMode::PackedGguf.prefill_device(Device::Cuda),
            Device::Cuda
        );
    }
}

/// Back-compat: env-only packed check ([`MetalGgufPrefillMode::Auto`]).
pub fn metal_use_packed_gguf_prefill() -> bool {
    MetalGgufPrefillMode::Auto.use_packed_gguf()
}

/// Back-compat: env-only prefill device ([`MetalGgufPrefillMode::Auto`]).
pub fn prefill_device_for(device: Device) -> Device {
    MetalGgufPrefillMode::Auto.prefill_device(device)
}
