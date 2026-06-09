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

//! Per-backend compile options and guards (Metal MPSGraph, wgpu/MLX fusion).

use rlx_core::flow_bridge::compile_options_from_profile;
use rlx_flow::CompileProfile;
use rlx_ir::logical_kernel::KernelDispatchConfig;
use rlx_runtime::{CompileOptions, Device};

/// MoonViT: MLX / wgpu / Vulkan lack in-graph [`Op::AxialRope2d`]; Metal `AxialRope2d` diverges from
/// the CPU reference on large grids — use the same decomposed 1D RoPE path.
pub fn moonvit_use_decomposed_rope(device: Device) -> bool {
    matches!(
        device,
        Device::Metal | Device::Mlx | Device::Gpu | Device::Vulkan
    )
}

/// True when parity-critical graphs (MoonViT, LM prefill/MTP/decode) run on CPU while `requested`
/// remains the user-facing device (logging, future GPU re-enable).
pub fn locateanything_uses_cpu_host(requested: Device) -> bool {
    locateanything_host_device(requested) != requested
}

/// Device for MoonViT, projector, and LM bucket graphs. All GPU backends use CPU host graphs until
/// per-backend HF/CPU parity is verified on real LocateAnything-3B weights.
pub fn locateanything_host_device(requested: Device) -> Device {
    match requested {
        Device::Cpu => Device::Cpu,
        _ => Device::Cpu,
    }
}

/// Device for MoonViT + projector compile/run.
pub fn vision_encode_device(lm_device: Device) -> Device {
    locateanything_host_device(lm_device)
}

/// Device for LM prefill + bucketed MTP/decode compile caches.
pub fn lm_host_device(lm_device: Device) -> Device {
    locateanything_host_device(lm_device)
}

/// GPU-resident KV handles: only when LM actually runs on a GPU backend that passes parity.
pub fn lm_gpu_kv_enabled(requested: Device) -> bool {
    if locateanything_uses_cpu_host(requested) {
        return false;
    }
    matches!(
        requested,
        Device::Mlx | Device::Cuda | Device::Rocm | Device::Gpu | Device::Vulkan
    )
}

/// Bucketed MTP/decode active-extent hints (Metal scaling still diverges from host slice semantics).
pub fn lm_active_extent_enabled(requested: Device) -> bool {
    !locateanything_uses_cpu_host(requested) && requested != Device::Metal
}

/// Disable fused decode layers on backends that do not lower `FusedResidualRmsNorm` yet.
pub fn lm_decode_compile_options(device: Device) -> CompileOptions {
    let mut profile = CompileProfile::llama32_decode();
    if matches!(device, Device::Gpu | Device::Vulkan | Device::Mlx) {
        profile.fusion.skip = true;
    }
    compile_options_from_profile(&profile, device, KernelDispatchConfig::default())
}

pub fn lm_prefill_compile_options(device: Device) -> CompileOptions {
    let mut profile = CompileProfile::llama32_prefill();
    if matches!(device, Device::Gpu | Device::Vulkan) {
        profile.fusion.skip = true;
    }
    compile_options_from_profile(&profile, device, KernelDispatchConfig::default())
}

/// Wrap LM graph compile on Metal (MPSGraph attention reshape issues on some shapes).
pub fn metal_lm_compile_guard<R, F>(device: Device, f: F) -> R
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_device_cpu_only_for_now() {
        assert_eq!(locateanything_host_device(Device::Cpu), Device::Cpu);
        assert_eq!(locateanything_host_device(Device::Metal), Device::Cpu);
        assert_eq!(locateanything_host_device(Device::Mlx), Device::Cpu);
        assert_eq!(locateanything_host_device(Device::Cuda), Device::Cpu);
        assert!(locateanything_uses_cpu_host(Device::Metal));
        assert!(!locateanything_uses_cpu_host(Device::Cpu));
    }

    #[test]
    fn gpu_kv_disabled_on_cpu_host() {
        assert!(!lm_gpu_kv_enabled(Device::Metal));
        assert!(!lm_gpu_kv_enabled(Device::Mlx));
        assert!(!lm_active_extent_enabled(Device::Metal));
    }
}
