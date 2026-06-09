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

//! FLUX.2 backend selection (Metal/MPS, MLX, CUDA, ROCm, wgpu, Vulkan, CPU).

use anyhow::{Result, bail};
use rlx_runtime::{Device, full_name, is_available};

/// True when the denoiser / VAE should use compiled HIR (non-CPU backends).
pub fn flux2_prefers_compiled_hir(device: Device) -> bool {
    !matches!(device, Device::Cpu)
}

/// Text encoder HIR on CUDA compiles a full Qwen3 trunk and can take hours + fill
/// VRAM while the denoiser is still resident. Native CPU encode once, then drop TE.
pub fn flux2_prefers_compiled_te(device: Device) -> bool {
    matches!(device, Device::Metal | Device::Mlx)
}

/// Human-readable Cargo feature needed to enable `device`.
pub fn flux2_device_feature(device: Device) -> Option<&'static str> {
    match device {
        Device::Cpu => None,
        Device::Metal => Some("metal"),
        Device::Mlx => Some("mlx"),
        Device::Cuda => Some("cuda"),
        Device::Gpu => Some("gpu"),
        Device::Rocm => Some("rocm"),
        Device::Vulkan => Some("vulkan"),
        _ => None,
    }
}

/// Fail fast when `--device` was requested but the runtime backend is missing.
pub fn assert_flux2_device_available(device: Device) -> Result<()> {
    if is_available(device) {
        return Ok(());
    }
    let name = full_name(device);
    if let Some(feat) = flux2_device_feature(device) {
        bail!(
            "FLUX.2 backend `{name}` is not available — rebuild with \
             `rlx-models` feature `{feat}` and install drivers/toolchain \
             (CUDA: NVIDIA driver + toolkit; MLX: `vendor/mlx`; ROCm: AMD HIP; \
             gpu/wgpu: portable adapter; vulkan: wgpu Vulkan/MoltenVK)"
        );
    }
    bail!("FLUX.2 backend `{name}` is not available on this host");
}
