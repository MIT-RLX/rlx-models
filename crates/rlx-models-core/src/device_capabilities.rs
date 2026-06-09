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

//! Shared backend policy for RLX model crates.
//!
//! Every model family in this workspace targets the same seven execution
//! backends. Call [`validate_standard_device`] at runner / loader build
//! time; enable matching `rlx-runtime` features on the model crate
//! (`metal`, `mlx`, `cuda`, `rocm`, `gpu`, `vulkan`, or `all-backends`).

use anyhow::{Result, bail};
use rlx_runtime::{Device, memory_estimate};

/// Backends every model crate is expected to support when the matching
/// `rlx-runtime` feature is enabled at build time.
pub const STANDARD_DEVICES: &[Device] = &[
    Device::Cpu,
    Device::Metal,
    Device::Mlx,
    Device::Cuda,
    Device::Rocm,
    Device::Gpu,
    Device::Vulkan,
];

/// CLI / help string for `--device`.
pub const STANDARD_DEVICE_NAMES: &str = "cpu|metal|mps|mlx|cuda|rocm|hip|gpu|wgpu|vulkan";

/// True when `device` is in [`STANDARD_DEVICES`].
pub fn is_standard_device(device: Device) -> bool {
    STANDARD_DEVICES.contains(&device)
}

/// Fail fast on exotic runtime devices (TPU, ANE, OpenGL, …).
pub fn validate_standard_device(family: &str, device: Device) -> Result<()> {
    if is_standard_device(device) {
        Ok(())
    } else {
        bail!(
            "{family}: device {device:?} is not supported \
             (use {STANDARD_DEVICE_NAMES})"
        )
    }
}

/// `(free_bytes, total_bytes)` for TIDE MoE VRAM budget sizing.
///
/// Override with `RLX_CUDA_FREE_BYTES` / `RLX_CUDA_TOTAL_BYTES` or
/// `RLX_DEVICE_FREE_BYTES` / `RLX_DEVICE_TOTAL_BYTES`. On Apple Silicon
/// (Metal / MLX), falls back to unified memory when env vars are unset.
pub fn device_memory_for_moe_offload(device: Device) -> Option<(usize, usize)> {
    if let (Ok(free), Ok(total)) = (
        std::env::var("RLX_CUDA_FREE_BYTES"),
        std::env::var("RLX_CUDA_TOTAL_BYTES"),
    ) {
        if let (Ok(f), Ok(t)) = (free.parse(), total.parse()) {
            return Some((f, t));
        }
    }
    if let (Ok(free), Ok(total)) = (
        std::env::var("RLX_DEVICE_FREE_BYTES"),
        std::env::var("RLX_DEVICE_TOTAL_BYTES"),
    ) {
        if let (Ok(f), Ok(t)) = (free.parse(), total.parse()) {
            return Some((f, t));
        }
    }
    match device {
        Device::Metal | Device::Mlx => memory_estimate::available_unified_memory().map(|t| (t, t)),
        Device::Cuda | Device::Rocm | Device::Gpu | Device::Vulkan => {
            memory_estimate::available_unified_memory().map(|t| (t, t))
        }
        _ => None,
    }
}

/// SAM v1 also documents `tpu` on [`rlx_sam::Sam::from_safetensors_on`].
pub fn validate_sam_device(family: &str, device: Device) -> Result<()> {
    if device == Device::Tpu || is_standard_device(device) {
        Ok(())
    } else {
        bail!(
            "{family}: device {device:?} is not supported \
             (use {STANDARD_DEVICE_NAMES} or tpu)"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_set_covers_cli_backends() {
        for dev in STANDARD_DEVICES {
            assert!(is_standard_device(*dev));
        }
        assert!(!is_standard_device(Device::Tpu));
    }
}
