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

//! Device selection (`--device`, `RLX_DEVICE`, auto GPU pick).

use anyhow::{Context, Result, ensure};
use rlx_cli::parse_standard_device;
use rlx_core::STANDARD_DEVICE_NAMES;
use rlx_runtime::{Device, is_available};
use std::env;

const FAMILY: &str = "locateanything";

/// Resolve execution device from CLI / env / auto.
///
/// `name` may be `auto`, `cpu`, `metal`, `cuda`, … (`RLX_DEVICE` when `name` is `None`).
pub fn resolve_device(name: Option<&str>) -> Result<Device> {
    let label = name
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| env::var("RLX_DEVICE").ok())
        .unwrap_or_else(|| "auto".to_string());

    if label.eq_ignore_ascii_case("auto") {
        let device = pick_auto_device();
        if device == Device::Cpu {
            eprintln!(
                "[rlx-locateanything] auto → Cpu (no GPU backend in this binary). \
                 On Apple Silicon: `just features=metal locateanything-demo` or \
                 `cargo build -p rlx-locateanything --features metal`"
            );
        }
        ensure_backend_ready(device)?;
        return Ok(device);
    }

    let device = parse_standard_device(FAMILY, &label)
        .with_context(|| format!("parse device {label} ({STANDARD_DEVICE_NAMES}|auto)"))?;
    ensure_backend_ready(device)?;
    Ok(device)
}

/// Prefer discrete / unified GPU backends, then CPU.
pub fn pick_auto_device() -> Device {
    for device in [
        Device::Cuda,
        Device::Metal,
        Device::Mlx,
        Device::Rocm,
        Device::Gpu, // wgpu
        Device::Vulkan,
    ] {
        if is_available(device) {
            return device;
        }
    }
    Device::Cpu
}

fn ensure_backend_ready(device: Device) -> Result<()> {
    if device == Device::Cpu {
        return Ok(());
    }
    ensure!(
        is_available(device),
        "{FAMILY}: {device:?} is not available.\n\
         Build with the matching feature, e.g. `cargo build -p rlx-locateanything --features {}`, \
         or pass `--device cpu`.",
        feature_hint(device)
    );
    Ok(())
}

fn feature_hint(device: Device) -> &'static str {
    match device {
        Device::Metal => "metal",
        Device::Mlx => "mlx",
        Device::Cuda => "cuda",
        Device::Rocm => "rocm",
        Device::Gpu => "all-backends",
        Device::Vulkan => "vulkan",
        Device::Cpu => "cpu",
        _ => "all-backends",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_resolves_to_available_backend() {
        let d = pick_auto_device();
        assert!(matches!(
            d,
            Device::Cpu
                | Device::Metal
                | Device::Mlx
                | Device::Cuda
                | Device::Rocm
                | Device::Gpu
                | Device::Vulkan
        ));
    }

    #[test]
    fn parse_cpu_device() {
        let d = resolve_device(Some("cpu")).expect("cpu");
        assert_eq!(d, Device::Cpu);
    }
}
