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

//! Training device selection (`--device`, `RLX_DEVICE`, auto GPU pick).

use anyhow::{Context, Result, ensure};
use rlx_cli::parse_standard_device;
use rlx_core::STANDARD_DEVICE_NAMES;
use rlx_runtime::{Device, is_available};
use std::env;

const FAMILY: &str = "voxtral-tts-train";

/// Resolve execution device from CLI name, `RLX_DEVICE`, or auto GPU pick.
pub fn resolve_train_device(requested: Option<&str>) -> Result<Device> {
    let name = requested
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| env::var("RLX_DEVICE").ok())
        .unwrap_or_else(|| "auto".to_string());

    if name.eq_ignore_ascii_case("auto") {
        let device = pick_auto_device();
        ensure_backend_ready(device)?;
        return Ok(device);
    }

    let device = parse_standard_device(FAMILY, &name)
        .with_context(|| format!("parse --device {name} ({STANDARD_DEVICE_NAMES}|auto)"))?;
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
        Device::Gpu,
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
         Build with the matching feature, e.g. `cargo build -p rlx-voxtral-tts-train --features {}`, \
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
        Device::Gpu => "gpu",
        Device::Vulkan => "vulkan",
        Device::Cpu => "cpu",
        _ => "all-backends",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_falls_back_to_cpu_when_no_gpu() {
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
        let d = resolve_train_device(Some("cpu")).expect("cpu");
        assert_eq!(d, Device::Cpu);
    }
}
