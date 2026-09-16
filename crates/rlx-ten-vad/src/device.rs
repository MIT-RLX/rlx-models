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

//! RLX device selection for TEN-VAD.
//!
//! The network graph compiles for whichever backend is requested. Note that
//! streaming scores one 16 ms frame per dispatch, where launch latency
//! dominates a network this small — `--device cpu` is usually fastest there,
//! and the GPU backends pay off on [`crate::TenVadBatch`], which scores a whole
//! 30 s window in one dispatch.

use anyhow::{Context, Result, ensure};
use rlx_cli::parse_standard_device;
use rlx_core::STANDARD_DEVICE_NAMES;
use rlx_runtime::{Device, is_available};

const FAMILY: &str = "rlx-ten-vad";

/// Stable label for bench output (`gpu` → `wgpu`).
pub fn device_label(device: Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Metal => "metal",
        Device::Mlx => "mlx",
        Device::Cuda => "cuda",
        Device::Rocm => "rocm",
        Device::Xdna => "xdna",
        Device::Gpu => "wgpu",
        Device::Vulkan => "vulkan",
        Device::Tpu => "tpu",
        Device::Ane => "ane",
        Device::OpenGl => "opengl",
        Device::DirectX => "directx",
        Device::WebGpu => "webgpu",
        Device::OneApi => "oneapi",
        Device::Hexagon => "hexagon",
        Device::Egpu => "egpu",
    }
}

/// Backends compiled into this build and present at run time.
pub fn available_devices() -> Vec<Device> {
    let mut out = vec![Device::Cpu];
    for dev in [
        Device::Metal,
        Device::Mlx,
        Device::Cuda,
        Device::Rocm,
        Device::Gpu,
        Device::Vulkan,
        Device::Ane,
    ] {
        if is_available(dev) {
            out.push(dev);
        }
    }
    out
}

pub fn available_device_labels() -> Vec<&'static str> {
    available_devices().into_iter().map(device_label).collect()
}

pub fn ensure_backend_ready(device: Device) -> Result<()> {
    if device == Device::Cpu {
        return Ok(());
    }
    ensure!(
        is_available(device),
        "{FAMILY}: {device:?} is not available — rebuild with the matching feature \
         (e.g. `--features metal`) or pass `--device cpu`"
    );
    Ok(())
}

pub fn resolve_device(name: &str) -> Result<Device> {
    let device = parse_standard_device(FAMILY, name)?;
    ensure_backend_ready(device)?;
    Ok(device)
}

/// Parse `--devices cpu,metal`, `all`, or `apple-silicon`.
pub fn parse_device_list(csv: &str) -> Result<Vec<Device>> {
    let csv = csv.trim();
    if csv.eq_ignore_ascii_case("all") {
        return Ok(available_devices());
    }
    if csv.eq_ignore_ascii_case("apple-silicon") {
        let mut out = vec![Device::Cpu];
        for dev in [Device::Metal, Device::Mlx, Device::Gpu, Device::Ane] {
            if is_available(dev) && !out.contains(&dev) {
                out.push(dev);
            }
        }
        return Ok(out);
    }
    let mut out = Vec::new();
    for part in csv.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let dev = resolve_device(part).with_context(|| {
            format!("parse device {part} ({STANDARD_DEVICE_NAMES}|all|apple-silicon)")
        })?;
        if !out.contains(&dev) {
            out.push(dev);
        }
    }
    ensure!(!out.is_empty(), "no devices selected");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_always_resolves() {
        assert_eq!(resolve_device("cpu").unwrap(), Device::Cpu);
    }

    #[test]
    fn device_list_dedupes() {
        assert_eq!(parse_device_list("cpu,cpu").unwrap(), vec![Device::Cpu]);
    }

    #[test]
    fn all_includes_cpu() {
        assert!(parse_device_list("all").unwrap().contains(&Device::Cpu));
    }
}
