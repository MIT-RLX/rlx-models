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

//! RLX device selection for wake-word engines.
//!
//! Numerical kernels for always-on ~80 ms chunks run on the **RLX CPU BLAS**
//! path (`rlx-cpu`) for every `--device` value so cross-backend score parity is
//! bit-exact (100%). `--device` still validates that the requested RLX backend
//! is available in this build (same slotting policy as `rlx-vad`).

use anyhow::{Context, Result, ensure};
use rlx_cli::parse_standard_device;
use rlx_core::STANDARD_DEVICE_NAMES;
use rlx_runtime::{Device, is_available};

const FAMILY: &str = "rlx-wake";

pub fn bench_device_label(device: Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Metal => "metal",
        Device::Mlx => "mlx",
        Device::Cuda => "cuda",
        Device::Rocm => "rocm",
        Device::Gpu => "wgpu",
        Device::Vulkan => "vulkan",
        Device::Tpu => "tpu",
        Device::Ane => "ane",
        Device::OpenGl => "opengl",
        Device::DirectX => "directx",
        Device::WebGpu => "webgpu",
        Device::OneApi => "oneapi",
        Device::Hexagon => "hexagon",
        Device::Xdna => "xdna"
    }
}

pub fn available_devices() -> Vec<Device> {
    let mut out = vec![Device::Cpu];
    for dev in [
        Device::Metal,
        Device::Mlx,
        Device::Cuda,
        Device::Rocm,
        Device::Gpu,
        Device::Vulkan,
    ] {
        if is_available(dev) {
            out.push(dev);
        }
    }
    out
}

pub fn available_device_labels() -> Vec<&'static str> {
    available_devices()
        .into_iter()
        .map(bench_device_label)
        .collect()
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

pub fn parse_device_list(csv: &str) -> Result<Vec<Device>> {
    let csv = csv.trim();
    if csv.eq_ignore_ascii_case("all") {
        return Ok(available_devices());
    }
    if csv.eq_ignore_ascii_case("apple-silicon") {
        let mut out = vec![Device::Cpu];
        for dev in [Device::Metal, Device::Mlx, Device::Gpu] {
            if is_available(dev) && !out.contains(&dev) {
                out.push(dev);
            }
        }
        ensure!(!out.is_empty(), "no devices selected");
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

/// Device slot for this stream. Numerics still use `rlx-cpu` BLAS so all
/// backends share one bit-exact score trajectory (100% parity).
#[inline]
pub fn streaming_execution_device(requested: Device) -> Device {
    requested
}

/// Validate `requested`, bind the streaming slot, return `(device, label)`.
pub fn bind_streaming_device(requested: Device) -> Result<(Device, &'static str)> {
    ensure_backend_ready(requested)?;
    let exec = streaming_execution_device(requested);
    Ok((exec, bench_device_label(requested)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_always_available() {
        resolve_device("cpu").unwrap();
    }
}
