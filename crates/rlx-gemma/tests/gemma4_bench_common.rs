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

//! Shared device selection for Gemma 4 integration benches / sweeps.

use rlx_runtime::{Device, device_ext::is_available};

/// Single device from `RLX_GEMMA4_DEVICE` (`mlx` | `metal` | `cpu`), else Metal on macOS.
pub fn bench_device_from_env() -> Device {
    if let Ok(raw) = std::env::var("RLX_GEMMA4_DEVICE") {
        match raw.to_ascii_lowercase().as_str() {
            "mlx"
                if cfg!(all(target_os = "macos", feature = "mlx")) && is_available(Device::Mlx) =>
            {
                return Device::Mlx;
            }
            "metal"
                if cfg!(all(target_os = "macos", feature = "metal"))
                    && is_available(Device::Metal) =>
            {
                return Device::Metal;
            }
            "cpu" => return Device::Cpu,
            _ => {}
        }
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if is_available(Device::Metal) {
        return Device::Metal;
    }
    Device::Cpu
}

/// Devices to run when `RLX_GEMMA4_DEVICE` is unset: Metal then MLX on macOS, else CPU.
#[allow(dead_code)]
pub fn bench_devices_from_env() -> Vec<Device> {
    if std::env::var("RLX_GEMMA4_DEVICE").is_ok() {
        return vec![bench_device_from_env()];
    }
    let mut out = Vec::new();
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if is_available(Device::Metal) {
        out.push(Device::Metal);
    }
    #[cfg(all(target_os = "macos", feature = "mlx"))]
    if is_available(Device::Mlx) {
        out.push(Device::Mlx);
    }
    if out.is_empty() {
        out.push(Device::Cpu);
    }
    out
}
