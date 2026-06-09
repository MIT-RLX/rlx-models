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

//! Backend EP mapping tests for rlx-kittentts.

#![cfg(feature = "onnx")]

use rlx_kittentts::{Device, execution_providers_for, validate_device};

#[test]
fn cpu_ep_only() {
    let eps = execution_providers_for(Device::Cpu);
    assert_eq!(eps.len(), 1);
}

#[test]
fn gpu_ep_includes_cpu_fallback() {
    let eps = execution_providers_for(Device::Cuda);
    assert!(!eps.is_empty());
}

#[test]
fn validate_standard_devices() {
    for dev in [
        Device::Cpu,
        Device::Metal,
        Device::Mlx,
        Device::Cuda,
        Device::Rocm,
        Device::Gpu,
        Device::Vulkan,
    ] {
        assert!(validate_device(dev).is_ok(), "{dev:?}");
    }
}

#[test]
fn validate_rejects_tpu() {
    assert!(validate_device(Device::Tpu).is_err());
}
