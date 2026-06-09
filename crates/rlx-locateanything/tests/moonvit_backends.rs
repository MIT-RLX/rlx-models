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

//! MoonViT compiled encoder on each RLX backend.
//!
//! ```bash
//! cargo test -p rlx-locateanything --test moonvit_backends --features all-backends
//! just features=all-backends test-locateanything-moonvit-backends
//! ```

mod backend_common;

use rlx_runtime::Device;

#[test]
fn moonvit_runs_on_cpu() {
    backend_common::run_moonvit_on_device(Device::Cpu);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn moonvit_runs_on_metal() {
    backend_common::run_moonvit_if_available(Device::Metal);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn moonvit_runs_on_mlx() {
    backend_common::run_moonvit_if_available(Device::Mlx);
}

#[cfg(feature = "cuda")]
#[test]
fn moonvit_runs_on_cuda() {
    backend_common::run_moonvit_if_available(Device::Cuda);
}

#[cfg(feature = "rocm")]
#[test]
fn moonvit_runs_on_rocm() {
    backend_common::run_moonvit_if_available(Device::Rocm);
}

#[cfg(feature = "gpu")]
#[test]
fn moonvit_runs_on_wgpu() {
    backend_common::run_moonvit_if_available(Device::Gpu);
}

#[cfg(feature = "vulkan")]
#[test]
fn moonvit_runs_on_vulkan() {
    backend_common::run_moonvit_if_available(Device::Vulkan);
}
