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

//! Synthetic projector + Qwen2.5 LM on each standard backend.
//!
//! ```bash
//! cargo test -p rlx-locateanything --test backend_quick_check --features all-backends
//! just features=all-backends test-locateanything-backends
//! ```

mod backend_common;

use rlx_runtime::Device;

#[test]
fn locateanything_projector_runs_on_cpu() {
    backend_common::run_projector_on_device(Device::Cpu);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn locateanything_projector_runs_on_metal() {
    backend_common::run_projector_if_available(Device::Metal);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn locateanything_projector_runs_on_mlx() {
    backend_common::run_projector_if_available(Device::Mlx);
}

#[cfg(feature = "cuda")]
#[test]
fn locateanything_projector_runs_on_cuda() {
    backend_common::run_projector_if_available(Device::Cuda);
}

#[cfg(feature = "rocm")]
#[test]
fn locateanything_projector_runs_on_rocm() {
    backend_common::run_projector_if_available(Device::Rocm);
}

#[cfg(feature = "gpu")]
#[test]
fn locateanything_projector_runs_on_wgpu() {
    backend_common::run_projector_if_available(Device::Gpu);
}

#[cfg(feature = "vulkan")]
#[test]
fn locateanything_projector_runs_on_vulkan() {
    backend_common::run_projector_if_available(Device::Vulkan);
}

#[test]
fn locateanything_prefill_runs_on_cpu() {
    backend_common::run_prefill_last_logits_on_device(Device::Cpu);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn locateanything_prefill_runs_on_metal() {
    backend_common::run_prefill_last_logits_if_available(Device::Metal);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn locateanything_prefill_runs_on_mlx() {
    backend_common::run_prefill_last_logits_if_available(Device::Mlx);
}

#[cfg(feature = "cuda")]
#[test]
fn locateanything_prefill_runs_on_cuda() {
    backend_common::run_prefill_last_logits_if_available(Device::Cuda);
}

#[cfg(feature = "rocm")]
#[test]
fn locateanything_prefill_runs_on_rocm() {
    backend_common::run_prefill_last_logits_if_available(Device::Rocm);
}

#[cfg(feature = "gpu")]
#[test]
fn locateanything_prefill_runs_on_wgpu() {
    backend_common::run_prefill_last_logits_if_available(Device::Gpu);
}

#[cfg(feature = "vulkan")]
#[test]
fn locateanything_prefill_runs_on_vulkan() {
    backend_common::run_prefill_last_logits_if_available(Device::Vulkan);
}

#[test]
fn locateanything_decode_runs_on_cpu() {
    backend_common::run_decode_step(Device::Cpu);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn locateanything_decode_runs_on_metal() {
    backend_common::run_decode_step_if_available(Device::Metal);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn locateanything_decode_runs_on_mlx() {
    backend_common::run_decode_step_if_available(Device::Mlx);
}

#[cfg(feature = "cuda")]
#[test]
fn locateanything_decode_runs_on_cuda() {
    backend_common::run_decode_step_if_available(Device::Cuda);
}

#[cfg(feature = "rocm")]
#[test]
fn locateanything_decode_runs_on_rocm() {
    backend_common::run_decode_step_if_available(Device::Rocm);
}

#[cfg(feature = "gpu")]
#[test]
fn locateanything_decode_runs_on_wgpu() {
    backend_common::run_decode_step_if_available(Device::Gpu);
}

#[cfg(feature = "vulkan")]
#[test]
fn locateanything_decode_runs_on_vulkan() {
    backend_common::run_decode_step_if_available(Device::Vulkan);
}
