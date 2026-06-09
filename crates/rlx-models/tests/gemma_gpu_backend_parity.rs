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

// CPU vs GPU F32 parity on a tiny synthetic Gemma prefill graph.
//
//   cargo test -p rlx-models --test gemma_gpu_backend_parity --features "metal,mlx,cuda,rocm,gpu,vulkan"

mod gemma_common;

use gemma_common::run_last_logits;
use rlx_runtime::Device;

#[test]
fn cpu_reference_logits_finite() {
    let logits = run_last_logits(Device::Cpu);
    assert_eq!(logits.len(), gemma_common::tiny_cfg().vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_matches_cpu_logits() {
    gemma_common::assert_logits_match_cpu(Device::Metal, "metal");
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn mlx_matches_cpu_logits() {
    gemma_common::assert_logits_match_cpu(Device::Mlx, "mlx");
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_matches_cpu_logits() {
    gemma_common::assert_logits_match_cpu(Device::Cuda, "cuda");
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_matches_cpu_logits() {
    gemma_common::assert_logits_match_cpu(Device::Rocm, "rocm");
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_matches_cpu_logits() {
    gemma_common::assert_logits_match_cpu(Device::Gpu, "wgpu");
}

#[cfg(feature = "vulkan")]
#[test]
fn vulkan_matches_cpu_logits() {
    gemma_common::assert_logits_match_cpu(Device::Vulkan, "vulkan");
}

#[test]
fn cpu_reference_gemma2_logits_finite() {
    let logits = gemma_common::run_last_logits_gemma2(Device::Cpu);
    assert_eq!(logits.len(), gemma_common::tiny_gemma2_cfg().vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_matches_cpu_gemma2_logits() {
    gemma_common::assert_logits_match_cpu_gemma2(Device::Metal, "metal");
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn mlx_matches_cpu_gemma2_logits() {
    gemma_common::assert_logits_match_cpu_gemma2(Device::Mlx, "mlx");
}
