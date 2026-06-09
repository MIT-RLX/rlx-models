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

// Synthetic Gemma prefill + generator on each standard backend (skips when unavailable).
//
//   cargo test -p rlx-models --test gemma_backend_quick_check --features all-backends
//   just features=all-backends test-gemma-backends

mod gemma_common;

use rlx_runtime::Device;

#[test]
fn gemma_tiny_graph_runs_on_cpu() {
    gemma_common::run_last_logits_prefill(Device::Cpu);
}

#[test]
fn gemma2_tiny_graph_runs_on_cpu() {
    gemma_common::run_last_logits_prefill_gemma2(Device::Cpu);
}

#[test]
fn gemma2_decode_runs_on_cpu() {
    gemma_common::run_decode_step_gemma2(Device::Cpu);
}

#[test]
fn gemma2_generator_runs_on_cpu() {
    gemma_common::run_generator_greedy_gemma2(Device::Cpu);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn gemma_tiny_graph_runs_on_metal() {
    gemma_common::run_last_logits_prefill_if_available(Device::Metal);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn gemma2_tiny_graph_runs_on_metal() {
    gemma_common::run_last_logits_prefill_gemma2_if_available(Device::Metal);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn gemma2_decode_runs_on_metal() {
    gemma_common::run_decode_step_gemma2_if_available(Device::Metal);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn gemma_tiny_graph_runs_on_mlx() {
    gemma_common::run_last_logits_prefill_if_available(Device::Mlx);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn gemma2_tiny_graph_runs_on_mlx() {
    gemma_common::run_last_logits_prefill_gemma2_if_available(Device::Mlx);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn gemma2_decode_runs_on_mlx() {
    gemma_common::run_decode_step_gemma2_if_available(Device::Mlx);
}

#[cfg(feature = "cuda")]
#[test]
fn gemma_tiny_graph_runs_on_cuda() {
    gemma_common::run_last_logits_prefill_if_available(Device::Cuda);
}

#[cfg(feature = "rocm")]
#[test]
fn gemma_tiny_graph_runs_on_rocm() {
    gemma_common::run_last_logits_prefill_if_available(Device::Rocm);
}

#[cfg(feature = "gpu")]
#[test]
fn gemma_tiny_graph_runs_on_wgpu() {
    gemma_common::run_last_logits_prefill_if_available(Device::Gpu);
}

#[cfg(feature = "vulkan")]
#[test]
fn gemma_tiny_graph_runs_on_vulkan() {
    gemma_common::run_last_logits_prefill_if_available(Device::Vulkan);
}

#[test]
fn gemma_generator_runs_on_cpu() {
    gemma_common::run_generator_greedy(Device::Cpu);
}

#[test]
fn gemma_decode_runs_on_cpu() {
    gemma_common::run_decode_step(Device::Cpu);
}

#[test]
fn gemma_cached_matches_naive_on_cpu() {
    gemma_common::run_cached_matches_naive(Device::Cpu);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn gemma_generator_runs_on_metal() {
    gemma_common::run_generator_greedy_if_available(Device::Metal);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn gemma_cached_matches_naive_on_metal() {
    gemma_common::run_cached_matches_naive_if_available(Device::Metal);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn gemma_generator_runs_on_mlx() {
    gemma_common::run_generator_greedy_if_available(Device::Mlx);
}

#[cfg(feature = "cuda")]
#[test]
fn gemma_generator_runs_on_cuda() {
    gemma_common::run_generator_greedy_if_available(Device::Cuda);
}

#[cfg(feature = "rocm")]
#[test]
fn gemma_generator_runs_on_rocm() {
    gemma_common::run_generator_greedy_if_available(Device::Rocm);
}

#[cfg(feature = "gpu")]
#[test]
fn gemma_generator_runs_on_wgpu() {
    gemma_common::run_generator_greedy_if_available(Device::Gpu);
}

#[cfg(feature = "vulkan")]
#[test]
fn gemma_generator_runs_on_vulkan() {
    gemma_common::run_generator_greedy_if_available(Device::Vulkan);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn gemma_prefill_kv_runs_on_metal() {
    gemma_common::run_prefill_with_kv_if_available(Device::Metal);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn gemma_decode_runs_on_metal() {
    gemma_common::run_decode_step_if_available(Device::Metal);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn gemma_decode_runs_on_mlx() {
    gemma_common::run_decode_step_if_available(Device::Mlx);
}

#[cfg(feature = "cuda")]
#[test]
fn gemma_decode_runs_on_cuda() {
    gemma_common::run_decode_step_if_available(Device::Cuda);
}

#[cfg(feature = "rocm")]
#[test]
fn gemma_decode_runs_on_rocm() {
    gemma_common::run_decode_step_if_available(Device::Rocm);
}

#[cfg(feature = "gpu")]
#[test]
fn gemma_decode_runs_on_wgpu() {
    gemma_common::run_decode_step_if_available(Device::Gpu);
}

#[cfg(feature = "vulkan")]
#[test]
fn gemma_decode_runs_on_vulkan() {
    gemma_common::run_decode_step_if_available(Device::Vulkan);
}
