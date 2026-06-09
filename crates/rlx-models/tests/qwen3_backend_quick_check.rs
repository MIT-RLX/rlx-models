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

// Synthetic qwen3 prefill + generator on each standard backend (skips when unavailable).
//
//   cargo test -p rlx-models --test qwen3_backend_quick_check --features all-backends
//   just features=all-backends test-qwen3-backends

mod qwen3_common;

use rlx_runtime::Device;

#[test]
fn qwen3_tiny_graph_runs_on_cpu() {
    qwen3_common::run_last_logits_prefill(Device::Cpu);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn qwen3_tiny_graph_runs_on_metal() {
    qwen3_common::run_last_logits_prefill_if_available(Device::Metal);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn qwen3_tiny_graph_runs_on_mlx() {
    qwen3_common::run_last_logits_prefill_if_available(Device::Mlx);
}

#[cfg(feature = "cuda")]
#[test]
fn qwen3_tiny_graph_runs_on_cuda() {
    qwen3_common::run_last_logits_prefill_if_available(Device::Cuda);
}

#[cfg(feature = "rocm")]
#[test]
fn qwen3_tiny_graph_runs_on_rocm() {
    qwen3_common::run_last_logits_prefill_if_available(Device::Rocm);
}

#[cfg(feature = "gpu")]
#[test]
fn qwen3_tiny_graph_runs_on_wgpu() {
    qwen3_common::run_last_logits_prefill_if_available(Device::Gpu);
}

#[cfg(feature = "vulkan")]
#[test]
fn qwen3_tiny_graph_runs_on_vulkan() {
    qwen3_common::run_last_logits_prefill_if_available(Device::Vulkan);
}

#[test]
fn qwen3_generator_runs_on_cpu() {
    qwen3_common::run_generator_greedy(Device::Cpu);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn qwen3_generator_runs_on_metal() {
    qwen3_common::run_generator_greedy_if_available(Device::Metal);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn qwen3_generator_runs_on_mlx() {
    qwen3_common::run_generator_greedy_if_available(Device::Mlx);
}

#[cfg(feature = "cuda")]
#[test]
fn qwen3_generator_runs_on_cuda() {
    qwen3_common::run_generator_greedy_if_available(Device::Cuda);
}

#[cfg(feature = "rocm")]
#[test]
fn qwen3_generator_runs_on_rocm() {
    qwen3_common::run_generator_greedy_if_available(Device::Rocm);
}

#[cfg(feature = "gpu")]
#[test]
fn qwen3_generator_runs_on_wgpu() {
    qwen3_common::run_generator_greedy_if_available(Device::Gpu);
}

#[cfg(feature = "vulkan")]
#[test]
fn qwen3_generator_runs_on_vulkan() {
    qwen3_common::run_generator_greedy_if_available(Device::Vulkan);
}
