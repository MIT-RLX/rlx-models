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

// Synthetic qwen35 prefill + runner on each standard backend (skips when unavailable).
//
//   cargo test -p rlx-models --test qwen35_backend_quick_check --features all-backends
//   just features=all-backends test-qwen35-backends

mod qwen35_common;

use rlx_runtime::Device;

#[test]
fn qwen35_tiny_graph_runs_on_cpu() {
    qwen35_common::run_prefill_last_logits(Device::Cpu);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn qwen35_tiny_graph_runs_on_metal() {
    qwen35_common::run_prefill_last_logits_if_available(Device::Metal);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn qwen35_tiny_graph_runs_on_mlx() {
    qwen35_common::run_prefill_last_logits_if_available(Device::Mlx);
}

#[cfg(feature = "cuda")]
#[test]
fn qwen35_tiny_graph_runs_on_cuda() {
    qwen35_common::run_prefill_last_logits_if_available(Device::Cuda);
}

#[cfg(feature = "rocm")]
#[test]
fn qwen35_tiny_graph_runs_on_rocm() {
    qwen35_common::run_prefill_last_logits_if_available(Device::Rocm);
}

#[cfg(feature = "gpu")]
#[test]
fn qwen35_tiny_graph_runs_on_wgpu() {
    qwen35_common::run_prefill_last_logits_if_available(Device::Gpu);
}

#[cfg(feature = "vulkan")]
#[test]
fn qwen35_tiny_graph_runs_on_vulkan() {
    qwen35_common::run_prefill_last_logits_if_available(Device::Vulkan);
}

#[test]
fn qwen35_runner_runs_on_cpu() {
    qwen35_common::run_runner_greedy(Device::Cpu);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn qwen35_runner_runs_on_metal() {
    qwen35_common::run_runner_greedy_if_available(Device::Metal);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn qwen35_runner_runs_on_mlx() {
    qwen35_common::run_runner_greedy_if_available(Device::Mlx);
}

#[cfg(feature = "cuda")]
#[test]
fn qwen35_runner_runs_on_cuda() {
    qwen35_common::run_runner_greedy_if_available(Device::Cuda);
}

#[cfg(feature = "rocm")]
#[test]
fn qwen35_runner_runs_on_rocm() {
    qwen35_common::run_runner_greedy_if_available(Device::Rocm);
}

#[cfg(feature = "gpu")]
#[test]
fn qwen35_runner_runs_on_wgpu() {
    qwen35_common::run_runner_greedy_if_available(Device::Gpu);
}

#[cfg(feature = "vulkan")]
#[test]
fn qwen35_runner_runs_on_vulkan() {
    qwen35_common::run_runner_greedy_if_available(Device::Vulkan);
}
