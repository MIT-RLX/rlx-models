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

//! Compile and run OCR detection + recognition on each standard RLX backend.
//!
//! ```bash
//! OCR_MODEL_DIR=~/.cache/ocrs cargo test -p rlx-ocr --test ocr_backend_quick_check --features rlx --release
//! just features=all-backends test-ocr-backends
//! ```

#![cfg(feature = "rlx")]

mod ocr_backend_common;

use rlx_runtime::Device;

#[test]
fn ocr_detection_runs_on_cpu() {
    ocr_backend_common::run_detection_forward_if_available(Device::Cpu);
}

#[test]
fn ocr_recognition_runs_on_cpu() {
    ocr_backend_common::run_recognition_logits_if_available(Device::Cpu);
}

#[test]
fn ocr_pipeline_runs_on_cpu() {
    ocr_backend_common::run_engine_get_text_if_available(Device::Cpu);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn ocr_detection_runs_on_metal() {
    ocr_backend_common::run_detection_forward_if_available(Device::Metal);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn ocr_recognition_runs_on_metal() {
    ocr_backend_common::run_recognition_logits_if_available(Device::Metal);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn ocr_pipeline_runs_on_metal() {
    ocr_backend_common::run_engine_get_text_if_available(Device::Metal);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn ocr_detection_runs_on_mlx() {
    ocr_backend_common::run_detection_forward_if_available(Device::Mlx);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn ocr_recognition_runs_on_mlx() {
    ocr_backend_common::run_recognition_logits_if_available(Device::Mlx);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn ocr_pipeline_runs_on_mlx() {
    ocr_backend_common::run_engine_get_text_if_available(Device::Mlx);
}

#[cfg(feature = "cuda")]
#[test]
fn ocr_detection_runs_on_cuda() {
    ocr_backend_common::run_detection_forward_if_available(Device::Cuda);
}

#[cfg(feature = "cuda")]
#[test]
fn ocr_recognition_runs_on_cuda() {
    ocr_backend_common::run_recognition_logits_if_available(Device::Cuda);
}

#[cfg(feature = "cuda")]
#[test]
fn ocr_pipeline_runs_on_cuda() {
    ocr_backend_common::run_engine_get_text_if_available(Device::Cuda);
}

#[cfg(feature = "rocm")]
#[test]
fn ocr_detection_runs_on_rocm() {
    ocr_backend_common::run_detection_forward_if_available(Device::Rocm);
}

#[cfg(feature = "rocm")]
#[test]
fn ocr_recognition_runs_on_rocm() {
    ocr_backend_common::run_recognition_logits_if_available(Device::Rocm);
}

#[cfg(feature = "rocm")]
#[test]
fn ocr_pipeline_runs_on_rocm() {
    ocr_backend_common::run_engine_get_text_if_available(Device::Rocm);
}

#[cfg(feature = "gpu")]
#[test]
fn ocr_detection_runs_on_wgpu() {
    ocr_backend_common::run_detection_forward_if_available(Device::Gpu);
}

#[cfg(feature = "gpu")]
#[test]
fn ocr_recognition_runs_on_wgpu() {
    ocr_backend_common::run_recognition_logits_if_available(Device::Gpu);
}

#[cfg(feature = "gpu")]
#[test]
fn ocr_pipeline_runs_on_wgpu() {
    ocr_backend_common::run_engine_get_text_if_available(Device::Gpu);
}

#[cfg(feature = "vulkan")]
#[test]
fn ocr_detection_runs_on_vulkan() {
    ocr_backend_common::run_detection_forward_if_available(Device::Vulkan);
}

#[cfg(feature = "vulkan")]
#[test]
fn ocr_recognition_runs_on_vulkan() {
    ocr_backend_common::run_recognition_logits_if_available(Device::Vulkan);
}

#[cfg(feature = "vulkan")]
#[test]
fn ocr_pipeline_runs_on_vulkan() {
    ocr_backend_common::run_engine_get_text_if_available(Device::Vulkan);
}
