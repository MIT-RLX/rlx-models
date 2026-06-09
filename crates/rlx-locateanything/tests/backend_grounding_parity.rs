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

//! Grounding token parity: CPU reference vs each available backend (real weights, env-gated).
//!
//! ```bash
//! RLX_LOCATEANYTHING_DIR=/path/to/LocateAnything-3B \
//!   cargo test -p rlx-locateanything --test backend_grounding_parity --features apple-silicon,tokenizer --release
//! ```

use rlx_locateanything::fixtures::sample_image_path;
use rlx_locateanything::{InferenceOptions, LocateAnythingSession};
use rlx_runtime::Device;

fn model_dir() -> Option<std::path::PathBuf> {
    std::env::var("RLX_LOCATEANYTHING_DIR")
        .ok()
        .map(std::path::PathBuf::from)
}

fn ground_tokens(device: Device, dir: &std::path::Path, max_new: usize) -> Vec<u32> {
    let opts = InferenceOptions::for_grounding()
        .device(device)
        .max_new_tokens(max_new);
    let mut session = LocateAnythingSession::open_with_options(dir, opts).expect("session");
    let prep = session
        .preprocess_file(sample_image_path())
        .expect("preprocess");
    let prompt = session
        .runner()
        .build_prompt_processor("person", &prep)
        .expect("prompt");
    let tokens = session
        .runner_mut()
        .generate(&prompt, &prep)
        .expect("generate");
    tokens[prompt.len()..].to_vec()
}

#[allow(dead_code)]
fn assert_parity_with_cpu(dir: &std::path::Path, device: Device) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip grounding parity {device:?}: backend not available");
        return;
    }
    let max_new = 8;
    let cpu = ground_tokens(Device::Cpu, dir, max_new);
    let other = ground_tokens(device, dir, max_new);
    assert_eq!(
        cpu, other,
        "grounding tokens mismatch: cpu={cpu:?} {device:?}={other:?}"
    );
}

#[test]
fn grounding_parity_cpu_reference() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    let cpu = ground_tokens(Device::Cpu, &dir, 8);
    assert!(
        !cpu.is_empty(),
        "expected at least one generated token, got {cpu:?}"
    );
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn grounding_parity_metal() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    assert_parity_with_cpu(&dir, Device::Metal);
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn grounding_parity_mlx() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    assert_parity_with_cpu(&dir, Device::Mlx);
}

#[cfg(feature = "gpu")]
#[test]
fn grounding_parity_wgpu() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    assert_parity_with_cpu(&dir, Device::Gpu);
}

#[cfg(feature = "cuda")]
#[test]
fn grounding_parity_cuda() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    assert_parity_with_cpu(&dir, Device::Cuda);
}

#[cfg(feature = "rocm")]
#[test]
fn grounding_parity_rocm() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    assert_parity_with_cpu(&dir, Device::Rocm);
}

#[cfg(feature = "vulkan")]
#[test]
fn grounding_parity_vulkan() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    assert_parity_with_cpu(&dir, Device::Vulkan);
}
