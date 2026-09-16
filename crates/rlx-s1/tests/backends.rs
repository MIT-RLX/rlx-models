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

//! Cross-backend agreement for S1-mini.
//!
//! Every enabled backend must reproduce the same normalizations the CPU path
//! does — which are themselves token-identical to HF f32 (see `hf_parity.rs`).
//! The expectations below are pinned literals rather than a CPU re-run, so a
//! backend test is meaningful on its own and a shared-path regression can't
//! move both sides at once.
//!
//! ```text
//! cargo test -p rlx-s1 --features metal,mlx --test backends -- --nocapture
//! ```
//!
//! Each test no-ops when the checkpoint or the device is unavailable.

use rlx_s1::{Controls, Device, S1Runner};
use std::path::PathBuf;

/// `(transcript, expected)` under the default control line. Verified
/// token-identical to `transformers` greedy in float32.
const CASES: &[(&str, &str)] = &[
    (
        "so um i need to like send the the report by uh friday no wait make that thursday",
        "So I need to send the report by Thursday.",
    ),
    (
        "i think the answer is forty two no sorry forty three",
        "I think the answer is 43.",
    ),
    (
        "send it to support at superwhisper dot com",
        "Send it to support@superwhisper.com.",
    ),
    // Filler-only input is documented to produce nothing at all.
    ("um", ""),
];

fn weights() -> Option<PathBuf> {
    let p: PathBuf = std::env::var("S1_WEIGHTS")
        .unwrap_or_else(|_| "/Volumes/FOUR/weights/lm/s1-mini".into())
        .into();
    p.exists().then_some(p)
}

fn run_on(device: Device, tag: &str) {
    let Some(weights) = weights() else {
        eprintln!("[s1 backends/{tag}] no S1_WEIGHTS checkpoint — skip");
        return;
    };
    let mut runner = S1Runner::builder()
        .weights(&weights)
        .device(device)
        .strict_shape(true)
        .build()
        .unwrap_or_else(|e| panic!("[{tag}] build: {e:#}"));

    for (transcript, expected) in CASES {
        let got = runner
            .normalize_with(transcript, Controls::new())
            .unwrap_or_else(|e| panic!("[{tag}] normalize {transcript:?}: {e:#}"));
        assert_eq!(&got, expected, "[{tag}] output differs for {transcript:?}");
    }
    eprintln!("[s1 backends/{tag}] {} cases match CPU/HF", CASES.len());
}

#[test]
fn agreement_cpu() {
    run_on(Device::Cpu, "cpu");
}

#[cfg(feature = "metal")]
#[test]
fn agreement_metal() {
    if !rlx_runtime::device_ext::is_available(Device::Metal) {
        eprintln!("[s1 backends] Metal unavailable — skip");
        return;
    }
    run_on(Device::Metal, "metal");
}

#[cfg(feature = "mlx")]
#[test]
fn agreement_mlx() {
    if !rlx_runtime::device_ext::is_available(Device::Mlx) {
        eprintln!("[s1 backends] MLX unavailable — skip");
        return;
    }
    run_on(Device::Mlx, "mlx");
}

#[cfg(feature = "gpu")]
#[test]
fn agreement_gpu() {
    if !rlx_runtime::device_ext::is_available(Device::Gpu) {
        eprintln!("[s1 backends] wgpu unavailable — skip");
        return;
    }
    run_on(Device::Gpu, "gpu");
}

#[cfg(feature = "vulkan")]
#[test]
fn agreement_vulkan() {
    if !rlx_runtime::device_ext::is_available(Device::Vulkan) {
        eprintln!("[s1 backends] Vulkan unavailable — skip");
        return;
    }
    run_on(Device::Vulkan, "vulkan");
}

#[cfg(feature = "cuda")]
#[test]
fn agreement_cuda() {
    if !rlx_runtime::device_ext::is_available(Device::Cuda) {
        eprintln!("[s1 backends] CUDA unavailable — skip");
        return;
    }
    run_on(Device::Cuda, "cuda");
}

#[cfg(feature = "rocm")]
#[test]
fn agreement_rocm() {
    if !rlx_runtime::device_ext::is_available(Device::Rocm) {
        eprintln!("[s1 backends] ROCm unavailable — skip");
        return;
    }
    run_on(Device::Rocm, "rocm");
}

#[cfg(feature = "coreml")]
#[test]
fn agreement_coreml() {
    if !rlx_runtime::device_ext::is_available(Device::Ane) {
        eprintln!("[s1 backends] CoreML/ANE unavailable — skip");
        return;
    }
    run_on(Device::Ane, "coreml");
}
