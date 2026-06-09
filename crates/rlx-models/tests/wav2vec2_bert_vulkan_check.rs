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

//! basic test: tiny synthetic Wav2Vec2-BERT graph forced through Vulkan via wgpu.
//!
//! Runs in its own test binary so `WGPU_BACKEND=vulkan` is read before the
//! wgpu singleton initializes. On macOS this typically uses MoltenVK when
//! available; skips gracefully when no Vulkan adapter is present.
//!
//! Run explicitly:
//! ```text
//! WGPU_BACKEND=vulkan cargo test -p rlx-models --features gpu --test wav2vec2_bert_vulkan_check
//! ```

#[path = "wav2vec2_bert/support.rs"]
mod support;

#[cfg(feature = "gpu")]
#[test]
fn wav2vec2_bert_tiny_graph_runs_on_vulkan() {
    if std::env::var("WGPU_BACKEND")
        .map(|v| !v.to_ascii_lowercase().contains("vulkan"))
        .unwrap_or(true)
    {
        // SAFETY: single-threaded test startup; no concurrent env reads.
        unsafe {
            std::env::set_var("WGPU_BACKEND", "vulkan");
        }
    }

    if !rlx_wgpu::is_available() {
        eprintln!("skip: no wgpu adapter for WGPU_BACKEND=vulkan");
        return;
    }

    let Some(dev) = rlx_wgpu::device::wgpu_device() else {
        eprintln!("skip: wgpu adapter init failed for WGPU_BACKEND=vulkan");
        return;
    };
    if format!("{:?}", dev.backend) != "Vulkan" {
        eprintln!(
            "skip: WGPU_BACKEND=vulkan but adapter is {:?} ({}) — \
             install/enable Vulkan (e.g. MoltenVK on macOS)",
            dev.backend, dev.name
        );
        return;
    }

    support::run_tiny_graph(rlx_runtime::Device::Gpu);
}

#[cfg(not(feature = "gpu"))]
#[test]
fn wav2vec2_bert_tiny_graph_runs_on_vulkan() {
    eprintln!("skip: build with --features gpu for wav2vec2_bert_vulkan_check");
}
