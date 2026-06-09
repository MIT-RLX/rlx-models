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

// Env-gated: packed GGUF prefill logits must match CPU on CUDA/ROCm/WGPU.
//
//   LLAMA32_GGUF_PATH=/path/to/model.gguf \
//   cargo test -p rlx-models --test llama32_backend_gguf_parity --features "cuda,rocm,gpu" --release

mod compile_support;

use rlx_models::run::Llama32RunnerBuilder;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

const DEFAULT_Q4: &str = "/tmp/rlx-models/Llama-3.2-1B-Instruct-Q4_K_M.gguf";

fn gguf_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("LLAMA32_GGUF_PATH") {
        if !p.is_empty() {
            let path = PathBuf::from(p);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    if std::env::var("LLAMA32_RUN_GGUF_PARITY").ok().as_deref() != Some("1") {
        return None;
    }
    let path = PathBuf::from(DEFAULT_Q4);
    path.is_file().then_some(path)
}

#[cfg(any(feature = "cuda", feature = "rocm", feature = "gpu"))]
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    (dot / na.sqrt() / nb.sqrt()) as f32
}

fn packed_logits(path: &Path, device: Device, prompt: &[u32]) -> Vec<f32> {
    let mut runner = Llama32RunnerBuilder::default()
        .weights(path)
        .device(device)
        .packed_weights(true)
        .max_seq(32)
        .build()
        .expect("runner");
    runner.predict_logits(prompt).expect("predict_logits")
}

#[test]
fn cpu_packed_reference_available_with_weights() {
    let Some(path) = gguf_path() else {
        eprintln!("skip: set LLAMA32_GGUF_PATH or LLAMA32_RUN_GGUF_PARITY=1");
        return;
    };
    let prompt: Vec<u32> = (1..=8).collect();
    let logits = packed_logits(&path, Device::Cpu, &prompt);
    assert!(logits.iter().all(|v| v.is_finite()));
    assert!(!logits.is_empty());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_packed_matches_cpu() {
    let Some(path) = gguf_path() else {
        eprintln!("skip: no GGUF path");
        return;
    };
    if !rlx_runtime::is_available(Device::Cuda) {
        eprintln!("skip: CUDA not available");
        return;
    }
    let prompt: Vec<u32> = (1..=8).collect();
    let cpu = packed_logits(&path, Device::Cpu, &prompt);
    let cuda = packed_logits(&path, Device::Cuda, &prompt);
    let c = cosine(&cpu, &cuda);
    eprintln!("llama32 packed cpu vs cuda cosine={c:.6}");
    assert!(c > 0.995, "packed cpu vs cuda cosine {c}");
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_packed_matches_cpu() {
    let Some(path) = gguf_path() else {
        eprintln!("skip: no GGUF path");
        return;
    };
    if !rlx_runtime::is_available(Device::Rocm) {
        eprintln!("skip: ROCm not available");
        return;
    }
    let prompt: Vec<u32> = (1..=8).collect();
    let cpu = packed_logits(&path, Device::Cpu, &prompt);
    let rocm = packed_logits(&path, Device::Rocm, &prompt);
    let c = cosine(&cpu, &rocm);
    eprintln!("llama32 packed cpu vs rocm cosine={c:.6}");
    assert!(c > 0.995, "packed cpu vs rocm cosine {c}");
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_packed_matches_cpu() {
    let Some(path) = gguf_path() else {
        eprintln!("skip: no GGUF path");
        return;
    };
    if !rlx_runtime::is_available(Device::Gpu) {
        eprintln!("skip: WGPU not available");
        return;
    }
    let prompt: Vec<u32> = (1..=8).collect();
    let cpu = packed_logits(&path, Device::Cpu, &prompt);
    let gpu = packed_logits(&path, Device::Gpu, &prompt);
    let c = cosine(&cpu, &gpu);
    eprintln!("llama32 packed cpu vs wgpu cosine={c:.6}");
    assert!(c > 0.995, "packed cpu vs wgpu cosine {c}");
}
