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

// Env-gated: real 0.8B GGUF forward on CPU / Metal / MLX.
//
//   QWEN35_GGUF_PATH=/tmp/rlx-models/Qwen3.5-0.8B-Q4_K_M.gguf \
//     cargo test -p rlx-models --test qwen35_backend_gguf_check --features "metal,mlx" --release -- --nocapture
//
// MLX: auto-selects lazy when the graph contains GatedDeltaNet.
// Metal: FuseMatMulBiasAct enabled; Exp/Log epilogues are not folded.

#![allow(dead_code)]

mod compile_support;

use rlx_models::Qwen35RunnerBuilder;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};
use std::time::Instant;

const DEFAULT_NON_MTP_Q4: &str = "/tmp/rlx-models/Qwen3.5-0.8B-Q4_K_M.gguf";
const DEFAULT_MTP_Q4: &str = "/tmp/rlx-models/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf";

fn gguf_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("QWEN35_GGUF_PATH") {
        if !p.is_empty() {
            let path = PathBuf::from(p);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    let path = PathBuf::from(DEFAULT_NON_MTP_Q4);
    path.is_file().then_some(path)
}

fn mtp_gguf_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("QWEN35_MTP_GGUF_PATH") {
        if !p.is_empty() {
            let path = PathBuf::from(p);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    if let Ok(p) = std::env::var("QWEN35_GGUF_PATH") {
        if !p.is_empty() {
            let path = PathBuf::from(p);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    let path = PathBuf::from(DEFAULT_MTP_Q4);
    path.is_file().then_some(path)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

fn top_k_tokens(logits: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.into_iter().take(k).map(|i| i as u32).collect()
}

fn run_predict(path: &Path, device: Device) -> (Vec<f32>, f64) {
    let prompt = vec![1u32, 2, 3];
    let max_seq = prompt.len().max(8);
    let t0 = Instant::now();
    let mut runner = Qwen35RunnerBuilder::default()
        .weights(path)
        .max_seq(max_seq)
        .device(device)
        .packed_weights(true)
        .last_logits_only(true)
        .build()
        .expect("build runner");
    let out = runner.predict_logits(&prompt).expect("predict_logits");
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    (out.logits, ms)
}

fn run_predict_mtp(path: &Path, device: Device) -> (Vec<f32>, Vec<f32>, f64) {
    let prompt = vec![1u32, 2, 3];
    let max_seq = prompt.len().max(8);
    let t0 = Instant::now();
    let mut runner = Qwen35RunnerBuilder::default()
        .weights(path)
        .max_seq(max_seq)
        .device(device)
        .packed_weights(true)
        .enable_mtp(true)
        .last_logits_only(true)
        .build()
        .expect("build MTP runner");
    let out = runner.predict_logits(&prompt).expect("predict_logits");
    let mtp = out.mtp_logits.expect("MTP logits with enable_mtp(true)");
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    (out.logits, mtp, ms)
}

fn assert_backend_matches_cpu(label: &str, cpu_logits: &[f32], backend_logits: &[f32], ms: f64) {
    assert_eq!(
        cpu_logits.len(),
        backend_logits.len(),
        "{label}: logit length mismatch"
    );
    let non_finite = backend_logits.iter().filter(|v| !v.is_finite()).count();
    assert!(
        non_finite == 0,
        "{label}: {non_finite}/{} non-finite logits",
        backend_logits.len()
    );
    let cos = cosine_similarity(cpu_logits, backend_logits);
    let cpu_top3 = top_k_tokens(cpu_logits, 3);
    let backend_top3 = top_k_tokens(backend_logits, 3);
    eprintln!(
        "qwen35 backend check {label}: cos={cos:.6} top3={backend_top3:?} cpu_top3={cpu_top3:?} {ms:.1}ms"
    );
    assert!(cos >= 0.999, "{label}: logits cosine {cos:.6} below 0.999");
    assert_eq!(
        cpu_top3, backend_top3,
        "{label}: top-3 tokens diverged from CPU"
    );
}

#[test]
fn qwen35_real_gguf_runs_on_cpu() {
    let path = match gguf_path() {
        Some(p) => p,
        None => {
            eprintln!("skip qwen35_backend_gguf_check: set QWEN35_GGUF_PATH");
            return;
        }
    };

    let (logits, ms) = run_predict(&path, Device::Cpu);
    assert!(logits.iter().all(|v| v.is_finite()));
    eprintln!(
        "qwen35 backend check cpu: n_vocab={} top3={:?} {ms:.1}ms",
        logits.len(),
        top_k_tokens(&logits, 3)
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn qwen35_real_gguf_runs_on_metal() {
    let path = match gguf_path() {
        Some(p) => p,
        None => return,
    };

    let (cpu_logits, _cpu_ms) = run_predict(&path, Device::Cpu);
    let (metal_logits, metal_ms) = run_predict(&path, Device::Metal);
    assert_backend_matches_cpu("metal", &cpu_logits, &metal_logits, metal_ms);
}

#[test]
#[cfg(all(target_os = "macos", feature = "mlx"))]
fn qwen35_real_gguf_runs_on_mlx() {
    let path = match gguf_path() {
        Some(p) => p,
        None => return,
    };

    // MLX compile auto-selects lazy for GDN graphs (see rlx-runtime backend).
    let (cpu_logits, _cpu_ms) = run_predict(&path, Device::Cpu);
    let (mlx_logits, mlx_ms) = run_predict(&path, Device::Mlx);
    assert_backend_matches_cpu("mlx", &cpu_logits, &mlx_logits, mlx_ms);
}

#[test]
#[cfg(feature = "gpu")]
fn qwen35_real_gguf_runs_on_wgpu() {
    let path = match gguf_path() {
        Some(p) => p,
        None => return,
    };
    if !rlx_runtime::is_available(Device::Gpu) {
        eprintln!("skip qwen35_real_gguf_runs_on_wgpu: wgpu unavailable");
        return;
    }

    let (cpu_logits, _cpu_ms) = run_predict(&path, Device::Cpu);
    let (wgpu_logits, wgpu_ms) = run_predict(&path, Device::Gpu);
    assert_backend_matches_cpu("wgpu", &cpu_logits, &wgpu_logits, wgpu_ms);
}

#[test]
fn qwen35_mtp_gguf_trunk_check_cpu() {
    let path = match mtp_gguf_path() {
        Some(p) => p,
        None => {
            eprintln!(
                "skip qwen35_mtp check: set QWEN35_MTP_GGUF_PATH or materialize {DEFAULT_MTP_Q4}"
            );
            return;
        }
    };
    let (trunk, mtp, ms) = run_predict_mtp(&path, Device::Cpu);
    assert!(trunk.iter().all(|v| v.is_finite()));
    assert!(mtp.iter().all(|v| v.is_finite()));
    eprintln!(
        "qwen35 MTP check cpu: trunk_top3={:?} mtp_top3={:?} {ms:.1}ms",
        top_k_tokens(&trunk, 3),
        top_k_tokens(&mtp, 3)
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn qwen35_mtp_gguf_trunk_check_metal() {
    let path = match mtp_gguf_path() {
        Some(p) => p,
        None => return,
    };
    let (cpu_trunk, cpu_mtp, _) = run_predict_mtp(&path, Device::Cpu);
    let (metal_trunk, metal_mtp, ms) = run_predict_mtp(&path, Device::Metal);
    assert_backend_matches_cpu("mtp-trunk-metal", &cpu_trunk, &metal_trunk, ms);
    let mtp_cos = cosine_similarity(&cpu_mtp, &metal_mtp);
    eprintln!(
        "qwen35 MTP check metal: mtp_cos={mtp_cos:.6} mtp_top3={:?} cpu_top3={:?}",
        top_k_tokens(&metal_mtp, 3),
        top_k_tokens(&cpu_mtp, 3)
    );
    assert!(
        mtp_cos >= 0.999,
        "mtp-trunk-metal: MTP logits cosine {mtp_cos:.6} below 0.999"
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "mlx"))]
fn qwen35_mtp_gguf_trunk_check_mlx() {
    let path = match mtp_gguf_path() {
        Some(p) => p,
        None => return,
    };
    let (cpu_trunk, cpu_mtp, _) = run_predict_mtp(&path, Device::Cpu);
    let (mlx_trunk, mlx_mtp, ms) = run_predict_mtp(&path, Device::Mlx);
    assert_backend_matches_cpu("mtp-trunk-mlx", &cpu_trunk, &mlx_trunk, ms);
    let mtp_cos = cosine_similarity(&cpu_mtp, &mlx_mtp);
    eprintln!(
        "qwen35 MTP check mlx: mtp_cos={mtp_cos:.6} mtp_top3={:?} cpu_top3={:?} {ms:.1}ms",
        top_k_tokens(&mlx_mtp, 3),
        top_k_tokens(&cpu_mtp, 3)
    );
    assert!(
        mtp_cos >= 0.999,
        "mtp-trunk-mlx: MTP logits cosine {mtp_cos:.6} below 0.999"
    );
}
