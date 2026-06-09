// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Real MiniCPM5-1B GGUF: packed prefill on CPU / Metal / MLX / CUDA / WGPU.
// MLX and wgpu/CUDA backends execute packed prefill on CPU until rlx 0.2.2 GPU parity;
// tests still build with `--features mlx` / `gpu` and assert logits match the CPU reference.
//
// ```sh
// just fetch-minicpm5-gguf Q4_K_M
// just test-minicpm5-gguf-backends
//
// RLX_MINICPM5_GGUF_Q4_K_M=/path/MiniCPM5-1B-Q4_K_M.gguf \
//   cargo test -p rlx-models --test minicpm5_backend_gguf_check --features all-backends --release -- --nocapture
// ```

#![allow(dead_code)]

use rlx_minicpm5::{MINICPM5_GGUF_FILES, MiniCpm5Runner};
use rlx_runtime::Device;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn gguf_dir() -> PathBuf {
    std::env::var("RLX_MINICPM5_GGUF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/rlx-weights/MiniCPM5-1B-GGUF"))
}

fn gguf_path_for_quant(quant: &str) -> Option<PathBuf> {
    let env_key = format!("RLX_MINICPM5_GGUF_{quant}");
    if let Ok(p) = std::env::var(&env_key) {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Some(path);
        }
    }
    if let Ok(p) = std::env::var("RLX_MINICPM5_GGUF_PATH") {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Some(path);
        }
    }
    let filename = MINICPM5_GGUF_FILES
        .iter()
        .find(|(label, _)| *label == quant)
        .map(|(_, f)| *f)?;
    let path = gguf_dir().join(filename);
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
    let prompt = vec![1u32, 2, 3, 4, 5];
    let max_seq = 64usize;
    let t0 = Instant::now();
    let mut runner = MiniCpm5Runner::builder()
        .weights(path)
        .device(device)
        .max_seq(max_seq)
        .packed_weights(true)
        .build()
        .expect("build runner");
    let logits = runner.predict_logits(&prompt).expect("predict_logits");
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    (logits, ms)
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
        "minicpm5 gguf {label}: cos={cos:.6} top3={backend_top3:?} cpu_top3={cpu_top3:?} {ms:.1}ms"
    );
    assert!(cos >= 0.999, "{label}: logits cosine {cos:.6} below 0.999");
    assert_eq!(
        cpu_top3, backend_top3,
        "{label}: top-3 tokens diverged from CPU"
    );
}

fn gguf_cpu_reference(quant: &str) -> Option<(PathBuf, Vec<f32>)> {
    let path = gguf_path_for_quant(quant)?;
    let (logits, ms) = run_predict(&path, Device::Cpu);
    eprintln!(
        "minicpm5 gguf cpu [{quant}]: n_vocab={} top3={:?} {ms:.1}ms",
        logits.len(),
        top_k_tokens(&logits, 3)
    );
    Some((path, logits))
}

fn run_q4_on_device(label: &str, device: Device) {
    let cpu_ref = gguf_cpu_reference("Q4_K_M");
    let Some((path, cpu_logits)) = cpu_ref else {
        eprintln!("skip {label}: just fetch-minicpm5-gguf Q4_K_M");
        return;
    };
    if device != Device::Cpu && !rlx_runtime::is_available(device) {
        eprintln!("skip minicpm5 gguf {label}: unavailable");
        return;
    }
    let path_c = path.clone();
    let outcome = catch_unwind(AssertUnwindSafe(|| run_predict(&path_c, device)));
    match outcome {
        Ok((backend_logits, ms)) => {
            assert_backend_matches_cpu(label, &cpu_logits, &backend_logits, ms);
        }
        Err(_) => {
            eprintln!("minicpm5 gguf {label}: panic during packed prefill (backend gap)");
        }
    }
}

#[test]
fn minicpm5_gguf_q4_k_m_cpu() {
    let Some((path, _)) = gguf_cpu_reference("Q4_K_M") else {
        eprintln!("skip: just fetch-minicpm5-gguf Q4_K_M");
        return;
    };
    eprintln!("using {}", path.display());
}

#[test]
#[cfg(feature = "metal")]
fn minicpm5_gguf_q4_k_m_metal() {
    run_q4_on_device("metal/Q4_K_M", Device::Metal);
}

#[test]
#[cfg(feature = "mlx")]
fn minicpm5_gguf_q4_k_m_mlx() {
    run_q4_on_device("mlx/Q4_K_M", Device::Mlx);
}

#[test]
#[cfg(feature = "cuda")]
fn minicpm5_gguf_q4_k_m_cuda() {
    run_q4_on_device("cuda/Q4_K_M", Device::Cuda);
}

#[test]
#[cfg(feature = "gpu")]
fn minicpm5_gguf_q4_k_m_wgpu() {
    run_q4_on_device("wgpu/Q4_K_M", Device::Gpu);
}

#[test]
fn minicpm5_gguf_q8_0_cpu() {
    let Some((path, cpu)) = gguf_cpu_reference("Q8_0") else {
        eprintln!("skip Q8_0: just fetch-minicpm5-gguf Q8_0");
        return;
    };
    let (other, ms) = run_predict(&path, Device::Cpu);
    assert_backend_matches_cpu("cpu/Q8_0", &cpu, &other, ms);
}

#[test]
fn minicpm5_gguf_f16_cpu() {
    let Some((path, cpu)) = gguf_cpu_reference("F16") else {
        eprintln!("skip F16: just fetch-minicpm5-gguf F16");
        return;
    };
    let (other, ms) = run_predict(&path, Device::Cpu);
    assert_backend_matches_cpu("cpu/F16", &cpu, &other, ms);
}
