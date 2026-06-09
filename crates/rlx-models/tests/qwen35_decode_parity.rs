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

// Env-gated: cached decode on Metal/MLX must match CPU token stream.
//
//   QWEN35_GGUF_PATH=/tmp/rlx-models/Qwen3.5-0.8B-Q4_K_M.gguf \
//     cargo test -p rlx-models --test qwen35_decode_parity --features "metal,mlx" --release -- --nocapture

#![allow(dead_code)]

mod compile_support;

use rlx_models::Qwen35RunnerBuilder;
use rlx_models::qwen3::SampleOpts;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

const DEFAULT: &str = "/tmp/rlx-models/Qwen3.5-0.8B-Q4_K_M.gguf";
const DEFAULT_MTP: &str = "/tmp/rlx-models/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf";
const STEPS: usize = 8;

fn gguf_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("QWEN35_GGUF_PATH") {
        if !p.is_empty() {
            let path = PathBuf::from(p);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    PathBuf::from(DEFAULT)
        .is_file()
        .then_some(PathBuf::from(DEFAULT))
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
    PathBuf::from(DEFAULT_MTP)
        .is_file()
        .then_some(PathBuf::from(DEFAULT_MTP))
}

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

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn prefill_and_two_decode_steps(path: &Path, device: Device) -> (u32, u32, u32, f32) {
    let prompt: Vec<u32> = (1..=8).collect();
    let mut runner = Qwen35RunnerBuilder::default()
        .weights(path)
        .device(device)
        .packed_weights(true)
        .max_seq(32)
        .bucketed_decode(false)
        .last_logits_only(true)
        .build()
        .expect("runner");
    let seed = runner
        .prefill_seed_for_decode(&prompt)
        .expect("prefill seed");
    let t1 = argmax(&seed.trunk_logits);
    let l2 = runner.decode_get_logits(t1).expect("decode 1");
    let t2 = argmax(&l2);
    let l3 = runner.decode_get_logits(t2).expect("decode 2");
    let t3 = argmax(&l3);
    (t1, t2, t3, l3[t3 as usize])
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn qwen35_metal_predict_seq_sweep_reports_cosine() {
    let path = match gguf_path() {
        Some(p) => p,
        None => return,
    };
    for len in [3usize, 4, 5, 6, 7, 8] {
        let prompt: Vec<u32> = (1..=len as u32).collect();
        let max_seq = len.max(8);
        let mut cpu = Qwen35RunnerBuilder::default()
            .weights(&path)
            .device(Device::Cpu)
            .packed_weights(true)
            .max_seq(max_seq)
            .last_logits_only(true)
            .build()
            .expect("cpu");
        let mut metal = Qwen35RunnerBuilder::default()
            .weights(&path)
            .device(Device::Metal)
            .packed_weights(true)
            .max_seq(max_seq)
            .last_logits_only(true)
            .build()
            .expect("metal");
        let cpu_logits = cpu.predict_logits(&prompt).expect("cpu").logits;
        let metal_logits = metal.predict_logits(&prompt).expect("metal").logits;
        let cos = cosine(&cpu_logits, &metal_logits);
        eprintln!("qwen35 predict seq={len} cpu vs metal cos={cos:.6}");
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn qwen35_metal_predict_len8_matches_cpu() {
    let path = match gguf_path() {
        Some(p) => p,
        None => return,
    };
    let prompt: Vec<u32> = (1..=8).collect();
    let max_seq = 16usize;
    let mut cpu = Qwen35RunnerBuilder::default()
        .weights(&path)
        .device(Device::Cpu)
        .packed_weights(true)
        .max_seq(max_seq)
        .last_logits_only(true)
        .build()
        .expect("cpu");
    let mut metal = Qwen35RunnerBuilder::default()
        .weights(&path)
        .device(Device::Metal)
        .packed_weights(true)
        .max_seq(max_seq)
        .last_logits_only(true)
        .build()
        .expect("metal");
    let cpu_logits = cpu.predict_logits(&prompt).expect("cpu predict").logits;
    let metal_logits = metal.predict_logits(&prompt).expect("metal predict").logits;
    let cos = cosine(&cpu_logits, &metal_logits);
    eprintln!("qwen35 predict len8 cpu vs metal cos={cos:.6}");
    assert!(cos >= 0.999, "predict len8 diverged: cos={cos}");
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn qwen35_metal_prefill_seed_matches_cpu() {
    let path = match gguf_path() {
        Some(p) => p,
        None => return,
    };
    let prompt: Vec<u32> = (1..=8).collect();
    let mut cpu = Qwen35RunnerBuilder::default()
        .weights(&path)
        .device(Device::Cpu)
        .packed_weights(true)
        .max_seq(32)
        .last_logits_only(true)
        .build()
        .expect("cpu");
    let mut metal = Qwen35RunnerBuilder::default()
        .weights(&path)
        .device(Device::Metal)
        .packed_weights(true)
        .max_seq(32)
        .last_logits_only(true)
        .build()
        .expect("metal");
    let cpu_seed = cpu.prefill_seed_for_decode(&prompt).expect("cpu seed");
    let metal_seed = metal.prefill_seed_for_decode(&prompt).expect("metal seed");
    let cos = cosine(&cpu_seed.trunk_logits, &metal_seed.trunk_logits);
    eprintln!("qwen35 prefill-seed cpu vs metal cos={cos:.6}");
    assert!(cos >= 0.999, "prefill seed logits diverged: cos={cos}");
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn qwen35_metal_decode_step3_logits_match_cpu() {
    let path = match gguf_path() {
        Some(p) => p,
        None => return,
    };
    let cpu = prefill_and_two_decode_steps(&path, Device::Cpu);
    let metal = prefill_and_two_decode_steps(&path, Device::Metal);
    eprintln!("qwen35 decode trace cpu:   {cpu:?}");
    eprintln!("qwen35 decode trace metal: {metal:?}");
    assert_eq!(cpu.0, metal.0, "token1");
    assert_eq!(cpu.1, metal.1, "token2");
    assert_eq!(cpu.2, metal.2, "token3 argmax");
}

fn greedy_decode(path: &Path, device: Device, steps: usize) -> Vec<u32> {
    greedy_decode_opts(path, device, steps, false)
}

fn greedy_decode_mtp(path: &Path, device: Device, steps: usize) -> Vec<u32> {
    greedy_decode_opts(path, device, steps, true)
}

fn greedy_decode_opts(path: &Path, device: Device, steps: usize, enable_mtp: bool) -> Vec<u32> {
    let prompt: Vec<u32> = (1..=8).collect();
    let max_seq = (prompt.len() + steps).max(16);
    let mut runner = Qwen35RunnerBuilder::default()
        .weights(path)
        .device(device)
        .packed_weights(true)
        .max_seq(max_seq)
        .bucketed_decode(true)
        .last_logits_only(true)
        .enable_mtp(enable_mtp)
        .build()
        .expect("build runner");
    runner
        .generate_with_opts(&prompt, steps, SampleOpts::greedy(), |_| true)
        .expect("generate")
}

#[test]
fn qwen35_cpu_greedy_decode_baseline() {
    let path = match gguf_path() {
        Some(p) => p,
        None => {
            eprintln!("skip: set QWEN35_GGUF_PATH");
            return;
        }
    };
    let toks = greedy_decode(&path, Device::Cpu, STEPS);
    eprintln!("qwen35 decode cpu baseline: {toks:?}");
    assert_eq!(toks.len(), STEPS);
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn qwen35_metal_greedy_decode_no_bucket_matches_cpu() {
    let path = match gguf_path() {
        Some(p) => p,
        None => return,
    };
    let prompt: Vec<u32> = (1..=8).collect();
    let max_seq = 32usize;
    let mut cpu_runner = Qwen35RunnerBuilder::default()
        .weights(&path)
        .device(Device::Cpu)
        .packed_weights(true)
        .max_seq(max_seq)
        .bucketed_decode(false)
        .last_logits_only(true)
        .build()
        .expect("cpu runner");
    let mut metal_runner = Qwen35RunnerBuilder::default()
        .weights(&path)
        .device(Device::Metal)
        .packed_weights(true)
        .max_seq(max_seq)
        .bucketed_decode(false)
        .last_logits_only(true)
        .build()
        .expect("metal runner");
    let cpu = cpu_runner
        .generate_with_opts(&prompt, STEPS, SampleOpts::greedy(), |_| true)
        .expect("cpu generate");
    let metal = metal_runner
        .generate_with_opts(&prompt, STEPS, SampleOpts::greedy(), |_| true)
        .expect("metal generate");
    eprintln!("qwen35 decode no-bucket cpu:   {cpu:?}");
    eprintln!("qwen35 decode no-bucket metal: {metal:?}");
    assert_eq!(cpu, metal, "Metal non-bucket decode diverged from CPU");
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn qwen35_metal_greedy_decode_matches_cpu() {
    let path = match gguf_path() {
        Some(p) => p,
        None => return,
    };
    let cpu = greedy_decode(&path, Device::Cpu, STEPS);
    let metal = greedy_decode(&path, Device::Metal, STEPS);
    eprintln!("qwen35 decode cpu:   {cpu:?}");
    eprintln!("qwen35 decode metal: {metal:?}");
    assert_eq!(cpu, metal, "Metal cached decode diverged from CPU");
}

#[test]
#[cfg(all(target_os = "macos", feature = "mlx"))]
fn qwen35_mlx_greedy_decode_matches_cpu() {
    let path = match gguf_path() {
        Some(p) => p,
        None => return,
    };
    let cpu = greedy_decode(&path, Device::Cpu, STEPS);
    let mlx = greedy_decode(&path, Device::Mlx, STEPS);
    eprintln!("qwen35 decode cpu: {cpu:?}");
    eprintln!("qwen35 decode mlx: {mlx:?}");
    assert_eq!(cpu, mlx, "MLX cached decode diverged from CPU");
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn qwen35_mtp_metal_greedy_decode_matches_cpu() {
    let path = match mtp_gguf_path() {
        Some(p) => p,
        None => {
            eprintln!("skip MTP decode parity: materialize {DEFAULT_MTP}");
            return;
        }
    };
    let cpu = greedy_decode_mtp(&path, Device::Cpu, STEPS);
    let metal = greedy_decode_mtp(&path, Device::Metal, STEPS);
    eprintln!("qwen35 MTP decode cpu:   {cpu:?}");
    eprintln!("qwen35 MTP decode metal: {metal:?}");
    assert_eq!(cpu, metal, "Metal MTP decode diverged from CPU");
}

#[test]
#[cfg(all(target_os = "macos", feature = "mlx"))]
fn qwen35_mtp_mlx_greedy_decode_matches_cpu() {
    let path = match mtp_gguf_path() {
        Some(p) => p,
        None => return,
    };
    let cpu = greedy_decode_mtp(&path, Device::Cpu, STEPS);
    let mlx = greedy_decode_mtp(&path, Device::Mlx, STEPS);
    eprintln!("qwen35 MTP decode cpu: {cpu:?}");
    eprintln!("qwen35 MTP decode mlx: {mlx:?}");
    assert_eq!(cpu, mlx, "MLX MTP decode diverged from CPU");
}
