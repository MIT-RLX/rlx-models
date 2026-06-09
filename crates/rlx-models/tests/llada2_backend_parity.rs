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

//! CPU vs accelerator backend logits parity on synthetic LLaDA2 (env-gated).
//!
//! ```text
//! cargo test -p rlx-models --test llada2_backend_parity --features metal -- --nocapture
//! cargo test -p rlx-models --test llada2_backend_parity --features mlx -- --nocapture
//! cargo test -p rlx-models --test llada2_backend_parity --features cuda -- --nocapture
//! cargo test -p rlx-models --test llada2_backend_parity --features rocm -- --nocapture
//! cargo test -p rlx-models --test llada2_backend_parity --features vulkan -- --nocapture
//! ```

#![allow(dead_code)]

mod compile_support;

use rlx_models::llada2::{build_llada2_forward_graph, synth};
use rlx_runtime::Device;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 1.0;
    }
    dot / (na * nb)
}

fn run_logits(device: Device, seq: usize) -> Vec<f32> {
    let cfg = synth::tiny_cfg();
    let weights = synth::tiny_weights(&cfg);
    let (graph, params) = build_llada2_forward_graph(&cfg, &weights, 1, seq).expect("graph");
    let mut compiled = compile_support::compile_llada2(device, graph, params);
    let ids: Vec<f32> = (0..seq).map(|i| (i % cfg.vocab_size) as f32).collect();
    let pos: Vec<f32> = (0..seq).map(|i| i as f32).collect();
    let mask = vec![0f32; seq * seq];
    compiled
        .run(&[
            ("input_ids", &ids),
            ("position_ids", &pos),
            ("attn_mask", &mask),
        ])
        .into_iter()
        .next()
        .unwrap_or_default()
}

fn assert_backend_parity_with_cfg(
    backend: Device,
    label: &str,
    cfg: &rlx_models::llada2::LLaDA2MoeConfig,
    weights: &rlx_models::llada2::LLaDA2Weights,
    seq: usize,
) {
    let (graph, params) = build_llada2_forward_graph(cfg, weights, 1, seq).expect("graph");
    let mut compiled_cpu =
        compile_support::compile_llada2(Device::Cpu, graph.clone(), params.clone());
    let mut compiled_acc = compile_support::compile_llada2(backend, graph, params);
    let ids: Vec<f32> = (0..seq).map(|i| (i % cfg.vocab_size) as f32).collect();
    let pos: Vec<f32> = (0..seq).map(|i| i as f32).collect();
    let mask = vec![0f32; seq * seq];
    let cpu = compiled_cpu
        .run(&[
            ("input_ids", &ids),
            ("position_ids", &pos),
            ("attn_mask", &mask),
        ])
        .into_iter()
        .next()
        .unwrap_or_default();
    let acc = compiled_acc
        .run(&[
            ("input_ids", &ids),
            ("position_ids", &pos),
            ("attn_mask", &mask),
        ])
        .into_iter()
        .next()
        .unwrap_or_default();
    assert_eq!(cpu.len(), acc.len(), "{label}: len");
    let cos = cosine(&cpu, &acc);
    eprintln!("llada2 {label} vs cpu cosine={cos:.6}");
    assert!(cos > 0.99, "{label}: cosine {cos} below 0.99");
}

fn assert_backend_parity(backend: Device, label: &str) {
    let seq = 8usize;
    let cfg = synth::tiny_cfg();
    let weights = synth::tiny_weights(&cfg);
    assert_backend_parity_with_cfg(backend, label, &cfg, &weights, seq);
}

#[test]
fn llada2_cpu_baseline() {
    let cfg = synth::tiny_cfg();
    let weights = synth::tiny_weights(&cfg);
    let (graph, params) = build_llada2_forward_graph(&cfg, &weights, 1, 4).expect("graph");
    let mut compiled = compile_support::compile_llada2(Device::Cpu, graph, params);
    let ids: Vec<f32> = (0..4).map(|i| (i % cfg.vocab_size) as f32).collect();
    let pos: Vec<f32> = (0..4).map(|i| i as f32).collect();
    let mask = vec![0f32; 16];
    let _ = compiled.run(&[
        ("input_ids", &ids),
        ("position_ids", &pos),
        ("attn_mask", &mask),
    ]);
}

#[cfg(feature = "metal")]
#[test]
fn llada2_metal_matches_cpu() {
    assert_backend_parity(Device::Metal, "metal");
}

#[cfg(feature = "mlx")]
#[test]
fn llada2_mlx_dense_only_matches_cpu() {
    let mut cfg = synth::tiny_cfg();
    cfg.num_hidden_layers = 1;
    cfg.first_k_dense_replace = 1;
    let weights = synth::tiny_weights(&cfg);
    let seq = 8usize;
    let (graph, params) = build_llada2_forward_graph(&cfg, &weights, 1, seq).expect("graph");
    let mut compiled_mlx =
        compile_support::compile_llada2(Device::Mlx, graph.clone(), params.clone());
    let mut compiled_cpu = compile_support::compile_llada2(Device::Cpu, graph, params);
    let ids: Vec<f32> = (0..seq).map(|i| (i % cfg.vocab_size) as f32).collect();
    let pos: Vec<f32> = (0..seq).map(|i| i as f32).collect();
    let mask = vec![0f32; seq * seq];
    let cpu = compiled_cpu
        .run(&[
            ("input_ids", &ids),
            ("position_ids", &pos),
            ("attn_mask", &mask),
        ])
        .into_iter()
        .next()
        .unwrap_or_default();
    let mlx = compiled_mlx
        .run(&[
            ("input_ids", &ids),
            ("position_ids", &pos),
            ("attn_mask", &mask),
        ])
        .into_iter()
        .next()
        .unwrap_or_default();
    assert_eq!(cpu.len(), mlx.len());
    let cos = cosine(&cpu, &mlx);
    assert!(cos > 0.99, "dense-only mlx cosine {cos}");
}

#[cfg(feature = "mlx")]
#[test]
fn llada2_mlx_two_dense_layers_matches_cpu() {
    let mut cfg = synth::tiny_cfg();
    cfg.num_hidden_layers = 2;
    cfg.first_k_dense_replace = 2;
    assert_backend_parity_with_cfg(
        Device::Mlx,
        "mlx-two-dense",
        &cfg,
        &synth::tiny_weights(&cfg),
        8,
    );
}

#[cfg(feature = "mlx")]
#[test]
fn llada2_mlx_moe_only_layer_matches_cpu() {
    let mut cfg = synth::tiny_cfg();
    cfg.num_hidden_layers = 1;
    cfg.first_k_dense_replace = 0;
    assert_backend_parity_with_cfg(
        Device::Mlx,
        "mlx-moe-only",
        &cfg,
        &synth::tiny_weights(&cfg),
        8,
    );
}

#[cfg(feature = "mlx")]
#[test]
fn llada2_mlx_one_moe_layer_matches_cpu() {
    let mut cfg = synth::tiny_cfg();
    cfg.num_hidden_layers = 2;
    cfg.first_k_dense_replace = 1;
    assert_backend_parity_with_cfg(
        Device::Mlx,
        "mlx-one-moe",
        &cfg,
        &synth::tiny_weights(&cfg),
        8,
    );
}

#[cfg(feature = "mlx")]
#[test]
fn llada2_mlx_matches_cpu() {
    assert_backend_parity(Device::Mlx, "mlx");
}

#[cfg(feature = "cuda")]
#[test]
fn llada2_cuda_matches_cpu() {
    assert_backend_parity(Device::Cuda, "cuda");
}

#[cfg(feature = "rocm")]
#[test]
fn llada2_rocm_matches_cpu() {
    assert_backend_parity(Device::Rocm, "rocm");
}

#[cfg(feature = "gpu")]
#[test]
fn llada2_wgpu_matches_cpu() {
    assert_backend_parity(Device::Gpu, "wgpu");
}

#[cfg(feature = "vulkan")]
#[test]
fn llada2_vulkan_matches_cpu() {
    if !rlx_runtime::is_available(Device::Vulkan) {
        eprintln!("skip: Vulkan backend not available");
        return;
    }
    assert_backend_parity(Device::Vulkan, "vulkan");
}
