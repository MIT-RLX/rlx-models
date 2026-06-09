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

//! Shared synthetic Gemma weights and multi-backend test helpers.

#![allow(dead_code)]

use rlx_flow::CompileProfile;
use rlx_models::flow_util::{
    compile_graph_gemma_prefill_with_params, compile_graph_with_kv_export_params,
};
use rlx_models::gemma::{
    GemmaArch, GemmaConfig, GemmaGenerator, build_gemma_graph_sized_last_logits,
    config::GemmaRopeMap, decode_profile_for_device,
};
use rlx_models::weight_map::WeightMap;
use rlx_qwen3::sampling::SampleOpts;
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

const SEQ: usize = 4;

pub fn tiny_cfg() -> GemmaConfig {
    GemmaConfig {
        arch: GemmaArch::Gemma,
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        tie_word_embeddings: true,
        attention_bias: false,
        head_dim: Some(8),
        attn_logit_softcapping: None,
        final_logit_softcapping: None,
        sliding_window: None,
        query_pre_attn_scalar: None,
        effective_num_layers: None,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        expert_weights_scale: 1.0,
        // Gemma 4 additions — defaults match the no-Gemma-4
        // behavior so the existing Gemma 1/2 paths stay unchanged.
        layer_types: Vec::new(),
        rope_parameters: GemmaRopeMap::default(),
        global_head_dim: None,
        num_global_key_value_heads: None,
        attention_k_eq_v: false,
        use_bidirectional_attention: None,
    }
}

pub fn tiny_gemma2_cfg() -> GemmaConfig {
    let mut cfg = tiny_cfg();
    cfg.arch = GemmaArch::Gemma2;
    cfg.attn_logit_softcapping = Some(50.0);
    cfg.final_logit_softcapping = Some(30.0);
    cfg.sliding_window = None;
    cfg.query_pre_attn_scalar = None;
    cfg
}

fn ramp(n: usize, scale: f32, salt: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(salt)) >> 8;
            (x as f32 / (1u32 << 24) as f32 - 0.5) * scale
        })
        .collect()
}

pub fn synthetic_weights(cfg: &GemmaConfig) -> WeightMap {
    let h = cfg.hidden_size;
    let q_dim = cfg.q_proj_dim();
    let kv_dim = cfg.kv_proj_dim();
    let int_dim = cfg.intermediate_size;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

    t.insert(
        "model.embed_tokens.weight".into(),
        (ramp(cfg.vocab_size * h, 0.02, 1), vec![cfg.vocab_size, h]),
    );
    for layer in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{layer}");
        let salt = layer as u32 * 17;
        t.insert(
            format!("{lp}.input_layernorm.weight"),
            (ramp(h, 0.001, salt), vec![h]),
        );
        t.insert(
            format!("{lp}.post_attention_layernorm.weight"),
            (ramp(h, 0.001, salt + 1), vec![h]),
        );
        if cfg.arch == GemmaArch::Gemma2 {
            t.insert(
                format!("{lp}.pre_feedforward_layernorm.weight"),
                (ramp(h, 0.001, salt + 10), vec![h]),
            );
            t.insert(
                format!("{lp}.post_feedforward_layernorm.weight"),
                (ramp(h, 0.001, salt + 11), vec![h]),
            );
        }
        t.insert(
            format!("{lp}.self_attn.q_proj.weight"),
            (ramp(q_dim * h, 0.01, salt + 2), vec![q_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.k_proj.weight"),
            (ramp(kv_dim * h, 0.01, salt + 3), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.v_proj.weight"),
            (ramp(kv_dim * h, 0.01, salt + 4), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.o_proj.weight"),
            (ramp(h * q_dim, 0.01, salt + 5), vec![h, q_dim]),
        );
        t.insert(
            format!("{lp}.mlp.gate_proj.weight"),
            (ramp(int_dim * h, 0.01, salt + 6), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.up_proj.weight"),
            (ramp(int_dim * h, 0.01, salt + 7), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.down_proj.weight"),
            (ramp(h * int_dim, 0.01, salt + 8), vec![h, int_dim]),
        );
    }
    t.insert("model.norm.weight".into(), (ramp(h, 0.001, 99), vec![h]));
    WeightMap::from_tensors(t)
}

struct CachedRun {
    compiled: CompiledGraph,
    ids: Vec<f32>,
    vocab: usize,
}

fn per_device_cache(gemma2: bool) -> &'static Mutex<HashMap<Device, CachedRun>> {
    static CACHE_G1: OnceLock<Mutex<HashMap<Device, CachedRun>>> = OnceLock::new();
    static CACHE_G2: OnceLock<Mutex<HashMap<Device, CachedRun>>> = OnceLock::new();
    if gemma2 {
        CACHE_G2.get_or_init(|| Mutex::new(HashMap::new()))
    } else {
        CACHE_G1.get_or_init(|| Mutex::new(HashMap::new()))
    }
}

fn compile_tiny(device: Device, gemma2: bool) -> CachedRun {
    let cfg = if gemma2 {
        tiny_gemma2_cfg()
    } else {
        tiny_cfg()
    };
    let mut wm = synthetic_weights(&cfg);
    let (graph, params) =
        build_gemma_graph_sized_last_logits(&cfg, &mut wm, 1, SEQ, false).expect("build");
    let compiled = compile_graph_gemma_prefill_with_params(device, graph, params).expect("compile");
    CachedRun {
        ids: (0..SEQ).map(|i| (i + 1) as f32).collect(),
        vocab: cfg.vocab_size,
        compiled,
    }
}

/// Last-position logits from a tiny Gemma 1 prefill graph on `device`.
pub fn run_last_logits(device: Device) -> Vec<f32> {
    run_last_logits_inner(device, false)
}

/// Last-position logits from a tiny Gemma 2 prefill graph on `device`.
pub fn run_last_logits_gemma2(device: Device) -> Vec<f32> {
    run_last_logits_inner(device, true)
}

fn run_last_logits_inner(device: Device, gemma2: bool) -> Vec<f32> {
    let mut cache = per_device_cache(gemma2).lock().unwrap();
    let entry = cache
        .entry(device)
        .or_insert_with(|| compile_tiny(device, gemma2));
    entry
        .compiled
        .run(&[("input_ids", entry.ids.as_slice())])
        .into_iter()
        .next()
        .expect("logits output")
}

pub fn run_last_logits_prefill(device: Device) {
    let logits = run_last_logits(device);
    assert_eq!(logits.len(), tiny_cfg().vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

pub fn run_last_logits_prefill_gemma2(device: Device) {
    let logits = run_last_logits_gemma2(device);
    assert_eq!(logits.len(), tiny_gemma2_cfg().vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

pub fn run_last_logits_prefill_if_available(device: Device) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip gemma {device:?}: backend not available");
        return;
    }
    run_last_logits_prefill(device);
}

/// Greedy tokens via full prefill [`GemmaGenerator::generate`] (no KV-cache path).
pub fn run_generator_greedy(device: Device) {
    let cfg = tiny_cfg();
    let mut wm = synthetic_weights(&cfg);
    let mut generator = GemmaGenerator::from_loader(cfg, &mut wm, device).expect("generator");
    generator.prefill(&[1, 2, 3]);
    let out = generator
        .generate(2, SampleOpts::greedy())
        .expect("generate");
    assert_eq!(out.len(), 2);
    let vocab = tiny_cfg().vocab_size as u32;
    assert!(out.iter().all(|&t| t < vocab));
}

pub fn run_generator_greedy_if_available(device: Device) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip gemma generator {device:?}: backend not available");
        return;
    }
    run_generator_greedy(device);
}

/// Prefill with KV side outputs (unfused compile — required for correct taps).
pub fn run_prefill_with_kv(device: Device) {
    let cfg = tiny_cfg();
    let mut wm = synthetic_weights(&cfg);
    use rlx_models::gemma::build_gemma_graph_sized_last_logits;
    let (graph, params) =
        build_gemma_graph_sized_last_logits(&cfg, &mut wm, 1, SEQ, true).expect("build kv");
    let mut compiled = compile_graph_with_kv_export_params(
        device,
        graph,
        params,
        &CompileProfile::gemma_prefill(),
    )
    .expect("compile kv");
    let outs = compiled.run(&[(
        "input_ids",
        &(1..=SEQ as u32).map(|i| i as f32).collect::<Vec<_>>(),
    )]);
    assert_eq!(outs.len(), 1 + 2 * cfg.num_hidden_layers);
    assert!(outs[0].iter().all(|v| v.is_finite()));
    for o in &outs[1..] {
        assert!(o.iter().all(|v| v.is_finite()), "kv side output non-finite");
    }
}

pub fn run_prefill_with_kv_if_available(device: Device) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip gemma prefill+kv {device:?}: backend not available");
        return;
    }
    run_prefill_with_kv(device);
}

/// One decode step after prefill (KV-cache graph + RoPE slice inputs).
pub fn run_decode_step(device: Device) {
    let cfg = tiny_cfg();
    let mut wm = synthetic_weights(&cfg);
    let mut generator = GemmaGenerator::from_loader(cfg.clone(), &mut wm, device)
        .expect("generator")
        .with_compile_profiles(
            CompileProfile::gemma_prefill(),
            decode_profile_for_device(device),
        );
    generator
        .prefill_get_last_logits(&[1, 2, 3])
        .expect("prefill");
    let logits = generator.decode_get_logits(5).expect("decode");
    assert_eq!(logits.len(), cfg.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

pub fn run_decode_step_if_available(device: Device) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip gemma decode {device:?}: backend not available");
        return;
    }
    run_decode_step(device);
}

pub fn run_decode_step_gemma2(device: Device) {
    let cfg = tiny_gemma2_cfg();
    let mut wm = synthetic_weights(&cfg);
    let mut generator = GemmaGenerator::from_loader(cfg.clone(), &mut wm, device)
        .expect("generator")
        .with_compile_profiles(
            CompileProfile::gemma_prefill(),
            decode_profile_for_device(device),
        );
    generator
        .prefill_get_last_logits(&[1, 2, 3])
        .expect("prefill");
    let logits = generator.decode_get_logits(5).expect("decode");
    assert_eq!(logits.len(), cfg.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

/// Greedy generation on a tiny Gemma 2 config (validates V2 decode graphs).
pub fn run_generator_greedy_gemma2(device: Device) {
    let cfg = tiny_gemma2_cfg();
    let mut wm = synthetic_weights(&cfg);
    let mut generator = GemmaGenerator::from_loader(cfg, &mut wm, device).expect("generator");
    generator.prefill(&[1, 2, 3]);
    let out = generator
        .generate(2, SampleOpts::greedy())
        .expect("generate");
    assert_eq!(out.len(), 2);
}

pub fn run_decode_step_gemma2_if_available(device: Device) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip gemma2 decode {device:?}: backend not available");
        return;
    }
    run_decode_step_gemma2(device);
}

pub fn run_last_logits_prefill_gemma2_if_available(device: Device) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip gemma2 prefill {device:?}: backend not available");
        return;
    }
    run_last_logits_prefill_gemma2(device);
}

/// Cached greedy generation (decode graph + KV cache) vs naive full prefill per step.
pub fn run_cached_matches_naive(device: Device) {
    let cfg = tiny_cfg();
    let prompt: Vec<u32> = vec![1, 2, 3];
    let steps = 2usize;
    let profile = CompileProfile::gemma_prefill();
    let decode = decode_profile_for_device(device);

    let mut wm_n = synthetic_weights(&cfg);
    let mut naive = GemmaGenerator::from_loader(cfg.clone(), &mut wm_n, device)
        .expect("naive gen")
        .with_compile_profiles(profile.clone(), decode.clone());
    naive.prefill(&prompt);
    let naive_tokens = naive.generate(steps, SampleOpts::greedy()).expect("naive");

    let mut wm_c = synthetic_weights(&cfg);
    let mut cached = GemmaGenerator::from_loader(cfg, &mut wm_c, device)
        .expect("cached gen")
        .with_compile_profiles(profile, decode);
    cached.prefill(&prompt);
    let cached_tokens = cached
        .generate_cached(steps, SampleOpts::greedy())
        .expect("cached");

    assert_eq!(
        cached_tokens, naive_tokens,
        "cached vs naive mismatch on {device:?}"
    );
}

pub fn run_cached_matches_naive_if_available(device: Device) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip gemma cached {device:?}: backend not available");
        return;
    }
    run_cached_matches_naive(device);
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
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

#[allow(dead_code)]
static CPU_LOGITS: OnceLock<Vec<f32>> = OnceLock::new();

#[allow(dead_code)]
fn cpu_logits_reference() -> &'static [f32] {
    CPU_LOGITS.get_or_init(|| run_last_logits(Device::Cpu))
}

#[allow(dead_code)]
pub fn assert_logits_match_cpu(device: Device, label: &str) {
    assert_logits_match_cpu_inner(device, label, false);
}

pub fn assert_logits_match_cpu_gemma2(device: Device, label: &str) {
    assert_logits_match_cpu_inner(device, label, true);
}

fn assert_logits_match_cpu_inner(device: Device, label: &str, gemma2: bool) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip gemma {label}: backend not available");
        return;
    }
    let cpu = if gemma2 {
        CPU_LOGITS_G2.get_or_init(|| run_last_logits_gemma2(Device::Cpu))
    } else {
        cpu_logits_reference()
    };
    let other = if gemma2 {
        run_last_logits_gemma2(device)
    } else {
        run_last_logits(device)
    };
    let c = cosine(cpu, &other);
    eprintln!(
        "gemma{} cpu vs {label} cosine={c:.6}",
        if gemma2 { "2" } else { "" }
    );
    assert!(c > 0.99, "cpu vs {label} cosine {c}");
}

static CPU_LOGITS_G2: OnceLock<Vec<f32>> = OnceLock::new();
