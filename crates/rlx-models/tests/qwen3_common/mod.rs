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

#![allow(dead_code)]

//! Shared synthetic Qwen3 weights and backend test helpers.

use rlx_models::qwen3::build_qwen3_graph_sized_last_logits;
use rlx_models::qwen3::{Qwen3Config, Qwen3Generator, SampleOpts};
use rlx_models::weight_map::WeightMap;
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

const SEQ: usize = 4;

pub fn tiny_cfg() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        hidden_act: "silu".into(),
        tie_word_embeddings: false,
        attention_bias: false,
        qk_norm: true,
        sliding_window: None,
        max_window_layers: usize::MAX,
        use_sliding_window: false,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        shared_expert_ffn_size: 0,
        expert_weights_scale: 1.0,
    }
}

fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| 0.001 + scale * (i as f32) * 0.01).collect()
}

pub fn synthetic_weights_with_lm_head(cfg: &Qwen3Config) -> WeightMap {
    let h = cfg.hidden_size;
    let q_dim = cfg.q_proj_dim();
    let kv_dim = cfg.kv_proj_dim();
    let int_dim = cfg.intermediate_size;
    let dh = cfg.head_dim;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

    t.insert(
        "model.embed_tokens.weight".into(),
        (ramp(cfg.vocab_size * h, 0.001), vec![cfg.vocab_size, h]),
    );
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        t.insert(
            format!("{lp}.input_layernorm.weight"),
            (vec![1.0; h], vec![h]),
        );
        t.insert(
            format!("{lp}.post_attention_layernorm.weight"),
            (vec![1.0; h], vec![h]),
        );
        t.insert(
            format!("{lp}.self_attn.q_proj.weight"),
            (ramp(q_dim * h, 0.01), vec![q_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.k_proj.weight"),
            (ramp(kv_dim * h, 0.01), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.v_proj.weight"),
            (ramp(kv_dim * h, 0.01), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.o_proj.weight"),
            (ramp(h * q_dim, 0.01), vec![h, q_dim]),
        );
        t.insert(
            format!("{lp}.self_attn.q_norm.weight"),
            (vec![1.0; dh], vec![dh]),
        );
        t.insert(
            format!("{lp}.self_attn.k_norm.weight"),
            (vec![1.0; dh], vec![dh]),
        );
        t.insert(
            format!("{lp}.mlp.gate_proj.weight"),
            (ramp(int_dim * h, 0.01), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.up_proj.weight"),
            (ramp(int_dim * h, 0.01), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.down_proj.weight"),
            (ramp(h * int_dim, 0.01), vec![h, int_dim]),
        );
    }
    t.insert("model.norm.weight".into(), (vec![1.0; h], vec![h]));
    t.insert(
        "lm_head.weight".into(),
        (ramp(cfg.vocab_size * h, 0.001), vec![cfg.vocab_size, h]),
    );
    WeightMap::from_tensors(t)
}

struct CachedTiny {
    compiled: CompiledGraph,
    ids: Vec<f32>,
    vocab: usize,
}

fn per_device_cache() -> &'static Mutex<HashMap<Device, CachedTiny>> {
    static CACHE: OnceLock<Mutex<HashMap<Device, CachedTiny>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn compile_tiny(device: Device) -> CachedTiny {
    let cfg = tiny_cfg();
    let mut wm = synthetic_weights_with_lm_head(&cfg);
    let (graph, params) =
        build_qwen3_graph_sized_last_logits(&cfg, &mut wm, 1, SEQ, false).expect("build");
    let compiled =
        rlx_models::flow_util::compile_graph_qwen3_prefill_with_params(device, graph, params)
            .expect("compile");
    CachedTiny {
        ids: (0..SEQ).map(|i| (i + 1) as f32).collect(),
        vocab: cfg.vocab_size,
        compiled,
    }
}

/// Last-position logits from a tiny prefill graph on `device` (compile once per device).
pub fn run_last_logits(device: Device) -> Vec<f32> {
    let mut cache = per_device_cache().lock().unwrap();
    let entry = cache.entry(device).or_insert_with(|| compile_tiny(device));
    entry
        .compiled
        .run(&[("input_ids", entry.ids.as_slice())])
        .into_iter()
        .next()
        .expect("logits output")
}

/// Compile + run prefill; assert finite last-position logits.
pub fn run_last_logits_prefill(device: Device) {
    let logits = run_last_logits(device);
    assert_eq!(logits.len(), tiny_cfg().vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

/// Like [`run_last_logits_prefill`], but returns early when `device` is unavailable.
pub fn run_last_logits_prefill_if_available(device: Device) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip qwen3 {device:?}: backend not available");
        return;
    }
    run_last_logits_prefill(device);
}

/// Greedy tokens via [`Qwen3Generator::generate`] (KV-cache decode path).
pub fn run_generator_greedy(device: Device) {
    let cfg = tiny_cfg();
    let mut wm = synthetic_weights_with_lm_head(&cfg);
    let mut generator = Qwen3Generator::from_loader(cfg, &mut wm, device).expect("generator");
    generator.prefill(&[1, 2, 3]);
    let out = generator
        .generate(2, SampleOpts::greedy())
        .expect("generate");
    assert_eq!(out.len(), 2);
    let vocab = tiny_cfg().vocab_size as u32;
    assert!(out.iter().all(|&t| t < vocab));
}

/// Like [`run_generator_greedy`], but returns early when `device` is unavailable.
pub fn run_generator_greedy_if_available(device: Device) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip qwen3 generator {device:?}: backend not available");
        return;
    }
    run_generator_greedy(device);
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

static CPU_LOGITS: OnceLock<Vec<f32>> = OnceLock::new();

fn cpu_logits_reference() -> &'static [f32] {
    CPU_LOGITS.get_or_init(|| run_last_logits(Device::Cpu))
}

/// CPU vs `device` logits cosine; skips when the backend is unavailable.
pub fn assert_logits_match_cpu(device: Device, label: &str) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip qwen3 {label}: backend not available");
        return;
    }
    let cpu = cpu_logits_reference();
    let other = run_last_logits(device);
    let c = cosine(cpu, &other);
    eprintln!("qwen3 cpu vs {label} cosine={c:.6}");
    assert!(c > 0.99, "cpu vs {label} cosine {c}");
}
