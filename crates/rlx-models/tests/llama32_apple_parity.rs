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

// CPU vs Apple backend parity on a tiny synthetic llama32 graph.
//
//   cargo test -p rlx-models --test llama32_apple_parity --features "metal,mlx" --release

#![allow(dead_code, clippy::unnecessary_lazy_evaluations)]

mod compile_support;

use rlx_models::weight_map::WeightMap;
use rlx_models::{Llama32Config, build_llama32_graph_sized_last_logits};
use rlx_runtime::Device;
use std::collections::HashMap;

fn tiny_cfg() -> Llama32Config {
    Llama32Config {
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-5,
        rope_theta: 500_000.0,
        hidden_act: "silu".into(),
        tie_word_embeddings: false,
        attention_bias: false,
        head_dim: None,
        rope_scaling: None,
        num_loops: 1,
        skip_loop_final_norm: false,
        rope_style: rlx_ir::RopeStyle::NeoX,
        gguf_arch: None,
        rope_dim: None,
    }
}

fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| 0.001 + scale * (i as f32) * 0.01).collect()
}

fn synthetic_weights(cfg: &Llama32Config) -> WeightMap {
    let h = cfg.hidden_size;
    let q_dim = cfg.q_proj_dim();
    let kv_dim = cfg.kv_proj_dim();
    let int_dim = cfg.intermediate_size;
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

fn run_last_logits(device: Device) -> Vec<f32> {
    let cfg = tiny_cfg();
    let mut wm = synthetic_weights(&cfg);
    let (graph, params) =
        build_llama32_graph_sized_last_logits(&cfg, &mut wm, 1, 4, false).expect("build");
    let mut compiled = compile_support::compile_llama32_prefill(device, graph, params.clone());

    let ids = vec![1.0f32, 2.0, 3.0, 4.0];
    let outs = compiled.run(&[("input_ids", &ids), ("last_token_idx", &[3.0f32])]);
    outs[0].to_vec()
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

#[test]
fn cpu_reference_logits_finite() {
    let logits = run_last_logits(Device::Cpu);
    assert_eq!(logits.len(), tiny_cfg().vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_matches_cpu_logits() {
    let cpu = run_last_logits(Device::Cpu);
    let metal = run_last_logits(Device::Metal);
    let c = cosine(&cpu, &metal);
    eprintln!("llama32 cpu vs metal cosine={c:.6}");
    assert!(c > 0.99, "cpu vs metal cosine {c}");
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
#[test]
fn mlx_matches_cpu_logits() {
    let cpu = run_last_logits(Device::Cpu);
    let mlx = run_last_logits(Device::Mlx);
    let c = cosine(&cpu, &mlx);
    eprintln!("llama32 cpu vs mlx cosine={c:.6}");
    assert!(c > 0.99, "cpu vs mlx cosine {c}");
}
