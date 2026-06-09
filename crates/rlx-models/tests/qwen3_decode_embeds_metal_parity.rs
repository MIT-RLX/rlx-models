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

//! Synthetic 1-layer `inputs_embeds` decode graph: CPU vs Metal thunk parity.
//!
//! ```bash
//! cargo test -p rlx-models --test qwen3_decode_embeds_metal_parity --release --features metal -- --nocapture
//! ```

use rlx_models::qwen3::{Qwen3Config, Qwen3DecodeOpts, build_qwen3_decode_embeds_built};
use rlx_models::weight_map::WeightMap;
use rlx_runtime::compile_cache::pad_rows;
use rlx_runtime::{CompileOptions, Device, Session, is_available};
use std::collections::HashMap;

fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| 0.001 + scale * (i as f32) * 0.01).collect()
}

fn talkerish_cfg() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 3072,
        hidden_size: 1024,
        intermediate_size: 3072,
        num_hidden_layers: 1,
        num_attention_heads: 16,
        num_key_value_heads: 8,
        head_dim: 128,
        max_position_embeddings: 32768,
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

fn synthetic_weights(cfg: &Qwen3Config) -> WeightMap {
    let h = cfg.hidden_size;
    let q_dim = cfg.q_proj_dim();
    let kv_dim = cfg.kv_proj_dim();
    let int_dim = cfg.intermediate_size;
    let dh = cfg.head_dim;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let lp = "model.layers.0";
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
    t.insert("model.norm.weight".into(), (vec![1.0; h], vec![h]));
    WeightMap::from_tensors(t)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn synthetic_1l_decode_embeds_cpu_vs_metal() {
    if !is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");

    let cfg = talkerish_cfg();
    let upper = 16usize;
    let past_seq = 13usize;
    let kv_dim = cfg.kv_proj_dim();
    let h = cfg.hidden_size;
    let half = cfg.head_dim / 2;

    let mut wm = synthetic_weights(&cfg);
    let built = build_qwen3_decode_embeds_built(
        &cfg,
        &mut wm,
        &Qwen3DecodeOpts {
            batch: 1,
            past_seq: upper,
            dynamic_past: false,
            use_custom_mask: true,
            profile: None,
        },
    )
    .expect("build decode");
    let (graph, params) = built.into_graph_parts().expect("graph");
    for (i, &oid) in graph.outputs.iter().enumerate() {
        let sh = &graph.node(oid).shape;
        eprintln!(
            "graph output[{i}] shape={sh:?} elems={}",
            sh.num_elements().unwrap_or(0)
        );
    }

    let mut opts = CompileOptions::default();
    opts.fusion_opts.skip_fusion = true;
    let mut cpu = Session::new(Device::Cpu).compile_with(graph.clone(), &opts);
    let mut metal = Session::new(Device::Metal).compile_with(graph, &opts);
    for (name, data) in &params {
        cpu.set_param(name, data);
        metal.set_param(name, data);
    }

    let emb: Vec<f32> = ramp(h, 0.02);
    let rope_cos: Vec<f32> = ramp(half, 0.03);
    let rope_sin: Vec<f32> = ramp(half, 0.04);
    let mut mask = vec![0f32; upper + 1];
    for (i, slot) in mask.iter_mut().enumerate() {
        *slot = if i < past_seq || i == upper { 1.0 } else { 0.0 };
    }
    let past_k: Vec<f32> = ramp(upper * kv_dim, 0.05);
    let past_v: Vec<f32> = ramp(upper * kv_dim, 0.06);
    let padded_k = pad_rows(&past_k, kv_dim, upper as u64);
    let padded_v = pad_rows(&past_v, kv_dim, upper as u64);

    let inputs = [
        ("inputs_embeds", emb.as_slice()),
        ("rope_cos", rope_cos.as_slice()),
        ("rope_sin", rope_sin.as_slice()),
        ("mask", mask.as_slice()),
        ("past_k_0", padded_k.as_slice()),
        ("past_v_0", padded_v.as_slice()),
    ];

    let cpu_all = cpu.run(&inputs);
    let metal_all = metal.run(&inputs);
    eprintln!(
        "run lens cpu={:?} metal={:?}",
        cpu_all.iter().map(|v| v.len()).collect::<Vec<_>>(),
        metal_all.iter().map(|v| v.len()).collect::<Vec<_>>()
    );
    let cpu_out = cpu_all[0].clone();
    let metal_out = metal_all[0].clone();
    let n = h.min(cpu_out.len()).min(metal_out.len());
    let d = max_abs(&cpu_out[..n], &metal_out[..n]);
    eprintln!(
        "synthetic 1L decode cpu vs metal max_abs={d} out_lens cpu={} metal={}",
        cpu_out.len(),
        metal_out.len()
    );
    if d >= 0.05 {
        eprintln!("cpu[:8]={:?}", &cpu_out[..8.min(n)]);
        eprintln!("metal[:8]={:?}", &metal_out[..8.min(n)]);
    }
    assert!(d < 0.05, "synthetic decode metal diverged (max_abs={d})");

    rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
}
