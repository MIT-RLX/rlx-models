// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Drive the real **rlx-llama4 MoE** text model (all-MoE layers, top-k router +
//! grouped expert matmul) through opscope, incl. the `Op::TopK` **routing** tap
//! (per-expert load) and the fused-attention tap. Tiny random config, no disk.
//!   cd rlx-models && cargo run -p rlx-llama4 --example opscope_llama4 --release -- /tmp/moe.csv

use std::collections::HashMap;

use rlx_core::weight_map::WeightMap;
use rlx_ir::Philox4x32;
use rlx_llama4::config::Llama4TextConfig;
use rlx_llama4::flow::build_llama4_text_flow;
use rlx_opscope::dataflow::repeated_flow_patterns;
use rlx_opscope::{
    Recorder, StatConfig, inject_attention_stats, inject_infer_stats, inject_matmul_stats,
};
use rlx_runtime::{Device, Session};

fn random_weights(cfg: &Llama4TextConfig, rng: &mut Philox4x32) -> WeightMap {
    let (h, hd) = (cfg.hidden_size, cfg.head_dim());
    let (q_dim, kv_dim) = (cfg.num_attention_heads * hd, cfg.num_key_value_heads * hd);
    let (inter, e) = (cfg.intermediate_size, cfg.num_local_experts);
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let put = |t: &mut HashMap<_, _>, k: String, shape: Vec<usize>, rng: &mut Philox4x32| {
        let n: usize = shape.iter().product();
        let mut d = vec![0f32; n];
        if k.contains("norm") {
            d.fill(1.0);
        } else {
            rng.fill_normal(&mut d);
            for v in &mut d {
                *v *= 0.02;
            }
        }
        t.insert(k, (d, shape));
    };
    put(
        &mut t,
        "model.embed_tokens.weight".into(),
        vec![cfg.vocab_size, h],
        rng,
    );
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        put(&mut t, format!("{lp}.input_layernorm.weight"), vec![h], rng);
        put(
            &mut t,
            format!("{lp}.post_attention_layernorm.weight"),
            vec![h],
            rng,
        );
        put(
            &mut t,
            format!("{lp}.self_attn.q_proj.weight"),
            vec![q_dim, h],
            rng,
        );
        put(
            &mut t,
            format!("{lp}.self_attn.k_proj.weight"),
            vec![kv_dim, h],
            rng,
        );
        put(
            &mut t,
            format!("{lp}.self_attn.v_proj.weight"),
            vec![kv_dim, h],
            rng,
        );
        put(
            &mut t,
            format!("{lp}.self_attn.o_proj.weight"),
            vec![h, q_dim],
            rng,
        );
        let ff = format!("{lp}.feed_forward");
        put(&mut t, format!("{ff}.router.weight"), vec![e, h], rng);
        put(
            &mut t,
            format!("{ff}.experts.gate_up_proj"),
            vec![e, h, 2 * inter],
            rng,
        );
        put(
            &mut t,
            format!("{ff}.experts.down_proj"),
            vec![e, inter, h],
            rng,
        );
        put(
            &mut t,
            format!("{ff}.shared_expert.gate_proj.weight"),
            vec![inter, h],
            rng,
        );
        put(
            &mut t,
            format!("{ff}.shared_expert.up_proj.weight"),
            vec![inter, h],
            rng,
        );
        put(
            &mut t,
            format!("{ff}.shared_expert.down_proj.weight"),
            vec![h, inter],
            rng,
        );
    }
    put(&mut t, "model.norm.weight".into(), vec![h], rng);
    put(
        &mut t,
        "lm_head.weight".into(),
        vec![cfg.vocab_size, h],
        rng,
    );
    WeightMap::from_tensors(t)
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "opscope_moe.csv".into());
    let cfg: Llama4TextConfig = serde_json::from_str(
        r#"{"vocab_size":64,"hidden_size":32,"intermediate_size":32,"intermediate_size_mlp":32,
            "num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
            "num_local_experts":8,"num_experts_per_tok":2,"no_rope_layer_interval":2,
            "interleave_moe_layer_step":1}"#,
    )
    .expect("cfg");
    let seq = 16usize;
    let half = cfg.head_dim() / 2;

    let mut rng = Philox4x32::new(0x4E11);
    let mut wm = random_weights(&cfg, &mut rng);
    let built = build_llama4_text_flow(&cfg, &mut wm, seq, true, false).expect("build llama4 flow");
    let (graph, params) = built.into_graph_parts().expect("graph parts");
    println!(
        "real rlx-llama4 MoE: {} layers, {} experts (top-{}), hidden {} — {} nodes, {} params\n",
        cfg.num_hidden_layers,
        cfg.num_local_experts,
        cfg.num_experts_per_tok,
        cfg.hidden_size,
        graph.nodes().len(),
        params.len()
    );

    // Structure: the repeated MoE expert block (TopK + GroupedMatMul).
    println!("[struct] top repeated dataflow blocks:");
    for p in repeated_flow_patterns(&graph, 3, 5, 2).iter().take(2) {
        println!(
            "   ×{} d{}  {}",
            p.count,
            p.depth,
            p.tree.chars().take(140).collect::<String>()
        );
    }
    println!();

    // Inject value + routing (TopK) + attention sketches.
    let scfg = StatConfig::default();
    let (g1, _mm) = inject_matmul_stats(&graph, &scfg);
    let (g2, inf1) = inject_infer_stats(&g1, &scfg); // TopK → route_load, Softmax
    let (g3, inf2) = inject_attention_stats(&g2, &scfg); // fused Op::Attention
    let mut opts = rlx_runtime::CompileOptions::default();
    opts.fusion_opts.skip_fusion = true;
    let mut c = Session::new(Device::Cpu).compile_with(g3, &opts);
    for (name, data) in &params {
        c.set_param(name, data);
    }
    let ids: Vec<f32> = (0..seq)
        .map(|_| (rng.next_u32() % cfg.vocab_size as u32) as f32)
        .collect();
    let cos: Vec<f32> = (0..seq * half).map(|k| ((k / half) as f32).cos()).collect();
    let sin: Vec<f32> = (0..seq * half).map(|k| ((k / half) as f32).sin()).collect();
    let outs = c.run(&[("input_ids", &ids), ("rope_cos", &cos), ("rope_sin", &sin)]);

    // ── MoE routing: per-expert load from each TopK router ──
    println!("[MoE] expert routing load (per top-k router):");
    for s in inf1.iter().filter(|s| s.stat == "route_load") {
        let load = &outs[s.out_idx];
        let total: f32 = load.iter().sum::<f32>().max(1.0);
        let mut pct: Vec<(usize, f32)> = load
            .iter()
            .enumerate()
            .map(|(e, &c)| (e, c / total))
            .collect();
        pct.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let cold = pct.iter().filter(|(_, f)| *f < 0.02).count();
        print!("   {:<10} ", s.site);
        for (e, f) in pct.iter().take(cfg.num_local_experts) {
            print!("e{e}:{:.0}% ", f * 100.0);
        }
        println!("  ({cold} cold → drop/merge, hottest → prefetch)");
    }
    // ── Attention concentration ──
    println!("\n[Tier1] attention concentration:");
    for s in inf2.iter().filter(|s| s.stat == "attn_qmax") {
        let qmax = &outs[s.out_idx];
        let m = qmax.iter().sum::<f32>() / qmax.len().max(1) as f32;
        println!("   {:<10} mean per-query peak {m:.2}", s.site);
    }

    let mut rec = Recorder::create(&out).expect("csv");
    rec.record(
        0,
        0,
        "cpu",
        "moe",
        seq,
        cfg.hidden_size,
        cfg.vocab_size,
        &_mm,
        &outs,
    )
    .unwrap();
    rec.flush().unwrap();
    println!("\n[value] wrote {} matmul-site sketches → {out}", _mm.len());
}
