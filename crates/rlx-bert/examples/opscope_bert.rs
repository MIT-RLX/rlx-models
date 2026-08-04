// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Drive the **real rlx-bert encoder** end-to-end through every opscope tool:
//!   Tier 3  op FLOPs/bytes → roofline + hot GEMM shapes
//!   struct  dataflow motifs → repeated attention/FFN blocks (fusion candidates)
//!   Tier 1  Softmax tap → attention concentration (if attention isn't fused)
//!   value   matmul sketches → CSV  (then opscope-mine / opscope-plan over ../rlx)
//!
//! Tiny random config (no disk, no checkpoint). Requires `just link-local`.
//!   cd rlx-models && cargo run -p rlx-bert --example opscope_bert --release -- /tmp/bert.csv
//!   cd ../rlx    && cargo run -p rlx-opscope --bin opscope-mine -- /tmp/bert.csv

use std::collections::HashMap;

use rlx_core::config::BertConfig;
use rlx_core::weight_map::WeightMap;
use rlx_ir::Philox4x32;
use rlx_opscope::dataflow::repeated_flow_patterns;
use rlx_opscope::shapes::{DEFAULT_RIDGE, gemm_shape_histogram, op_costs, roofline_class};
use rlx_opscope::{Recorder, StatConfig, inject_attention_stats, inject_matmul_stats};
use rlx_runtime::{Device, Session};

/// Full BERT weight set for `cfg`, randomly initialized (LayerNorm γ=1, biases 0).
fn random_weights(cfg: &BertConfig, rng: &mut Philox4x32) -> WeightMap {
    let (h, int) = (cfg.hidden_size, cfg.intermediate_size);
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let add = |t: &mut HashMap<_, _>, key: String, shape: Vec<usize>, rng: &mut Philox4x32| {
        let n: usize = shape.iter().product();
        let mut d = vec![0f32; n];
        if key.ends_with("LayerNorm.weight") {
            d.fill(1.0);
        } else if key.ends_with(".bias") {
            // zeros
        } else {
            rng.fill_normal(&mut d);
            for v in &mut d {
                *v *= 0.02; // standard BERT init scale
            }
        }
        t.insert(key, (d, shape));
    };
    add(
        &mut t,
        "embeddings.word_embeddings.weight".into(),
        vec![cfg.vocab_size, h],
        rng,
    );
    add(
        &mut t,
        "embeddings.position_embeddings.weight".into(),
        vec![cfg.max_position_embeddings, h],
        rng,
    );
    add(
        &mut t,
        "embeddings.token_type_embeddings.weight".into(),
        vec![cfg.type_vocab_size, h],
        rng,
    );
    add(&mut t, "embeddings.LayerNorm.weight".into(), vec![h], rng);
    add(&mut t, "embeddings.LayerNorm.bias".into(), vec![h], rng);
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("encoder.layer.{i}");
        for p in ["query", "key", "value"] {
            add(
                &mut t,
                format!("{lp}.attention.self.{p}.weight"),
                vec![h, h],
                rng,
            );
            add(
                &mut t,
                format!("{lp}.attention.self.{p}.bias"),
                vec![h],
                rng,
            );
        }
        add(
            &mut t,
            format!("{lp}.attention.output.dense.weight"),
            vec![h, h],
            rng,
        );
        add(
            &mut t,
            format!("{lp}.attention.output.dense.bias"),
            vec![h],
            rng,
        );
        add(
            &mut t,
            format!("{lp}.attention.output.LayerNorm.weight"),
            vec![h],
            rng,
        );
        add(
            &mut t,
            format!("{lp}.attention.output.LayerNorm.bias"),
            vec![h],
            rng,
        );
        add(
            &mut t,
            format!("{lp}.intermediate.dense.weight"),
            vec![int, h],
            rng,
        );
        add(
            &mut t,
            format!("{lp}.intermediate.dense.bias"),
            vec![int],
            rng,
        );
        add(
            &mut t,
            format!("{lp}.output.dense.weight"),
            vec![h, int],
            rng,
        );
        add(&mut t, format!("{lp}.output.dense.bias"), vec![h], rng);
        add(
            &mut t,
            format!("{lp}.output.LayerNorm.weight"),
            vec![h],
            rng,
        );
        add(&mut t, format!("{lp}.output.LayerNorm.bias"), vec![h], rng);
    }
    WeightMap::from_tensors(t)
}

fn mean(v: &[f32]) -> f32 {
    v.iter().sum::<f32>() / v.len().max(1) as f32
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "opscope_bert.csv".into());
    let (batch, seq) = (1usize, 32usize);
    let cfg = BertConfig {
        vocab_size: 128,
        hidden_size: 64,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        intermediate_size: 128,
        max_position_embeddings: 64,
        type_vocab_size: 2,
        layer_norm_eps: 1e-12,
        hidden_act: "gelu".into(),
    };
    let mut rng = Philox4x32::new(0xB3E7);
    let mut wm = random_weights(&cfg, &mut rng);
    let (graph, params) = rlx_bert::bert::build_bert_graph_sized(&cfg, &mut wm, batch, seq)
        .expect("build bert graph");
    println!(
        "real rlx-bert: {} layers, seq {seq}, hidden {} — {} graph nodes, {} params\n",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        graph.nodes().len(),
        params.len()
    );

    // ── Tier 3: roofline + hot GEMM shapes ──
    let costs = op_costs(&graph);
    let (tf, tb): (u64, u64) = (
        costs.iter().map(|c| c.flops).sum(),
        costs.iter().map(|c| c.bytes).sum(),
    );
    let (mut mem, mut comp) = (0u64, 0u64);
    for c in &costs {
        match roofline_class(c, DEFAULT_RIDGE) {
            "memory-bound" => mem += c.flops,
            "compute-bound" => comp += c.flops,
            _ => {}
        }
    }
    println!(
        "[Tier3] {:.3} GFLOP, {:.2} MB, {:.1} FLOP/byte — {:.0}% memory-bound",
        tf as f64 / 1e9,
        tb as f64 / 1e6,
        tf as f64 / tb.max(1) as f64,
        mem as f64 / (mem + comp).max(1) as f64 * 100.0
    );
    print!("[Tier3] hot GEMM shapes: ");
    for ((m, k, n), (ct, _)) in gemm_shape_histogram(&costs).iter().take(5) {
        print!("{m}×{k}×{n}(×{ct}) ");
    }
    println!("\n");

    // ── Structural: repeated dataflow blocks ──
    println!("[struct] top repeated dataflow blocks (fusion candidates):");
    for p in repeated_flow_patterns(&graph, 3, 5, 2).iter().take(3) {
        println!("   ×{} d{}  {}", p.count, p.depth, p.tree);
    }
    println!();

    // ── Inject value + inference-dynamics sketches, then run ──
    let scfg = StatConfig::default();
    let (g1, mm) = inject_matmul_stats(&graph, &scfg);
    let (g2, inf) = inject_attention_stats(&g1, &scfg); // taps the FUSED Op::Attention
    let mut c = Session::new(Device::Cpu).compile(g2);
    for (name, data) in &params {
        c.set_param(name, data);
    }

    let ids: Vec<f32> = (0..batch * seq)
        .map(|_| (rng.next_u32() % cfg.vocab_size as u32) as f32)
        .collect();
    let mask = vec![1.0f32; batch * seq];
    let types = vec![0.0f32; batch * seq];
    let pos: Vec<f32> = (0..batch * seq).map(|i| (i % seq) as f32).collect();
    let outs = c.run(&[
        ("input_ids", &ids),
        ("attention_mask", &mask),
        ("token_type_ids", &types),
        ("position_ids", &pos),
    ]);

    // ── Tier 1: attention concentration, tapped from the FUSED Op::Attention ──
    let attn: Vec<_> = inf.iter().filter(|s| s.stat == "attn_qmax").collect();
    println!("[Tier1] attention concentration (decomposed from fused Op::Attention):");
    for s in &attn {
        let qmax = &outs[s.out_idx];
        // matching per-key received (sinks): fraction on the single hottest key.
        let krecv = inf
            .iter()
            .find(|x| x.stat == "attn_krecv" && x.site == s.site)
            .map(|x| &outs[x.out_idx]);
        let sink = krecv
            .map(|kr| kr.iter().cloned().fold(0.0f32, f32::max) / kr.iter().sum::<f32>().max(1.0))
            .unwrap_or(0.0);
        println!(
            "   {:<10} mean per-query peak {:.2}   hottest key gets {:.0}% of its head's mass",
            s.site,
            mean(qmax),
            sink * 100.0
        );
    }
    println!();

    // ── Value sketches → CSV ──
    let mut rec = Recorder::create(&out).expect("create csv");
    rec.record(
        0,
        0,
        "cpu",
        "bert",
        seq,
        cfg.hidden_size,
        cfg.vocab_size,
        &mm,
        &outs,
    )
    .unwrap();
    rec.flush().unwrap();
    println!("[value] wrote {} matmul-site sketches → {out}", mm.len());
    println!("        next (from ../rlx): opscope-mine {out}  |  opscope-plan {out}");
}
