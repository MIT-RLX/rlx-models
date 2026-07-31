// RLX — versatile ML compiler + runtime. GPLv3.
//! End-to-end **model-side bridge** check: build the DeepSeek-V4 prefill graph
//! (tiny synthetic config, ratio-4 overlap compressor + Indexer active), then
//! run it as a multi-node PIPELINE via the model-agnostic `rlx-distributed` API
//! and assert it reproduces the single-node logits — both in-process
//! ([`run_decoder_pipeline_local`]) and over real localhost TCP sockets (one
//! worker per stage, each holding only its own weight shard). This is the exact
//! path a real DeepSeek-V4 run across several hosts would take; here on synthetic
//! weights so it runs on one machine.
//!
//!   cargo run --release -p rlx-models-core --example dsv4_distributed_probe

use anyhow::Result;
use rlx_distributed::NamedTensor;
use rlx_distributed::graph::{bind_stage, partition, run_pipeline_tcp, serve_bound};
use rlx_ir::quant::QuantScheme;
use rlx_models_core::distributed_bridge::{MapParamSource, run_decoder_pipeline_local};
use rlx_models_core::standard_decoder::{DeepseekV4Spec, build_deepseek_v4_prefill};
use rlx_models_core::weight_loader::WeightLoader;
use rlx_runtime::{CompileOptions, Device, Session};
use std::collections::HashMap;

fn rnd(seed: f64, i: usize) -> f32 {
    let x = ((i as f64 + 1.0) * (seed + 1.3) * 12.9898).sin() * 43758.5453;
    ((x - x.floor()) as f32 - 0.5) * 0.3
}

struct MemLoader {
    t: HashMap<String, (Vec<f32>, Vec<usize>)>,
}
impl WeightLoader for MemLoader {
    fn take(&mut self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.t
            .get(k)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing {k}"))
    }
    fn take_transposed(&mut self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (d, s) = self.take(k)?;
        assert_eq!(s.len(), 2, "{k}");
        let (r, c) = (s[0], s[1]);
        let mut o = vec![0f32; d.len()];
        for i in 0..r {
            for j in 0..c {
                o[j * r + i] = d[i * c + j];
            }
        }
        Ok((o, vec![c, r]))
    }
    fn len(&self) -> usize {
        self.t.len()
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.t.keys().cloned().collect()
    }
}

fn build() -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>, Vec<u32>)> {
    let (vocab, dim, nl, hc, nh, hd, rd, ql) = (16usize, 8, 2, 2, 2, 4, 2, 6);
    let (ngrp, olora, inter) = (2usize, 3, 10);
    let (ihd, in_heads, itopk, ratio) = (4usize, 2, 2, 4);
    let mix_hc = (2 + hc) * hc;
    let hcd = hc * dim;
    let dpg = nh * hd / ngrp;
    let spec = DeepseekV4Spec {
        vocab_size: vocab,
        dim,
        n_layers: nl,
        hc_mult: hc,
        n_heads: nh,
        head_dim: hd,
        rope_head_dim: rd,
        q_lora_rank: ql,
        n_groups: ngrp,
        o_lora_rank: olora,
        compress_ratios: vec![ratio; nl],
        index_head_dim: ihd,
        index_n_heads: in_heads,
        index_topk: itopk,
        window_size: 64,
        first_k_dense_replace: nl,
        n_hash_layers: 0,
        moe_intermediate_size: 8,
        n_routed_experts: 4,
        n_activated_experts: 2,
        n_shared_experts: 1,
        intermediate_size: inter,
        route_scale: 1.0,
        rope_theta: 10000.0,
        compress_rope_theta: 160000.0,
        swiglu_limit: 10.0,
        rms_norm_eps: 1e-5,
        hc_sinkhorn_iters: 5,
        hc_eps: 1e-6,
        original_seq_len: 0,
        rope_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        n_mtp_layers: 0,
        dspark_block_size: 0,
        dspark_noise_token_id: 0,
        dspark_target_layer_ids: vec![],
        dspark_markov_rank: 256,
    };
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut sd = 1.0;
    let mut put = |k: String, shape: Vec<usize>| {
        sd += 1.0;
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| rnd(sd, i) + 0.05).collect();
        t.insert(k, (data, shape));
    };
    put("model.embed_tokens.weight".into(), vec![vocab, dim]);
    for il in 0..nl {
        let p = format!("model.layers.{il}");
        put(format!("{p}.attn_hc.fn"), vec![mix_hc, hcd]);
        put(format!("{p}.attn_hc.scale"), vec![3]);
        put(format!("{p}.attn_hc.base"), vec![mix_hc]);
        put(format!("{p}.attn_norm.weight"), vec![dim]);
        put(format!("{p}.attn.wq_a.weight"), vec![ql, dim]);
        put(format!("{p}.attn.q_norm.weight"), vec![ql]);
        put(format!("{p}.attn.wq_b.weight"), vec![nh * hd, ql]);
        put(format!("{p}.attn.wkv.weight"), vec![hd, dim]);
        put(format!("{p}.attn.kv_norm.weight"), vec![hd]);
        put(format!("{p}.attn.attn_sink"), vec![nh]);
        put(format!("{p}.attn.wo_a.weight"), vec![ngrp * olora, dpg]);
        put(format!("{p}.attn.wo_b.weight"), vec![dim, ngrp * olora]);
        put(format!("{p}.attn.compressor.wkv.weight"), vec![2 * hd, dim]);
        put(
            format!("{p}.attn.compressor.wgate.weight"),
            vec![2 * hd, dim],
        );
        put(format!("{p}.attn.compressor.ape"), vec![ratio, 2 * hd]);
        put(format!("{p}.attn.compressor.norm.weight"), vec![hd]);
        put(
            format!("{p}.attn.indexer.compressor.wkv.weight"),
            vec![2 * ihd, dim],
        );
        put(
            format!("{p}.attn.indexer.compressor.wgate.weight"),
            vec![2 * ihd, dim],
        );
        put(
            format!("{p}.attn.indexer.compressor.ape"),
            vec![ratio, 2 * ihd],
        );
        put(
            format!("{p}.attn.indexer.compressor.norm.weight"),
            vec![ihd],
        );
        put(
            format!("{p}.attn.indexer.wq_b.weight"),
            vec![in_heads * ihd, ql],
        );
        put(
            format!("{p}.attn.indexer.weights_proj.weight"),
            vec![in_heads, dim],
        );
        put(format!("{p}.ffn_hc.fn"), vec![mix_hc, hcd]);
        put(format!("{p}.ffn_hc.scale"), vec![3]);
        put(format!("{p}.ffn_hc.base"), vec![mix_hc]);
        put(format!("{p}.ffn_norm.weight"), vec![dim]);
        put(format!("{p}.ffn.gate_proj.weight"), vec![inter, dim]);
        put(format!("{p}.ffn.up_proj.weight"), vec![inter, dim]);
        put(format!("{p}.ffn.down_proj.weight"), vec![dim, inter]);
    }
    put("model.hc_head.fn".into(), vec![hc, hcd]);
    put("model.hc_head.scale".into(), vec![1]);
    put("model.hc_head.base".into(), vec![hc]);
    put("model.norm.weight".into(), vec![dim]);
    put("lm_head.weight".into(), vec![vocab, dim]);

    let ids: Vec<u32> = (0..16u32).map(|i| (i * 7 + 1) % vocab as u32).collect();
    let mut loader = MemLoader { t };
    let mut packed: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
    let (graph, params) = build_deepseek_v4_prefill(&spec, &mut loader, ids.len(), &mut packed)?;
    assert!(packed.is_empty(), "synthetic config is all-f32");
    Ok((graph, params, ids))
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    d / (na.sqrt() * nb.sqrt()).max(1e-12)
}

fn main() -> Result<()> {
    let (graph, params, ids) = build()?;
    let opts = CompileOptions::default();

    // ── Single-node reference ──
    let mut c = Session::new(Device::Cpu).compile_with(graph.clone(), &opts);
    for (n, d) in &params {
        c.set_param(n, d);
    }
    let ids_f: Vec<f32> = ids.iter().map(|&x| x as f32).collect();
    let reference = c
        .run(&[("input_ids", ids_f.as_slice())])
        .into_iter()
        .next()
        .unwrap();

    // ── Distributed, in-process (bridge partitions + shards) ──
    let empty = HashMap::new();
    let local = run_decoder_pipeline_local(
        graph.clone(),
        params.clone(),
        empty,
        &ids,
        2,
        Device::Cpu,
        &opts,
    );
    let cl = cos(&local[0].data, &reference);
    let el = local[0]
        .data
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("── DeepSeek-V4 across a 2-stage pipeline via the model bridge ──");
    println!("in-process : cos {cl:.8}  max|err| {el:.3e}");

    // ── Distributed over real TCP (one worker thread per stage, own shard) ──
    let stages = partition(&graph, 2);
    let mut addrs = Vec::new();
    let mut handles = Vec::new();
    for stage in &stages {
        let (addr, listener) = bind_stage("127.0.0.1:0").expect("bind");
        addrs.push(addr.to_string());
        let stage = stage.clone();
        // Each worker gets ONLY its stage's param shard.
        let shard: HashMap<String, Vec<f32>> = stage
            .params
            .iter()
            .filter_map(|p| params.get(p).map(|v| (p.clone(), v.clone())))
            .collect();
        handles.push(std::thread::spawn(move || {
            let opts = CompileOptions::default();
            let mut src = MapParamSource::new(shard, HashMap::new());
            serve_bound(listener, stage, &mut src, Device::Cpu, &opts, 1).expect("serve");
        }));
    }
    let input = NamedTensor::new("input_ids", vec![1, ids.len()], ids_f.clone());
    let tcp = run_pipeline_tcp(&stages, &addrs, vec![input]).expect("coordinator");
    for h in handles {
        h.join().expect("worker");
    }
    let ct = cos(&tcp[0].data, &reference);
    let et = tcp[0]
        .data
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!(
        "TCP 2-node : cos {ct:.8}  max|err| {et:.3e}  (stage shards: {} + {} params)",
        stages[0].params.len(),
        stages[1].params.len()
    );

    if cl > 0.999999 && el < 1e-4 && ct > 0.999999 && et < 1e-4 {
        println!("✅ V4 runs across pipeline stages (in-process + TCP) matching single-node");
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "distributed V4 diverged: local cos {cl} tcp cos {ct}"
        ))
    }
}
