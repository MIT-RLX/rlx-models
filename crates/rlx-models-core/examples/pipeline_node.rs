// RLX — versatile ML compiler + runtime. GPLv3.
//! Runnable **worker / coordinator binary** for the graph-node pipeline — the
//! launchable artifact behind running a model split across several SSH hosts.
//!
//! Roles:
//!   * `--role worker --rank R --addr HOST:PORT` — build the model, take stage R,
//!     serve it (holding only stage R's weight shard) until one forward is done.
//!   * `--role coordinator --peers a,b[,…]` — build the model, partition it,
//!     drive the pipeline over TCP against the workers at `--peers`.
//!   * no args — **launcher**: pick free loopback ports, spawn one worker PROCESS
//!     per stage, then run as coordinator and check the result == single-node.
//!     (The same binary scales to real hosts — only the addresses change.)
//!
//! Demonstrated on a tiny synthetic DeepSeek-V4 (HC + o-LoRA MLA + overlap
//! compressor + Indexer + FFN) so it runs on one machine; a real run swaps the
//! synthetic build for a checkpoint-backed one.
//!
//!   cargo run --release -p rlx-models-core --example pipeline_node

use anyhow::Result;
use rlx_distributed::graph::{partition, run_pipeline_tcp, serve_stage};
use rlx_distributed::{NamedTensor, free_loopback_ports};
use rlx_ir::quant::QuantScheme;
use rlx_models_core::distributed_bridge::MapParamSource;
use rlx_models_core::standard_decoder::{DeepseekV4Spec, build_deepseek_v4_prefill};
use rlx_models_core::weight_loader::WeightLoader;
use rlx_runtime::{CompileOptions, Device, Session};
use std::collections::HashMap;

const N_STAGES: usize = 2;

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

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

/// Deterministic tiny DeepSeek-V4 build (same on every process/rank).
fn build() -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>, Vec<u32>)> {
    let (vocab, dim, nl, hc, nh, hd, rd, ql) = (16usize, 8, 2, 2, 2, 4, 2, 6);
    let (ngrp, olora, inter) = (2usize, 3, 10);
    let (ihd, in_heads, itopk, ratio) = (4usize, 2, 2, 4);
    let (mix_hc, hcd, dpg) = ((2 + hc) * hc, hc * dim, nh * hd / ngrp);
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
        t.insert(k, ((0..n).map(|i| rnd(sd, i) + 0.05).collect(), shape));
    };
    put("model.embed_tokens.weight".into(), vec![vocab, dim]);
    for il in 0..nl {
        let p = format!("model.layers.{il}");
        for (suf, sh) in [
            ("attn_hc.fn", vec![mix_hc, hcd]),
            ("attn_hc.scale", vec![3]),
            ("attn_hc.base", vec![mix_hc]),
            ("attn_norm.weight", vec![dim]),
            ("attn.wq_a.weight", vec![ql, dim]),
            ("attn.q_norm.weight", vec![ql]),
            ("attn.wq_b.weight", vec![nh * hd, ql]),
            ("attn.wkv.weight", vec![hd, dim]),
            ("attn.kv_norm.weight", vec![hd]),
            ("attn.attn_sink", vec![nh]),
            ("attn.wo_a.weight", vec![ngrp * olora, dpg]),
            ("attn.wo_b.weight", vec![dim, ngrp * olora]),
            ("attn.compressor.wkv.weight", vec![2 * hd, dim]),
            ("attn.compressor.wgate.weight", vec![2 * hd, dim]),
            ("attn.compressor.ape", vec![ratio, 2 * hd]),
            ("attn.compressor.norm.weight", vec![hd]),
            ("attn.indexer.compressor.wkv.weight", vec![2 * ihd, dim]),
            ("attn.indexer.compressor.wgate.weight", vec![2 * ihd, dim]),
            ("attn.indexer.compressor.ape", vec![ratio, 2 * ihd]),
            ("attn.indexer.compressor.norm.weight", vec![ihd]),
            ("attn.indexer.wq_b.weight", vec![in_heads * ihd, ql]),
            ("attn.indexer.weights_proj.weight", vec![in_heads, dim]),
            ("ffn_hc.fn", vec![mix_hc, hcd]),
            ("ffn_hc.scale", vec![3]),
            ("ffn_hc.base", vec![mix_hc]),
            ("ffn_norm.weight", vec![dim]),
            ("ffn.gate_proj.weight", vec![inter, dim]),
            ("ffn.up_proj.weight", vec![inter, dim]),
            ("ffn.down_proj.weight", vec![dim, inter]),
        ] {
            put(format!("{p}.{suf}"), sh);
        }
    }
    put("model.hc_head.fn".into(), vec![hc, hcd]);
    put("model.hc_head.scale".into(), vec![1]);
    put("model.hc_head.base".into(), vec![hc]);
    put("model.norm.weight".into(), vec![dim]);
    put("lm_head.weight".into(), vec![vocab, dim]);
    let ids: Vec<u32> = (0..16u32).map(|i| (i * 7 + 1) % vocab as u32).collect();
    let mut loader = MemLoader { t };
    let mut packed: HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)> = HashMap::new();
    let (g, params) = build_deepseek_v4_prefill(&spec, &mut loader, ids.len(), &mut packed)?;
    Ok((g, params, ids))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let opts = CompileOptions::default();

    // ── Worker: serve one stage, holding only its shard ──
    if flag(&args, "--role").as_deref() == Some("worker") {
        let rank: usize = flag(&args, "--rank").unwrap().parse()?;
        let addr = flag(&args, "--addr").unwrap();
        let (g, params, _) = build()?;
        let stage = partition(&g, N_STAGES).into_iter().nth(rank).unwrap();
        let shard: HashMap<String, Vec<f32>> = stage
            .params
            .iter()
            .filter_map(|p| params.get(p).map(|v| (p.clone(), v.clone())))
            .collect();
        eprintln!(
            "[worker {rank}] serving on {addr} with {} params",
            shard.len()
        );
        let mut src = MapParamSource::new(shard, HashMap::new());
        serve_stage(&addr, stage, &mut src, Device::Cpu, &opts, 1)?;
        return Ok(());
    }

    // ── Launcher / coordinator ──
    let (g, params, ids) = build()?;
    // single-node reference
    let mut c = Session::new(Device::Cpu).compile_with(g.clone(), &opts);
    for (n, d) in &params {
        c.set_param(n, d);
    }
    let ids_f: Vec<f32> = ids.iter().map(|&x| x as f32).collect();
    let reference = c
        .run(&[("input_ids", ids_f.as_slice())])
        .into_iter()
        .next()
        .unwrap();

    let peers: Vec<String> = match flag(&args, "--peers") {
        Some(p) => p.split(',').map(String::from).collect(),
        None => {
            // Launcher: pick ports, spawn a worker process per stage.
            let ports = free_loopback_ports(N_STAGES as u32)?;
            let addrs: Vec<String> = ports.iter().map(|p| format!("127.0.0.1:{p}")).collect();
            let exe = std::env::current_exe()?;
            let mut children = Vec::new();
            for (rank, addr) in addrs.iter().enumerate() {
                children.push(
                    std::process::Command::new(&exe)
                        .args([
                            "--role",
                            "worker",
                            "--rank",
                            &rank.to_string(),
                            "--addr",
                            addr,
                        ])
                        .spawn()?,
                );
            }
            eprintln!(
                "[launcher] spawned {} worker processes: {addrs:?}",
                children.len()
            );
            // Coordinator drives; workers exit after one forward. Reap after.
            let stages = partition(&g, N_STAGES);
            let input = NamedTensor::new("input_ids", vec![1, ids.len()], ids_f.clone());
            let out = run_pipeline_tcp(&stages, &addrs, vec![input])?;
            for mut ch in children {
                let _ = ch.wait();
            }
            return report(&out[0].data, &reference);
        }
    };

    // Explicit coordinator (workers already running at --peers).
    let stages = partition(&g, N_STAGES);
    let input = NamedTensor::new("input_ids", vec![1, ids.len()], ids_f);
    let out = run_pipeline_tcp(&stages, &peers, vec![input])?;
    report(&out[0].data, &reference)
}

fn report(dist: &[f32], reference: &[f32]) -> Result<()> {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (a, b) in dist.iter().zip(reference) {
        dot += *a as f64 * *b as f64;
        na += *a as f64 * *a as f64;
        nb += *b as f64 * *b as f64;
    }
    let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
    let err = dist
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let am = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    };
    println!("── graph-node pipeline across worker PROCESSES vs single-node ──");
    println!(
        "cos {cos:.8}  max|err| {err:.3e}  argmax dist={} ref={}",
        am(dist),
        am(reference)
    );
    if cos > 0.999999 && err < 1e-4 {
        println!("✅ multi-process pipeline matches single-node");
        Ok(())
    } else {
        Err(anyhow::anyhow!("pipeline diverged: cos {cos}"))
    }
}
