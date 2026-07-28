// RLX — versatile ML compiler + runtime. GPLv3.
//! Validates the **structure-only build** — the piece that lets a coordinator
//! partition a model too big to hold in RAM. `StructureLoader` builds the
//! DeepSeek-V4 graph returning each weight's SHAPE but dropping its data (peak
//! RAM = one weight), recording a per-key load manifest; each pipeline stage
//! then re-loads only its shard via `ManifestParamSource`. Asserts (a) the
//! structure build retains ~no real weight data (only in-graph synth consts),
//! and (b) the distributed run reproduces the normally-built single-node logits.
//!
//!   cargo run --release -p rlx-models-core --example dsv4_structure_only_probe

use anyhow::Result;
use rlx_distributed::NamedTensor;
use rlx_distributed::graph::{partition, run_pipeline_local};
use rlx_ir::quant::QuantScheme;
use rlx_models_core::distributed_bridge::{ManifestParamSource, StructureLoader};
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

fn spec() -> DeepseekV4Spec {
    let (dim, nl, hc, nh, hd, rd, ql) = (8usize, 2, 2, 2, 4, 2, 6);
    let (ngrp, olora, inter) = (2usize, 3, 10);
    DeepseekV4Spec {
        vocab_size: 16,
        dim,
        n_layers: nl,
        hc_mult: hc,
        n_heads: nh,
        head_dim: hd,
        rope_head_dim: rd,
        q_lora_rank: ql,
        n_groups: ngrp,
        o_lora_rank: olora,
        compress_ratios: vec![4; nl],
        index_head_dim: 4,
        index_n_heads: 2,
        index_topk: 2,
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
    }
}

fn tensor_map(s: &DeepseekV4Spec) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let (vocab, dim, nl, hc, nh, hd, ql) = (
        s.vocab_size,
        s.dim,
        s.n_layers,
        s.hc_mult,
        s.n_heads,
        s.head_dim,
        s.q_lora_rank,
    );
    let (ngrp, olora, inter) = (s.n_groups, s.o_lora_rank, s.intermediate_size);
    let (ihd, in_heads, ratio) = (s.index_head_dim, s.index_n_heads, 4usize);
    let (mix_hc, hcd, dpg) = ((2 + hc) * hc, hc * dim, nh * hd / ngrp);
    let mut t = HashMap::new();
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
    t
}

fn main() -> Result<()> {
    let spec = spec();
    let ids: Vec<u32> = (0..16u32)
        .map(|i| (i * 7 + 1) % spec.vocab_size as u32)
        .collect();
    let ids_f: Vec<f32> = ids.iter().map(|&x| x as f32).collect();
    let opts = CompileOptions::default();

    // ── Normal single-node reference ──
    let mut real1 = MemLoader {
        t: tensor_map(&spec),
    };
    let mut pk = HashMap::<String, (Vec<u8>, QuantScheme, Vec<usize>)>::new();
    let (g_ref, params_ref) = build_deepseek_v4_prefill(&spec, &mut real1, ids.len(), &mut pk)?;
    let mut c = Session::new(Device::Cpu).compile_with(g_ref, &opts);
    for (n, d) in &params_ref {
        c.set_param(n, d);
    }
    let reference = c
        .run(&[("input_ids", ids_f.as_slice())])
        .into_iter()
        .next()
        .unwrap();
    let full_elems: usize = params_ref.values().map(|v| v.len()).sum();

    // ── Structure-only build (shapes only, data dropped) ──
    let mut real2 = MemLoader {
        t: tensor_map(&spec),
    };
    let (graph, sparams, manifest) = {
        let mut sl = StructureLoader::new(&mut real2);
        let mut pk2 = HashMap::<String, (Vec<u8>, QuantScheme, Vec<usize>)>::new();
        let (g, sp) = build_deepseek_v4_prefill(&spec, &mut sl, ids.len(), &mut pk2)?;
        (g, sp, std::mem::take(&mut sl.manifest))
    };
    let held_elems: usize = sparams.values().map(|v| v.len()).sum();
    let deferred: usize = manifest.len();
    println!("── DeepSeek-V4 structure-only build + per-stage shard load ──");
    println!(
        "structure build held {held_elems} f32 (synth consts) vs full {full_elems} → \
         {:.1}% of weights deferred across {deferred} tensors",
        100.0 * (1.0 - held_elems as f64 / full_elems as f64)
    );
    // Every manifested (real) weight must be empty in the structure params.
    let leaked: Vec<String> = manifest
        .keys()
        .filter(|k| sparams.get(*k).is_some_and(|v| !v.is_empty()))
        .cloned()
        .collect();
    assert!(
        leaked.is_empty(),
        "structure build leaked weight data: {leaked:?}"
    );

    // ── Distributed run: each stage re-loads only its shard via the manifest ──
    let synth: HashMap<String, Vec<f32>> =
        sparams.into_iter().filter(|(_, v)| !v.is_empty()).collect();
    let stages = partition(&graph, 2);
    let mut real3 = MemLoader {
        t: tensor_map(&spec),
    };
    let mut src = ManifestParamSource {
        loader: &mut real3,
        manifest,
        synth,
        synth_packed: HashMap::new(),
    };
    let input = NamedTensor::new("input_ids", vec![1, ids.len()], ids_f);
    let out = run_pipeline_local(stages, &mut src, vec![input], Device::Cpu, &opts);

    let cos = {
        let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (a, b) in out[0].data.iter().zip(&reference) {
            d += *a as f64 * *b as f64;
            na += *a as f64 * *a as f64;
            nb += *b as f64 * *b as f64;
        }
        d / (na.sqrt() * nb.sqrt()).max(1e-12)
    };
    let err = out[0]
        .data
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("distributed (structure-only build) vs single-node: cos {cos:.8}  max|err| {err:.3e}");
    if held_elems < full_elems && leaked.is_empty() && cos > 0.999999 && err < 1e-4 {
        println!(
            "✅ structure-only build defers weights + per-stage shard load matches single-node"
        );
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "structure-only mismatch: cos {cos} held {held_elems}/{full_elems}"
        ))
    }
}
