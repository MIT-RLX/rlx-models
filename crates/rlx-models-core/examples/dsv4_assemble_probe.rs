// RLX — versatile ML compiler + runtime. GPLv3.
//! Construction / finite check for the assembled DeepSeek-V4 forward
//! ([`build_deepseek_v4_prefill`]): embed → HC-expand → per-block [HC(o-LoRA
//! MLA) + HC(dense FFN)] → hc_head → norm → lm_head, composing the validated
//! cores. Runs on a tiny synthetic config with **ratio-4 compression on every
//! layer** and a small `index_topk` so the overlapping KV Compressor AND the
//! learned Indexer top-k gate are both exercised (each cos-exact-validated in
//! its own probe), asserting the whole graph builds + runs finite. No checkpoint.
//!
//!   cargo run --release -p rlx-models-core --example dsv4_assemble_probe

use anyhow::Result;
use rlx_models_core::standard_decoder::{DeepseekV4Spec, build_deepseek_v4_prefill};
use rlx_models_core::weight_loader::WeightLoader;
use rlx_runtime::{Device, Session};
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

fn main() -> Result<()> {
    let (vocab, dim, nl, hc, nh, hd, rd, ql) = (
        16usize, 8usize, 2usize, 2usize, 2usize, 4usize, 2usize, 6usize,
    );
    let (ngrp, olora, inter) = (2usize, 3usize, 10usize);
    // ratio-4 compression on both layers with a small index_topk so the overlap
    // compressor AND the learned Indexer top-k gate are both exercised
    // (ncomp = seq/ratio = 4 > index_topk = 2 → indexer_on).
    let (ihd, in_heads, itopk, ratio) = (4usize, 2usize, 2usize, 4usize);
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
    let mut put = |k: String, shape: Vec<usize>, seed: f64| {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| rnd(seed, i) + 0.05).collect();
        t.insert(k, (data, shape));
    };
    let mut sd = 1.0;
    let mut s = || {
        sd += 1.0;
        sd
    };
    put("model.embed_tokens.weight".into(), vec![vocab, dim], s());
    for il in 0..nl {
        let p = format!("model.layers.{il}");
        put(format!("{p}.attn_hc.fn"), vec![mix_hc, hcd], s());
        put(format!("{p}.attn_hc.scale"), vec![3], s());
        put(format!("{p}.attn_hc.base"), vec![mix_hc], s());
        put(format!("{p}.attn_norm.weight"), vec![dim], s());
        put(format!("{p}.attn.wq_a.weight"), vec![ql, dim], s());
        put(format!("{p}.attn.q_norm.weight"), vec![ql], s());
        put(format!("{p}.attn.wq_b.weight"), vec![nh * hd, ql], s());
        put(format!("{p}.attn.wkv.weight"), vec![hd, dim], s());
        put(format!("{p}.attn.kv_norm.weight"), vec![hd], s());
        put(format!("{p}.attn.attn_sink"), vec![nh], s());
        put(
            format!("{p}.attn.wo_a.weight"),
            vec![ngrp * olora, dpg],
            s(),
        );
        put(
            format!("{p}.attn.wo_b.weight"),
            vec![dim, ngrp * olora],
            s(),
        );
        // ratio-4 overlapping compressor (wkv/wgate output coff*hd = 2*hd).
        put(
            format!("{p}.attn.compressor.wkv.weight"),
            vec![2 * hd, dim],
            s(),
        );
        put(
            format!("{p}.attn.compressor.wgate.weight"),
            vec![2 * hd, dim],
            s(),
        );
        put(format!("{p}.attn.compressor.ape"), vec![ratio, 2 * hd], s());
        put(format!("{p}.attn.compressor.norm.weight"), vec![hd], s());
        // learned Indexer: own overlapping compressor (index_head_dim) + q/weights.
        put(
            format!("{p}.attn.indexer.compressor.wkv.weight"),
            vec![2 * ihd, dim],
            s(),
        );
        put(
            format!("{p}.attn.indexer.compressor.wgate.weight"),
            vec![2 * ihd, dim],
            s(),
        );
        put(
            format!("{p}.attn.indexer.compressor.ape"),
            vec![ratio, 2 * ihd],
            s(),
        );
        put(
            format!("{p}.attn.indexer.compressor.norm.weight"),
            vec![ihd],
            s(),
        );
        put(
            format!("{p}.attn.indexer.wq_b.weight"),
            vec![in_heads * ihd, ql],
            s(),
        );
        put(
            format!("{p}.attn.indexer.weights_proj.weight"),
            vec![in_heads, dim],
            s(),
        );
        put(format!("{p}.ffn_hc.fn"), vec![mix_hc, hcd], s());
        put(format!("{p}.ffn_hc.scale"), vec![3], s());
        put(format!("{p}.ffn_hc.base"), vec![mix_hc], s());
        put(format!("{p}.ffn_norm.weight"), vec![dim], s());
        put(format!("{p}.ffn.gate_proj.weight"), vec![inter, dim], s());
        put(format!("{p}.ffn.up_proj.weight"), vec![inter, dim], s());
        put(format!("{p}.ffn.down_proj.weight"), vec![dim, inter], s());
    }
    put("model.hc_head.fn".into(), vec![hc, hcd], s());
    put("model.hc_head.scale".into(), vec![1], s());
    put("model.hc_head.base".into(), vec![hc], s());
    put("model.norm.weight".into(), vec![dim], s());
    put("lm_head.weight".into(), vec![vocab, dim], s());

    let seq = 16usize; // ncomp = seq/ratio = 4 > index_topk = 2 → Indexer active
    let ids: Vec<u32> = (0..seq).map(|i| ((i * 7 + 1) % vocab) as u32).collect();
    let mut loader = MemLoader { t };
    let mut packed: HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)> =
        HashMap::new();
    let (graph, params) = build_deepseek_v4_prefill(&spec, &mut loader, seq, &mut packed)?;
    eprintln!("[dsv4] graph built: {} params", params.len());
    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    );
    let mut compiled = Session::new(Device::Cpu).compile_with(graph, &opts);
    for (n, d) in &params {
        compiled.set_param(n, d);
    }
    let ids_f32: Vec<f32> = ids.iter().map(|&x| x as f32).collect();
    let logits = compiled
        .run(&[("input_ids", ids_f32.as_slice())])
        .into_iter()
        .next()
        .unwrap();

    let finite = logits.iter().all(|v| v.is_finite());
    let (mn, mx) = logits
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let last = &logits[(seq - 1) * vocab..seq * vocab];
    let argmax = last
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    println!("── DeepSeek-V4 assembled forward: construction + finite check ──");
    println!(
        "logits = [{seq}, {vocab}]  finite = {finite}  range = [{mn:.3}, {mx:.3}]  argmax = {argmax}"
    );
    if finite && logits.len() == seq * vocab {
        println!(
            "✅ assembled V4 forward (HC-streams + o-LoRA MLA + hc_head) builds + runs finite"
        );
        Ok(())
    } else {
        Err(anyhow::anyhow!("V4 assembly not finite / wrong shape"))
    }
}
