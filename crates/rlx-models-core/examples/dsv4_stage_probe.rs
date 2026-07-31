// RLX — versatile ML compiler + runtime. GPLv3.
//! Validates the DeepSeek-V4 **layer-range stage builder**
//! ([`build_deepseek_v4_stage`]) — the piece that lets each machine build only
//! its slice of the model from its own shards. Builds the tiny synthetic V4 two
//! ways: (a) the whole forward; (b) as two pipeline STAGES (layers 0..1 with
//! embed, then layers 1..2 with the LM head) chained through the hidden-state
//! boundary. Asserts the staged run reproduces the whole-forward logits.
//!
//!   cargo run --release -p rlx-models-core --example dsv4_stage_probe

use anyhow::Result;
use rlx_ir::quant::QuantScheme;
use rlx_models_core::standard_decoder::{
    DeepseekV4Spec, build_deepseek_v4_prefill, build_deepseek_v4_stage,
};
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
    DeepseekV4Spec {
        vocab_size: 16,
        dim: 8,
        n_layers: 2,
        hc_mult: 2,
        n_heads: 2,
        head_dim: 4,
        rope_head_dim: 2,
        q_lora_rank: 6,
        n_groups: 2,
        o_lora_rank: 3,
        compress_ratios: vec![4; 2],
        index_head_dim: 4,
        index_n_heads: 2,
        index_topk: 2,
        window_size: 64,
        first_k_dense_replace: 2,
        n_hash_layers: 0,
        moe_intermediate_size: 8,
        n_routed_experts: 4,
        n_activated_experts: 2,
        n_shared_experts: 1,
        intermediate_size: 10,
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
    }
}
fn tmap(s: &DeepseekV4Spec) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
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
    let mut put = |k: String, sh: Vec<usize>| {
        sd += 1.0;
        let n: usize = sh.iter().product();
        t.insert(k, ((0..n).map(|i| rnd(sd, i) + 0.05).collect(), sh));
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

fn run(
    graph: rlx_ir::Graph,
    params: &HashMap<String, Vec<f32>>,
    input: (&str, &[f32]),
) -> Vec<f32> {
    let opts = CompileOptions::default();
    let mut c = Session::new(Device::Cpu).compile_with(graph, &opts);
    for (n, d) in params {
        c.set_param(n, d);
    }
    c.run(&[input]).into_iter().next().unwrap()
}

fn main() -> Result<()> {
    let spec = spec();
    let ids: Vec<u32> = (0..16u32)
        .map(|i| (i * 7 + 1) % spec.vocab_size as u32)
        .collect();
    let ids_f: Vec<f32> = ids.iter().map(|&x| x as f32).collect();

    // Whole forward → reference logits.
    let mut l0 = MemLoader { t: tmap(&spec) };
    let mut pk = HashMap::<String, (Vec<u8>, QuantScheme, Vec<usize>)>::new();
    let (g_ref, p_ref) = build_deepseek_v4_prefill(&spec, &mut l0, ids.len(), &mut pk)?;
    let reference = run(g_ref, &p_ref, ("input_ids", &ids_f));

    // Two stages: [0,1) with embed, then [1,2) with head. Boundary = hidden state.
    let mut la = MemLoader { t: tmap(&spec) };
    let (g0, p0) = build_deepseek_v4_stage(&spec, &mut la, ids.len(), 0..1, true, false, &mut pk)?;
    let hidden = run(g0, &p0, ("input_ids", &ids_f)); // [rows, hc, d] flattened

    let mut lb = MemLoader { t: tmap(&spec) };
    let (g1, p1) = build_deepseek_v4_stage(&spec, &mut lb, ids.len(), 1..2, false, true, &mut pk)?;
    let staged = run(g1, &p1, ("hidden_in", &hidden));

    let cos = {
        let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (a, b) in staged.iter().zip(&reference) {
            d += *a as f64 * *b as f64;
            na += (*a as f64).powi(2);
            nb += (*b as f64).powi(2);
        }
        d / (na.sqrt() * nb.sqrt()).max(1e-12)
    };
    let err = staged
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("── DeepSeek-V4 2-stage split (layers 0|1) via hidden boundary vs whole forward ──");
    println!(
        "boundary hidden = {} floats; logits cos {cos:.8}  max|err| {err:.3e}",
        hidden.len()
    );
    if cos > 0.999999 && err < 1e-4 {
        println!("✅ staged build reproduces the whole forward — ready for per-node stages");
        Ok(())
    } else {
        Err(anyhow::anyhow!("stage split mismatch: cos {cos}"))
    }
}
