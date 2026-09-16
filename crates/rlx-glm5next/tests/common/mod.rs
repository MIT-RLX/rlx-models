// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! The tiny synthetic `glm5next` checkpoint shared by the integration tests.
//!
//! GLM-5.3-Flash is 320 B parameters and its smallest published quantization is
//! 93 GB, so nothing here can run the real model. What this fixture preserves is
//! the *architecture*: the shipped layer schedule (`[KDA, KDA, KDA, MLA]`,
//! dense-then-MoE), mHC at every site, and the DSA indexer — only the widths
//! shrink. Every tensor is synthesized under its GGUF name in the orientation
//! the GGUF loader hands over (torch `[out, in]`).
#![allow(dead_code)]

use rlx_core::weight_map::WeightMap;
use rlx_glm5next::Glm5NextConfig;
use rlx_glm5next::config::{AttnKind, IndexerKind};
use rlx_runtime::Device;
use std::collections::HashMap;

pub const HIDDEN: usize = 32;
pub const HEADS: usize = 2;
pub const NOPE: usize = 16;
pub const KV_LORA: usize = 8;
pub const Q_LORA: usize = 12;
pub const KDA_HEADS: usize = 2;
pub const KDA_HD: usize = 16;
pub const CONV_K: usize = 4;
pub const IDX_HEADS: usize = 2;
pub const IDX_HD: usize = 8;
pub const KPOOL: usize = 4;
pub const EXPERTS: usize = 6;
pub const TOPK: usize = 2;
pub const MOE_INTER: usize = 12;
pub const INTER: usize = 24;
pub const VOCAB: usize = 16;
pub const HC: usize = 4;
pub const LAYERS: usize = 4;

pub fn dev() -> Device {
    std::env::var("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

pub fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.5
        })
        .collect()
}

pub fn cfg(index_topk: usize) -> Glm5NextConfig {
    Glm5NextConfig {
        vocab_size: VOCAB,
        hidden_size: HIDDEN,
        intermediate_size: INTER,
        num_hidden_layers: LAYERS,
        num_attention_heads: HEADS,
        rms_norm_eps: 1e-5,
        max_position_embeddings: 1024,
        tie_word_embeddings: false,
        // The shipped schedule: three KDA layers then one MLA layer.
        layer_types: vec![
            AttnKind::Kda,
            AttnKind::Kda,
            AttnKind::Kda,
            AttnKind::MlaDsa,
        ],
        indexer_types: vec![IndexerKind::Full; LAYERS],
        first_k_dense_replace: 3,
        q_lora_rank: Q_LORA,
        kv_lora_rank: KV_LORA,
        qk_nope_head_dim: NOPE,
        qk_rope_head_dim: 0,
        v_head_dim: NOPE,
        index_n_heads: IDX_HEADS,
        index_head_dim: IDX_HD,
        index_topk,
        index_kpool: KPOOL,
        index_kpool_always_select_tail: true,
        linear_num_heads: KDA_HEADS,
        linear_head_dim: KDA_HD,
        linear_conv_kernel_dim: CONV_K,
        linear_lower_bound: Some(-5.0),
        hc_mult: HC,
        hc_sinkhorn_iters: 20,
        hc_eps: 1e-6,
        n_routed_experts: EXPERTS,
        num_experts_per_tok: TOPK,
        n_shared_experts: 1,
        moe_intermediate_size: MOE_INTER,
        n_group: 1,
        topk_group: 1,
        routed_scaling_factor: 2.5,
        norm_topk_prob: true,
        swiglu_limit: 10.0,
        num_nextn_predict_layers: 1,
        with_mtp: false,
    }
}

/// Synthesize every tensor the flow asks for, under its GGUF name and in the
/// orientation the GGUF loader hands over (torch `[out, in]`).
pub fn weights(c: &Glm5NextConfig) -> WeightMap {
    WeightMap::from_tensors(tensor_map(c))
}

/// The same checkpoint as a plain name -> (data, shape) map, which is what a
/// pipeline stage shards.
pub fn tensor_map(c: &Glm5NextConfig) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 1u64;
    let mut put = |t: &mut HashMap<_, _>, k: String, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        seed += 7;
        t.insert(k, (fill(n, seed), shape));
    };

    put(&mut t, "token_embd.weight".into(), vec![VOCAB, HIDDEN]);
    put(&mut t, "output.weight".into(), vec![VOCAB, HIDDEN]);
    t.insert(
        "output_norm.weight".into(),
        (vec![1.0; HIDDEN], vec![HIDDEN]),
    );

    let kda_proj = KDA_HEADS * KDA_HD;
    let qk = HEADS * NOPE;
    let hc_mix = (2 + HC) * HC;

    for i in 0..c.num_hidden_layers {
        let b = format!("blk.{i}");
        for n in ["attn_norm", "ffn_norm"] {
            t.insert(format!("{b}.{n}.weight"), (vec![1.0; HIDDEN], vec![HIDDEN]));
        }
        // mHC, at both sites.
        for site in ["hc_attn", "hc_ffn"] {
            put(
                &mut t,
                format!("{b}.{site}_fn.weight"),
                vec![hc_mix, HC * HIDDEN],
            );
            put(&mut t, format!("{b}.{site}_base.weight"), vec![hc_mix]);
            t.insert(
                format!("{b}.{site}_scale.weight"),
                (vec![1.0, 1.0, 1.0], vec![3]),
            );
        }

        match c.attn_kind(i) {
            AttnKind::Kda => {
                for p in ["attn_q", "attn_k", "attn_v"] {
                    put(&mut t, format!("{b}.{p}.weight"), vec![kda_proj, HIDDEN]);
                }
                put(
                    &mut t,
                    format!("{b}.attn_output.weight"),
                    vec![HIDDEN, kda_proj],
                );
                for cv in ["ssm_conv1d_q", "ssm_conv1d_k", "ssm_conv1d_v"] {
                    put(
                        &mut t,
                        format!("{b}.{cv}.weight"),
                        vec![kda_proj, 1, CONV_K],
                    );
                }
                put(&mut t, format!("{b}.ssm_a"), vec![KDA_HEADS]);
                put(&mut t, format!("{b}.ssm_dt.bias"), vec![kda_proj]);
                put(
                    &mut t,
                    format!("{b}.ssm_beta.weight"),
                    vec![KDA_HEADS, HIDDEN],
                );
                put(&mut t, format!("{b}.ssm_f_a.weight"), vec![KDA_HD, HIDDEN]);
                put(
                    &mut t,
                    format!("{b}.ssm_f_b.weight"),
                    vec![kda_proj, KDA_HD],
                );
                put(&mut t, format!("{b}.ssm_g_a.weight"), vec![KDA_HD, HIDDEN]);
                put(
                    &mut t,
                    format!("{b}.ssm_g_b.weight"),
                    vec![kda_proj, KDA_HD],
                );
                t.insert(
                    format!("{b}.ssm_norm.weight"),
                    (vec![1.0; KDA_HD], vec![KDA_HD]),
                );
            }
            AttnKind::MlaDsa => {
                put(&mut t, format!("{b}.attn_q_a.weight"), vec![Q_LORA, HIDDEN]);
                t.insert(
                    format!("{b}.attn_q_a_norm.weight"),
                    (vec![1.0; Q_LORA], vec![Q_LORA]),
                );
                put(&mut t, format!("{b}.attn_q_b.weight"), vec![qk, Q_LORA]);
                put(
                    &mut t,
                    format!("{b}.attn_kv_a_mqa.weight"),
                    vec![KV_LORA, HIDDEN],
                );
                t.insert(
                    format!("{b}.attn_kv_a_norm.weight"),
                    (vec![1.0; KV_LORA], vec![KV_LORA]),
                );
                // [head, kv_lora, nope] and [head, v_head, kv_lora] — the two
                // orientations GGUF stores these in are NOT the same.
                put(
                    &mut t,
                    format!("{b}.attn_k_b.weight"),
                    vec![HEADS, KV_LORA, NOPE],
                );
                put(
                    &mut t,
                    format!("{b}.attn_v_b.weight"),
                    vec![HEADS, NOPE, KV_LORA],
                );
                put(&mut t, format!("{b}.attn_output.weight"), vec![HIDDEN, qk]);

                // DSA indexer.
                put(
                    &mut t,
                    format!("{b}.indexer.attn_q_b.weight"),
                    vec![IDX_HEADS * IDX_HD, Q_LORA],
                );
                put(
                    &mut t,
                    format!("{b}.indexer.attn_k.weight"),
                    vec![IDX_HD, HIDDEN],
                );
                t.insert(
                    format!("{b}.indexer.k_norm.weight"),
                    (vec![1.0; IDX_HD], vec![IDX_HD]),
                );
                t.insert(
                    format!("{b}.indexer.k_norm.bias"),
                    (vec![0.0; IDX_HD], vec![IDX_HD]),
                );
                put(
                    &mut t,
                    format!("{b}.indexer.proj.weight"),
                    vec![IDX_HEADS, HIDDEN],
                );
                put(
                    &mut t,
                    format!("{b}.indexer_compressor_gate.weight"),
                    vec![IDX_HD, HIDDEN],
                );
                put(
                    &mut t,
                    format!("{b}.indexer_compressor_ape.weight"),
                    vec![KPOOL, IDX_HD],
                );
            }
        }

        if c.is_moe_layer(i) {
            put(
                &mut t,
                format!("{b}.ffn_gate_inp.weight"),
                vec![EXPERTS, HIDDEN],
            );
            put(&mut t, format!("{b}.exp_probs_b.bias"), vec![EXPERTS]);
            put(
                &mut t,
                format!("{b}.ffn_gate_exps.weight"),
                vec![EXPERTS, MOE_INTER, HIDDEN],
            );
            put(
                &mut t,
                format!("{b}.ffn_up_exps.weight"),
                vec![EXPERTS, MOE_INTER, HIDDEN],
            );
            put(
                &mut t,
                format!("{b}.ffn_down_exps.weight"),
                vec![EXPERTS, HIDDEN, MOE_INTER],
            );
            put(
                &mut t,
                format!("{b}.ffn_gate_shexp.weight"),
                vec![MOE_INTER, HIDDEN],
            );
            put(
                &mut t,
                format!("{b}.ffn_up_shexp.weight"),
                vec![MOE_INTER, HIDDEN],
            );
            put(
                &mut t,
                format!("{b}.ffn_down_shexp.weight"),
                vec![HIDDEN, MOE_INTER],
            );
        } else {
            put(&mut t, format!("{b}.ffn_gate.weight"), vec![INTER, HIDDEN]);
            put(&mut t, format!("{b}.ffn_up.weight"), vec![INTER, HIDDEN]);
            put(&mut t, format!("{b}.ffn_down.weight"), vec![HIDDEN, INTER]);
        }
    }
    t
}
