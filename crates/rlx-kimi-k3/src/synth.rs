// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Synthetic Kimi-K3 checkpoints: a small HF-layout directory (`config.json`,
//! `model.safetensors.index.json`, sharded `*.safetensors`) with **the real
//! tensor names, layouts and dtypes** — bf16 matrices in `nn.Linear` `[out, in]`
//! order, f32 conv / `A_log` / `dt_bias` / score-correction bias, and MXFP4
//! routed experts (E2M1 nibbles + E8M0 group-32 scales). Random values, tiny
//! dims.
//!
//! The real checkpoint is ~1.6 TB; this is what lets every checkpoint-reading
//! path — [`crate::loader::CheckpointLoader`], the streamed runner, training,
//! activation/weight dumps, the distributed workers — run end to end in a test
//! or on a laptop. `rlx-kimi-k3 synth --out DIR` writes one.

use anyhow::{Context, Result};
use safetensors::{Dtype, tensor::TensorView};
use std::collections::BTreeMap;
use std::path::Path;

/// Shape of a synthetic checkpoint. [`Default`] is the smallest model that still
/// has every feature: KDA and MLA layers, a dense first layer then LatentMoE,
/// shared experts, and an Attention-Residual block boundary mid-stack.
#[derive(Debug, Clone)]
pub struct SynthSpec {
    pub layers: usize,
    pub hidden: usize,
    pub vocab: usize,
    /// 1-indexed KDA layers (the rest are MLA), as in `linear_attn_config`.
    pub kda_layers: Vec<usize>,
    pub kda_heads: usize,
    pub kda_head_dim: usize,
    pub conv_kernel: usize,
    pub mla_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub num_shared: usize,
    /// Routed-expert FFN width; a multiple of 32 (MXFP4 block).
    pub moe_inter: usize,
    /// Routed-expert latent width; a multiple of 32 (MXFP4 block).
    pub latent: usize,
    pub dense_inter: usize,
    pub first_k_dense: usize,
    pub attn_res_block_size: usize,
    pub routed_scaling: f32,
    /// Tensors per shard file (several shards exercise the index lookup).
    pub layers_per_shard: usize,
    pub seed: u64,
}

impl Default for SynthSpec {
    fn default() -> Self {
        Self {
            layers: 4,
            hidden: 32,
            vocab: 64,
            kda_layers: vec![1, 2, 4],
            kda_heads: 2,
            kda_head_dim: 8,
            conv_kernel: 4,
            mla_heads: 2,
            q_lora_rank: 16,
            kv_lora_rank: 8,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 4,
            v_head_dim: 4,
            num_experts: 8,
            top_k: 2,
            num_shared: 1,
            moe_inter: 32,
            latent: 32,
            dense_inter: 48,
            first_k_dense: 1,
            attn_res_block_size: 2,
            routed_scaling: 2.5,
            layers_per_shard: 2,
            seed: 7,
        }
    }
}

/// splitmix64 → uniform `[-1, 1)`.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

enum Data {
    Bf16(Vec<u8>),
    F32(Vec<u8>),
    U8(Vec<u8>),
}

struct Writer {
    rng: Rng,
    /// shard index → name → (shape, data)
    shards: Vec<BTreeMap<String, (Vec<usize>, Data)>>,
}

impl Writer {
    fn push(&mut self, shard: usize, name: String, shape: &[usize], data: Data) {
        while self.shards.len() <= shard {
            self.shards.push(BTreeMap::new());
        }
        self.shards[shard].insert(name, (shape.to_vec(), data));
    }

    /// bf16 `nn.Linear` weight `[out, in]`, uniform with std ≈ `gain/√in`.
    fn linear(&mut self, shard: usize, name: String, out: usize, inp: usize, gain: f32) {
        let a = gain * (3.0f32 / inp.max(1) as f32).sqrt();
        let v: Vec<u8> = (0..out * inp)
            .flat_map(|_| half::bf16::from_f32(self.rng.uniform() * a).to_le_bytes())
            .collect();
        self.push(shard, name, &[out, inp], Data::Bf16(v));
    }

    /// bf16 vector around `center` (norm weights ≈ 1).
    fn vec_bf16(&mut self, shard: usize, name: String, shape: &[usize], center: f32, spread: f32) {
        let n: usize = shape.iter().product();
        let v: Vec<u8> = (0..n)
            .flat_map(|_| half::bf16::from_f32(center + self.rng.uniform() * spread).to_le_bytes())
            .collect();
        self.push(shard, name, shape, Data::Bf16(v));
    }

    fn vec_f32(&mut self, shard: usize, name: String, shape: &[usize], f: impl Fn(f32) -> f32) {
        let n: usize = shape.iter().product();
        let v: Vec<u8> = (0..n)
            .flat_map(|_| f(self.rng.uniform()).to_le_bytes())
            .collect();
        self.push(shard, name, shape, Data::F32(v));
    }

    /// MXFP4 `[rows, cols]`: random E2M1 codes, E8M0 scales near `2^-4`.
    fn mxfp4(&mut self, shard: usize, base: String, rows: usize, cols: usize) {
        let codes: Vec<u8> = (0..rows * cols / 2)
            .map(|_| (self.rng.next_u64() & 0xFF) as u8)
            .collect();
        let scales: Vec<u8> = (0..rows * cols / 32)
            .map(|_| 123 + (self.rng.next_u64() % 2) as u8)
            .collect();
        self.push(
            shard,
            format!("{base}.weight_packed"),
            &[rows, cols / 2],
            Data::U8(codes),
        );
        self.push(
            shard,
            format!("{base}.weight_scale"),
            &[rows, cols / 32],
            Data::U8(scales),
        );
    }
}

/// The `config.json` a [`SynthSpec`] describes.
pub fn synth_config_json(s: &SynthSpec) -> serde_json::Value {
    let full: Vec<usize> = (1..=s.layers)
        .filter(|l| !s.kda_layers.contains(l))
        .collect();
    serde_json::json!({
        "model_type": "kimi_k3",
        "media_placeholder_token_id": 163605,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "text_config": {
            "vocab_size": s.vocab,
            "hidden_size": s.hidden,
            "intermediate_size": s.dense_inter,
            "num_hidden_layers": s.layers,
            "num_attention_heads": s.mla_heads,
            "num_key_value_heads": s.mla_heads,
            "rms_norm_eps": 1e-5,
            "hidden_act": "situ",
            "q_lora_rank": s.q_lora_rank,
            "kv_lora_rank": s.kv_lora_rank,
            "qk_nope_head_dim": s.qk_nope_head_dim,
            "qk_rope_head_dim": s.qk_rope_head_dim,
            "v_head_dim": s.v_head_dim,
            "mla_use_nope": true,
            "mla_use_output_gate": true,
            "num_experts": s.num_experts,
            "num_experts_per_token": s.top_k,
            "num_shared_experts": s.num_shared,
            "routed_scaling_factor": s.routed_scaling,
            "moe_intermediate_size": s.moe_inter,
            "routed_expert_hidden_size": s.latent,
            "first_k_dense_replace": s.first_k_dense,
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "attn_res_block_size": s.attn_res_block_size,
            "linear_attn_config": {
                "kda_layers": s.kda_layers,
                "full_attn_layers": full,
                "head_dim": s.kda_head_dim,
                "num_heads": s.kda_heads,
                "short_conv_kernel_size": s.conv_kernel,
                "gate_lower_bound": -5.0,
                "use_full_rank_gate": true
            }
        }
    })
}

/// Write a synthetic checkpoint described by `spec` into `dir` (created if
/// missing). Deterministic in `spec.seed`.
pub fn write_synthetic_checkpoint(dir: impl AsRef<Path>, spec: &SynthSpec) -> Result<()> {
    let dir = dir.as_ref();
    anyhow::ensure!(
        spec.latent.is_multiple_of(32) && spec.moe_inter.is_multiple_of(32),
        "MXFP4 experts need latent and moe_inter to be multiples of 32"
    );
    std::fs::create_dir_all(dir).with_context(|| format!("creating {dir:?}"))?;
    let s = spec;
    let h = s.hidden;
    let proj = s.kda_heads * s.kda_head_dim;
    let qk = s.qk_nope_head_dim + s.qk_rope_head_dim;
    let si = s.num_shared * s.moe_inter;
    let mut w = Writer {
        rng: Rng(s.seed),
        shards: Vec::new(),
    };
    let per = s.layers_per_shard.max(1);
    let n_layer_shards = s.layers.div_ceil(per);
    let tail = n_layer_shards; // embed / head shard

    w.linear(
        tail,
        "language_model.model.embed_tokens.weight".into(),
        s.vocab,
        h,
        1.0,
    );
    for i in 0..s.layers {
        let sh = i / per;
        let lp = format!("language_model.model.layers.{i}");
        w.vec_bf16(sh, format!("{lp}.input_layernorm.weight"), &[h], 1.0, 0.1);
        w.vec_bf16(
            sh,
            format!("{lp}.post_attention_layernorm.weight"),
            &[h],
            1.0,
            0.1,
        );
        w.vec_bf16(
            sh,
            format!("{lp}.self_attention_res_norm.weight"),
            &[h],
            1.0,
            0.1,
        );
        w.vec_bf16(
            sh,
            format!("{lp}.self_attention_res_proj.weight"),
            &[1, h],
            0.0,
            0.5,
        );
        w.vec_bf16(sh, format!("{lp}.mlp_res_norm.weight"), &[h], 1.0, 0.1);
        w.vec_bf16(sh, format!("{lp}.mlp_res_proj.weight"), &[1, h], 0.0, 0.5);
        let a = format!("{lp}.self_attn");
        if s.kda_layers.contains(&(i + 1)) {
            for p in ["q_proj", "k_proj", "v_proj", "g_proj"] {
                w.linear(sh, format!("{a}.{p}.weight"), proj, h, 1.0);
            }
            for c in ["q_conv1d", "k_conv1d", "v_conv1d"] {
                w.vec_f32(
                    sh,
                    format!("{a}.{c}.weight"),
                    &[proj, 1, s.conv_kernel],
                    |u| u * 0.5,
                );
            }
            w.linear(sh, format!("{a}.f_a_proj.weight"), s.kda_head_dim, h, 1.0);
            w.linear(
                sh,
                format!("{a}.f_b_proj.weight"),
                proj,
                s.kda_head_dim,
                1.0,
            );
            w.vec_f32(sh, format!("{a}.dt_bias"), &[proj], |u| u * 0.5);
            // A_log ~ ln U(1, 8): decays exp(A_log)·softplus(·) in a sane range.
            w.vec_f32(sh, format!("{a}.A_log"), &[s.kda_head_dim], |u| {
                (1.0 + (u + 1.0) * 3.5).ln()
            });
            w.linear(sh, format!("{a}.b_proj.weight"), s.kda_heads, h, 1.0);
            w.vec_bf16(
                sh,
                format!("{a}.o_norm.weight"),
                &[s.kda_head_dim],
                1.0,
                0.1,
            );
            w.linear(sh, format!("{a}.o_proj.weight"), h, proj, 0.5);
        } else {
            w.linear(sh, format!("{a}.q_a_proj.weight"), s.q_lora_rank, h, 1.0);
            w.vec_bf16(
                sh,
                format!("{a}.q_a_layernorm.weight"),
                &[s.q_lora_rank],
                1.0,
                0.1,
            );
            w.linear(
                sh,
                format!("{a}.q_b_proj.weight"),
                s.mla_heads * qk,
                s.q_lora_rank,
                1.0,
            );
            w.linear(
                sh,
                format!("{a}.kv_a_proj_with_mqa.weight"),
                s.kv_lora_rank + s.qk_rope_head_dim,
                h,
                1.0,
            );
            w.vec_bf16(
                sh,
                format!("{a}.kv_a_layernorm.weight"),
                &[s.kv_lora_rank],
                1.0,
                0.1,
            );
            w.linear(
                sh,
                format!("{a}.kv_b_proj.weight"),
                s.mla_heads * (s.qk_nope_head_dim + s.v_head_dim),
                s.kv_lora_rank,
                1.0,
            );
            w.linear(
                sh,
                format!("{a}.g_proj.weight"),
                s.mla_heads * s.v_head_dim,
                h,
                1.0,
            );
            w.linear(
                sh,
                format!("{a}.o_proj.weight"),
                h,
                s.mla_heads * s.v_head_dim,
                0.5,
            );
        }
        if i < s.first_k_dense || s.num_experts == 0 {
            let m = format!("{lp}.mlp");
            w.linear(sh, format!("{m}.gate_proj.weight"), s.dense_inter, h, 1.0);
            w.linear(sh, format!("{m}.up_proj.weight"), s.dense_inter, h, 1.0);
            w.linear(sh, format!("{m}.down_proj.weight"), h, s.dense_inter, 0.5);
        } else {
            let m = format!("{lp}.block_sparse_moe");
            w.linear(sh, format!("{m}.gate.weight"), s.num_experts, h, 1.0);
            w.vec_f32(
                sh,
                format!("{m}.gate.e_score_correction_bias"),
                &[s.num_experts],
                |u| u * 0.05,
            );
            w.linear(
                sh,
                format!("{m}.routed_expert_down_proj.weight"),
                s.latent,
                h,
                1.0,
            );
            w.linear(
                sh,
                format!("{m}.routed_expert_up_proj.weight"),
                h,
                s.latent,
                0.5,
            );
            w.vec_bf16(
                sh,
                format!("{m}.routed_expert_norm.weight"),
                &[s.latent],
                1.0,
                0.1,
            );
            w.linear(
                sh,
                format!("{m}.shared_experts.gate_proj.weight"),
                si,
                h,
                1.0,
            );
            w.linear(sh, format!("{m}.shared_experts.up_proj.weight"), si, h, 1.0);
            w.linear(
                sh,
                format!("{m}.shared_experts.down_proj.weight"),
                h,
                si,
                0.5,
            );
            for e in 0..s.num_experts {
                let p = format!("{m}.experts.{e}");
                w.mxfp4(sh, format!("{p}.w1"), s.moe_inter, s.latent);
                w.mxfp4(sh, format!("{p}.w3"), s.moe_inter, s.latent);
                w.mxfp4(sh, format!("{p}.w2"), s.latent, s.moe_inter);
            }
        }
    }
    w.vec_bf16(
        tail,
        "language_model.model.norm.weight".into(),
        &[h],
        1.0,
        0.1,
    );
    w.vec_bf16(
        tail,
        "language_model.model.output_attn_res_norm.weight".into(),
        &[h],
        1.0,
        0.1,
    );
    w.vec_bf16(
        tail,
        "language_model.model.output_attn_res_proj.weight".into(),
        &[1, h],
        0.0,
        0.5,
    );
    w.linear(
        tail,
        "language_model.lm_head.weight".into(),
        s.vocab,
        h,
        1.0,
    );

    let n = w.shards.len();
    let mut weight_map = serde_json::Map::new();
    let mut total = 0usize;
    for (si, shard) in w.shards.iter().enumerate() {
        let file = format!("model-{:05}-of-{:05}.safetensors", si + 1, n);
        let views: Vec<(String, TensorView<'_>)> = shard
            .iter()
            .map(|(name, (shape, data))| {
                let (dt, bytes) = match data {
                    Data::Bf16(b) => (Dtype::BF16, b),
                    Data::F32(b) => (Dtype::F32, b),
                    Data::U8(b) => (Dtype::U8, b),
                };
                total += bytes.len();
                weight_map.insert(name.clone(), serde_json::Value::String(file.clone()));
                TensorView::new(dt, shape.clone(), bytes)
                    .map(|v| (name.clone(), v))
                    .map_err(|e| anyhow::anyhow!("{name}: {e:?}"))
            })
            .collect::<Result<_>>()?;
        safetensors::serialize_to_file(views, None, &dir.join(&file))
            .map_err(|e| anyhow::anyhow!("writing {file}: {e:?}"))?;
    }
    let index = serde_json::json!({
        "metadata": { "total_size": total },
        "weight_map": weight_map,
    });
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&index)?,
    )?;
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec_pretty(&synth_config_json(spec))?,
    )?;
    Ok(())
}
