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

//! A tiny Flint with deterministic synthetic weights, written to disk as a
//! real HF checkpoint (`config.json` + safetensors).
//!
//! Every test goes through this one fixture, and through the same
//! [`crate::checkpoint::Checkpoint`] loader the published model uses, so the
//! tensor contract lives in exactly one place: [`tensor_manifest`], which
//! `tests/tensor_manifest.rs` also checks against the real checkpoint's index.
//!
//! The tiny config keeps every architectural feature of the real one: a dense
//! prefix, full NoPE layers among RoPE'd sliding ones, a sliding window short
//! enough for a test prompt to cross it, GQA, top-k over several experts and a
//! shared expert of a different width.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::Path;

use crate::config::FlintConfig;
use crate::moe::{EXPERT_PROJS, expert_key};

/// `config.json` for the tiny model.
pub fn tiny_config_json() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["FlintForCausalLM"],
        "model_type": "flint",
        "vocab_size": 96,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "sliding_window": 4,
        "use_sliding_window": true,
        "layer_types": ["full_attention", "full_attention", "sliding_attention",
                        "sliding_attention", "full_attention", "sliding_attention"],
        "no_rope_layer_indices": [0, 1, 4],
        "mlp_only_layers": [0, 1],
        "num_experts": 8,
        "num_experts_per_tok": 3,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 24,
        "norm_topk_prob": true,
        "router_score_function": "sigmoid",
        "softmax_type": "learnable",
        "decoder_sparse_step": 1,
        "tie_word_embeddings": false,
        "bos_token_id": 90,
        "eos_token_id": [91, 92],
        "pad_token_id": 91,
        "max_position_embeddings": 4096
    })
}

pub fn tiny_config() -> FlintConfig {
    FlintConfig::from_value(&tiny_config_json()).expect("the tiny config is valid")
}

/// Every tensor the model reads, with its HF shape, in a stable order.
pub fn tensor_manifest(cfg: &FlintConfig) -> BTreeMap<String, Vec<usize>> {
    let (h, v) = (cfg.hidden_size, cfg.vocab_size);
    let mut m = BTreeMap::new();
    m.insert("model.embed_tokens.weight".into(), vec![v, h]);
    m.insert("model.norm.weight".into(), vec![h]);
    if !cfg.tie_word_embeddings {
        m.insert("lm_head.weight".into(), vec![v, h]);
    }
    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{l}");
        m.insert(format!("{p}.input_layernorm.weight"), vec![h]);
        m.insert(format!("{p}.post_attention_layernorm.weight"), vec![h]);
        let a = format!("{p}.self_attn");
        m.insert(format!("{a}.q_proj.weight"), vec![cfg.q_dim(), h]);
        m.insert(format!("{a}.k_proj.weight"), vec![cfg.kv_dim(), h]);
        m.insert(format!("{a}.v_proj.weight"), vec![cfg.kv_dim(), h]);
        m.insert(format!("{a}.o_proj.weight"), vec![h, cfg.q_dim()]);
        m.insert(format!("{a}.q_norm.weight"), vec![cfg.head_dim]);
        m.insert(format!("{a}.k_norm.weight"), vec![cfg.head_dim]);
        m.insert(format!("{a}.softmax_offset"), vec![cfg.num_attention_heads]);
        let mlp = |m: &mut BTreeMap<String, Vec<usize>>, prefix: String, inter: usize| {
            m.insert(format!("{prefix}.gate_proj.weight"), vec![inter, h]);
            m.insert(format!("{prefix}.up_proj.weight"), vec![inter, h]);
            m.insert(format!("{prefix}.down_proj.weight"), vec![h, inter]);
        };
        if cfg.layer(l).dense {
            mlp(&mut m, format!("{p}.mlp"), cfg.intermediate_size);
        } else {
            m.insert(format!("{p}.mlp.gate.weight"), vec![cfg.num_experts, h]);
            if cfg.shared_expert_intermediate_size > 0 {
                mlp(
                    &mut m,
                    format!("{p}.mlp.shared_expert"),
                    cfg.shared_expert_intermediate_size,
                );
            }
            let i = cfg.moe_intermediate_size;
            for e in 0..cfg.num_experts {
                for proj in EXPERT_PROJS {
                    let shape = if proj == "down_proj" {
                        vec![h, i]
                    } else {
                        vec![i, h]
                    };
                    m.insert(expert_key(l, e, proj), shape);
                }
            }
        }
    }
    m
}

/// SplitMix64 — deterministic, dependency-free.
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// Synthetic values for one tensor: norm gains near 1, sinks around 0,
/// matrices scaled by `1/√fan_in` so activations stay O(1) through the stack.
fn values(name: &str, shape: &[usize], rng: &mut Rng) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let fan_in = *shape.last().unwrap_or(&1) as f32;
    let (base, scale) = if name.ends_with("norm.weight") {
        (1.0, 0.2)
    } else if name.ends_with("softmax_offset") {
        (0.0, 2.0)
    } else if name.contains("embed_tokens") {
        (0.0, 1.0)
    } else {
        (0.0, 1.5 / fan_in.sqrt())
    };
    (0..n).map(|_| base + scale * rng.next_f32()).collect()
}

/// Storage dtype for [`write_checkpoint`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynthDtype {
    F32,
    BF16,
}

/// Write `config` and synthetic weights for it into `dir`, split across
/// `shards` safetensors files with a `model.safetensors.index.json`, the way
/// the published checkpoint is laid out.
pub fn write_checkpoint(
    dir: &Path,
    config: &serde_json::Value,
    dtype: SynthDtype,
    shards: usize,
    seed: u64,
) -> Result<()> {
    let cfg = FlintConfig::from_value(config)?;
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(config)?)?;
    let manifest = tensor_manifest(&cfg);
    let mut rng = Rng(seed);
    let shards = shards.max(1);
    let mut groups: Vec<Vec<(String, Vec<usize>, Vec<u8>)>> = vec![Vec::new(); shards];
    for (i, (name, shape)) in manifest.iter().enumerate() {
        let v = values(name, shape, &mut rng);
        let bytes: Vec<u8> = match dtype {
            SynthDtype::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            SynthDtype::BF16 => v
                .iter()
                .flat_map(|x| half::bf16::from_f32(*x).to_bits().to_le_bytes())
                .collect(),
        };
        groups[i % shards].push((name.clone(), shape.clone(), bytes));
    }
    let st_dtype = match dtype {
        SynthDtype::F32 => safetensors::Dtype::F32,
        SynthDtype::BF16 => safetensors::Dtype::BF16,
    };
    let mut weight_map = serde_json::Map::new();
    for (si, group) in groups.iter().enumerate() {
        let file = format!("model-{:05}-of-{:05}.safetensors", si + 1, shards);
        let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = group
            .iter()
            .map(|(n, s, b)| {
                safetensors::tensor::TensorView::new(st_dtype, s.clone(), b)
                    .map(|v| (n.clone(), v))
                    .context("synthetic tensor view")
            })
            .collect::<Result<_>>()?;
        safetensors::serialize_to_file(views, None, &dir.join(&file))
            .map_err(|e| anyhow::anyhow!("writing {file}: {e}"))?;
        for (n, _, _) in group {
            weight_map.insert(n.clone(), serde_json::Value::String(file.clone()));
        }
    }
    let index = serde_json::json!({ "metadata": {}, "weight_map": weight_map });
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&index)?,
    )?;
    Ok(())
}
