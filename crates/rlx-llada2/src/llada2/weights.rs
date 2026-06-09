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

// RLX — LLaDA2 MoE weight layout (HuggingFace / TIDE naming).

use crate::config::LLaDA2MoeConfig;
use anyhow::{Result, anyhow};
use rlx_core::weight_loader::WeightLoader;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct DenseFfnWeights {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct MoeLayerWeights {
    pub router: Vec<f32>,
    pub expert_bias: Vec<f32>,
    pub gate_exps: Vec<f32>,
    pub up_exps: Vec<f32>,
    pub down_exps: Vec<f32>,
    pub shared_gate: Option<Vec<f32>>,
    pub shared_up: Option<Vec<f32>>,
    pub shared_down: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub input_norm: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    pub qkv: Vec<f32>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub o_proj: Vec<f32>,
    pub ffn: LayerFfn,
}

#[derive(Debug, Clone)]
pub enum LayerFfn {
    Dense(DenseFfnWeights),
    Moe(MoeLayerWeights),
}

#[derive(Debug, Clone)]
pub struct LLaDA2Weights {
    pub embed: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub lm_head: Vec<f32>,
    pub layers: Vec<LayerWeights>,
}

/// HF tensor names required to build a graph with `cfg.num_hidden_layers` blocks.
pub fn tensor_keys_for_config(cfg: &LLaDA2MoeConfig) -> HashSet<String> {
    let mut keys = HashSet::new();
    keys.insert("model.word_embeddings.weight".into());
    keys.insert("model.embed_tokens.weight".into());
    keys.insert("model.norm.weight".into());
    keys.insert("lm_head.weight".into());
    for il in 0..cfg.num_hidden_layers {
        keys.extend(layer_tensor_keys(cfg, il));
    }
    keys
}

fn layer_tensor_keys(cfg: &LLaDA2MoeConfig, il: usize) -> HashSet<String> {
    let mut keys = HashSet::new();
    let p = |tail: &str| format!("model.layers.{il}.{tail}");
    for stem in ["attention", "self_attn"] {
        keys.insert(p(&format!("{stem}.query_key_value.weight")));
        keys.insert(p(&format!("{stem}.dense.weight")));
        if cfg.use_qk_norm {
            keys.insert(p(&format!("{stem}.query_layernorm.weight")));
            keys.insert(p(&format!("{stem}.key_layernorm.weight")));
        }
    }
    keys.insert(p("input_layernorm.weight"));
    keys.insert(p("post_attention_layernorm.weight"));
    if cfg.is_moe_layer(il) {
        keys.insert(format!("model.layers.{il}.mlp.gate.weight"));
        keys.insert(format!("model.layers.{il}.mlp.gate.expert_bias"));
        for ei in 0..cfg.num_experts {
            let base = format!("model.layers.{il}.mlp.experts.{ei}");
            keys.insert(format!("{base}.gate_proj.weight"));
            keys.insert(format!("{base}.up_proj.weight"));
            keys.insert(format!("{base}.down_proj.weight"));
        }
        if cfg.num_shared_experts.unwrap_or(0) > 0 {
            keys.insert(format!(
                "model.layers.{il}.mlp.shared_experts.gate_proj.weight"
            ));
            keys.insert(format!(
                "model.layers.{il}.mlp.shared_experts.up_proj.weight"
            ));
            keys.insert(format!(
                "model.layers.{il}.mlp.shared_experts.down_proj.weight"
            ));
        }
    } else {
        keys.insert(p("mlp.gate_proj.weight"));
        keys.insert(p("mlp.up_proj.weight"));
        keys.insert(p("mlp.down_proj.weight"));
    }
    keys
}

fn take_any(loader: &mut dyn WeightLoader, keys: &[&str]) -> Result<(Vec<f32>, Vec<usize>)> {
    for key in keys {
        if let Ok(v) = loader.take(key) {
            return Ok(v);
        }
    }
    Err(anyhow!("weight not found: {}", keys.join(" | ")))
}

fn take_transposed_any(
    loader: &mut dyn WeightLoader,
    keys: &[&str],
) -> Result<(Vec<f32>, Vec<usize>)> {
    for key in keys {
        if let Ok(v) = loader.take_transposed(key) {
            return Ok(v);
        }
    }
    Err(anyhow!("weight not found: {}", keys.join(" | ")))
}

impl LLaDA2Weights {
    pub fn load(cfg: &LLaDA2MoeConfig, loader: &mut dyn WeightLoader) -> Result<Self> {
        let h = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let embed = take_any(
            loader,
            &["model.word_embeddings.weight", "model.embed_tokens.weight"],
        )?
        .0;
        let final_norm = loader.take("model.norm.weight")?.0;
        let lm_head = take_any(
            loader,
            &[
                "lm_head.weight",
                "model.word_embeddings.weight",
                "model.embed_tokens.weight",
            ],
        )?
        .0;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for il in 0..cfg.num_hidden_layers {
            layers.push(load_layer(cfg, loader, il)?);
        }

        if embed.len() != vocab * h {
            return Err(anyhow!(
                "embed len {} != vocab*hidden ({vocab}*{h})",
                embed.len()
            ));
        }
        Ok(Self {
            embed,
            final_norm,
            lm_head,
            layers,
        })
    }
}

fn load_layer(
    cfg: &LLaDA2MoeConfig,
    loader: &mut dyn WeightLoader,
    il: usize,
) -> Result<LayerWeights> {
    let p = |tail: &str| format!("model.layers.{il}.{tail}");
    let h = cfg.hidden_size;
    let qkv_out = (cfg.num_attention_heads + 2 * cfg.num_kv_heads()) * cfg.head_dim();

    let qkv = take_transposed_any(
        loader,
        &[
            &p("attention.query_key_value.weight"),
            &p("self_attn.query_key_value.weight"),
        ],
    )?
    .0;
    let o_proj = take_transposed_any(
        loader,
        &[&p("attention.dense.weight"), &p("self_attn.dense.weight")],
    )?
    .0;

    let q_norm = if cfg.use_qk_norm {
        Some(
            take_any(
                loader,
                &[
                    &p("attention.query_layernorm.weight"),
                    &p("self_attn.query_layernorm.weight"),
                ],
            )?
            .0,
        )
    } else {
        None
    };
    let k_norm = if cfg.use_qk_norm {
        Some(
            take_any(
                loader,
                &[
                    &p("attention.key_layernorm.weight"),
                    &p("self_attn.key_layernorm.weight"),
                ],
            )?
            .0,
        )
    } else {
        None
    };

    if qkv.len() != h * qkv_out {
        return Err(anyhow!("layer {il} qkv size mismatch"));
    }

    let ffn = if cfg.is_moe_layer(il) {
        let e = cfg.num_experts;
        let ff = cfg.expert_ffn_dim();
        let router =
            take_transposed_any(loader, &[&format!("model.layers.{il}.mlp.gate.weight")])?.0;
        let expert_bias = loader
            .take(&format!("model.layers.{il}.mlp.gate.expert_bias"))
            .map(|(d, _)| d)
            .unwrap_or_else(|_| vec![0f32; e]);
        let mut gate_exps = vec![0f32; e * h * ff];
        let mut up_exps = vec![0f32; e * h * ff];
        let mut down_exps = vec![0f32; e * ff * h];
        for ei in 0..e {
            let base = format!("model.layers.{il}.mlp.experts.{ei}");
            let g = take_transposed_any(loader, &[&format!("{base}.gate_proj.weight")])?.0;
            let u = take_transposed_any(loader, &[&format!("{base}.up_proj.weight")])?.0;
            let d = take_transposed_any(loader, &[&format!("{base}.down_proj.weight")])?.0;
            let stride_in = h * ff;
            let stride_out = ff * h;
            gate_exps[ei * stride_in..(ei + 1) * stride_in].copy_from_slice(&g);
            up_exps[ei * stride_in..(ei + 1) * stride_in].copy_from_slice(&u);
            down_exps[ei * stride_out..(ei + 1) * stride_out].copy_from_slice(&d);
        }
        let (shared_gate, shared_up, shared_down) = if cfg.num_shared_experts.unwrap_or(0) > 0 {
            let sg = take_transposed_any(
                loader,
                &[&format!(
                    "model.layers.{il}.mlp.shared_experts.gate_proj.weight"
                )],
            )?
            .0;
            let su = take_transposed_any(
                loader,
                &[&format!(
                    "model.layers.{il}.mlp.shared_experts.up_proj.weight"
                )],
            )?
            .0;
            let sd = take_transposed_any(
                loader,
                &[&format!(
                    "model.layers.{il}.mlp.shared_experts.down_proj.weight"
                )],
            )?
            .0;
            (Some(sg), Some(su), Some(sd))
        } else {
            (None, None, None)
        };
        LayerFfn::Moe(MoeLayerWeights {
            router,
            expert_bias,
            gate_exps,
            up_exps,
            down_exps,
            shared_gate,
            shared_up,
            shared_down,
        })
    } else {
        LayerFfn::Dense(DenseFfnWeights {
            gate: take_transposed_any(loader, &[&p("mlp.gate_proj.weight")])?.0,
            up: take_transposed_any(loader, &[&p("mlp.up_proj.weight")])?.0,
            down: take_transposed_any(loader, &[&p("mlp.down_proj.weight")])?.0,
        })
    };

    Ok(LayerWeights {
        input_norm: loader.take(&p("input_layernorm.weight"))?.0,
        post_attn_norm: loader.take(&p("post_attention_layernorm.weight"))?.0,
        qkv,
        q_norm,
        k_norm,
        o_proj,
        ffn,
    })
}

/// Register all tensors into `params` for graph compile.
pub fn register_params(
    cfg: &LLaDA2MoeConfig,
    weights: &LLaDA2Weights,
    params: &mut HashMap<String, Vec<f32>>,
) {
    params.insert("model.embed_tokens.weight".into(), weights.embed.clone());
    params.insert("model.norm.weight".into(), weights.final_norm.clone());
    params.insert("lm_head.weight".into(), weights.lm_head.clone());
    let inv = crate::rope::inv_freq(cfg);
    let (cos, sin) = crate::rope::build_rope_tables(cfg, &inv, cfg.max_position_embeddings);
    params.insert("rope.cos".into(), cos);
    params.insert("rope.sin".into(), sin);
    for (il, layer) in weights.layers.iter().enumerate() {
        let p = |t: &str| format!("model.layers.{il}.{t}");
        params.insert(p("input_layernorm.weight"), layer.input_norm.clone());
        params.insert(
            p("post_attention_layernorm.weight"),
            layer.post_attn_norm.clone(),
        );
        params.insert(p("self_attn.query_key_value.weight"), layer.qkv.clone());
        params.insert(p("self_attn.dense.weight"), layer.o_proj.clone());
        if let Some(q) = &layer.q_norm {
            params.insert(p("self_attn.query_layernorm.weight"), q.clone());
        }
        if let Some(k) = &layer.k_norm {
            params.insert(p("self_attn.key_layernorm.weight"), k.clone());
        }
        match &layer.ffn {
            LayerFfn::Dense(d) => {
                params.insert(p("mlp.gate_proj.weight"), d.gate.clone());
                params.insert(p("mlp.up_proj.weight"), d.up.clone());
                params.insert(p("mlp.down_proj.weight"), d.down.clone());
            }
            LayerFfn::Moe(m) => {
                params.insert(p("mlp.gate.weight"), m.router.clone());
                params.insert(p("mlp.gate.expert_bias"), m.expert_bias.clone());
                params.insert(p("mlp.gate_exps.weight"), m.gate_exps.clone());
                params.insert(p("mlp.up_exps.weight"), m.up_exps.clone());
                params.insert(p("mlp.down_exps.weight"), m.down_exps.clone());
                if let Some(w) = &m.shared_gate {
                    params.insert(p("mlp.shared_experts.gate_proj.weight"), w.clone());
                }
                if let Some(w) = &m.shared_up {
                    params.insert(p("mlp.shared_experts.up_proj.weight"), w.clone());
                }
                if let Some(w) = &m.shared_down {
                    params.insert(p("mlp.shared_experts.down_proj.weight"), w.clone());
                }
            }
        }
    }
}
