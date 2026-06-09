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

//! FLUX.2 text encoder (Qwen3) weight loading.

use super::super::weights::{LinearWeights, RmsNormWeight, load_linear, load_rms};
use anyhow::{Context, Result, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_qwen3::Qwen3Config;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Flux2TextEncoderMlpWeights {
    pub gate: LinearWeights,
    pub up: LinearWeights,
    pub down: LinearWeights,
}

#[derive(Debug, Clone)]
pub struct Flux2TextEncoderAttnWeights {
    pub q: LinearWeights,
    pub k: LinearWeights,
    pub v: LinearWeights,
    pub o: LinearWeights,
    pub q_norm: RmsNormWeight,
    pub k_norm: RmsNormWeight,
}

#[derive(Debug, Clone)]
pub struct Flux2TextEncoderLayerWeights {
    pub input_layernorm: RmsNormWeight,
    pub post_attention_layernorm: RmsNormWeight,
    pub attn: Flux2TextEncoderAttnWeights,
    pub mlp: Flux2TextEncoderMlpWeights,
}

#[derive(Debug, Clone)]
pub struct Flux2TextEncoderWeights {
    pub embed_tokens: (Vec<f32>, usize, usize),
    pub norm: RmsNormWeight,
    pub layers: Vec<Flux2TextEncoderLayerWeights>,
}

fn normalize_text_encoder_keys(mut wm: WeightMap) -> WeightMap {
    wm.remap_keys(|k| k.strip_prefix("model.").unwrap_or(&k).to_string());
    wm
}

pub fn load_text_encoder_weights(
    path: &Path,
    cfg: &Qwen3Config,
) -> Result<Flux2TextEncoderWeights> {
    let wm = if path.is_dir() {
        WeightMap::from_safetensors_dir(path)?
    } else {
        WeightMap::from_file(path.to_str().context("non-utf8 path")?)?
    };
    extract_text_encoder_weights(normalize_text_encoder_keys(wm), cfg)
}

pub fn extract_text_encoder_weights(
    mut wm: WeightMap,
    cfg: &Qwen3Config,
) -> Result<Flux2TextEncoderWeights> {
    let (embed_data, embed_shape) = wm.take("embed_tokens.weight")?;
    ensure!(
        embed_shape.len() == 2,
        "embed_tokens.weight: expected [vocab, hidden]"
    );
    let vocab = embed_shape[0];
    let hidden = embed_shape[1];
    ensure!(
        hidden == cfg.hidden_size,
        "embed hidden {} != config {}",
        hidden,
        cfg.hidden_size
    );

    let norm = load_rms(&mut wm, "norm.weight")?;
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("layers.{i}");
        layers.push(Flux2TextEncoderLayerWeights {
            input_layernorm: load_rms(&mut wm, &format!("{lp}.input_layernorm.weight"))?,
            post_attention_layernorm: load_rms(
                &mut wm,
                &format!("{lp}.post_attention_layernorm.weight"),
            )?,
            attn: Flux2TextEncoderAttnWeights {
                q: load_linear(
                    &mut wm,
                    &format!("{lp}.self_attn.q_proj.weight"),
                    &format!("{lp}.self_attn.q_proj.bias"),
                    cfg.attention_bias,
                )?,
                k: load_linear(
                    &mut wm,
                    &format!("{lp}.self_attn.k_proj.weight"),
                    &format!("{lp}.self_attn.k_proj.bias"),
                    cfg.attention_bias,
                )?,
                v: load_linear(
                    &mut wm,
                    &format!("{lp}.self_attn.v_proj.weight"),
                    &format!("{lp}.self_attn.v_proj.bias"),
                    cfg.attention_bias,
                )?,
                o: load_linear(
                    &mut wm,
                    &format!("{lp}.self_attn.o_proj.weight"),
                    &format!("{lp}.self_attn.o_proj.bias"),
                    cfg.attention_bias,
                )?,
                q_norm: load_rms(&mut wm, &format!("{lp}.self_attn.q_norm.weight"))?,
                k_norm: load_rms(&mut wm, &format!("{lp}.self_attn.k_norm.weight"))?,
            },
            mlp: Flux2TextEncoderMlpWeights {
                gate: load_linear(
                    &mut wm,
                    &format!("{lp}.mlp.gate_proj.weight"),
                    &format!("{lp}.mlp.gate_proj.bias"),
                    false,
                )?,
                up: load_linear(
                    &mut wm,
                    &format!("{lp}.mlp.up_proj.weight"),
                    &format!("{lp}.mlp.up_proj.bias"),
                    false,
                )?,
                down: load_linear(
                    &mut wm,
                    &format!("{lp}.mlp.down_proj.weight"),
                    &format!("{lp}.mlp.down_proj.bias"),
                    false,
                )?,
            },
        });
    }

    Ok(Flux2TextEncoderWeights {
        embed_tokens: (embed_data, vocab, hidden),
        norm,
        layers,
    })
}

/// Tiny zero weights for unit tests (2 layers, hidden 8).
pub fn synthetic_text_encoder_weights(cfg: &Qwen3Config) -> Flux2TextEncoderWeights {
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let h = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let ff = cfg.intermediate_size;
    let hd = cfg.head_dim;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;

    t.insert(
        "embed_tokens.weight".into(),
        (vec![0.01f32; vocab * h], vec![vocab, h]),
    );
    t.insert("norm.weight".into(), (vec![1.0f32; h], vec![h]));

    for i in 0..cfg.num_hidden_layers {
        let lp = format!("layers.{i}");
        for (name, out_d, in_d) in [
            (format!("{lp}.self_attn.q_proj"), nh * hd, h),
            (format!("{lp}.self_attn.k_proj"), nkv * hd, h),
            (format!("{lp}.self_attn.v_proj"), nkv * hd, h),
            (format!("{lp}.self_attn.o_proj"), h, nh * hd),
            (format!("{lp}.mlp.gate_proj"), ff, h),
            (format!("{lp}.mlp.up_proj"), ff, h),
            (format!("{lp}.mlp.down_proj"), h, ff),
        ] {
            t.insert(
                format!("{name}.weight"),
                (vec![0.01f32; out_d * in_d], vec![out_d, in_d]),
            );
            if name.contains("self_attn") {
                t.insert(format!("{name}.bias"), (vec![0.0f32; out_d], vec![out_d]));
            }
        }
        for suffix in [
            "input_layernorm",
            "post_attention_layernorm",
            "self_attn.q_norm",
            "self_attn.k_norm",
        ] {
            let dim = if suffix.contains("norm") && suffix.contains("attn") {
                hd
            } else {
                h
            };
            t.insert(
                format!("{lp}.{suffix}.weight"),
                (vec![1.0f32; dim], vec![dim]),
            );
        }
    }

    extract_text_encoder_weights(WeightMap::from_tensors(t), cfg).expect("synthetic text encoder")
}
