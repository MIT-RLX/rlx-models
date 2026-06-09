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

//! Weight extraction for V-JEPA2 checkpoints (HF + Meta key layouts).

use super::config::Vjepa2Config;
use super::preprocess::{Vjepa2PatchEmbedWeights, extract_patch_embed_weights};
use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;

#[derive(Clone)]
pub struct Vjepa2BlockWeights {
    pub norm1_w: Vec<f32>,
    pub norm1_b: Vec<f32>,
    pub q_w_t: Vec<f32>,
    pub q_b: Vec<f32>,
    pub k_w_t: Vec<f32>,
    pub k_b: Vec<f32>,
    pub v_w_t: Vec<f32>,
    pub v_b: Vec<f32>,
    pub proj_w_t: Vec<f32>,
    pub proj_b: Vec<f32>,
    pub norm2_w: Vec<f32>,
    pub norm2_b: Vec<f32>,
    pub mlp_fc1_w_t: Vec<f32>,
    pub mlp_fc1_b: Vec<f32>,
    pub mlp_fc2_w_t: Vec<f32>,
    pub mlp_fc2_b: Vec<f32>,
}

#[derive(Clone)]
pub struct Vjepa2EncoderWeights {
    pub patch: Vjepa2PatchEmbedWeights,
    pub blocks: Vec<Vjepa2BlockWeights>,
    pub norm_w: Vec<f32>,
    pub norm_b: Vec<f32>,
}

#[derive(Clone)]
pub struct Vjepa2PredictorWeights {
    pub embed_w_t: Vec<f32>,
    pub embed_b: Vec<f32>,
    pub mask_tokens: Vec<f32>,
    pub blocks: Vec<Vjepa2BlockWeights>,
    pub norm_w: Vec<f32>,
    pub norm_b: Vec<f32>,
    pub proj_w_t: Vec<f32>,
    pub proj_b: Vec<f32>,
}

#[derive(Clone)]
pub struct Vjepa2PoolerSelfBlockWeights {
    pub norm1_w: Vec<f32>,
    pub norm1_b: Vec<f32>,
    pub q_w_t: Vec<f32>,
    pub q_b: Vec<f32>,
    pub k_w_t: Vec<f32>,
    pub k_b: Vec<f32>,
    pub v_w_t: Vec<f32>,
    pub v_b: Vec<f32>,
    pub out_w_t: Vec<f32>,
    pub out_b: Vec<f32>,
    pub norm2_w: Vec<f32>,
    pub norm2_b: Vec<f32>,
    pub mlp_fc1_w_t: Vec<f32>,
    pub mlp_fc1_b: Vec<f32>,
    pub mlp_fc2_w_t: Vec<f32>,
    pub mlp_fc2_b: Vec<f32>,
}

#[derive(Clone)]
pub struct Vjepa2PoolerCrossWeights {
    pub norm1_w: Vec<f32>,
    pub norm1_b: Vec<f32>,
    pub q_w_t: Vec<f32>,
    pub q_b: Vec<f32>,
    pub k_w_t: Vec<f32>,
    pub k_b: Vec<f32>,
    pub v_w_t: Vec<f32>,
    pub v_b: Vec<f32>,
    pub norm2_w: Vec<f32>,
    pub norm2_b: Vec<f32>,
    pub mlp_fc1_w_t: Vec<f32>,
    pub mlp_fc1_b: Vec<f32>,
    pub mlp_fc2_w_t: Vec<f32>,
    pub mlp_fc2_b: Vec<f32>,
}

#[derive(Clone)]
pub struct Vjepa2PoolerWeights {
    pub query_tokens: Vec<f32>,
    pub self_blocks: Vec<Vjepa2PoolerSelfBlockWeights>,
    pub cross: Vjepa2PoolerCrossWeights,
    pub classifier_w_t: Option<Vec<f32>>,
    pub classifier_b: Option<Vec<f32>>,
}

#[derive(Clone)]
pub struct Vjepa2ModelWeights {
    pub encoder: Vjepa2EncoderWeights,
    pub predictor: Option<Vjepa2PredictorWeights>,
    pub pooler: Option<Vjepa2PoolerWeights>,
}

pub fn extract_encoder_weights(
    weights: &mut WeightMap,
    cfg: &Vjepa2Config,
) -> Result<Vjepa2EncoderWeights> {
    let patch = extract_patch_embed_weights(weights, cfg)?;
    let e = cfg.hidden_size;
    let hidden = cfg.intermediate_size();
    let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);

    for i in 0..cfg.num_hidden_layers {
        let hf = format!("encoder.layer.{i}");
        let meta = format!("blocks.{i}");
        blocks.push(extract_transformer_block(
            weights,
            &[hf, meta],
            e,
            hidden,
            "attention",
            "attn",
        )?);
    }

    let norm_w = take_first_vec(
        weights,
        &["encoder.layernorm.weight", "norm.weight"],
        vec![e],
    )?;
    let norm_b = take_first_vec(weights, &["encoder.layernorm.bias", "norm.bias"], vec![e])?;

    Ok(Vjepa2EncoderWeights {
        patch,
        blocks,
        norm_w,
        norm_b,
    })
}

pub fn extract_predictor_weights(
    weights: &mut WeightMap,
    cfg: &Vjepa2Config,
) -> Result<Vjepa2PredictorWeights> {
    let enc = cfg.hidden_size;
    let pred = cfg.pred_hidden_size;
    let hidden = cfg.pred_intermediate_size();

    let embed_key = pick_key(
        weights,
        &[
            "predictor.embeddings.predictor_embeddings.weight",
            "predictor_embed.weight",
        ],
    )?;
    let embed_w_t = take_linear_w_key(weights, &embed_key, enc, pred)?;
    let embed_b = take_first_vec(
        weights,
        &[
            "predictor.embeddings.predictor_embeddings.bias",
            "predictor_embed.bias",
        ],
        vec![pred],
    )?;

    let n_masks = cfg.pred_num_mask_tokens;
    let mask_tokens = take_first_vec(
        weights,
        &["predictor.embeddings.mask_tokens", "mask_tokens"],
        vec![n_masks, 1, 1, pred],
    )?;

    let mut blocks = Vec::with_capacity(cfg.pred_num_hidden_layers);
    for i in 0..cfg.pred_num_hidden_layers {
        let hf = format!("predictor.layer.{i}");
        let meta = format!("predictor_blocks.{i}");
        blocks.push(extract_transformer_block(
            weights,
            &[hf, meta],
            pred,
            hidden,
            "attention",
            "attn",
        )?);
    }

    let norm_w = take_first_vec(
        weights,
        &["predictor.layernorm.weight", "predictor_norm.weight"],
        vec![pred],
    )?;
    let norm_b = take_first_vec(
        weights,
        &["predictor.layernorm.bias", "predictor_norm.bias"],
        vec![pred],
    )?;
    let proj_key = pick_key(weights, &["predictor.proj.weight", "predictor_proj.weight"])?;
    let proj_w_t = take_linear_w_key(weights, &proj_key, pred, enc)?;
    let proj_b = take_first_vec(
        weights,
        &["predictor.proj.bias", "predictor_proj.bias"],
        vec![enc],
    )?;

    Ok(Vjepa2PredictorWeights {
        embed_w_t,
        embed_b,
        mask_tokens,
        blocks,
        norm_w,
        norm_b,
        proj_w_t,
        proj_b,
    })
}

pub fn extract_pooler_weights(
    weights: &mut WeightMap,
    cfg: &Vjepa2Config,
) -> Result<Vjepa2PoolerWeights> {
    let e = cfg.hidden_size;
    let hidden = cfg.pooler_intermediate_size();

    let query_tokens = take_first_vec(weights, &["pooler.query_tokens"], vec![1, 1, e])?;

    let mut self_blocks = Vec::with_capacity(cfg.num_pooler_layers);
    for i in 0..cfg.num_pooler_layers {
        let p = format!("pooler.self_attention_layers.{i}");
        self_blocks.push(Vjepa2PoolerSelfBlockWeights {
            norm1_w: take_ln_w(weights, &[&p], "layer_norm1", e)?,
            norm1_b: take_ln_b(weights, &[&p], "layer_norm1", e)?,
            q_w_t: take_linear_w_key(weights, &format!("{p}.self_attn.q_proj.weight"), e, e)?,
            q_b: take_first_vec(weights, &[&format!("{p}.self_attn.q_proj.bias")], vec![e])?,
            k_w_t: take_linear_w_key(weights, &format!("{p}.self_attn.k_proj.weight"), e, e)?,
            k_b: take_first_vec(weights, &[&format!("{p}.self_attn.k_proj.bias")], vec![e])?,
            v_w_t: take_linear_w_key(weights, &format!("{p}.self_attn.v_proj.weight"), e, e)?,
            v_b: take_first_vec(weights, &[&format!("{p}.self_attn.v_proj.bias")], vec![e])?,
            out_w_t: take_linear_w_key(weights, &format!("{p}.self_attn.out_proj.weight"), e, e)?,
            out_b: take_first_vec(weights, &[&format!("{p}.self_attn.out_proj.bias")], vec![e])?,
            norm2_w: take_ln_w(weights, &[&p], "layer_norm2", e)?,
            norm2_b: take_ln_b(weights, &[&p], "layer_norm2", e)?,
            mlp_fc1_w_t: take_linear_w_key(weights, &format!("{p}.mlp.fc1.weight"), e, hidden)?,
            mlp_fc1_b: take_first_vec(weights, &[&format!("{p}.mlp.fc1.bias")], vec![hidden])?,
            mlp_fc2_w_t: take_linear_w_key(weights, &format!("{p}.mlp.fc2.weight"), hidden, e)?,
            mlp_fc2_b: take_first_vec(weights, &[&format!("{p}.mlp.fc2.bias")], vec![e])?,
        });
    }

    let cp = "pooler.cross_attention_layer";
    let cross = Vjepa2PoolerCrossWeights {
        norm1_w: take_ln_w(weights, &[cp], "layer_norm1", e)?,
        norm1_b: take_ln_b(weights, &[cp], "layer_norm1", e)?,
        q_w_t: take_linear_w_key(weights, &format!("{cp}.cross_attn.q_proj.weight"), e, e)?,
        q_b: take_first_vec(weights, &[&format!("{cp}.cross_attn.q_proj.bias")], vec![e])?,
        k_w_t: take_linear_w_key(weights, &format!("{cp}.cross_attn.k_proj.weight"), e, e)?,
        k_b: take_first_vec(weights, &[&format!("{cp}.cross_attn.k_proj.bias")], vec![e])?,
        v_w_t: take_linear_w_key(weights, &format!("{cp}.cross_attn.v_proj.weight"), e, e)?,
        v_b: take_first_vec(weights, &[&format!("{cp}.cross_attn.v_proj.bias")], vec![e])?,
        norm2_w: take_ln_w(weights, &[cp], "layer_norm2", e)?,
        norm2_b: take_ln_b(weights, &[cp], "layer_norm2", e)?,
        mlp_fc1_w_t: take_linear_w_key(weights, &format!("{cp}.mlp.fc1.weight"), e, hidden)?,
        mlp_fc1_b: take_first_vec(weights, &[&format!("{cp}.mlp.fc1.bias")], vec![hidden])?,
        mlp_fc2_w_t: take_linear_w_key(weights, &format!("{cp}.mlp.fc2.weight"), hidden, e)?,
        mlp_fc2_b: take_first_vec(weights, &[&format!("{cp}.mlp.fc2.bias")], vec![e])?,
    };

    let classifier_w_t = if weights.has("classifier.weight") {
        let (data, shape) = weights.take_transposed("classifier.weight")?;
        ensure!(shape[1] == e, "classifier weight second dim must be {e}");
        Some(data)
    } else {
        None
    };
    let classifier_b = if weights.has("classifier.bias") {
        let (data, shape) = weights.take("classifier.bias")?;
        ensure!(shape.len() == 1, "classifier bias must be 1d");
        Some(data)
    } else {
        None
    };

    Ok(Vjepa2PoolerWeights {
        query_tokens,
        self_blocks,
        cross,
        classifier_w_t,
        classifier_b,
    })
}

pub fn extract_model_weights(
    weights: &mut WeightMap,
    cfg: &Vjepa2Config,
) -> Result<Vjepa2ModelWeights> {
    let encoder = extract_encoder_weights(weights, cfg)?;
    let predictor = if weights.has("predictor.layer.0.attention.query.weight")
        || weights.has("predictor_blocks.0.attn.qkv.weight")
    {
        Some(extract_predictor_weights(weights, cfg)?)
    } else {
        None
    };
    let pooler = if weights.has("pooler.query_tokens") {
        Some(extract_pooler_weights(weights, cfg)?)
    } else {
        None
    };
    Ok(Vjepa2ModelWeights {
        encoder,
        predictor,
        pooler,
    })
}

pub(crate) fn extract_transformer_block(
    weights: &mut WeightMap,
    prefixes: &[String],
    embed: usize,
    hidden: usize,
    attn_hf: &str,
    attn_meta: &str,
) -> Result<Vjepa2BlockWeights> {
    let pref_refs: Vec<&str> = prefixes.iter().map(String::as_str).collect();
    Ok(Vjepa2BlockWeights {
        norm1_w: take_ln_w(weights, &pref_refs, "norm1", embed)?,
        norm1_b: take_ln_b(weights, &pref_refs, "norm1", embed)?,
        q_w_t: take_linear_w(
            weights, &pref_refs, "query", embed, embed, attn_hf, attn_meta,
        )?,
        q_b: take_linear_b(weights, &pref_refs, "query", embed, attn_hf, attn_meta)?,
        k_w_t: take_linear_w(weights, &pref_refs, "key", embed, embed, attn_hf, attn_meta)?,
        k_b: take_linear_b(weights, &pref_refs, "key", embed, attn_hf, attn_meta)?,
        v_w_t: take_linear_w(
            weights, &pref_refs, "value", embed, embed, attn_hf, attn_meta,
        )?,
        v_b: take_linear_b(weights, &pref_refs, "value", embed, attn_hf, attn_meta)?,
        proj_w_t: take_attn_proj_w(weights, &pref_refs, embed, attn_hf, attn_meta)?,
        proj_b: take_attn_proj_b(weights, &pref_refs, embed, attn_hf, attn_meta)?,
        norm2_w: take_ln_w(weights, &pref_refs, "norm2", embed)?,
        norm2_b: take_ln_b(weights, &pref_refs, "norm2", embed)?,
        mlp_fc1_w_t: take_mlp_w(weights, &pref_refs, "fc1", embed, hidden)?,
        mlp_fc1_b: take_mlp_b(weights, &pref_refs, "fc1", hidden)?,
        mlp_fc2_w_t: take_mlp_w(weights, &pref_refs, "fc2", hidden, embed)?,
        mlp_fc2_b: take_mlp_b(weights, &pref_refs, "fc2", embed)?,
    })
}

fn pick_key(weights: &WeightMap, keys: &[&str]) -> Result<String> {
    for k in keys {
        if weights.has(k) {
            return Ok((*k).to_string());
        }
    }
    anyhow::bail!("none of keys found: {keys:?}")
}

fn take_attn_proj_w(
    weights: &mut WeightMap,
    prefixes: &[&str],
    e: usize,
    attn_hf: &str,
    attn_meta: &str,
) -> Result<Vec<f32>> {
    for p in prefixes {
        let hf = format!("{p}.{attn_hf}.proj.weight");
        if weights.has(&hf) {
            return take_linear_w_key(weights, &hf, e, e);
        }
        let meta = format!("{p}.{attn_meta}.proj.weight");
        if weights.has(&meta) {
            return take_linear_w_key(weights, &meta, e, e);
        }
    }
    anyhow::bail!("attention proj weight not found for {prefixes:?}")
}

fn take_attn_proj_b(
    weights: &mut WeightMap,
    prefixes: &[&str],
    e: usize,
    attn_hf: &str,
    attn_meta: &str,
) -> Result<Vec<f32>> {
    for p in prefixes {
        for suffix in [
            format!("{attn_hf}.proj.bias"),
            format!("{attn_meta}.proj.bias"),
        ] {
            let key = format!("{p}.{suffix}");
            if weights.has(&key) {
                let (data, shape) = weights.take(&key)?;
                ensure!(shape == vec![e]);
                return Ok(data);
            }
        }
    }
    anyhow::bail!("attention proj bias not found")
}

fn take_linear_w(
    weights: &mut WeightMap,
    prefixes: &[&str],
    name: &str,
    in_dim: usize,
    out_dim: usize,
    attn_hf: &str,
    attn_meta: &str,
) -> Result<Vec<f32>> {
    for p in prefixes {
        let hf = format!("{p}.{attn_hf}.{name}.weight");
        if weights.has(&hf) {
            return take_linear_w_key(weights, &hf, in_dim, out_dim);
        }
    }
    for p in prefixes {
        if !p.starts_with("blocks.") && !p.starts_with("predictor_blocks.") {
            continue;
        }
        let key = format!("{p}.{attn_meta}.qkv.weight");
        if weights.has(&key) {
            let (data, shape) = weights.take_transposed(&key)?;
            ensure!(shape == vec![in_dim, 3 * out_dim]);
            return Ok(split_qkv_w(&data, in_dim, out_dim, name));
        }
    }
    anyhow::bail!("linear weight {name} not found for {prefixes:?}")
}

fn take_linear_b(
    weights: &mut WeightMap,
    prefixes: &[&str],
    name: &str,
    dim: usize,
    attn_hf: &str,
    attn_meta: &str,
) -> Result<Vec<f32>> {
    for p in prefixes {
        let hf = format!("{p}.{attn_hf}.{name}.bias");
        if weights.has(&hf) {
            let (data, shape) = weights.take(&hf)?;
            ensure!(shape == vec![dim]);
            return Ok(data);
        }
    }
    for p in prefixes {
        if !p.starts_with("blocks.") && !p.starts_with("predictor_blocks.") {
            continue;
        }
        let key = format!("{p}.{attn_meta}.qkv.bias");
        if weights.has(&key) {
            let (data, shape) = weights.take(&key)?;
            ensure!(shape == vec![3 * dim]);
            return Ok(split_qkv_b(&data, dim, name));
        }
    }
    anyhow::bail!("linear bias {name} not found")
}

fn split_qkv_w(data: &[f32], in_dim: usize, out_dim: usize, which: &str) -> Vec<f32> {
    let off = match which {
        "query" => 0,
        "key" => out_dim,
        "value" => 2 * out_dim,
        _ => panic!("bad qkv split {which}"),
    };
    let mut out = vec![0f32; in_dim * out_dim];
    for i in 0..in_dim {
        for j in 0..out_dim {
            out[i * out_dim + j] = data[i * 3 * out_dim + off + j];
        }
    }
    out
}

fn split_qkv_b(data: &[f32], dim: usize, which: &str) -> Vec<f32> {
    let off = match which {
        "query" => 0,
        "key" => dim,
        "value" => 2 * dim,
        _ => panic!("bad qkv split {which}"),
    };
    data[off..off + dim].to_vec()
}

fn take_mlp_w(
    weights: &mut WeightMap,
    prefixes: &[&str],
    fc: &str,
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>> {
    for p in prefixes {
        let key = format!("{p}.mlp.{fc}.weight");
        if weights.has(&key) {
            return take_linear_w_key(weights, &key, in_dim, out_dim);
        }
    }
    anyhow::bail!("mlp {fc} weight not found")
}

fn take_mlp_b(
    weights: &mut WeightMap,
    prefixes: &[&str],
    fc: &str,
    dim: usize,
) -> Result<Vec<f32>> {
    for p in prefixes {
        let key = format!("{p}.mlp.{fc}.bias");
        if weights.has(&key) {
            let (data, shape) = weights.take(&key)?;
            ensure!(shape == vec![dim]);
            return Ok(data);
        }
    }
    anyhow::bail!("mlp {fc} bias not found")
}

fn take_ln_w(
    weights: &mut WeightMap,
    prefixes: &[&str],
    norm: &str,
    dim: usize,
) -> Result<Vec<f32>> {
    for p in prefixes {
        let key = format!("{p}.{norm}.weight");
        if weights.has(&key) {
            let (data, shape) = weights.take(&key)?;
            ensure!(shape == vec![dim]);
            return Ok(data);
        }
    }
    anyhow::bail!("{norm} weight not found")
}

fn take_ln_b(
    weights: &mut WeightMap,
    prefixes: &[&str],
    norm: &str,
    dim: usize,
) -> Result<Vec<f32>> {
    for p in prefixes {
        let key = format!("{p}.{norm}.bias");
        if weights.has(&key) {
            let (data, shape) = weights.take(&key)?;
            ensure!(shape == vec![dim]);
            return Ok(data);
        }
    }
    anyhow::bail!("{norm} bias not found")
}

fn take_linear_w_key(
    weights: &mut WeightMap,
    key: &str,
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>> {
    let (data, shape) = weights.take_transposed(key)?;
    ensure!(
        shape == vec![in_dim, out_dim],
        "{key} expected [{in_dim}, {out_dim}], got {shape:?}"
    );
    Ok(data)
}

fn take_first_vec(
    weights: &mut WeightMap,
    keys: &[&str],
    expected: Vec<usize>,
) -> Result<Vec<f32>> {
    for key in keys {
        if weights.has(key) {
            let (data, shape) = weights.take(key)?;
            ensure!(
                shape == expected,
                "{key} shape mismatch: {shape:?} vs {expected:?}"
            );
            return Ok(data);
        }
    }
    anyhow::bail!("keys not found: {keys:?}")
}
