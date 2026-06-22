use crate::checkpoint::MoshiCheckpoint;
use crate::config::LmConfig;
use crate::gguf::load_gguf_weight_map;
use crate::lm::LmModel;
use anyhow::{Context, Result};
use rlx_core::safetensors_checkpoint::SafetensorsCheckpoint;
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub fn load_weight_map(model_dir: &Path) -> Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    let ckpt = SafetensorsCheckpoint::open(model_dir)?;
    let keys: HashSet<String> = ckpt.keys().map(str::to_string).collect();
    load_weight_map_keys(model_dir, &keys)
}

pub fn load_lm_weights(
    model_dir: &Path,
    cfg: &LmConfig,
) -> Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    let keys: HashSet<String> = expected_lm_keys(cfg).into_iter().collect();
    load_weight_map_keys(model_dir, &keys)
}

pub fn load_lm_weights_from_checkpoint(
    model_dir: &Path,
    cfg: &LmConfig,
    checkpoint: MoshiCheckpoint,
) -> Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    let path = checkpoint.lm_weights_path(model_dir);
    if checkpoint.is_gguf() {
        load_gguf_weight_map(&path, cfg)
    } else {
        load_lm_weights(model_dir, cfg)
    }
}

fn load_weight_map_keys(
    model_dir: &Path,
    keys: &HashSet<String>,
) -> Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    let ckpt = SafetensorsCheckpoint::open(model_dir)?;
    let mut wm = ckpt.load_selected(keys)?;
    let mut map = HashMap::with_capacity(keys.len());
    for key in keys {
        let (data, shape) = wm
            .take(key)
            .with_context(|| format!("tensor {key} missing after load"))?;
        map.insert(key.clone(), (data, shape));
    }
    Ok(map)
}

pub fn open_lm(model_dir: &Path, cfg: LmConfig) -> Result<LmModel> {
    let weights = load_lm_weights(model_dir, &cfg)?;
    open_lm_from_weights(cfg, weights)
}

pub fn open_lm_from_weights(
    cfg: LmConfig,
    weights: HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<LmModel> {
    LmModel::open(cfg, weights)
}

pub fn open_lm_from_checkpoint(
    model_dir: &Path,
    cfg: LmConfig,
    checkpoint: MoshiCheckpoint,
) -> Result<LmModel> {
    let weights = load_lm_weights_from_checkpoint(model_dir, &cfg, checkpoint)?;
    open_lm_from_weights(cfg, weights)
}

pub fn expected_lm_keys(cfg: &LmConfig) -> Vec<String> {
    let mut keys = vec![
        "text_emb.weight".into(),
        "text_linear.weight".into(),
        "out_norm.alpha".into(),
    ];
    for i in 0..cfg.audio_codebooks {
        keys.push(format!("emb.{i}.weight"));
    }
    for li in 0..cfg.transformer.num_layers {
        let p = format!("transformer.layers.{li}.");
        keys.extend([
            format!("{p}norm1.alpha"),
            format!("{p}norm2.alpha"),
            format!("{p}self_attn.in_proj_weight"),
            format!("{p}self_attn.out_proj.weight"),
            format!("{p}gating.linear_in.weight"),
            format!("{p}gating.linear_out.weight"),
        ]);
    }
    if let Some(df) = &cfg.depformer {
        for si in 0..df.num_slices {
            let p = format!("depformer.{si}.");
            keys.extend([
                format!("{p}emb.weight"),
                format!("{p}linear_in.weight"),
                format!("{p}linear_out.weight"),
            ]);
            for li in 0..df.transformer.num_layers {
                let lp = format!("{p}transformer.layers.{li}.");
                keys.extend([
                    format!("{lp}norm1.alpha"),
                    format!("{lp}norm2.alpha"),
                    format!("{lp}self_attn.in_proj_weight"),
                    format!("{lp}self_attn.out_proj.weight"),
                    format!("{lp}gating.linear_in.weight"),
                    format!("{lp}gating.linear_out.weight"),
                ]);
            }
        }
    }
    keys
}
