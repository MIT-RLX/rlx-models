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

//! Tier-0 NomicVision encoder flow — native [`ModelFlow`] ViT assembly.

use anyhow::Result;
use rlx_flow::{BuiltModel, CompileProfile, ModelFlow};
use rlx_ir::{DType, Shape};

use crate::vision::VisionPreprocessWeights;
use rlx_core::config::NomicVisionConfig;
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;

#[derive(Debug, Clone)]
pub struct NomicVisionFlow<'a> {
    cfg: &'a NomicVisionConfig,
    batch: usize,
}

impl<'a> NomicVisionFlow<'a> {
    pub fn new(cfg: &'a NomicVisionConfig, batch: usize) -> Self {
        Self { cfg, batch }
    }

    pub fn build(self, weights: &mut WeightMap) -> Result<NomicVisionBuilt> {
        build_nomic_vision_built(self.cfg, weights, self.batch)
    }
}

pub struct NomicVisionBuilt {
    pub model: BuiltModel,
    pub preprocess: VisionPreprocessWeights,
}

pub fn build_nomic_vision_built(
    cfg: &NomicVisionConfig,
    weights: &mut WeightMap,
    batch: usize,
) -> Result<NomicVisionBuilt> {
    let preprocess = extract_vision_preprocess(weights)?;
    let final_ln = resolve_final_norm_prefix(weights);

    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let eps = cfg.layer_norm_eps() as f32;
    let ps = cfg.patch_size;
    let np = (cfg.img_size / ps) * (cfg.img_size / ps);
    let seq = np + 1;
    let f = DType::F32;

    let model = ModelFlow::new("nomic_vision")
        .with_profile(CompileProfile::encoder())
        .input("hidden", Shape::new(&[batch, seq, h], f))
        .attn_mask_ones(batch, seq)
        .repeat_vision_layers(cfg.num_hidden_layers, h, nh, eps)
        .layer_norm(
            format!("{final_ln}.weight"),
            format!("{final_ln}.bias"),
            eps,
        )
        .cls_token_pool(batch, h)
        .output("cls")
        .build(&mut WeightMapSource(weights))?;

    Ok(NomicVisionBuilt { model, preprocess })
}

fn extract_vision_preprocess(weights: &mut WeightMap) -> Result<VisionPreprocessWeights> {
    let (proj_w_data, proj_w_shape) = weights.take_transposed("embeddings.proj.weight")?;
    let (proj_b_data, _) = weights.take("embeddings.proj.bias")?;
    let (cls_token_data, _) = weights.take("embeddings.cls_token")?;
    let (pos_embed_data, _) = weights.take("embeddings.pos_embed")?;
    Ok(VisionPreprocessWeights {
        proj_w: proj_w_data,
        proj_w_cols: proj_w_shape.last().copied().unwrap_or(0),
        proj_b: proj_b_data,
        cls_token: cls_token_data,
        pos_embed: pos_embed_data,
    })
}

fn resolve_final_norm_prefix(weights: &WeightMap) -> &'static str {
    if weights.has("norm.weight") {
        "norm"
    } else if weights.has("selector.norm1.weight") {
        "selector.norm1"
    } else {
        "encoder.norm"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn tiny_cfg() -> NomicVisionConfig {
        NomicVisionConfig {
            hidden_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            n_inner: 32,
            img_size: 32,
            patch_size: 16,
            layer_norm_epsilon: 1e-5,
        }
    }

    fn synth_weights(cfg: &NomicVisionConfig) -> WeightMap {
        let h = cfg.hidden_size;
        let int_dim = cfg.intermediate_size();
        let ps = cfg.patch_size;
        let patch_dim = 3 * ps * ps;
        let np = (cfg.img_size / ps) * (cfg.img_size / ps);
        let seq = np + 1;
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let z = |n: usize| vec![0.0f32; n];
        t.insert(
            "embeddings.proj.weight".into(),
            (z(patch_dim * h), vec![h, patch_dim]),
        );
        t.insert("embeddings.proj.bias".into(), (z(h), vec![h]));
        t.insert("embeddings.cls_token".into(), (z(h), vec![1, 1, h]));
        t.insert("embeddings.pos_embed".into(), (z(seq * h), vec![1, seq, h]));
        let lp = "layers.0";
        t.insert(format!("{lp}.norm1.weight"), (z(h), vec![h]));
        t.insert(format!("{lp}.norm1.bias"), (z(h), vec![h]));
        t.insert(format!("{lp}.norm2.weight"), (z(h), vec![h]));
        t.insert(format!("{lp}.norm2.bias"), (z(h), vec![h]));
        t.insert(
            format!("{lp}.attn.Wqkv.weight"),
            (z(3 * h * h), vec![3 * h, h]),
        );
        t.insert(format!("{lp}.attn.Wqkv.bias"), (z(3 * h), vec![3 * h]));
        t.insert(format!("{lp}.attn.out_proj.weight"), (z(h * h), vec![h, h]));
        t.insert(format!("{lp}.attn.out_proj.bias"), (z(h), vec![h]));
        t.insert(
            format!("{lp}.mlp.fc11.weight"),
            (z(int_dim * h), vec![int_dim, h]),
        );
        t.insert(format!("{lp}.mlp.fc11.bias"), (z(int_dim), vec![int_dim]));
        t.insert(
            format!("{lp}.mlp.fc12.weight"),
            (z(int_dim * h), vec![int_dim, h]),
        );
        t.insert(format!("{lp}.mlp.fc12.bias"), (z(int_dim), vec![int_dim]));
        t.insert(
            format!("{lp}.mlp.fc2.weight"),
            (z(h * int_dim), vec![h, int_dim]),
        );
        t.insert(format!("{lp}.mlp.fc2.bias"), (z(h), vec![h]));
        t.insert(format!("{lp}.mlp.norm.weight"), (z(int_dim), vec![int_dim]));
        t.insert(format!("{lp}.mlp.norm.bias"), (z(int_dim), vec![int_dim]));
        t.insert("norm.weight".into(), (z(h), vec![h]));
        t.insert("norm.bias".into(), (z(h), vec![h]));
        WeightMap::from_tensors(t)
    }

    #[test]
    fn vision_flow_builds() {
        let cfg = tiny_cfg();
        let mut wm = synth_weights(&cfg);
        let built = NomicVisionFlow::new(&cfg, 1).build(&mut wm).unwrap();
        assert_eq!(
            *built.model.primary_shape(),
            Shape::new(&[1, cfg.hidden_size], DType::F32)
        );
    }
}
