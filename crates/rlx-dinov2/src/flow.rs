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

//! Tier-0 DINOv2 flow — native [`ModelFlow`] ViT assembly.

use anyhow::Result;
use rlx_flow::{BuiltModel, CompileProfile, GgufPackedParams, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Shape};

use super::config::DinoV2Config;
use super::preprocess::DinoV2PreprocessWeights;
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;

#[derive(Debug, Clone)]
pub struct DinoV2Flow<'a> {
    cfg: &'a DinoV2Config,
    batch: usize,
}

impl<'a> DinoV2Flow<'a> {
    pub fn new(cfg: &'a DinoV2Config, batch: usize) -> Self {
        Self { cfg, batch }
    }

    pub fn build(self, weights: &mut WeightMap) -> Result<DinoV2Built> {
        build_dinov2_built(self.cfg, weights, self.batch)
    }
}

pub struct DinoV2Built {
    pub model: BuiltModel,
    pub preprocess: DinoV2PreprocessWeights,
}

pub fn build_dinov2_built(
    cfg: &DinoV2Config,
    weights: &mut WeightMap,
    batch: usize,
) -> Result<DinoV2Built> {
    build_dinov2_built_with_packed(cfg, weights, batch, None)
}

pub fn build_dinov2_built_with_packed(
    cfg: &DinoV2Config,
    weights: &mut WeightMap,
    batch: usize,
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<DinoV2Built> {
    let preprocess = super::preprocess::extract_preprocess_weights(weights, cfg)?;
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let eps = cfg.layer_norm_eps as f32;
    let seq = cfg.seq_len();
    let f = DType::F32;

    let mut flow = ModelFlow::new("dinov2")
        .with_profile(CompileProfile::encoder())
        .input("hidden", Shape::new(&[batch, seq, h], f))
        .attn_mask_ones(batch, seq)
        .repeat_dinov2_layers(cfg.num_hidden_layers, h, nh, eps)
        .layer_norm("norm.weight", "norm.bias", eps);

    if cfg.num_classes > 0 {
        let patch_start = 1 + cfg.num_register_tokens;
        let num_patches = cfg.num_patches();
        let num_classes = cfg.num_classes;
        flow = flow.plugin_named("dinov2.head", move |emit, hidden| {
            let encoded = hidden.ok_or_else(|| anyhow::anyhow!("dinov2 head requires hidden"))?;
            let head_w = emit.load_param("head.weight", true)?;
            let head_b = emit.load_param("head.bias", false)?;
            let mut gb = HirMut::new(emit.hir());
            let cls_slice = gb.narrow_(encoded.hir_id(), 1, 0, 1);
            let cls_flat = gb.reshape_(cls_slice, vec![batch as i64, h as i64]);
            let patch_tokens = gb.narrow_(encoded.hir_id(), 1, patch_start, num_patches);
            let mean_patches = gb.mean(patch_tokens, vec![1], false);
            let features = gb.concat_(vec![cls_flat, mean_patches], 1);
            let logits_mm = gb.mm(features, head_w);
            let logits = gb.add(logits_mm, head_b);
            Ok(Some(emit.wrap(
                logits,
                Shape::new(&[batch, num_classes], DType::F32),
            )))
        });
        flow = flow.output("logits");
    } else {
        flow = flow.output("hidden");
    }

    Ok(DinoV2Built {
        model: flow.build_with(&mut WeightMapSource(weights), gguf_packed)?,
        preprocess,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn tiny_cfg() -> DinoV2Config {
        DinoV2Config {
            hidden_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            img_size: 32,
            patch_size: 16,
            mlp_ratio: 4.0,
            layer_norm_eps: 1e-5,
            num_register_tokens: 0,
            num_classes: 0,
        }
    }

    fn synth_weights(cfg: &DinoV2Config) -> WeightMap {
        let h = cfg.hidden_size;
        let int_dim = (h as f64 * cfg.mlp_ratio) as usize;
        let patch_dim = cfg.patch_dim();
        let seq = cfg.seq_len();
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let z = |n: usize| vec![0.0f32; n];
        t.insert(
            "patch_embed.proj.weight".into(),
            (z(h * patch_dim), vec![h, 3, cfg.patch_size, cfg.patch_size]),
        );
        t.insert("patch_embed.proj.bias".into(), (z(h), vec![h]));
        t.insert("cls_token".into(), (z(h), vec![1, 1, h]));
        t.insert("pos_embed".into(), (z(seq * h), vec![1, seq, h]));
        let lp = "blocks.0";
        t.insert(format!("{lp}.norm1.weight"), (z(h), vec![h]));
        t.insert(format!("{lp}.norm1.bias"), (z(h), vec![h]));
        t.insert(format!("{lp}.norm2.weight"), (z(h), vec![h]));
        t.insert(format!("{lp}.norm2.bias"), (z(h), vec![h]));
        t.insert(
            format!("{lp}.attn.qkv.weight"),
            (z(3 * h * h), vec![3 * h, h]),
        );
        t.insert(format!("{lp}.attn.qkv.bias"), (z(3 * h), vec![3 * h]));
        t.insert(format!("{lp}.attn.proj.weight"), (z(h * h), vec![h, h]));
        t.insert(format!("{lp}.attn.proj.bias"), (z(h), vec![h]));
        t.insert(format!("{lp}.ls1.gamma"), (z(h), vec![h]));
        t.insert(format!("{lp}.ls2.gamma"), (z(h), vec![h]));
        t.insert(
            format!("{lp}.mlp.fc1.weight"),
            (z(int_dim * h), vec![int_dim, h]),
        );
        t.insert(format!("{lp}.mlp.fc1.bias"), (z(int_dim), vec![int_dim]));
        t.insert(
            format!("{lp}.mlp.fc2.weight"),
            (z(h * int_dim), vec![h, int_dim]),
        );
        t.insert(format!("{lp}.mlp.fc2.bias"), (z(h), vec![h]));
        t.insert("norm.weight".into(), (z(h), vec![h]));
        t.insert("norm.bias".into(), (z(h), vec![h]));
        WeightMap::from_tensors(t)
    }

    #[test]
    fn dinov2_flow_builds() {
        let cfg = tiny_cfg();
        let mut wm = synth_weights(&cfg);
        let built = DinoV2Flow::new(&cfg, 1).build(&mut wm).unwrap();
        assert_eq!(built.model.primary_shape().rank(), 3);
    }
}
