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

//! LFM2.5-VL vision encoder + LLaVA-style multimodal projector.

use anyhow::Result;
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::blocks::siglip_layer_fused_with_prefix;
use rlx_flow::{BuiltModel, CompileProfile, GgufPackedParams, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Shape};

use super::config::LfmVlVisionConfig;
use super::preprocess::{LfmVlPreprocessWeights, extract_preprocess_weights};

const ENCODER_PREFIX: &str = "vision_tower.vision_model.encoder";
const POST_LN_W: &str = "vision_tower.vision_model.post_layernorm.weight";
const POST_LN_B: &str = "vision_tower.vision_model.post_layernorm.bias";

pub struct LfmVlVisionBuilt {
    pub model: BuiltModel,
    pub preprocess: LfmVlPreprocessWeights,
}

pub fn build_lfm_vl_vision(
    cfg: &LfmVlVisionConfig,
    weights: &mut WeightMap,
) -> Result<LfmVlVisionBuilt> {
    build_lfm_vl_vision_with_packed(cfg, weights, None)
}

pub fn build_lfm_vl_vision_with_packed(
    cfg: &LfmVlVisionConfig,
    weights: &mut WeightMap,
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<LfmVlVisionBuilt> {
    let preprocess = extract_preprocess_weights(weights, cfg)?;

    let batch = 1usize;
    let seq = cfg.seq_len();
    let h = cfg.hidden_size;
    let lm_h = cfg.projector_output_dim;
    let nh = cfg.num_attention_heads;
    let eps = cfg.layer_norm_eps as f32;
    let f = DType::F32;

    let mut flow = ModelFlow::new("lfm_vl_vision")
        .with_profile(CompileProfile::encoder())
        .input("hidden", Shape::new(&[batch, seq, h], f))
        .attn_mask_ones(batch, seq);

    for i in 0..cfg.num_hidden_layers {
        let layer_prefix = format!("{ENCODER_PREFIX}.layers.{i}");
        flow = flow.named_layer(
            format!("layer{i}"),
            siglip_layer_fused_with_prefix(layer_prefix, i, h, nh, eps),
        );
    }

    flow = flow
        .layer_norm(POST_LN_W, POST_LN_B, eps)
        .plugin_named("lfm_vl.projector", move |emit, hidden| {
            let v = hidden.ok_or_else(|| anyhow::anyhow!("projector requires hidden"))?;
            let l1_w = emit.load_param("multi_modal_projector.linear_1.weight", true)?;
            let l1_b = emit.load_param("multi_modal_projector.linear_1.bias", false)?;
            let l2_w = emit.load_param("multi_modal_projector.linear_2.weight", true)?;
            let l2_b = emit.load_param("multi_modal_projector.linear_2.bias", false)?;
            let mut gb = HirMut::new(emit.hir());
            let m1 = gb.mm(v.hir_id(), l1_w);
            let a1 = gb.add(m1, l1_b);
            let act = gb.gelu(a1);
            let m2 = gb.mm(act, l2_w);
            let out = gb.add(m2, l2_b);
            Ok(Some(
                emit.wrap(out, Shape::new(&[batch, seq, lm_h], DType::F32)),
            ))
        })
        .output("lm_embeds");

    Ok(LfmVlVisionBuilt {
        model: flow.build_with(&mut WeightMapSource(weights), gguf_packed)?,
        preprocess,
    })
}
