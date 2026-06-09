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

//! Qwen3-VL vision encoder + projector graph.
//!
//! Compiles a SigLIP-variant encoder (`repeat_siglip_layers` ×
//! `num_hidden_layers`) followed by the multimodal projector
//! (`merger.ln_q` LayerNorm → two linears w/ GELU) that maps the
//! vision-hidden into the Qwen3 LM hidden size.

use anyhow::Result;
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, GgufPackedParams, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Shape};

use super::config::Qwen3VlVisionConfig;
use super::preprocess::{Qwen3VlPreprocessWeights, extract_preprocess_weights};

pub struct Qwen3VlVisionBuilt {
    /// Compiled encoder + projector graph.
    /// Input: `"hidden"` `[1, num_patches, hidden_size]`.
    /// Output: `"lm_embeds"` `[1, num_patches, projector_output_dim]`.
    pub model: BuiltModel,
    /// Host-side patch-embed + pos-embed weights.
    pub preprocess: Qwen3VlPreprocessWeights,
}

pub fn build_qwen3_vl_vision(
    cfg: &Qwen3VlVisionConfig,
    weights: &mut WeightMap,
) -> Result<Qwen3VlVisionBuilt> {
    build_qwen3_vl_vision_with_packed(cfg, weights, None)
}

pub fn build_qwen3_vl_vision_with_packed(
    cfg: &Qwen3VlVisionConfig,
    weights: &mut WeightMap,
    gguf_packed: Option<&GgufPackedParams>,
) -> Result<Qwen3VlVisionBuilt> {
    let preprocess = extract_preprocess_weights(weights, cfg)?;

    let batch = 1usize;
    let seq = cfg.seq_len();
    let h = cfg.hidden_size;
    let lm_h = cfg.projector_output_dim;
    let nh = cfg.num_attention_heads;
    let eps = cfg.layer_norm_eps as f32;
    let f = DType::F32;

    let flow = ModelFlow::new("qwen3_vl_vision")
        .with_profile(CompileProfile::encoder())
        .input("hidden", Shape::new(&[batch, seq, h], f))
        .attn_mask_ones(batch, seq)
        .repeat_siglip_layers(cfg.num_hidden_layers, h, nh, eps)
        .layer_norm("post_layernorm.weight", "post_layernorm.bias", eps)
        .plugin_named("qwen3_vl.projector", move |emit, hidden| {
            let v = hidden.ok_or_else(|| anyhow::anyhow!("projector requires hidden"))?;
            let ln_g = emit.load_param("merger.ln_q.weight", false)?;
            let ln_b = emit.load_param("merger.ln_q.bias", false)?;
            let fc1_w = emit.load_param("merger.mlp.0.weight", true)?;
            let fc1_b = emit.load_param("merger.mlp.0.bias", false)?;
            let fc2_w = emit.load_param("merger.mlp.2.weight", true)?;
            let fc2_b = emit.load_param("merger.mlp.2.bias", false)?;
            let mut gb = HirMut::new(emit.hir());
            let normed = gb.ln(v.hir_id(), ln_g, ln_b, eps);
            let fc1_mm = gb.mm(normed, fc1_w);
            let fc1 = gb.add(fc1_mm, fc1_b);
            let act = gb.gelu(fc1);
            let fc2_mm = gb.mm(act, fc2_w);
            let out = gb.add(fc2_mm, fc2_b);
            Ok(Some(
                emit.wrap(out, Shape::new(&[batch, seq, lm_h], DType::F32)),
            ))
        })
        .output("lm_embeds");

    Ok(Qwen3VlVisionBuilt {
        model: flow.build_with(&mut WeightMapSource(weights), gguf_packed)?,
        preprocess,
    })
}
