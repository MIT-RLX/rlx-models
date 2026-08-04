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

//! PaliGemma SigLIP-So400m/14 vision tower (no MAP pooling head).
//!
//! Turns SigLIP-normalized images `[B, 3, 224, 224]` into the projected patch
//! sequence `[B, 256, 2048]` (`multi_modal_projector.linear` output), which
//! becomes the leading image tokens of the joint prefix. Unlike SigLIP-2's
//! classifier, π₀ keeps the **full patch sequence** (no attention pooling).
//!
//! The Conv2d patch stem + position embeddings run on host (reusing
//! [`rlx_siglip2::assemble_vision_hidden`]); the graph is 27 bidirectional
//! pre-LN encoder layers → `post_layernorm` → the projector linear. At the
//! pinned `transformers` commit `get_image_features` applies no scaling, so
//! this graph's output is used as the image tokens directly (see `prefix.rs`).

use anyhow::{Result, ensure};
use rlx_flow::{BuiltModel, CompileProfile, Emit, FlowValue, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};
use rlx_siglip2::VisionEmbedWeights;

use crate::config::VisionConfig;
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;

/// Build [`rlx_siglip2::VisionEmbedWeights`] from the canonical VLASH keys
/// (`vision.embeddings.*`). Consumes the tensors from `wm`.
pub fn extract_vision_embed(wm: &mut WeightMap, cfg: &VisionConfig) -> Result<VisionEmbedWeights> {
    let width = cfg.width;
    let patch_dim = cfg.patch_dim();
    let num_patches = cfg.num_patches();

    let (conv, conv_shape) = wm.take("vision.embeddings.patch_embedding.weight")?;
    ensure!(
        conv_shape.len() == 4
            && conv_shape[0] == width
            && conv_shape[1] * conv_shape[2] * conv_shape[3] == patch_dim,
        "patch_embedding.weight expected [{width},3,ps,ps] (patch_dim={patch_dim}), got {conv_shape:?}"
    );
    // [width, patch_dim] → transpose → [patch_dim, width] (sgemm-friendly).
    let mut patch_w = vec![0f32; width * patch_dim];
    for e in 0..width {
        for d in 0..patch_dim {
            patch_w[d * width + e] = conv[e * patch_dim + d];
        }
    }
    let (patch_b, _) = wm.take("vision.embeddings.patch_embedding.bias")?;
    ensure!(patch_b.len() == width, "patch_embedding.bias != width");
    let (pos_embed, _) = wm.take("vision.embeddings.position_embedding.weight")?;
    ensure!(
        pos_embed.len() == num_patches * width,
        "position_embedding {} != num_patches*width ({num_patches}*{width})",
        pos_embed.len()
    );

    Ok(VisionEmbedWeights {
        patch_w,
        patch_b,
        pos_embed,
        width,
        patch_dim,
        num_patches,
    })
}

/// A `nn.Linear` `x @ Wᵀ + b` under `prefix` (`{prefix}.weight` `[out,in]`).
fn emit_linear(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
    let w = emit.load_param(&format!("{prefix}.weight"), true)?;
    let b = emit.load_param(&format!("{prefix}.bias"), false)?;
    let mut gb = HirMut::new(emit.hir());
    let mm = gb.mm(x, w);
    Ok(gb.add(mm, b))
}

/// One bidirectional pre-LN SigLIP encoder layer (`vision.encoder.layers.{i}`).
fn emit_encoder_layer(
    emit: &mut Emit<'_>,
    prefix: &str,
    width: usize,
    heads: usize,
    eps: f32,
    input: &FlowValue,
) -> Result<FlowValue> {
    let head_dim = width / heads;
    let shape = input.shape.clone();
    let x = input.hir_id();

    // Attention sub-block.
    let ln1_g = emit.load_param(&format!("{prefix}.layer_norm1.weight"), false)?;
    let ln1_b = emit.load_param(&format!("{prefix}.layer_norm1.bias"), false)?;
    let normed1 = {
        let mut gb = HirMut::new(emit.hir());
        gb.ln(x, ln1_g, ln1_b, eps)
    };
    let q = emit_linear(emit, &format!("{prefix}.self_attn.q_proj"), normed1)?;
    let k = emit_linear(emit, &format!("{prefix}.self_attn.k_proj"), normed1)?;
    let v = emit_linear(emit, &format!("{prefix}.self_attn.v_proj"), normed1)?;
    let attn = {
        let mut gb = HirMut::new(emit.hir());
        let attn_shape = rlx_ir::shape::attention_shape(gb.shape(q));
        gb.attention_kind(q, k, v, heads, head_dim, MaskKind::None, attn_shape)
    };
    let attn_out = emit_linear(emit, &format!("{prefix}.self_attn.out_proj"), attn)?;
    let res1 = {
        let mut gb = HirMut::new(emit.hir());
        gb.add(x, attn_out)
    };

    // MLP sub-block.
    let ln2_g = emit.load_param(&format!("{prefix}.layer_norm2.weight"), false)?;
    let ln2_b = emit.load_param(&format!("{prefix}.layer_norm2.bias"), false)?;
    let normed2 = {
        let mut gb = HirMut::new(emit.hir());
        gb.ln(res1, ln2_g, ln2_b, eps)
    };
    let fc1 = emit_linear(emit, &format!("{prefix}.mlp.fc1"), normed2)?;
    let act = {
        let mut gb = HirMut::new(emit.hir());
        gb.gelu_approx(fc1)
    };
    let fc2 = emit_linear(emit, &format!("{prefix}.mlp.fc2"), act)?;
    let out = {
        let mut gb = HirMut::new(emit.hir());
        gb.add(res1, fc2)
    };
    Ok(emit.wrap(out, shape))
}

/// Build the vision-tower flow. Input `"hidden"` is `[batch, 256, width]`
/// (patch projections + position embeddings, assembled on host). Output
/// `"image_features"` is `[batch, 256, projection_dim]` (raw projector output;
/// the `/√hidden` scaling is applied in `prefix.rs`).
pub fn build_vision_flow(
    cfg: &VisionConfig,
    wm: &mut WeightMap,
    batch: usize,
) -> Result<BuiltModel> {
    let width = cfg.width;
    let heads = cfg.heads;
    let seq = cfg.num_patches();
    let eps = cfg.ln_eps;
    let proj = cfg.projection_dim;
    let f = DType::F32;

    let mut flow = ModelFlow::new("vlash_vision")
        .with_profile(CompileProfile::encoder())
        .input("hidden", Shape::new(&[batch, seq, width], f));

    for i in 0..cfg.layers {
        let prefix = format!("vision.encoder.layers.{i}");
        flow = flow.plugin_named(format!("vision.layer{i}"), move |emit, prev| {
            let input = prev.ok_or_else(|| anyhow::anyhow!("vision layer requires hidden"))?;
            let out = emit_encoder_layer(emit, &prefix, width, heads, eps, &input)?;
            Ok(Some(out))
        });
    }

    let post_shape = Shape::new(&[batch, seq, width], f);
    flow = flow.plugin_named("vision.post_ln", move |emit, prev| {
        let encoded = prev.ok_or_else(|| anyhow::anyhow!("post_ln requires hidden"))?;
        let g = emit.load_param("vision.post_layernorm.weight", false)?;
        let b = emit.load_param("vision.post_layernorm.bias", false)?;
        let mut gb = HirMut::new(emit.hir());
        let out = gb.ln(encoded.hir_id(), g, b, eps);
        Ok(Some(emit.wrap(out, post_shape.clone())))
    });

    let proj_shape = Shape::new(&[batch, seq, proj], f);
    flow = flow.plugin_named("vision.projector", move |emit, prev| {
        let x = prev.ok_or_else(|| anyhow::anyhow!("projector requires hidden"))?;
        let out = emit_linear(emit, "vision.projector", x.hir_id())?;
        Ok(Some(emit.wrap(out, proj_shape.clone())))
    });

    flow.output("image_features")
        .build_with(&mut WeightMapSource(wm), None)
}
