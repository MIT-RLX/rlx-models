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

//! Tier-0 BioCLIP-2 flows — native [`ModelFlow`] assembly of the OpenCLIP
//! vision and text transformer towers.
//!
//! Both towers are stacks of pre-LayerNorm CLIP residual blocks:
//! ```text
//!   x = x + out_proj(MHSA(ln_1(x)))
//!   x = x + c_proj(gelu(c_fc(ln_2(x))))
//! ```
//! Vision uses bidirectional attention ([`MaskKind::None`]); text uses
//! causal attention ([`MaskKind::Causal`]). Activation is exact `nn.GELU`
//! (BioCLIP-2 derives from LAION-2B weights, not OpenAI → not QuickGELU).

use anyhow::Result;
use rlx_flow::{BuiltModel, CompileProfile, Emit, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, HirGraphExt, Shape};

use crate::config::{BioClip2Config, LN_EPS};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;

/// Emit one pre-LN CLIP residual block under `prefix` (e.g.
/// `"visual.transformer.resblocks.3"`). `input` is the block input;
/// returns the block output (same shape).
fn emit_clip_resblock(
    emit: &mut Emit<'_>,
    prefix: &str,
    width: usize,
    heads: usize,
    eps: f32,
    mask: MaskKind,
    input: &rlx_flow::FlowValue,
) -> Result<rlx_flow::FlowValue> {
    let head_dim = width / heads;

    let ln1_g = emit.load_param(&format!("{prefix}.ln_1.weight"), false)?;
    let ln1_b = emit.load_param(&format!("{prefix}.ln_1.bias"), false)?;
    // nn.MultiheadAttention packs Q|K|V into `in_proj_weight` [3·w, w].
    let in_w = emit.load_param(&format!("{prefix}.attn.in_proj_weight"), true)?;
    let in_b = emit.load_param(&format!("{prefix}.attn.in_proj_bias"), false)?;
    let out_w = emit.load_param(&format!("{prefix}.attn.out_proj.weight"), true)?;
    let out_b = emit.load_param(&format!("{prefix}.attn.out_proj.bias"), false)?;
    let ln2_g = emit.load_param(&format!("{prefix}.ln_2.weight"), false)?;
    let ln2_b = emit.load_param(&format!("{prefix}.ln_2.bias"), false)?;
    let fc_w = emit.load_param(&format!("{prefix}.mlp.c_fc.weight"), true)?;
    let fc_b = emit.load_param(&format!("{prefix}.mlp.c_fc.bias"), false)?;
    let proj_w = emit.load_param(&format!("{prefix}.mlp.c_proj.weight"), true)?;
    let proj_b = emit.load_param(&format!("{prefix}.mlp.c_proj.bias"), false)?;

    let shape = input.shape.clone();
    let mut gb = HirMut::new(emit.hir());
    let x = input.hir_id();

    // --- Attention sub-block ---
    let normed1 = gb.ln(x, ln1_g, ln1_b, eps);
    let qkv_mm = gb.mm(normed1, in_w);
    let qkv = gb.add(qkv_mm, in_b);
    let last_ax = gb.shape(qkv).rank() - 1;
    let q = gb.narrow_(qkv, last_ax, 0, width);
    let k = gb.narrow_(qkv, last_ax, width, width);
    let v = gb.narrow_(qkv, last_ax, 2 * width, width);
    let attn_shape = rlx_ir::shape::attention_shape(gb.shape(q));
    let attn = gb.attention_kind(q, k, v, heads, head_dim, mask, attn_shape);
    let proj_attn = gb.mm(attn, out_w);
    let attn_out = gb.add(proj_attn, out_b);
    let res1 = gb.add(x, attn_out);

    // --- MLP sub-block ---
    let normed2 = gb.ln(res1, ln2_g, ln2_b, eps);
    let fc_mm = gb.mm(normed2, fc_w);
    let fc = gb.add(fc_mm, fc_b);
    let act = gb.gelu(fc);
    let down_mm = gb.mm(act, proj_w);
    let down = gb.add(down_mm, proj_b);
    let out = gb.add(res1, down);

    Ok(emit.wrap(out, shape))
}

/// Build the vision tower flow. Input `"hidden"` is `[batch, seq, width]`
/// (class + patch tokens + positional embeddings, assembled on host).
/// Output `"image_features"` is `[batch, embed_dim]`.
pub fn build_vision_flow(
    cfg: &BioClip2Config,
    weights: &mut WeightMap,
    batch: usize,
) -> Result<BuiltModel> {
    build_vision_flow_inner(cfg, weights, batch, VisionOutput::ClsProjection)
}

/// Build the vision tower outputting per-patch features instead of the
/// CLS-projected image embedding. Output shape:
/// `[batch, n_patches, width]` (CLS token + register tokens stripped;
/// just the spatial patch grid, ln-post applied).
///
/// Used by `rlx-bioclip2-batch` to mirror the DINOv2 dense-feature
/// pipeline so we can compare like-for-like clustering quality.
pub fn build_vision_features_flow(
    cfg: &BioClip2Config,
    weights: &mut WeightMap,
    batch: usize,
) -> Result<BuiltModel> {
    build_vision_flow_inner(cfg, weights, batch, VisionOutput::PatchTokens)
}

/// Internal vision-tower switch. Same transformer body; the head is
/// what differs between the two graph variants.
#[derive(Copy, Clone)]
enum VisionOutput {
    /// Original BioCLIP-2 image-features path: ln_post → CLS row →
    /// @ visual.proj → `[batch, embed_dim]`.
    ClsProjection,
    /// Dense path: ln_post on every token, drop the CLS row, output
    /// `[batch, n_patches, width]`.
    PatchTokens,
}

fn build_vision_flow_inner(
    cfg: &BioClip2Config,
    weights: &mut WeightMap,
    batch: usize,
    out_kind: VisionOutput,
) -> Result<BuiltModel> {
    let v = cfg.vision;
    let width = v.width;
    let heads = v.heads;
    let seq = v.seq_len();
    let embed_dim = cfg.embed_dim;
    let eps = LN_EPS;
    let f = DType::F32;
    let n_patches = seq - 1; // CLS is row 0; remaining rows are patches.

    let mut flow = ModelFlow::new("bioclip2_vision")
        .with_profile(CompileProfile::encoder())
        .input("hidden", Shape::new(&[batch, seq, width], f));

    // ln_pre.
    let in_shape = Shape::new(&[batch, seq, width], f);
    flow = flow.plugin_named("vision.ln_pre", move |emit, _prev| {
        let x = emit.flow_input("hidden")?;
        let g = emit.load_param("visual.ln_pre.weight", false)?;
        let b = emit.load_param("visual.ln_pre.bias", false)?;
        let mut gb = HirMut::new(emit.hir());
        let out = gb.ln(x.hir_id(), g, b, eps);
        Ok(Some(emit.wrap(out, in_shape.clone())))
    });

    // Transformer resblocks (bidirectional).
    for i in 0..v.layers {
        let prefix = format!("visual.transformer.resblocks.{i}");
        flow = flow.plugin_named(format!("vision.resblock{i}"), move |emit, prev| {
            let input = prev.ok_or_else(|| anyhow::anyhow!("vision resblock requires hidden"))?;
            let out = emit_clip_resblock(emit, &prefix, width, heads, eps, MaskKind::None, &input)?;
            Ok(Some(out))
        });
    }

    // Head differs by `out_kind`. Both variants apply `visual.ln_post`
    // to every token first; the original BioCLIP-2 path then narrows to
    // CLS and projects, the dense path drops CLS and outputs all patch
    // rows directly (the equivalent of DINOv2's `patch_tokens`).
    let output_name = match out_kind {
        VisionOutput::ClsProjection => "image_features",
        VisionOutput::PatchTokens => "patch_features",
    };
    flow = flow.plugin_named("vision.head", move |emit, prev| {
        let encoded = prev.ok_or_else(|| anyhow::anyhow!("vision head requires hidden"))?;
        let ln_g = emit.load_param("visual.ln_post.weight", false)?;
        let ln_b = emit.load_param("visual.ln_post.bias", false)?;
        // Load proj before borrowing `emit` for the HIR builder (only the
        // CLS-projection head needs it).
        let proj = match out_kind {
            VisionOutput::ClsProjection => Some(emit.load_param("visual.proj", false)?),
            VisionOutput::PatchTokens => None,
        };
        let mut gb = HirMut::new(emit.hir());
        let normed = gb.ln(encoded.hir_id(), ln_g, ln_b, eps);
        match out_kind {
            VisionOutput::ClsProjection => {
                let proj = proj.expect("cls projection head loads visual.proj");
                let cls = gb.narrow_(normed, 1, 0, 1);
                let cls_flat = gb.reshape_(cls, vec![batch as i64, width as i64]);
                let out = gb.mm(cls_flat, proj);
                Ok(Some(emit.wrap(out, Shape::new(&[batch, embed_dim], f))))
            }
            VisionOutput::PatchTokens => {
                // Drop CLS (row 0), keep [1..seq] → n_patches rows.
                let patches = gb.narrow_(normed, 1, 1, n_patches);
                Ok(Some(
                    emit.wrap(patches, Shape::new(&[batch, n_patches, width], f)),
                ))
            }
        }
    });

    flow.output(output_name)
        .build_with(&mut WeightMapSource(weights), None)
}

/// Build the text tower flow. Input `"hidden"` is `[batch, ctx, width]`
/// — the token + positional embeddings assembled on host (see
/// [`crate::text_embed`]); keeping the embedding lookup off-graph mirrors
/// the vision conv1 stem and avoids an in-graph integer `Gather`. Output
/// `"text_hidden"` is the full post-`ln_final` sequence `[batch, ctx,
/// width]`; EOT-row selection and `text_projection` are applied on host.
pub fn build_text_flow(
    cfg: &BioClip2Config,
    weights: &mut WeightMap,
    batch: usize,
) -> Result<BuiltModel> {
    let t = cfg.text;
    let width = t.width;
    let heads = t.heads;
    let seq = t.context_length;
    let eps = LN_EPS;
    let f = DType::F32;

    let mut flow = ModelFlow::new("bioclip2_text")
        .with_profile(CompileProfile::encoder())
        .input("hidden", Shape::new(&[batch, seq, width], f));

    // Bind the host-assembled embeddings as the first flow value.
    flow = flow.plugin_named("text.input", move |emit, _prev| {
        let x = emit.flow_input("hidden")?;
        Ok(Some(x))
    });

    // Transformer resblocks (causal).
    for i in 0..t.layers {
        let prefix = format!("transformer.resblocks.{i}");
        flow = flow.plugin_named(format!("text.resblock{i}"), move |emit, prev| {
            let input = prev.ok_or_else(|| anyhow::anyhow!("text resblock requires hidden"))?;
            let out =
                emit_clip_resblock(emit, &prefix, width, heads, eps, MaskKind::Causal, &input)?;
            Ok(Some(out))
        });
    }

    // ln_final over the full sequence.
    flow = flow.plugin_named("text.ln_final", move |emit, prev| {
        let encoded = prev.ok_or_else(|| anyhow::anyhow!("text ln_final requires hidden"))?;
        let g = emit.load_param("ln_final.weight", false)?;
        let b = emit.load_param("ln_final.bias", false)?;
        let mut gb = HirMut::new(emit.hir());
        let out = gb.ln(encoded.hir_id(), g, b, eps);
        Ok(Some(emit.wrap(out, Shape::new(&[batch, seq, width], f))))
    });

    flow.output("text_hidden")
        .build_with(&mut WeightMapSource(weights), None)
}
