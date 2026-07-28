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

//! Llama-4 vision tower + adapter + multi-modal projector as a native flow.
//!
//! Input `hidden [1, num_patches, hidden]` (host: unfold patch-embed + class
//! token appended last + learned position embedding) and the 2D-axial RoPE
//! `cos/sin [num_patches, head_dim/2]`. Graph:
//! ```text
//!   x = layernorm_pre(hidden)
//!   for l in 0..N: x = enc_layer[l](x)           # biased q/k/v/o, GptJ 2D-rope, full attn, GELU MLP
//!   x = layernorm_post(x)[:, :-1, :]              # drop the class token
//!   x = pixel_shuffle(x, ratio)                    # [1, N', hidden/ratio²]
//!   x = mlp2(x)                                     # fc1 → GELU → fc2 → GELU
//!   image_features = x @ projector.linear_1ᵀ       # → [1, N', text_hidden]
//! ```

use anyhow::Result;
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, Emit, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, HirGraphExt, HirNodeId, RopeStyle, Shape};

use crate::config::Llama4VisionConfig;

pub const V_ROPE_COS: &str = "v_rope_cos";
pub const V_ROPE_SIN: &str = "v_rope_sin";

fn linear_bias(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
    let w = emit.load_param(&format!("{prefix}.weight"), true)?;
    let b = emit.load_param(&format!("{prefix}.bias"), false)?;
    let mut gb = HirMut::new(emit.hir());
    let mm = gb.mm(x, w);
    Ok(gb.add(mm, b))
}

fn linear_nobias(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
    let w = emit.load_param(&format!("{prefix}.weight"), true)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.mm(x, w))
}

fn layer_norm(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId, eps: f32) -> Result<HirNodeId> {
    let g = emit.load_param(&format!("{prefix}.weight"), false)?;
    let b = emit.load_param(&format!("{prefix}.bias"), false)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.ln(x, g, b, eps))
}

fn enc_layer(
    emit: &mut Emit<'_>,
    prefix: &str,
    x: HirNodeId,
    n: usize,
    hidden: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
    cos: HirNodeId,
    sin: HirNodeId,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let normed = layer_norm(emit, &format!("{prefix}.input_layernorm"), x, eps)?;
    let q = linear_bias(emit, &format!("{prefix}.self_attn.q_proj"), normed)?;
    let k = linear_bias(emit, &format!("{prefix}.self_attn.k_proj"), normed)?;
    let v = linear_bias(emit, &format!("{prefix}.self_attn.v_proj"), normed)?;
    let attn = {
        let mut gb = HirMut::new(emit.hir());
        let qr = gb.rope_styled(q, cos, sin, head_dim, RopeStyle::GptJ);
        let kr = gb.rope_styled(k, cos, sin, head_dim, RopeStyle::GptJ);
        gb.attention_kind(
            qr,
            kr,
            v,
            heads,
            head_dim,
            MaskKind::None,
            Shape::new(&[1, n, hidden], f),
        )
    };
    let attn = linear_bias(emit, &format!("{prefix}.self_attn.o_proj"), attn)?;
    let x = {
        let mut gb = HirMut::new(emit.hir());
        gb.add(x, attn)
    };
    let normed = layer_norm(emit, &format!("{prefix}.post_attention_layernorm"), x, eps)?;
    let fc1 = linear_bias(emit, &format!("{prefix}.mlp.fc1"), normed)?;
    let act = {
        let mut gb = HirMut::new(emit.hir());
        gb.gelu(fc1)
    };
    let fc2 = linear_bias(emit, &format!("{prefix}.mlp.fc2"), act)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.add(x, fc2))
}

/// Build the vision tower + adapter + projector. `num_patches` includes the
/// class token; the output has `(num_patches-1) * ratio²` tokens of width
/// `text_hidden`.
pub fn build_llama4_vision_flow(
    cfg: &Llama4VisionConfig,
    weights: &mut WeightMap,
    text_hidden: usize,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let hidden = cfg.hidden_size;
    let heads = cfg.num_attention_heads;
    let head_dim = cfg.head_dim();
    let half = head_dim / 2;
    let eps = cfg.norm_eps;
    let np = cfg.num_patches();
    let patches = np - 1; // without class token
    let ratio = cfg.pixel_shuffle_ratio;
    let ps = (patches as f64).sqrt() as usize; // grid side (32)
    let ps_out = (ps as f64 * ratio as f64) as usize; // 16
    let n_out = ps_out * ps_out; // 256
    let c_shuf = ((hidden as f64) / (ratio as f64 * ratio as f64)) as usize; // hidden*4
    let n_layers = cfg.num_hidden_layers;

    let flow = ModelFlow::new("llama4_vision")
        .with_profile(CompileProfile::encoder())
        .input("hidden", Shape::new(&[1, np, hidden], f))
        .input(V_ROPE_COS, Shape::new(&[np, half], f))
        .input(V_ROPE_SIN, Shape::new(&[np, half], f));

    let flow = flow.plugin_named("vision", move |emit, _prev| {
        let hidden_in = emit.flow_input("hidden")?.hir_id();
        let cos = emit.flow_input(V_ROPE_COS)?.hir_id();
        let sin = emit.flow_input(V_ROPE_SIN)?.hir_id();

        let mut x = layer_norm(emit, "vision_model.layernorm_pre", hidden_in, eps)?;
        for l in 0..n_layers {
            let prefix = format!("vision_model.model.layers.{l}");
            x = enc_layer(emit, &prefix, x, np, hidden, heads, head_dim, eps, cos, sin)?;
        }
        x = layer_norm(emit, "vision_model.layernorm_post", x, eps)?;

        // Drop the class token, then pixel-shuffle [1, patches, hidden] → [1, n_out, c_shuf].
        let shuffled = {
            let mut gb = HirMut::new(emit.hir());
            let x = gb.narrow_(x, 1, 0, patches); // [1, patches, hidden]
            let x = gb.reshape_(x, vec![1, ps as i64, ps as i64, hidden as i64]);
            let x = gb.reshape_(
                x,
                vec![1, ps as i64, ps_out as i64, (hidden * ps / ps_out) as i64],
            );
            let x = gb.transpose_(x, vec![0, 2, 1, 3]);
            let x = gb.reshape_(x, vec![1, ps_out as i64, ps_out as i64, c_shuf as i64]);
            let x = gb.transpose_(x, vec![0, 2, 1, 3]);
            gb.reshape_(x, vec![1, n_out as i64, c_shuf as i64])
        };

        // Adapter MLP2 (no bias): fc1 → GELU → fc2 → GELU.
        let fc1 = linear_nobias(emit, "vision_model.vision_adapter.mlp.fc1", shuffled)?;
        let a1 = {
            let mut gb = HirMut::new(emit.hir());
            gb.gelu(fc1)
        };
        let fc2 = linear_nobias(emit, "vision_model.vision_adapter.mlp.fc2", a1)?;
        let a2 = {
            let mut gb = HirMut::new(emit.hir());
            gb.gelu(fc2)
        };

        // Multi-modal projector into the text embedding space.
        let feats = linear_nobias(emit, "multi_modal_projector.linear_1", a2)?;
        Ok(Some(
            emit.wrap(feats, Shape::new(&[1, n_out, text_hidden], f)),
        ))
    });

    flow.output("image_features")
        .build_with(&mut WeightMapSource(weights), None)
}
