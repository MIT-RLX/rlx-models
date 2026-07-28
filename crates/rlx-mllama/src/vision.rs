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

//! mllama vision tower + multi-modal projector as a native [`ModelFlow`].
//!
//! The graph takes an already-assembled `hidden` `[1, seq, width]`
//! (patch-embed + pre-tile + class token + gated-position embeddings, computed
//! host-side in [`crate::preprocess`]) plus the `post_tile` positional
//! additive tensor `[1, seq, width]`, and produces `cross_states`
//! `[1, seq, text_hidden]` — the vision features already projected into the
//! text embedding space, ready to feed the language model's cross-attention.
//!
//! `seq = num_tiles * num_patches`. We run with the image's *actual* tile
//! count and no padding: in HF all tile / alignment padding is fully masked
//! out, so an exact-tile run with [`MaskKind::None`] is numerically
//! equivalent while avoiding the aspect-ratio attention mask entirely.
//!
//! Structure (mirrors `MllamaVisionModel.forward` + `multi_modal_projector`):
//! ```text
//!   x = layernorm_pre(hidden)
//!   for l in 0..32:  x = local_layer[l](x)          # non-gated; tap [3,7,15,23,30]
//!   x = layernorm_post(x)
//!   x = x + post_tile
//!   for l in 0..8:   x = global_layer[l](x)          # tanh-gated attn + ffn
//!   feats = concat(x, interleaved_stack(taps))        # [.., width*(1+5)] = vision_output_dim
//!   cross_states = feats @ projᵀ + proj_bias          # [.., text_hidden]
//! ```

use anyhow::{Result, anyhow};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, Emit, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::config::MllamaVisionConfig;

/// Linear `x @ Wᵀ` (no bias). `{prefix}.weight` is HF `[out, in]`.
fn linear_nobias(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
    let w = emit.load_param(&format!("{prefix}.weight"), true)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.mm(x, w))
}

/// Linear `x @ Wᵀ + b`. `{prefix}.weight` `[out,in]`, `{prefix}.bias` `[out]`.
fn linear_bias(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
    let w = emit.load_param(&format!("{prefix}.weight"), true)?;
    let b = emit.load_param(&format!("{prefix}.bias"), false)?;
    let mut gb = HirMut::new(emit.hir());
    let mm = gb.mm(x, w);
    Ok(gb.add(mm, b))
}

/// LayerNorm with learned weight+bias under `{prefix}.{weight,bias}`.
fn layer_norm(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId, eps: f32) -> Result<HirNodeId> {
    let g = emit.load_param(&format!("{prefix}.weight"), false)?;
    let b = emit.load_param(&format!("{prefix}.bias"), false)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.ln(x, g, b, eps))
}

/// One vision encoder layer (`MllamaVisionEncoderLayer`).
///
/// `gated` selects the global-transformer layers, which multiply the attention
/// and FFN sub-block outputs by `tanh(gate_attn)` / `tanh(gate_ffn)` before the
/// residual add. Attention projections have no bias; the MLP `fc1`/`fc2` do.
fn vision_layer(
    emit: &mut Emit<'_>,
    prefix: &str,
    x: HirNodeId,
    seq: usize,
    width: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
    gated: bool,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let attn_shape = Shape::new(&[1, seq, width], f);

    // --- Self-attention sub-block (pre-norm) ---
    let normed = layer_norm(emit, &format!("{prefix}.input_layernorm"), x, eps)?;
    let q = linear_nobias(emit, &format!("{prefix}.self_attn.q_proj"), normed)?;
    let k = linear_nobias(emit, &format!("{prefix}.self_attn.k_proj"), normed)?;
    let v = linear_nobias(emit, &format!("{prefix}.self_attn.v_proj"), normed)?;
    let attn = {
        let mut gb = HirMut::new(emit.hir());
        gb.attention_kind(q, k, v, heads, head_dim, MaskKind::None, attn_shape.clone())
    };
    let mut attn = linear_nobias(emit, &format!("{prefix}.self_attn.o_proj"), attn)?;
    if gated {
        let gate = emit.load_param(&format!("{prefix}.gate_attn"), false)?;
        let mut gb = HirMut::new(emit.hir());
        let g = gb.tanh(gate);
        attn = gb.mul(attn, g);
    }
    let x = {
        let mut gb = HirMut::new(emit.hir());
        gb.add(x, attn)
    };

    // --- MLP sub-block (pre-norm) ---
    let normed = layer_norm(emit, &format!("{prefix}.post_attention_layernorm"), x, eps)?;
    let fc1 = linear_bias(emit, &format!("{prefix}.mlp.fc1"), normed)?;
    let act = {
        let mut gb = HirMut::new(emit.hir());
        gb.gelu(fc1) // mllama vision uses exact (erf) GELU
    };
    let mut fc2 = linear_bias(emit, &format!("{prefix}.mlp.fc2"), act)?;
    if gated {
        let gate = emit.load_param(&format!("{prefix}.gate_ffn"), false)?;
        let mut gb = HirMut::new(emit.hir());
        let g = gb.tanh(gate);
        fc2 = gb.mul(fc2, g);
    }
    let out = {
        let mut gb = HirMut::new(emit.hir());
        gb.add(x, fc2)
    };
    Ok(out)
}

/// Build the mllama vision + projector flow for a fixed sequence length
/// `seq = num_tiles * num_patches`. Weights are the mllama safetensors under
/// the `vision_model.*` and `multi_modal_projector.*` prefixes.
pub fn build_vision_flow(
    cfg: &MllamaVisionConfig,
    weights: &mut WeightMap,
    text_hidden: usize,
    num_tiles: usize,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let width = cfg.hidden_size;
    let heads = cfg.num_attention_heads;
    let head_dim = cfg.head_dim();
    let np = cfg.num_patches();
    let seq = num_tiles * np;
    let eps = cfg.norm_eps;
    let n_local = cfg.num_hidden_layers;
    let n_global = cfg.num_global_layers;
    let taps: Vec<usize> = cfg.intermediate_layers_indices.clone();
    let concat_width = cfg.concat_width();

    let flow = ModelFlow::new("mllama_vision")
        .with_profile(CompileProfile::encoder())
        .input("hidden", Shape::new(&[1, seq, width], f))
        .input("post_tile", Shape::new(&[1, seq, width], f));

    let flow = flow.plugin_named("vision.encoder", move |emit, _prev| {
        let hidden = emit.flow_input("hidden")?.hir_id();
        let post_tile = emit.flow_input("post_tile")?.hir_id();

        // layernorm_pre → local transformer (collecting intermediate taps)
        let mut x = layer_norm(emit, "vision_model.layernorm_pre", hidden, eps)?;
        let mut tapped: Vec<HirNodeId> = Vec::with_capacity(taps.len());
        for l in 0..n_local {
            let prefix = format!("vision_model.transformer.layers.{l}");
            x = vision_layer(emit, &prefix, x, seq, width, heads, head_dim, eps, false)?;
            if taps.contains(&l) {
                tapped.push(x);
            }
        }
        if tapped.len() != taps.len() {
            return Err(anyhow!(
                "collected {} intermediate taps, expected {}",
                tapped.len(),
                taps.len()
            ));
        }

        // layernorm_post → + post-tile positional → global transformer
        x = layer_norm(emit, "vision_model.layernorm_post", x, eps)?;
        x = {
            let mut gb = HirMut::new(emit.hir());
            gb.add(x, post_tile)
        };
        for l in 0..n_global {
            let prefix = format!("vision_model.global_transformer.layers.{l}");
            x = vision_layer(emit, &prefix, x, seq, width, heads, head_dim, eps, true)?;
        }

        // Interleave the taps (feature-major, layer-minor) exactly like HF's
        // `torch.stack(taps, dim=-1).reshape(.., width*n_taps)`, then
        // concat the final hidden in front: [final | interleaved].
        let n_taps = tapped.len();
        let feats = {
            let mut gb = HirMut::new(emit.hir());
            let stacked: Vec<HirNodeId> = tapped
                .iter()
                .map(|&t| gb.reshape_(t, vec![1, seq as i64, width as i64, 1]))
                .collect();
            let inter4 = gb.concat_(stacked, 3); // [1, seq, width, n_taps]
            let inter = gb.reshape_(inter4, vec![1, seq as i64, (width * n_taps) as i64]);
            gb.concat_(vec![x, inter], 2) // [1, seq, width*(1+n_taps)]
        };

        // multi_modal_projector: Linear(concat_width, text_hidden, bias=True)
        let cross = linear_bias(emit, "multi_modal_projector", feats)?;
        let shape = Shape::new(&[1, seq, text_hidden], f);
        Ok(Some(emit.wrap(cross, shape)))
    });

    let _ = concat_width; // documented invariant; projector weight enforces it
    flow.output("cross_states")
        .build_with(&mut WeightMapSource(weights), None)
}
