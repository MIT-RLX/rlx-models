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

//! MiniMax-M3 CLIP-style vision tower (`vision_tower.vision_model`) + the
//! spatial-merge multimodal projector.
//!
//! The Conv3d patch embed is expressed as a matmul over pre-patchified
//! `pixel_values [num_patches, patch_dim]` (`patch_dim = C·temporal·patch²`),
//! matching the flattened conv weight. Each encoder layer is CLIP-standard —
//! `LN1 → biased q/k/v/out attn (3D RoPE on Q/K, non-causal) → +res → LN2 →
//! fc1/gelu/fc2 → +res` — with cos/sin fed as graph inputs. The projector maps
//! `1280 → text hidden 6144`, grouping `spatial_merge_size²` patches into the
//! channel dim between its two GELU MLPs.

use anyhow::{Result, anyhow};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, Emit, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, HirGraphExt, Shape};

use super::config::M3VisionConfig;

/// Graph-input keys for the vision 3D-RoPE tables (`[num_patches, rot_dim/2]`).
pub const VCOS: &str = "vcos";
pub const VSIN: &str = "vsin";

/// Build the vision 3D-RoPE cos/sin tables `[num_patches, 3·axis_dim/2]` for a
/// single image grid `(t, h, w)` in raster order. `inv_freq[j] = theta^(-2j/axis_dim)`;
/// each patch's `(t,h,w)` coordinates scale their own axis band, concatenated T|H|W.
pub fn vision_rope_tables(
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
    axis_dim: usize,
    theta: f64,
) -> (Vec<f32>, Vec<f32>) {
    let per = axis_dim / 2; // freqs per axis
    let half = 3 * per; // total freqs
    let np = grid_t * grid_h * grid_w;
    let mut cos = vec![0.0f32; np * half];
    let mut sin = vec![0.0f32; np * half];
    let inv: Vec<f64> = (0..per)
        .map(|j| theta.powf(-(2.0 * j as f64) / axis_dim as f64))
        .collect();
    let mut p = 0usize;
    for t in 0..grid_t {
        for h in 0..grid_h {
            for w in 0..grid_w {
                let coords = [t as f64, h as f64, w as f64];
                for (a, &c) in coords.iter().enumerate() {
                    for j in 0..per {
                        let ang = c * inv[j];
                        let idx = p * half + a * per + j;
                        cos[idx] = ang.cos() as f32;
                        sin[idx] = ang.sin() as f32;
                    }
                }
                p += 1;
            }
        }
    }
    (cos, sin)
}

fn linear_bias(
    emit: &mut Emit<'_>,
    prefix: &str,
    x: rlx_ir::HirNodeId,
) -> Result<rlx_ir::HirNodeId> {
    let w = emit.load_param(&format!("{prefix}.weight"), true)?;
    let b = emit.load_param(&format!("{prefix}.bias"), false)?;
    let mut gb = HirMut::new(emit.hir());
    let m = gb.mm(x, w);
    Ok(gb.add(m, b))
}

fn layernorm(
    emit: &mut Emit<'_>,
    prefix: &str,
    x: rlx_ir::HirNodeId,
    eps: f32,
) -> Result<rlx_ir::HirNodeId> {
    let g = emit.load_param(&format!("{prefix}.weight"), false)?;
    let b = emit.load_param(&format!("{prefix}.bias"), false)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.ln(x, g, b, eps))
}

/// Build the vision tower prefill graph over `num_patches`, output
/// `vision_hidden [1, num_patches, hidden]`.
pub fn build_m3_vision_flow(
    cfg: &M3VisionConfig,
    weights: &mut WeightMap,
    num_patches: usize,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let np = num_patches;
    let embed = cfg.hidden_size;
    let heads = cfg.num_attention_heads;
    let hd = cfg.head_dim();
    let rot = cfg.rot_dim();
    let rot_half = cfg.rot_dim() / 2;
    let eps = cfg.layer_norm_eps;
    let inter = cfg.intermediate_size;
    let patch_dim = cfg.patch_dim();

    let mut flow = ModelFlow::new("minimax_m3_vision")
        .with_profile(CompileProfile::encoder())
        .input("pixel_values", Shape::new(&[np, patch_dim], f))
        .input(VCOS, Shape::new(&[np, rot_half], f))
        .input(VSIN, Shape::new(&[np, rot_half], f));

    // Patch embed (conv-as-matmul) + pre_layrnorm.
    flow = flow.plugin_named("patch_embed", move |emit, _prev| {
        let px = emit.flow_input("pixel_values")?.hir_id();
        let pw = emit.load_param(
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            true,
        )?;
        let h = {
            let mut gb = HirMut::new(emit.hir());
            let e = gb.mm(px, pw); // [np, embed]
            gb.reshape_(e, vec![1, np as i64, embed as i64])
        };
        let h = layernorm(emit, "vision_tower.vision_model.pre_layrnorm", h, eps)?;
        Ok(Some(emit.wrap(h, Shape::new(&[1, np, embed], f))))
    });

    let hs = Shape::new(&[1, np, embed], f);
    for l in 0..cfg.num_hidden_layers {
        let lp = format!("vision_tower.vision_model.encoder.layers.{l}");
        let hs = hs.clone();
        flow = flow.plugin_named(format!("vlayer{l}"), move |emit, prev| {
            let x = prev.ok_or_else(|| anyhow!("vlayer needs input"))?.hir_id();
            // Attention block.
            let normed = layernorm(emit, &format!("{lp}.layer_norm1"), x, eps)?;
            let q = linear_bias(emit, &format!("{lp}.self_attn.q_proj"), normed)?;
            let k = linear_bias(emit, &format!("{lp}.self_attn.k_proj"), normed)?;
            let v = linear_bias(emit, &format!("{lp}.self_attn.v_proj"), normed)?;
            let cos = emit.flow_input(VCOS)?.hir_id();
            let sin = emit.flow_input(VSIN)?.hir_id();
            let attn = {
                let mut gb = HirMut::new(emit.hir());
                let q = gb.rope_n(q, cos, sin, hd, rot);
                let k = gb.rope_n(k, cos, sin, hd, rot);
                gb.attention_kind(
                    q,
                    k,
                    v,
                    heads,
                    hd,
                    MaskKind::None,
                    Shape::new(&[1, np, embed], f),
                )
            };
            let attn_out = linear_bias(emit, &format!("{lp}.self_attn.out_proj"), attn)?;
            let x = {
                let mut gb = HirMut::new(emit.hir());
                gb.add(x, attn_out)
            };
            // MLP block.
            let normed2 = layernorm(emit, &format!("{lp}.layer_norm2"), x, eps)?;
            let fc1 = linear_bias(emit, &format!("{lp}.mlp.fc1"), normed2)?;
            let act = {
                let mut gb = HirMut::new(emit.hir());
                gb.gelu(fc1)
            };
            let fc2 = linear_bias(emit, &format!("{lp}.mlp.fc2"), act)?;
            let out = {
                let mut gb = HirMut::new(emit.hir());
                gb.add(x, fc2)
            };
            Ok(Some(emit.wrap(out, hs.clone())))
        });
    }

    let _ = inter;
    let flow = flow.output("vision_hidden");
    flow.build_with(&mut WeightMapSource(weights), None)
}

/// Emit the multimodal projector: `[np, embed] → [np/merge², text_hidden]`.
/// `multi_modal_projector` (linear_1→gelu→linear_2), reshape grouping
/// `merge²` patches into channels, then `patch_merge_mlp` (linear_1→gelu→linear_2).
pub fn build_m3_projector_flow(
    cfg: &M3VisionConfig,
    weights: &mut WeightMap,
    num_patches: usize,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let np = num_patches;
    let embed = cfg.hidden_size;
    let ph = cfg.projector_hidden_size;
    let text = cfg.projection_dim;
    let merge2 = cfg.spatial_merge_size * cfg.spatial_merge_size;
    let np_out = np / merge2;

    let mut flow = ModelFlow::new("minimax_m3_projector")
        .with_profile(CompileProfile::encoder())
        .input("vision_hidden", Shape::new(&[np, embed], f));

    flow = flow.plugin_named("projector", move |emit, _prev| {
        let x = emit.flow_input("vision_hidden")?.hir_id();
        // multi_modal_projector: linear_1 → gelu → linear_2 → [np, ph].
        let h = linear_bias(emit, "multi_modal_projector.linear_1", x)?;
        let h = {
            let mut gb = HirMut::new(emit.hir());
            gb.gelu(h)
        };
        let h = linear_bias(emit, "multi_modal_projector.linear_2", h)?;
        // Group merge² patches into channels: [np, ph] → [np_out, ph·merge²].
        let h = {
            let mut gb = HirMut::new(emit.hir());
            gb.reshape_(h, vec![np_out as i64, (ph * merge2) as i64])
        };
        // patch_merge_mlp: linear_1 → gelu → linear_2 → [np_out, text].
        let h = linear_bias(emit, "patch_merge_mlp.linear_1", h)?;
        let h = {
            let mut gb = HirMut::new(emit.hir());
            gb.gelu(h)
        };
        let h = linear_bias(emit, "patch_merge_mlp.linear_2", h)?;
        Ok(Some(emit.wrap(h, Shape::new(&[np_out, text], f))))
    });

    let flow = flow.output("image_features");
    flow.build_with(&mut WeightMapSource(weights), None)
}
