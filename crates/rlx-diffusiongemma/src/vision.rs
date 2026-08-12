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

//! The Gemma 4 vision tower (`model_type = "gemma4_vision"`) and the projector
//! that lifts its soft tokens into the language model's space.
//!
//! Pipeline: variable-resolution patches → patch embed → 27 bidirectional
//! encoder layers → 3×3 average pool → `sqrt(hidden)` scale → standardize →
//! project to the LM width. The result is the 280 soft tokens per image the
//! text encoder splices in at `image_token_id` positions.
//!
//! Two things differ from the text tower:
//!
//! * **2-D RoPE.** Each 72-wide head is split in half: the low 36 channels
//!   rotate by the patch's *x* position, the high 36 by its *y*. Within each
//!   half it is an ordinary NeoX rotate-half over 18 frequency slots, so the
//!   rotation is built explicitly as `x·cos + [-x_hi, x_lo, -y_hi, y_lo]·sin`
//!   rather than by calling the packed RoPE op (which would pair channel `j`
//!   with `j+36` and mix the two axes).
//! * **Position embeddings are looked up, not computed** — a learned
//!   `[2, position_embedding_size, hidden]` table indexed by (x, y) and summed.
//!
//! Patches are `[0,1]` pixels; the tower rescales them to `[-1,1]` itself
//! (`do_normalize: false` in the processor config, mean 0 / std 1).

use anyhow::{Result, anyhow};
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, Emit, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::op::{Activation, MaskKind, Op};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::config::{DiffusionGemmaConfig, VisionConfig};
use crate::flow::SharedWeights;

/// Vision tower prefix in a DiffusionGemma checkpoint. (Stock Gemma 4 drops the
/// `encoder.` segment; DiffusionGemma nests the tower under its encoder.)
pub const VISION_PREFIX: &str = "model.encoder.vision_tower";
/// Vision → LM projector.
pub const VISION_PROJ_PREFIX: &str = "model.encoder.embed_vision";

/// Graph input: flattened patch pixels in `[0,1]`, `[1, patches, 3·patch²]`.
pub const PIXELS_INPUT: &str = "vision_pixels";
/// Graph input: per-patch x index into the position table, `[1, patches]`.
pub const POS_X_INPUT: &str = "vision_pos_x";
/// Graph input: per-patch y index, `[1, patches]`.
pub const POS_Y_INPUT: &str = "vision_pos_y";
/// Graph input: 2-D RoPE cos table `[1, patches, head_dim]` (see
/// [`vision_rope_tables`]).
pub const ROPE_COS_INPUT: &str = "vision_rope_cos";
/// Graph input: 2-D RoPE sin table `[1, patches, head_dim]`.
pub const ROPE_SIN_INPUT: &str = "vision_rope_sin";
/// Graph input: `1.0` for a real patch, `0.0` for padding, `[1, patches]`.
pub const VALID_INPUT: &str = "vision_valid";
/// Graph input: average-pooling matrix `[soft_tokens, patches]` (see
/// [`vision_pool_matrix`]).
pub const POOL_INPUT: &str = "vision_pool";
/// Graph output: soft tokens in LM space, `[1, soft_tokens, text_hidden]`.
pub const SOFT_TOKENS_OUTPUT: &str = "soft_tokens";
/// Side output: patch embeddings before the encoder, `[1, patches, hidden]`.
pub const PATCH_EMBED_TAP: &str = "vision_patch_embed";
/// Side output: encoder last hidden state, `[1, patches, hidden]`.
pub const ENCODER_TAP: &str = "vision_encoder_out";
/// Side output: pooled + standardized features, `[soft_tokens, hidden]`.
pub const POOLED_TAP: &str = "vision_pooled";

/// Per-axis inverse frequencies: `spatial = head_dim/2`, stepping by 2, so a
/// 72-wide head yields 18 frequencies per axis.
pub fn vision_inv_freq(cfg: &VisionConfig) -> Vec<f64> {
    let spatial = cfg.head_dim / 2;
    let theta = cfg.rope_theta();
    (0..spatial)
        .step_by(2)
        .map(|i| 1.0 / theta.powf(i as f64 / spatial as f64))
        .collect()
}

/// 2-D RoPE tables `[patches, head_dim]` (flattened) for patch grid positions.
///
/// Each row is `[cos(x·f)⊗2 (36) | cos(y·f)⊗2 (36)]`, matching HF
/// `apply_multidimensional_rope`, which applies `cat(freqs, freqs)` per axis.
pub fn vision_rope_tables(cfg: &VisionConfig, positions: &[(u32, u32)]) -> (Vec<f32>, Vec<f32>) {
    let inv = vision_inv_freq(cfg);
    let hd = cfg.head_dim;
    let half = hd / 2;
    let nf = inv.len();
    let p = positions.len();
    let mut cos = vec![0f32; p * hd];
    let mut sin = vec![0f32; p * hd];
    for (pi, &(x, y)) in positions.iter().enumerate() {
        let base = pi * hd;
        for (axis, coord) in [x, y].into_iter().enumerate() {
            let off = base + axis * half;
            for (j, f) in inv.iter().enumerate() {
                let a = coord as f64 * f;
                let (c, s) = (a.cos() as f32, a.sin() as f32);
                cos[off + j] = c;
                cos[off + nf + j] = c;
                sin[off + j] = s;
                sin[off + nf + j] = s;
            }
        }
    }
    (cos, sin)
}

/// Average-pooling matrix `[out_len, patches]` for a `k×k` patch grid pool.
///
/// Mirrors HF `_avg_pool_by_positions`: patches are bucketed by
/// `(x/k) + (max_x/k)·(y/k)` and each bucket averages its `k²` members.
pub fn vision_pool_matrix(positions: &[(u32, u32)], k: usize, out_len: usize) -> Vec<f32> {
    let p = positions.len();
    let mut m = vec![0f32; out_len * p];
    if k == 0 {
        return m;
    }
    let max_x = positions.iter().map(|&(x, _)| x).max().unwrap_or(0) + 1;
    let cols = (max_x as usize) / k;
    let w = 1.0 / (k * k) as f32;
    for (pi, &(x, y)) in positions.iter().enumerate() {
        let bucket = (x as usize / k) + cols * (y as usize / k);
        if bucket < out_len {
            m[bucket * p + pi] = w;
        }
    }
    m
}

/// Patch grid positions for a `width × height` patch grid, row-major.
pub fn grid_positions(cols: usize, rows: usize) -> Vec<(u32, u32)> {
    (0..rows)
        .flat_map(|y| (0..cols).map(move |x| (x as u32, y as u32)))
        .collect()
}

/// Build the encoder's `inputs_embeds` for a multimodal prompt.
///
/// Text tokens are looked up and scaled by `sqrt(hidden)`; every position whose
/// id is `image_token_id` is overwritten by the next vision soft token, left
/// unscaled. That ordering matters — HF scales the token embeddings first and
/// then `masked_scatter`s the projected image features over them, so the soft
/// tokens must *not* pick up the embedding scale.
///
/// `soft_tokens` is the `[n_soft, hidden]` block from [`build_vision_flow`].
pub fn merge_multimodal_embeds(
    cfg: &DiffusionGemmaConfig,
    weights: &WeightMap,
    ids: &[u32],
    soft_tokens: &[f32],
) -> Result<Vec<f32>> {
    let t = &cfg.text_config;
    let hidden = t.hidden_size;
    let (table, shape) = weights
        .get(crate::flow::EMBED_KEY)
        .ok_or_else(|| anyhow!("checkpoint is missing {}", crate::flow::EMBED_KEY))?;
    anyhow::ensure!(
        shape == [t.vocab_size, hidden],
        "{}: expected [{}, {hidden}], got {shape:?}",
        crate::flow::EMBED_KEY,
        t.vocab_size
    );
    anyhow::ensure!(
        soft_tokens.len().is_multiple_of(hidden),
        "soft_tokens length {} is not a multiple of hidden {hidden}",
        soft_tokens.len()
    );
    let n_soft = soft_tokens.len() / hidden;
    let n_slots = ids.iter().filter(|&&i| i == cfg.image_token_id).count();
    anyhow::ensure!(
        n_slots == n_soft,
        "prompt has {n_slots} image-token slots but {n_soft} soft tokens were supplied"
    );

    let scale = t.embed_scale();
    let mut out = vec![0f32; ids.len() * hidden];
    let mut next_soft = 0usize;
    for (pos, &id) in ids.iter().enumerate() {
        let dst = &mut out[pos * hidden..(pos + 1) * hidden];
        if id == cfg.image_token_id {
            dst.copy_from_slice(&soft_tokens[next_soft * hidden..(next_soft + 1) * hidden]);
            next_soft += 1;
        } else {
            let row = id as usize;
            anyhow::ensure!(row < t.vocab_size, "token id {row} out of range");
            for (d, s) in dst.iter_mut().zip(&table[row * hidden..(row + 1) * hidden]) {
                *d = s * scale;
            }
        }
    }
    Ok(out)
}

fn rms(emit: &mut Emit<'_>, key: &str, x: HirNodeId, dim: usize, eps: f32) -> Result<HirNodeId> {
    let gamma = emit.load_param(&format!("{key}.weight"), false)?;
    let beta = emit.synth_param(
        &format!("{key}.beta"),
        vec![0.0; dim],
        Shape::new(&[dim], DType::F32),
    );
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.rms_norm(x, gamma, beta, eps))
}

fn rms_no_scale(emit: &mut Emit<'_>, tag: &str, x: HirNodeId, dim: usize, eps: f32) -> HirNodeId {
    let ones = emit.synth_param(
        &format!("{tag}.ones"),
        vec![1.0; dim],
        Shape::new(&[dim], DType::F32),
    );
    let zeros = emit.synth_param(
        &format!("{tag}.zeros"),
        vec![0.0; dim],
        Shape::new(&[dim], DType::F32),
    );
    let mut gb = HirMut::new(emit.hir());
    gb.rms_norm(x, ones, zeros, eps)
}

/// `Gemma4ClippableLinear` with `use_clipped_linears: false` — the weight lives
/// one level down, under `.linear.weight`.
fn clipped_linear(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
    let w = emit.load_param(&format!("{prefix}.linear.weight"), true)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.mm(x, w))
}

/// 2-D RoPE on a packed `[1, patches, heads·head_dim]` tensor.
///
/// `out = x·cos + [-x_hi, x_lo, -y_hi, y_lo]·sin`, i.e. an independent NeoX
/// rotate-half within each 36-wide axis block.
fn apply_2d_rope(
    gb: &mut HirMut<'_>,
    x: HirNodeId,
    cos: HirNodeId,
    sin: HirNodeId,
    patches: usize,
    heads: usize,
    hd: usize,
) -> HirNodeId {
    let (p, h, d) = (patches as i64, heads as i64, hd as i64);
    let quarter = hd / 4;
    let half = hd / 2;
    let x4 = gb.reshape_(x, vec![1, p, h, d]);
    // cos/sin are per-patch, shared across heads.
    let cos4 = gb.reshape_(cos, vec![1, p, 1, d]);
    let sin4 = gb.reshape_(sin, vec![1, p, 1, d]);

    let seg = |gb: &mut HirMut<'_>, start: usize| gb.narrow_(x4, 3, start, quarter);
    let x_lo = seg(gb, 0);
    let x_hi = seg(gb, quarter);
    let y_lo = seg(gb, half);
    let y_hi = seg(gb, half + quarter);
    let shape = Shape::new(&[1, patches, heads, quarter], DType::F32);
    let nx_hi = gb.add_node(Op::Activation(Activation::Neg), vec![x_hi], shape.clone());
    let ny_hi = gb.add_node(Op::Activation(Activation::Neg), vec![y_hi], shape);
    let rot = gb.concat_(vec![nx_hi, x_lo, ny_hi, y_lo], 3);

    let a = gb.mul(x4, cos4);
    let b = gb.mul(rot, sin4);
    let out = gb.add(a, b);
    gb.reshape_(out, vec![1, p, (heads * hd) as i64])
}

fn emit_vision_attention(
    emit: &mut Emit<'_>,
    prefix: &str,
    x: HirNodeId,
    cfg: &VisionConfig,
    patches: usize,
    cos: HirNodeId,
    sin: HirNodeId,
    mask: HirNodeId,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let hd = cfg.head_dim;
    let nh = cfg.num_attention_heads;
    let dim = nh * hd;
    let eps = cfg.rms_norm_eps;
    let (p, hi, di) = (patches as i64, nh as i64, hd as i64);

    let q = clipped_linear(emit, &format!("{prefix}.q_proj"), x)?;
    let k = clipped_linear(emit, &format!("{prefix}.k_proj"), x)?;
    let v = clipped_linear(emit, &format!("{prefix}.v_proj"), x)?;

    let q4 = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(q, vec![1, p, hi, di])
    };
    let q4 = rms(emit, &format!("{prefix}.q_norm"), q4, hd, eps)?;
    let k4 = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(k, vec![1, p, hi, di])
    };
    let k4 = rms(emit, &format!("{prefix}.k_norm"), k4, hd, eps)?;
    let v4 = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(v, vec![1, p, hi, di])
    };
    let v4 = rms_no_scale(emit, &format!("{prefix}.v_norm"), v4, hd, eps);

    let attn = {
        let mut gb = HirMut::new(emit.hir());
        let q = gb.reshape_(q4, vec![1, p, dim as i64]);
        let k = gb.reshape_(k4, vec![1, p, dim as i64]);
        let v = gb.reshape_(v4, vec![1, p, dim as i64]);
        let q = apply_2d_rope(&mut gb, q, cos, sin, patches, nh, hd);
        let k = apply_2d_rope(&mut gb, k, cos, sin, patches, nh, hd);
        // Bidirectional over all patches; `mask` is the key-padding mask
        // (1.0 = real patch), which is what `MaskKind::Custom` reads.
        //
        // Emitted directly rather than via `gb.attention()`: that helper leaves
        // `score_scale` unset, and the kernel then defaults it to
        // `head_dim^-0.5`. Gemma 4 vision — like the text tower — uses
        // `scaling = 1.0`, because Q/K are already per-head RMS-normed.
        gb.add_node(
            Op::Attention {
                num_heads: nh,
                head_dim: hd,
                v_head_dim: None,
                mask_kind: MaskKind::Custom,
                score_scale: Some(1.0),
                attn_logit_softcap: None,
            },
            vec![q, k, v, mask],
            Shape::new(&[1, patches, dim], f),
        )
    };
    clipped_linear(emit, &format!("{prefix}.o_proj"), attn)
}

/// One vision encoder layer — the text layer's shape without the MoE branch.
fn emit_vision_layer(
    emit: &mut Emit<'_>,
    prefix: &str,
    x: HirNodeId,
    cfg: &VisionConfig,
    patches: usize,
    cos: HirNodeId,
    sin: HirNodeId,
    mask: HirNodeId,
) -> Result<HirNodeId> {
    let h = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;

    let normed = rms(emit, &format!("{prefix}.input_layernorm"), x, h, eps)?;
    let attn = emit_vision_attention(
        emit,
        &format!("{prefix}.self_attn"),
        normed,
        cfg,
        patches,
        cos,
        sin,
        mask,
    )?;
    let attn = rms(
        emit,
        &format!("{prefix}.post_attention_layernorm"),
        attn,
        h,
        eps,
    )?;
    let res = {
        let mut gb = HirMut::new(emit.hir());
        gb.add(x, attn)
    };

    let normed = rms(
        emit,
        &format!("{prefix}.pre_feedforward_layernorm"),
        res,
        h,
        eps,
    )?;
    // The vision MLP's projections also sit under `.linear.weight`.
    let gate = clipped_linear(emit, &format!("{prefix}.mlp.gate_proj"), normed)?;
    let up = clipped_linear(emit, &format!("{prefix}.mlp.up_proj"), normed)?;
    let ffn = {
        let mut gb = HirMut::new(emit.hir());
        let act = gb.gelu_approx(gate);
        gb.mul(act, up)
    };
    let ffn = clipped_linear(emit, &format!("{prefix}.mlp.down_proj"), ffn)?;
    let ffn = rms(
        emit,
        &format!("{prefix}.post_feedforward_layernorm"),
        ffn,
        h,
        eps,
    )?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.add(res, ffn))
}

/// Build the vision tower + projector for a fixed patch count.
///
/// Inputs: [`PIXELS_INPUT`], [`POS_X_INPUT`], [`POS_Y_INPUT`],
/// [`ROPE_COS_INPUT`], [`ROPE_SIN_INPUT`], [`VALID_INPUT`], [`POOL_INPUT`].
/// Output: [`SOFT_TOKENS_OUTPUT`] `[1, soft_tokens, text_hidden]`.
pub fn build_vision_flow(
    cfg: &DiffusionGemmaConfig,
    weights: &WeightMap,
    patches: usize,
    soft_tokens: usize,
) -> Result<BuiltModel> {
    let v = cfg
        .vision_config
        .as_ref()
        .ok_or_else(|| anyhow!("config has no vision_config; this checkpoint is text-only"))?;
    let f = DType::F32;
    let h = v.hidden_size;
    let patch_dim = 3 * v.patch_size * v.patch_size;
    let text_hidden = cfg.text_config.hidden_size;
    let eps = v.rms_norm_eps;

    let mut flow = ModelFlow::new("diffusiongemma_vision")
        .with_profile(CompileProfile::llama32_prefill())
        .input(PIXELS_INPUT, Shape::new(&[1, patches, patch_dim], f))
        .input(POS_X_INPUT, Shape::new(&[1, patches], f))
        .input(POS_Y_INPUT, Shape::new(&[1, patches], f))
        .input(ROPE_COS_INPUT, Shape::new(&[1, patches, v.head_dim], f))
        .input(ROPE_SIN_INPUT, Shape::new(&[1, patches, v.head_dim], f))
        .input(VALID_INPUT, Shape::new(&[1, patches], f))
        .input(POOL_INPUT, Shape::new(&[soft_tokens, patches], f));

    let hs = Shape::new(&[1, patches, h], f);

    // Patch embed: [0,1] → [-1,1], project, add the looked-up 2-D position
    // embedding, and zero out padding patches.
    {
        let v = v.clone();
        let hs = hs.clone();
        flow = flow.plugin_named("patch_embed", move |emit, _prev| {
            let pixels = emit.flow_input(PIXELS_INPUT)?.hir_id();
            let px = emit.flow_input(POS_X_INPUT)?.hir_id();
            let py = emit.flow_input(POS_Y_INPUT)?.hir_id();
            let valid = emit.flow_input(VALID_INPUT)?.hir_id();
            let proj = emit.load_param(
                &format!("{VISION_PREFIX}.patch_embedder.input_proj.weight"),
                true,
            )?;
            let table = emit.load_param(
                &format!("{VISION_PREFIX}.patch_embedder.position_embedding_table"),
                false,
            )?;
            let two = emit.synth_param("vis.two", vec![2.0], Shape::new(&[1], f));
            let one = emit.synth_param("vis.one", vec![1.0], Shape::new(&[1], f));

            let mut gb = HirMut::new(emit.hir());
            let scaled = {
                let a = gb.mul(pixels, two);
                gb.sub(a, one)
            };
            let embeds = gb.mm(scaled, proj);

            // Position embedding: table is [2, pos_size, hidden]; axis 0 picks
            // the x/y plane, then a plain gather by index.
            let pos_size = v.position_embedding_size as i64;
            let hi = v.hidden_size as i64;
            let x_plane = {
                let t = gb.narrow_(table, 0, 0, 1);
                gb.reshape_(t, vec![pos_size, hi])
            };
            let y_plane = {
                let t = gb.narrow_(table, 0, 1, 1);
                gb.reshape_(t, vec![pos_size, hi])
            };
            let px2 = gb.reshape_(px, vec![1, patches as i64]);
            let py2 = gb.reshape_(py, vec![1, patches as i64]);
            let x_emb = gb.gather_(x_plane, px2, 0);
            let y_emb = gb.gather_(y_plane, py2, 0);
            let pos = gb.add(x_emb, y_emb);
            let out = gb.add(embeds, pos);
            // Padding patches contribute nothing downstream.
            let valid3 = gb.reshape_(valid, vec![1, patches as i64, 1]);
            let out = gb.mul(out, valid3);
            emit.state
                .side_outputs
                .push((PATCH_EMBED_TAP.to_string(), out));
            Ok(Some(emit.wrap(out, hs.clone())))
        });
    }

    for i in 0..v.num_hidden_layers {
        let prefix = format!("{VISION_PREFIX}.encoder.layers.{i}");
        let v = v.clone();
        let hs = hs.clone();
        flow = flow.plugin_named(format!("vlayer{i}"), move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("vision layer{i} needs a hidden input"))?
                .hir_id();
            let cos = emit.flow_input(ROPE_COS_INPUT)?.hir_id();
            let sin = emit.flow_input(ROPE_SIN_INPUT)?.hir_id();
            let mask = emit.flow_input(VALID_INPUT)?.hir_id();
            let out = emit_vision_layer(emit, &prefix, x, &v, patches, cos, sin, mask)?;
            Ok(Some(emit.wrap(out, hs.clone())))
        });
    }

    // Pool → sqrt(hidden) scale → standardize → project into LM space.
    {
        let v = v.clone();
        let out_shape = Shape::new(&[1, soft_tokens, text_hidden], f);
        flow = flow.plugin_named("pool_project", move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("pool_project needs a hidden input"))?
                .hir_id();
            emit.state.side_outputs.push((ENCODER_TAP.to_string(), x));
            let pool = emit.flow_input(POOL_INPUT)?.hir_id();
            let valid = emit.flow_input(VALID_INPUT)?.hir_id();
            let root = emit.synth_param(
                "vis.root_hidden",
                vec![(v.hidden_size as f32).sqrt()],
                Shape::new(&[1], f),
            );

            let pooled = {
                let mut gb = HirMut::new(emit.hir());
                // Zero padding patches before averaging, as HF does.
                let valid3 = gb.reshape_(valid, vec![1, patches as i64, 1]);
                let masked = gb.mul(x, valid3);
                let x2 = gb.reshape_(masked, vec![patches as i64, v.hidden_size as i64]);
                let p = gb.mm(pool, x2); // [soft_tokens, hidden]
                gb.mul(p, root)
            };

            let standardized = if v.standardize {
                let bias = emit.load_param(&format!("{VISION_PREFIX}.std_bias"), false)?;
                let scale = emit.load_param(&format!("{VISION_PREFIX}.std_scale"), false)?;
                let mut gb = HirMut::new(emit.hir());
                let b = gb.reshape_(bias, vec![1, v.hidden_size as i64]);
                let s = gb.reshape_(scale, vec![1, v.hidden_size as i64]);
                let c = gb.sub(pooled, b);
                gb.mul(c, s)
            } else {
                pooled
            };

            emit.state
                .side_outputs
                .push((POOLED_TAP.to_string(), standardized));
            // Projector: scale-free RMS norm, then a linear into LM width.
            let normed = rms_no_scale(
                emit,
                &format!("{VISION_PROJ_PREFIX}.embedding_pre_projection_norm"),
                standardized,
                v.hidden_size,
                eps,
            );
            let w = emit.load_param(
                &format!("{VISION_PROJ_PREFIX}.embedding_projection.weight"),
                true,
            )?;
            let mut gb = HirMut::new(emit.hir());
            let projected = gb.mm(normed, w);
            let out = gb.reshape_(projected, vec![1, soft_tokens as i64, text_hidden as i64]);
            Ok(Some(emit.wrap(out, out_shape.clone())))
        });
    }

    flow.output(SOFT_TOKENS_OUTPUT)
        .build_with(&mut SharedWeights(weights), None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vcfg() -> VisionConfig {
        serde_json::from_str(
            r#"{"hidden_size":1152,"num_hidden_layers":27,"num_attention_heads":16,
                "head_dim":72,"intermediate_size":4304,"patch_size":16,
                "pooling_kernel_size":3,"position_embedding_size":10240,
                "rms_norm_eps":1e-6,"standardize":true,
                "rope_parameters":{"rope_theta":100.0,"rope_type":"default"}}"#,
        )
        .unwrap()
    }

    #[test]
    fn inv_freq_matches_the_reference_partitioning() {
        let c = vcfg();
        // spatial = head_dim/2 = 36, stepping by 2 → 18 frequencies per axis.
        let inv = vision_inv_freq(&c);
        assert_eq!(inv.len(), 18);
        assert!((inv[0] - 1.0).abs() < 1e-12);
        let want = 1.0 / 100f64.powf(2.0 / 36.0);
        assert!((inv[1] - want).abs() < 1e-12);
    }

    #[test]
    fn rope_tables_split_x_and_y_across_the_head() {
        let c = vcfg();
        let (cos, sin) = vision_rope_tables(&c, &[(0, 0), (3, 5)]);
        assert_eq!(cos.len(), 2 * 72);
        // Position (0,0) → every angle is 0 → cos 1, sin 0.
        assert!(cos[..72].iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert!(sin[..72].iter().all(|&v| v.abs() < 1e-6));

        // Position (3,5): low half tracks x=3, high half tracks y=5, and each
        // half repeats its 18 frequencies twice (cat(freqs, freqs)).
        let inv = vision_inv_freq(&c);
        let row = 72;
        for j in 0..18 {
            let cx = (3.0 * inv[j]).cos() as f32;
            let cy = (5.0 * inv[j]).cos() as f32;
            assert!((cos[row + j] - cx).abs() < 1e-6, "x slot {j}");
            assert!((cos[row + 18 + j] - cx).abs() < 1e-6, "x mirror {j}");
            assert!((cos[row + 36 + j] - cy).abs() < 1e-6, "y slot {j}");
            assert!((cos[row + 54 + j] - cy).abs() < 1e-6, "y mirror {j}");
        }
    }

    #[test]
    fn pool_matrix_averages_each_k_by_k_block() {
        // 4×4 grid pooled by k=2 → 4 buckets of 4 patches, weight 1/4 each.
        let pos = grid_positions(4, 4);
        let m = vision_pool_matrix(&pos, 2, 4);
        assert_eq!(m.len(), 4 * 16);
        for b in 0..4 {
            let row = &m[b * 16..(b + 1) * 16];
            let nonzero: Vec<usize> = row
                .iter()
                .enumerate()
                .filter(|(_, w)| **w != 0.0)
                .map(|(i, _)| i)
                .collect();
            assert_eq!(nonzero.len(), 4, "bucket {b} must average 4 patches");
            assert!(row.iter().all(|&w| w == 0.0 || (w - 0.25).abs() < 1e-9));
        }
        // Bucket 0 is the top-left 2×2 block: patches 0, 1, 4, 5.
        let row0 = &m[0..16];
        for i in [0usize, 1, 4, 5] {
            assert!(row0[i] > 0.0, "patch {i} belongs to bucket 0");
        }
        // Every column sums to 1/4 — each patch lands in exactly one bucket.
        for pi in 0..16 {
            let s: f32 = (0..4).map(|b| m[b * 16 + pi]).sum();
            assert!((s - 0.25).abs() < 1e-9);
        }
    }

    #[test]
    fn grid_positions_are_row_major() {
        assert_eq!(
            grid_positions(3, 2),
            vec![(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
        );
    }
}
