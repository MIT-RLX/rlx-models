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

//! SAM 2 mask decoder — host-side.
//!
//! Mirrors `sam2/modeling/sam/mask_decoder.py::MaskDecoder` exactly.
//! Five differences vs the SAM v1 mask decoder:
//!
//!   1. **Object-score token + head.** When `pred_obj_scores=True`
//!      (the default for SAM 2), an extra `obj_score_token` is
//!      prepended to the iou+mask tokens; the decoder's first output
//!      slot becomes `obj_score_logits = pred_obj_score_head(hs[:, 0])`.
//!   2. **High-res features.** The upscaler eats the FpnNeck's
//!      stride-4 and stride-8 levels via two 1×1 lateral convs
//!      (`conv_s0`, `conv_s1`), additively fused into the upscaling
//!      stack between the two `ConvTranspose2d` layers.
//!   3. **`use_multimask_token_for_obj_ptr=True`.** When multimask
//!      output is selected, the object-pointer projection consumes
//!      `mask_tokens_out[:, 1:]` (the three multimask tokens) instead
//!      of `mask_tokens_out[:, 0:1]` (the single).
//!   4. **Dynamic multimask via stability** (`dynamic_multimask_via_
//!      stability=True` in some configs) — if multimask_output is
//!      False but the model thinks the single token's stability is
//!      below a threshold, fall back to the best of the multimask
//!      outputs. Implemented per the reference.
//!   5. **Object-pointer projection.** A small MLP that turns the
//!      selected mask token(s) into the pointer fed to the memory
//!      attention layer (Phase 3 path). Weights live here on the
//!      decoder side.
//!
//! Weight key prefix is `sam_mask_decoder.*` (SAM 2 nests the mask
//! decoder under `sam_mask_decoder` in the published checkpoints).

use super::config::Sam2DecoderConfig;
use super::transformer::{
    Sam2TwoWayTransformerWeights, add_inplace, extract_two_way_transformer_weights, linear,
    two_way_transformer_forward,
};
use super::upscale_ir::Sam2MaskUpscaleCompiled;
use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_sam_ir::mask_hyper_matmul_ir::MaskHyperMatmulCompiled;
use rlx_sam_ir::mlp_relu_ir::MlpReluCompiled;
use rlx_sam_ir::twoway_transformer_ir::TwoWayTransformerCompiled;

pub struct Sam2MaskDecoderWeights {
    pub iou_token: Vec<f32>,   // [1, transformer_dim]
    pub mask_tokens: Vec<f32>, // [num_mask_tokens, transformer_dim]
    /// Optional object-score token, populated when `pred_obj_scores=true`.
    pub obj_score_token: Option<Vec<f32>>,
    pub transformer: Sam2TwoWayTransformerWeights,

    /// ConvTranspose2d in=transformer_dim, out=transformer_dim/4, k=2, s=2.
    pub upscale_conv1_w: Vec<f32>,
    pub upscale_conv1_b: Vec<f32>,
    pub upscale_ln_g: Vec<f32>,
    pub upscale_ln_b: Vec<f32>,
    /// ConvTranspose2d in=transformer_dim/4, out=transformer_dim/8.
    pub upscale_conv2_w: Vec<f32>,
    pub upscale_conv2_b: Vec<f32>,

    /// Optional high-res fusion 1×1 convs.
    ///   - `conv_s0`: stride-4 features → transformer_dim/8 channels
    ///   - `conv_s1`: stride-8 features → transformer_dim/4 channels
    pub conv_s0_w: Option<Vec<f32>>,
    pub conv_s0_b: Option<Vec<f32>>,
    pub conv_s1_w: Option<Vec<f32>>,
    pub conv_s1_b: Option<Vec<f32>>,

    /// `num_mask_tokens` × 3-layer ReLU MLPs producing
    /// `transformer_dim/8` per mask token.
    pub hyper_mlps: Vec<Sam2HypernetMlp>,

    /// IoU prediction head: 3-layer ReLU MLP `transformer_dim →
    /// iou_head_hidden_dim → iou_head_hidden_dim → num_mask_tokens`.
    pub iou_head: Sam2HypernetMlp,
    /// `iou_prediction_use_sigmoid` flag.
    pub iou_use_sigmoid: bool,

    /// Optional object-score prediction head (3-layer MLP when the
    /// `pred_obj_scores_mlp` flag is set; otherwise a plain Linear).
    pub obj_score_head: Option<Sam2HypernetMlp>,

    /// Optional object-pointer projection MLP. Reference shape:
    /// `MLP(transformer_dim, transformer_dim, transformer_dim, 3)` if
    /// `use_mlp_for_obj_ptr_proj=True`, otherwise `Linear(...)`.
    pub obj_ptr_proj: Option<Sam2HypernetMlp>,

    pub transformer_dim: usize,
    pub num_mask_tokens: usize,
    pub use_high_res_features: bool,
    pub pred_obj_scores: bool,
    pub use_multimask_token_for_obj_ptr: bool,
    pub dynamic_multimask_via_stability: bool,
    pub dynamic_multimask_stability_delta: f32,
    pub dynamic_multimask_stability_thresh: f32,
}

pub struct Sam2HypernetMlp {
    pub layers: Vec<Sam2MlpLayer>,
    pub sigmoid_output: bool,
}

pub struct Sam2MlpLayer {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub in_d: usize,
    pub out_d: usize,
}

pub fn extract_mask_decoder_weights(
    weights: &mut WeightMap,
    cfg: &Sam2DecoderConfig,
) -> Result<Sam2MaskDecoderWeights> {
    let transformer_dim = cfg.transformer_dim;
    let num_mask_tokens = cfg.num_mask_tokens;

    let (iou_token, sh) = weights.take("sam_mask_decoder.iou_token.weight")?;
    ensure!(
        sh == vec![1, transformer_dim],
        "iou_token shape {sh:?} not [1, {transformer_dim}]"
    );
    let (mask_tokens, sh) = weights.take("sam_mask_decoder.mask_tokens.weight")?;
    ensure!(
        sh == vec![num_mask_tokens, transformer_dim],
        "mask_tokens shape {sh:?} not [{num_mask_tokens}, {transformer_dim}]"
    );

    let obj_score_token = if cfg.pred_obj_scores {
        let (data, sh) = weights.take("sam_mask_decoder.obj_score_token.weight")?;
        ensure!(
            sh == vec![1, transformer_dim],
            "obj_score_token shape {sh:?} not [1, {transformer_dim}]"
        );
        Some(data)
    } else {
        None
    };

    // ConvTranspose2d weight convention in PyTorch: [in, out, kH, kW].
    let q4 = transformer_dim / 4;
    let q8 = transformer_dim / 8;
    let (upscale_conv1_w, sh) = weights.take("sam_mask_decoder.output_upscaling.0.weight")?;
    ensure!(
        sh == vec![transformer_dim, q4, 2, 2],
        "output_upscaling.0.weight shape {sh:?} not [{transformer_dim}, {q4}, 2, 2]"
    );
    let (upscale_conv1_b, _) = weights.take("sam_mask_decoder.output_upscaling.0.bias")?;
    let (upscale_ln_g, _) = weights.take("sam_mask_decoder.output_upscaling.1.weight")?;
    let (upscale_ln_b, _) = weights.take("sam_mask_decoder.output_upscaling.1.bias")?;
    let (upscale_conv2_w, sh) = weights.take("sam_mask_decoder.output_upscaling.3.weight")?;
    ensure!(
        sh == vec![q4, q8, 2, 2],
        "output_upscaling.3.weight shape {sh:?} not [{q4}, {q8}, 2, 2]"
    );
    let (upscale_conv2_b, _) = weights.take("sam_mask_decoder.output_upscaling.3.bias")?;

    // High-res fusion convs (gated on `use_high_res_features`).
    let (conv_s0_w, conv_s0_b, conv_s1_w, conv_s1_b) = if cfg.use_high_res_features {
        let (s0w, sh) = weights.take("sam_mask_decoder.conv_s0.weight")?;
        ensure!(
            sh == vec![q8, transformer_dim, 1, 1],
            "conv_s0.weight shape {sh:?} not [{q8}, {transformer_dim}, 1, 1]"
        );
        let (s0b, _) = weights.take("sam_mask_decoder.conv_s0.bias")?;
        let (s1w, sh) = weights.take("sam_mask_decoder.conv_s1.weight")?;
        ensure!(
            sh == vec![q4, transformer_dim, 1, 1],
            "conv_s1.weight shape {sh:?} not [{q4}, {transformer_dim}, 1, 1]"
        );
        let (s1b, _) = weights.take("sam_mask_decoder.conv_s1.bias")?;
        (Some(s0w), Some(s0b), Some(s1w), Some(s1b))
    } else {
        (None, None, None, None)
    };

    // Hypernetwork MLPs.
    let mut hyper_mlps = Vec::with_capacity(num_mask_tokens);
    for i in 0..num_mask_tokens {
        let mlp = extract_mlp(
            weights,
            &format!("sam_mask_decoder.output_hypernetworks_mlps.{i}"),
            transformer_dim,
            transformer_dim,
            q8,
            3,
            false,
        )?;
        hyper_mlps.push(mlp);
    }

    // IoU prediction head.
    let iou_head = extract_mlp(
        weights,
        "sam_mask_decoder.iou_prediction_head",
        transformer_dim,
        cfg.iou_head_hidden_dim,
        num_mask_tokens,
        cfg.iou_head_depth,
        cfg.iou_prediction_use_sigmoid,
    )?;

    // Object-score head: 3-layer MLP when pred_obj_scores_mlp,
    // else a plain Linear(d, 1).
    let obj_score_head = if cfg.pred_obj_scores {
        if cfg.pred_obj_scores_mlp {
            Some(extract_mlp(
                weights,
                "sam_mask_decoder.pred_obj_score_head",
                transformer_dim,
                transformer_dim,
                1,
                3,
                false,
            )?)
        } else {
            let (w, sh) = weights.take("sam_mask_decoder.pred_obj_score_head.weight")?;
            ensure!(
                sh == vec![1, transformer_dim],
                "pred_obj_score_head.weight shape {sh:?} not [1, {transformer_dim}]"
            );
            let (b, _) = weights.take("sam_mask_decoder.pred_obj_score_head.bias")?;
            Some(Sam2HypernetMlp {
                layers: vec![Sam2MlpLayer {
                    w,
                    b,
                    in_d: transformer_dim,
                    out_d: 1,
                }],
                sigmoid_output: false,
            })
        }
    } else {
        None
    };

    // Object-pointer projection. NB: lives at the *top level* of the
    // SAM2Base module (sibling of `sam_mask_decoder`, not nested), so
    // the checkpoint keys are `obj_ptr_proj.layers.{i}.weight` (not
    // `sam_mask_decoder.obj_ptr_proj.…`). Same for `no_obj_ptr`.
    let obj_ptr_proj = if cfg.use_object_pointer {
        if cfg.use_mlp_for_obj_ptr_proj {
            Some(extract_mlp(
                weights,
                "obj_ptr_proj",
                transformer_dim,
                transformer_dim,
                transformer_dim,
                3,
                false,
            )?)
        } else {
            let (w, sh) = weights.take("obj_ptr_proj.weight")?;
            ensure!(
                sh == vec![transformer_dim, transformer_dim],
                "obj_ptr_proj.weight shape {sh:?} not [{transformer_dim}, {transformer_dim}]"
            );
            let (b, _) = weights.take("obj_ptr_proj.bias")?;
            Some(Sam2HypernetMlp {
                layers: vec![Sam2MlpLayer {
                    w,
                    b,
                    in_d: transformer_dim,
                    out_d: transformer_dim,
                }],
                sigmoid_output: false,
            })
        }
    } else {
        None
    };

    let transformer = extract_two_way_transformer_weights(
        weights,
        transformer_dim,
        cfg.transformer_depth,
        cfg.transformer_num_heads,
        cfg.transformer_mlp_dim,
    )?;

    Ok(Sam2MaskDecoderWeights {
        iou_token,
        mask_tokens,
        obj_score_token,
        transformer,
        upscale_conv1_w,
        upscale_conv1_b,
        upscale_ln_g,
        upscale_ln_b,
        upscale_conv2_w,
        upscale_conv2_b,
        conv_s0_w,
        conv_s0_b,
        conv_s1_w,
        conv_s1_b,
        hyper_mlps,
        iou_head,
        iou_use_sigmoid: cfg.iou_prediction_use_sigmoid,
        obj_score_head,
        obj_ptr_proj,
        transformer_dim,
        num_mask_tokens,
        use_high_res_features: cfg.use_high_res_features,
        pred_obj_scores: cfg.pred_obj_scores,
        use_multimask_token_for_obj_ptr: cfg.use_multimask_token_for_obj_ptr,
        dynamic_multimask_via_stability: cfg.dynamic_multimask_via_stability,
        dynamic_multimask_stability_delta: cfg.dynamic_multimask_stability_delta,
        dynamic_multimask_stability_thresh: cfg.dynamic_multimask_stability_thresh,
    })
}

fn extract_mlp(
    weights: &mut WeightMap,
    prefix: &str,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    num_layers: usize,
    sigmoid_output: bool,
) -> Result<Sam2HypernetMlp> {
    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let in_d = if i == 0 { input_dim } else { hidden_dim };
        let out_d = if i + 1 == num_layers {
            output_dim
        } else {
            hidden_dim
        };
        let (w, sh) = weights.take(&format!("{prefix}.layers.{i}.weight"))?;
        ensure!(
            sh == vec![out_d, in_d],
            "{prefix}.layers.{i}.weight shape {sh:?} not [{out_d}, {in_d}]"
        );
        let (b, _) = weights.take(&format!("{prefix}.layers.{i}.bias"))?;
        layers.push(Sam2MlpLayer { w, b, in_d, out_d });
    }
    Ok(Sam2HypernetMlp {
        layers,
        sigmoid_output,
    })
}

/// Forward through a ReLU MLP. Final layer is NOT followed by ReLU;
/// optional sigmoid is applied to the output.
pub fn mlp_forward(mlp: &Sam2HypernetMlp, x: &[f32], rows: usize) -> Vec<f32> {
    let mut cur = x.to_vec();
    let n = mlp.layers.len();
    for (i, layer) in mlp.layers.iter().enumerate() {
        cur = linear(&cur, &layer.w, &layer.b, rows, layer.in_d, layer.out_d);
        if i + 1 < n {
            for v in cur.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
    }
    if mlp.sigmoid_output {
        for v in cur.iter_mut() {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
    }
    cur
}

/// Output of [`mask_decoder_forward`].
pub struct Sam2MaskDecoderOutput {
    /// `[num_masks, h_out, w_out]` mask logits. `num_masks` is 1 or 3
    /// depending on `multimask_output` (and dynamic-stability fallback).
    pub masks: Vec<f32>,
    pub iou_pred: Vec<f32>, // [num_masks]
    pub num_masks: usize,
    pub h_out: usize,
    pub w_out: usize,
    /// Selected mask token(s) for the object-pointer projection.
    /// Shape `[num_ptr_tokens, transformer_dim]`. None if
    /// `use_object_pointer=false`.
    pub sam_tokens_out: Vec<f32>,
    pub num_ptr_tokens: usize,
    /// Object-score logits — `[1]` per batch when pred_obj_scores=true,
    /// else a constant +10 (matching the reference) so downstream
    /// `obj_score_prob` evaluates to ~1.
    pub object_score_logits: Vec<f32>,
    /// Object-pointer projection output `[num_ptr_tokens,
    /// transformer_dim]`. None if `use_object_pointer=false`.
    pub object_pointer: Option<Vec<f32>>,
}

/// Run the SAM 2 mask decoder.
///
/// `image_embeddings`: NCHW `[1, C=transformer_dim, grid, grid]`.
/// `image_pe`: NCHW `[1, C=transformer_dim, grid, grid]`.
/// `sparse_prompt_embeddings`: `[num_sparse, transformer_dim]`.
/// `dense_prompt_embeddings`: `[transformer_dim, grid, grid]`.
/// `high_res_features`: optional `(feat_s0, feat_s1)` where:
///
///   - `feat_s0`: stride-4 features `[transformer_dim, 4·grid, 4·grid]`
///   - `feat_s1`: stride-8 features `[transformer_dim, 2·grid, 2·grid]`
///
///   Reference passes these from the FpnNeck.
/// `grid`: spatial side of the image embeddings (64 for SAM 2).
#[allow(clippy::too_many_arguments)]
pub fn mask_decoder_forward(
    w: &Sam2MaskDecoderWeights,
    upscale: &mut Sam2MaskUpscaleCompiled,
    hyper_matmul: Option<&mut MaskHyperMatmulCompiled>,
    hyper_mlps_ir: Option<&mut [MlpReluCompiled]>,
    iou_head_ir: Option<&mut MlpReluCompiled>,
    obj_score_head_ir: Option<&mut MlpReluCompiled>,
    obj_ptr_proj_ir: Option<&mut MlpReluCompiled>,
    tw_ir: Option<&mut TwoWayTransformerCompiled>,
    image_embeddings: &[f32],
    image_pe: &[f32],
    sparse_prompt_embeddings: &[f32],
    num_sparse_tokens: usize,
    dense_prompt_embeddings: &[f32],
    high_res_features: Option<(&[f32], &[f32])>,
    multimask_output: bool,
    grid: usize,
) -> Result<Sam2MaskDecoderOutput> {
    let e = w.transformer_dim;
    let nm = w.num_mask_tokens;
    let g = grid;
    ensure!(
        image_embeddings.len() == e * g * g,
        "image_embeddings len {} ≠ E·g·g ({e}·{g}·{g})",
        image_embeddings.len()
    );
    ensure!(
        image_pe.len() == e * g * g,
        "image_pe len {} ≠ E·g·g",
        image_pe.len()
    );
    ensure!(
        dense_prompt_embeddings.len() == e * g * g,
        "dense_prompt_embeddings len {} ≠ E·g·g",
        dense_prompt_embeddings.len()
    );
    ensure!(
        sparse_prompt_embeddings.len() == num_sparse_tokens * e,
        "sparse_prompt_embeddings len {} ≠ num_sparse·E ({num_sparse_tokens}·{e})",
        sparse_prompt_embeddings.len()
    );
    if w.use_high_res_features {
        let (s0, s1) = high_res_features.ok_or_else(|| {
            anyhow::anyhow!("use_high_res_features=true requires (feat_s0, feat_s1)")
        })?;
        ensure!(
            s0.len() == e * (4 * g) * (4 * g),
            "feat_s0 len {} ≠ E·4g·4g ({e}·{}·{})",
            s0.len(),
            4 * g,
            4 * g
        );
        ensure!(
            s1.len() == e * (2 * g) * (2 * g),
            "feat_s1 len {} ≠ E·2g·2g ({e}·{}·{})",
            s1.len(),
            2 * g,
            2 * g
        );
    }

    // ── Build tokens = cat(maybe obj_score, iou, mask, sparse) ──
    let s = if w.obj_score_token.is_some() { 1 } else { 0 };
    let n_out_tokens = s + 1 + nm;
    let q_n = n_out_tokens + num_sparse_tokens;
    let mut tokens = Vec::with_capacity(q_n * e);
    if let Some(obj) = &w.obj_score_token {
        tokens.extend_from_slice(obj);
    }
    tokens.extend_from_slice(&w.iou_token);
    tokens.extend_from_slice(&w.mask_tokens);
    tokens.extend_from_slice(sparse_prompt_embeddings);

    // ── src = image_embeddings + dense_prompt_embeddings ──
    let mut src = image_embeddings.to_vec();
    for i in 0..src.len() {
        src[i] += dense_prompt_embeddings[i];
    }
    let pos_src = image_pe.to_vec();

    // ── Run the two-way transformer ──
    let k_n = g * g;
    let (hs, src_post) = if let Some(tw) = tw_ir {
        if tw.masked && q_n <= tw.max_q_n && tw.k_n == k_n {
            tw.run_nchw_masked(&tokens, q_n, &src, &pos_src, g)?
        } else if !tw.masked && q_n == tw.max_q_n && tw.k_n == k_n {
            tw.run_nchw(&tokens, &src, &pos_src, g)?
        } else {
            two_way_transformer_forward(&w.transformer, &src, &pos_src, &tokens, 1, e, g, g, q_n)
        }
    } else {
        two_way_transformer_forward(&w.transformer, &src, &pos_src, &tokens, 1, e, g, g, q_n)
    };

    let obj_score_logits_pre = if let Some(ir) = obj_score_head_ir {
        ir.run(&hs[..e], 1)?
    } else if let Some(head) = &w.obj_score_head {
        let token = &hs[..e];
        mlp_forward(head, token, 1)
    } else {
        // Reference returns a constant +10 logit when pred_obj_scores=false.
        vec![10.0]
    };

    let iou_token_out: Vec<f32> = hs[s * e..(s + 1) * e].to_vec();
    let mask_tokens_out = hs[(s + 1) * e..(s + 1 + nm) * e].to_vec();

    // ── Reshape src_post [1, g·g, E] → [1, E, g, g] NCHW ──
    let mut src_nchw = vec![0f32; e * g * g];
    for ss in 0..g * g {
        for c in 0..e {
            src_nchw[c * g * g + ss] = src_post[ss * e + c];
        }
    }

    // ── Upscaling via IR (optional high-res 1×1 fuse inside graph) ──
    let q8 = e / 8;
    let h2 = g * 4;
    let w2 = g * 4;
    let (feat_s0, feat_s1) = high_res_features.unwrap_or((&[] as &[f32], &[] as &[f32]));
    let up2 = upscale.run(&src_nchw, feat_s1, feat_s0, g)?;

    // ── Hypernetwork MLPs → [nm, q8] ──
    let mut hyper_in = vec![0f32; nm * q8];
    if let Some(mlps) = hyper_mlps_ir {
        ensure!(
            mlps.len() == nm,
            "hyper_mlps_ir len {} ≠ num_mask_tokens {}",
            mlps.len(),
            nm
        );
        for i in 0..nm {
            let token = &mask_tokens_out[i * e..(i + 1) * e];
            let h = mlps[i].run(token, 1)?;
            hyper_in[i * q8..(i + 1) * q8].copy_from_slice(&h);
        }
    } else {
        for i in 0..nm {
            let token = &mask_tokens_out[i * e..(i + 1) * e];
            let h = mlp_forward(&w.hyper_mlps[i], token, 1);
            hyper_in[i * q8..(i + 1) * q8].copy_from_slice(&h);
        }
    }
    let spat = h2 * w2;
    let mut masks_all = vec![0f32; nm * spat];
    if let Some(hm) = hyper_matmul {
        hm.run(&hyper_in, &up2, &mut masks_all)?;
    } else {
        rlx_cpu::blas::sgemm_auto(&hyper_in, &up2, &mut masks_all, nm, q8, spat);
    }

    // ── IoU head ──
    let iou_pred_all = if let Some(head) = iou_head_ir {
        head.run(&iou_token_out, 1)?
    } else {
        mlp_forward(&w.iou_head, &iou_token_out, 1)
    };

    // ── Multimask selection (with optional dynamic stability fallback) ──
    let (masks, iou_pred, num_masks, ptr_indices): (Vec<f32>, Vec<f32>, usize, Vec<usize>) =
        if multimask_output {
            // [1:nm] = 3 masks for nm=4.
            let masks = masks_all[spat..].to_vec();
            let iou = iou_pred_all[1..].to_vec();
            let ptr = if w.use_multimask_token_for_obj_ptr {
                (1..nm).collect()
            } else {
                vec![0]
            };
            (masks, iou, nm - 1, ptr)
        } else if w.dynamic_multimask_via_stability {
            dynamic_multimask_via_stability(
                &masks_all,
                &iou_pred_all,
                nm,
                spat,
                w.dynamic_multimask_stability_delta,
                w.dynamic_multimask_stability_thresh,
            )
        } else {
            let masks = masks_all[..spat].to_vec();
            let iou = iou_pred_all[..1].to_vec();
            (masks, iou, 1, vec![0])
        };

    let num_ptr_tokens = ptr_indices.len();
    let mut sam_tokens_out = Vec::with_capacity(num_ptr_tokens * e);
    for &pi in &ptr_indices {
        sam_tokens_out.extend_from_slice(&mask_tokens_out[pi * e..(pi + 1) * e]);
    }

    let object_pointer = if let Some(ir) = obj_ptr_proj_ir {
        if ir.compiled_rows() == num_ptr_tokens {
            Some(ir.run(&sam_tokens_out, num_ptr_tokens)?)
        } else {
            w.obj_ptr_proj
                .as_ref()
                .map(|proj| mlp_forward(proj, &sam_tokens_out, num_ptr_tokens))
        }
    } else {
        w.obj_ptr_proj
            .as_ref()
            .map(|proj| mlp_forward(proj, &sam_tokens_out, num_ptr_tokens))
    };

    Ok(Sam2MaskDecoderOutput {
        masks,
        iou_pred,
        num_masks,
        h_out: h2,
        w_out: w2,
        sam_tokens_out,
        num_ptr_tokens,
        object_score_logits: obj_score_logits_pre,
        object_pointer,
    })
}

/// Reference's `_dynamic_multimask_via_stability`: pick between the
/// single-mask token (index 0) and the best multimask token (1..nm)
/// based on a stability score that compares mask area at two thresholds.
fn dynamic_multimask_via_stability(
    masks_all: &[f32],
    iou_pred_all: &[f32],
    _nm: usize,
    spat: usize,
    delta: f32,
    thresh: f32,
) -> (Vec<f32>, Vec<f32>, usize, Vec<usize>) {
    // multimask logits [nm-1, spat], iou [nm-1]
    let mm_masks = &masks_all[spat..];
    let mm_iou = &iou_pred_all[1..];
    // Best multimask by predicted IoU.
    let best = mm_iou
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv { (i, v) } else { (bi, bv) }
        })
        .0;

    // Stability score of single-mask token (index 0).
    let single_mask = &masks_all[..spat];
    let stability = mask_stability_score(single_mask, delta);
    if stability >= thresh {
        // Single mask is stable enough; use it.
        (single_mask.to_vec(), iou_pred_all[..1].to_vec(), 1, vec![0])
    } else {
        // Fall back to the best multimask token.
        let masks = mm_masks[best * spat..(best + 1) * spat].to_vec();
        let iou = vec![mm_iou[best]];
        // Pointer index in the *original* nm tokens: best+1.
        (masks, iou, 1, vec![best + 1])
    }
}

/// Stability score: `area(masks > +delta) / area(masks > -delta)`.
/// Mirrors `_get_stability_scores` in the reference.
fn mask_stability_score(mask_logits: &[f32], delta: f32) -> f32 {
    let mut hi = 0u32;
    let mut lo = 0u32;
    for &v in mask_logits {
        if v > delta {
            hi += 1;
        }
        if v > -delta {
            lo += 1;
        }
    }
    if lo == 0 { 1.0 } else { hi as f32 / lo as f32 }
}

#[allow(dead_code)]
fn _silence_add_inplace(x: &mut [f32], y: &[f32]) {
    add_inplace(x, y);
}
