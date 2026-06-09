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

//! SAM 2 — Meta's Segment Anything Model 2 (image + video segmentation).
//!
//! Mirrors `facebookresearch/sam2` so the published
//! `sam2_hiera_{t,s,b+,l}.{pt,safetensors}` checkpoints load with no
//! weight-key remapping.
//!
//! ## Components
//!
//! - **Phase 1** — Hiera image encoder + FpnNeck
//!   ([`image_encoder`], [`fpn_neck`], [`preprocess`]).
//! - **Phase 2** — prompt encoder + TwoWayTransformer + mask decoder
//!   with object-pointer / object-score / high-res mask path
//!   ([`prompt_encoder`], [`transformer`], [`mask_decoder`]).
//! - **Phase 3** — memory encoder + memory attention for video
//!   tracking ([`memory_encoder`], [`memory_attention`]).
//! - **Top-level wrapper** — [`Sam2`] orchestrator with
//!   `predict_image()` and `predict_video_frame()` APIs.
//!
//! ## Parity status
//!
//! Synthetic-weights build tests in [`tests`] exercise every component
//! (encoder, prompt enc, decoder, memory enc/attn, end-to-end Sam2
//! object) for every Hiera variant. Numerical parity against the
//! pytorch reference is wired up in `tests/sam2_parity.rs` behind the
//! `parity-pytorch` feature flag — turning the bisect options there
//! against a real `sam2_hiera_*.safetensors` checkpoint is the
//! follow-up bisect work (analogous to how SAM v1 Phase 1 landed
//! parity in iterative passes after the initial graph was wired).

pub mod axial_rope;
pub mod cli;
pub mod config;
pub mod flow;
pub mod fpn_neck;
pub mod fpn_neck_ir;
pub mod image_encoder;
pub mod mask_decoder;
pub mod memory_attention;
pub mod memory_attention_ir;
pub mod memory_encoder;
pub mod memory_mask_ir;
pub mod mlp_ir;
pub mod preprocess;
pub mod prompt_encoder;
pub mod prompt_mask_ir;
#[allow(clippy::module_inception)]
pub mod sam2;
pub mod transformer;
pub mod transformer_ir;
pub mod upscale_ir;

pub use rlx_sam::profile::{
    SAM_PROFILE_FILE, sam_profile_near_weights, sam2_profile_default, sam2_profile_near_weights,
};

pub use config::{
    SAM2_IMG_SIZE, SAM2_PATCH_GRID, SAM2_PATCH_KERNEL, SAM2_PATCH_PADDING, SAM2_PATCH_STRIDE,
    SAM2_PIXEL_MEAN, SAM2_PIXEL_STD, SAM2_PROMPT_EMBED_DIM, SAM2_Q_POOL_COUNT, SAM2_Q_STRIDE,
    Sam2Config, Sam2DecoderConfig, Sam2FpnConfig, Sam2HieraConfig, Sam2MemoryConfig,
    Sam2MemoryEncoderConfig,
};
pub use flow::{Sam2ImageEncoderBuilt, Sam2ImageEncoderFlow, build_sam2_image_encoder_built};
pub use fpn_neck::{FpnLevel, FpnNeckWeights, apply_fpn_neck, apply_fpn_neck_host};
pub use fpn_neck_ir::{Sam2FpnNeckIr, compile_fpn_neck_ir};
pub use image_encoder::{build_sam2_image_encoder_graph, build_sam2_image_encoder_hir};
pub use mask_decoder::{Sam2MaskDecoderOutput, Sam2MaskDecoderWeights, mask_decoder_forward};
pub use memory_attention::{Sam2MemoryAttentionWeights, memory_attention_forward};
pub use memory_attention_ir::MemoryAttentionCompiled;
pub use memory_encoder::{
    Sam2MemoryEncoderOutput, Sam2MemoryEncoderWeights, memory_encoder_forward,
};
pub use preprocess::{Sam2PreprocessWeights, assemble_patch_tokens, preprocess_image};
pub use prompt_encoder::{
    SAM2_MASK_IN_CHANS, SAM2_PROMPT_GRID, Sam2PromptEncoderOutput, Sam2PromptEncoderWeights,
    prompt_encoder_forward,
};
pub use rlx_sam_ir::twoway_transformer_ir::TwoWayTransformerCompiled;
pub use sam2::{Sam2, Sam2ImagePrediction, Sam2VideoState};
pub use transformer::{Sam2TwoWayTransformerWeights, two_way_transformer_forward};
pub use transformer_ir::compile_two_way_transformer;

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_core::weight_map::WeightMap;
    use rlx_runtime::Device;
    use std::collections::HashMap;

    type T = HashMap<String, (Vec<f32>, Vec<usize>)>;

    fn z(n: usize) -> Vec<f32> {
        vec![0.0f32; n]
    }

    /// Insert every key required by the Hiera image encoder + FPN
    /// neck. Mirrors the assertions in `image_encoder.rs` and
    /// `fpn_neck.rs::extract_fpn_weights`.
    fn add_hiera_weights(t: &mut T, cfg: &Sam2HieraConfig) {
        let e0 = cfg.embed_dim;
        let k = SAM2_PATCH_KERNEL;
        let [ph, pw] = cfg.window_pos_embed_bkg_spatial_size;
        let mu = cfg.window_size_at_stage(0);

        t.insert(
            "image_encoder.trunk.patch_embed.proj.weight".into(),
            (z(e0 * 3 * k * k), vec![e0, 3, k, k]),
        );
        t.insert(
            "image_encoder.trunk.patch_embed.proj.bias".into(),
            (z(e0), vec![e0]),
        );
        t.insert(
            "image_encoder.trunk.pos_embed".into(),
            (z(e0 * ph * pw), vec![1, e0, ph, pw]),
        );
        t.insert(
            "image_encoder.trunk.pos_embed_window".into(),
            (z(e0 * mu * mu), vec![1, e0, mu, mu]),
        );

        let q_pool = cfg.q_pool_block_indices();
        let total = cfg.total_blocks();
        let mut stage = 0usize;
        let mut dim_curr = e0;
        for i in 0..total {
            let is_q_pool = q_pool.contains(&i);
            let dim_in = dim_curr;
            let stage_after = if is_q_pool { stage + 1 } else { stage };
            let dim_out = cfg.embed_dim_at_stage(stage_after);
            let lp = format!("image_encoder.trunk.blocks.{i}");

            t.insert(format!("{lp}.norm1.weight"), (z(dim_in), vec![dim_in]));
            t.insert(format!("{lp}.norm1.bias"), (z(dim_in), vec![dim_in]));
            if dim_in != dim_out {
                t.insert(
                    format!("{lp}.proj.weight"),
                    (z(dim_in * dim_out), vec![dim_out, dim_in]),
                );
                t.insert(format!("{lp}.proj.bias"), (z(dim_out), vec![dim_out]));
            }
            t.insert(
                format!("{lp}.attn.qkv.weight"),
                (z(dim_in * 3 * dim_out), vec![3 * dim_out, dim_in]),
            );
            if cfg.qkv_bias {
                t.insert(
                    format!("{lp}.attn.qkv.bias"),
                    (z(3 * dim_out), vec![3 * dim_out]),
                );
            }
            t.insert(
                format!("{lp}.attn.proj.weight"),
                (z(dim_out * dim_out), vec![dim_out, dim_out]),
            );
            t.insert(format!("{lp}.attn.proj.bias"), (z(dim_out), vec![dim_out]));
            t.insert(format!("{lp}.norm2.weight"), (z(dim_out), vec![dim_out]));
            t.insert(format!("{lp}.norm2.bias"), (z(dim_out), vec![dim_out]));

            let hidden = (dim_out as f64 * cfg.mlp_ratio) as usize;
            t.insert(
                format!("{lp}.mlp.layers.0.weight"),
                (z(dim_out * hidden), vec![hidden, dim_out]),
            );
            t.insert(format!("{lp}.mlp.layers.0.bias"), (z(hidden), vec![hidden]));
            t.insert(
                format!("{lp}.mlp.layers.1.weight"),
                (z(hidden * dim_out), vec![dim_out, hidden]),
            );
            t.insert(
                format!("{lp}.mlp.layers.1.bias"),
                (z(dim_out), vec![dim_out]),
            );

            if is_q_pool {
                stage += 1;
                dim_curr = dim_out;
            }
        }

        let fpn = Sam2FpnConfig::for_hiera(cfg);
        for (i, &cin) in fpn.backbone_channel_list.iter().enumerate() {
            t.insert(
                format!("image_encoder.neck.convs.{i}.conv.weight"),
                (z(fpn.d_model * cin), vec![fpn.d_model, cin, 1, 1]),
            );
            t.insert(
                format!("image_encoder.neck.convs.{i}.conv.bias"),
                (z(fpn.d_model), vec![fpn.d_model]),
            );
        }
    }

    fn add_prompt_encoder_weights(t: &mut T, embed_dim: usize, mask_in_chans: usize) {
        let half = embed_dim / 2;
        let q = mask_in_chans / 4;
        t.insert(
            "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix".into(),
            (z(2 * half), vec![2, half]),
        );
        t.insert(
            "sam_prompt_encoder.not_a_point_embed.weight".into(),
            (z(embed_dim), vec![1, embed_dim]),
        );
        t.insert(
            "sam_prompt_encoder.no_mask_embed.weight".into(),
            (z(embed_dim), vec![1, embed_dim]),
        );
        for i in 0..4 {
            t.insert(
                format!("sam_prompt_encoder.point_embeddings.{i}.weight"),
                (z(embed_dim), vec![1, embed_dim]),
            );
        }
        t.insert(
            "sam_prompt_encoder.mask_downscaling.0.weight".into(),
            (z(q * 4), vec![q, 1, 2, 2]),
        );
        t.insert(
            "sam_prompt_encoder.mask_downscaling.0.bias".into(),
            (z(q), vec![q]),
        );
        t.insert(
            "sam_prompt_encoder.mask_downscaling.1.weight".into(),
            (z(q), vec![q]),
        );
        t.insert(
            "sam_prompt_encoder.mask_downscaling.1.bias".into(),
            (z(q), vec![q]),
        );
        t.insert(
            "sam_prompt_encoder.mask_downscaling.3.weight".into(),
            (z(mask_in_chans * q * 4), vec![mask_in_chans, q, 2, 2]),
        );
        t.insert(
            "sam_prompt_encoder.mask_downscaling.3.bias".into(),
            (z(mask_in_chans), vec![mask_in_chans]),
        );
        t.insert(
            "sam_prompt_encoder.mask_downscaling.4.weight".into(),
            (z(mask_in_chans), vec![mask_in_chans]),
        );
        t.insert(
            "sam_prompt_encoder.mask_downscaling.4.bias".into(),
            (z(mask_in_chans), vec![mask_in_chans]),
        );
        t.insert(
            "sam_prompt_encoder.mask_downscaling.6.weight".into(),
            (
                z(embed_dim * mask_in_chans),
                vec![embed_dim, mask_in_chans, 1, 1],
            ),
        );
        t.insert(
            "sam_prompt_encoder.mask_downscaling.6.bias".into(),
            (z(embed_dim), vec![embed_dim]),
        );
    }

    fn add_two_way_transformer_weights(t: &mut T, cfg: &Sam2DecoderConfig) {
        let e = cfg.transformer_dim;
        let id = e / 2;
        let mlp = cfg.transformer_mlp_dim;
        for i in 0..cfg.transformer_depth {
            let p = format!("sam_mask_decoder.transformer.layers.{i}");
            // self_attn (downsample_rate=1 → internal_dim=e)
            {
                let sub = "self_attn";
                t.insert(format!("{p}.{sub}.q_proj.weight"), (z(e * e), vec![e, e]));
                t.insert(format!("{p}.{sub}.q_proj.bias"), (z(e), vec![e]));
                t.insert(format!("{p}.{sub}.k_proj.weight"), (z(e * e), vec![e, e]));
                t.insert(format!("{p}.{sub}.k_proj.bias"), (z(e), vec![e]));
                t.insert(format!("{p}.{sub}.v_proj.weight"), (z(e * e), vec![e, e]));
                t.insert(format!("{p}.{sub}.v_proj.bias"), (z(e), vec![e]));
                t.insert(format!("{p}.{sub}.out_proj.weight"), (z(e * e), vec![e, e]));
                t.insert(format!("{p}.{sub}.out_proj.bias"), (z(e), vec![e]));
            }
            t.insert(format!("{p}.norm1.weight"), (z(e), vec![e]));
            t.insert(format!("{p}.norm1.bias"), (z(e), vec![e]));
            // cross_attn_token_to_image, cross_attn_image_to_token (downsample_rate=2 → internal=e/2)
            for sub in ["cross_attn_token_to_image", "cross_attn_image_to_token"] {
                t.insert(format!("{p}.{sub}.q_proj.weight"), (z(e * id), vec![id, e]));
                t.insert(format!("{p}.{sub}.q_proj.bias"), (z(id), vec![id]));
                t.insert(format!("{p}.{sub}.k_proj.weight"), (z(e * id), vec![id, e]));
                t.insert(format!("{p}.{sub}.k_proj.bias"), (z(id), vec![id]));
                t.insert(format!("{p}.{sub}.v_proj.weight"), (z(e * id), vec![id, e]));
                t.insert(format!("{p}.{sub}.v_proj.bias"), (z(id), vec![id]));
                t.insert(
                    format!("{p}.{sub}.out_proj.weight"),
                    (z(e * id), vec![e, id]),
                );
                t.insert(format!("{p}.{sub}.out_proj.bias"), (z(e), vec![e]));
            }
            t.insert(format!("{p}.norm2.weight"), (z(e), vec![e]));
            t.insert(format!("{p}.norm2.bias"), (z(e), vec![e]));
            t.insert(
                format!("{p}.mlp.layers.0.weight"),
                (z(mlp * e), vec![mlp, e]),
            );
            t.insert(format!("{p}.mlp.layers.0.bias"), (z(mlp), vec![mlp]));
            t.insert(
                format!("{p}.mlp.layers.1.weight"),
                (z(mlp * e), vec![e, mlp]),
            );
            t.insert(format!("{p}.mlp.layers.1.bias"), (z(e), vec![e]));
            t.insert(format!("{p}.norm3.weight"), (z(e), vec![e]));
            t.insert(format!("{p}.norm3.bias"), (z(e), vec![e]));
            t.insert(format!("{p}.norm4.weight"), (z(e), vec![e]));
            t.insert(format!("{p}.norm4.bias"), (z(e), vec![e]));
        }
        // final_attn_token_to_image (downsample_rate=2)
        let p = "sam_mask_decoder.transformer.final_attn_token_to_image";
        t.insert(format!("{p}.q_proj.weight"), (z(e * id), vec![id, e]));
        t.insert(format!("{p}.q_proj.bias"), (z(id), vec![id]));
        t.insert(format!("{p}.k_proj.weight"), (z(e * id), vec![id, e]));
        t.insert(format!("{p}.k_proj.bias"), (z(id), vec![id]));
        t.insert(format!("{p}.v_proj.weight"), (z(e * id), vec![id, e]));
        t.insert(format!("{p}.v_proj.bias"), (z(id), vec![id]));
        t.insert(format!("{p}.out_proj.weight"), (z(e * id), vec![e, id]));
        t.insert(format!("{p}.out_proj.bias"), (z(e), vec![e]));
        t.insert(
            "sam_mask_decoder.transformer.norm_final_attn.weight".into(),
            (z(e), vec![e]),
        );
        t.insert(
            "sam_mask_decoder.transformer.norm_final_attn.bias".into(),
            (z(e), vec![e]),
        );
    }

    fn add_mask_decoder_weights(t: &mut T, cfg: &Sam2DecoderConfig) {
        let e = cfg.transformer_dim;
        let q4 = e / 4;
        let q8 = e / 8;
        t.insert(
            "sam_mask_decoder.iou_token.weight".into(),
            (z(e), vec![1, e]),
        );
        t.insert(
            "sam_mask_decoder.mask_tokens.weight".into(),
            (z(cfg.num_mask_tokens * e), vec![cfg.num_mask_tokens, e]),
        );
        if cfg.pred_obj_scores {
            t.insert(
                "sam_mask_decoder.obj_score_token.weight".into(),
                (z(e), vec![1, e]),
            );
        }
        t.insert(
            "sam_mask_decoder.output_upscaling.0.weight".into(),
            (z(e * q4 * 4), vec![e, q4, 2, 2]),
        );
        t.insert(
            "sam_mask_decoder.output_upscaling.0.bias".into(),
            (z(q4), vec![q4]),
        );
        t.insert(
            "sam_mask_decoder.output_upscaling.1.weight".into(),
            (z(q4), vec![q4]),
        );
        t.insert(
            "sam_mask_decoder.output_upscaling.1.bias".into(),
            (z(q4), vec![q4]),
        );
        t.insert(
            "sam_mask_decoder.output_upscaling.3.weight".into(),
            (z(q4 * q8 * 4), vec![q4, q8, 2, 2]),
        );
        t.insert(
            "sam_mask_decoder.output_upscaling.3.bias".into(),
            (z(q8), vec![q8]),
        );
        if cfg.use_high_res_features {
            t.insert(
                "sam_mask_decoder.conv_s0.weight".into(),
                (z(q8 * e), vec![q8, e, 1, 1]),
            );
            t.insert("sam_mask_decoder.conv_s0.bias".into(), (z(q8), vec![q8]));
            t.insert(
                "sam_mask_decoder.conv_s1.weight".into(),
                (z(q4 * e), vec![q4, e, 1, 1]),
            );
            t.insert("sam_mask_decoder.conv_s1.bias".into(), (z(q4), vec![q4]));
        }
        for i in 0..cfg.num_mask_tokens {
            let p = format!("sam_mask_decoder.output_hypernetworks_mlps.{i}");
            // 3-layer ReLU MLP: e → e → e → q8
            t.insert(format!("{p}.layers.0.weight"), (z(e * e), vec![e, e]));
            t.insert(format!("{p}.layers.0.bias"), (z(e), vec![e]));
            t.insert(format!("{p}.layers.1.weight"), (z(e * e), vec![e, e]));
            t.insert(format!("{p}.layers.1.bias"), (z(e), vec![e]));
            t.insert(format!("{p}.layers.2.weight"), (z(e * q8), vec![q8, e]));
            t.insert(format!("{p}.layers.2.bias"), (z(q8), vec![q8]));
        }
        // IoU prediction head: e → hidden → hidden → num_masks
        let p = "sam_mask_decoder.iou_prediction_head";
        let hidden = cfg.iou_head_hidden_dim;
        t.insert(
            format!("{p}.layers.0.weight"),
            (z(e * hidden), vec![hidden, e]),
        );
        t.insert(format!("{p}.layers.0.bias"), (z(hidden), vec![hidden]));
        t.insert(
            format!("{p}.layers.1.weight"),
            (z(hidden * hidden), vec![hidden, hidden]),
        );
        t.insert(format!("{p}.layers.1.bias"), (z(hidden), vec![hidden]));
        t.insert(
            format!("{p}.layers.2.weight"),
            (
                z(hidden * cfg.num_mask_tokens),
                vec![cfg.num_mask_tokens, hidden],
            ),
        );
        t.insert(
            format!("{p}.layers.2.bias"),
            (z(cfg.num_mask_tokens), vec![cfg.num_mask_tokens]),
        );
        // pred_obj_score_head MLP
        if cfg.pred_obj_scores {
            if cfg.pred_obj_scores_mlp {
                let p = "sam_mask_decoder.pred_obj_score_head";
                t.insert(format!("{p}.layers.0.weight"), (z(e * e), vec![e, e]));
                t.insert(format!("{p}.layers.0.bias"), (z(e), vec![e]));
                t.insert(format!("{p}.layers.1.weight"), (z(e * e), vec![e, e]));
                t.insert(format!("{p}.layers.1.bias"), (z(e), vec![e]));
                t.insert(format!("{p}.layers.2.weight"), (z(e), vec![1, e]));
                t.insert(format!("{p}.layers.2.bias"), (z(1), vec![1]));
            } else {
                t.insert(
                    "sam_mask_decoder.pred_obj_score_head.weight".into(),
                    (z(e), vec![1, e]),
                );
                t.insert(
                    "sam_mask_decoder.pred_obj_score_head.bias".into(),
                    (z(1), vec![1]),
                );
            }
        }
        // obj_ptr_proj MLP — top-level under SAM2Base, not nested.
        if cfg.use_object_pointer {
            if cfg.use_mlp_for_obj_ptr_proj {
                let p = "obj_ptr_proj";
                t.insert(format!("{p}.layers.0.weight"), (z(e * e), vec![e, e]));
                t.insert(format!("{p}.layers.0.bias"), (z(e), vec![e]));
                t.insert(format!("{p}.layers.1.weight"), (z(e * e), vec![e, e]));
                t.insert(format!("{p}.layers.1.bias"), (z(e), vec![e]));
                t.insert(format!("{p}.layers.2.weight"), (z(e * e), vec![e, e]));
                t.insert(format!("{p}.layers.2.bias"), (z(e), vec![e]));
            } else {
                t.insert("obj_ptr_proj.weight".into(), (z(e * e), vec![e, e]));
                t.insert("obj_ptr_proj.bias".into(), (z(e), vec![e]));
            }
        }
        add_two_way_transformer_weights(t, cfg);
    }

    fn add_memory_encoder_weights(t: &mut T, cfg: &Sam2MemoryEncoderConfig) {
        // MaskDownSampler levels.
        let mut in_c = 1usize;
        let stride2 = cfg.mask_downsampler_stride * cfg.mask_downsampler_stride;
        let mut num_levels = 0;
        let mut acc = 1usize;
        while acc < cfg.mask_downsampler_total_stride {
            acc *= cfg.mask_downsampler_stride;
            num_levels += 1;
        }
        for li in 0..num_levels {
            let out_c = in_c * stride2;
            let conv_idx = li * 3;
            let ln_idx = conv_idx + 1;
            let k = cfg.mask_downsampler_kernel;
            t.insert(
                format!("memory_encoder.mask_downsampler.encoder.{conv_idx}.weight"),
                (z(out_c * in_c * k * k), vec![out_c, in_c, k, k]),
            );
            t.insert(
                format!("memory_encoder.mask_downsampler.encoder.{conv_idx}.bias"),
                (z(out_c), vec![out_c]),
            );
            t.insert(
                format!("memory_encoder.mask_downsampler.encoder.{ln_idx}.weight"),
                (z(out_c), vec![out_c]),
            );
            t.insert(
                format!("memory_encoder.mask_downsampler.encoder.{ln_idx}.bias"),
                (z(out_c), vec![out_c]),
            );
            in_c = out_c;
        }
        let final_idx = num_levels * 3;
        t.insert(
            format!("memory_encoder.mask_downsampler.encoder.{final_idx}.weight"),
            (z(cfg.in_dim * in_c), vec![cfg.in_dim, in_c, 1, 1]),
        );
        t.insert(
            format!("memory_encoder.mask_downsampler.encoder.{final_idx}.bias"),
            (z(cfg.in_dim), vec![cfg.in_dim]),
        );
        // pix_feat_proj
        t.insert(
            "memory_encoder.pix_feat_proj.weight".into(),
            (
                z(cfg.in_dim * cfg.in_dim),
                vec![cfg.in_dim, cfg.in_dim, 1, 1],
            ),
        );
        t.insert(
            "memory_encoder.pix_feat_proj.bias".into(),
            (z(cfg.in_dim), vec![cfg.in_dim]),
        );
        // Fuser
        for i in 0..cfg.fuser_num_layers {
            let p = format!("memory_encoder.fuser.layers.{i}");
            let dim = cfg.fuser_dim;
            let k = cfg.fuser_kernel;
            if cfg.fuser_use_dwconv {
                t.insert(
                    format!("{p}.dwconv.weight"),
                    (z(dim * k * k), vec![dim, 1, k, k]),
                );
            } else {
                t.insert(
                    format!("{p}.dwconv.weight"),
                    (z(dim * dim * k * k), vec![dim, dim, k, k]),
                );
            }
            t.insert(format!("{p}.dwconv.bias"), (z(dim), vec![dim]));
            t.insert(format!("{p}.norm.weight"), (z(dim), vec![dim]));
            t.insert(format!("{p}.norm.bias"), (z(dim), vec![dim]));
            t.insert(
                format!("{p}.pwconv1.weight"),
                (z(4 * dim * dim), vec![4 * dim, dim]),
            );
            t.insert(format!("{p}.pwconv1.bias"), (z(4 * dim), vec![4 * dim]));
            t.insert(
                format!("{p}.pwconv2.weight"),
                (z(dim * 4 * dim), vec![dim, 4 * dim]),
            );
            t.insert(format!("{p}.pwconv2.bias"), (z(dim), vec![dim]));
            if cfg.fuser_layer_scale_init_value > 0.0 {
                t.insert(format!("{p}.gamma"), (z(dim), vec![dim]));
            }
        }
        // out_proj (only when dims differ)
        if cfg.in_dim != cfg.out_dim {
            t.insert(
                "memory_encoder.out_proj.weight".into(),
                (
                    z(cfg.in_dim * cfg.out_dim),
                    vec![cfg.out_dim, cfg.in_dim, 1, 1],
                ),
            );
            t.insert(
                "memory_encoder.out_proj.bias".into(),
                (z(cfg.out_dim), vec![cfg.out_dim]),
            );
        }
    }

    fn add_memory_attention_weights(t: &mut T, cfg: &Sam2MemoryConfig) {
        let d = cfg.d_model;
        let kv = cfg.kv_in_dim;
        let dff = cfg.dim_feedforward;
        for i in 0..cfg.num_layers {
            let p = format!("memory_attention.layers.{i}");
            // self_attn: q/k/v all from d → d
            {
                let sub = "self_attn";
                t.insert(format!("{p}.{sub}.q_proj.weight"), (z(d * d), vec![d, d]));
                t.insert(format!("{p}.{sub}.q_proj.bias"), (z(d), vec![d]));
                t.insert(format!("{p}.{sub}.k_proj.weight"), (z(d * d), vec![d, d]));
                t.insert(format!("{p}.{sub}.k_proj.bias"), (z(d), vec![d]));
                t.insert(format!("{p}.{sub}.v_proj.weight"), (z(d * d), vec![d, d]));
                t.insert(format!("{p}.{sub}.v_proj.bias"), (z(d), vec![d]));
                t.insert(format!("{p}.{sub}.out_proj.weight"), (z(d * d), vec![d, d]));
                t.insert(format!("{p}.{sub}.out_proj.bias"), (z(d), vec![d]));
            }
            // cross_attn_image: q from d → d, k/v from kv → d
            {
                let sub = "cross_attn_image";
                t.insert(format!("{p}.{sub}.q_proj.weight"), (z(d * d), vec![d, d]));
                t.insert(format!("{p}.{sub}.q_proj.bias"), (z(d), vec![d]));
                t.insert(format!("{p}.{sub}.k_proj.weight"), (z(d * kv), vec![d, kv]));
                t.insert(format!("{p}.{sub}.k_proj.bias"), (z(d), vec![d]));
                t.insert(format!("{p}.{sub}.v_proj.weight"), (z(d * kv), vec![d, kv]));
                t.insert(format!("{p}.{sub}.v_proj.bias"), (z(d), vec![d]));
                t.insert(format!("{p}.{sub}.out_proj.weight"), (z(d * d), vec![d, d]));
                t.insert(format!("{p}.{sub}.out_proj.bias"), (z(d), vec![d]));
            }
            t.insert(format!("{p}.norm1.weight"), (z(d), vec![d]));
            t.insert(format!("{p}.norm1.bias"), (z(d), vec![d]));
            t.insert(format!("{p}.norm2.weight"), (z(d), vec![d]));
            t.insert(format!("{p}.norm2.bias"), (z(d), vec![d]));
            t.insert(format!("{p}.norm3.weight"), (z(d), vec![d]));
            t.insert(format!("{p}.norm3.bias"), (z(d), vec![d]));
            t.insert(format!("{p}.linear1.weight"), (z(dff * d), vec![dff, d]));
            t.insert(format!("{p}.linear1.bias"), (z(dff), vec![dff]));
            t.insert(format!("{p}.linear2.weight"), (z(d * dff), vec![d, dff]));
            t.insert(format!("{p}.linear2.bias"), (z(d), vec![d]));
        }
        t.insert("memory_attention.norm.weight".into(), (z(d), vec![d]));
        t.insert("memory_attention.norm.bias".into(), (z(d), vec![d]));
    }

    fn synthetic_full_sam2_weights(cfg: &Sam2Config) -> WeightMap {
        let mut t: T = HashMap::new();
        add_hiera_weights(&mut t, &cfg.hiera);
        add_prompt_encoder_weights(&mut t, cfg.decoder.transformer_dim, SAM2_MASK_IN_CHANS);
        add_mask_decoder_weights(&mut t, &cfg.decoder);
        add_memory_encoder_weights(&mut t, &cfg.memory_encoder);
        add_memory_attention_weights(&mut t, &cfg.memory);
        WeightMap::from_tensors(t)
    }

    fn assert_encoder_builds(cfg: Sam2HieraConfig) {
        let mut t: T = HashMap::new();
        add_hiera_weights(&mut t, &cfg);
        let mut wm = WeightMap::from_tensors(t);
        let (g, _params, _pre, _fpn) = build_sam2_image_encoder_graph(&cfg, &mut wm)
            .unwrap_or_else(|e| panic!("encoder build failed: {e}"));
        assert_eq!(g.outputs.len(), cfg.stages.len());
        for (s, out_id) in g.outputs.iter().copied().enumerate() {
            let shape = g.shape(out_id);
            let dims: Vec<usize> = shape.dims().iter().map(|d| d.unwrap_static()).collect();
            let hw_s = cfg.grid_size_at_stage(s);
            let dim_s = cfg.embed_dim_at_stage(s);
            assert_eq!(dims, vec![1, hw_s, hw_s, dim_s], "stage {s} shape mismatch");
        }
        let leftovers: Vec<&str> = wm.keys().collect();
        assert!(leftovers.is_empty(), "leftover weights: {leftovers:?}");
    }

    #[test]
    fn encoder_graph_builds_tiny() {
        assert_encoder_builds(Sam2HieraConfig::tiny());
    }

    #[test]
    fn encoder_graph_builds_small() {
        assert_encoder_builds(Sam2HieraConfig::small());
    }

    #[test]
    fn encoder_graph_builds_base_plus() {
        assert_encoder_builds(Sam2HieraConfig::base_plus());
    }

    #[test]
    fn encoder_graph_builds_large() {
        assert_encoder_builds(Sam2HieraConfig::large());
    }

    #[test]
    fn preprocess_round_trip_shapes() {
        let img = vec![64u8; 80 * 120 * 3];
        let nchw = preprocess_image(&img, 80, 120);
        assert_eq!(nchw.len(), 3 * 1024 * 1024);
    }

    #[test]
    fn fpn_neck_runs_on_synth_outputs() {
        let cfg = Sam2HieraConfig::base_plus();
        let mut t: T = HashMap::new();
        add_hiera_weights(&mut t, &cfg);
        let mut wm = WeightMap::from_tensors(t);
        let (_g, _p, _pre, neck) = build_sam2_image_encoder_graph(&cfg, &mut wm).unwrap();

        let stage_hw: Vec<(usize, usize)> = (0..cfg.stages.len())
            .map(|s| (cfg.grid_size_at_stage(s), cfg.grid_size_at_stage(s)))
            .collect();
        let stage_dims: Vec<usize> = (0..cfg.stages.len())
            .map(|s| cfg.embed_dim_at_stage(s))
            .collect();
        let stage_outputs: Vec<Vec<f32>> = stage_hw
            .iter()
            .zip(&stage_dims)
            .map(|(&(h, w), &d)| vec![0f32; h * w * d])
            .collect();

        let mut fpn_ir = super::fpn_neck_ir::compile_fpn_neck_ir(
            &neck,
            &stage_hw,
            &stage_dims,
            Device::Cpu,
            &rlx_flow::CompileProfile::sam2(),
        )
        .unwrap();
        let levels =
            apply_fpn_neck(&neck, &mut fpn_ir, &stage_outputs, &stage_hw, &stage_dims).unwrap();
        let levels_host = apply_fpn_neck_host(&neck, &stage_outputs, &stage_hw, &stage_dims);
        assert_eq!(levels.len(), levels_host.len());
        for (a, b) in levels.iter().zip(&levels_host) {
            assert_eq!(a.features.len(), b.features.len());
            assert_eq!(a.h, b.h);
            assert_eq!(a.w, b.w);
            let fd = a
                .features
                .iter()
                .zip(&b.features)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert!(
                fd < 1e-4,
                "FPN IR vs host max |Δ| = {fd:.3e} at level {}×{}",
                a.h,
                a.w
            );
        }
        assert_eq!(levels.len(), 4);
        assert_eq!((levels[0].h, levels[0].w), (256, 256));
        assert_eq!((levels[3].h, levels[3].w), (32, 32));
    }

    #[test]
    fn full_weight_extraction_drains_map() {
        // End-to-end: build the synthetic WeightMap for *every* SAM 2
        // component, instantiate via the same code paths Sam2 uses, and
        // assert no expected keys are left over.
        let cfg = Sam2Config::hiera_base_plus();
        let mut wm = synthetic_full_sam2_weights(&cfg);

        // Mirror Sam2::from_safetensors_on weight extraction order.
        let (_g, _p, _pre, _fpn) = build_sam2_image_encoder_graph(&cfg.hiera, &mut wm).unwrap();
        let _ = prompt_encoder::extract_prompt_encoder_weights(
            &mut wm,
            cfg.decoder.transformer_dim,
            SAM2_MASK_IN_CHANS,
        )
        .unwrap();
        let _ = mask_decoder::extract_mask_decoder_weights(&mut wm, &cfg.decoder).unwrap();
        let _ =
            memory_encoder::extract_memory_encoder_weights(&mut wm, &cfg.memory_encoder).unwrap();
        let _ = memory_attention::extract_memory_attention_weights(&mut wm, &cfg.memory).unwrap();

        let leftovers: Vec<&str> = wm.keys().collect();
        assert!(
            leftovers.is_empty(),
            "leftover weights after full extraction: {leftovers:?}"
        );
    }

    #[test]
    fn prompt_encoder_no_prompt_produces_pe_and_no_mask() {
        let cfg = Sam2Config::hiera_base_plus();
        let mut wm = synthetic_full_sam2_weights(&cfg);
        // Drain encoder + FPN keys to keep the test focused.
        let (_g, _p, _pre, _fpn) = build_sam2_image_encoder_graph(&cfg.hiera, &mut wm).unwrap();
        let pe = prompt_encoder::extract_prompt_encoder_weights(
            &mut wm,
            cfg.decoder.transformer_dim,
            SAM2_MASK_IN_CHANS,
        )
        .unwrap();
        let mut mask_stack =
            super::prompt_mask_ir::Sam2PromptMaskCompiled::compile(&pe, Device::Cpu).unwrap();
        let out = prompt_encoder_forward(&pe, &mut mask_stack, None, None, None).unwrap();
        assert_eq!(out.num_sparse_tokens, 0);
        assert_eq!(
            out.dense_embeddings.len(),
            cfg.decoder.transformer_dim * SAM2_PROMPT_GRID * SAM2_PROMPT_GRID
        );
        assert_eq!(
            out.image_pe.len(),
            cfg.decoder.transformer_dim * SAM2_PROMPT_GRID * SAM2_PROMPT_GRID
        );
    }

    #[test]
    fn mask_decoder_runs_on_zero_inputs() {
        let cfg = Sam2Config::hiera_base_plus();
        let mut wm = synthetic_full_sam2_weights(&cfg);
        let (_g, _p, _pre, _fpn) = build_sam2_image_encoder_graph(&cfg.hiera, &mut wm).unwrap();
        let _pe = prompt_encoder::extract_prompt_encoder_weights(
            &mut wm,
            cfg.decoder.transformer_dim,
            SAM2_MASK_IN_CHANS,
        )
        .unwrap();
        let dec = mask_decoder::extract_mask_decoder_weights(&mut wm, &cfg.decoder).unwrap();
        let _ =
            memory_encoder::extract_memory_encoder_weights(&mut wm, &cfg.memory_encoder).unwrap();
        let _ = memory_attention::extract_memory_attention_weights(&mut wm, &cfg.memory).unwrap();

        let e = cfg.decoder.transformer_dim;
        let g = SAM2_PROMPT_GRID;
        let image_emb = vec![0f32; e * g * g];
        let image_pe = vec![0f32; e * g * g];
        let dense = vec![0f32; e * g * g];
        let sparse: Vec<f32> = Vec::new();
        let s0 = vec![0f32; e * (4 * g) * (4 * g)];
        let s1 = vec![0f32; e * (2 * g) * (2 * g)];

        let mut upscale =
            super::upscale_ir::Sam2MaskUpscaleCompiled::compile(&dec, g, Device::Cpu).unwrap();
        let mut hyper_matmul = rlx_sam_ir::mask_hyper_matmul_ir::MaskHyperMatmulCompiled::compile(
            dec.num_mask_tokens,
            cfg.decoder.transformer_dim / 8,
            g,
            Device::Cpu,
        )
        .unwrap();
        let mut hyper_mlps_ir =
            super::mlp_ir::compile_hyper_mlps(&dec.hyper_mlps, Device::Cpu).unwrap();
        let mut iou_head_ir = super::mlp_ir::compile_hyper_mlp(&dec.iou_head, Device::Cpu).unwrap();
        let mut obj_score_head_ir =
            super::mlp_ir::compile_optional_hyper_mlp(&dec.obj_score_head, 1, Device::Cpu).unwrap();
        let obj_ptr_rows = super::mlp_ir::obj_ptr_proj_rows(
            dec.num_mask_tokens,
            dec.use_multimask_token_for_obj_ptr,
        );
        let mut obj_ptr_proj_ir =
            super::mlp_ir::compile_optional_hyper_mlp(&dec.obj_ptr_proj, obj_ptr_rows, Device::Cpu)
                .unwrap();
        let s_tok = if dec.obj_score_token.is_some() { 1 } else { 0 };
        let base_q_n = s_tok + 1 + dec.num_mask_tokens;
        let mut tw_ir = super::transformer_ir::compile_two_way_transformer(
            &dec.transformer,
            base_q_n,
            g,
            Device::Cpu,
        )
        .unwrap();
        let out = mask_decoder_forward(
            &dec,
            &mut upscale,
            Some(&mut hyper_matmul),
            Some(&mut hyper_mlps_ir),
            Some(&mut iou_head_ir),
            obj_score_head_ir.as_mut(),
            obj_ptr_proj_ir.as_mut(),
            Some(&mut tw_ir),
            &image_emb,
            &image_pe,
            &sparse,
            0,
            &dense,
            Some((&s0, &s1)),
            /*multimask_output=*/ true,
            g,
        )
        .unwrap();
        assert_eq!(out.num_masks, 3);
        assert_eq!(out.h_out, 4 * g);
        assert_eq!(out.w_out, 4 * g);
        assert_eq!(out.masks.len(), 3 * out.h_out * out.w_out);
        assert_eq!(out.iou_pred.len(), 3);
        // pred_obj_scores=true → object_score_logits is from the MLP head (single scalar).
        assert_eq!(out.object_score_logits.len(), 1);
    }

    #[test]
    fn memory_encoder_prefix_matches_split_ir() {
        let cfg = Sam2Config::hiera_base_plus();
        let mut wm = synthetic_full_sam2_weights(&cfg);
        let (_g, _p, _pre, _fpn) = build_sam2_image_encoder_graph(&cfg.hiera, &mut wm).unwrap();
        let mem =
            memory_encoder::extract_memory_encoder_weights(&mut wm, &cfg.memory_encoder).unwrap();

        let pix = vec![0.1f32; cfg.memory_encoder.in_dim * SAM2_PROMPT_GRID * SAM2_PROMPT_GRID];
        let mask = vec![0.5f32; SAM2_IMG_SIZE * SAM2_IMG_SIZE];

        let mut md = memory_mask_ir::Sam2MemoryMaskDownCompiled::compile(
            &mem.mask_downsampler,
            SAM2_IMG_SIZE,
            SAM2_IMG_SIZE,
            Device::Cpu,
        )
        .unwrap();
        let mut pp = memory_mask_ir::Sam2MemoryConv1x1Compiled::compile(
            mem.in_dim,
            mem.in_dim,
            SAM2_PROMPT_GRID,
            SAM2_PROMPT_GRID,
            &mem.pix_feat_proj_w,
            &mem.pix_feat_proj_b,
            Device::Cpu,
        )
        .unwrap();
        let m_down = md.run(&mask).unwrap();
        let mut split = pp.run(&pix).unwrap();
        for i in 0..split.len() {
            split[i] += m_down[i];
        }

        let mut prefix = memory_mask_ir::Sam2MemoryPrefixCompiled::compile(
            &mem.mask_downsampler,
            mem.in_dim,
            SAM2_IMG_SIZE,
            SAM2_IMG_SIZE,
            SAM2_PROMPT_GRID,
            SAM2_PROMPT_GRID,
            &mem.pix_feat_proj_w,
            &mem.pix_feat_proj_b,
            Device::Cpu,
        )
        .unwrap();
        let fused = prefix.run(&mask, &pix).unwrap();
        assert_eq!(split.len(), fused.len());
        let fd = split
            .iter()
            .zip(&fused)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(fd < 1e-4, "prefix vs split max |Δ| = {fd:.3e}");
    }

    #[test]
    fn memory_encoder_shapes_match_for_b_plus() {
        let cfg = Sam2Config::hiera_base_plus();
        let mut wm = synthetic_full_sam2_weights(&cfg);
        let (_g, _p, _pre, _fpn) = build_sam2_image_encoder_graph(&cfg.hiera, &mut wm).unwrap();
        let _ = prompt_encoder::extract_prompt_encoder_weights(
            &mut wm,
            cfg.decoder.transformer_dim,
            SAM2_MASK_IN_CHANS,
        )
        .unwrap();
        let _ = mask_decoder::extract_mask_decoder_weights(&mut wm, &cfg.decoder).unwrap();
        let mut mem =
            memory_encoder::extract_memory_encoder_weights(&mut wm, &cfg.memory_encoder).unwrap();
        memory_encoder::compile_memory_encoder_ir(
            &mut mem,
            SAM2_IMG_SIZE,
            SAM2_IMG_SIZE,
            SAM2_PROMPT_GRID,
            SAM2_PROMPT_GRID,
            Device::Cpu,
            &rlx_flow::CompileProfile::sam2(),
        )
        .unwrap();
        let _ = memory_attention::extract_memory_attention_weights(&mut wm, &cfg.memory).unwrap();

        let pix = vec![0f32; cfg.memory_encoder.in_dim * SAM2_PROMPT_GRID * SAM2_PROMPT_GRID];
        let mask = vec![0f32; SAM2_IMG_SIZE * SAM2_IMG_SIZE];
        let out = memory_encoder_forward(
            &mut mem,
            &pix,
            &mask,
            SAM2_PROMPT_GRID,
            SAM2_PROMPT_GRID,
            /*skip_mask_sigmoid=*/ true,
        )
        .unwrap();
        assert_eq!(out.h, SAM2_PROMPT_GRID);
        assert_eq!(out.w, SAM2_PROMPT_GRID);
        assert_eq!(
            out.features.len(),
            cfg.memory_encoder.out_dim * SAM2_PROMPT_GRID * SAM2_PROMPT_GRID
        );
        // PE channel count = 2 · num_pos_feats.
        assert_eq!(
            out.pos.len(),
            2 * cfg.memory_encoder.pe_num_pos_feats * SAM2_PROMPT_GRID * SAM2_PROMPT_GRID
        );
    }

    #[test]
    fn memory_attention_runs_on_zero_inputs() {
        let cfg = Sam2Config::hiera_base_plus();
        let mut wm = synthetic_full_sam2_weights(&cfg);
        let (_g, _p, _pre, _fpn) = build_sam2_image_encoder_graph(&cfg.hiera, &mut wm).unwrap();
        let _ = prompt_encoder::extract_prompt_encoder_weights(
            &mut wm,
            cfg.decoder.transformer_dim,
            SAM2_MASK_IN_CHANS,
        )
        .unwrap();
        let _ = mask_decoder::extract_mask_decoder_weights(&mut wm, &cfg.decoder).unwrap();
        let _ =
            memory_encoder::extract_memory_encoder_weights(&mut wm, &cfg.memory_encoder).unwrap();
        let mat = memory_attention::extract_memory_attention_weights(&mut wm, &cfg.memory).unwrap();

        let [end_x, end_y] = cfg.memory.rope_feat_size;
        let n_img = end_x * end_y;
        let d = cfg.memory.d_model;
        let kv = cfg.memory.kv_in_dim;
        let curr = vec![0f32; n_img * d];
        let curr_pos = vec![0f32; n_img * d];
        // 1 frame of memory.
        let n_mem = end_x * end_y;
        let memory = vec![0f32; n_mem * kv];
        let memory_pos = vec![0f32; n_mem * kv];
        let out = memory_attention_forward(
            &mat,
            &curr,
            &curr_pos,
            &memory,
            &memory_pos,
            n_img,
            n_mem,
            kv,
            /*num_obj_ptr_tokens=*/ 0,
        )
        .unwrap();
        assert_eq!(out.len(), n_img * d);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
