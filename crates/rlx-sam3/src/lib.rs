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

//! SAM 3 — Meta's Segment Anything with Concepts.
//!
//! This module targets the base SAM3 image + video architecture from
//! `facebookresearch/sam3`. SAM3.1 multiplex is intentionally separate.

pub mod cli;
pub mod config;
pub use rlx_sam::profile::{
    SAM_PROFILE_FILE, sam_profile_near_weights, sam3_profile_default, sam3_profile_near_weights,
};
pub mod detector;
pub mod detector_decoder;
pub mod detector_decoder_ir;
pub mod detector_encoder;
pub mod detector_encoder_ir;
pub mod flow;
pub mod geometry;
pub mod gguf_ir;
pub mod neck;
pub mod neck_branch_ir;
pub mod packed_gguf;
pub mod preprocess;
#[allow(clippy::module_inception)]
pub mod sam3;
pub mod segmentation_head;
pub mod segmentation_pixel_ir;
/// Host tensor kernels (shared with FLUX.2 / V-JEPA2 via `rlx-tensor`).
pub mod tensor {
    pub use rlx_core::host_kernels::*;
}
pub mod text_encoder;
pub mod tracker;
pub mod vision_encoder;
pub mod vision_encoder_ir;

pub use config::{
    SAM3_DET_DIM, SAM3_IMG_SIZE, SAM3_PATCH_GRID, SAM3_PATCH_SIZE, SAM3_PIXEL_MEAN, SAM3_PIXEL_STD,
    SAM3_VISION_DIM, Sam3Config, Sam3DetectorConfig, Sam3TextConfig, Sam3TrackerConfig,
    Sam3VitConfig,
};
pub use detector_decoder_ir::Sam3CompiledDecoder;
pub use detector_decoder_ir::{forward_decoder_ir_on, forward_decoder_ir_on_with_profile};
pub use detector_encoder_ir::build_sam3_detector_encoder_graph;
pub use detector_encoder_ir::{forward_encoder_ir_on, forward_encoder_ir_on_with_profile};
pub use flow::{
    Sam3DetectorDecoderBuilt, Sam3DetectorDecoderFlow, Sam3DetectorEncoderFlow,
    build_sam3_detector_decoder_built, build_sam3_detector_decoder_built_with_profile,
    build_sam3_detector_encoder_built, build_sam3_detector_encoder_built_with_profile,
    build_sam3_detector_encoder_model_flow,
};
pub use packed_gguf::{
    gguf_has_packed_linears, load_sam3_from_gguf, take_or_gguf, take_transposed_or_gguf,
};
pub use preprocess::{Sam3PreprocessWeights, assemble_patch_tokens, preprocess_image};
pub use sam3::{
    Sam3, Sam3EncodedImage, Sam3ImagePrediction, Sam3VideoFramePrediction, Sam3VideoState,
};

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_core::weight_map::WeightMap;
    use std::collections::HashMap;

    #[test]
    fn sam3_base_config_matches_public_builder() {
        let cfg = Sam3Config::base();
        assert_eq!(cfg.vit.img_size, 1008);
        assert_eq!(cfg.vit.patch_size, 14);
        assert_eq!(cfg.vit.patch_grid(), 72);
        assert_eq!(cfg.vit.embed_dim, 1024);
        assert_eq!(cfg.vit.global_att_blocks, vec![7, 15, 23, 31]);
        assert_eq!(cfg.detector.num_queries, 200);
        assert_eq!(cfg.tracker.num_maskmem, 7);
    }

    #[test]
    fn preprocess_extracts_sam3_patch_weights() {
        let mut cfg = Sam3VitConfig::base();
        cfg.use_abs_pos = false;
        let ps = cfg.patch_size;
        let e = cfg.embed_dim;
        let pd = 3 * ps * ps;
        let mut tensors = HashMap::new();
        tensors.insert(
            "detector.backbone.visual.trunk.patch_embed.proj.weight".to_string(),
            (vec![0.0f32; e * pd], vec![e, 3, ps, ps]),
        );
        let mut wm = WeightMap::from_tensors(tensors);
        let pre = preprocess::extract_preprocess_weights(&mut wm, &cfg).unwrap();
        assert_eq!(pre.patch_proj_w.len(), e * pd);
        assert_eq!(pre.patch_proj_b.len(), e);
        assert!(pre.pos_embed.is_none());
        assert!(wm.is_empty());
    }

    #[test]
    fn assemble_patch_tokens_shape_is_sam3_grid() {
        let pre = Sam3PreprocessWeights {
            patch_proj_w: vec![0.0; 3 * SAM3_PATCH_SIZE * SAM3_PATCH_SIZE * SAM3_VISION_DIM],
            patch_proj_b: vec![1.0; SAM3_VISION_DIM],
            pos_embed: None,
            embed_dim: SAM3_VISION_DIM,
            patch_size: SAM3_PATCH_SIZE,
            grid: SAM3_PATCH_GRID,
        };
        let image = vec![0.0f32; 3 * SAM3_IMG_SIZE * SAM3_IMG_SIZE];
        let tokens = assemble_patch_tokens(&pre, &image).unwrap();
        assert_eq!(
            tokens.len(),
            SAM3_PATCH_GRID * SAM3_PATCH_GRID * SAM3_VISION_DIM
        );
        assert_eq!(tokens[0], 1.0);
    }
}
