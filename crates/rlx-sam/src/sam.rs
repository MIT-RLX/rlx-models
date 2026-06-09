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

//! SAM v1 top-level orchestrator — ties the IR-graph image encoder
//! together with the host-side prompt encoder + mask decoder.
//!
//! Mirrors `candle-transformers/src/models/segment_anything/sam.rs`
//! at the API level. The image encoder runs on the rlx-runtime
//! `Session`; mask/prompt conv stacks use IR ops on the same device.

use super::config::{SAM_EMBED_HW, SAM_IMG_SIZE, SAM_PROMPT_EMBED_DIM, SamConfig};
use super::image_encoder::build_sam_encoder_graph;
use super::mask_decoder::{MaskDecoderWeights, extract_mask_decoder_weights, mask_decoder_forward};
use super::preprocess::{SamPreprocessWeights, assemble_patch_tokens, preprocess_image};
use super::prompt_encoder::{
    PromptEncoderOutput, PromptEncoderWeights, extract_prompt_encoder_weights,
    prompt_encoder_forward,
};
use super::prompt_mask_ir::SamPromptMaskCompiled;
use super::upscale_ir::SamMaskUpscaleCompiled;
use anyhow::Result;
use rlx_runtime::{CompiledGraph, Device, Session};
use rlx_sam_ir::mask_hyper_matmul_ir::MaskHyperMatmulCompiled;
use rlx_sam_ir::mlp_relu_ir::MlpReluCompiled;
use std::path::Path;

/// Mask channels used by the prompt encoder's mask-downscaling stack.
/// candle's `Sam::new` hardcodes 16 across all ViT variants.
pub const SAM_MASK_IN_CHANS: usize = 16;

/// Full SAM model — owns the compiled image encoder + all decoder
/// weights. Stateless wrt prompts: every call to [`Sam::forward`]
/// runs the cached encoder + a fresh decoder forward.
pub struct Sam {
    cfg: SamConfig,
    encoder: CompiledGraph,
    pre: SamPreprocessWeights,
    prompt_enc: PromptEncoderWeights,
    mask_stack: SamPromptMaskCompiled,
    mask_dec: MaskDecoderWeights,
    upscale: SamMaskUpscaleCompiled,
    hyper_matmul: MaskHyperMatmulCompiled,
    hyper_mlps_ir: Vec<MlpReluCompiled>,
    iou_head_ir: MlpReluCompiled,
    tw_ir: rlx_sam_ir::twoway_transformer_ir::TwoWayTransformerCompiled,
}

impl Sam {
    /// Load SAM ViT-B (or L/H — pass the matching config) from a
    /// safetensors checkpoint, compiling the image encoder for the
    /// CPU backend. For GPU/Metal/MLX, use
    /// [`Sam::from_safetensors_on`].
    pub fn from_safetensors(weights_path: &str, cfg: SamConfig) -> Result<Self> {
        Self::from_safetensors_on(weights_path, cfg, Device::Cpu)
    }

    /// Same as [`Sam::from_safetensors`] but compiles the image
    /// encoder for the given device. Requires the matching backend
    /// feature on `rlx-models`:
    ///
    /// | feature   | backend           |
    /// |-----------|-------------------|
    /// | `metal`   | `Device::Metal`   |
    /// | `mlx`     | `Device::Mlx`     |
    /// | `gpu`     | `Device::Gpu`     |
    /// | `cuda`    | `Device::Cuda`    |
    /// | `rocm`    | `Device::Rocm`    |
    /// | `tpu`     | `Device::Tpu`     |
    ///
    pub fn from_safetensors_on(weights_path: &str, cfg: SamConfig, device: Device) -> Result<Self> {
        rlx_core::validate_sam_device("sam", device)?;
        let mut wm = rlx_core::load_weight_map(Path::new(weights_path), rlx_core::SAM_GGUF_ARCHES)?;
        let (graph, params, pre) = build_sam_encoder_graph(&cfg.encoder, &mut wm)?;
        let profile = crate::profile::sam_profile_near_weights(std::path::Path::new(weights_path));
        let opts = rlx_core::flow_bridge::compile_options_for_profile(&profile, device);
        let mut encoder = Session::new(device).compile_with(graph, &opts);
        for (name, data) in &params {
            encoder.set_param(name, data);
        }
        let prompt_enc =
            extract_prompt_encoder_weights(&mut wm, cfg.encoder.out_chans, SAM_MASK_IN_CHANS)?;
        let mask_stack =
            SamPromptMaskCompiled::compile_with_profile(&prompt_enc, device, &profile)?;
        let mask_dec = extract_mask_decoder_weights(
            &mut wm,
            cfg.decoder.transformer_dim,
            cfg.decoder.num_mask_tokens,
            cfg.decoder.iou_head_depth,
            cfg.decoder.iou_head_hidden_dim,
            cfg.decoder.transformer_depth,
            cfg.decoder.transformer_num_heads,
            cfg.decoder.transformer_mlp_dim,
        )?;
        let upscale = SamMaskUpscaleCompiled::compile_with_profile(&mask_dec, device, &profile)?;
        let hyper_matmul = MaskHyperMatmulCompiled::compile_with_profile(
            mask_dec.num_mask_tokens,
            cfg.decoder.transformer_dim / 8,
            SAM_EMBED_HW,
            device,
            &profile,
        )?;
        let hyper_mlps_ir =
            super::mlp_ir::compile_hyper_mlps_with_profile(&mask_dec.hyper_mlps, device, &profile)?;
        let iou_head_ir =
            super::mlp_ir::compile_iou_head_with_profile(&mask_dec.iou_head, device, &profile)?;
        let base_q_n = 1 + mask_dec.num_mask_tokens;
        let tw_ir = super::transformer_ir::compile_two_way_transformer_with_profile(
            &mask_dec.transformer,
            base_q_n,
            SAM_EMBED_HW,
            device,
            &profile,
        )?;
        Ok(Self {
            cfg,
            encoder,
            pre,
            prompt_enc,
            mask_stack,
            mask_dec,
            upscale,
            hyper_matmul,
            hyper_mlps_ir,
            iou_head_ir,
            tw_ir,
        })
    }

    /// Encode an image into the `[256, 64, 64]` image embedding.
    /// `image_nchw`: pre-padded `[3, 1024, 1024]` NCHW f32 tensor
    /// (see [`super::preprocess::preprocess_image`]).
    pub fn encode_image(&mut self, image_nchw: &[f32]) -> Vec<f32> {
        let hidden = assemble_patch_tokens(&self.pre, image_nchw).expect("assemble_patch_tokens");
        let outputs = self.encoder.run(&[("hidden", hidden.as_slice())]);
        outputs.into_iter().next().expect("encoder output")
    }

    /// Run the prompt encoder + mask decoder on a pre-encoded image.
    pub fn predict_masks(
        &mut self,
        image_embeddings: &[f32],
        points: Option<(&[f32], &[f32])>,
        boxes: Option<&[f32]>,
        masks: Option<&[f32]>,
        multimask_output: bool,
    ) -> Result<MaskPrediction> {
        let pe: PromptEncoderOutput =
            prompt_encoder_forward(&self.prompt_enc, &mut self.mask_stack, points, boxes, masks)?;
        let (mask_logits, iou_pred, num_masks, mask_side) = mask_decoder_forward(
            &self.mask_dec,
            &mut self.upscale,
            Some(&mut self.hyper_matmul),
            Some(&mut self.hyper_mlps_ir),
            Some(&mut self.iou_head_ir),
            Some(&mut self.tw_ir),
            image_embeddings,
            &pe.image_pe,
            &pe.sparse_embeddings,
            pe.num_sparse_tokens,
            &pe.dense_embeddings,
            multimask_output,
        )?;
        Ok(MaskPrediction {
            mask_logits,
            iou_pred,
            num_masks,
            mask_side,
        })
    }

    /// End-to-end forward: image bytes → masks. `rgb` is HWC u8.
    pub fn forward(
        &mut self,
        rgb: &[u8],
        h_in: usize,
        w_in: usize,
        points: Option<(&[f32], &[f32])>,
        boxes: Option<&[f32]>,
        masks: Option<&[f32]>,
        multimask_output: bool,
    ) -> Result<(MaskPrediction, (usize, usize))> {
        let (image_nchw, (resized_h, resized_w)) = preprocess_image(rgb, h_in, w_in);
        let image_embeddings = self.encode_image(&image_nchw);
        let pred = self.predict_masks(&image_embeddings, points, boxes, masks, multimask_output)?;
        Ok((pred, (resized_h, resized_w)))
    }

    pub fn config(&self) -> &SamConfig {
        &self.cfg
    }

    /// Spatial side length of the predicted mask logits (= 4 · hw = 256
    /// for ViT-B at 1024 input).
    pub fn mask_side(&self) -> usize {
        4 * SAM_EMBED_HW
    }

    /// Image side that the model operates on internally.
    pub fn input_image_size(&self) -> usize {
        SAM_IMG_SIZE
    }
}

/// Output of [`Sam::predict_masks`] / [`Sam::forward`].
pub struct MaskPrediction {
    /// `[num_masks, mask_side, mask_side]` mask logits in the encoder's
    /// 4×-upscaled space. Threshold > 0 to get binary masks; further
    /// upscale + crop back to the original image as needed.
    pub mask_logits: Vec<f32>,
    /// `[num_masks]` per-mask IoU prediction (model-self-estimated
    /// mask quality).
    pub iou_pred: Vec<f32>,
    pub num_masks: usize,
    pub mask_side: usize,
}

impl MaskPrediction {
    /// Convenience: drop the largest predicted-IoU index. Returns
    /// `Some((index, iou))`.
    pub fn best_by_iou(&self) -> Option<(usize, f32)> {
        self.iou_pred
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, v)| (i, *v))
    }
}

/// Drop-in default config matching candle's `Sam::new()` for ViT-B
/// (the `lmz/candle-sam/sam_vit_b_01ec64.safetensors` checkpoint).
pub fn sam_vit_b_config() -> SamConfig {
    SamConfig::vit_b()
}

#[allow(dead_code)]
fn _silence_unused() {
    let _ = SAM_PROMPT_EMBED_DIM;
}
