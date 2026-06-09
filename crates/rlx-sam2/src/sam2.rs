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

//! SAM 2 top-level orchestrator — ties together the IR-graph Hiera
//! image encoder, the host-side FpnNeck, prompt encoder, mask decoder,
//! memory encoder, and memory attention into the two reference APIs:
//!
//!   - [`Sam2::predict_image`] — single-image segmentation (matches
//!     `SAM2ImagePredictor.predict` in spirit).
//!   - [`Sam2::predict_video_frame`] — stateful per-frame call with a
//!     [`Sam2VideoState`] memory bank (mirrors `SAM2VideoPredictor`).
//!
//! The image encoder is compiled once on the chosen
//! [`rlx_runtime::Device`]; every other component runs host-side
//! because their compute is < 1 % of total per inference and the IR
//! surface to support them all (cross-attention with kv_in_dim,
//! depthwise Conv2d, ConvTranspose2d, sigmoid, etc.) isn't worth
//! growing for a fraction of a millisecond's win.

use super::config::{SAM2_IMG_SIZE, Sam2Config, Sam2DecoderConfig};
use super::fpn_neck::{FpnLevel, FpnNeckWeights, apply_fpn_neck};
use super::fpn_neck_ir::{Sam2FpnNeckIr, compile_fpn_neck_ir};
use super::image_encoder::build_sam2_image_encoder_graph;
use super::mask_decoder::{
    Sam2MaskDecoderOutput, Sam2MaskDecoderWeights, extract_mask_decoder_weights,
    mask_decoder_forward,
};
use super::memory_attention::{
    Sam2MemoryAttentionWeights, extract_memory_attention_weights, memory_attention_forward,
};
use super::memory_attention_ir::{MemoryAttentionCompiled, max_memory_slots};
use super::memory_encoder::{
    Sam2MemoryEncoderOutput, Sam2MemoryEncoderWeights, extract_memory_encoder_weights,
    memory_encoder_forward,
};
use super::preprocess::{Sam2PreprocessWeights, assemble_patch_tokens, preprocess_image};
use super::prompt_encoder::{
    SAM2_MASK_IN_CHANS, SAM2_PROMPT_GRID, Sam2PromptEncoderOutput, Sam2PromptEncoderWeights,
    extract_prompt_encoder_weights, prompt_encoder_forward,
};
use super::prompt_mask_ir::Sam2PromptMaskCompiled;
use super::upscale_ir::Sam2MaskUpscaleCompiled;
use anyhow::{Result, ensure};
use rlx_flow::CompileProfile;
use rlx_runtime::{CompiledGraph, Device, Session};
use rlx_sam::profile::sam2_profile_near_weights;
use rlx_sam_ir::mask_hyper_matmul_ir::MaskHyperMatmulCompiled;
use rlx_sam_ir::mlp_relu_ir::MlpReluCompiled;
use std::path::Path;

/// SAM 2 image-encoder hiera stage spec — needed by the host-side FPN.
#[derive(Clone)]
struct HieraOutputShapes {
    stage_hw: Vec<(usize, usize)>,
    stage_dims: Vec<usize>,
}

/// Full SAM 2 model — owns the compiled image encoder + every
/// host-side weight bundle. The encoder result is recomputed per call
/// (no encoder-caching here; layer above can wrap if needed).
pub struct Sam2 {
    cfg: Sam2Config,
    encoder: CompiledGraph,
    pre: Sam2PreprocessWeights,
    fpn: FpnNeckWeights,
    fpn_ir: Sam2FpnNeckIr,
    prompt_enc: Sam2PromptEncoderWeights,
    mask_dec: Sam2MaskDecoderWeights,
    mask_stack: Sam2PromptMaskCompiled,
    upscale: Sam2MaskUpscaleCompiled,
    hyper_matmul: MaskHyperMatmulCompiled,
    hyper_mlps_ir: Vec<MlpReluCompiled>,
    iou_head_ir: MlpReluCompiled,
    obj_score_head_ir: Option<MlpReluCompiled>,
    obj_ptr_proj_ir: Option<MlpReluCompiled>,
    tw_ir: rlx_sam_ir::twoway_transformer_ir::TwoWayTransformerCompiled,
    mem_enc: Sam2MemoryEncoderWeights,
    mem_attn: Sam2MemoryAttentionWeights,
    /// Compiled on first video frame (avoids multi-minute compile at checkpoint load).
    mem_attn_ir: Option<MemoryAttentionCompiled>,
    mem_attn_device: Device,
    hiera_shapes: HieraOutputShapes,
    compile_profile: CompileProfile,
}

impl Sam2 {
    /// Load every SAM 2 component from a safetensors checkpoint and
    /// compile the image encoder for the CPU backend. For GPU/Metal,
    /// see [`Sam2::from_safetensors_on`].
    pub fn from_safetensors(weights_path: &str, cfg: Sam2Config) -> Result<Self> {
        Self::from_safetensors_on(weights_path, cfg, Device::Cpu)
    }

    /// Same as [`Sam2::from_safetensors`] but compiles the image
    /// encoder for the given backend. The cross-backend feature flags
    /// match SAM v1's [`rlx_sam::Sam::from_safetensors_on`].
    pub fn from_safetensors_on(
        weights_path: &str,
        cfg: Sam2Config,
        device: Device,
    ) -> Result<Self> {
        rlx_core::validate_sam_device("sam2", device)?;
        let mut wm =
            rlx_core::load_weight_map(Path::new(weights_path), rlx_core::SAM2_GGUF_ARCHES)?;
        let compile_profile = sam2_profile_near_weights(Path::new(weights_path));

        // 1) Hiera image encoder graph (drains its weight keys + the
        //    preprocess + FPN-neck weights).
        let (graph, params, pre, fpn) = build_sam2_image_encoder_graph(&cfg.hiera, &mut wm)?;

        let hiera_shapes = HieraOutputShapes {
            stage_hw: (0..cfg.hiera.stages.len())
                .map(|s| {
                    (
                        cfg.hiera.grid_size_at_stage(s),
                        cfg.hiera.grid_size_at_stage(s),
                    )
                })
                .collect(),
            stage_dims: (0..cfg.hiera.stages.len())
                .map(|s| cfg.hiera.embed_dim_at_stage(s))
                .collect(),
        };

        // 2) Prompt encoder.
        let prompt_enc = extract_prompt_encoder_weights(
            &mut wm,
            cfg.decoder.transformer_dim,
            SAM2_MASK_IN_CHANS,
        )?;

        // 3) Mask decoder.
        let mask_dec = extract_mask_decoder_weights(&mut wm, &cfg.decoder)?;

        // 4) Memory encoder.
        let mut mem_enc = extract_memory_encoder_weights(&mut wm, &cfg.memory_encoder)?;
        super::memory_encoder::compile_memory_encoder_ir(
            &mut mem_enc,
            SAM2_IMG_SIZE,
            SAM2_IMG_SIZE,
            SAM2_PROMPT_GRID,
            SAM2_PROMPT_GRID,
            device,
            &compile_profile,
        )?;

        // 5) Memory attention.
        let mem_attn = extract_memory_attention_weights(&mut wm, &cfg.memory)?;
        let grid = cfg.hiera.grid_size_at_stage(cfg.hiera.stages.len() - 1);
        let mask_stack =
            Sam2PromptMaskCompiled::compile_with_profile(&prompt_enc, device, &compile_profile)?;
        let upscale = Sam2MaskUpscaleCompiled::compile_with_profile(
            &mask_dec,
            grid,
            device,
            &compile_profile,
        )?;
        let hyper_matmul = MaskHyperMatmulCompiled::compile_with_profile(
            mask_dec.num_mask_tokens,
            cfg.decoder.transformer_dim / 8,
            grid,
            device,
            &compile_profile,
        )?;
        let hyper_mlps_ir = super::mlp_ir::compile_hyper_mlps_with_profile(
            &mask_dec.hyper_mlps,
            device,
            &compile_profile,
        )?;
        let iou_head_ir = super::mlp_ir::compile_hyper_mlp_with_profile(
            &mask_dec.iou_head,
            device,
            &compile_profile,
        )?;
        let obj_score_head_ir = super::mlp_ir::compile_optional_hyper_mlp_with_profile(
            &mask_dec.obj_score_head,
            1,
            device,
            &compile_profile,
        )?;
        let obj_ptr_rows = super::mlp_ir::obj_ptr_proj_rows(
            mask_dec.num_mask_tokens,
            mask_dec.use_multimask_token_for_obj_ptr,
        );
        let obj_ptr_proj_ir = super::mlp_ir::compile_optional_hyper_mlp_with_profile(
            &mask_dec.obj_ptr_proj,
            obj_ptr_rows,
            device,
            &compile_profile,
        )?;
        let s_tok = if mask_dec.obj_score_token.is_some() {
            1
        } else {
            0
        };
        let base_q_n = s_tok + 1 + mask_dec.num_mask_tokens;
        let grid = cfg.hiera.grid_size_at_stage(cfg.hiera.stages.len() - 1);
        let tw_ir = super::transformer_ir::compile_two_way_transformer_with_profile(
            &mask_dec.transformer,
            base_q_n,
            grid,
            device,
            &compile_profile,
        )?;
        let fpn_ir = compile_fpn_neck_ir(
            &fpn,
            &hiera_shapes.stage_hw,
            &hiera_shapes.stage_dims,
            device,
            &compile_profile,
        )?;

        let opts = rlx_core::flow_bridge::compile_options_for_profile(&compile_profile, device);
        let mut encoder = Session::new(device).compile_with(graph, &opts);
        for (name, data) in &params {
            encoder.set_param(name, data);
        }

        // Preflight: at least the most-important keys should be drained.
        // We don't assert the full map is empty because the published
        // sam2 checkpoints include training-only buffers we choose to
        // ignore (e.g. `maskmem_tpos_enc`, optimizer state remnants).
        Ok(Self {
            cfg,
            encoder,
            pre,
            fpn,
            fpn_ir,
            prompt_enc,
            mask_dec,
            mask_stack,
            upscale,
            hyper_matmul,
            hyper_mlps_ir,
            iou_head_ir,
            obj_score_head_ir,
            obj_ptr_proj_ir,
            tw_ir,
            mem_enc,
            mem_attn,
            mem_attn_ir: None,
            mem_attn_device: device,
            hiera_shapes,
            compile_profile,
        })
    }

    /// Tier-1 compile profile (`sam.rlx.toml` next to weights when present).
    pub fn compile_profile(&self) -> &CompileProfile {
        &self.compile_profile
    }

    pub fn config(&self) -> &Sam2Config {
        &self.cfg
    }

    fn ensure_mem_attn_ir(&mut self) -> Result<()> {
        if self.mem_attn_ir.is_some() {
            return Ok(());
        }
        let [rope_x, rope_y] = self.cfg.memory.rope_feat_size;
        let n_img_mem = rope_x * rope_y;
        let max_n_mem = max_memory_slots(n_img_mem, self.cfg.memory.max_obj_ptrs_in_encoder);
        self.mem_attn_ir = Some(if self.cfg.memory.mem_attn_in_graph_rope {
            MemoryAttentionCompiled::compile_in_graph_rope_with_profile(
                &self.mem_attn,
                n_img_mem,
                max_n_mem,
                self.cfg.memory.max_obj_ptrs_in_encoder,
                self.mem_attn_device,
                &self.compile_profile,
            )?
        } else {
            MemoryAttentionCompiled::compile_with_profile(
                &self.mem_attn,
                n_img_mem,
                max_n_mem,
                self.cfg.memory.max_obj_ptrs_in_encoder,
                self.mem_attn_device,
                &self.compile_profile,
            )?
        });
        Ok(())
    }

    /// Run the encoder + FPN host-side neck and return per-level
    /// features ordered fine → coarse (stride 4, 8, 16, 32).
    fn encode(&mut self, image_u8: &[u8], h_in: usize, w_in: usize) -> Result<Vec<FpnLevel>> {
        let image_nchw = preprocess_image(image_u8, h_in, w_in);
        let hidden = assemble_patch_tokens(&self.pre, &image_nchw)?;
        let outputs = self.encoder.run(&[("hidden", hidden.as_slice())]);
        ensure!(
            outputs.len() == self.hiera_shapes.stage_dims.len(),
            "encoder produced {} outputs (expected {})",
            outputs.len(),
            self.hiera_shapes.stage_dims.len()
        );
        apply_fpn_neck(
            &self.fpn,
            &mut self.fpn_ir,
            &outputs,
            &self.hiera_shapes.stage_hw,
            &self.hiera_shapes.stage_dims,
        )
    }

    /// Image-segmentation API.
    ///
    /// `image_u8`: row-major RGB `h_in × w_in × 3` u8.
    /// `points`: optional `(coords [N,2], labels [N])` — coords in
    ///     input-image pixels (0..max(h_in, w_in)), labels per
    ///     [`prompt_encoder_forward`].
    /// `boxes`: optional `[M, 4]` boxes (x0, y0, x1, y1) in input
    ///     pixels.
    /// `mask_input`: optional `[1, 256, 256]` low-res mask logits.
    /// `multimask_output`: true → 3 masks; false → 1 (with optional
    ///     dynamic-stability fallback).
    ///
    /// Returns `(mask_logits, iou_pred, num_masks, h_out, w_out)`
    /// where `(h_out, w_out)` = `(4·SAM2_PROMPT_GRID, 4·SAM2_PROMPT_GRID)`
    /// = 256×256 — caller resizes to the original image resolution.
    pub fn predict_image(
        &mut self,
        image_u8: &[u8],
        h_in: usize,
        w_in: usize,
        points: Option<(&[f32], &[f32])>,
        boxes: Option<&[f32]>,
        mask_input: Option<&[f32]>,
        multimask_output: bool,
    ) -> Result<Sam2ImagePrediction> {
        let levels = self.encode(image_u8, h_in, w_in)?;
        // FPN levels are fine→coarse: stride 4, 8, 16, 32.
        // Image embedding for the mask decoder is the stride-16 level
        // (index 2). High-res features are stride-4 + stride-8.
        let prompt = self.run_prompt(points, boxes, mask_input)?;
        let dec = self.run_decoder(&levels, &prompt, multimask_output)?;

        Ok(Sam2ImagePrediction {
            masks: dec.masks,
            iou_pred: dec.iou_pred,
            num_masks: dec.num_masks,
            h_out: dec.h_out,
            w_out: dec.w_out,
            object_score_logits: dec.object_score_logits,
            object_pointer: dec.object_pointer,
        })
    }

    fn run_prompt(
        &mut self,
        points: Option<(&[f32], &[f32])>,
        boxes: Option<&[f32]>,
        mask_input: Option<&[f32]>,
    ) -> Result<Sam2PromptEncoderOutput> {
        prompt_encoder_forward(
            &self.prompt_enc,
            &mut self.mask_stack,
            points,
            boxes,
            mask_input,
        )
    }

    fn run_decoder(
        &mut self,
        levels: &[FpnLevel],
        prompt: &Sam2PromptEncoderOutput,
        multimask_output: bool,
    ) -> Result<Sam2MaskDecoderOutput> {
        let lvl_stride16 = &levels[2]; // stride 16 → 64×64
        let lvl_stride8 = &levels[1]; // stride 8  → 128×128
        let lvl_stride4 = &levels[0]; // stride 4  → 256×256

        let high_res_features = if self.mask_dec.use_high_res_features {
            Some((
                lvl_stride4.features.as_slice(),
                lvl_stride8.features.as_slice(),
            ))
        } else {
            None
        };

        ensure!(
            lvl_stride16.h == SAM2_PROMPT_GRID && lvl_stride16.w == SAM2_PROMPT_GRID,
            "stride-16 FPN level must be {}×{} (got {}×{})",
            SAM2_PROMPT_GRID,
            SAM2_PROMPT_GRID,
            lvl_stride16.h,
            lvl_stride16.w
        );

        mask_decoder_forward(
            &self.mask_dec,
            &mut self.upscale,
            Some(&mut self.hyper_matmul),
            Some(&mut self.hyper_mlps_ir),
            Some(&mut self.iou_head_ir),
            self.obj_score_head_ir.as_mut(),
            self.obj_ptr_proj_ir.as_mut(),
            Some(&mut self.tw_ir),
            &lvl_stride16.features,
            &lvl_stride16.pos,
            &prompt.sparse_embeddings,
            prompt.num_sparse_tokens,
            &prompt.dense_embeddings,
            high_res_features,
            multimask_output,
            SAM2_PROMPT_GRID,
        )
    }

    /// Per-frame video API. Wraps [`Sam2::predict_image`] with the
    /// memory-attention path (cross-attend the current frame's stride-32
    /// features to the bank) and the memory-encoder path (encode the
    /// chosen mask + features into the bank).
    ///
    /// Mirrors `SAM2VideoPredictor.add_new_points_or_box` +
    /// `propagate_in_video` semantics: when `state` is empty, this acts
    /// as image-predict; otherwise it conditions on stored frames.
    pub fn predict_video_frame(
        &mut self,
        state: &mut Sam2VideoState,
        image_u8: &[u8],
        h_in: usize,
        w_in: usize,
        points: Option<(&[f32], &[f32])>,
        boxes: Option<&[f32]>,
        mask_input: Option<&[f32]>,
        multimask_output: bool,
    ) -> Result<Sam2ImagePrediction> {
        let levels = self.encode(image_u8, h_in, w_in)?;

        // Stride-32 level (index 3) is the queries source for memory
        // attention — matches the reference's `vision_features` at
        // 32×32 resolution.
        let stride32 = &levels[3];
        let mut conditioned_stride32: Vec<f32> = stride32.features.clone();
        if !state.memory.is_empty() {
            let curr = nchw_to_seq_c(
                &stride32.features,
                self.cfg.memory.d_model,
                stride32.h,
                stride32.w,
            );
            let curr_pos = nchw_to_seq_c(
                &stride32.pos,
                self.cfg.memory.d_model,
                stride32.h,
                stride32.w,
            );

            let (memory_flat, memory_pos_flat, n_mem) =
                state.assembled_memory(self.cfg.memory.kv_in_dim, self.cfg.memory.mem_dim);
            let n_img = stride32.h * stride32.w;
            let num_ptr = state.num_obj_ptr_tokens(self.cfg.memory.mem_dim);
            self.ensure_mem_attn_ir()?;
            let ir = self.mem_attn_ir.as_mut().expect("mem_attn_ir");
            let attn_out = if n_img == ir.n_img && n_mem <= ir.max_n_mem {
                ir.run(
                    &curr,
                    &curr_pos,
                    &memory_flat,
                    &memory_pos_flat,
                    n_mem,
                    num_ptr,
                )?
            } else {
                memory_attention_forward(
                    &self.mem_attn,
                    &curr,
                    &curr_pos,
                    &memory_flat,
                    &memory_pos_flat,
                    n_img,
                    n_mem,
                    self.cfg.memory.kv_in_dim,
                    num_ptr,
                )?
            };
            // Reshape back to NCHW.
            conditioned_stride32 =
                seq_c_to_nchw(&attn_out, self.cfg.memory.d_model, stride32.h, stride32.w);
        }

        // Splice the conditioned features back into level[3] for the
        // decoder. Decoder reads stride-16 (level[2]) for image_emb +
        // dense, so we only condition the memory-attention output for
        // *propagation* — the stride-16 path is unmodified per the
        // reference.
        let mut levels = levels;
        levels[3].features = conditioned_stride32;

        let prompt = self.run_prompt(points, boxes, mask_input)?;
        let dec = self.run_decoder(&levels, &prompt, multimask_output)?;

        // Encode the chosen mask + stride-16 features into memory and
        // push them onto the state's bank.
        let stride16 = &levels[2];
        let mem = run_memory_encoder(&mut self.mem_enc, &stride16.features, &dec)?;
        state.push_frame_memory(
            mem,
            dec.object_pointer.clone(),
            self.cfg.memory.max_obj_ptrs_in_encoder,
        );

        Ok(Sam2ImagePrediction {
            masks: dec.masks,
            iou_pred: dec.iou_pred,
            num_masks: dec.num_masks,
            h_out: dec.h_out,
            w_out: dec.w_out,
            object_score_logits: dec.object_score_logits,
            object_pointer: dec.object_pointer,
        })
    }
}

/// One frame's worth of mask-decoder output, as returned by both
/// [`Sam2::predict_image`] and [`Sam2::predict_video_frame`].
pub struct Sam2ImagePrediction {
    pub masks: Vec<f32>,
    pub iou_pred: Vec<f32>,
    pub num_masks: usize,
    pub h_out: usize,
    pub w_out: usize,
    pub object_score_logits: Vec<f32>,
    pub object_pointer: Option<Vec<f32>>,
}

/// Per-track state for [`Sam2::predict_video_frame`]. Stores up to
/// `max_obj_ptrs_in_encoder` past memory tokens + the rolling
/// object-pointer queue.
pub struct Sam2VideoState {
    /// Each entry: `(features [out_dim, h, w] flat, pos [..., h, w] flat, h, w)`.
    pub memory: Vec<Sam2MemoryEncoderOutput>,
    pub obj_ptr_queue: Vec<Vec<f32>>,
}

impl Sam2VideoState {
    pub fn new() -> Self {
        Self {
            memory: Vec::new(),
            obj_ptr_queue: Vec::new(),
        }
    }

    /// Total number of memory tokens (spatial + obj-ptr) in the
    /// concatenated memory bank. `mem_dim` is the obj-pointer
    /// channel dim (typically 64).
    pub fn num_obj_ptr_tokens(&self, _mem_dim: usize) -> usize {
        // Each stored obj-ptr is a single token (the reference splits a
        // higher-dim ptr into 4 sub-tokens via `obj_ptr_proj`, but at
        // the level we expose here we treat each frame's pointer as a
        // single token). When training a sub-token split, the user can
        // extend this fn.
        self.obj_ptr_queue.len()
    }

    /// Concatenate the per-frame memories into a single
    /// `(memory [N_mem, kv_in_dim], memory_pos [N_mem, kv_in_dim])`
    /// pair for the memory-attention call. Spatial tokens go first,
    /// object-pointer tokens at the tail (so `num_k_exclude_rope`
    /// works correctly).
    pub fn assembled_memory(
        &self,
        kv_in_dim: usize,
        _mem_dim: usize,
    ) -> (Vec<f32>, Vec<f32>, usize) {
        let mut features = Vec::new();
        let mut positions = Vec::new();
        let mut total_tokens = 0usize;

        for m in &self.memory {
            let tokens = m.h * m.w;
            // Flatten [out_dim, h, w] → [tokens, out_dim] (matches kv_in_dim).
            let mut feat_seq = vec![0f32; tokens * kv_in_dim];
            let mut pos_seq = vec![0f32; tokens * kv_in_dim];
            let pe_chans = m.pos.len() / (m.h * m.w);
            for t in 0..tokens {
                for c in 0..kv_in_dim {
                    feat_seq[t * kv_in_dim + c] = m.features[c * tokens + t];
                }
                // PE may have more channels than kv_in_dim (e.g. 128 vs 64).
                // We only copy the first `kv_in_dim` to match memory's channel layout.
                for c in 0..kv_in_dim.min(pe_chans) {
                    pos_seq[t * kv_in_dim + c] = m.pos[c * tokens + t];
                }
            }
            features.extend_from_slice(&feat_seq);
            positions.extend_from_slice(&pos_seq);
            total_tokens += tokens;
        }

        // Append object-pointer tokens (no PE — they go in the
        // `num_k_exclude_rope` band).
        for ptr in &self.obj_ptr_queue {
            ensure_or_zero(&mut features, &mut positions, ptr, kv_in_dim);
            total_tokens += 1;
        }

        (features, positions, total_tokens)
    }

    fn push_frame_memory(
        &mut self,
        mem: Sam2MemoryEncoderOutput,
        obj_ptr: Option<Vec<f32>>,
        max_ptrs: usize,
    ) {
        self.memory.push(mem);
        if let Some(p) = obj_ptr {
            self.obj_ptr_queue.push(p);
            while self.obj_ptr_queue.len() > max_ptrs {
                self.obj_ptr_queue.remove(0);
            }
        }
    }
}

impl Default for Sam2VideoState {
    fn default() -> Self {
        Self::new()
    }
}

fn ensure_or_zero(
    features: &mut Vec<f32>,
    positions: &mut Vec<f32>,
    ptr: &[f32],
    kv_in_dim: usize,
) {
    if ptr.len() == kv_in_dim {
        features.extend_from_slice(ptr);
    } else {
        // Reference's `obj_ptr_proj` produces `transformer_dim`-sized
        // pointers (256), which the loader reshape-projects into
        // `mem_dim` (64) chunks via `obj_ptr_proj.layers.{i}.weight`.
        // We approximate by taking the first `kv_in_dim` channels — a
        // correct full split requires the loader's reshape; the user
        // can pre-project before calling.
        let take = ptr.len().min(kv_in_dim);
        features.extend_from_slice(&ptr[..take]);
        for _ in take..kv_in_dim {
            features.push(0.0);
        }
    }
    for _ in 0..kv_in_dim {
        positions.push(0.0);
    }
}

fn run_memory_encoder(
    mem_enc: &mut Sam2MemoryEncoderWeights,
    pix_feat: &[f32],
    dec: &Sam2MaskDecoderOutput,
) -> Result<Sam2MemoryEncoderOutput> {
    // We always pick the first (top-IoU) mask to encode. Reference
    // `SAM2Base._encode_new_memory` does the same when caller doesn't
    // override.
    // dec.masks shape: [num_masks, h_out, w_out]. Take mask 0.
    let m_chunk = dec.h_out * dec.w_out;
    ensure!(
        dec.masks.len() >= m_chunk,
        "decoder produced empty mask buffer"
    );
    let mask0 = &dec.masks[..m_chunk];

    // Reference upsamples the 256×256 mask to 1024×1024 before
    // memory-encoding (`F.interpolate(masks, size=(1024, 1024),
    // mode="bilinear")`). We do the same with a cheap bilinear.
    let mut up_mask = vec![0f32; SAM2_IMG_SIZE * SAM2_IMG_SIZE];
    bilinear_upsample_1ch(
        mask0,
        dec.h_out,
        dec.w_out,
        &mut up_mask,
        SAM2_IMG_SIZE,
        SAM2_IMG_SIZE,
    );

    memory_encoder_forward(
        mem_enc,
        pix_feat,
        &up_mask,
        SAM2_PROMPT_GRID,
        SAM2_PROMPT_GRID,
        /*skip_mask_sigmoid=*/ false,
    )
}

fn bilinear_upsample_1ch(src: &[f32], sh: usize, sw: usize, dst: &mut [f32], dh: usize, dw: usize) {
    let sx = (sw as f32) / (dw as f32);
    let sy = (sh as f32) / (dh as f32);
    for y in 0..dh {
        let yf = (y as f32 + 0.5) * sy - 0.5;
        let y0 = yf.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let dy = (yf - yf.floor()).clamp(0.0, 1.0);
        for x in 0..dw {
            let xf = (x as f32 + 0.5) * sx - 0.5;
            let x0 = xf.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let dx = (xf - xf.floor()).clamp(0.0, 1.0);
            let p00 = src[y0 * sw + x0];
            let p01 = src[y0 * sw + x1];
            let p10 = src[y1 * sw + x0];
            let p11 = src[y1 * sw + x1];
            let top = p00 * (1.0 - dx) + p01 * dx;
            let bot = p10 * (1.0 - dx) + p11 * dx;
            dst[y * dw + x] = top * (1.0 - dy) + bot * dy;
        }
    }
}

fn nchw_to_seq_c(src: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0f32; h * w * c];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                out[(y * w + x) * c + ch] = src[ch * h * w + y * w + x];
            }
        }
    }
    out
}

fn seq_c_to_nchw(src: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0f32; c * h * w];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                out[ch * h * w + y * w + x] = src[(y * w + x) * c + ch];
            }
        }
    }
    out
}

#[allow(dead_code)]
fn _silence_decoder_cfg(_d: &Sam2DecoderConfig) {}
