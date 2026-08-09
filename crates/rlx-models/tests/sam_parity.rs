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

//! SAM v1 image-encoder parity test against candle.
//!
//! Phase-1 scope: validate that the rlx-models SAM ViT-B image encoder
//! produces the same `[1, 256, 64, 64]` image embeddings as candle's
//! `image_encoder::ImageEncoderViT::forward()` on identical input,
//! using the same `sam_vit_b_01ec64.safetensors` checkpoint.
//!
//! The prompt encoder + mask decoder land in a follow-up commit; once
//! those are wired, this file will grow an end-to-end mask-output
//! parity test too.
//!
//! ## Running
//!
//! ```bash
//! huggingface-cli download lmz/candle-sam sam_vit_b_01ec64.safetensors \
//!   --local-dir /tmp/rlx_sam
//!
//! RLX_SAM_WEIGHTS=/tmp/rlx_sam/sam_vit_b_01ec64.safetensors \
//!   cargo test -p rlx-models --features parity-candle --release sam_encoder_parity \
//!   -- --nocapture
//! ```

#![cfg(feature = "parity-candle")]

mod compile_support;

use anyhow::Result;
use candle_core::{DType as CDType, Device as CDevice, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::segment_anything::image_encoder::ImageEncoderViT;
use candle_transformers::models::segment_anything::mask_decoder::MaskDecoder;
use candle_transformers::models::segment_anything::prompt_encoder::PromptEncoder;
use rlx_models::WeightMap;
use rlx_models::sam::{
    Device, SAM_EMBED_HW, SAM_IMG_SIZE, SAM_MASK_IN_CHANS, SAM_PROMPT_EMBED_DIM, Sam, SamConfig,
    SamEncoderConfig, assemble_patch_tokens, build_sam_encoder_graph,
};

const PARITY_TOL: f32 = 5e-3;

fn weights_path() -> Option<String> {
    rlx_ir::env::var("RLX_SAM_WEIGHTS")
}

/// Deterministic 1024×1024 NCHW image — same sine pattern recipe as
/// the DINOv2 parity test, scaled to SAM's pixel-space input range
/// (post-normalisation it's roughly [-2, 2]).
// 6.28 / 3.14 are deliberate 2-dp values, not TAU/PI: they define the
// synthetic parity input, and changing them would change every dumped
// reference blob compared against here.
#[allow(clippy::approx_constant)]
fn synthesize_image() -> Vec<f32> {
    let n = 3 * SAM_IMG_SIZE * SAM_IMG_SIZE;
    let mut v = vec![0f32; n];
    for c in 0..3 {
        let phase = (c as f32) * 0.7;
        for y in 0..SAM_IMG_SIZE {
            for x in 0..SAM_IMG_SIZE {
                let fx = x as f32 / SAM_IMG_SIZE as f32;
                let fy = y as f32 / SAM_IMG_SIZE as f32;
                v[c * SAM_IMG_SIZE * SAM_IMG_SIZE + y * SAM_IMG_SIZE + x] =
                    (6.28 * fx + phase).sin() * (3.14 * fy + phase).cos();
            }
        }
    }
    v
}

fn debug_depth() -> usize {
    rlx_ir::env::var("RLX_SAM_DEBUG_DEPTH")
        .and_then(|s| s.parse().ok())
        .unwrap_or(12)
}

fn run_candle(weights: &str, image_nchw: &[f32]) -> Result<Vec<f32>> {
    let device = CDevice::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], CDType::F32, &device)? };
    let use_rel_pos = rlx_ir::env::is_unset("RLX_SAM_DEBUG_NO_RELPOS");
    let depth = debug_depth();
    // RLX_SAM_DEBUG_FORCE_GLOBAL=1 makes every block use global
    // attention. Only meaningful in combination with depth ≤ 4 (since
    // the safetensors file's rel_pos tensors are only sized for
    // window_size=14 in the windowed blocks).
    let global_attn: Vec<usize> = if rlx_ir::env::flag("RLX_SAM_DEBUG_FORCE_GLOBAL") {
        (0..depth).collect()
    } else {
        [2, 5, 8, 11].into_iter().filter(|&i| i < depth).collect()
    };
    let encoder = ImageEncoderViT::new(
        /* img_size           */ SAM_IMG_SIZE,
        /* patch_size         */ 16,
        /* in_chans           */ 3,
        /* embed_dim          */ 768,
        /* depth              */ depth,
        /* num_heads          */ 12,
        /* out_chans          */ SAM_PROMPT_EMBED_DIM,
        /* qkv_bias           */ true,
        /* use_rel_pos        */ use_rel_pos,
        /* use_abs_pos        */ true,
        /* window_size        */ 14,
        /* global_attn_idx    */ &global_attn,
        vb.pp("image_encoder"),
    )?;
    let img = Tensor::from_slice(image_nchw, (1, 3, SAM_IMG_SIZE, SAM_IMG_SIZE), &device)?;
    let out = encoder.forward(&img)?;
    // out: [1, 256, 64, 64] NCHW
    let flat = out.flatten_all()?.to_vec1::<f32>()?;
    Ok(flat)
}

fn run_rlx(weights: &str, image_nchw: &[f32]) -> Result<Vec<f32>> {
    let mut cfg = SamEncoderConfig::vit_b();
    if rlx_ir::env::flag("RLX_SAM_DEBUG_NO_RELPOS") {
        cfg.use_rel_pos = false;
    }
    cfg.depth = debug_depth();
    if rlx_ir::env::flag("RLX_SAM_DEBUG_FORCE_GLOBAL") {
        cfg.global_attn_indexes = (0..cfg.depth).collect();
    } else {
        cfg.global_attn_indexes.retain(|&i| i < cfg.depth);
    }
    let mut wm = WeightMap::from_file(weights)?;
    let (graph, params, pre) = build_sam_encoder_graph(&cfg, &mut wm)?;

    let mut compiled =
        rlx_models::flow_util::compile_graph_sam_with_params(pick_device(), graph, params)?;

    // Host-side patch tokens (Conv2d 16×16 stride 16) + abs pos_embed.
    let hidden = assemble_patch_tokens(&pre, image_nchw)?;
    let outputs = compiled.run(&[("hidden", hidden.as_slice())]);
    let body_out = outputs
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("rlx encoder produced no output"))?;

    // The neck (conv1x1 → LN2d → conv3x3 → LN2d) is part of the compiled
    // graph now, so the single output already is `[256, 64, 64]` NCHW.
    Ok(body_out)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    let mut max = 0f32;
    let mut idx = 0;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > max {
            max = d;
            idx = i;
        }
    }
    (max, idx)
}

/// Pick the runtime device for `Sam::from_safetensors_on`. Respects
/// `RLX_SAM_DEVICE=metal|mlx|gpu|cuda|cpu` (default `cpu`); falls
/// back to CPU when the requested backend isn't compiled in.
fn pick_device() -> Device {
    let want = rlx_ir::env::var("RLX_SAM_DEVICE").unwrap_or_default();
    match want.to_ascii_lowercase().as_str() {
        "metal" => {
            #[cfg(feature = "metal")]
            return Device::Metal;
            #[cfg(not(feature = "metal"))]
            {
                eprintln!("RLX_SAM_DEVICE=metal but rlx-models not built with --features metal");
                Device::Cpu
            }
        }
        "mlx" => {
            #[cfg(feature = "mlx")]
            return Device::Mlx;
            #[cfg(not(feature = "mlx"))]
            {
                eprintln!("RLX_SAM_DEVICE=mlx but rlx-models not built with --features mlx");
                Device::Cpu
            }
        }
        "gpu" => {
            #[cfg(feature = "gpu")]
            return Device::Gpu;
            #[cfg(not(feature = "gpu"))]
            {
                eprintln!("RLX_SAM_DEVICE=gpu but rlx-models not built with --features gpu");
                Device::Cpu
            }
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            return Device::Cuda;
            #[cfg(not(feature = "cuda"))]
            {
                eprintln!("RLX_SAM_DEVICE=cuda but rlx-models not built with --features cuda");
                Device::Cpu
            }
        }
        _ => Device::Cpu,
    }
}

/// End-to-end parity: image → mask logits + IoU pred, with a single
/// foreground point prompt at (512, 512). Compares rlx-models `Sam`
/// against candle's `ImageEncoderViT + PromptEncoder + MaskDecoder`
/// wired together by hand (since candle doesn't expose a top-level
/// `Sam::forward` here — its `sam.rs` does).
///
/// Set `RLX_SAM_DEVICE=metal` (and build `--features metal`) to run
/// the rlx encoder on Apple GPU.
#[test]
fn sam_end_to_end_parity_vs_candle() -> Result<()> {
    let Some(weights) = weights_path() else {
        eprintln!("skipping — set RLX_SAM_WEIGHTS=/path/to/sam_vit_b_01ec64.safetensors");
        return Ok(());
    };

    let image = synthesize_image();

    // ── candle reference: build encoder + prompt encoder + mask decoder ──
    let device = CDevice::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], CDType::F32, &device)? };
    let encoder = ImageEncoderViT::new(
        SAM_IMG_SIZE,
        16,
        3,
        768,
        12,
        12,
        SAM_PROMPT_EMBED_DIM,
        true,
        true,
        true,
        14,
        &[2, 5, 8, 11],
        vb.pp("image_encoder"),
    )?;
    let prompt_encoder = PromptEncoder::new(
        SAM_PROMPT_EMBED_DIM,
        (SAM_EMBED_HW, SAM_EMBED_HW),
        (SAM_IMG_SIZE, SAM_IMG_SIZE),
        SAM_MASK_IN_CHANS,
        vb.pp("prompt_encoder"),
    )?;
    let mask_decoder = MaskDecoder::new(
        SAM_PROMPT_EMBED_DIM,
        /*num_multimask_outputs*/ 3,
        /*iou_head_depth*/ 3,
        /*iou_head_hidden_dim*/ SAM_PROMPT_EMBED_DIM,
        vb.pp("mask_decoder"),
    )?;

    let img_t = Tensor::from_slice(&image, (1, 3, SAM_IMG_SIZE, SAM_IMG_SIZE), &device)?;
    let candle_image_emb = encoder.forward(&img_t)?; // [1, 256, 64, 64]

    // Point prompt at (512, 512) with label 1 (foreground).
    let point_coords_c = Tensor::from_slice(&[512.0f32, 512.0], (1, 1, 2), &device)?;
    let point_labels_c = Tensor::from_slice(&[1.0f32], (1, 1), &device)?;
    let (sparse_c, dense_c) =
        prompt_encoder.forward(Some((&point_coords_c, &point_labels_c)), None, None)?;
    let image_pe_c = prompt_encoder.get_dense_pe()?;
    let (candle_masks, candle_iou) = mask_decoder.forward(
        &candle_image_emb,
        &image_pe_c,
        &sparse_c,
        &dense_c,
        /*multimask=*/ true,
    )?;
    let candle_masks_flat = candle_masks.flatten_all()?.to_vec1::<f32>()?;
    let candle_iou_flat = candle_iou.flatten_all()?.to_vec1::<f32>()?;

    // ── rlx-models Sam ──
    let device = pick_device();
    eprintln!("[sam e2e] rlx device = {device:?}");
    let mut sam = Sam::from_safetensors_on(&weights, SamConfig::vit_b(), device)?;
    let rlx_image_emb = sam.encode_image(&image);

    let pt_coords = vec![512.0f32, 512.0];
    let pt_labels = vec![1.0f32];
    let pred = sam.predict_masks(
        &rlx_image_emb,
        Some((&pt_coords, &pt_labels)),
        None,
        None,
        /*multimask=*/ true,
    )?;

    // Compare image embeddings
    let candle_emb_flat = candle_image_emb.flatten_all()?.to_vec1::<f32>()?;
    let (emb_diff, emb_idx) = max_abs_diff(&rlx_image_emb, &candle_emb_flat);
    eprintln!("[sam e2e] image_emb max |Δ| = {emb_diff:.4e} at idx {emb_idx}");

    // Compare masks
    let (mask_diff, mask_idx) = max_abs_diff(&pred.mask_logits, &candle_masks_flat);
    eprintln!(
        "[sam e2e] masks: {} values; max |Δ| = {mask_diff:.4e} at idx {mask_idx} \
         (rlx={}, candle={})",
        pred.mask_logits.len(),
        pred.mask_logits[mask_idx],
        candle_masks_flat[mask_idx],
    );

    // Compare IoU
    let (iou_diff, iou_idx) = max_abs_diff(&pred.iou_pred, &candle_iou_flat);
    eprintln!(
        "[sam e2e] iou_pred = rlx {:?} vs candle {:?}; max |Δ| = {iou_diff:.4e}",
        pred.iou_pred, candle_iou_flat,
    );
    let _ = iou_idx;

    // Threshold the mask logits at 0 (SAM convention) and verify the
    // boolean masks agree on at least 99.9% of pixels.
    let mut agree = 0usize;
    for i in 0..pred.mask_logits.len() {
        let r = pred.mask_logits[i] > 0.0;
        let c = candle_masks_flat[i] > 0.0;
        if r == c {
            agree += 1;
        }
    }
    let agreement = agree as f64 / pred.mask_logits.len() as f64;
    eprintln!(
        "[sam e2e] binary mask agreement: {agree}/{} = {:.6}%",
        pred.mask_logits.len(),
        100.0 * agreement
    );

    // Tolerances depend on the backend:
    //   - CPU: same BLAS (rlx NEON ↔ candle NEON), so f32 noise floor.
    //   - Metal: MPS sgemm uses a different reduction order from CPU
    //     sgemm; over 12 layers + decoder hypernetwork that compounds
    //     to ~1e-1 in mask logits even though every individual op is
    //     algorithmically correct. The functional output (thresholded
    //     binary mask) still matches > 99.9% of pixels.
    let cpu = matches!(device, Device::Cpu);
    let mask_tol = if cpu { 1e-2 } else { 5e-1 };
    let iou_tol = 1e-3;
    let agree_min = 0.999;
    assert!(
        mask_diff <= mask_tol,
        "rlx vs candle mask logits diverge: max |Δ| = {mask_diff:.4e} > {mask_tol:.0e} (device {device:?})"
    );
    assert!(
        iou_diff <= iou_tol,
        "rlx vs candle IoU pred diverge: max |Δ| = {iou_diff:.4e} > {iou_tol:.0e} (device {device:?})"
    );
    assert!(
        agreement >= agree_min,
        "binary mask agreement only {:.4}% (< {:.1}%) on {device:?}",
        100.0 * agreement,
        100.0 * agree_min
    );
    Ok(())
}

#[test]
fn sam_encoder_parity_vs_candle() -> Result<()> {
    let Some(weights) = weights_path() else {
        eprintln!("skipping — set RLX_SAM_WEIGHTS=/path/to/sam_vit_b_01ec64.safetensors");
        return Ok(());
    };

    let image = synthesize_image();
    let candle_emb = run_candle(&weights, &image)?;
    let rlx_emb = run_rlx(&weights, &image)?;

    let (diff, idx) = max_abs_diff(&rlx_emb, &candle_emb);
    eprintln!(
        "[sam encoder parity] {} f32 values; max |Δ| = {diff:.4e} at idx {idx} \
         (rlx={}, candle={})",
        rlx_emb.len(),
        rlx_emb[idx],
        candle_emb[idx],
    );

    // Backend-aware tolerance — MPS sgemm uses a different reduction
    // order than NEON sgemm, so Metal compounds ~1e-2 of BLAS noise
    // across 12 layers. CPU and MLX both land at the f32 noise floor.
    let device = pick_device();
    let tol = match device {
        Device::Metal => 5e-2,
        _ => PARITY_TOL,
    };
    assert!(
        diff <= tol,
        "rlx vs candle SAM encoder diverge on {device:?}: max |Δ| = {diff:.4e} > {tol:.4e}"
    );
    Ok(())
}
