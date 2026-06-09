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

//! CPU preprocessing for Gemma 4 **unified** (12B) image + audio inputs.
//!
//! Matches the HuggingFace `Gemma4UnifiedImageProcessor` pipeline:
//! aspect-ratio resize → `[0,1]` rescale → 16px teacher patchify →
//! 3×3 patch merge → pad to `max_soft_tokens`.

use anyhow::{Result, bail};

type PatchGrid = Vec<(i32, i32)>;
type PatchGridResult = (Vec<f32>, PatchGrid);

/// Supported `max_soft_tokens` values (HF Gemma 4 unified).
pub const SUPPORTED_MAX_SOFT_TOKENS: [usize; 5] = [70, 140, 280, 560, 1120];

/// Max raw audio samples (~30s @ 16 kHz).
pub const MAX_AUDIO_SAMPLES: usize = 480_000;

/// Pad audio frame count up to a multiple (HF batches pad to 128).
pub const AUDIO_FRAME_PAD_MULTIPLE: usize = 128;

/// Compute dynamic soft-token count for an image size (before padding).
pub fn compute_num_soft_tokens_from_size(
    height: usize,
    width: usize,
    patch_size: usize,
    pooling_kernel_size: usize,
    max_soft_tokens: usize,
) -> Result<usize> {
    let max_patches = max_soft_tokens * pooling_kernel_size * pooling_kernel_size;
    let (th, tw) =
        aspect_ratio_preserving_size(height, width, patch_size, max_patches, pooling_kernel_size)?;
    let teacher = (th / patch_size) * (tw / patch_size);
    Ok(teacher / (pooling_kernel_size * pooling_kernel_size))
}

/// Keep only non-padding vision rows from a projected `[max_slots × hidden]` buffer.
pub fn strip_valid_vision_rows(
    projected: &[f32],
    positions: &[(i32, i32)],
    hidden: usize,
) -> Vec<f32> {
    let mut out = Vec::new();
    let slots = projected.len() / hidden.max(1);
    for i in 0..slots {
        let (x, y) = positions.get(i).copied().unwrap_or((-1, -1));
        if x >= 0 && y >= 0 {
            out.extend_from_slice(&projected[i * hidden..(i + 1) * hidden]);
        }
    }
    out
}

/// Truncate + frame-count for unified 12B raw PCM (640 samples/token).
pub fn unified_audio_token_count(
    num_samples: usize,
    samples_per_token: usize,
    max_tokens: usize,
) -> usize {
    let capped = num_samples.min(MAX_AUDIO_SAMPLES);
    capped.div_ceil(samples_per_token).max(1).min(max_tokens)
}

pub fn prepare_unified_audio_samples(
    samples: &[f32],
    samples_per_token: usize,
    max_tokens: usize,
) -> Vec<f32> {
    let capped_len = samples.len().min(MAX_AUDIO_SAMPLES);
    let mut truncated = samples[..capped_len].to_vec();
    let mut num_frames = truncated.len().div_ceil(samples_per_token).max(1);
    num_frames = num_frames.min(max_tokens);
    let padded_frames = num_frames.div_ceil(AUDIO_FRAME_PAD_MULTIPLE) * AUDIO_FRAME_PAD_MULTIPLE;
    truncated.resize(padded_frames * samples_per_token, 0.0);
    truncated
}

#[derive(Debug, Clone)]
pub struct UnifiedImageBatch {
    /// Merged 48×48 RGB patches, row-major `[num_slots, 6912]`.
    pub patches: Vec<f32>,
    /// `(x, y)` grid coords per slot; `(-1, -1)` marks padding.
    pub positions: Vec<(i32, i32)>,
    /// Non-padding patch count (before pad-to-max).
    pub num_valid: usize,
}

/// Compute target `(height, width)` preserving aspect ratio within the
/// teacher-patch budget. Port of HF `get_aspect_ratio_preserving_size`.
pub fn aspect_ratio_preserving_size(
    height: usize,
    width: usize,
    patch_size: usize,
    max_patches: usize,
    pooling_kernel_size: usize,
) -> Result<(usize, usize)> {
    let total_px = height * width;
    let target_px = max_patches * patch_size * patch_size;
    let factor = (target_px as f64 / total_px as f64).sqrt();
    let ideal_height = factor * height as f64;
    let ideal_width = factor * width as f64;
    let side_mult = pooling_kernel_size * patch_size;

    let mut target_height = (ideal_height / side_mult as f64).floor() as usize * side_mult;
    let mut target_width = (ideal_width / side_mult as f64).floor() as usize * side_mult;

    if target_height == 0 && target_width == 0 {
        bail!(
            "resize target is 0×0; image too small for patch_size={patch_size} \
             pooling_kernel_size={pooling_kernel_size}"
        );
    }

    let max_side_length = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;
    if target_height == 0 {
        target_height = side_mult;
        target_width =
            ((width as f64 / height as f64).floor() as usize * side_mult).min(max_side_length);
    }
    if target_width == 0 {
        target_width = side_mult;
        target_height =
            ((height as f64 / width as f64).floor() as usize * side_mult).min(max_side_length);
    }
    Ok((target_height.max(side_mult), target_width.max(side_mult)))
}

/// Teacher-level 16×16 patchify from interleaved RGB u8, values in `[0, 1]`.
pub fn teacher_patches_from_rgb(
    rgb: &[u8],
    width: usize,
    height: usize,
    patch_size: usize,
) -> Result<PatchGridResult> {
    if rgb.len() != width * height * 3 {
        bail!("rgb len {} != {width}×{height}×3", rgb.len());
    }
    let patch_cols = width / patch_size;
    let patch_rows = height / patch_size;
    let num = patch_rows * patch_cols;
    let per = patch_size * patch_size * 3;
    let inv = 1.0 / 255.0;
    let mut patches = vec![0f32; num * per];
    let mut positions = Vec::with_capacity(num);
    for pr in 0..patch_rows {
        for pc in 0..patch_cols {
            let idx = pr * patch_cols + pc;
            positions.push((pc as i32, pr as i32));
            let dst_base = idx * per;
            for py in 0..patch_size {
                for px in 0..patch_size {
                    let src = ((pr * patch_size + py) * width + (pc * patch_size + px)) * 3;
                    let dst = dst_base + (py * patch_size + px) * 3;
                    patches[dst] = rgb[src] as f32 * inv;
                    patches[dst + 1] = rgb[src + 1] as f32 * inv;
                    patches[dst + 2] = rgb[src + 2] as f32 * inv;
                }
            }
        }
    }
    Ok((patches, positions))
}

/// Merge `k×k` teacher patches into 48×48 model patches. Port of HF
/// `patches_merge` (single batch, no torch).
pub fn patches_merge(
    patches: &[f32],
    positions: &[(i32, i32)],
    num_model_patches: usize,
    teacher_patch_dim: usize,
) -> Result<PatchGridResult> {
    let l = patches.len() / teacher_patch_dim;
    if l != num_model_patches {
        let k2 = l / num_model_patches;
        if k2 * num_model_patches != l {
            bail!("cannot merge {l} teacher patches into {num_model_patches} model patches");
        }
    }
    let k = ((l / num_model_patches) as f64).sqrt() as usize;
    if k * k * num_model_patches != l {
        bail!("patch count {l} is not num_model×k²");
    }
    let patch_size = (teacher_patch_dim / 3).isqrt();
    let model_dim = (k * patch_size) * (k * patch_size) * 3;

    // Build target ordering (argsort of kernel-group indices).
    let max_x = positions.iter().map(|(x, _)| *x).max().unwrap_or(0).max(0) as usize + 1;
    let mut order: Vec<usize> = (0..l).collect();
    order.sort_by_key(|&i| {
        let (x, y) = positions[i];
        let kx = (x as usize) / k;
        let ky = (y as usize) / k;
        let num_from_tl = k * k * kx + k * max_x * ky;
        let px = (x as usize) % k;
        let py = (y as usize) % k;
        num_from_tl + px + py * k
    });

    let mut kernel_ordered: Vec<f32> = vec![0.0; l * teacher_patch_dim];
    let mut kernel_pos: Vec<(i32, i32)> = vec![(0, 0); l];
    for (out_i, &src_i) in order.iter().enumerate() {
        kernel_ordered[out_i * teacher_patch_dim..(out_i + 1) * teacher_patch_dim]
            .copy_from_slice(&patches[src_i * teacher_patch_dim..(src_i + 1) * teacher_patch_dim]);
        kernel_pos[out_i] = positions[src_i];
    }

    let mut merged = vec![0f32; num_model_patches * model_dim];
    let mut merged_pos = vec![(-1, -1); num_model_patches];

    for mp in 0..num_model_patches {
        let base = mp * k * k;
        let mut min_x = i32::MAX;
        let mut min_y = i32::MAX;
        let mut out_off = 0usize;
        for ky in 0..k {
            for kx in 0..k {
                let ti = base + ky * k + kx;
                let (x, y) = kernel_pos[ti];
                if x >= 0 {
                    min_x = min_x.min(x / k as i32);
                    min_y = min_y.min(y / k as i32);
                }
                // Spatial merge: place k×k teacher tiles into model patch grid.
                for py in 0..patch_size {
                    for px in 0..patch_size {
                        for c in 0..3 {
                            let src = ti * teacher_patch_dim + (py * patch_size + px) * 3 + c;
                            let dst = mp * model_dim
                                + ((ky * patch_size + py) * (k * patch_size)
                                    + (kx * patch_size + px))
                                    * 3
                                + c;
                            merged[dst] = kernel_ordered[src];
                        }
                    }
                }
                out_off += 1;
            }
        }
        let _ = out_off;
        if min_x != i32::MAX {
            merged_pos[mp] = (min_x, min_y);
        }
    }
    Ok((merged, merged_pos))
}

pub fn pad_patches_to_max(
    patches: Vec<f32>,
    positions: Vec<(i32, i32)>,
    model_dim: usize,
    max_slots: usize,
) -> (Vec<f32>, Vec<(i32, i32)>) {
    let n = patches.len() / model_dim;
    let mut out = vec![0f32; max_slots * model_dim];
    let mut pos = vec![(-1, -1); max_slots];
    out[..n * model_dim].copy_from_slice(&patches);
    pos[..n].copy_from_slice(&positions);
    (out, pos)
}

/// Full unified image pipeline from a JPEG/PNG path.
pub fn load_unified_image(
    path: impl AsRef<std::path::Path>,
    patch_size: usize,
    pooling_kernel_size: usize,
    max_soft_tokens: usize,
) -> Result<UnifiedImageBatch> {
    let img = image::open(path.as_ref())
        .map_err(|e| anyhow::anyhow!("decode {:?}: {e}", path.as_ref()))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let max_patches = max_soft_tokens * pooling_kernel_size * pooling_kernel_size;
    let (th, tw) =
        aspect_ratio_preserving_size(h, w, patch_size, max_patches, pooling_kernel_size)?;
    let resized = if (tw, th) != (w, h) {
        image::DynamicImage::ImageRgb8(rgb)
            .resize_exact(tw as u32, th as u32, image::imageops::FilterType::Triangle)
            .to_rgb8()
    } else {
        rgb
    };
    let (teacher, tpos) = teacher_patches_from_rgb(
        resized.as_raw(),
        resized.width() as usize,
        resized.height() as usize,
        patch_size,
    )?;
    let teacher_dim = patch_size * patch_size * 3;
    let num_model = teacher.len() / teacher_dim / (pooling_kernel_size * pooling_kernel_size);
    let (merged, mpos) = patches_merge(&teacher, &tpos, num_model, teacher_dim)?;
    let model_dim = (patch_size * pooling_kernel_size).pow(2) * 3;
    let num_valid = num_model;
    let (patches, positions) = pad_patches_to_max(merged, mpos, model_dim, max_soft_tokens);
    Ok(UnifiedImageBatch {
        patches,
        positions,
        num_valid,
    })
}

/// Factorized 2D positional bias: `pos_embedding[p, axis, d]` with
/// shape `[posemb_size, 2, dim]` (row-major).
pub fn factorized_pos_bias(
    pos_embedding: &[f32],
    posemb_size: usize,
    dim: usize,
    positions: &[(i32, i32)],
) -> Vec<f32> {
    let mut out = vec![0f32; positions.len() * dim];
    for (i, &(x, y)) in positions.iter().enumerate() {
        if x < 0 || y < 0 {
            continue;
        }
        let x = x as usize;
        let y = y as usize;
        if x >= posemb_size || y >= posemb_size {
            continue;
        }
        let x_base = (x * 2) * dim;
        let y_base = (y * 2 + 1) * dim;
        for d in 0..dim {
            out[i * dim + d] = pos_embedding[x_base + d] + pos_embedding[y_base + d];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patches_merge_square_grid() {
        let k = 3;
        let ps = 16;
        let td = ps * ps * 3;
        let _side = k * 3; // 3 model patches per side → 9 teacher per side? 
        // 2 model patches: 2*3=6 teacher per side
        let cols = 6;
        let rows = 6;
        let l = cols * rows;
        let mut patches = vec![0f32; l * td];
        let mut pos = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let i = r * cols + c;
                pos.push((c as i32, r as i32));
                patches[i * td] = (i + 1) as f32;
            }
        }
        let num_model = l / (k * k);
        let (merged, mpos) = patches_merge(&patches, &pos, num_model, td).unwrap();
        assert_eq!(merged.len(), num_model * (k * ps).pow(2) * 3);
        assert_eq!(mpos.len(), num_model);
        assert!(mpos[0].0 >= 0);
    }
}
