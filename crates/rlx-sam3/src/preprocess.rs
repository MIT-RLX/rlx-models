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

//! Host-side SAM3 preprocessing and patch embedding.
//!
//! SAM3's public builder uses a ViT patch-14 backbone at 1008x1008.
//! RLX does patch projection on the host, matching the existing SAM and
//! DINOv2 ports, because the IR surface does not currently include a
//! general f32 Conv2d forward.

use super::config::{
    SAM3_IMG_SIZE, SAM3_PATCH_GRID, SAM3_PIXEL_MEAN, SAM3_PIXEL_STD, Sam3VitConfig,
};
use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;

#[derive(Clone)]
pub struct Sam3PreprocessWeights {
    /// Patch projection `[patch_dim, embed_dim]`, transposed for row-major matmul.
    pub patch_proj_w: Vec<f32>,
    pub patch_proj_b: Vec<f32>,
    pub pos_embed: Option<Vec<f32>>,
    pub embed_dim: usize,
    pub patch_size: usize,
    pub grid: usize,
}

pub(crate) fn extract_preprocess_weights(
    weights: &mut WeightMap,
    cfg: &Sam3VitConfig,
) -> Result<Sam3PreprocessWeights> {
    let e = cfg.embed_dim;
    let ps = cfg.patch_size;
    let grid = cfg.patch_grid();
    let pd = 3 * ps * ps;

    let (proj_raw, proj_shape) = take_first(
        weights,
        &[
            "detector.backbone.vision_backbone.trunk.patch_embed.proj.weight",
            "detector.backbone.visual.trunk.patch_embed.proj.weight",
            "backbone.vision_backbone.trunk.patch_embed.proj.weight",
            "backbone.visual.trunk.patch_embed.proj.weight",
            "visual.trunk.patch_embed.proj.weight",
            "trunk.patch_embed.proj.weight",
        ],
    )?;
    ensure!(
        proj_shape == vec![e, 3, ps, ps],
        "SAM3 patch_embed.proj.weight expected [{e}, 3, {ps}, {ps}], got {proj_shape:?}"
    );

    let mut patch_proj_w = vec![0f32; e * pd];
    for ei in 0..e {
        for d in 0..pd {
            patch_proj_w[d * e + ei] = proj_raw[ei * pd + d];
        }
    }

    let patch_proj_b = if cfg.bias_patch_embed {
        let (data, shape) = take_first(
            weights,
            &[
                "detector.backbone.vision_backbone.trunk.patch_embed.proj.bias",
                "detector.backbone.visual.trunk.patch_embed.proj.bias",
                "backbone.vision_backbone.trunk.patch_embed.proj.bias",
                "backbone.visual.trunk.patch_embed.proj.bias",
                "visual.trunk.patch_embed.proj.bias",
                "trunk.patch_embed.proj.bias",
            ],
        )?;
        ensure!(
            shape == vec![e],
            "SAM3 patch bias expected [{e}], got {shape:?}"
        );
        data
    } else {
        vec![0.0; e]
    };

    let pos_embed = if cfg.use_abs_pos {
        take_optional_first(
            weights,
            &[
                "detector.backbone.vision_backbone.trunk.pos_embed",
                "detector.backbone.visual.trunk.pos_embed",
                "backbone.vision_backbone.trunk.pos_embed",
                "backbone.visual.trunk.pos_embed",
                "visual.trunk.pos_embed",
                "trunk.pos_embed",
            ],
        )?
        .map(|(data, shape)| materialize_pos_embed(&data, &shape, cfg, grid, e))
        .transpose()?
    } else {
        None
    };

    Ok(Sam3PreprocessWeights {
        patch_proj_w,
        patch_proj_b,
        pos_embed,
        embed_dim: e,
        patch_size: ps,
        grid,
    })
}

/// Resize an RGB u8 image to fit in SAM3's square canvas, normalize, and pad.
pub fn preprocess_image(rgb: &[u8], h_in: usize, w_in: usize) -> (Vec<f32>, (usize, usize)) {
    let scale = (SAM3_IMG_SIZE as f32) / (h_in.max(w_in) as f32);
    let new_h = ((h_in as f32) * scale).round() as usize;
    let new_w = ((w_in as f32) * scale).round() as usize;

    let mut resized = vec![0f32; 3 * new_h * new_w];
    let sx = (w_in as f32 - 1.0) / (new_w.max(1) as f32 - 1.0).max(1.0);
    let sy = (h_in as f32 - 1.0) / (new_h.max(1) as f32 - 1.0).max(1.0);
    for y in 0..new_h {
        let fy = y as f32 * sy;
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(h_in - 1);
        let dy = fy - y0 as f32;
        for x in 0..new_w {
            let fx = x as f32 * sx;
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(w_in - 1);
            let dx = fx - x0 as f32;
            for c in 0..3 {
                let p00 = rgb[(y0 * w_in + x0) * 3 + c] as f32 / 255.0;
                let p01 = rgb[(y0 * w_in + x1) * 3 + c] as f32 / 255.0;
                let p10 = rgb[(y1 * w_in + x0) * 3 + c] as f32 / 255.0;
                let p11 = rgb[(y1 * w_in + x1) * 3 + c] as f32 / 255.0;
                let top = p00 * (1.0 - dx) + p01 * dx;
                let bot = p10 * (1.0 - dx) + p11 * dx;
                let v = top * (1.0 - dy) + bot * dy;
                resized[c * new_h * new_w + y * new_w + x] =
                    (v - SAM3_PIXEL_MEAN[c]) / SAM3_PIXEL_STD[c];
            }
        }
    }

    let mut padded = vec![0f32; 3 * SAM3_IMG_SIZE * SAM3_IMG_SIZE];
    for c in 0..3 {
        for y in 0..new_h {
            let src_row = c * new_h * new_w + y * new_w;
            let dst_row = c * SAM3_IMG_SIZE * SAM3_IMG_SIZE + y * SAM3_IMG_SIZE;
            padded[dst_row..dst_row + new_w].copy_from_slice(&resized[src_row..src_row + new_w]);
        }
    }
    (padded, (new_h, new_w))
}

pub fn assemble_patch_tokens(pre: &Sam3PreprocessWeights, image_nchw: &[f32]) -> Result<Vec<f32>> {
    let e = pre.embed_dim;
    let ps = pre.patch_size;
    let grid = pre.grid;
    let pd = 3 * ps * ps;
    ensure!(
        image_nchw.len() == 3 * SAM3_IMG_SIZE * SAM3_IMG_SIZE,
        "SAM3 image must be [3, {SAM3_IMG_SIZE}, {SAM3_IMG_SIZE}] NCHW, got len {}",
        image_nchw.len()
    );
    ensure!(
        grid == SAM3_PATCH_GRID,
        "SAM3 base grid must be {SAM3_PATCH_GRID}"
    );

    let mut out = vec![0f32; grid * grid * e];
    let mut patch_buf = vec![0f32; pd];
    for py in 0..grid {
        for px in 0..grid {
            for c in 0..3 {
                for ry in 0..ps {
                    let src_y = py * ps + ry;
                    for rx in 0..ps {
                        let src_x = px * ps + rx;
                        let src = c * SAM3_IMG_SIZE * SAM3_IMG_SIZE + src_y * SAM3_IMG_SIZE + src_x;
                        let dst = c * ps * ps + ry * ps + rx;
                        patch_buf[dst] = image_nchw[src];
                    }
                }
            }
            let row = py * grid + px;
            let dst = &mut out[row * e..(row + 1) * e];
            dst.copy_from_slice(&pre.patch_proj_b);
            for d in 0..pd {
                let v = patch_buf[d];
                if v == 0.0 {
                    continue;
                }
                let w_row = &pre.patch_proj_w[d * e..(d + 1) * e];
                for k in 0..e {
                    dst[k] += v * w_row[k];
                }
            }
        }
    }

    if let Some(pos) = &pre.pos_embed {
        ensure!(pos.len() == out.len(), "SAM3 pos_embed size mismatch");
        for i in 0..out.len() {
            out[i] += pos[i];
        }
    }

    Ok(out)
}

/// Materialise the absolute positional embedding so the trunk can add it
/// directly to the [grid, grid, embed_dim] patch tokens. Upstream stores a
/// `[1, num_positions, embed_dim]` pretrain table: when
/// `pretrain_use_cls_token` is set the first row is the CLS position and the
/// rest is a `pretrain_grid x pretrain_grid` table. We then tile (or, when
/// `tile_abs_pos=False`, bicubic-interpolate) to the deployment grid.
fn materialize_pos_embed(
    data: &[f32],
    shape: &[usize],
    cfg: &Sam3VitConfig,
    grid: usize,
    e: usize,
) -> Result<Vec<f32>> {
    if shape == [1, grid, grid, e] || shape == [grid, grid, e] {
        return Ok(data.to_vec());
    }
    ensure!(
        shape.len() == 3 && shape[0] == 1 && shape[2] == e,
        "SAM3 pos_embed expected [1, *, {e}], got {shape:?}"
    );
    let num_positions = shape[1];
    let has_cls = num_positions % 2 == 1;
    let spatial = if has_cls {
        num_positions - 1
    } else {
        num_positions
    };
    let pretrain_grid = (spatial as f64).sqrt().round() as usize;
    ensure!(
        pretrain_grid * pretrain_grid == spatial,
        "SAM3 pos_embed spatial portion not square: {spatial} positions"
    );

    let src = if has_cls { &data[e..] } else { data };
    let mut out = vec![0f32; grid * grid * e];

    if cfg.tile_abs_pos {
        for y in 0..grid {
            for x in 0..grid {
                let sy = y % pretrain_grid;
                let sx = x % pretrain_grid;
                let src_row = (sy * pretrain_grid + sx) * e;
                let dst_row = (y * grid + x) * e;
                out[dst_row..dst_row + e].copy_from_slice(&src[src_row..src_row + e]);
            }
        }
    } else {
        // Bicubic interpolation (matches torch.nn.functional.interpolate
        // mode="bicubic", align_corners=False).
        bicubic_interp_nhwc(src, pretrain_grid, pretrain_grid, &mut out, grid, grid, e);
    }

    Ok(out)
}

fn bicubic_interp_nhwc(
    src: &[f32],
    src_h: usize,
    src_w: usize,
    dst: &mut [f32],
    dst_h: usize,
    dst_w: usize,
    c: usize,
) {
    // Convert to [C, H, W] for sampling, then back.
    let mut src_chw = vec![0f32; c * src_h * src_w];
    for y in 0..src_h {
        for x in 0..src_w {
            for ch in 0..c {
                src_chw[ch * src_h * src_w + y * src_w + x] = src[(y * src_w + x) * c + ch];
            }
        }
    }
    let scale_y = src_h as f32 / dst_h as f32;
    let scale_x = src_w as f32 / dst_w as f32;
    for y in 0..dst_h {
        let fy = (y as f32 + 0.5) * scale_y - 0.5;
        let y_floor = fy.floor() as i32;
        let dy = fy - y_floor as f32;
        let wy = cubic_weights(dy);
        for x in 0..dst_w {
            let fx = (x as f32 + 0.5) * scale_x - 0.5;
            let x_floor = fx.floor() as i32;
            let dx = fx - x_floor as f32;
            let wx = cubic_weights(dx);
            for ch in 0..c {
                let plane = &src_chw[ch * src_h * src_w..(ch + 1) * src_h * src_w];
                let mut v = 0.0f32;
                for j in -1..=2 {
                    let sy = (y_floor + j).clamp(0, src_h as i32 - 1) as usize;
                    let mut row_acc = 0.0f32;
                    for i in -1..=2 {
                        let sx = (x_floor + i).clamp(0, src_w as i32 - 1) as usize;
                        row_acc += plane[sy * src_w + sx] * wx[(i + 1) as usize];
                    }
                    v += row_acc * wy[(j + 1) as usize];
                }
                dst[(y * dst_w + x) * c + ch] = v;
            }
        }
    }
}

fn cubic_weights(t: f32) -> [f32; 4] {
    // Cubic convolution kernel with a=-0.75 (matches PyTorch's bicubic).
    let a = -0.75f32;
    let t1 = 1.0 + t; // distance to leftmost
    let t2 = t; // distance to next
    let t3 = 1.0 - t; // distance to next
    let t4 = 2.0 - t; // distance to rightmost
    [
        cubic_kernel(t1, a),
        cubic_kernel(t2, a),
        cubic_kernel(t3, a),
        cubic_kernel(t4, a),
    ]
}

fn cubic_kernel(x: f32, a: f32) -> f32 {
    let x = x.abs();
    if x < 1.0 {
        (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0
    } else if x < 2.0 {
        a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a
    } else {
        0.0
    }
}

fn take_first(weights: &mut WeightMap, keys: &[&str]) -> Result<(Vec<f32>, Vec<usize>)> {
    for key in keys {
        if weights.has(key) {
            return weights.take(key);
        }
    }
    anyhow::bail!("none of the SAM3 weight keys were found: {keys:?}")
}

fn take_optional_first(
    weights: &mut WeightMap,
    keys: &[&str],
) -> Result<Option<(Vec<f32>, Vec<usize>)>> {
    for key in keys {
        if weights.has(key) {
            return weights.take(key).map(Some);
        }
    }
    Ok(None)
}
