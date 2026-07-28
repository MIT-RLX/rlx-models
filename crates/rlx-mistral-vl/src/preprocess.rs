// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Pixtral image preprocess — CLIP mean/std + longest-edge resize.

use crate::config::PixtralVisionConfig;
use anyhow::{Result, ensure};
use image::imageops::FilterType;

/// Resize preserving aspect so the longest edge ≤ `longest`, aligned to `align`.
pub fn smart_resize_longest_edge(
    w: usize,
    h: usize,
    align: usize,
    longest_edge: usize,
) -> (usize, usize) {
    if w == 0 || h == 0 || longest_edge == 0 || align == 0 {
        return (0, 0);
    }
    let scale = (longest_edge as f32 / w as f32).min(longest_edge as f32 / h as f32);
    let tw = ceil_by_factor(w as f32 * scale, align).max(align);
    let th = ceil_by_factor(h as f32 * scale, align).max(align);
    (tw, th)
}

fn ceil_by_factor(x: f32, f: usize) -> usize {
    ((x / f as f32).ceil() as usize) * f
}

/// RGB bytes → NCHW f32 normalized patches flattened as `[n_patches * patch_dim]`.
pub fn image_to_patch_rows(
    rgb: &[u8],
    img_w: usize,
    img_h: usize,
    cfg: &PixtralVisionConfig,
) -> Result<(Vec<f32>, usize, usize)> {
    ensure!(
        rgb.len() == img_w.saturating_mul(img_h).saturating_mul(3),
        "rgb len {} != {img_w}×{img_h}×3",
        rgb.len()
    );
    let align = cfg.align_size();
    let (tw, th) = smart_resize_longest_edge(img_w, img_h, align, cfg.image_size);
    ensure!(tw > 0 && th > 0, "invalid resize {tw}×{th}");

    let img = image::RgbImage::from_raw(img_w as u32, img_h as u32, rgb.to_vec())
        .ok_or_else(|| anyhow::anyhow!("wrap rgb"))?;
    let dynimg = image::DynamicImage::ImageRgb8(img);
    let resized = dynimg.resize_exact(tw as u32, th as u32, FilterType::CatmullRom);
    let rgb8 = resized.to_rgb8();

    let ps = cfg.patch_size;
    let grid_y = th / ps;
    let grid_x = tw / ps;
    let n_patches = grid_y * grid_x;
    let patch_dim = cfg.num_channels * ps * ps;
    let mut patches = vec![0f32; n_patches * patch_dim];

    for gy in 0..grid_y {
        for gx in 0..grid_x {
            let row = gy * grid_x + gx;
            for py in 0..ps {
                for px in 0..ps {
                    let x = (gx * ps + px) as u32;
                    let y = (gy * ps + py) as u32;
                    let p = rgb8.get_pixel(x, y);
                    for c in 0..3 {
                        let v = (p[c] as f32 / 255.0 - cfg.image_mean[c]) / cfg.image_std[c];
                        // Channel-major within patch (matches conv unfold C·ps·ps).
                        let off = row * patch_dim + c * ps * ps + py * ps + px;
                        patches[off] = v;
                    }
                }
            }
        }
    }
    Ok((patches, grid_x, grid_y))
}

/// Host patch-embed: patches `[n, patch_dim]` × W where W is GGUF
/// `v.patch_embd.weight` with shape (ps, ps, 3, hidden) → `[n, hidden]`.
pub fn apply_patch_embed(
    patches: &[f32],
    n_patches: usize,
    patch_dim: usize,
    hidden: usize,
    weight: &[f32],
) -> Result<Vec<f32>> {
    ensure!(
        patches.len() == n_patches * patch_dim,
        "patches len {}",
        patches.len()
    );
    ensure!(
        weight.len() == patch_dim * hidden,
        "patch_embd weight len {} != {patch_dim}×{hidden}",
        weight.len()
    );
    // weight layout ne=(ps,ps,3,hidden): index = px + ps*(py + ps*(c + 3*h))
    // which is the same as flattening patch_dim×hidden with patch_dim = ps*ps*3
    // innermost over spatial then channel then out — matches our patch packing
    // (c, py, px) if we reorder. Our packing is c*ps*ps + py*ps + px.
    // GGUF: ne0=ps(x), ne1=ps(y), ne2=3, ne3=hidden
    // flat index for (px,py,c,h) = px + ps*(py + ps*(c + 3*h))
    // Our patch index for (c,py,px) = c*ps*ps + py*ps + px
    // These differ — convert weight to [patch_dim, hidden] row-major for our packing.
    let mut w_rm = vec![0f32; patch_dim * hidden];
    let ps = ((patch_dim / 3) as f32).sqrt() as usize;
    ensure!(ps * ps * 3 == patch_dim, "patch_dim {patch_dim} not 3×ps²");
    for h in 0..hidden {
        for c in 0..3 {
            for py in 0..ps {
                for px in 0..ps {
                    let src = px + ps * (py + ps * (c + 3 * h));
                    let dst_row = c * ps * ps + py * ps + px;
                    w_rm[dst_row * hidden + h] = weight[src];
                }
            }
        }
    }
    let mut out = vec![0f32; n_patches * hidden];
    for n in 0..n_patches {
        for h in 0..hidden {
            let mut acc = 0f32;
            for d in 0..patch_dim {
                acc += patches[n * patch_dim + d] * w_rm[d * hidden + h];
            }
            out[n * hidden + h] = acc;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cfg(image_size: usize) -> PixtralVisionConfig {
        PixtralVisionConfig {
            image_size,
            patch_size: 14,
            spatial_merge_size: 2,
            num_channels: 3,
            // Identity normalization so a constant image is predictable.
            image_mean: [0.0, 0.0, 0.0],
            image_std: [1.0, 1.0, 1.0],
            ..PixtralVisionConfig::default()
        }
    }

    #[test]
    fn resize_scales_to_longest_edge_aligned() {
        // align = patch(14) * merge(2) = 28; scales the long edge to `longest`.
        assert_eq!(smart_resize_longest_edge(56, 28, 28, 1540), (1540, 784));
        // Square within the cap stays square, aligned up.
        assert_eq!(smart_resize_longest_edge(28, 28, 28, 28), (28, 28));
        // Degenerate inputs → (0, 0) rather than a panic.
        assert_eq!(smart_resize_longest_edge(0, 10, 28, 1540), (0, 0));
        assert_eq!(smart_resize_longest_edge(10, 10, 0, 1540), (0, 0));
    }

    #[test]
    fn patch_rows_grid_and_channel_layout() {
        let cfg = test_cfg(28); // 28×28 image → 2×2 grid of 14px patches.
        let (r, g, b) = (100u8, 150u8, 200u8);
        let mut rgb = Vec::with_capacity(28 * 28 * 3);
        for _ in 0..28 * 28 {
            rgb.extend_from_slice(&[r, g, b]);
        }
        let (patches, gx, gy) = image_to_patch_rows(&rgb, 28, 28, &cfg).unwrap();
        assert_eq!((gx, gy), (2, 2));
        let ps = cfg.patch_size;
        let patch_dim = 3 * ps * ps;
        assert_eq!(patches.len(), gx * gy * patch_dim);
        // Channel-major within a patch: [c=0 block | c=1 block | c=2 block].
        // A constant image survives resize unchanged, so each channel block is flat.
        let approx = |a: f32, b: f32| (a - b).abs() < 1e-4;
        assert!(approx(patches[0], r as f32 / 255.0));
        assert!(approx(patches[ps * ps], g as f32 / 255.0));
        assert!(approx(patches[2 * ps * ps], b as f32 / 255.0));
    }
}
