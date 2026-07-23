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

//! Image preprocessing — exact port of `baidu/Unlimited-OCR`'s
//! `BasicImageTransform` + `dynamic_preprocess` + `infer`/`infer_multi`
//! image-handling from `modeling_unlimitedocr.py`.
//!
//! Three checkpoint-supported modes (HF README):
//! - **Base**: `base_size=1024, image_size=1024, crop_mode=False` — one
//!   square "global view", no tiles.
//! - **Gundam**: `base_size=1024, image_size=640, crop_mode=True` — one
//!   square global view at `base_size` plus dynamically-tiled `image_size`
//!   crops when the page doesn't already fit in 640×640.
//! - **Multi** (page/PDF batches): `image_size=1024`, `infer_multi` — each
//!   page gets an independent Base-style view; tiling is never used.
//!
//! Pixel pipeline: EXIF-correct orientation → RGB → `(x/255 - 0.5)/0.5`
//! (mean/std 0.5 → `[-1, 1]`) → `NCHW` (`[3, H, W]`, no batch dim — batch by
//! stacking [`PreprocessedImage`]s upstream).

use crate::config;
use anyhow::{Context, Result, bail, ensure};
use image::imageops::{FilterType, overlay};
use image::metadata::Orientation;
use image::{DynamicImage, GenericImageView, ImageDecoder, Rgb, RgbImage};
use std::path::Path;

/// HF `image_transform.mean` scaled to `u8` — border-fill color for padded views.
pub const PAD_COLOR: [u8; 3] = [127, 127, 127];
/// `dynamic_preprocess(min_num=2, ...)` default.
pub const DYNAMIC_MIN_NUM: u32 = 2;
/// `dynamic_preprocess(max_num=32, ...)` default.
pub const DYNAMIC_MAX_NUM: u32 = 32;
/// Tiling never engages below this size on either axis (`infer`'s
/// `image.size[0] <= 640 and image.size[1] <= 640` check).
pub const GUNDAM_NO_CROP_THRESHOLD: u32 = 640;
/// Resample filter approximating PIL's default `Resampling.BICUBIC`.
const RESAMPLE: FilterType = FilterType::CatmullRom;

/// Which of the checkpoint's three supported layouts to preprocess for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageMode {
    /// `base_size=1024, image_size=1024, crop_mode=False` — single square view.
    Base { size: u32 },
    /// `base_size=1024, image_size=640, crop_mode=True` — global view + dynamic tiles.
    Gundam { base: u32, tile: u32 },
    /// Multi-page/PDF: one Base-style view per page, no tiling.
    Multi { size: u32 },
}

impl Default for ImageMode {
    fn default() -> Self {
        ImageMode::Gundam {
            base: 1024,
            tile: 640,
        }
    }
}

impl ImageMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "base" => Some(Self::Base { size: 1024 }),
            "gundam" => Some(Self::default()),
            "multi" | "multipage" | "pdf" => Some(Self::Multi { size: 1024 }),
            _ => None,
        }
    }
}

/// Rasterize a PDF to PNG pages via `pdftoppm` (300 DPI).
pub fn pdf_to_page_images(pdf_path: &Path, out_dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    std::fs::create_dir_all(out_dir)?;
    let prefix = out_dir.join("page");
    let status = std::process::Command::new("pdftoppm")
        .args([
            "-png",
            "-r",
            "300",
            pdf_path.to_str().context("pdf utf-8")?,
            prefix.to_str().context("prefix utf-8")?,
        ])
        .status()
        .context("spawn pdftoppm (install poppler)")?;
    if !status.success() {
        bail!("pdftoppm failed with {status}");
    }
    let mut pages: Vec<std::path::PathBuf> = std::fs::read_dir(out_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("png"))
        })
        .collect();
    pages.sort();
    ensure!(!pages.is_empty(), "pdftoppm produced no pages");
    Ok(pages)
}

/// EXIF-transposed pixels + tiling metadata for one page/document image.
#[derive(Debug, Clone)]
pub struct PreprocessedImage {
    /// `[3, global_size, global_size]` NCHW, normalized to `[-1, 1]`.
    pub global: Vec<f32>,
    pub global_size: u32,
    /// Dynamic-preprocess crops, each `[3, tile_size, tile_size]` NCHW. Empty
    /// outside Gundam mode or when the source already fits `tile_size`.
    pub tiles: Vec<Vec<f32>>,
    pub tile_size: u32,
    /// `[width_crop_num, height_crop_num]` (HF `images_spatial_crop` row); `[1, 1]` when untiled.
    pub spatial_crop: [u32; 2],
    /// Original (pre-transform) pixel dimensions, for downstream box/coord scaling.
    pub orig_w: u32,
    pub orig_h: u32,
}

impl PreprocessedImage {
    pub fn has_tiles(&self) -> bool {
        !self.tiles.is_empty()
    }

    /// Vision-query tokens contributed by the global view alone (`q*(q+1)+1`).
    pub fn global_query_tokens(&self, patch_size: usize, downsample_ratio: usize) -> usize {
        base_image_tokens(config::num_queries_with(
            self.global_size as usize,
            patch_size,
            downsample_ratio,
        ))
    }

    /// Vision-query tokens contributed by the tile grid (`0` when untiled).
    pub fn tile_query_tokens(&self, patch_size: usize, downsample_ratio: usize) -> usize {
        if !self.has_tiles() {
            return 0;
        }
        let q = config::num_queries_with(self.tile_size as usize, patch_size, downsample_ratio);
        let [w, h] = self.spatial_crop;
        gundam_tile_tokens(q, w, h)
    }

    /// Total `<image>`-placeholder tokens this image expands to in the prompt.
    pub fn token_count(&self, patch_size: usize, downsample_ratio: usize) -> usize {
        self.global_query_tokens(patch_size, downsample_ratio)
            + self.tile_query_tokens(patch_size, downsample_ratio)
    }

    /// The `image_token_id` placeholder run this image expands to, in the
    /// exact order HF concatenates global-view then tile-grid tokens.
    pub fn image_token_ids(
        &self,
        image_token_id: u32,
        patch_size: usize,
        downsample_ratio: usize,
    ) -> Vec<u32> {
        let q = config::num_queries_with(self.global_size as usize, patch_size, downsample_ratio);
        let mut ids = base_view_token_ids(image_token_id, q);
        if self.has_tiles() {
            let qt =
                config::num_queries_with(self.tile_size as usize, patch_size, downsample_ratio);
            let [w, h] = self.spatial_crop;
            ids.extend(tile_view_token_ids(image_token_id, qt, w, h));
        }
        ids
    }
}

/// A batch of pages (single image for Base/Gundam, N pages for Multi).
#[derive(Debug, Clone, Default)]
pub struct PreprocessedBatch {
    pub images: Vec<PreprocessedImage>,
}

// ---------------------------------------------------------------------
// Loading (EXIF orientation).
// ---------------------------------------------------------------------

/// `load_image()` in `modeling_unlimitedocr.py`: `Image.open` +
/// `ImageOps.exif_transpose`. JPEG/TIFF/WebP decoders in the `image` crate
/// read EXIF orientation natively; other formats have no orientation tag and
/// this is a no-op for them.
pub fn load_image_exif_corrected(path: &Path) -> Result<DynamicImage> {
    let reader = image::ImageReader::open(path)
        .with_context(|| format!("open image {path:?}"))?
        .with_guessed_format()
        .with_context(|| format!("guess image format {path:?}"))?;
    let mut decoder = reader
        .into_decoder()
        .with_context(|| format!("create decoder {path:?}"))?;
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut img =
        DynamicImage::from_decoder(decoder).with_context(|| format!("decode pixels {path:?}"))?;
    img.apply_orientation(orientation);
    Ok(img)
}

// ---------------------------------------------------------------------
// Pixel transform: RGB, mean/std 0.5 -> [-1, 1], NCHW.
// ---------------------------------------------------------------------

/// `BasicImageTransform(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), normalize=True)`.
pub fn rgb_to_chw_normalized(img: &RgbImage) -> Vec<f32> {
    let (w, h) = img.dimensions();
    let (w, h) = (w as usize, h as usize);
    let hw = h * w;
    let mut out = vec![0f32; 3 * hw];
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let v = p[c] as f32 / 255.0;
                out[c * hw + y * w + x] = (v - 0.5) / 0.5;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------
// ImageOps.pad equivalent (letterbox to square, centered, pad-color border).
// ---------------------------------------------------------------------

/// `ImageOps.contain` target size for a square destination: preserve aspect
/// ratio, one axis exactly `size`, the other `floor(size * short/long)`.
fn contain_size(w: u32, h: u32, size: u32) -> (u32, u32) {
    if w == h {
        return (size, size);
    }
    let im_ratio = w as f64 / h as f64;
    if im_ratio > 1.0 {
        let new_height = ((h as f64 / w as f64) * size as f64) as u32;
        (size, new_height.max(1))
    } else {
        let new_width = ((w as f64 / h as f64) * size as f64) as u32;
        (new_width.max(1), size)
    }
}

/// `ImageOps.pad(image, (size, size), color=pad_color)` — letterbox to a
/// square canvas of `size`×`size`, centered, borders filled with `pad_color`.
pub fn pad_to_square(image: &DynamicImage, size: u32, pad_color: [u8; 3]) -> RgbImage {
    let rgb = image.to_rgb8();
    let (w, h) = rgb.dimensions();
    let (new_w, new_h) = contain_size(w, h, size);
    let resized = if new_w == w && new_h == h {
        rgb
    } else {
        DynamicImage::ImageRgb8(rgb)
            .resize_exact(new_w, new_h, RESAMPLE)
            .to_rgb8()
    };
    if new_w == size && new_h == size {
        return resized;
    }
    let mut canvas = RgbImage::from_pixel(size, size, Rgb(pad_color));
    // PIL pastes at (x, 0) or (0, y) depending on which axis is short;
    // `contain_size` guarantees exactly one of these offsets is non-zero.
    let off_x = ((size - new_w) as f64 * 0.5) as i64;
    let off_y = ((size - new_h) as f64 * 0.5) as i64;
    overlay(&mut canvas, &resized, off_x, off_y);
    canvas
}

// ---------------------------------------------------------------------
// dynamic_preprocess (Gundam tiling).
// ---------------------------------------------------------------------

/// One page's dynamic tile grid.
#[derive(Debug, Clone)]
pub struct DynamicTiles {
    /// Row-major tile order (`height_crop_num` rows of `width_crop_num` tiles).
    pub tiles: Vec<RgbImage>,
    pub width_crop_num: u32,
    pub height_crop_num: u32,
}

/// `target_ratios` set/sort from `dynamic_preprocess` — every `(i, j)` with
/// `1 <= i, j <= n` and `min_num <= i*j <= max_num` for some `n` in
/// `[min_num, max_num]`, deduplicated, sorted ascending by `i*j` (block count).
fn target_ratios(min_num: u32, max_num: u32) -> Vec<(u32, u32)> {
    let mut set = std::collections::BTreeSet::new();
    for n in min_num..=max_num {
        for i in 1..=n {
            for j in 1..=n {
                let blocks = i * j;
                if blocks <= max_num && blocks >= min_num {
                    set.insert((i, j));
                }
            }
        }
    }
    let mut ratios: Vec<(u32, u32)> = set.into_iter().collect();
    ratios.sort_by_key(|&(i, j)| i * j);
    ratios
}

/// `find_closest_aspect_ratio` — picks the `(i, j)` from `ratios` whose
/// `i/j` is closest to `aspect_ratio`; ties broken toward the larger block
/// count when the source image area justifies the extra resolution.
fn find_closest_aspect_ratio(
    aspect_ratio: f64,
    ratios: &[(u32, u32)],
    width: u32,
    height: u32,
    image_size: u32,
) -> (u32, u32) {
    let mut best_diff = f64::INFINITY;
    let mut best = (1u32, 1u32);
    let area = width as f64 * height as f64;
    for &(i, j) in ratios {
        let target = i as f64 / j as f64;
        let diff = (aspect_ratio - target).abs();
        if diff < best_diff {
            best_diff = diff;
            best = (i, j);
        } else if diff == best_diff
            && area > 0.5 * image_size as f64 * image_size as f64 * i as f64 * j as f64
        {
            best = (i, j);
        }
    }
    best
}

/// `dynamic_preprocess(image, min_num=2, max_num=32, image_size=640)`.
///
/// Always returns `width_crop_num * height_crop_num >= 2` tiles (since
/// `min_num=2` forbids the `(1, 1)` ratio from ever being selected).
pub fn dynamic_preprocess(
    image: &DynamicImage,
    min_num: u32,
    max_num: u32,
    image_size: u32,
) -> DynamicTiles {
    let (orig_w, orig_h) = image.dimensions();
    let aspect_ratio = orig_w as f64 / orig_h as f64;
    let ratios = target_ratios(min_num, max_num);
    let (width_crop_num, height_crop_num) =
        find_closest_aspect_ratio(aspect_ratio, &ratios, orig_w, orig_h, image_size);

    let target_w = image_size * width_crop_num;
    let target_h = image_size * height_crop_num;
    let resized = image.resize_exact(target_w, target_h, RESAMPLE);

    let blocks = width_crop_num * height_crop_num;
    let mut tiles = Vec::with_capacity(blocks as usize);
    for idx in 0..blocks {
        let col = idx % width_crop_num;
        let row = idx / width_crop_num;
        let tile = resized
            .crop_imm(col * image_size, row * image_size, image_size, image_size)
            .to_rgb8();
        tiles.push(tile);
    }

    DynamicTiles {
        tiles,
        width_crop_num,
        height_crop_num,
    }
}

// ---------------------------------------------------------------------
// image_token_id placeholder-run construction (mirrors HF `tokenized_image`).
// ---------------------------------------------------------------------

/// `([image_token_id] * q + [image_token_id]) * q + [image_token_id]`
/// (base/global-view token run: `q` rows of `q` tokens + row separator, plus
/// one trailing separator). Length: `q*(q+1) + 1`.
fn base_view_token_ids(image_token_id: u32, q: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(q * (q + 1) + 1);
    for _ in 0..q {
        out.extend(std::iter::repeat_n(image_token_id, q));
        out.push(image_token_id);
    }
    out.push(image_token_id);
    out
}

/// `([image_token_id] * (q*W) + [image_token_id]) * (q*H)` (tile-grid token
/// run). Length: `(q*H) * (q*W + 1)`.
fn tile_view_token_ids(
    image_token_id: u32,
    q: usize,
    width_crop_num: u32,
    height_crop_num: u32,
) -> Vec<u32> {
    let cols = q * width_crop_num as usize;
    let rows = q * height_crop_num as usize;
    let mut out = Vec::with_capacity(rows * (cols + 1));
    for _ in 0..rows {
        out.extend(std::iter::repeat_n(image_token_id, cols));
        out.push(image_token_id);
    }
    out
}

/// `q*(q+1) + 1` — token count of a single (untiled) view of query-grid side `q`.
pub fn base_image_tokens(q: usize) -> usize {
    q * (q + 1) + 1
}

/// `(q*H) * (q*W + 1)` — token count of a `width_crop_num x height_crop_num`
/// tile grid, each tile contributing a `q x q` query grid.
pub fn gundam_tile_tokens(q: usize, width_crop_num: u32, height_crop_num: u32) -> usize {
    (q * height_crop_num as usize) * (q * width_crop_num as usize + 1)
}

// ---------------------------------------------------------------------
// Per-mode page preprocessing.
// ---------------------------------------------------------------------

/// Base-style single square view: hard-resize below the Gundam tiling
/// threshold, else letterbox-pad. Shared by [`ImageMode::Base`] and
/// [`ImageMode::Multi`] (both use this exact HF branch, just at different
/// default `size`s: 1024 for Base's `base_size`/Multi pages, 1024 or smaller
/// custom sizes are equally valid).
fn preprocess_base_view(image: &DynamicImage, size: u32) -> RgbImage {
    if size <= GUNDAM_NO_CROP_THRESHOLD {
        // `image.resize((size, size))`: a direct (aspect-ratio-ignoring) stretch.
        image.resize_exact(size, size, RESAMPLE).to_rgb8()
    } else {
        // `ImageOps.pad(image, (size, size), color=pad_color)`.
        pad_to_square(image, size, PAD_COLOR)
    }
}

/// Preprocess one page/document image under `mode`.
pub fn preprocess_one(image: &DynamicImage, mode: ImageMode) -> PreprocessedImage {
    let (orig_w, orig_h) = image.dimensions();
    match mode {
        ImageMode::Base { size } | ImageMode::Multi { size } => {
            let view = preprocess_base_view(image, size);
            PreprocessedImage {
                global: rgb_to_chw_normalized(&view),
                global_size: size,
                tiles: Vec::new(),
                tile_size: 0,
                spatial_crop: [1, 1],
                orig_w,
                orig_h,
            }
        }
        ImageMode::Gundam { base, tile } => {
            let global_view = pad_to_square(image, base, PAD_COLOR);
            let (tiles_rgb, width_crop_num, height_crop_num) =
                if orig_w <= GUNDAM_NO_CROP_THRESHOLD && orig_h <= GUNDAM_NO_CROP_THRESHOLD {
                    (Vec::new(), 1, 1)
                } else {
                    let dt = dynamic_preprocess(image, DYNAMIC_MIN_NUM, DYNAMIC_MAX_NUM, tile);
                    (dt.tiles, dt.width_crop_num, dt.height_crop_num)
                };
            let tiles: Vec<Vec<f32>> = tiles_rgb.iter().map(rgb_to_chw_normalized).collect();
            PreprocessedImage {
                global: rgb_to_chw_normalized(&global_view),
                global_size: base,
                tiles,
                tile_size: tile,
                spatial_crop: [width_crop_num, height_crop_num],
                orig_w,
                orig_h,
            }
        }
    }
}

/// Preprocess a batch of pages under `mode` (one element for Base/Gundam,
/// any number of pages for Multi).
pub fn preprocess_batch(images: &[DynamicImage], mode: ImageMode) -> PreprocessedBatch {
    PreprocessedBatch {
        images: images.iter().map(|img| preprocess_one(img, mode)).collect(),
    }
}

/// Load (EXIF-corrected) + preprocess a single page from disk.
pub fn preprocess_path(path: &Path, mode: ImageMode) -> Result<PreprocessedImage> {
    let img = load_image_exif_corrected(path)?;
    Ok(preprocess_one(&img, mode))
}

/// Load (EXIF-corrected) + preprocess several pages from disk (Multi mode).
pub fn preprocess_paths(
    paths: &[std::path::PathBuf],
    mode: ImageMode,
) -> Result<PreprocessedBatch> {
    let images = paths
        .iter()
        .map(|p| load_image_exif_corrected(p))
        .collect::<Result<Vec<_>>>()?;
    Ok(preprocess_batch(&images, mode))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(w, h, Rgb([200, 30, 30])))
    }

    #[test]
    fn contain_size_preserves_wider_than_tall() {
        // 800x400 (2:1) -> width becomes 1024, height floor(400/800*1024)=512.
        assert_eq!(contain_size(800, 400, 1024), (1024, 512));
    }

    #[test]
    fn contain_size_preserves_taller_than_wide() {
        assert_eq!(contain_size(400, 800, 1024), (512, 1024));
    }

    #[test]
    fn contain_size_square_is_identity() {
        assert_eq!(contain_size(500, 500, 1024), (1024, 1024));
    }

    #[test]
    fn pad_to_square_produces_exact_target_dimensions() {
        let img = solid(300, 700);
        let out = pad_to_square(&img, 1024, PAD_COLOR);
        assert_eq!(out.dimensions(), (1024, 1024));
        // Border pixel should be the pad color (short axis was letterboxed).
        assert_eq!(*out.get_pixel(0, 0), Rgb(PAD_COLOR));
    }

    #[test]
    fn base_mode_small_image_is_hard_resized_no_padding() {
        let img = solid(300, 100);
        let pre = preprocess_one(&img, ImageMode::Base { size: 640 });
        assert_eq!(pre.global_size, 640);
        assert!(!pre.has_tiles());
        assert_eq!(pre.spatial_crop, [1, 1]);
        assert_eq!(pre.global.len(), 3 * 640 * 640);
    }

    #[test]
    fn base_mode_large_image_uses_letterbox_pad() {
        let img = solid(2000, 1000);
        let pre = preprocess_one(&img, ImageMode::Base { size: 1024 });
        assert_eq!(pre.global_size, 1024);
        assert_eq!(pre.global.len(), 3 * 1024 * 1024);
        assert!(!pre.has_tiles());
    }

    #[test]
    fn gundam_mode_small_image_has_no_tiles() {
        let img = solid(500, 300);
        let pre = preprocess_one(
            &img,
            ImageMode::Gundam {
                base: 1024,
                tile: 640,
            },
        );
        assert_eq!(pre.spatial_crop, [1, 1]);
        assert!(!pre.has_tiles());
        assert_eq!(pre.global_size, 1024);
    }

    #[test]
    fn gundam_mode_large_image_produces_tiles() {
        let img = solid(3000, 1000);
        let pre = preprocess_one(
            &img,
            ImageMode::Gundam {
                base: 1024,
                tile: 640,
            },
        );
        assert!(pre.has_tiles());
        let [w, h] = pre.spatial_crop;
        assert!(w >= 1 && h >= 1);
        assert_eq!(pre.tiles.len(), (w * h) as usize);
        for t in &pre.tiles {
            assert_eq!(t.len(), 3 * 640 * 640);
        }
    }

    #[test]
    fn multi_mode_matches_base_mode_per_page() {
        let img = solid(2000, 1000);
        let base = preprocess_one(&img, ImageMode::Base { size: 1024 });
        let multi = preprocess_one(&img, ImageMode::Multi { size: 1024 });
        assert_eq!(base.global, multi.global);
        assert_eq!(base.spatial_crop, multi.spatial_crop);
    }

    #[test]
    fn dynamic_preprocess_always_yields_at_least_two_tiles() {
        let img = solid(1300, 1290);
        let dt = dynamic_preprocess(&img, DYNAMIC_MIN_NUM, DYNAMIC_MAX_NUM, 640);
        assert!(dt.width_crop_num * dt.height_crop_num >= 2);
        assert_eq!(
            dt.tiles.len(),
            (dt.width_crop_num * dt.height_crop_num) as usize
        );
    }

    #[test]
    fn dynamic_preprocess_wide_image_prefers_wide_grid() {
        let img = solid(4000, 1000); // very wide -> width_crop_num > height_crop_num
        let dt = dynamic_preprocess(&img, DYNAMIC_MIN_NUM, DYNAMIC_MAX_NUM, 640);
        assert!(dt.width_crop_num >= dt.height_crop_num);
    }

    #[test]
    fn num_queries_matches_config_defaults() {
        assert_eq!(config::num_queries(1024), 16);
        assert_eq!(config::num_queries(640), 10);
    }

    #[test]
    fn base_image_tokens_matches_hf_formula() {
        // 1024-px global view: q=16 -> 16*17 + 1 = 273 placeholder tokens.
        assert_eq!(base_image_tokens(16), 273);
        assert_eq!(base_view_token_ids(999, 16).len(), 273);
    }

    #[test]
    fn gundam_tile_tokens_matches_hf_formula() {
        // q=10 tile grid, 2x1 crop: (10*1)*(10*2+1) = 210.
        assert_eq!(gundam_tile_tokens(10, 2, 1), 210);
        assert_eq!(tile_view_token_ids(999, 10, 2, 1).len(), 210);
    }

    #[test]
    fn preprocessed_image_token_count_matches_helpers() {
        let img = solid(3000, 1000);
        let pre = preprocess_one(
            &img,
            ImageMode::Gundam {
                base: 1024,
                tile: 640,
            },
        );
        let ids = pre.image_token_ids(128_815, 16, 4);
        assert_eq!(ids.len(), pre.token_count(16, 4));
        assert!(ids.iter().all(|&t| t == 128_815));
        // Global view contributes exactly base_image_tokens(16) = 273 tokens.
        assert_eq!(pre.global_query_tokens(16, 4), 273);
    }

    #[test]
    fn probe_sample_image_exif_load_is_graceful_when_missing() {
        // No bundled fixture yet in this crate; just confirm the loader
        // reports a clean error rather than panicking.
        let missing = Path::new("/nonexistent/rlx-unlimited-ocr-probe.jpg");
        assert!(load_image_exif_corrected(missing).is_err());
    }
}
