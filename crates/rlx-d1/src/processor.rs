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

//! LFM2-VL's image processor (`Lfm2VlImageProcessor` + `Lfm2VlProcessor`'s
//! token layout) and the pixel unshuffle after the tower: plain host math
//! over pictures, which the exported package's program builds too
//! ([`crate::vision`] runs the tower and the projector).

use std::path::Path;

use anyhow::{Context, Result, ensure};
use rlx_core::image_preprocess::{Filter, torch_resize_rgb8_aa};

/// An RGB8 picture, row-major `[height, width, 3]`.
#[derive(Debug, Clone)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl RgbImage {
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Result<Self> {
        ensure!(
            data.len() == width * height * 3,
            "RGB buffer of {} bytes is not {width}x{height}x3",
            data.len()
        );
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Decode any format the `image` crate reads, converted to RGB.
    pub fn open(path: &Path) -> Result<Self> {
        let img = image::open(path)
            .with_context(|| format!("opening image {}", path.display()))?
            .to_rgb8();
        let (w, h) = img.dimensions();
        Self::new(w as usize, h as usize, img.into_raw())
    }
}

/// Every picture is bounded at this many pixels before the processor
/// (`runner.VISION_MAX_PIXELS`).
pub const VISION_MAX_PIXELS: usize = 1024 * 1024;

/// d1's `cap_pixels`: downscale (Pillow bicubic, bit-exact) to at most
/// `max_pixels`, keeping the aspect ratio; smaller pictures pass through.
pub fn cap_pixels(img: &RgbImage, max_pixels: usize) -> RgbImage {
    let (w, h) = (img.width, img.height);
    if w * h <= max_pixels {
        return img.clone();
    }
    let scale = (max_pixels as f64 / (w * h) as f64).sqrt();
    let nw = ((w as f64 * scale) as usize).max(1);
    let nh = ((h as f64 * scale) as usize).max(1);
    let data =
        rlx_core::image_preprocess::pil_resize_rgb8_exact(&img.data, w, h, nw, nh, Filter::Bicubic);
    RgbImage {
        width: nw,
        height: nh,
        data,
    }
}

/// `Lfm2VlImageProcessor` settings (`processor_config.json`).
#[derive(Debug, Clone)]
pub struct ProcessorConfig {
    pub downsample_factor: usize,
    pub do_image_splitting: bool,
    pub min_tiles: usize,
    pub max_tiles: usize,
    pub use_thumbnail: bool,
    pub min_image_tokens: usize,
    pub max_image_tokens: usize,
    pub encoder_patch_size: usize,
    pub tile_size: usize,
    pub max_pixels_tolerance: f64,
    pub use_image_special_tokens: bool,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            downsample_factor: 2,
            do_image_splitting: true,
            min_tiles: 2,
            max_tiles: 10,
            use_thumbnail: true,
            min_image_tokens: 64,
            max_image_tokens: 256,
            encoder_patch_size: 16,
            tile_size: 512,
            max_pixels_tolerance: 2.0,
            use_image_special_tokens: true,
        }
    }
}

impl ProcessorConfig {
    /// Read `processor_config.json` (its `image_processor` block) and
    /// `config.json`'s `use_image_special_tokens`; absent keys keep the
    /// defaults. Mean and std must be 0.5 and the resample bicubic — the
    /// values every LFM2-VL checkpoint ships — or this refuses.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let mut cfg = Self::default();
        let read = |name: &str| -> Result<Option<serde_json::Value>> {
            let p = dir.join(name);
            if !p.is_file() {
                return Ok(None);
            }
            let raw = std::fs::read_to_string(&p).with_context(|| format!("reading {name}"))?;
            Ok(Some(
                serde_json::from_str(&raw).with_context(|| format!("parsing {name}"))?,
            ))
        };
        if let Some(pc) = read("processor_config.json")? {
            let ip = pc.get("image_processor").unwrap_or(&pc);
            let u = |k: &str, d: usize| {
                ip.get(k)
                    .and_then(serde_json::Value::as_u64)
                    .map_or(d, |v| v as usize)
            };
            let b = |k: &str, d: bool| ip.get(k).and_then(serde_json::Value::as_bool).unwrap_or(d);
            cfg.downsample_factor = u("downsample_factor", cfg.downsample_factor);
            cfg.do_image_splitting = b("do_image_splitting", cfg.do_image_splitting);
            cfg.min_tiles = u("min_tiles", cfg.min_tiles);
            cfg.max_tiles = u("max_tiles", cfg.max_tiles);
            cfg.use_thumbnail = b("use_thumbnail", cfg.use_thumbnail);
            cfg.min_image_tokens = u("min_image_tokens", cfg.min_image_tokens);
            cfg.max_image_tokens = u("max_image_tokens", cfg.max_image_tokens);
            cfg.encoder_patch_size = u("encoder_patch_size", cfg.encoder_patch_size);
            cfg.tile_size = u("tile_size", cfg.tile_size);
            cfg.max_pixels_tolerance = ip
                .get("max_pixels_tolerance")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(cfg.max_pixels_tolerance);
            for key in ["image_mean", "image_std"] {
                if let Some(v) = ip.get(key).and_then(serde_json::Value::as_array) {
                    ensure!(
                        v.iter().all(|x| x.as_f64() == Some(0.5)),
                        "{key} {v:?} is not [0.5, 0.5, 0.5]; only SigLIP normalization is ported"
                    );
                }
            }
            if let Some(r) = ip.get("resample").and_then(serde_json::Value::as_u64) {
                ensure!(
                    r == 3,
                    "resample {r} is not bicubic (3); only bicubic is ported"
                );
            }
        }
        if let Some(c) = read("config.json")?
            && let Some(v) = c
                .get("use_image_special_tokens")
                .and_then(serde_json::Value::as_bool)
        {
            cfg.use_image_special_tokens = v;
        }
        Ok(cfg)
    }

    /// The tower's compiled sequence length: the larger of a thumbnail's
    /// patch budget and one tile's patch count.
    pub fn max_num_patches(&self) -> usize {
        let thumb = self.max_image_tokens * self.downsample_factor.pow(2);
        let tile = if self.do_image_splitting {
            (self.tile_size / self.encoder_patch_size).pow(2)
        } else {
            0
        };
        thumb.max(tile)
    }
}

/// Python `round(number / factor) * factor` — `round` ties to even.
fn round_by_factor(number: usize, factor: usize) -> usize {
    ((number as f64 / factor as f64).round_ties_even() as usize) * factor
}

/// CPython's order of `sorted(set(ratios), key=w*h)` for the shipped tile
/// range (2..=10). Ties in area keep the `set`'s iteration order, which a
/// near-tie in aspect ratio can turn into a different grid, so the order is
/// copied rather than re-derived.
const RATIOS_2_10: [(usize, usize); 26] = [
    (1, 2),
    (2, 1),
    (3, 1),
    (1, 3),
    (2, 2),
    (4, 1),
    (1, 4),
    (5, 1),
    (1, 5),
    (1, 6),
    (6, 1),
    (3, 2),
    (2, 3),
    (7, 1),
    (1, 7),
    (4, 2),
    (2, 4),
    (1, 8),
    (8, 1),
    (1, 9),
    (3, 3),
    (9, 1),
    (2, 5),
    (5, 2),
    (10, 1),
    (1, 10),
];

/// Candidate `(grid_w, grid_h)` layouts by area. Other ranges than the
/// shipped 2..=10 break area ties by `(w, h)`, which can differ from Python
/// on an exact aspect-ratio tie.
fn target_ratios(min_tiles: usize, max_tiles: usize) -> Vec<(usize, usize)> {
    if (min_tiles, max_tiles) == (2, 10) {
        return RATIOS_2_10.to_vec();
    }
    let mut set: Vec<(usize, usize)> = Vec::new();
    for n in min_tiles..=max_tiles {
        for w in 1..=n {
            for h in 1..=n {
                if (min_tiles..=max_tiles).contains(&(w * h)) && !set.contains(&(w, h)) {
                    set.push((w, h));
                }
            }
        }
    }
    set.sort_by_key(|&(w, h)| (w * h, w, h));
    set
}

/// `find_closest_aspect_ratio`: the layout nearest the image's aspect ratio;
/// on a tie, a larger layout wins if the image covers half its area.
fn closest_aspect_ratio(
    aspect: f64,
    ratios: &[(usize, usize)],
    width: usize,
    height: usize,
    tile: usize,
) -> (usize, usize) {
    let mut best_diff = f64::INFINITY;
    let mut best = (1, 1);
    let area = (width * height) as f64;
    for &(rw, rh) in ratios {
        let diff = (aspect - rw as f64 / rh as f64).abs();
        if diff < best_diff {
            best_diff = diff;
            best = (rw, rh);
        } else if diff == best_diff && area > 0.5 * (tile * tile * rw * rh) as f64 {
            best = (rw, rh);
        }
    }
    best
}

/// One tile ready for the tower: normalized patches and its patch grid.
#[derive(Debug, Clone)]
pub struct Tile {
    /// `[ph · pw, 3 · ps²]`, each row `(ry, rx, c)` within its patch.
    pub pixel_values: Vec<f32>,
    pub ph: usize,
    pub pw: usize,
}

/// An image as the processor leaves it.
#[derive(Debug, Clone)]
pub struct ProcessedImage {
    /// Grid tiles row-major, then the thumbnail; or the single tile.
    pub tiles: Vec<Tile>,
    pub rows: usize,
    pub cols: usize,
    /// The smart-resized size `(height, width)` (the thumbnail's, when tiled).
    pub image_size: (usize, usize),
}

impl ProcessedImage {
    /// The `<image>` replacement text, as `Lfm2VlProcessor` writes it.
    pub fn placeholder(&self, cfg: &ProcessorConfig) -> String {
        let ds = cfg.downsample_factor;
        let ps = cfg.encoder_patch_size;
        let per_tile = (cfg.tile_size / ps).div_ceil(ds).pow(2);
        let (h, w) = self.image_size;
        let for_image = (h / ps).div_ceil(ds) * (w / ps).div_ceil(ds);
        let special = cfg.use_image_special_tokens;
        let mut s = String::new();
        if special {
            s.push_str("<|image_start|>");
        }
        if self.rows > 1 || self.cols > 1 {
            for r in 0..self.rows {
                for c in 0..self.cols {
                    if special {
                        s.push_str(&format!("<|img_row_{}_col_{}|>", r + 1, c + 1));
                    }
                    s.push_str(&IMAGE_TOKEN.repeat(per_tile));
                }
            }
            if cfg.use_thumbnail {
                if special {
                    s.push_str("<|img_thumbnail|>");
                }
                s.push_str(&IMAGE_TOKEN.repeat(for_image));
            }
        } else {
            s.push_str(&IMAGE_TOKEN.repeat(for_image));
        }
        if special {
            s.push_str("<|image_end|>");
        }
        s
    }
}

/// The image placeholder token's text.
pub const IMAGE_TOKEN: &str = "<image>";

/// `rescale` + `normalize` fused as torchvision runs them:
/// `(x - 127.5) / 127.5` in f32 (mean = std = 0.5, scaled by 255).
fn patchify(rgb: &[u8], h: usize, w: usize, ps: usize) -> Tile {
    let (ph, pw) = (h / ps, w / ps);
    let pd = 3 * ps * ps;
    let mut pv = vec![0f32; ph * pw * pd];
    for py in 0..ph {
        for px in 0..pw {
            let base = (py * pw + px) * pd;
            for ry in 0..ps {
                for rx in 0..ps {
                    let src = ((py * ps + ry) * w + px * ps + rx) * 3;
                    for c in 0..3 {
                        pv[base + (ry * ps + rx) * 3 + c] = (rgb[src + c] as f32 - 127.5) / 127.5;
                    }
                }
            }
        }
    }
    Tile {
        pixel_values: pv,
        ph,
        pw,
    }
}

/// `Lfm2VlImageProcessor` for one (already capped) image.
pub fn process_image(img: &RgbImage, cfg: &ProcessorConfig) -> Result<ProcessedImage> {
    let (h, w) = (img.height, img.width);
    ensure!(h > 0 && w > 0, "empty image");
    let ps = cfg.encoder_patch_size;
    let ds = cfg.downsample_factor;
    let total = ps * ds;
    let (min_tiles, max_tiles) = if cfg.do_image_splitting {
        ensure!(
            cfg.min_tiles <= cfg.max_tiles,
            "min_tiles must be <= max_tiles"
        );
        (cfg.min_tiles, cfg.max_tiles)
    } else {
        (1, 1)
    };
    let splitting = !(min_tiles == 1 && max_tiles == 1);

    // `_is_image_too_large`.
    let h_bar = ps.max(round_by_factor(h, total));
    let w_bar = ps.max(round_by_factor(w, total));
    let too_large = (h_bar * w_bar) as f64
        > (cfg.max_image_tokens * ps * ps * ds * ds) as f64 * cfg.max_pixels_tolerance;

    // `smart_resize`.
    let min_px = cfg.min_image_tokens * ps * ps * ds * ds;
    let max_px = cfg.max_image_tokens * ps * ps * ds * ds;
    let mut nh = total.max(round_by_factor(h, total));
    let mut nw = total.max(round_by_factor(w, total));
    if nh * nw > max_px {
        let beta = ((h * w) as f64 / max_px as f64).sqrt();
        nh = total.max(((h as f64 / beta / total as f64).floor() as usize) * total);
        nw = total.max(((w as f64 / beta / total as f64).floor() as usize) * total);
    } else if nh * nw < min_px {
        let beta = (min_px as f64 / (h * w) as f64).sqrt();
        nh = ((h as f64 * beta / total as f64).ceil() as usize) * total;
        nw = ((w as f64 * beta / total as f64).ceil() as usize) * total;
    }

    let resize =
        |th: usize, tw: usize| torch_resize_rgb8_aa(&img.data, w, h, tw, th, Filter::Bicubic);
    let mut tiles = Vec::new();
    let (mut rows, mut cols) = (1, 1);
    if too_large && splitting {
        let tile = cfg.tile_size;
        let (gw, gh) = closest_aspect_ratio(
            w as f64 / h as f64,
            &target_ratios(min_tiles, max_tiles),
            w,
            h,
            tile,
        );
        let (th, tw) = (tile * gh, tile * gw);
        let big = resize(th, tw);
        for ty in 0..gh {
            for tx in 0..gw {
                let mut buf = Vec::with_capacity(tile * tile * 3);
                for y in 0..tile {
                    let row = ((ty * tile + y) * tw + tx * tile) * 3;
                    buf.extend_from_slice(&big[row..row + tile * 3]);
                }
                tiles.push(patchify(&buf, tile, tile, ps));
            }
        }
        if cfg.use_thumbnail && gw * gh != 1 {
            tiles.push(patchify(&resize(nh, nw), nh, nw, ps));
        }
        rows = gh;
        cols = gw;
    } else {
        tiles.push(patchify(&resize(nh, nw), nh, nw, ps));
    }
    Ok(ProcessedImage {
        tiles,
        rows,
        cols,
        image_size: (nh, nw),
    })
}

/// LFM2-VL's `pixel_unshuffle` over a `[ph, pw, c]` feature map: each 2x2
/// block becomes one `4c` row, `out[i, j, (di·f + dj)·c + k] =
/// x[f·i + di, f·j + dj, k]`, rows in `(i, j)` order.
pub fn pixel_unshuffle(x: &[f32], ph: usize, pw: usize, c: usize, f: usize) -> Vec<f32> {
    let (oh, ow) = (ph / f, pw / f);
    let mut out = vec![0f32; oh * ow * f * f * c];
    for i in 0..oh {
        for j in 0..ow {
            let dst = (i * ow + j) * f * f * c;
            for di in 0..f {
                for dj in 0..f {
                    let src = ((f * i + di) * pw + f * j + dj) * c;
                    let o = dst + (di * f + dj) * c;
                    out[o..o + c].copy_from_slice(&x[src..src + c]);
                }
            }
        }
    }
    out
}
