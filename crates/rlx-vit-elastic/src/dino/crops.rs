// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! DINO-style multi-crop augmentation (host-side).
//!
//! Following DINO we take `n_global` large-scale views and `n_local`
//! small-scale views of an image and enforce cross-view consistency.
//!
//! **Deviation from the paper (documented):** every crop is resized to the
//! model's native `img_size` rather than DINO's 96² locals. This keeps a
//! **single** graph shape (one compiled runner, no position-embedding
//! interpolation) while preserving the multi-scale/region cross-view
//! objective — a global crop is a large region, a local crop a small region,
//! both rendered at `img_size`. BYOL augmentations here are random-resized
//! crop + horizontal flip + gaussian blur (color jitter/solarize omitted).

use super::super::vit::config::{IMAGENET_MEAN, IMAGENET_STD};

/// Small deterministic PRNG (SplitMix64) — reproducible crops/augs/noise.
#[derive(Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E3779B97F4A7C15,
        }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, 1)`.
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Uniform in `[a, b)`.
    pub fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f32()
    }
    /// Standard normal (Box–Muller).
    pub fn gauss(&mut self) -> f32 {
        let u1 = (self.f32()).max(1e-7);
        let u2 = self.f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Multi-crop parameters.
#[derive(Debug, Clone)]
pub struct CropConfig {
    pub n_global: usize,
    pub n_local: usize,
    pub global_scale: (f32, f32),
    pub local_scale: (f32, f32),
    pub img_size: usize,
    pub flip_prob: f32,
    pub blur_prob: f32,
}

impl Default for CropConfig {
    fn default() -> Self {
        // DINO defaults: 2 global + N local views (all rendered at img_size).
        Self {
            n_global: 2,
            n_local: 6,
            global_scale: (0.4, 1.0),
            local_scale: (0.05, 0.4),
            img_size: 224,
            flip_prob: 0.5,
            blur_prob: 0.5,
        }
    }
}

impl CropConfig {
    pub fn n_crops(&self) -> usize {
        self.n_global + self.n_local
    }
}

/// A crop box in source-image pixel coordinates.
struct Box {
    x0: f32,
    y0: f32,
    bw: f32,
    bh: f32,
}

fn sample_box(rng: &mut Rng, w: usize, h: usize, scale: (f32, f32)) -> Box {
    let area = (w * h) as f32;
    // A few attempts to find an in-bounds box with a valid aspect ratio.
    for _ in 0..10 {
        let target = area * rng.range(scale.0, scale.1);
        let log_ratio = rng.range((3.0f32 / 4.0).ln(), (4.0f32 / 3.0).ln());
        let ar = log_ratio.exp();
        let bw = (target * ar).sqrt();
        let bh = (target / ar).sqrt();
        if bw <= w as f32 && bh <= h as f32 && bw >= 1.0 && bh >= 1.0 {
            let x0 = rng.range(0.0, w as f32 - bw);
            let y0 = rng.range(0.0, h as f32 - bh);
            return Box { x0, y0, bw, bh };
        }
    }
    // Fallback: center crop of the whole image.
    Box {
        x0: 0.0,
        y0: 0.0,
        bw: w as f32,
        bh: h as f32,
    }
}

/// Bilinear-resize the source box to `img×img`, ImageNet-normalize → NCHW f32.
fn render_box(rgb: &[u8], w: usize, h: usize, b: &Box, img: usize) -> Vec<f32> {
    let mut out = vec![0f32; 3 * img * img];
    let sx = if img > 1 {
        b.bw / (img as f32 - 1.0).max(1.0)
    } else {
        0.0
    };
    let sy = if img > 1 {
        b.bh / (img as f32 - 1.0).max(1.0)
    } else {
        0.0
    };
    for y in 0..img {
        let fy = (b.y0 + y as f32 * sy).clamp(0.0, h as f32 - 1.0);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(h - 1);
        let dy = fy - y0 as f32;
        for x in 0..img {
            let fx = (b.x0 + x as f32 * sx).clamp(0.0, w as f32 - 1.0);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);
            let dx = fx - x0 as f32;
            for c in 0..3 {
                let p00 = rgb[(y0 * w + x0) * 3 + c] as f32;
                let p01 = rgb[(y0 * w + x1) * 3 + c] as f32;
                let p10 = rgb[(y1 * w + x0) * 3 + c] as f32;
                let p11 = rgb[(y1 * w + x1) * 3 + c] as f32;
                let top = p00 * (1.0 - dx) + p01 * dx;
                let bot = p10 * (1.0 - dx) + p11 * dx;
                let v = (top * (1.0 - dy) + bot * dy) / 255.0;
                out[c * img * img + y * img + x] = (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
        }
    }
    out
}

/// Horizontal flip in place (NCHW).
fn hflip(nchw: &mut [f32], img: usize) {
    for c in 0..3 {
        for y in 0..img {
            let row = c * img * img + y * img;
            for x in 0..img / 2 {
                nchw.swap(row + x, row + img - 1 - x);
            }
        }
    }
}

/// Separable 3×3 gaussian blur (NCHW), sigma≈1.
fn blur3(nchw: &[f32], img: usize) -> Vec<f32> {
    let k = [0.25f32, 0.5, 0.25];
    let mut tmp = vec![0f32; nchw.len()];
    // Horizontal.
    for c in 0..3 {
        for y in 0..img {
            for x in 0..img {
                let base = c * img * img + y * img;
                let mut acc = 0.0;
                for (i, &kw) in k.iter().enumerate() {
                    let xx = (x as isize + i as isize - 1).clamp(0, img as isize - 1) as usize;
                    acc += kw * nchw[base + xx];
                }
                tmp[base + x] = acc;
            }
        }
    }
    // Vertical.
    let mut out = vec![0f32; nchw.len()];
    for c in 0..3 {
        for y in 0..img {
            for x in 0..img {
                let base = c * img * img;
                let mut acc = 0.0;
                for (i, &kw) in k.iter().enumerate() {
                    let yy = (y as isize + i as isize - 1).clamp(0, img as isize - 1) as usize;
                    acc += kw * tmp[base + yy * img + x];
                }
                out[base + y * img + x] = acc;
            }
        }
    }
    out
}

/// Produce `cfg.n_crops()` augmented views of `rgb` (HWC u8), each a
/// normalized NCHW f32 tensor of length `3·img·img`. Globals come first.
pub fn multi_crop(
    rng: &mut Rng,
    rgb: &[u8],
    h_in: usize,
    w_in: usize,
    cfg: &CropConfig,
) -> Vec<Vec<f32>> {
    let img = cfg.img_size;
    let mut crops = Vec::with_capacity(cfg.n_crops());
    for i in 0..cfg.n_crops() {
        let scale = if i < cfg.n_global {
            cfg.global_scale
        } else {
            cfg.local_scale
        };
        let b = sample_box(rng, w_in, h_in, scale);
        let mut view = render_box(rgb, w_in, h_in, &b, img);
        if rng.f32() < cfg.flip_prob {
            hflip(&mut view, img);
        }
        if rng.f32() < cfg.blur_prob {
            view = blur3(&view, img);
        }
        crops.push(view);
    }
    crops
}

/// Stack per-crop NCHW tensors into a single `[n_crops·3·img·img]` batch (the
/// ViT runner's `batch = n_crops` input).
pub fn stack_crops(crops: &[Vec<f32>]) -> Vec<f32> {
    let mut out = Vec::with_capacity(crops.iter().map(|c| c.len()).sum());
    for c in crops {
        out.extend_from_slice(c);
    }
    out
}
