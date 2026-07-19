// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Patch-augmentation consistency (GLARE §4.3, L_loc1): strong-blur a fraction
//! of patches on the student view. The two views are spatially **aligned**
//! (same crop, photometric distortion only) so patch/region correspondence is
//! by position — a documented simplification of the paper's crop back-tracking.

use crate::dino::Rng;

/// Apply a strong box blur to a random `frac` of the patches of an NCHW image
/// (`3·img·img`), leaving the rest untouched. Returns the distorted copy.
pub fn strong_blur_patches(
    nchw: &[f32],
    img: usize,
    patch: usize,
    frac: f32,
    rng: &mut Rng,
) -> Vec<f32> {
    let mut out = nchw.to_vec();
    let n_side = img / patch;
    let radius = 2i64; // strong 5×5 box blur
    for py in 0..n_side {
        for px in 0..n_side {
            if rng.f32() >= frac {
                continue;
            }
            for c in 0..3 {
                let base = c * img * img;
                for ry in 0..patch {
                    let y = py * patch + ry;
                    for rx in 0..patch {
                        let x = px * patch + rx;
                        let mut acc = 0.0f32;
                        let mut cnt = 0.0f32;
                        for dy in -radius..=radius {
                            for dx in -radius..=radius {
                                let yy = (y as i64 + dy).clamp(0, img as i64 - 1) as usize;
                                let xx = (x as i64 + dx).clamp(0, img as i64 - 1) as usize;
                                acc += nchw[base + yy * img + xx];
                                cnt += 1.0;
                            }
                        }
                        out[base + y * img + x] = acc / cnt;
                    }
                }
            }
        }
    }
    out
}
