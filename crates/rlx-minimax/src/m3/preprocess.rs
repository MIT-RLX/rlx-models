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

//! MiniMax-M3 image preprocessing: RGB → pre-patchified `pixel_values` + grid.
//!
//! Produces the [`ImageInput`] the vision tower
//! consumes: an `[num_patches, C·temporal·patch²]` matrix whose row layout
//! matches the flattened Conv3d patch-embed weight (`[embed, C, T, P, P]`
//! flattened to `[embed, C·T·P·P]`), plus the `(t, h, w)` patch grid. A single
//! image is `grid_t = 1` with the frame replicated across the `temporal_patch_size`.
//!
//! Bilinear resize to a patch multiple + CLIP mean/std normalization. Exact HF
//! parity (smart-resize rounding / interpolation) is a refinement — the real
//! 428B checkpoint can't be validated locally regardless.

use anyhow::{Result, anyhow};

use super::config::M3VisionConfig;
use super::vl_runner::ImageInput;

/// CLIP/SigLIP normalization constants (OpenAI CLIP defaults).
pub const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
pub const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// Image preprocessor bound to a vision config.
#[derive(Debug, Clone)]
pub struct M3ImagePreprocessor {
    /// Spatial patch size (`patch × patch`).
    pub patch_size: usize,
    /// Temporal patch depth (the frame is replicated across it).
    pub temporal_patch_size: usize,
    /// Input channels (clamped to 3).
    pub num_channels: usize,
    /// Per-channel normalization mean.
    pub mean: [f32; 3],
    /// Per-channel normalization std.
    pub std: [f32; 3],
    /// Upper bound on `grid_h · grid_w` (the resize shrinks to fit).
    pub max_patches: usize,
}

impl M3ImagePreprocessor {
    /// Build a preprocessor from a vision config, capping the patch grid at
    /// `max_patches` and using the CLIP mean/std.
    pub fn from_vision_config(cfg: &M3VisionConfig, max_patches: usize) -> Self {
        Self {
            patch_size: cfg.patch_size,
            temporal_patch_size: cfg.temporal_patch_size,
            num_channels: cfg.num_channels.min(3),
            mean: CLIP_MEAN,
            std: CLIP_STD,
            max_patches: max_patches.max(1),
        }
    }

    /// Choose the `(grid_h, grid_w)` patch grid for a `w×h` image: at least one
    /// patch per axis, shrunk proportionally so `gh·gw ≤ max_patches`.
    pub fn grid_for(&self, w: usize, h: usize) -> (usize, usize) {
        let p = self.patch_size;
        let mut gh = ((h + p / 2) / p).max(1);
        let mut gw = ((w + p / 2) / p).max(1);
        while gh * gw > self.max_patches && (gh > 1 || gw > 1) {
            if gh >= gw && gh > 1 {
                gh -= 1;
            } else if gw > 1 {
                gw -= 1;
            } else {
                break;
            }
        }
        (gh, gw)
    }

    /// Preprocess an interleaved RGB `u8` buffer (`h·w·3`, HWC, row-major) into
    /// an [`ImageInput`].
    pub fn preprocess_rgb_u8(&self, rgb: &[u8], w: usize, h: usize) -> Result<ImageInput> {
        if rgb.len() != w * h * 3 {
            return Err(anyhow!("rgb len {} != w·h·3 {}", rgb.len(), w * h * 3));
        }
        let (gh, gw) = self.grid_for(w, h);
        let p = self.patch_size;
        let (oh, ow) = (gh * p, gw * p);
        let c = self.num_channels;
        let t = self.temporal_patch_size;

        // Bilinear resize + normalize into a planar [c, oh, ow] buffer.
        let mut norm = vec![0f32; c * oh * ow];
        for ch in 0..c {
            for oy in 0..oh {
                // Map output pixel centers back to input space.
                let sy = ((oy as f32 + 0.5) * h as f32 / oh as f32) - 0.5;
                let y0 = sy.floor().clamp(0.0, (h - 1) as f32);
                let y1 = (y0 + 1.0).min((h - 1) as f32);
                let wy = (sy - y0).clamp(0.0, 1.0);
                for ox in 0..ow {
                    let sx = ((ox as f32 + 0.5) * w as f32 / ow as f32) - 0.5;
                    let x0 = sx.floor().clamp(0.0, (w - 1) as f32);
                    let x1 = (x0 + 1.0).min((w - 1) as f32);
                    let wx = (sx - x0).clamp(0.0, 1.0);
                    let (y0, y1, x0, x1) = (y0 as usize, y1 as usize, x0 as usize, x1 as usize);
                    let at = |yy: usize, xx: usize| rgb[(yy * w + xx) * 3 + ch] as f32 / 255.0;
                    let top = at(y0, x0) * (1.0 - wx) + at(y0, x1) * wx;
                    let bot = at(y1, x0) * (1.0 - wx) + at(y1, x1) * wx;
                    let v = top * (1.0 - wy) + bot * wy;
                    norm[(ch * oh + oy) * ow + ox] = (v - self.mean[ch]) / self.std[ch];
                }
            }
        }

        // Patchify → [num_patches, C·T·P·P] with the frame replicated across T.
        let np = gh * gw;
        let patch_dim = c * t * p * p;
        let mut pixels = vec![0f32; np * patch_dim];
        for gy in 0..gh {
            for gx in 0..gw {
                let pidx = gy * gw + gx;
                let base = pidx * patch_dim;
                let mut o = 0usize;
                for ch in 0..c {
                    for _tt in 0..t {
                        for py in 0..p {
                            for px in 0..p {
                                let iy = gy * p + py;
                                let ix = gx * p + px;
                                pixels[base + o] = norm[(ch * oh + iy) * ow + ix];
                                o += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(ImageInput {
            pixel_values: pixels,
            grid_t: 1,
            grid_h: gh,
            grid_w: gw,
        })
    }
}
