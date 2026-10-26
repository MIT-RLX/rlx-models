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

//! Running the vendor's subject-lifting network: preprocessing, the compiled graph,
//! and the two heads.

use anyhow::{Context, Result, ensure};
use image::{GrayImage, RgbImage, imageops::FilterType};
use rlx_neuralhash::netfile::NetFile;
use rlx_runtime::{CompiledGraph, Device};
use std::path::{Path, PathBuf};

use crate::graph::{Upsample, build};

/// Where macOS ships the model (Vision.framework). It is read in place, never
/// copied: this crate carries no vendor weights.
pub const SYSTEM_MODEL: &str = "/System/Library/Frameworks/Vision.framework/Versions/A/Resources/subject_lifting_gen1_rev5_gv8dsz6vxu_multihead_int8.espresso.net";

/// Pixel encoding fed to `input_image`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputNorm {
    /// `v / 255`.
    Unit,
    /// `v` (0..255).
    Byte,
    /// `v / 127.5 - 1`.
    Signed,
    /// ImageNet mean/std on `v / 255`.
    ImageNet,
}

/// Settings fixed by parity against the Vision framework (`examples/parity.rs`):
/// over nine images (emoji on white, dark and saturated backgrounds, a photo),
/// `[-1, 1]` RGB with bilinear upsampling matches Vision's own masks at
/// correlation 0.998 / mean |Δα| 0.0038; `[0, 1]` gives 0.994, BGR 0.988,
/// ImageNet 0.976, nearest upsampling 0.90, raw 0..255 ≈ 0. Those figures
/// predate [`FEATHER`], which accounts for most of the remaining residual.
#[derive(Debug, Clone, Copy)]
pub struct LiftOptions {
    pub norm: InputNorm,
    pub bgr: bool,
    pub upsample: Upsample,
    /// Gaussian σ, in network pixels, applied to the saliency before it is
    /// resized to the image (see [`FEATHER`]).
    pub feather: f32,
}

/// The edge softening Vision applies when it scales the mask to the image.
/// Fitted against `generateScaledMaskForImage` on super-resolved emoji at
/// 640 and 1280 px: our bilinear masks had 2.5× fewer partial-alpha pixels
/// than Vision's, all of the difference sat in the edge band with zero mean,
/// and the width scaled with the image, so it is defined at the network's
/// resolution. σ = 0.75 cuts mean |Δα| 0.0027 → 0.0003; an edge-aware
/// (colour guided filter) upsample went the other way, to 0.0035–0.008.
pub const FEATHER: f32 = 0.75;

impl Default for LiftOptions {
    fn default() -> Self {
        Self {
            norm: InputNorm::Signed,
            bgr: false,
            upsample: Upsample::Bilinear,
            feather: FEATHER,
        }
    }
}

/// Subject mask for one image.
#[derive(Debug, Clone)]
pub struct Lifted {
    /// Soft foreground probability at the network's resolution (`side²`).
    pub saliency: Vec<f32>,
    pub side: usize,
    /// "Is there a liftable subject" confidence (0..1).
    pub gating: f32,
    /// [`LiftOptions::feather`] it was produced with.
    pub feather: f32,
}

/// Separable Gaussian blur of a `side²` plane, borders clamped.
fn gaussian(p: &[f32], side: usize, sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 {
        return p.to_vec();
    }
    let r = (3.0 * sigma).ceil() as isize;
    let k: Vec<f32> = (-r..=r)
        .map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let norm: f32 = k.iter().sum();
    let tap = |i: isize| i.clamp(0, side as isize - 1) as usize;
    let mut tmp = vec![0f32; p.len()];
    for y in 0..side {
        for x in 0..side {
            tmp[y * side + x] = (-r..=r)
                .map(|d| k[(d + r) as usize] * p[y * side + tap(x as isize + d)])
                .sum::<f32>()
                / norm;
        }
    }
    let mut out = vec![0f32; p.len()];
    for y in 0..side {
        for x in 0..side {
            out[y * side + x] = (-r..=r)
                .map(|d| k[(d + r) as usize] * tmp[tap(y as isize + d) * side + x])
                .sum::<f32>()
                / norm;
        }
    }
    out
}

impl Lifted {
    /// The saliency with the [`feather`](Self::feather) applied.
    pub fn feathered(&self) -> Vec<f32> {
        gaussian(&self.saliency, self.side, self.feather)
    }

    /// Saliency resampled to `w × h` as an 8-bit mask.
    pub fn mask(&self, w: u32, h: u32) -> GrayImage {
        let s = self.side as u32;
        let g = image::ImageBuffer::<image::Luma<f32>, Vec<f32>>::from_raw(s, s, self.feathered())
            .expect("side²");
        let r = image::imageops::resize(&g, w, h, FilterType::Triangle);
        GrayImage::from_fn(w, h, |x, y| {
            image::Luma([(r.get_pixel(x, y).0[0].clamp(0.0, 1.0) * 255.0).round() as u8])
        })
    }

    /// Saliency resampled to `w × h`, 0..1.
    pub fn alpha(&self, w: u32, h: u32) -> Vec<f32> {
        let s = self.side as u32;
        let g = image::ImageBuffer::<image::Luma<f32>, Vec<f32>>::from_raw(s, s, self.feathered())
            .expect("side²");
        image::imageops::resize(&g, w, h, FilterType::Triangle)
            .into_raw()
            .into_iter()
            .map(|v| v.clamp(0.0, 1.0))
            .collect()
    }
}

/// A compiled subject-lifting network.
pub struct SubjectLift {
    graph: CompiledGraph,
    side: usize,
    opts: LiftOptions,
}

impl SubjectLift {
    /// Load from `path` (default [`SYSTEM_MODEL`]).
    pub fn open(path: Option<&Path>, device: Device, opts: LiftOptions) -> Result<Self> {
        let path: PathBuf = path
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(SYSTEM_MODEL));
        ensure!(
            path.is_file(),
            "{} not found — the subject-lifting model ships with macOS 14+ (Vision.framework)",
            path.display()
        );
        let net = NetFile::open(&path).with_context(|| format!("reading {}", path.display()))?;
        let side = net.shapes.get("input_image").map(|s| s.h).unwrap_or(512);
        let built = build(
            &net,
            "input_image",
            side,
            &["saliency", "gating_confidence"],
            opts.upsample,
        )?;
        let graph = rlx_core::flow_util::compile_built(built, device)?;
        Ok(Self { graph, side, opts })
    }

    pub fn side(&self) -> usize {
        self.side
    }

    /// Network input `[3, side, side]` for `img` (stretched to the square, as
    /// the input is square and emoji are too).
    pub fn preprocess(&self, img: &RgbImage) -> Vec<f32> {
        let s = self.side as u32;
        let r = image::imageops::resize(img, s, s, FilterType::Triangle);
        let n = (s * s) as usize;
        let mut out = vec![0.0f32; 3 * n];
        const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
        const STD: [f32; 3] = [0.229, 0.224, 0.225];
        for (i, p) in r.pixels().enumerate() {
            for c in 0..3 {
                let src = if self.opts.bgr { 2 - c } else { c };
                let v = p.0[src] as f32;
                out[c * n + i] = match self.opts.norm {
                    InputNorm::Unit => v / 255.0,
                    InputNorm::Byte => v,
                    InputNorm::Signed => v / 127.5 - 1.0,
                    InputNorm::ImageNet => (v / 255.0 - MEAN[c]) / STD[c],
                };
            }
        }
        out
    }

    /// Run on an image.
    pub fn run(&mut self, img: &RgbImage) -> Result<Lifted> {
        let x = self.preprocess(img);
        let out = self
            .graph
            .run(&[("input_image", &x)])
            .into_iter()
            .next()
            .context("no output")?;
        let n = self.side * self.side;
        ensure!(out.len() == n + 1, "unexpected output length {}", out.len());
        Ok(Lifted {
            saliency: out[..n].to_vec(),
            side: self.side,
            gating: out[n],
            feather: self.opts.feather,
        })
    }
}
