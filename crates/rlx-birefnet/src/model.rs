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

//! Running BiRefNet: ImageNet-normalized square input, sigmoid of the logits.

use anyhow::{Context, Result, ensure};
use image::{RgbImage, imageops::FilterType};
use rlx_runtime::{CompiledGraph, Device};
use std::path::Path;
use std::sync::Arc;

use crate::graph::build;
use crate::weights::Weights;

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// A compiled BiRefNet at one input size.
pub struct BiRefNet {
    graph: CompiledGraph,
    pub side: usize,
}

impl BiRefNet {
    /// Load `dir` (holding `model.safetensors`) and compile for `side × side`
    /// input (a multiple of 32; the checkpoints are trained at 1024).
    pub fn open(dir: &Path, side: usize, device: Device) -> Result<Self> {
        let w = Arc::new(Weights::load(dir)?);
        let built = build(w, side, &["logits"])?;
        Ok(Self {
            graph: rlx_core::flow_util::compile_built(built, device)?,
            side,
        })
    }

    /// Network input `[3, side, side]`.
    pub fn preprocess(&self, img: &RgbImage) -> Vec<f32> {
        let s = self.side as u32;
        let r = image::imageops::resize(img, s, s, FilterType::Triangle);
        let n = (s * s) as usize;
        let mut out = vec![0.0f32; 3 * n];
        for (i, p) in r.pixels().enumerate() {
            for c in 0..3 {
                out[c * n + i] = (p.0[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
        out
    }

    /// Raw logits `[side²]` for a preprocessed input.
    pub fn logits(&mut self, x: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            x.len() == 3 * self.side * self.side,
            "input length {}",
            x.len()
        );
        self.graph
            .run(&[("image", x)])
            .into_iter()
            .next()
            .context("no output")
    }

    /// Foreground alpha (0..1) of `img`, resampled to its size.
    pub fn alpha(&mut self, img: &RgbImage) -> Result<Vec<f32>> {
        let x = self.preprocess(img);
        let l = self.logits(&x)?;
        let s = self.side as u32;
        let a: Vec<f32> = l.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
        let g = image::ImageBuffer::<image::Luma<f32>, Vec<f32>>::from_raw(s, s, a).expect("side²");
        Ok(
            image::imageops::resize(&g, img.width(), img.height(), FilterType::Triangle)
                .into_raw()
                .into_iter()
                .map(|v| v.clamp(0.0, 1.0))
                .collect(),
        )
    }
}
