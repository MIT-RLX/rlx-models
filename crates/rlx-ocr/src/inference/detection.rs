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

//! Text detection via ocrs-compatible RTen segmentation models.

use crate::config::DetectionParams;
use crate::detection::postprocess::word_rects_from_mask;
use crate::inference::load_rten_model;
use crate::preprocess::BLACK_VALUE;
use anyhow::{Result, anyhow};
use rten::{Dimension, FloatOperators, Operators, RunOptions};
use rten_imageproc::RotatedRect;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, Tensor};
use std::path::Path;

/// Word detector backed by an ocrs `.rten` segmentation model.
pub struct RtenTextDetector {
    model: rten::Model,
    params: DetectionParams,
    input_shape: Vec<Dimension>,
}

impl RtenTextDetector {
    pub fn from_path(path: impl AsRef<Path>, params: DetectionParams) -> Result<Self> {
        let model = load_rten_model(path.as_ref())?;
        let input_shape = model_input_shape(&model)?;
        Ok(Self {
            model,
            params,
            input_shape,
        })
    }

    pub fn threshold(&self) -> f32 {
        self.params.text_threshold
    }

    /// Fixed `(height, width)` the RTen model expects after padding (if known).
    pub fn fixed_input_hw(&self) -> Option<(usize, usize)> {
        let [_, _, Dimension::Fixed(h), Dimension::Fixed(w)] = self.input_shape[..] else {
            return None;
        };
        Some((h, w))
    }

    pub fn detect_words(&self, image: NdTensorView<f32, 3>) -> Result<Vec<RotatedRect>> {
        let mask = self.detect_text_pixels(image)?;
        Ok(word_rects_from_mask(
            mask.view(),
            self.params.text_threshold,
            self.params.min_area,
        ))
    }

    pub fn detect_text_pixels(&self, image: NdTensorView<f32, 3>) -> Result<NdTensor<f32, 2>> {
        let [img_chans, img_height, img_width] = image.shape();
        let image = image.reshaped([1, img_chans, img_height, img_width]);

        let [
            _,
            _,
            Dimension::Fixed(in_height),
            Dimension::Fixed(in_width),
        ] = self.input_shape[..]
        else {
            return Err(anyhow!("detection model has dynamic input shape"));
        };

        let pad_bottom = (in_height as i32 - img_height as i32).max(0);
        let pad_right = (in_width as i32 - img_width as i32).max(0);
        let image = (pad_bottom > 0 || pad_right > 0)
            .then(|| {
                let pads = &[0, 0, 0, 0, 0, 0, pad_bottom, pad_right];
                image.pad(pads.into(), BLACK_VALUE)
            })
            .transpose()?
            .map(|t| t.into_cow())
            .unwrap_or(image.as_dyn().as_cow());

        let image = (image.size(2) != in_height || image.size(3) != in_width)
            .then(|| image.resize_image([in_height, in_width]))
            .transpose()?
            .map(|t| t.into_cow())
            .unwrap_or(image);

        let mut opts = RunOptions::default();
        opts.timing = false;
        let text_mask: Tensor<f32> = self
            .model
            .run_one(image.view().into(), Some(opts))?
            .try_into()
            .map_err(|_| anyhow!("detection model output was not f32"))?;

        let text_mask = text_mask
            .slice((
                ..,
                ..,
                ..(in_height - pad_bottom as usize),
                ..(in_width - pad_right as usize),
            ))
            .resize_image([img_height, img_width])?;

        Ok(text_mask.into_shape([img_height, img_width]))
    }
}

fn model_input_shape(model: &rten::Model) -> Result<Vec<Dimension>> {
    let input_id = model
        .input_ids()
        .first()
        .copied()
        .ok_or_else(|| anyhow!("detection model has no inputs"))?;
    model
        .node_info(input_id)
        .and_then(|info| info.shape())
        .ok_or_else(|| anyhow!("detection model does not specify input shape"))
}
