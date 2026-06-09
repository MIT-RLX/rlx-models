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

use crate::capabilities::validate_device;
use crate::config::DetectionParams;
use crate::detection::postprocess::word_rects_from_mask;
use crate::model::{DetectionGraphConfig, build_detection_graph};
use crate::preprocess::BLACK_VALUE;
use crate::weights::{
    HF_DETECTION_ST, HF_DETECTION_ST_FULL, SafetensorsFile, prefer_safetensors_path,
};
use anyhow::{Result, anyhow};
use rlx_core::flow_bridge::compile_options_for_profile;
use rlx_core::flow_util::attach_built_params;
use rlx_flow::CompileProfile;
use rlx_runtime::{CompiledGraph, Device, Session};
#[cfg(feature = "tensor-ops")]
use rten::{FloatOperators, Operators};
use rten_imageproc::RotatedRect;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};
use std::path::Path;
use std::sync::Mutex;

/// Text detector using a compiled native RLX U-Net graph.
pub struct RlxTextDetector {
    compiled: Mutex<CompiledGraph>,
    params: DetectionParams,
    input_hw: (usize, usize),
    #[allow(dead_code)]
    device: Device,
}

impl RlxTextDetector {
    pub fn from_path(
        path: impl AsRef<Path>,
        params: DetectionParams,
        device: Device,
    ) -> Result<Self> {
        Self::from_safetensors(path.as_ref(), params, device)
    }

    pub fn from_safetensors(path: &Path, params: DetectionParams, device: Device) -> Result<Self> {
        Self::from_safetensors_sized(path, params, DetectionGraphConfig::default(), device)
    }

    pub fn from_safetensors_sized(
        path: &Path,
        params: DetectionParams,
        cfg: DetectionGraphConfig,
        device: Device,
    ) -> Result<Self> {
        validate_device(device)?;
        let mut wm = SafetensorsFile::open(path)?.weight_map()?;
        let input_hw = (cfg.height, cfg.width);
        let (graph, param_map) = build_detection_graph(&mut wm, cfg)?;
        let opts = compile_options_for_profile(&CompileProfile::encoder(), device);
        let mut compiled = Session::new(device).compile_with(graph, &opts);
        attach_built_params(&mut compiled, param_map, &[]);
        Ok(Self {
            compiled: Mutex::new(compiled),
            params,
            input_hw,
            device,
        })
    }

    pub fn from_model_dir(dir: &Path, params: DetectionParams, device: Device) -> Result<Self> {
        let path =
            prefer_safetensors_path(dir, crate::weights::HF_DETECTION_ST, HF_DETECTION_ST_FULL);
        if !path.is_file() {
            let _fallback = dir.join(HF_DETECTION_ST);
            anyhow::bail!(
                "missing detection safetensors in {dir:?} (need ocr-detection-full.safetensors); \
                 run `rlx-ocr-convert` on {:?}",
                dir.join("text-detection-ssfbcj81.rten")
            );
        }
        Self::from_safetensors(&path, params, device)
    }

    pub fn fixed_input_hw(&self) -> Option<(usize, usize)> {
        Some(self.input_hw)
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
        let (in_height, in_width) = self.input_hw;

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

        let mut compiled = self.compiled.lock().map_err(|_| anyhow!("lock poisoned"))?;
        let input: Vec<f32> = image.iter().copied().collect();
        let outputs = compiled.run(&[("image", input.as_slice())]);
        let flat = outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("detection returned no output"))?;

        let valid_h = in_height - pad_bottom as usize;
        let valid_w = in_width - pad_right as usize;
        let mask = NdTensor::from_data([1, 1, in_height, in_width], flat);
        // Keep NCHW rank: `slice((0, 0, ..))` squeezes batch/channel and breaks `resize_image`.
        let mask = mask
            .slice((.., .., ..valid_h, ..valid_w))
            .resize_image([img_height, img_width])?;
        Ok(mask.into_shape([img_height, img_width]))
    }
}
