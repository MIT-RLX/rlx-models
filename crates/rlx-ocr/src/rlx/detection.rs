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
use crate::host_resize::{pad_hw_end, resize_bilinear};
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
use rten_imageproc::RotatedRect;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

/// Network stride: the U-Net has 6 pooling levels. Round adaptive input dims up
/// to this so the encoder/decoder skip sizes stay well-behaved across images.
const DET_STRIDE: usize = 64;
/// Cap the longest detection-input side; larger images are downscaled
/// (preserving aspect) to bound compute and memory.
const DET_MAX_SIDE: usize = 1280;

/// Text detector over native RLX U-Net graphs, compiled lazily per input size
/// (cached). Each image runs at an aspect-matched resolution instead of being
/// squished into one fixed shape — a wide line stays one line instead of
/// fragmenting. Pin a fixed size with `OCR_DETECTION_HW=H,W` for debugging.
pub struct RlxTextDetector {
    weights: SafetensorsFile,
    params: DetectionParams,
    device: Device,
    cache: Mutex<HashMap<(usize, usize), CompiledGraph>>,
    /// `Some` pins a fixed input size (`OCR_DETECTION_HW` or an explicit
    /// `DetectionGraphConfig`); `None` enables per-image aspect-preserving sizing.
    forced_hw: Option<(usize, usize)>,
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
        // Default entry: per-image aspect-preserving sizing, unless
        // `OCR_DETECTION_HW=H,W` pins a fixed input size.
        let forced = std::env::var("OCR_DETECTION_HW").ok().and_then(|s| {
            let (h, w) = s.split_once(',')?;
            Some((h.trim().parse().ok()?, w.trim().parse().ok()?))
        });
        Self::new(path, params, device, forced)
    }

    /// Pin the detector to the explicit config's `(height, width)`.
    pub fn from_safetensors_sized(
        path: &Path,
        params: DetectionParams,
        cfg: DetectionGraphConfig,
        device: Device,
    ) -> Result<Self> {
        Self::new(path, params, device, Some((cfg.height, cfg.width)))
    }

    fn new(
        path: &Path,
        params: DetectionParams,
        device: Device,
        forced_hw: Option<(usize, usize)>,
    ) -> Result<Self> {
        validate_device(device)?;
        Ok(Self {
            weights: SafetensorsFile::open(path)?,
            params,
            device,
            cache: Mutex::new(HashMap::new()),
            forced_hw,
        })
    }

    /// Compile (and cache) the detection graph at input size `(in_h, in_w)`.
    fn ensure_compiled(&self, in_h: usize, in_w: usize) -> Result<()> {
        let mut cache = self.cache.lock().map_err(|_| anyhow!("lock poisoned"))?;
        if cache.contains_key(&(in_h, in_w)) {
            return Ok(());
        }
        let mut wm = self.weights.weight_map()?;
        let cfg = DetectionGraphConfig {
            batch: 1,
            height: in_h,
            width: in_w,
        };
        let (graph, param_map) = build_detection_graph(&mut wm, cfg)?;
        let opts = compile_options_for_profile(&CompileProfile::encoder(), self.device);
        let mut compiled = Session::new(self.device).compile_with(graph, &opts);
        attach_built_params(&mut compiled, param_map, &[]);
        cache.insert((in_h, in_w), compiled);
        Ok(())
    }

    /// Choose the compiled input size `(in_h, in_w)` and the scaled image region
    /// `(scaled_h, scaled_w)` it occupies inside it (the remainder is padding).
    fn plan(&self, img_h: usize, img_w: usize) -> (usize, usize, usize, usize) {
        let round_up = |v: usize, m: usize| v.div_ceil(m) * m;
        match self.forced_hw {
            // Pinned size acts as an upper bound: fit the image inside it
            // preserving aspect (downscale only), pad the rest.
            Some((fh, fw)) => {
                let scale = (fh as f32 / img_h as f32)
                    .min(fw as f32 / img_w as f32)
                    .min(1.0);
                let sh = ((img_h as f32 * scale).round() as usize).clamp(1, fh);
                let sw = ((img_w as f32 * scale).round() as usize).clamp(1, fw);
                (fh, fw, sh, sw)
            }
            // Adaptive: keep native resolution (downscale only if the longest
            // side exceeds the cap), then round up to the network stride.
            None => {
                let long = img_h.max(img_w);
                let scale = if long > DET_MAX_SIDE {
                    DET_MAX_SIDE as f32 / long as f32
                } else {
                    1.0
                };
                let sh = ((img_h as f32 * scale).round() as usize).max(1);
                let sw = ((img_w as f32 * scale).round() as usize).max(1);
                (round_up(sh, DET_STRIDE), round_up(sw, DET_STRIDE), sh, sw)
            }
        }
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
        self.forced_hw
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
        let (in_h, in_w, scaled_h, scaled_w) = self.plan(img_height, img_width);
        self.ensure_compiled(in_h, in_w)?;

        // Resize (preserving aspect) into the valid region, pad the rest of the
        // compiled input with the background value.
        let scaled: NdTensor<f32, 4> = if scaled_h != img_height || scaled_w != img_width {
            resize_bilinear(image.view(), scaled_h, scaled_w)
        } else {
            NdTensor::from_data(image.shape(), image.to_vec())
        };
        let padded: NdTensor<f32, 4> = if scaled_h != in_h || scaled_w != in_w {
            pad_hw_end(scaled.view(), in_h - scaled_h, in_w - scaled_w, BLACK_VALUE)
        } else {
            scaled
        };

        let input: Vec<f32> = padded.iter().copied().collect();
        let mut cache = self.cache.lock().map_err(|_| anyhow!("lock poisoned"))?;
        let compiled = cache
            .get_mut(&(in_h, in_w))
            .ok_or_else(|| anyhow!("detection graph not compiled for {in_h}x{in_w}"))?;
        let flat = compiled
            .run(&[("image", input.as_slice())])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("detection returned no output"))?;

        let mask = NdTensor::from_data([1, 1, in_h, in_w], flat);
        // Crop the valid (scaled) region, resize the mask back to the original
        // size. Keep NCHW rank: `slice((0, 0, ..))` squeezes batch/channel.
        let mask = resize_bilinear(
            mask.slice((.., .., ..scaled_h, ..scaled_w)),
            img_height,
            img_width,
        );
        Ok(mask.into_shape([img_height, img_width]))
    }
}
