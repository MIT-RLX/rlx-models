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

//! High-level OCR engine — detect, layout, recognize.

use crate::config::{DEFAULT_ALPHABET, DecodeMethod, DetectionParams, OcrConfig};
use crate::layout::find_text_lines;
use crate::preprocess::{ImageSource, prepare_image};
use crate::text::TextLine;
use anyhow::{Context, Result, anyhow};
use rlx_runtime::Device;
use rten_imageproc::RotatedRect;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};

#[cfg(feature = "rlx")]
use crate::rlx::{RlxTextDetector, RlxTextRecognizer};

#[cfg(all(feature = "rten-inference", not(feature = "rlx")))]
use crate::inference::{RtenTextDetector, RtenTextRecognizer};

/// Parameters for constructing an [`OcrEngine`].
pub struct OcrEngineParams {
    pub detection_model: Option<std::path::PathBuf>,
    pub recognition_model: Option<std::path::PathBuf>,
    pub detection: DetectionParams,
    pub decode_method: DecodeMethod,
    pub alphabet: Option<String>,
    pub allowed_chars: Option<String>,
    pub device: Device,
}

impl Default for OcrEngineParams {
    fn default() -> Self {
        Self {
            detection_model: None,
            recognition_model: None,
            detection: DetectionParams::default(),
            decode_method: DecodeMethod::default(),
            alphabet: None,
            allowed_chars: None,
            device: Device::Cpu,
        }
    }
}

/// Preprocessed greyscale input image `[1, H, W]`.
pub struct OcrInput {
    pub(crate) image: NdTensor<f32, 3>,
}

/// End-to-end OCR pipeline (ocrs-compatible API).
pub struct OcrEngine {
    #[cfg(feature = "rlx")]
    detector: Option<RlxTextDetector>,
    #[cfg(feature = "rlx")]
    recognizer: Option<RlxTextRecognizer>,
    #[cfg(all(feature = "rten-inference", not(feature = "rlx")))]
    detector: Option<RtenTextDetector>,
    #[cfg(all(feature = "rten-inference", not(feature = "rlx")))]
    recognizer: Option<RtenTextRecognizer>,
    detection: DetectionParams,
    decode_method: DecodeMethod,
    alphabet: String,
    excluded_char_labels: Option<Vec<usize>>,
}

impl OcrEngine {
    /// Build from explicit model paths (`.safetensors` for native RLX; `.rten` only with `rten-inference`).
    pub fn from_paths(
        detection: impl AsRef<std::path::Path>,
        recognition: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        Self::from_paths_on_device(detection, recognition, Device::Cpu)
    }

    pub fn from_paths_on_device(
        detection: impl AsRef<std::path::Path>,
        recognition: impl AsRef<std::path::Path>,
        device: Device,
    ) -> Result<Self> {
        Self::new(OcrEngineParams {
            detection_model: Some(detection.as_ref().to_path_buf()),
            recognition_model: Some(recognition.as_ref().to_path_buf()),
            device,
            ..Default::default()
        })
    }

    pub fn new(params: OcrEngineParams) -> Result<Self> {
        let detection = params.detection;
        let device = params.device;
        #[cfg(feature = "rlx")]
        let detector = params
            .detection_model
            .as_ref()
            .map(|p| RlxTextDetector::from_path(p, detection.clone(), device))
            .transpose()
            .context("load detection model")?;
        #[cfg(feature = "rlx")]
        let recognizer = params
            .recognition_model
            .as_ref()
            .map(|p| RlxTextRecognizer::from_path(p, device))
            .transpose()
            .context("load recognition model")?;

        #[cfg(all(feature = "rten-inference", not(feature = "rlx")))]
        let detector = params
            .detection_model
            .as_ref()
            .map(|p| RtenTextDetector::from_path(p, detection.clone()))
            .transpose()
            .context("load detection model")?;
        #[cfg(all(feature = "rten-inference", not(feature = "rlx")))]
        let recognizer = params
            .recognition_model
            .as_ref()
            .map(RtenTextRecognizer::from_path)
            .transpose()
            .context("load recognition model")?;

        let alphabet = params
            .alphabet
            .unwrap_or_else(|| DEFAULT_ALPHABET.to_string());
        let excluded_char_labels = params.allowed_chars.as_ref().map(|allowed| {
            alphabet
                .chars()
                .enumerate()
                .filter_map(|(i, ch)| {
                    if allowed.contains(ch) {
                        None
                    } else {
                        Some(i + 1)
                    }
                })
                .collect()
        });

        Ok(Self {
            detector,
            recognizer,
            detection,
            decode_method: params.decode_method,
            alphabet,
            excluded_char_labels,
        })
    }

    /// Load default HuggingFace checkpoint filenames from a model directory (CPU).
    pub fn from_model_dir(dir: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_model_dir_on_device(dir, Device::Cpu)
    }

    /// Load checkpoints from `dir` and compile graphs on `device`.
    pub fn from_model_dir_on_device(
        dir: impl AsRef<std::path::Path>,
        device: Device,
    ) -> Result<Self> {
        let (det, rec) = crate::weights::resolve_model_dir(dir.as_ref())?;
        Self::from_paths_on_device(det, rec, device)
    }

    pub fn prepare_input(&self, source: ImageSource<'_>) -> Result<OcrInput> {
        Ok(OcrInput {
            image: prepare_image(source),
        })
    }

    pub fn detection_threshold(&self) -> f32 {
        self.detection.text_threshold
    }

    pub fn detect_words(&self, input: &OcrInput) -> Result<Vec<RotatedRect>> {
        let detector = self
            .detector
            .as_ref()
            .ok_or_else(|| anyhow!("detection model not configured"))?;
        detector.detect_words(input.image.view())
    }

    pub fn detect_text_pixels(&self, input: &OcrInput) -> Result<NdTensor<f32, 2>> {
        let detector = self
            .detector
            .as_ref()
            .ok_or_else(|| anyhow!("detection model not configured"))?;
        detector.detect_text_pixels(input.image.view())
    }

    pub fn find_text_lines(
        &self,
        _input: &OcrInput,
        words: &[RotatedRect],
    ) -> Vec<Vec<RotatedRect>> {
        find_text_lines(words)
    }

    pub fn prepare_recognition_input(
        &self,
        input: &OcrInput,
        line: &[RotatedRect],
    ) -> Result<NdTensor<f32, 2>> {
        let recognizer = self
            .recognizer
            .as_ref()
            .ok_or_else(|| anyhow!("recognition model not configured"))?;
        Ok(recognizer.prepare_input(input.image.view(), line))
    }

    pub fn recognize_text(
        &self,
        input: &OcrInput,
        lines: &[Vec<RotatedRect>],
    ) -> Result<Vec<Option<TextLine>>> {
        let recognizer = self
            .recognizer
            .as_ref()
            .ok_or_else(|| anyhow!("recognition model not configured"))?;
        recognizer.recognize_text_lines(
            input.image.view(),
            lines,
            self.decode_method,
            &self.alphabet,
            self.excluded_char_labels.as_deref(),
        )
    }

    pub fn get_text(&self, input: &OcrInput) -> Result<String> {
        let words = self.detect_words(input)?;
        let lines = self.find_text_lines(input, &words);
        let recognized = self.recognize_text(input, &lines)?;
        Ok(recognized
            .into_iter()
            .filter_map(|l| l.map(|tl| tl.text()))
            .collect::<Vec<_>>()
            .join("\n"))
    }

    pub fn config(&self) -> OcrConfig {
        OcrConfig {
            detection: self.detection.clone(),
            decode_method: self.decode_method,
            alphabet: self.alphabet.clone(),
        }
    }
}

pub fn input_image(input: &OcrInput) -> NdTensorView<'_, f32, 3> {
    input.image.view()
}

pub fn ocr_rgb_bytes(engine: &OcrEngine, rgb: &[u8], width: u32, height: u32) -> Result<String> {
    let source = ImageSource::from_bytes(rgb, (width, height))?;
    let input = engine.prepare_input(source)?;
    engine.get_text(&input)
}
