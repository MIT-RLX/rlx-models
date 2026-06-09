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

//! OCR configuration — detection thresholds, alphabet, decode method.

/// Default character alphabet matching ocrs pretrained recognition models.
pub const DEFAULT_ALPHABET: &str = " 0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~€ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// Post-processing parameters for the text detection segmentation mask.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectionParams {
    /// Minimum area of word bounding boxes returned from connected components.
    pub min_area: f32,
    /// Per-pixel score threshold for classifying a pixel as text.
    pub text_threshold: f32,
}

impl Default for DetectionParams {
    fn default() -> Self {
        Self {
            min_area: 100.,
            text_threshold: 0.2,
        }
    }
}

/// Method used to decode CRNN sequence outputs.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum DecodeMethod {
    #[default]
    Greedy,
    BeamSearch {
        width: u32,
    },
}

/// Fixed input height for the recognition model (ocrs default).
pub const RECOGNITION_INPUT_HEIGHT: u32 = 64;

/// Shared OCR settings.
#[derive(Clone, Debug)]
pub struct OcrConfig {
    pub detection: DetectionParams,
    pub decode_method: DecodeMethod,
    pub alphabet: String,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            detection: DetectionParams::default(),
            decode_method: DecodeMethod::default(),
            alphabet: DEFAULT_ALPHABET.to_string(),
        }
    }
}
