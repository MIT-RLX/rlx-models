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

use anyhow::Result;
use std::path::Path;

use super::embedded;

use crate::SampleRate;

/// Silero VAD weights (16 kHz ONNX branch).
#[derive(Clone)]
pub struct SileroWeights {
    pub stft_conv: Vec<f32>,
    pub conv1_w: Vec<f32>,
    pub conv1_b: Vec<f32>,
    pub conv2_w: Vec<f32>,
    pub conv2_b: Vec<f32>,
    pub conv3_w: Vec<f32>,
    pub conv3_b: Vec<f32>,
    pub conv4_w: Vec<f32>,
    pub conv4_b: Vec<f32>,
    pub lstm_w_ih: Vec<f32>,
    pub lstm_w_hh: Vec<f32>,
    pub lstm_b_ih: Vec<f32>,
    pub lstm_b_hh: Vec<f32>,
    pub final_w: Vec<f32>,
    pub final_b: Vec<f32>,
}

impl SileroWeights {
    /// Default embedded weights (`silero_vad_16k.safetensors`, no external files).
    pub fn embedded() -> Self {
        embedded::embedded().clone()
    }

    /// Load the same safetensors layout from disk (optional override).
    pub fn load(path: &Path) -> Result<Self> {
        embedded::load_file(path)
    }
}

pub fn frame_samples(sr: SampleRate) -> usize {
    match sr {
        SampleRate::Hz8000 => super::FRAME_SAMPLES_8K,
        SampleRate::Hz16000 => super::FRAME_SAMPLES_16K,
    }
}

pub fn context_samples(sr: SampleRate) -> usize {
    match sr {
        SampleRate::Hz8000 => super::CONTEXT_8K,
        SampleRate::Hz16000 => super::CONTEXT_16K,
    }
}
