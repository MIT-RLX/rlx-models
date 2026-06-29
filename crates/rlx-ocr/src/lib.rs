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

//! OCR engine for RLX — native compiled detection + recognition graphs.
//!
//! Load HuggingFace [`robertknight/ocrs`](https://huggingface.co/robertknight/ocrs) weights as
//! `.safetensors` (use `rlx-ocr-convert` with feature `convert-rten` to export from legacy `.rten`).
//!
//! Detection + recognition run on every standard RLX backend when the matching `rlx-runtime`
//! feature is enabled (`cpu`, `metal`, `mlx`, `cuda`, `rocm`, `gpu`, `vulkan`). Build with
//! `all-backends` for a single binary that accepts all `--device` values.
//!
//! **Parity / baseline:** `parity-ocrs` (upstream ocrs + RTen) or `rten-inference` (RTen graphs only).

pub mod capabilities;
pub mod config;
pub mod ctc;
pub mod detection;
pub mod engine;
pub mod geom;
pub mod host_resize;
pub mod layout;
pub mod model;
pub mod preprocess;
pub mod recognition;
pub mod runner;
pub mod text;
pub mod weights;

#[cfg(feature = "rlx")]
pub mod rlx;

#[cfg(feature = "rten-inference")]
pub mod inference;

pub mod cli;

pub use capabilities::{STANDARD_DEVICE_NAMES, STANDARD_DEVICES, validate_device};
pub use config::{DEFAULT_ALPHABET, DecodeMethod, DetectionParams, OcrConfig};
pub use ctc::{CtcHypothesis, CtcStep, decode};
pub use engine::{OcrEngine, OcrEngineParams, OcrInput, input_image, ocr_rgb_bytes};
pub use preprocess::{BLACK_VALUE, DimOrder, ImageSource, ImageSourceError, prepare_image};
pub use runner::{OcrOutput, OcrRunner, OcrRunnerBuilder};
pub use text::{TextChar, TextItem, TextLine, TextWord};
pub use weights::{
    HF_DETECTION_RTEN, HF_DETECTION_ST, HF_DETECTION_ST_FULL, HF_RECOGNITION_RTEN,
    HF_RECOGNITION_ST, HF_RECOGNITION_ST_FULL, SafetensorsFile, is_rten_checkpoint,
    load_rlx_weights, load_safetensors, load_safetensors_weights, resolve_model_dir,
};

pub use model::{DetectionGraphConfig, RecognitionGraphConfig};
#[cfg(feature = "rlx")]
pub use rlx::{RlxTextDetector, RlxTextRecognizer};

#[cfg(feature = "rten-inference")]
pub use inference::{RtenTextDetector, RtenTextRecognizer};

#[cfg(feature = "convert-rten")]
pub use weights::{export_rten_to_safetensors, weight_map_from_rten};

pub use rten_imageproc::RotatedRect;
