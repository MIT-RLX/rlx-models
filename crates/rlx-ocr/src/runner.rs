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

//! High-level OCR runner with image loading.

use crate::capabilities::validate_device;
use crate::config::OcrConfig;
use crate::engine::{OcrEngine, OcrEngineParams};
use crate::text::TextLine;
use crate::weights::resolve_model_dir;
use anyhow::{Context, Result, anyhow};
use rlx_runtime::Device;
use rten_imageproc::RotatedRect;
use std::path::{Path, PathBuf};

/// Structured OCR output.
#[derive(Debug, Clone)]
pub struct OcrOutput {
    pub text: String,
    pub lines: Vec<Option<TextLine>>,
    pub words: Vec<RotatedRect>,
}

/// Builder for [`OcrRunner`] (mirrors whisper / dinov2 runners).
#[derive(Debug, Clone, Default)]
pub struct OcrRunnerBuilder {
    model_dir: Option<PathBuf>,
    detection_model: Option<PathBuf>,
    recognition_model: Option<PathBuf>,
    device: Option<Device>,
    alphabet: Option<String>,
}

impl OcrRunnerBuilder {
    pub fn model_dir<P: Into<PathBuf>>(mut self, dir: P) -> Self {
        self.model_dir = Some(dir.into());
        self
    }

    pub fn detection_model<P: Into<PathBuf>>(mut self, p: P) -> Self {
        self.detection_model = Some(p.into());
        self
    }

    pub fn recognition_model<P: Into<PathBuf>>(mut self, p: P) -> Self {
        self.recognition_model = Some(p.into());
        self
    }

    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }

    pub fn alphabet(mut self, alphabet: impl Into<String>) -> Self {
        self.alphabet = Some(alphabet.into());
        self
    }

    pub fn build(self) -> Result<OcrRunner> {
        let (detection, recognition) = match (
            self.detection_model,
            self.recognition_model,
            self.model_dir,
        ) {
            (Some(d), Some(r), _) => (d, r),
            (_, _, Some(dir)) => resolve_model_dir(&dir)?,
            _ => {
                return Err(anyhow!(
                    "provide model_dir(...) or both detection_model(...) and recognition_model(...)"
                ));
            }
        };
        let device = self.device.unwrap_or(Device::Cpu);
        validate_device(device)?;

        let engine = OcrEngine::new(OcrEngineParams {
            detection_model: Some(detection),
            recognition_model: Some(recognition),
            alphabet: self.alphabet,
            device,
            ..Default::default()
        })?;

        Ok(OcrRunner { engine, device })
    }
}

/// OCR session wrapping a fully loaded [`OcrEngine`].
pub struct OcrRunner {
    engine: OcrEngine,
    device: Device,
}

impl OcrRunner {
    pub fn builder() -> OcrRunnerBuilder {
        OcrRunnerBuilder::default()
    }

    pub fn engine(&self) -> &OcrEngine {
        &self.engine
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn config(&self) -> OcrConfig {
        self.engine.config()
    }

    /// Run OCR on an image file (JPEG/PNG via `image` crate).
    pub fn predict_path(&self, path: &Path) -> Result<OcrOutput> {
        let img = image::open(path)
            .with_context(|| format!("open image {path:?}"))?
            .into_rgb8();
        let (w, h) = img.dimensions();
        self.predict_rgb(img.as_raw(), w, h)
    }

    /// Run OCR on RGB8 bytes.
    pub fn predict_rgb(&self, rgb: &[u8], width: u32, height: u32) -> Result<OcrOutput> {
        let source = crate::ImageSource::from_bytes(rgb, (width, height))?;
        let input = self.engine.prepare_input(source)?;
        let words = self.engine.detect_words(&input)?;
        let line_rects = self.engine.find_text_lines(&input, &words);
        let lines = self.engine.recognize_text(&input, &line_rects)?;
        let text = lines
            .iter()
            .filter_map(|l| l.as_ref().map(TextLine::text))
            .collect::<Vec<_>>()
            .join("\n");
        Ok(OcrOutput { text, lines, words })
    }

    /// Convenience: text only.
    pub fn predict_text(&self, path: &Path) -> Result<String> {
        Ok(self.predict_path(path)?.text)
    }
}
