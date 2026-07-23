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

//! End-to-end PP-OCRv6 engine: detect → crop → recognize (native RLX sessions).

use crate::capabilities::{configure_coreml_for_ocr, validate_device};
use crate::config::{DetectionParams, RecognitionParams, Tier};
use crate::detection::DetBox;
use crate::preprocess::{RgbPage, crop_quad, load_rgb};
use crate::recognition::CharDict;
use crate::rlx::{RlxDetector, RlxRecognizer};
use crate::weights::resolve_model_dir;
use anyhow::Result;
use rlx_runtime::Device;
use std::path::Path;

pub struct EngineParams {
    pub tier: Tier,
    pub model_dir: std::path::PathBuf,
    pub device: Device,
    pub detection: DetectionParams,
    pub recognition: RecognitionParams,
}

pub struct PpOcrV6Engine {
    pub tier: Tier,
    detector: RlxDetector,
    recognizer: RlxRecognizer,
    device: Device,
}

impl PpOcrV6Engine {
    pub fn new(p: EngineParams) -> Result<Self> {
        let (det, rec, dict_path) = resolve_model_dir(&p.model_dir, p.tier)?;
        let dict = CharDict::load(&dict_path).unwrap_or_else(|_| match p.tier {
            Tier::Tiny => CharDict::from_embedded(crate::recognition::TINY_DICT),
            Tier::Small => CharDict::from_embedded(crate::recognition::SMALL_DICT),
        });
        validate_device(p.device)?;
        configure_coreml_for_ocr(p.device);
        let detector = RlxDetector::from_safetensors(&det, p.tier, p.detection, p.device)?;
        let recognizer =
            RlxRecognizer::from_safetensors(&rec, p.tier, dict, p.recognition, p.device)?;
        Ok(Self {
            tier: p.tier,
            detector,
            recognizer,
            device: p.device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn detect_path(&self, path: &Path) -> Result<Vec<DetBox>> {
        let page = load_rgb(path)?;
        self.detector.detect(&page)
    }

    pub fn ocr_page(&self, page: &RgbPage) -> Result<OcrResult> {
        let boxes = self.detector.detect(page)?;
        let mut lines = Vec::with_capacity(boxes.len());
        for b in &boxes {
            let crop = crop_quad(page, &b.points);
            let text = self.recognizer.recognize_crop(&crop)?;
            lines.push(OcrLine {
                text,
                score: b.score,
                points: b.points,
            });
        }
        // reading order: top-to-bottom, then left-to-right
        lines.sort_by(|a, b| {
            let ay = a.points.iter().map(|p| p[1]).sum::<f32>() / 4.0;
            let by = b.points.iter().map(|p| p[1]).sum::<f32>() / 4.0;
            ay.partial_cmp(&by)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    let ax = a.points.iter().map(|p| p[0]).sum::<f32>() / 4.0;
                    let bx = b.points.iter().map(|p| p[0]).sum::<f32>() / 4.0;
                    ax.partial_cmp(&bx).unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        let text = lines
            .iter()
            .map(|l| l.text.as_str())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        Ok(OcrResult { text, lines })
    }

    pub fn ocr_path(&self, path: &Path) -> Result<OcrResult> {
        let page = load_rgb(path)?;
        self.ocr_page(&page)
    }
}

#[derive(Debug, Clone)]
pub struct OcrLine {
    pub text: String,
    pub score: f32,
    pub points: [[f32; 2]; 4],
}

#[derive(Debug, Clone)]
pub struct OcrResult {
    pub text: String,
    pub lines: Vec<OcrLine>,
}
