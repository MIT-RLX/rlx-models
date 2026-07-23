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

//! High-level PP-OCRv6 runner (builder → [`crate::engine::PpOcrV6Engine`]).

use crate::capabilities::validate_device;
use crate::config::{DetectionParams, RecognitionParams, Tier};
use crate::engine::{EngineParams, OcrResult, PpOcrV6Engine};
use anyhow::{Result, anyhow};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct PpOcrV6RunnerBuilder {
    tier: Tier,
    model_dir: Option<PathBuf>,
    device: Option<Device>,
    detection: DetectionParams,
    recognition: RecognitionParams,
}

impl Default for PpOcrV6RunnerBuilder {
    fn default() -> Self {
        Self {
            tier: Tier::Tiny,
            model_dir: None,
            device: None,
            detection: DetectionParams::from_tier(Tier::Tiny),
            recognition: RecognitionParams::default(),
        }
    }
}

impl PpOcrV6RunnerBuilder {
    pub fn tier(mut self, tier: Tier) -> Self {
        self.tier = tier;
        self.detection = DetectionParams::from_tier(tier);
        self
    }

    pub fn model_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.model_dir = Some(dir.into());
        self
    }

    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }

    pub fn detection(mut self, p: DetectionParams) -> Self {
        self.detection = p;
        self
    }

    pub fn recognition(mut self, p: RecognitionParams) -> Self {
        self.recognition = p;
        self
    }

    pub fn build(self) -> Result<PpOcrV6Runner> {
        let model_dir = self
            .model_dir
            .ok_or_else(|| anyhow!("model_dir(...) is required"))?;
        let device = self.device.unwrap_or(Device::Cpu);
        validate_device(device)?;
        let engine = PpOcrV6Engine::new(EngineParams {
            tier: self.tier,
            model_dir,
            device,
            detection: self.detection,
            recognition: self.recognition,
        })?;
        Ok(PpOcrV6Runner { engine })
    }
}

pub struct PpOcrV6Runner {
    engine: PpOcrV6Engine,
}

impl PpOcrV6Runner {
    pub fn builder() -> PpOcrV6RunnerBuilder {
        PpOcrV6RunnerBuilder::default()
    }

    pub fn engine(&self) -> &PpOcrV6Engine {
        &self.engine
    }

    pub fn predict_path(&self, path: impl AsRef<Path>) -> Result<OcrResult> {
        self.engine.ocr_path(path.as_ref())
    }

    pub fn predict_text(&self, path: impl AsRef<Path>) -> Result<String> {
        Ok(self.predict_path(path)?.text)
    }
}
