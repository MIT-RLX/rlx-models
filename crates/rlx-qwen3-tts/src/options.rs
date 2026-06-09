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
use rlx_core::validate_standard_device;
use rlx_runtime::Device;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Qwen3TtsOptions {
    pub device: Device,
    pub eager_talker: bool,
    /// Hard cap on codec frames (0 = auto budget from text; synthesis stops at talker EOS).
    pub max_frames: usize,
}

impl Default for Qwen3TtsOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            eager_talker: false,
            max_frames: 0,
        }
    }
}

impl Qwen3TtsOptions {
    pub fn validate(&self) -> Result<()> {
        validate_standard_device("qwen3-tts", self.device)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Qwen3TtsRunnerBuilder {
    model_dir: Option<PathBuf>,
    options: Qwen3TtsOptions,
}

impl Qwen3TtsRunnerBuilder {
    pub fn model_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_dir = Some(path.into());
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.options.device = device;
        self
    }

    pub fn eager_talker(mut self, on: bool) -> Self {
        self.options.eager_talker = on;
        self
    }

    pub fn max_frames(mut self, n: usize) -> Self {
        self.options.max_frames = n;
        self
    }

    pub fn build(self) -> Result<crate::runner::Qwen3TtsRunner> {
        let model_dir = self
            .model_dir
            .ok_or_else(|| anyhow::anyhow!("model_dir required"))?;
        self.options.validate()?;
        crate::runner::Qwen3TtsRunner::open_with_options(&model_dir, self.options)
    }
}
