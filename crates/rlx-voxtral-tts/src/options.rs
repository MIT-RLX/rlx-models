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

//! Runner options — device and eager fallbacks.

use anyhow::Result;
use rlx_core::validate_standard_device;
use rlx_runtime::Device;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct VoxtralTtsOptions {
    pub device: Device,
    /// Hand-ported CPU LM (parity / debugging). Default: compiled RLX graph.
    pub eager_lm: bool,
    /// Hand-ported CPU acoustic stack. Default: compiled RLX graph.
    pub eager_acoustic: bool,
}

impl Default for VoxtralTtsOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            eager_lm: std::env::var("RLX_VOXTRAL_TTS_EAGER").ok().as_deref() == Some("1"),
            eager_acoustic: std::env::var("RLX_VOXTRAL_TTS_ACOUSTIC_EAGER")
                .ok()
                .as_deref()
                == Some("1"),
        }
    }
}

impl VoxtralTtsOptions {
    pub fn validate(&self) -> Result<()> {
        validate_standard_device("voxtral-tts", self.device)
    }
}

#[derive(Debug, Clone, Default)]
pub struct VoxtralTtsRunnerBuilder {
    model_dir: Option<PathBuf>,
    options: VoxtralTtsOptions,
}

impl VoxtralTtsRunnerBuilder {
    pub fn model_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_dir = Some(path.into());
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.options.device = device;
        self
    }

    pub fn eager_lm(mut self, on: bool) -> Self {
        self.options.eager_lm = on;
        self
    }

    pub fn eager_acoustic(mut self, on: bool) -> Self {
        self.options.eager_acoustic = on;
        self
    }

    pub fn options(mut self, options: VoxtralTtsOptions) -> Self {
        self.options = options;
        self
    }

    pub fn build(self) -> Result<crate::runner::VoxtralTtsRunner> {
        let model_dir = self
            .model_dir
            .ok_or_else(|| anyhow::anyhow!("model_dir required (call .model_dir(...))"))?;
        self.options.validate()?;
        crate::runner::VoxtralTtsRunner::open_with_options(&model_dir, self.options)
    }
}
