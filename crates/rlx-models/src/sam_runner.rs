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

use anyhow::{Result, anyhow, bail};
use rlx_core::validate_sam_device;
use rlx_runtime::Device;
use std::path::PathBuf;

/// Which SAM generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamArch {
    Sam1,
    Sam2,
    Sam3,
}

/// Builder for the SAM family.
#[derive(Debug, Clone)]
pub struct SamRunnerBuilder {
    arch: SamArch,
    weights: Option<PathBuf>,
    device: Option<Device>,
    config_path: Option<PathBuf>,
}

impl SamRunnerBuilder {
    pub fn weights<P: Into<PathBuf>>(mut self, p: P) -> Self {
        self.weights = Some(p.into());
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }
    pub fn config<P: Into<PathBuf>>(mut self, p: P) -> Self {
        self.config_path = Some(p.into());
        self
    }

    /// Build (validates inputs, but does not load weights — SAM
    /// loaders today take ownership of the file path and load on
    /// demand to keep memory peaks lower).
    pub fn build(self) -> Result<SamRunner> {
        let weights = self
            .weights
            .ok_or_else(|| anyhow!("weights path required"))?;
        if !weights.exists() {
            bail!("weights file not found: {weights:?}");
        }
        let device = self.device.unwrap_or(Device::Cpu);
        validate_sam_device("sam", device)?;
        Ok(SamRunner {
            arch: self.arch,
            weights,
            device,
            config_path: self.config_path,
        })
    }
}

/// SAM runner — owns the resolved config and dispatches the
/// per-arch forward pass. SAM 1 / 2 / 3 differ enough in their
/// prompting that we keep the heavy result type
/// (`SamPredictionAny`) as a discriminated union the caller
/// matches on.
pub struct SamRunner {
    pub arch: SamArch,
    pub weights: PathBuf,
    pub device: Device,
    pub config_path: Option<PathBuf>,
}

/// Union of per-arch image-prediction outputs. Caller matches on
/// the arch they asked for.
pub enum SamPredictionAny {
    Sam1(rlx_sam::MaskPrediction),
    Sam2(rlx_sam2::Sam2ImagePrediction),
    Sam3(rlx_sam3::Sam3ImagePrediction),
}

impl SamRunner {
    pub fn builder(arch: SamArch) -> SamRunnerBuilder {
        SamRunnerBuilder {
            arch,
            weights: None,
            device: None,
            config_path: None,
        }
    }

    /// Print a human-readable summary — what the CLI prints before
    /// any per-arch image processing.
    pub fn summary(&self) -> String {
        format!(
            "SAM{} runner — weights={:?} device={:?} config={:?}",
            match self.arch {
                SamArch::Sam1 => "1",
                SamArch::Sam2 => "2",
                SamArch::Sam3 => "3",
            },
            self.weights,
            self.device,
            self.config_path
        )
    }

    /// End-to-end forward: image bytes → masks. Dispatches to the
    /// right per-arch entrypoint:
    ///   * SAM 1 → `Sam::forward` (multimask = true)
    ///   * SAM 2 → `Sam2::predict_image` (multimask = true)
    ///   * SAM 3 → `Sam3::predict_image_text` with the supplied
    ///     `text_tokens` (required for SAM 3 — its decoder is
    ///     text-conditioned). Pass an empty slice for arches that
    ///     don't use it.
    ///
    /// `rgb` is HWC u8; `points` is `(xy_pairs, labels)` with one
    /// label per (x, y) pair (1 = foreground, 0 = background).
    ///
    /// SAM-arch-specific defaults applied:
    ///   * `cfg` derived from environment variables (`RLX_SAM_VARIANT`
    ///     for v1: vit_b/l/h; `RLX_SAM2_VARIANT` for v2: tiny/small/
    ///     base_plus/large); falls back to the smallest variant.
    ///   * `multimask_output = true` for v1 + v2.
    ///   * SAM 3 vit defaults to `base`.
    pub fn predict_image(
        &self,
        rgb: &[u8],
        h_in: usize,
        w_in: usize,
        points: Option<(&[f32], &[f32])>,
        boxes: Option<&[f32]>,
        text_tokens: &[u32],
    ) -> Result<SamPredictionAny> {
        let weights_str = self
            .weights
            .to_str()
            .ok_or_else(|| anyhow!("non-utf8 weights path"))?;
        match self.arch {
            SamArch::Sam1 => {
                use rlx_sam::{Sam, SamConfig};
                let cfg = match rlx_ir::env::var("RLX_SAM_VARIANT")
                    .unwrap_or_else(|| "vit_b".into())
                    .as_str()
                {
                    "vit_b" => SamConfig::vit_b(),
                    "vit_l" => SamConfig::vit_l(),
                    "vit_h" => SamConfig::vit_h(),
                    other => bail!("RLX_SAM_VARIANT must be vit_b|vit_l|vit_h, got {other}"),
                };
                let mut sam = Sam::from_safetensors_on(weights_str, cfg, self.device)?;
                let (pred, _resized) = sam.forward(
                    rgb, h_in, w_in, points, boxes, None, /*multimask*/ true,
                )?;
                Ok(SamPredictionAny::Sam1(pred))
            }
            SamArch::Sam2 => {
                use rlx_sam2::{Sam2, Sam2Config};
                let cfg = match rlx_ir::env::var("RLX_SAM2_VARIANT")
                    .unwrap_or_else(|| "tiny".into())
                    .as_str()
                {
                    "tiny" => Sam2Config::hiera_tiny(),
                    "small" => Sam2Config::hiera_small(),
                    "base_plus" => Sam2Config::hiera_base_plus(),
                    "large" => Sam2Config::hiera_large(),
                    other => {
                        bail!("RLX_SAM2_VARIANT must be tiny|small|base_plus|large, got {other}")
                    }
                };
                let mut sam = Sam2::from_safetensors_on(weights_str, cfg, self.device)?;
                let pred = sam.predict_image(
                    rgb, h_in, w_in, points, boxes, None, /*multimask*/ true,
                )?;
                Ok(SamPredictionAny::Sam2(pred))
            }
            SamArch::Sam3 => {
                use rlx_sam3::{Sam3, Sam3Config};
                let cfg = Sam3Config::base();
                let mut sam = Sam3::from_checkpoint_on(weights_str, cfg, self.device)?;
                if text_tokens.is_empty() {
                    bail!("SAM 3 is text-conditioned — pass non-empty text_tokens");
                }
                let pred = sam.predict_image_text(rgb, h_in, w_in, text_tokens)?;
                Ok(SamPredictionAny::Sam3(pred))
            }
        }
    }
}
