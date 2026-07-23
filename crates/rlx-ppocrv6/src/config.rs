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

//! PP-OCRv6 tier + detection / recognition hyper-parameters.

use anyhow::{Result, bail};
use std::str::FromStr;

/// Model capacity tier (medium is out of scope for this crate).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tier {
    Tiny,
    Small,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tiny => "tiny",
            Self::Small => "small",
        }
    }
}

impl FromStr for Tier {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "tiny" => Ok(Self::Tiny),
            "small" => Ok(Self::Small),
            other => bail!("unknown PP-OCRv6 tier `{other}` (expected tiny|small)"),
        }
    }
}

/// DB detection post-process (matches official `inference.yml`).
#[derive(Debug, Clone, Copy)]
pub struct DetectionParams {
    pub thresh: f32,
    pub box_thresh: f32,
    pub unclip_ratio: f32,
    pub max_candidates: usize,
    pub min_size: f32,
    /// Longest side after aspect-preserving resize (Paddle DetResizeForTest).
    pub limit_side_len: usize,
}

impl Default for DetectionParams {
    fn default() -> Self {
        Self {
            thresh: 0.3,
            box_thresh: 0.6,
            unclip_ratio: 1.5,
            max_candidates: 1000,
            min_size: 3.0,
            limit_side_len: 960,
        }
    }
}

impl DetectionParams {
    pub fn from_tier(tier: Tier) -> Self {
        // Official tiny/small ONNX yml: thresh=0.2, box_thresh=0.4, unclip=1.4
        let _ = tier;
        Self {
            thresh: 0.2,
            box_thresh: 0.4,
            unclip_ratio: 1.4,
            max_candidates: 3000,
            min_size: 3.0,
            limit_side_len: 960,
        }
    }
}

/// Recognition crop / CTC settings.
#[derive(Debug, Clone)]
pub struct RecognitionParams {
    pub image_height: usize,
    pub max_width: usize,
    pub batch_size: usize,
}

impl Default for RecognitionParams {
    fn default() -> Self {
        Self {
            image_height: 48,
            max_width: 3200,
            batch_size: 8,
        }
    }
}

/// ImageNet-style normalize used by PP-OCR detection.
pub const DET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const DET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Recognition normalize `(x / 255 - 0.5) / 0.5` → `[-1, 1]` (Paddle RecResizeImg).
pub const REC_SCALE: f32 = 1.0 / 255.0;
pub const REC_MEAN: f32 = 0.5;
pub const REC_STD: f32 = 0.5;
