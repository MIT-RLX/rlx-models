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

//! NeuCodec decoder — FSQ lookup + Vocos backbone + ISTFT head.
//!
//! | Feature     | Path                                      |
//! |-------------|-------------------------------------------|
//! | `codec`     | Eager ndarray CPU ([`eager`]) — default   |
//! | `burn`      | Burn wgpu → NdArray → eager fallback      |
//! | `rlx`       | RLX runtime (eager parity until graph)    |
//!
//! Decoder weights are **not** shipped in this crate. Set [`decoder_weights_path`] via
//! `NEUTTS_DECODER_PATH` (path to `neucodec_decoder.safetensors`).

use std::path::PathBuf;

use anyhow::{Result, bail};

mod eager;

#[cfg(feature = "burn")]
mod burn;

#[cfg(feature = "rlx")]
mod rlx;

pub use eager::{
    ENCODER_DEFAULT_INPUT_SAMPLES, ENCODER_SAMPLE_RATE, ENCODER_SAMPLES_PER_TOKEN, NeuCodecDecoder,
    NeuCodecEncoder, SAMPLE_RATE, SAMPLES_PER_TOKEN,
};

pub use crate::features::{burn_feature_enabled, rlx_feature_enabled, wgpu_feature_enabled};

/// Resolve NeuCodec decoder weights from `NEUTTS_DECODER_PATH`.
pub fn decoder_weights_path() -> Result<PathBuf> {
    let path = std::env::var("NEUTTS_DECODER_PATH")
        .ok()
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "NEUTTS_DECODER_PATH is not set (NeuCodec weights are not bundled in rlx-neutts)"
            )
        })?;
    if !path.exists() {
        bail!("NEUTTS_DECODER_PATH does not exist: {}", path.display());
    }
    Ok(path)
}

/// Like [`decoder_weights_path`], but returns `None` when unset or missing (for tests).
pub fn decoder_weights_path_if_available() -> Option<PathBuf> {
    let path = std::env::var("NEUTTS_DECODER_PATH")
        .ok()
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)?;
    if path.exists() { Some(path) } else { None }
}
