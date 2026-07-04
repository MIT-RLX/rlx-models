// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! SNAC 24 kHz RVQ decoder for Orpheus speech tokens.
//!
//! - **Eager** ([`SnacDecoder`]): host safetensors + ndarray (default).
//! - **CoreML** ([`SnacBackend`] with [`SnacLoadOptions::coreml`]): quantizer on CPU,
//!   conv decoder compiled to [`Device::Ane`] via `rlx-ir` (feature `coreml`).
//!
//! Weights: [`decoder_weights_path`] (`ORPHEUS_SNAC_PATH`).

mod backend;
mod config;
mod eager;
mod ops;

#[cfg(feature = "snac-rlx")]
mod compiled;

pub use backend::{SnacBackend, SnacExec, SnacLoadOptions};
pub use config::SnacConfig;
pub use eager::{SAMPLE_RATE, SAMPLES_PER_FRAME, SnacDecoder};

use std::path::PathBuf;

use anyhow::{Result, bail};

/// Resolve SNAC decoder weights from `ORPHEUS_SNAC_PATH`.
pub fn decoder_weights_path() -> Result<PathBuf> {
    let path = std::env::var("ORPHEUS_SNAC_PATH")
        .ok()
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "ORPHEUS_SNAC_PATH is not set (SNAC weights are not bundled in rlx-orpheus)"
            )
        })?;
    if !path.exists() {
        bail!("ORPHEUS_SNAC_PATH does not exist: {}", path.display());
    }
    Ok(path)
}

/// Like [`decoder_weights_path`], but returns `None` when unset or missing (for tests).
pub fn decoder_weights_path_if_available() -> Option<PathBuf> {
    let path = std::env::var("ORPHEUS_SNAC_PATH")
        .ok()
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)?;
    if path.exists() { Some(path) } else { None }
}
