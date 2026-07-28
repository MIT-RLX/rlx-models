// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! Weight I/O helpers (little-endian f32 blobs).

use anyhow::{Context, Result, bail};
use std::fs;
use std::path::Path;

/// Read a little-endian f32 blob.
pub(crate) fn read_f32_bin(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() % 4 != 0 {
        bail!("{} size not multiple of 4", path.display());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
