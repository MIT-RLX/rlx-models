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

//! Supertonic-3 voice styles (`voice_styles/<name>.json`).
//!
//! Each file holds two style tensors: `style_ttl` `[1, 50, 256]` (used by the
//! text encoder + vector estimator) and `style_dp` `[1, 8, 16]` (duration
//! predictor). We flatten each to a row-major `Vec<f32>` plus its `[rows, cols]`.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct RawStyle {
    dims: Vec<usize>,
    data: serde_json::Value,
}

#[derive(Deserialize)]
struct RawVoice {
    style_ttl: RawStyle,
    style_dp: RawStyle,
}

/// A flattened style tensor plus its trailing 2 dims (`[rows, cols]`).
#[derive(Debug, Clone)]
pub struct StyleTensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

/// A voice: the `style_ttl` and `style_dp` tensors.
#[derive(Debug, Clone)]
pub struct Voice {
    pub ttl: StyleTensor,
    pub dp: StyleTensor,
}

fn flatten(v: &serde_json::Value, out: &mut Vec<f32>) {
    match v {
        serde_json::Value::Array(a) => {
            for e in a {
                flatten(e, out);
            }
        }
        serde_json::Value::Number(n) => out.push(n.as_f64().unwrap_or(0.0) as f32),
        _ => {}
    }
}

impl StyleTensor {
    fn from_raw(raw: RawStyle) -> Result<Self> {
        anyhow::ensure!(
            raw.dims.len() == 3,
            "expected 3-D style dims, got {:?}",
            raw.dims
        );
        let rows = raw.dims[1];
        let cols = raw.dims[2];
        let mut data = Vec::with_capacity(rows * cols);
        flatten(&raw.data, &mut data);
        anyhow::ensure!(
            data.len() == rows * cols,
            "style data len {} != {rows}*{cols}",
            data.len()
        );
        Ok(Self { rows, cols, data })
    }
}

impl Voice {
    /// Load a voice-style JSON.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("read voice: {}", path.display()))?;
        let raw: RawVoice =
            serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        Ok(Self {
            ttl: StyleTensor::from_raw(raw.style_ttl)?,
            dp: StyleTensor::from_raw(raw.style_dp)?,
        })
    }
}

/// List available voice names in a `voice_styles/` directory (sorted).
pub fn list_voices(voice_dir: &Path) -> Result<Vec<String>> {
    let mut names: Vec<String> = std::fs::read_dir(voice_dir)
        .with_context(|| format!("read voice dir: {}", voice_dir.display()))?
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            (p.extension().and_then(|x| x.to_str()) == Some("json"))
                .then(|| p.file_stem()?.to_str().map(str::to_string))
                .flatten()
        })
        .collect();
    names.sort();
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flattens_nested() {
        let raw = RawStyle {
            dims: vec![1, 2, 3],
            data: serde_json::json!([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),
        };
        let t = StyleTensor::from_raw(raw).unwrap();
        assert_eq!(t.rows, 2);
        assert_eq!(t.cols, 3);
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
