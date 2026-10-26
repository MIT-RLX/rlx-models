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

use anyhow::{Result, anyhow};
use std::path::Path;

/// What the model file is. Used by runners to pick the right loader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    Safetensors,
    Gguf,
}

impl WeightFormat {
    /// Infer format from a file extension, or from directory contents.
    ///
    /// Defers to [`rlx_core::weight_registry::resolve_format`], which is the
    /// one place that decides what a path means. This used to decide for
    /// itself and preferred GGUF, while the loader registry preferred
    /// safetensors — so a directory holding both ran different weights
    /// depending on which code path reached it first. For a directory
    /// holding one format, or a path naming a file, the answer is unchanged;
    /// a directory holding several is now an error that names them instead
    /// of a silent pick.
    pub fn detect(path: &Path) -> Result<Self> {
        if path.is_dir() {
            return match rlx_core::weight_registry::resolve_format(path)? {
                "gguf" => Ok(Self::Gguf),
                // An mlx-quantised directory is safetensors on disk; the
                // loader routes it onwards by itself.
                "safetensors" | "mlx" => Ok(Self::Safetensors),
                other => Err(anyhow!(
                    "{path:?} holds {other} weights, which this command cannot \
                     load — it understands safetensors and GGUF"
                )),
            };
        }
        Self::from_path(path)
    }

    pub fn from_path(path: &Path) -> Result<Self> {
        match path.extension().and_then(|s| s.to_str()) {
            Some("safetensors") => Ok(Self::Safetensors),
            Some("gguf") => Ok(Self::Gguf),
            other => {
                let hint = rlx_core::registered_extensions_hint();
                Err(anyhow!(
                    "cannot autodetect weight format from extension {:?} on {path:?}\n\
                     Registered extensions: .{hint}\n\
                     Pass --format safetensors|gguf, or register a custom extension via rlx_core::weights",
                    other
                ))
            }
        }
    }

    /// Parse CLI `--format` values (`safetensors` | `gguf`).
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "safetensors" => Ok(Self::Safetensors),
            "gguf" => Ok(Self::Gguf),
            other => Err(anyhow!("expected safetensors|gguf, got {other}")),
        }
    }

    /// Use an explicit override when set; otherwise infer from the file extension.
    pub fn resolve(path: &Path, override_fmt: Option<Self>) -> Result<Self> {
        match override_fmt {
            Some(f) => Ok(f),
            None => Self::detect(path),
        }
    }
}
