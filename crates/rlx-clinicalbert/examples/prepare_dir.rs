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

//! Convert a HuggingFace ClinicalBERT snapshot dir to RLX layout
//! (`tokenizer.json`, `model.safetensors`).

#[path = "support/common.rs"]
mod common;

use anyhow::{Context, Result, bail};
use rlx_clinicalbert::prepare_clinicalbert_dir;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = common::require_flag(&args, "--dir")?;
    let dir = PathBuf::from(dir);
    if !dir.is_dir() {
        bail!("--dir must be an existing directory: {}", dir.display());
    }
    prepare_clinicalbert_dir(&dir).with_context(|| format!("prepare {}", dir.display()))?;
    println!("prepared {}", dir.display());
    Ok(())
}
