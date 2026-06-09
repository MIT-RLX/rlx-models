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

//! Download ocrs checkpoints and ensure `.safetensors` exports for native RLX benches.

use anyhow::{Context, Result};
use rlx_ocr::weights::{
    HF_DETECTION_RTEN, HF_DETECTION_ST, HF_DETECTION_ST_FULL, HF_RECOGNITION_RTEN,
    HF_RECOGNITION_ST, HF_RECOGNITION_ST_FULL, export_rten_to_safetensors, prefer_safetensors_path,
    resolve_model_dir,
};
use std::path::{Path, PathBuf};

const DET_URL: &str =
    "https://huggingface.co/robertknight/ocrs/resolve/main/text-detection-ssfbcj81.rten";
const REC_URL: &str =
    "https://huggingface.co/robertknight/ocrs/resolve/main/text-rec-checkpoint-s52qdbqt.rten";
#[allow(dead_code)]
const TEST_IMAGE_URL: &str =
    "https://raw.githubusercontent.com/robertknight/ocrs/main/ocrs-cli/test-data/why-rust.png";

pub fn download(url: &str, dest: &Path) -> Result<()> {
    eprintln!("downloading {url} -> {}", dest.display());
    let resp = ureq::get(url)
        .call()
        .with_context(|| format!("GET {url}"))?;
    let mut reader = resp.into_reader();
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut reader, &mut bytes)?;
    std::fs::write(dest, bytes)?;
    Ok(())
}

pub fn ensure_rten_checkpoints(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let det = dir.join(HF_DETECTION_RTEN);
    let rec = dir.join(HF_RECOGNITION_RTEN);
    if !det.is_file() {
        download(DET_URL, &det)?;
    }
    if !rec.is_file() {
        download(REC_URL, &rec)?;
    }
    Ok(())
}

/// Export `.rten` → `.safetensors` when missing (native RLX path).
pub fn ensure_safetensors_exports(dir: &Path) -> Result<()> {
    ensure_rten_checkpoints(dir)?;
    let det_st = prefer_safetensors_path(dir, HF_DETECTION_ST, HF_DETECTION_ST_FULL);
    let rec_st = prefer_safetensors_path(dir, HF_RECOGNITION_ST, HF_RECOGNITION_ST_FULL);
    if !det_st.is_file() {
        export_rten_to_safetensors(&dir.join(HF_DETECTION_RTEN), &det_st)?;
    }
    if !rec_st.is_file() {
        export_rten_to_safetensors(&dir.join(HF_RECOGNITION_RTEN), &rec_st)?;
    }
    Ok(())
}

#[allow(dead_code)]
pub fn ensure_rlx_model_dir(dir: &Path) -> Result<(PathBuf, PathBuf)> {
    ensure_safetensors_exports(dir)?;
    resolve_model_dir(dir)
}

#[allow(dead_code)]
pub fn ensure_test_image(path: &Path) -> Result<()> {
    if path.is_file() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    download(TEST_IMAGE_URL, path)
}

pub fn default_model_dir() -> PathBuf {
    std::env::temp_dir().join("rlx-ocr-parity-models")
}

#[allow(dead_code)]
pub fn default_test_image() -> PathBuf {
    std::env::temp_dir().join("rlx-ocr-parity-why-rust.png")
}
