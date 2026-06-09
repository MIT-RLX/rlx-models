// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! HuggingFace `robertknight/ocrs` checkpoint filenames and model-dir resolution.

use anyhow::{Result, bail};
use std::path::{Path, PathBuf};

/// Safetensors export of detection weights (short name).
pub const HF_DETECTION_ST: &str = "ocr-detection.safetensors";
/// Full detection export (includes int32 BN scalars from RTen).
pub const HF_DETECTION_ST_FULL: &str = "ocr-detection-full.safetensors";
/// Safetensors export of recognition weights.
pub const HF_RECOGNITION_ST: &str = "ocr-recognition.safetensors";
pub const HF_RECOGNITION_ST_FULL: &str = "ocr-recognition-full.safetensors";

/// Production detection checkpoint (legacy RTen graph).
pub const HF_DETECTION_RTEN: &str = "text-detection-ssfbcj81.rten";
/// Production recognition checkpoint (legacy RTen CRNN + GRU graph).
pub const HF_RECOGNITION_RTEN: &str = "text-rec-checkpoint-s52qdbqt.rten";

/// True when `path` is an ocrs `.rten` checkpoint.
pub fn is_rten_checkpoint(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("rten"))
}

pub fn prefer_safetensors_path(base: &Path, short: &str, full: &str) -> PathBuf {
    let full_path = base.join(full);
    if full_path.is_file() {
        return full_path;
    }
    base.join(short)
}

/// Resolve detection + recognition weight paths under `dir` (safetensors for native RLX).
pub fn resolve_model_dir(dir: &Path) -> Result<(PathBuf, PathBuf)> {
    let det_st = prefer_safetensors_path(dir, HF_DETECTION_ST, HF_DETECTION_ST_FULL);
    let rec_st = prefer_safetensors_path(dir, HF_RECOGNITION_ST, HF_RECOGNITION_ST_FULL);
    if det_st.is_file() && rec_st.is_file() {
        return Ok((det_st, rec_st));
    }

    #[cfg(feature = "rten-inference")]
    {
        let det_rten = dir.join(HF_DETECTION_RTEN);
        let rec_rten = dir.join(HF_RECOGNITION_RTEN);
        if det_rten.is_file() && rec_rten.is_file() {
            return Ok((det_rten, rec_rten));
        }
    }

    #[cfg(feature = "rten-inference")]
    let legacy =
        format!(" or legacy {HF_DETECTION_RTEN} + {HF_RECOGNITION_RTEN} (enable `rten-inference`)");
    #[cfg(not(feature = "rten-inference"))]
    let legacy = String::new();

    bail!(
        "missing ocrs checkpoints in {dir:?}: need {HF_DETECTION_ST_FULL} + {HF_RECOGNITION_ST_FULL} \
         (run `rlx-ocr-convert` on .rten files from https://huggingface.co/robertknight/ocrs){legacy}"
    );
}
