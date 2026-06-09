// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Weight paths, safetensors load, and optional `.rten` export.

mod load;
mod paths;

#[cfg(feature = "convert-rten")]
mod rten_export;

pub use crate::model::weights::{detection_input_hw, parse_hw};
pub use load::{SafetensorsFile, load_rlx_weights, load_safetensors, load_safetensors_weights};
pub use paths::prefer_safetensors_path;
pub use paths::{
    HF_DETECTION_RTEN, HF_DETECTION_ST, HF_DETECTION_ST_FULL, HF_RECOGNITION_RTEN,
    HF_RECOGNITION_ST, HF_RECOGNITION_ST_FULL, is_rten_checkpoint, resolve_model_dir,
};

#[cfg(feature = "convert-rten")]
pub use rten_export::{export_rten_to_safetensors, weight_map_from_rten};

#[cfg(not(feature = "convert-rten"))]
pub fn export_rten_to_safetensors(
    _rten_path: &std::path::Path,
    _out_path: &std::path::Path,
) -> anyhow::Result<()> {
    anyhow::bail!("rlx-ocr: rebuild with --features convert-rten to export .rten → .safetensors")
}

#[cfg(not(feature = "convert-rten"))]
pub fn weight_map_from_rten(
    _path: &std::path::Path,
) -> anyhow::Result<rlx_core::weight_map::WeightMap> {
    anyhow::bail!("rlx-ocr: rebuild with --features convert-rten to load .rten weights")
}
