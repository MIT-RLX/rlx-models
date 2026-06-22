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

//! Convenience model-kind detection and openers.

use std::path::Path;

use rlx_runtime::Device;

use crate::config::ModelKind;

/// Guess the model family from a checkpoint directory's `config.yaml`.
pub fn detect_kind(dir: &Path) -> Option<ModelKind> {
    let text = std::fs::read_to_string(dir.join("config.yaml"))
        .unwrap_or_default()
        .to_lowercase();
    if text.contains("sensevoice") || text.contains("sense_voice") {
        Some(ModelKind::SenseVoice)
    } else if text.contains("campplus") || text.contains("cam++") || text.contains("sv_zh") {
        Some(ModelKind::CamPlus)
    } else if (text.contains("fsmn") && text.contains("vad")) || text.contains("fsmnvad") {
        Some(ModelKind::FsmnVad)
    } else if text.contains("cttransformer")
        || text.contains("ct_transformer")
        || text.contains("punc")
    {
        Some(ModelKind::CtTransformer)
    } else if text.contains("paraformer") {
        Some(ModelKind::Paraformer)
    } else {
        None
    }
}

/// Open the model in `dir` as the appropriate ASR backbone for the pipeline.
pub fn open_asr(dir: &Path, device: Device) -> anyhow::Result<crate::pipeline::AsrModel> {
    match detect_kind(dir) {
        Some(ModelKind::SenseVoice) => Ok(crate::pipeline::AsrModel::SenseVoice(
            crate::sensevoice::SenseVoice::open(dir, device)?,
        )),
        _ => Ok(crate::pipeline::AsrModel::Paraformer(
            crate::paraformer::Paraformer::open(dir, device)?,
        )),
    }
}
