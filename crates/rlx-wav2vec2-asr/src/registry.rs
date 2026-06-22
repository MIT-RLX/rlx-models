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

use crate::config::AlignModelSpec;

/// Language code → HuggingFace Wav2Vec2 CTC model (WhisperX registry subset).
pub fn align_model_for_language(lang: &str) -> Option<AlignModelSpec> {
    match lang.split('-').next().unwrap_or(lang) {
        "en" => Some(AlignModelSpec {
            hf_repo: "facebook/wav2vec2-large-960h-lv60",
            config_file: "config.json",
        }),
        "de" => Some(AlignModelSpec {
            hf_repo: "facebook/wav2vec2-base-960h",
            config_file: "config.json",
        }),
        "fr" => Some(AlignModelSpec {
            hf_repo: "LeBenchmark/wav2vec2-FR-7K-large",
            config_file: "config.json",
        }),
        "es" => Some(AlignModelSpec {
            hf_repo: "facebook/wav2vec2-large-xlsr-53-spanish",
            config_file: "config.json",
        }),
        _ => None,
    }
}
