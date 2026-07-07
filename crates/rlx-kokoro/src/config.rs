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

//! Kokoro model directory layout.
//!
//! A Kokoro ONNX bundle (e.g. `onnx-community/Kokoro-82M-v1.0-ONNX`) looks like:
//!
//! ```text
//! config.json
//! tokenizer.json
//! onnx/model.onnx           (also model_fp16, model_q8f16, …)
//! voices/af_heart.bin       (one .bin per voice)
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// Hugging Face repo shipping the ONNX export + voice packs.
pub const DEFAULT_HF_REPO: &str = "onnx-community/Kokoro-82M-v1.0-ONNX";

/// Default local checkout directory (centralized under `weights/tts/`, which is
/// gitignored and destined for a standalone Hugging Face weights repo).
pub const DEFAULT_LOCAL_DIR: &str = "weights/tts/kokoro-82m";

/// Audio sample rate produced by the model.
pub const SAMPLE_RATE: u32 = 24_000;

/// Resolved paths for one Kokoro checkpoint directory.
#[derive(Debug, Clone)]
pub struct ModelLayout {
    pub dir: PathBuf,
    pub onnx: PathBuf,
    pub tokenizer: PathBuf,
    pub voices_dir: PathBuf,
}

impl ModelLayout {
    /// Resolve the layout, selecting the ONNX variant named `model_file`
    /// (e.g. `model.onnx`, `model_fp16.onnx`, `model_q8f16.onnx`).
    pub fn resolve(model_dir: &Path, model_file: &str) -> Result<Self> {
        let dir = model_dir
            .canonicalize()
            .unwrap_or_else(|_| model_dir.to_path_buf());

        // ONNX file may live under onnx/ or directly in the dir.
        let onnx = [dir.join("onnx").join(model_file), dir.join(model_file)]
            .into_iter()
            .find(|p| p.is_file())
            .with_context(|| {
                format!(
                    "ONNX model '{model_file}' not found under {} (onnx/ or root)",
                    dir.display()
                )
            })?;

        let tokenizer = dir.join("tokenizer.json");
        if !tokenizer.is_file() {
            bail!("tokenizer.json missing: {}", tokenizer.display());
        }

        let voices_dir = [dir.join("voices"), dir.clone()]
            .into_iter()
            .find(|p| has_voice_bins(p))
            .with_context(|| format!("no voices/*.bin found under {}", dir.display()))?;

        Ok(Self {
            dir,
            onnx,
            tokenizer,
            voices_dir,
        })
    }
}

fn has_voice_bins(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .any(|e| e.path().extension().and_then(|x| x.to_str()) == Some("bin"))
}

/// Map a Kokoro voice name to its espeak-ng language tag from the name prefix.
///
/// `af_/am_` → American English, `bf_/bm_` → British English, and the other
/// language prefixes (`e` Spanish, `f` French, `h` Hindi, `i` Italian,
/// `j` Japanese, `p` Portuguese, `z` Mandarin). The second character (`f`/`m`)
/// is the speaker gender and does not affect the language.
pub fn voice_lang(voice: &str) -> &'static str {
    match voice.as_bytes().first().copied().map(|b| b as char) {
        Some('a') => "en-us",
        Some('b') => "en-gb",
        Some('e') => "es",
        Some('f') => "fr-fr",
        Some('h') => "hi",
        Some('i') => "it",
        Some('j') => "ja",
        Some('p') => "pt-br",
        Some('z') => "cmn",
        _ => "en-us",
    }
}

/// Whether a voice uses (bundled) English espeak data.
pub fn is_english_voice(voice: &str) -> bool {
    matches!(voice_lang(voice), "en-us" | "en-gb")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lang_from_prefix() {
        assert_eq!(voice_lang("af_heart"), "en-us");
        assert_eq!(voice_lang("am_michael"), "en-us");
        assert_eq!(voice_lang("bf_emma"), "en-gb");
        assert_eq!(voice_lang("bm_george"), "en-gb");
        assert_eq!(voice_lang("jf_alpha"), "ja");
        assert!(is_english_voice("af_heart"));
        assert!(!is_english_voice("jf_alpha"));
    }
}
