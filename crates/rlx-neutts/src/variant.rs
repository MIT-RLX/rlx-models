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

//! NeuTTS model variants. `Air` is the base Neuphonic model; `VieNeu` is the
//! Vietnamese fine-tune (VieNeu-TTS) — same GGUF-Llama backbone + NeuCodec, but
//! trained for Vietnamese and paired with a Vietnamese IPA frontend. The variant
//! only differs in language + weights, so it rides the existing native pipeline.

/// Which NeuTTS variant / language to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NeuTtsVariant {
    /// NeuTTS-Air — the base Neuphonic model.
    #[default]
    Air,
    /// VieNeu-TTS — the Vietnamese fine-tune.
    VieNeu,
}

impl NeuTtsVariant {
    /// Primary language (ISO 639-1).
    pub fn language(self) -> &'static str {
        match self {
            NeuTtsVariant::Air => "en",
            NeuTtsVariant::VieNeu => "vi",
        }
    }

    /// Hosted weights repo.
    pub fn hf_repo(self) -> &'static str {
        match self {
            NeuTtsVariant::Air => "neuphonic/neutts-air",
            NeuTtsVariant::VieNeu => "pnnbao-ump/VieNeu-TTS",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            NeuTtsVariant::Air => "NeuTTS-Air",
            NeuTtsVariant::VieNeu => "VieNeu-TTS",
        }
    }

    /// All NeuTTS variants are voice-cloning (reference audio + transcript).
    pub fn supports_voice_cloning(self) -> bool {
        true
    }

    pub fn is_vietnamese(self) -> bool {
        matches!(self, NeuTtsVariant::VieNeu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn languages_differ() {
        assert_eq!(NeuTtsVariant::Air.language(), "en");
        assert_eq!(NeuTtsVariant::VieNeu.language(), "vi");
        assert_eq!(NeuTtsVariant::default(), NeuTtsVariant::Air);
    }

    #[test]
    fn vietnamese_flag_and_distinct_repos() {
        assert!(NeuTtsVariant::VieNeu.is_vietnamese());
        assert!(!NeuTtsVariant::Air.is_vietnamese());
        assert_ne!(
            NeuTtsVariant::Air.hf_repo(),
            NeuTtsVariant::VieNeu.hf_repo()
        );
        assert!(NeuTtsVariant::VieNeu.supports_voice_cloning());
        assert_ne!(
            NeuTtsVariant::Air.display_name(),
            NeuTtsVariant::VieNeu.display_name()
        );
    }
}
