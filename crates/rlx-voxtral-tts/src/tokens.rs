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

//! Audio + text special tokens (vLLM-Omni `AudioSpecialTokens`).

pub const AUDIO_TOKEN_OFFSET: u32 = 2;
pub const EMPTY_AUDIO: u32 = 0;
pub const END_AUDIO: u32 = 1;

pub const DEFAULT_CFG_ALPHA: f32 = 1.2;
pub const DEFAULT_EULER_STEPS: usize = 7;

/// Preset voices shipped under `voice_embedding/` on HuggingFace.
pub const PRESET_VOICES: &[&str] = &[
    "casual_female",
    "casual_male",
    "cheerful_female",
    "neutral_female",
    "neutral_male",
    "pt_male",
    "pt_female",
    "nl_male",
    "nl_female",
    "it_male",
    "it_female",
    "fr_male",
    "fr_female",
    "es_male",
    "es_female",
    "de_male",
    "de_female",
    "ar_male",
    "hi_male",
    "hi_female",
];

pub fn strip_acoustic_offset(codes: &mut [u32]) {
    for c in codes.iter_mut() {
        if *c >= AUDIO_TOKEN_OFFSET {
            *c -= AUDIO_TOKEN_OFFSET;
        }
    }
}

/// Split flat vLLM frames into semantic (raw) + acoustic (offset stripped).
pub fn split_voxtral_frames(codes: &[u32], n_frames: usize) -> (Vec<usize>, Vec<u32>, usize) {
    let mut semantic = Vec::with_capacity(n_frames);
    let mut acoustic = Vec::with_capacity(n_frames * 36);
    let mut actual_frames = 0usize;
    for fi in 0..n_frames {
        let sem = codes[fi * 37];
        if sem == END_AUDIO {
            break;
        }
        semantic.push(sem.saturating_sub(AUDIO_TOKEN_OFFSET) as usize);
        for ai in 0..36 {
            let v = codes[fi * 37 + 1 + ai];
            acoustic.push(v.saturating_sub(AUDIO_TOKEN_OFFSET));
        }
        actual_frames += 1;
    }
    (semantic, acoustic, actual_frames)
}

pub fn is_end_of_audio_semantic(semantic: u32) -> bool {
    semantic == END_AUDIO
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_strips_semantic_offset_for_max_codebook() {
        let mut codes = vec![2u32; 37];
        codes[0] = 8193; // codebook index 8191 with +2 offset
        let (sem, _, n) = split_voxtral_frames(&codes, 1);
        assert_eq!(n, 1);
        assert_eq!(sem, vec![8191]);
    }
}
