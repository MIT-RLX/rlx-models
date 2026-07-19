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

//! MiraTTS token layout + prompt builder (STEP 1 of the port — reverse-engineered
//! from `FastBiCodec` (`ncodec`) `codec.py` + the model's `added_tokens.json`).
//!
//! The Qwen2 LM's extended vocab (166 000) embeds the codec tokens inline:
//! - `<|context_token_N|>` — the **semantic / reference** codebook (4096 codes),
//!   `id = CONTEXT_BASE + N`. Produced by the LinaCodec/BiCodec **encoder** from
//!   the reference clip and placed in the prompt (voice cloning).
//! - `<|speech_token_N|>` — the **acoustic** codebook (8192 codes),
//!   `id = SPEECH_BASE + N`. **Generated** autoregressively by the LM.
//!
//! Reference `format_prompt` (FastBiCodec `codec.py`):
//! ```text
//! <|task_tts|><|start_text|>{text}<|end_text|>
//! <|context_audio_start|>{context_tokens}<|context_audio_end|>
//! <|prompt_speech_start|>
//! ```
//! then the LM emits `<|speech_token_N|>*` until `eos` / `<|end_acoustic_token|>`.
//! Decoding parses the acoustic codes (`id - SPEECH_BASE`) + the context codes
//! and feeds them to the BiCodec decoder.

/// `<|context_token_0|>` id — semantic/reference codebook base. `id = base + code`.
pub const CONTEXT_BASE: u32 = 151_665;
/// Number of entries in the semantic/context codebook.
pub const CONTEXT_CODEBOOK: u32 = 4096;
/// `<|speech_token_0|>` id — acoustic codebook base (LM-generated). `id = base + code`.
pub const SPEECH_BASE: u32 = 155_761;
/// Number of entries in the acoustic codebook.
pub const SPEECH_CODEBOOK: u32 = 8192;

// ── Structural / control token ids (from added_tokens.json) ─────────────────
pub const TASK_TTS: u32 = 165_137;
pub const START_TEXT: u32 = 165_146;
pub const END_TEXT: u32 = 165_152;
pub const CONTEXT_AUDIO_START: u32 = 165_150;
pub const CONTEXT_AUDIO_END: u32 = 165_156;
pub const PROMPT_SPEECH_START: u32 = 165_151;
pub const START_ACOUSTIC_TOKEN: u32 = 165_149;
pub const END_ACOUSTIC_TOKEN: u32 = 165_155;
pub const END_SEMANTIC_TOKEN: u32 = 165_157;
/// `<|im_end|>` — the generation eos (generation_config `eos_token_id`).
pub const EOS: u32 = 151_645;

/// Map a semantic/context code (0..4096) to its LM vocab id.
#[inline]
pub fn context_id(code: u32) -> u32 {
    CONTEXT_BASE + code
}

/// Map an acoustic/speech code (0..8192) to its LM vocab id.
#[inline]
pub fn speech_id(code: u32) -> u32 {
    SPEECH_BASE + code
}

/// Build the LM prompt token-id sequence for text-to-speech, mirroring
/// FastBiCodec `TTSCodec.format_prompt`. `text_ids` are the Qwen2 BPE ids of the
/// input text; `context_codes` are the semantic codes from the reference clip.
pub fn build_tts_prompt(text_ids: &[u32], context_codes: &[u32]) -> Vec<u32> {
    let mut p = Vec::with_capacity(text_ids.len() + context_codes.len() + 6);
    p.push(TASK_TTS);
    p.push(START_TEXT);
    p.extend_from_slice(text_ids);
    p.push(END_TEXT);
    p.push(CONTEXT_AUDIO_START);
    p.extend(context_codes.iter().map(|&c| context_id(c)));
    p.push(CONTEXT_AUDIO_END);
    p.push(PROMPT_SPEECH_START);
    p
}

/// Extract the acoustic codes from a generated id stream: keep ids in the
/// acoustic range, subtract the base, stop at eos / `<|end_acoustic_token|>`.
pub fn parse_speech_codes(generated: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    for &id in generated {
        if id == EOS || id == END_ACOUSTIC_TOKEN {
            break;
        }
        if (SPEECH_BASE..SPEECH_BASE + SPEECH_CODEBOOK).contains(&id) {
            out.push(id - SPEECH_BASE);
        }
    }
    out
}

/// A generated id ends acoustic generation (eos / end-of-acoustic).
#[inline]
pub fn is_stop(id: u32) -> bool {
    id == EOS || id == END_ACOUSTIC_TOKEN
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_ranges_partition_the_extended_vocab() {
        // context [151665, 155761), speech [155761, 163953) — contiguous, no overlap.
        assert_eq!(context_id(0), 151_665);
        assert_eq!(context_id(CONTEXT_CODEBOOK - 1), 155_760);
        assert_eq!(speech_id(0), 155_761);
        assert_eq!(speech_id(SPEECH_CODEBOOK - 1), 163_952);
        assert_eq!(CONTEXT_BASE + CONTEXT_CODEBOOK, SPEECH_BASE);
    }

    #[test]
    fn prompt_has_the_reference_structure() {
        let text = [10u32, 20, 30];
        let ctx = [0u32, 5, 4095];
        let p = build_tts_prompt(&text, &ctx);
        assert_eq!(p[0], TASK_TTS);
        assert_eq!(p[1], START_TEXT);
        assert_eq!(&p[2..5], &text);
        assert_eq!(p[5], END_TEXT);
        assert_eq!(p[6], CONTEXT_AUDIO_START);
        assert_eq!(&p[7..10], &[151_665, 151_670, 155_760]); // context ids
        assert_eq!(p[10], CONTEXT_AUDIO_END);
        assert_eq!(p[11], PROMPT_SPEECH_START);
        assert_eq!(p.len(), 12);
    }

    #[test]
    fn parse_extracts_acoustic_codes_and_stops() {
        let stream = [
            speech_id(7),
            speech_id(100),
            speech_id(8191),
            EOS,
            speech_id(1),
        ];
        assert_eq!(parse_speech_codes(&stream), vec![7, 100, 8191]);
        // stops at end-of-acoustic too, ignores out-of-range noise
        let stream2 = [speech_id(3), 999_999, END_ACOUSTIC_TOKEN, speech_id(9)];
        assert_eq!(parse_speech_codes(&stream2), vec![3]);
    }
}
