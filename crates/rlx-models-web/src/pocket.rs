// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! Kyutai Pocket TTS in the browser: one language's model and a voice,
//! text in, 24 kHz mono samples out (CPU).

use wasm_bindgen::prelude::*;

use crate::{Files, js_err};

/// Pocket TTS (`rlx-pocket-tts`): text → speech in one language.
#[wasm_bindgen]
pub struct PocketTts {
    tts: rlx_pocket_tts::engine::PocketTts,
    voice: Option<rlx_pocket_tts::voice_state::VoiceState>,
}

#[wasm_bindgen]
impl PocketTts {
    /// `language`: a bundled config (`"french"`, `"english"`…). Consumes
    /// `model.safetensors` and `tokenizer.model` from `files`.
    #[wasm_bindgen(constructor)]
    pub fn new(files: &mut Files, language: &str) -> Result<PocketTts, JsError> {
        let weights = files.take_file("model.safetensors")?;
        let tokenizer = files.take_file("tokenizer.model")?;
        let tts = rlx_pocket_tts::engine::PocketTts::from_bytes(
            language,
            weights,
            &tokenizer,
            rlx_runtime::Device::Cpu,
        )
        .map_err(js_err)?;
        Ok(PocketTts { tts, voice: None })
    }

    /// The voice to speak in: one of the model's `embeddings/<name>.safetensors`.
    pub fn set_voice(&mut self, bytes: Vec<u8>) -> Result<(), JsError> {
        self.voice = Some(self.tts.voice_from_bytes(bytes).map_err(js_err)?);
        Ok(())
    }

    /// `text` spoken, as mono samples at [`sample_rate`](Self::sample_rate).
    pub fn speak(&mut self, text: &str) -> Result<Vec<f32>, JsError> {
        let voice = self
            .voice
            .as_ref()
            .ok_or_else(|| JsError::new("set_voice first"))?;
        Ok(self.tts.generate(text, voice).map_err(js_err)?.samples)
    }

    #[wasm_bindgen(getter)]
    pub fn sample_rate(&self) -> u32 {
        self.tts.sample_rate()
    }
}
