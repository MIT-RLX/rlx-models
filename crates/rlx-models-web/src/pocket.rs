// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! Kyutai Pocket TTS in the browser: one language's model and a voice,
//! text in, 24 kHz mono samples out — on the CPU (SIMD), WebGPU or WebGL.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;

use crate::{Files, js_err, parse_backend};

struct Model {
    tts: rlx_pocket_tts::engine::PocketTts,
    voice: Option<rlx_pocket_tts::voice_state::VoiceState>,
}

/// Pocket TTS (`rlx-pocket-tts`): text → speech in one language.
#[wasm_bindgen]
pub struct PocketTts {
    // Shared so an async method can hold it across awaits.
    model: Rc<RefCell<Model>>,
    backend: String,
}

#[wasm_bindgen]
impl PocketTts {
    /// `language`: a bundled config (`"french"`, `"english"`…). Consumes
    /// `model.safetensors` and `tokenizer.model` from `files`. `backend`:
    /// `"cpu"` (default), `"webgpu"` (after `init_webgpu()`; then use the
    /// `*_async` methods), `"webgl"`, or a FlowLM step's and a decoder's
    /// apart: `"cpu+webgl"` (the step on the CPU, the decoder on WebGL —
    /// each where it is quickest in a browser).
    #[wasm_bindgen(constructor)]
    pub fn new(
        files: &mut Files,
        language: &str,
        backend: Option<String>,
    ) -> Result<PocketTts, JsError> {
        let (device, decoder) = match backend.as_deref().and_then(|b| b.split_once('+')) {
            Some((step, dec)) => (parse_backend(Some(step))?, Some(parse_backend(Some(dec))?)),
            None => (parse_backend(backend.as_deref())?, None),
        };
        let weights = files.take_file("model.safetensors")?;
        let tokenizer = files.take_file("tokenizer.model")?;
        let mut tts =
            rlx_pocket_tts::engine::PocketTts::from_bytes(language, weights, &tokenizer, device)
                .map_err(js_err)?;
        if let Some(d) = decoder.filter(|d| *d != device) {
            tts.set_decoder_device(d).map_err(js_err)?;
        }
        Ok(PocketTts {
            model: Rc::new(RefCell::new(Model { tts, voice: None })),
            backend: backend.unwrap_or_else(|| "cpu".into()),
        })
    }

    /// The backend it runs on, as given.
    #[wasm_bindgen(getter)]
    pub fn backend(&self) -> String {
        self.backend.clone()
    }

    /// The voice to speak in: one of the model's `embeddings/<name>.safetensors`.
    pub fn set_voice(&mut self, bytes: Vec<u8>) -> Result<(), JsError> {
        let mut m = self
            .model
            .try_borrow_mut()
            .map_err(|_| JsError::new("busy speaking"))?;
        m.voice = Some(m.tts.voice_from_bytes(bytes).map_err(js_err)?);
        Ok(())
    }

    /// `text` spoken, as mono samples at [`sample_rate`](Self::sample_rate).
    pub fn speak(&mut self, text: &str) -> Result<Vec<f32>, JsError> {
        let mut m = self
            .model
            .try_borrow_mut()
            .map_err(|_| JsError::new("busy speaking"))?;
        let Model { tts, voice } = &mut *m;
        let voice = voice
            .as_ref()
            .ok_or_else(|| JsError::new("set_voice first"))?;
        Ok(tts.generate(text, voice).map_err(js_err)?.samples)
    }

    /// [`speak`](Self::speak), calling `on_block(samples)` with each decoder
    /// block (about 0.64 s) as soon as it is made, so it can play while the
    /// rest is made. Returns the whole, as `speak` would.
    pub fn speak_streaming(
        &mut self,
        text: &str,
        on_block: &js_sys::Function,
    ) -> Result<Vec<f32>, JsError> {
        let mut m = self
            .model
            .try_borrow_mut()
            .map_err(|_| JsError::new("busy speaking"))?;
        let Model { tts, voice } = &mut *m;
        let voice = voice
            .as_ref()
            .ok_or_else(|| JsError::new("set_voice first"))?;
        let mut failed = None;
        let mut sink = |block: &[f32]| {
            if failed.is_none()
                && let Err(e) = on_block.call1(&JsValue::NULL, &js_sys::Float32Array::from(block))
            {
                failed = Some(e);
            }
        };
        let audio = tts
            .generate_streaming(text, voice, &mut sink)
            .map_err(js_err)?;
        if let Some(e) = failed {
            return Err(JsError::new(&format!("on_block: {e:?}")));
        }
        Ok(audio.samples)
    }

    /// [`speak_streaming`](Self::speak_streaming) awaiting each graph: what
    /// WebGPU needs (its results come back asynchronously). Resolves to the
    /// whole as a `Float32Array`.
    #[allow(clippy::await_holding_refcell_ref)]
    pub fn speak_streaming_async(
        &self,
        text: String,
        on_block: js_sys::Function,
    ) -> js_sys::Promise {
        let model = self.model.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let mut m = model
                .try_borrow_mut()
                .map_err(|_| JsValue::from_str("busy speaking"))?;
            let Model { tts, voice } = &mut *m;
            let voice = voice
                .as_ref()
                .ok_or_else(|| JsValue::from_str("set_voice first"))?;
            let mut failed = None;
            let mut sink = |block: &[f32]| {
                if failed.is_none()
                    && let Err(e) =
                        on_block.call1(&JsValue::NULL, &js_sys::Float32Array::from(block))
                {
                    failed = Some(e);
                }
            };
            let audio = tts
                .generate_streaming_async(&text, voice, &mut sink)
                .await
                .map_err(|e| JsValue::from_str(&format!("{e:#}")))?;
            if let Some(e) = failed {
                return Err(e);
            }
            Ok(js_sys::Float32Array::from(audio.samples.as_slice()).into())
        })
    }

    /// What the last sentences cost, a line each (graph compile, prompt,
    /// frames, decoder), and clears it: with `set_setting(
    /// "RLX_POCKET_TTS_TIMING", "1")`, else empty.
    #[wasm_bindgen(getter)]
    pub fn timing(&self) -> String {
        rlx_pocket_tts::last_timing()
    }

    #[wasm_bindgen(getter)]
    pub fn sample_rate(&self) -> u32 {
        self.model.borrow().tts.sample_rate()
    }
}
