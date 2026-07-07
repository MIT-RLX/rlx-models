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

//! Kokoro-82M runner (StyleTTS2 + ISTFTNet) over ONNX Runtime.
//!
//! The exported graph takes three inputs and returns a mono waveform:
//!
//! | Name        | Shape           | dtype   |
//! |-------------|-----------------|---------|
//! | `input_ids` | `[1, seq_len]`  | int64   |
//! | `style`     | `[1, 256]`      | float32 |
//! | `speed`     | `[1]`           | float32 |
//! | `waveform`  | `[1, n_samples]`| float32 |

use std::path::Path;

use anyhow::{Context, Result};
use rlx_runtime::Device;

use crate::config::{ModelLayout, SAMPLE_RATE};
use crate::tokenize::Vocab;
use crate::voices::VoiceBank;

#[cfg(feature = "onnx")]
use std::sync::Mutex;
#[cfg(feature = "onnx")]
use ort::{session::Session, value::Tensor};

/// Peak amplitude below this is treated as silent (failed) output.
pub const MIN_AUDIBLE_PEAK: f32 = 1e-3;

/// Max phoneme characters per synthesis chunk before sentence splitting.
#[cfg(feature = "espeak")]
const CHUNK_MAX_CHARS: usize = 400;

/// Crossfade length (samples at 24 kHz) when joining chunks.
#[cfg(feature = "espeak")]
const CHUNK_CROSSFADE: usize = 240;

/// Peak absolute amplitude of a waveform.
pub fn peak_amplitude(audio: &[f32]) -> f32 {
    audio
        .iter()
        .filter(|s| s.is_finite())
        .map(|s| s.abs())
        .fold(0.0f32, f32::max)
}

/// A loaded Kokoro model.
pub struct Kokoro {
    device: Device,
    vocab: Vocab,
    voices: VoiceBank,
    #[cfg(feature = "onnx")]
    session: Mutex<Session>,
    ort_ep: String,
}

impl Kokoro {
    /// Load from a checkpoint directory on CPU, using `model.onnx`.
    pub fn load_from_dir(model_dir: &Path) -> Result<Self> {
        Self::load_on(model_dir, "model.onnx", Device::Cpu)
    }

    /// Load a specific ONNX variant on a specific device.
    ///
    /// `model_file` is the ONNX filename (e.g. `model.onnx`, `model_fp16.onnx`,
    /// `model_q8f16.onnx`). ONNX inference runs on the matching ONNX Runtime
    /// execution provider (CoreML/CUDA/DirectML) with a CPU fallback.
    pub fn load_on(model_dir: &Path, model_file: &str, device: Device) -> Result<Self> {
        let layout = ModelLayout::resolve(model_dir, model_file)?;
        let vocab = Vocab::load(&layout.tokenizer)?;
        let voices = VoiceBank::load_dir(&layout.voices_dir)?;

        #[cfg(not(feature = "onnx"))]
        {
            let _ = (device, &layout);
            anyhow::bail!("rlx-kokoro built without the `onnx` feature");
        }
        #[cfg(feature = "onnx")]
        {
            let built = rlx_kittentts::build_onnx_session(&layout.onnx, device)
                .context("build Kokoro ONNX session")?;
            eprintln!(
                "[kokoro] loaded {} voices on {device:?} (ort_ep={})",
                voices.len(),
                built.ort_ep
            );
            Ok(Self {
                device,
                vocab,
                voices,
                session: Mutex::new(built.session),
                ort_ep: built.ort_ep,
            })
        }
    }

    /// Execution device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// ONNX Runtime execution provider label that actually loaded the model.
    pub fn ort_ep(&self) -> &str {
        &self.ort_ep
    }

    /// Model sample rate (24 kHz).
    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Sorted list of available voice names.
    pub fn voice_names(&self) -> Vec<String> {
        self.voices.names()
    }

    /// Whether a voice exists.
    pub fn has_voice(&self, voice: &str) -> bool {
        self.voices.get(voice).is_some()
    }

    /// The phoneme vocabulary.
    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    /// Synthesize directly from a phoneme string (skips text normalization/G2P).
    ///
    /// `speed` scales duration: `1.0` = natural, `>1.0` faster, `<1.0` slower.
    pub fn infer_phonemes(&self, phonemes: &str, voice: &str, speed: f32) -> Result<Vec<f32>> {
        let voice_data = self.voices.get(voice).with_context(|| {
            format!(
                "voice '{voice}' not found. Available: {:?}",
                self.voice_names()
            )
        })?;

        let content_len = self.vocab.content_len(phonemes);
        anyhow::ensure!(
            content_len > 0,
            "phoneme input produced no in-vocabulary tokens: {phonemes:?}"
        );
        let unknown = self.vocab.unknown_chars(phonemes);
        if !unknown.is_empty() {
            eprintln!("[kokoro] warning: dropped {} unknown phoneme chars: {unknown:?}", unknown.len());
        }

        let ids = self.vocab.to_input_ids(phonemes);
        let style = voice_data.style_row(content_len).to_vec();

        let audio = self.run_onnx(&ids, &style, speed)?;

        let peak = peak_amplitude(&audio);
        anyhow::ensure!(
            peak >= MIN_AUDIBLE_PEAK,
            "synthesized audio is effectively silent (peak={peak:.2e})"
        );
        Ok(audio)
    }

    #[cfg(feature = "onnx")]
    fn run_onnx(&self, ids: &[i64], style: &[f32], speed: f32) -> Result<Vec<f32>> {
        let seq_len = ids.len();
        let style_dim = style.len();
        let t_input_ids = Tensor::<i64>::from_array(([1usize, seq_len], ids.to_vec()))
            .context("build input_ids tensor")?;
        let t_style = Tensor::<f32>::from_array(([1usize, style_dim], style.to_vec()))
            .context("build style tensor")?;
        let t_speed =
            Tensor::<f32>::from_array(([1usize], vec![speed])).context("build speed tensor")?;

        // Graph input order is [input_ids, style, speed]; single output `waveform`.
        let mut session = self.session.lock().expect("ORT session mutex poisoned");
        let outputs = session
            .run(ort::inputs![t_input_ids, t_style, t_speed])
            .context("Kokoro ONNX inference failed")?;

        let (_shape, audio) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("extract waveform tensor")?;
        Ok(audio.to_vec())
    }

    #[cfg(not(feature = "onnx"))]
    fn run_onnx(&self, _ids: &[i64], _style: &[f32], _speed: f32) -> Result<Vec<f32>> {
        anyhow::bail!("rlx-kokoro built without the `onnx` feature")
    }

    /// Synthesize from plain text (espeak-ng G2P + text normalization).
    ///
    /// Long text is split into sentence chunks and crossfaded. The espeak
    /// language is derived from the voice prefix (`af_/am_` → en-us,
    /// `bf_/bm_` → en-gb).
    #[cfg(feature = "espeak")]
    pub fn generate_from_text(&self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>> {
        let lang = crate::config::voice_lang(voice);
        self.generate_from_text_lang(text, voice, speed, lang)
    }

    /// Like [`generate_from_text`](Self::generate_from_text) with an explicit
    /// espeak language tag.
    #[cfg(feature = "espeak")]
    pub fn generate_from_text_lang(
        &self,
        text: &str,
        voice: &str,
        speed: f32,
        lang: &str,
    ) -> Result<Vec<f32>> {
        use rlx_kittentts::preprocess::TextPreprocessor;

        anyhow::ensure!(
            self.has_voice(voice),
            "voice '{voice}' not found. Available: {:?}",
            self.voice_names()
        );

        let processed = TextPreprocessor::new().process(text);
        let chunks = chunk_text(&processed, CHUNK_MAX_CHARS);
        anyhow::ensure!(!chunks.is_empty(), "text input is empty");

        let mut audio = Vec::new();
        for chunk in chunks {
            let ipa = phonemize_with_fallback(lang, &chunk)?;
            if self.vocab.content_len(&ipa) == 0 {
                continue;
            }
            let chunk_audio = self.infer_phonemes(&ipa, voice, speed)?;
            crossfade_extend(&mut audio, &chunk_audio);
        }
        anyhow::ensure!(!audio.is_empty(), "no audio produced for input text");
        Ok(audio)
    }

    /// Write mono 16-bit PCM WAV at 24 kHz.
    pub fn write_wav(&self, audio: &[f32], output_path: &Path) -> Result<()> {
        write_wav(audio, output_path)
    }
}

/// Write mono 16-bit PCM WAV at 24 kHz.
pub fn write_wav(audio: &[f32], output_path: &Path) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(output_path, spec)
        .with_context(|| format!("create WAV: {}", output_path.display()))?;
    for &s in audio {
        let s16 = (s * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        writer.write_sample(s16).context("WAV write")?;
    }
    writer.finalize().context("WAV finalize")?;
    Ok(())
}

#[cfg(feature = "espeak")]
fn ensure_punctuation(text: &str) -> String {
    let text = text.trim();
    if text.is_empty() {
        return text.to_string();
    }
    match text.chars().last() {
        Some(c) if ".!?,;:".contains(c) => text.to_string(),
        _ => format!("{text},"),
    }
}

/// Phonemize `text`, falling back through progressively more available espeak
/// language tables when the requested one is not installed. The bundled espeak
/// data only ships English (`en` / `en-us`), so e.g. `en-gb` degrades to
/// `en-us` rather than failing outright.
#[cfg(feature = "espeak")]
fn phonemize_with_fallback(primary: &str, text: &str) -> Result<String> {
    use rlx_kittentts::phonemize::phonemize_lang;

    let mut langs = vec![primary];
    for fallback in ["en-us", "en"] {
        if !langs.contains(&fallback) {
            langs.push(fallback);
        }
    }
    let mut last_err = None;
    for (i, lang) in langs.iter().enumerate() {
        match phonemize_lang(lang, text) {
            Ok(ipa) => {
                if i > 0 {
                    eprintln!("[kokoro] espeak language '{primary}' unavailable; used '{lang}'");
                }
                return Ok(ipa);
            }
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no espeak language available")))
        .with_context(|| format!("espeak phonemize failed for {text:?}"))
}

/// Split into sentence-ish chunks under `max_len` characters.
#[cfg(feature = "espeak")]
fn chunk_text(text: &str, max_len: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    for sentence in text.split_terminator(['.', '!', '?']) {
        let sentence = sentence.trim();
        if sentence.is_empty() {
            continue;
        }
        if sentence.len() <= max_len {
            chunks.push(ensure_punctuation(sentence));
        } else {
            let mut current = String::new();
            for word in sentence.split_whitespace() {
                if !current.is_empty() && current.len() + 1 + word.len() > max_len {
                    chunks.push(ensure_punctuation(current.trim()));
                    current = word.to_string();
                } else {
                    if !current.is_empty() {
                        current.push(' ');
                    }
                    current.push_str(word);
                }
            }
            if !current.trim().is_empty() {
                chunks.push(ensure_punctuation(current.trim()));
            }
        }
    }
    chunks
}

/// Append `chunk` to `dst`, crossfading the overlap to avoid clicks.
#[cfg(feature = "espeak")]
fn crossfade_extend(dst: &mut Vec<f32>, chunk: &[f32]) {
    if dst.is_empty() {
        dst.extend_from_slice(chunk);
        return;
    }
    let fade = CHUNK_CROSSFADE.min(dst.len()).min(chunk.len());
    if fade == 0 {
        dst.extend_from_slice(chunk);
        return;
    }
    for i in 0..fade {
        let t = i as f32 / fade as f32;
        let tail = dst.len() - fade + i;
        dst[tail] = dst[tail] * (1.0 - t) + chunk[i] * t;
    }
    dst.extend_from_slice(&chunk[fade..]);
}

#[cfg(all(test, feature = "espeak"))]
mod tests {
    use super::*;

    #[test]
    fn chunks_short_sentence() {
        assert_eq!(chunk_text("Hello world.", 400), vec!["Hello world,"]);
    }
}
