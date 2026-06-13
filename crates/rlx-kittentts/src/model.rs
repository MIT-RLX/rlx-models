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

//! ONNX model runner — mirrors Python's `KittenTTS_1_Onnx`.
//!
//! Uses [`ort`] (ONNX Runtime Rust bindings) for inference.
//! The three model inputs are:
//!
//! | Name        | Shape         | dtype   |
//! |-------------|---------------|---------|
//! | `input_ids` | `[1, seq_len]`| int64   |
//! | `style`     | `[1, style_d]`| float32 |
//! | `speed`     | `[1]`         | float32 |

use std::{collections::HashMap, path::Path};

use anyhow::{Context, Result};
use rlx_runtime::Device;

#[cfg(feature = "onnx")]
use ort::value::Tensor;

use crate::{
    assets::ModelLayout,
    backend_kind::BackendKind,
    npz::{NpyArray, load_npz},
    tokenize::{ipa_content_len, ipa_style_index, ipa_text_style_index, ipa_to_ids, warn_unknown_ipa_chars},
};

#[cfg(feature = "onnx")]
use crate::backend::build_onnx_session;

/// Samples trimmed from the tail of every generated chunk to remove the model's
/// trailing silence artifact. Override with `KITTENTTS_TAIL_TRIM`.
const DEFAULT_TAIL_TRIM: usize = 2_000;

/// Crossfade length when concatenating espeak chunks (samples at 24 kHz).
#[cfg(feature = "espeak")]
const CHUNK_CROSSFADE: usize = 240;

/// Max UTF-8 bytes per espeak → infer chunk (matches kittentts-rs).
#[cfg(feature = "espeak")]
const CHUNK_MAX_CHARS: usize = 400;

/// Audio sample rate produced by the model.
pub const SAMPLE_RATE: u32 = 24_000;

/// Peak amplitude below this is treated as silent output (post tail-trim).
pub const MIN_AUDIBLE_PEAK: f32 = 1e-3;

pub fn peak_amplitude(audio: &[f32]) -> f32 {
    audio
        .iter()
        .filter(|s| s.is_finite())
        .map(|s| s.abs())
        .fold(0.0f32, f32::max)
}

struct Voice {
    nrows: usize,
    ncols: usize,
    data: Vec<f32>,
}

impl Voice {
    fn from_npy(arr: NpyArray) -> Self {
        Self {
            nrows: arr.nrows(),
            ncols: arr.ncols(),
            data: arr.data,
        }
    }

    fn style_row(&self, text_len: usize) -> &[f32] {
        let i = text_len.min(self.nrows.saturating_sub(1));
        &self.data[i * self.ncols..(i + 1) * self.ncols]
    }
}

/// The main KittenTTS handle (ONNX Runtime and/or native RLX graph).
pub struct KittenTTS {
    backend: BackendKind,
    device: Device,
    voices: HashMap<String, Voice>,
    speed_priors: HashMap<String, f32>,
    voice_aliases: HashMap<String, String>,
    pub available_voices: Vec<String>,
}

impl KittenTTS {
    /// Load on CPU (default).
    pub fn load(
        model_path: &Path,
        voices_path: &Path,
        speed_priors: HashMap<String, f32>,
        voice_aliases: HashMap<String, String>,
    ) -> Result<Self> {
        Self::load_on(
            model_path,
            voices_path,
            speed_priors,
            voice_aliases,
            Device::Cpu,
        )
    }

    /// Load ONNX model + voices from a checkpoint directory (`config.json` + files).
    pub fn load_from_dir(model_dir: &Path, device: Device) -> Result<Self> {
        let layout = ModelLayout::resolve(model_dir)?;
        Self::load_on(
            &layout.onnx,
            &layout.voices,
            layout.config.speed_priors.clone(),
            layout.config.voice_aliases.clone(),
            device,
        )
    }

    /// Load decomposed RLX weights + voices from a checkpoint directory.
    #[cfg(feature = "native")]
    pub fn load_native_from_dir(
        model_dir: &Path,
        device: Device,
        sequence_length: usize,
        max_waveform_samples: usize,
    ) -> Result<Self> {
        let layout = ModelLayout::resolve(model_dir)?;
        let weights = layout.native_weights.as_ref().with_context(|| {
            format!(
                "native weights not found for {} (set KITTEN_RLX_WEIGHTS or place model.safetensors)",
                model_dir.display()
            )
        })?;
        Self::load_native(
            weights,
            &layout.voices,
            layout.config.speed_priors.clone(),
            layout.config.voice_aliases.clone(),
            device,
            sequence_length,
            max_waveform_samples,
        )
    }

    /// Load the model on a specific RLX execution device.
    ///
    /// ONNX inference uses the matching ONNX Runtime execution provider
    /// (CoreML, CUDA, ROCm, DirectML, …) with CPU as fallback.
    pub fn load_on(
        model_path: &Path,
        voices_path: &Path,
        speed_priors: HashMap<String, f32>,
        voice_aliases: HashMap<String, String>,
        device: Device,
    ) -> Result<Self> {
        #[cfg(not(feature = "onnx"))]
        {
            let _ = (model_path, device);
            anyhow::bail!("rlx-kittentts built without `onnx` feature");
        }
        #[cfg(feature = "onnx")]
        {
            let built = build_onnx_session(model_path, device)?;
            Self::from_parts(
                BackendKind::Onnx {
                    session: std::sync::Mutex::new(built.session),
                    ort_ep: built.ort_ep,
                },
                device,
                voices_path,
                speed_priors,
                voice_aliases,
            )
        }
    }

    /// Load the decomposed RLX graph from `weights_dir` (`model.safetensors` inside).
    ///
    /// `sequence_length` must be ≥ the longest IPA token sequence you will synthesize
    /// (graph shapes are fixed at compile time).
    #[cfg(feature = "native")]
    pub fn load_native(
        weights_dir: &Path,
        voices_path: &Path,
        speed_priors: HashMap<String, f32>,
        voice_aliases: HashMap<String, String>,
        device: Device,
        sequence_length: usize,
        max_waveform_samples: usize,
    ) -> Result<Self> {
        let engine = crate::native::NativeEngine::load(
            weights_dir,
            device,
            sequence_length,
            max_waveform_samples,
        )?;
        Self::from_parts(
            BackendKind::Native(engine),
            device,
            voices_path,
            speed_priors,
            voice_aliases,
        )
    }

    fn from_parts(
        backend: BackendKind,
        device: Device,
        voices_path: &Path,
        speed_priors: HashMap<String, f32>,
        voice_aliases: HashMap<String, String>,
    ) -> Result<Self> {
        let raw = load_npz(voices_path)
            .with_context(|| format!("Cannot load voices: {}", voices_path.display()))?;

        let available_voices: Vec<String> = raw.keys().cloned().collect();
        let voices: HashMap<String, Voice> = raw
            .into_iter()
            .map(|(k, v)| (k, Voice::from_npy(v)))
            .collect();

        eprintln!(
            "[kittentts] loaded on {device:?} (backend={})",
            backend.backend_label()
        );

        Ok(Self {
            backend,
            device,
            voices,
            speed_priors,
            voice_aliases,
            available_voices,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn ort_ep(&self) -> &str {
        self.backend.backend_label()
    }

    /// Compiled sequence length when using the `native` backend; `None` for ONNX.
    #[cfg(feature = "native")]
    pub fn native_sequence_length(&self) -> Option<usize> {
        match &self.backend {
            BackendKind::Native(e) => Some(e.sequence_length),
            #[cfg(feature = "onnx")]
            _ => None,
        }
    }

    pub fn resolve_voice<'a>(&'a self, voice: &'a str) -> &'a str {
        self.voice_aliases
            .get(voice)
            .map(String::as_str)
            .unwrap_or(voice)
    }

    pub fn effective_speed(&self, voice_key: &str, speed: f32) -> f32 {
        speed * self.speed_priors.get(voice_key).copied().unwrap_or(1.0)
    }

    pub fn voice_names(&self) -> &[String] {
        &self.available_voices
    }

    pub fn has_voice(&self, voice_key: &str) -> bool {
        self.voices.contains_key(voice_key)
    }

    /// Core inference step: IPA string → audio samples.
    pub fn infer_ipa(
        &self,
        ipa: &str,
        style_idx: usize,
        voice_key: &str,
        effective_speed: f32,
    ) -> Result<Vec<f32>> {
        let voice_data = self.voices.get(voice_key).with_context(|| {
            format!(
                "Voice '{}' not found. Available: {:?}",
                voice_key, self.available_voices
            )
        })?;

        if ipa_content_len(ipa) == 0 {
            anyhow::bail!(
                "IPA input tokenized to no phoneme characters (got only pad tokens). \
                 Pass IPA symbols (e.g. həˈloʊ), not arbitrary Unicode text."
            );
        }
        warn_unknown_ipa_chars(ipa);

        let ids = ipa_to_ids(ipa);
        let style_slice = voice_data.style_row(style_idx);

        let audio_flat = match &self.backend {
            #[cfg(feature = "onnx")]
            BackendKind::Onnx { session, .. } => {
                let seq_len = ids.len();
                let style_dim = style_slice.len();
                let t_input_ids = Tensor::<i64>::from_array(([1usize, seq_len], ids))
                    .context("Failed to build input_ids tensor")?;
                let t_style =
                    Tensor::<f32>::from_array(([1usize, style_dim], style_slice.to_vec()))
                        .context("Failed to build style tensor")?;
                let t_speed = Tensor::<f32>::from_array(([1usize], vec![effective_speed]))
                    .context("Failed to build speed tensor")?;

                let mut session = session.lock().expect("ORT session mutex poisoned");
                let outputs = session
                    .run(ort::inputs![t_input_ids, t_style, t_speed])
                    .context("ONNX inference failed")?;

                let (_shape, audio_data) = outputs[0]
                    .try_extract_tensor::<f32>()
                    .context("Failed to extract audio tensor")?;
                audio_data.to_vec()
            }
            #[cfg(feature = "native")]
            BackendKind::Native(engine) => engine.infer(&ids, style_slice, effective_speed)?,
        };

        let trimmed_len = if audio_flat.len() > tail_trim() {
            audio_flat.len() - tail_trim()
        } else {
            audio_flat.len()
        };
        let audio = audio_flat[..trimmed_len].to_vec();
        let peak = peak_amplitude(&audio);
        if peak < MIN_AUDIBLE_PEAK {
            anyhow::bail!(
                "synthesized audio is effectively silent (peak={peak:.2e}). \
                 Check that --ipa uses IPA phoneme symbols."
            );
        }
        Ok(audio)
    }

    /// Synthesize plain text via espeak phonemization (`espeak` feature).
    ///
    /// Long input is split into sentence/word chunks (max ~400 chars). Applies
    /// [`preprocess::TextPreprocessor`](crate::preprocess::TextPreprocessor) before
    /// phonemization. Style row index uses each chunk's UTF-8 byte length (kittentts-rs).
    #[cfg(feature = "espeak")]
    pub fn generate_from_text(
        &self,
        text: &str,
        voice: &str,
        speed: f32,
        lang: &str,
    ) -> Result<Vec<f32>> {
        self.generate_from_text_with_options(text, voice, speed, lang, true)
    }

    /// Plain text → espeak without the preprocessor (raw user string).
    #[cfg(feature = "espeak")]
    pub fn generate_from_text_raw(
        &self,
        text: &str,
        voice: &str,
        speed: f32,
        lang: &str,
    ) -> Result<Vec<f32>> {
        self.generate_from_text_with_options(text, voice, speed, lang, false)
    }

    #[cfg(feature = "espeak")]
    pub fn generate_from_text_with_options(
        &self,
        text: &str,
        voice: &str,
        speed: f32,
        lang: &str,
        clean_text: bool,
    ) -> Result<Vec<f32>> {
        use crate::phonemize::phonemize_lang;

        let voice_key = self.resolve_voice(voice);
        if !self.has_voice(voice_key) {
            anyhow::bail!(
                "Unknown voice '{voice}'. Available: {:?}",
                self.available_voices
            );
        }

        let processed = if clean_text {
            crate::preprocess::TextPreprocessor::new().process(text)
        } else {
            text.to_string()
        };

        let chunks = chunk_text(&processed, CHUNK_MAX_CHARS);
        if chunks.is_empty() {
            anyhow::bail!("text input is empty");
        }

        let mut audio = Vec::new();
        for chunk in chunks {
            let ipa = phonemize_lang(lang, &chunk)?;
            if ipa_content_len(&ipa) == 0 {
                anyhow::bail!("espeak produced no IPA for chunk {chunk:?} (lang={lang})");
            }
            let style = ipa_text_style_index(&chunk);
            let chunk_audio = self.generate_from_ipa(&ipa, voice, speed, style)?;
            crossfade_extend(&mut audio, &chunk_audio);
        }
        Ok(audio)
    }

    pub fn generate_from_ipa(
        &self,
        ipa: &str,
        voice: &str,
        speed: f32,
        style_idx: usize,
    ) -> Result<Vec<f32>> {
        let voice_key = self.resolve_voice(voice);
        let effective_speed = self.effective_speed(voice_key, speed);
        self.infer_ipa(ipa, style_idx, voice_key, effective_speed)
    }

    pub fn generate_from_ipa_chunks(
        &self,
        chunks: &[&str],
        voice: &str,
        speed: f32,
    ) -> Result<Vec<f32>> {
        let voice_key = self.resolve_voice(voice);
        if !self.has_voice(voice_key) {
            anyhow::bail!(
                "Unknown voice '{}'. Available: {:?}",
                voice,
                self.available_voices
            );
        }
        let mut audio = Vec::new();
        for &ipa in chunks {
            audio.extend(self.generate_from_ipa(ipa, voice, speed, ipa_style_index(ipa))?);
        }
        Ok(audio)
    }

    pub fn generate_to_file_from_ipa(
        &self,
        ipa: &str,
        output_path: &Path,
        voice: &str,
        speed: f32,
        style_idx: usize,
    ) -> Result<()> {
        let audio = self.generate_from_ipa(ipa, voice, speed, style_idx)?;
        self.write_wav(&audio, output_path)
    }

    pub fn write_wav(&self, audio: &[f32], output_path: &Path) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(output_path, spec)
            .with_context(|| format!("Cannot create WAV: {}", output_path.display()))?;
        for &s in audio {
            let s16 = (s * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            writer.write_sample(s16).context("WAV write error")?;
        }
        writer.finalize().context("WAV finalise error")?;
        println!(
            "Saved {} samples ({} s) to {}",
            audio.len(),
            audio.len() as f32 / SAMPLE_RATE as f32,
            output_path.display()
        );
        Ok(())
    }
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

fn tail_trim() -> usize {
    std::env::var("KITTENTTS_TAIL_TRIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TAIL_TRIM)
}

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
        let tail_idx = dst.len() - fade + i;
        dst[tail_idx] = dst[tail_idx] * (1.0 - t) + chunk[i] * t;
    }
    dst.extend_from_slice(&chunk[fade..]);
}

#[cfg(all(test, feature = "espeak"))]
mod chunk_tests {
    use super::*;

    #[test]
    fn chunk_short_sentence() {
        let c = chunk_text("Hello world.", 400);
        assert_eq!(c, vec!["Hello world,"]);
    }
}
