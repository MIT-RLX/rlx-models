// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! High-level Orpheus TTS runner — Llama GGUF backbone + SNAC decoder.

use std::path::Path;

use anyhow::{Context, Result};

use crate::decoder::{SAMPLE_RATE, SAMPLES_PER_FRAME, SnacBackend, decoder_weights_path};
use crate::tokens::{build_prompt_ids, build_voice_clone_prompt_ids, pack_orpheus_codes};

#[cfg(feature = "llama")]
use crate::backbone::{BackboneLoadOptions, BackboneModel, DEFAULT_N_CTX};
#[cfg(feature = "llama")]
use crate::device::OrpheusRuntimeDevice;
#[cfg(feature = "llama")]
use rlx_runtime::Device;

/// LM + decoder hyper-parameters (Orpheus defaults from upstream).
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub seed: u64,
    pub greedy: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 1200,
            temperature: 0.6,
            top_p: 0.8,
            top_k: 0,
            repetition_penalty: 1.3,
            seed: 42,
            greedy: false,
        }
    }
}

/// Synthesized mono PCM at [`SAMPLE_RATE`].
#[derive(Debug, Clone)]
pub struct SynthesisResult {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub code_count: usize,
    pub codes: Vec<i32>,
}

/// Orpheus TTS session: GGUF backbone + SNAC decoder.
pub struct OrpheusTts {
    #[cfg(feature = "llama")]
    pub backbone: BackboneModel,
    pub snac: SnacBackend,
    pub config: GenerationConfig,
}

impl OrpheusTts {
    #[cfg(feature = "llama")]
    pub fn load(backbone_path: &Path, snac_path: &Path) -> Result<Self> {
        Self::load_on(backbone_path, snac_path, Device::Cpu)
    }

    #[cfg(feature = "llama")]
    pub fn load_on(backbone_path: &Path, snac_path: &Path, device: Device) -> Result<Self> {
        Self::load_on_with(
            backbone_path,
            snac_path,
            OrpheusRuntimeDevice {
                lm: device,
                snac_coreml: false,
            },
            BackboneLoadOptions::for_device(device),
        )
    }

    #[cfg(feature = "llama")]
    pub fn load_on_with(
        backbone_path: &Path,
        snac_path: &Path,
        runtime: OrpheusRuntimeDevice,
        backbone_opts: BackboneLoadOptions,
    ) -> Result<Self> {
        let backbone =
            BackboneModel::load_on_with(backbone_path, DEFAULT_N_CTX, runtime.lm, backbone_opts)
                .context("load Orpheus GGUF backbone")?;
        let snac = SnacBackend::open(snac_path, runtime.snac_load_options())
            .context("load SNAC decoder")?;
        Ok(Self {
            backbone,
            snac,
            config: GenerationConfig::default(),
        })
    }

    #[cfg(feature = "llama")]
    pub fn load_on_with_device(
        backbone_path: &Path,
        snac_path: &Path,
        device: Device,
        backbone_opts: BackboneLoadOptions,
    ) -> Result<Self> {
        Self::load_on_with(
            backbone_path,
            snac_path,
            OrpheusRuntimeDevice {
                lm: device,
                snac_coreml: false,
            },
            backbone_opts,
        )
    }

    #[cfg(feature = "llama")]
    pub fn load_with_env_decoder(backbone_path: &Path) -> Result<Self> {
        let snac_path = decoder_weights_path()?;
        Self::load(backbone_path, &snac_path)
    }

    #[cfg(feature = "llama")]
    pub fn load_with_env_decoder_on(backbone_path: &Path, device: Device) -> Result<Self> {
        let snac_path = decoder_weights_path()?;
        Self::load_on(backbone_path, &snac_path, device)
    }

    /// Synthesize speech for `text` with an optional built-in voice prefix.
    #[cfg(feature = "llama")]
    pub fn synthesize(&self, text: &str, voice: Option<&str>) -> Result<SynthesisResult> {
        let body = match voice {
            Some(v) => format!("{v}: {text}"),
            None => text.to_string(),
        };
        let prompt_ids = build_prompt_ids(self.backbone.weights_path(), &body)?;
        self.synthesize_from_prompt_ids(&prompt_ids)
    }

    /// Zero-shot voice clone using a **pretrained** Orpheus checkpoint.
    ///
    /// `ref_text` must match what was spoken in the reference clip (use Whisper
    /// or a manual transcript). `ref_audio_token_ids` come from SNAC encoding —
    /// see [`crate::tokens::VoiceCloneReference`] and
    /// `scripts/orpheus_encode_reference.py`.
    #[cfg(feature = "llama")]
    pub fn synthesize_voice_clone(
        &self,
        ref_text: &str,
        ref_audio_token_ids: &[u32],
        target_text: &str,
    ) -> Result<SynthesisResult> {
        let prompt_ids = build_voice_clone_prompt_ids(
            self.backbone.weights_path(),
            ref_text,
            ref_audio_token_ids,
            target_text,
        )?;
        self.synthesize_from_prompt_ids(&prompt_ids)
    }

    /// Stream PCM chunks (~2048 samples / ~85 ms) as SNAC codes arrive (after 4 frames buffered).
    #[cfg(feature = "llama")]
    pub fn synthesize_stream(
        &self,
        text: &str,
        voice: Option<&str>,
        mut on_pcm: impl FnMut(&[f32]) -> Result<()>,
    ) -> Result<SynthesisResult> {
        let body = match voice {
            Some(v) => format!("{v}: {text}"),
            None => text.to_string(),
        };
        let prompt_ids = build_prompt_ids(self.backbone.weights_path(), &body)?;
        let mut codes = Vec::new();
        let mut pcm = Vec::new();

        self.backbone
            .generate_codes_from_prompt_streaming(&prompt_ids, &self.config, |code| {
                codes.push(code);
                if codes.len() >= 28 && codes.len() % 7 == 0 {
                    let chunk = decode_stream_chunk(&self.snac, &codes[codes.len() - 28..])?;
                    on_pcm(&chunk)?;
                    pcm.extend_from_slice(&chunk);
                }
                Ok(())
            })?;

        if codes.len() >= 7 {
            let samples = decode_orpheus_codes(&self.snac, &codes)?;
            if samples.len() > pcm.len() {
                let tail = &samples[pcm.len()..];
                on_pcm(tail)?;
                pcm.extend_from_slice(tail);
            }
        }

        Ok(SynthesisResult {
            samples: pcm,
            sample_rate: SAMPLE_RATE,
            code_count: codes.len(),
            codes,
        })
    }

    /// Synthesize from pre-built prompt token ids (incl. Orpheus control tokens).
    #[cfg(feature = "llama")]
    pub fn synthesize_from_prompt_ids(&self, prompt_ids: &[u32]) -> Result<SynthesisResult> {
        let codes = self
            .backbone
            .generate_codes_from_prompt(prompt_ids, &self.config)
            .context("LM code generation")?;

        let samples = self.decode_all_codes(&codes)?;
        Ok(SynthesisResult {
            samples,
            sample_rate: SAMPLE_RATE,
            code_count: codes.len(),
            codes,
        })
    }

    /// Decode SNAC codes to PCM. Uses Orpheus streaming overlap when enough frames exist.
    pub fn decode_all_codes(&self, codes: &[i32]) -> Result<Vec<f32>> {
        decode_orpheus_codes(&self.snac, codes)
    }

    pub fn decode_codes_to_pcm(&self, codes: &[i32]) -> Result<Vec<f32>> {
        self.decode_all_codes(codes)
    }
}

const WINDOW_FRAMES: usize = 4;
const WINDOW_CODES: usize = WINDOW_FRAMES * 7;

/// One streaming chunk: center 2048 samples from the latest 4-frame SNAC window.
pub fn decode_stream_chunk(snac: &SnacBackend, recent_28_codes: &[i32]) -> Result<Vec<f32>> {
    anyhow::ensure!(
        recent_28_codes.len() == WINDOW_CODES,
        "stream chunk expects {WINDOW_CODES} codes, got {}",
        recent_28_codes.len()
    );
    let audio = snac.decode_orpheus_frames(recent_28_codes)?;
    let mut chunk = Vec::new();
    append_center_pcm(&audio, &mut chunk);
    Ok(chunk)
}

/// Decode interleaved Orpheus SNAC code indices to mono PCM (no LM).
///
/// Returns the **full** SNAC waveform for all frames (batch / WAV export).
/// Streaming uses [`decode_stream_chunk`] (center 2048 samples per 4-frame window);
/// [`OrpheusTts::synthesize_stream`] appends the remaining tail from this function.
pub fn decode_orpheus_codes(snac: &SnacBackend, codes: &[i32]) -> Result<Vec<f32>> {
    if codes.is_empty() {
        return Ok(Vec::new());
    }

    let num_frames = codes.len() / 7;
    if num_frames == 0 {
        return Ok(Vec::new());
    }

    let aligned = &codes[..num_frames * 7];
    let (c0, c1, c2) = pack_orpheus_codes(aligned).ok_or_else(|| {
        anyhow::anyhow!(
            "invalid code layout ({} codes, first={:?})",
            aligned.len(),
            &aligned[..aligned.len().min(8)]
        )
    })?;
    snac.decode_codes(&c0, &c1, &c2)
}

/// Sliding-window starts for streaming (matches Python `tokens_decoder`).
pub fn streaming_window_starts(num_codes: usize) -> Vec<usize> {
    if num_codes < WINDOW_CODES {
        return Vec::new();
    }
    (0..=num_codes - WINDOW_CODES).step_by(7).collect()
}

fn append_center_pcm(audio: &[f32], pcm: &mut Vec<f32>) {
    let take_start = SAMPLES_PER_FRAME.min(audio.len());
    let take_end = (SAMPLES_PER_FRAME * 2).min(audio.len());
    if take_end > take_start {
        pcm.extend_from_slice(&audio[take_start..take_end]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_window_starts_match_python() {
        // 5 frames (35 codes): emit at counts 28 and 35 → token starts 0, 7.
        assert_eq!(streaming_window_starts(35), vec![0, 7]);
        // 8 frames (56 codes): emit at 28,35,42,49,56 → starts 0,7,14,21,28.
        assert_eq!(streaming_window_starts(56), vec![0, 7, 14, 21, 28]);
    }

    #[test]
    fn batch_decode_longer_than_streaming_centers() {
        // 12 frames: 9 streaming chunks × 2048 + 6144 tail = full decode.
        let streaming_samples = streaming_window_starts(84).len() * SAMPLES_PER_FRAME;
        let full_samples = 12 * SAMPLES_PER_FRAME;
        assert_eq!(streaming_samples + 3 * SAMPLES_PER_FRAME, full_samples);
    }
}
