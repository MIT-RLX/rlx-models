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
        Self::long_form()
    }
}

impl GenerationConfig {
    /// Long-form / narration — high token ceiling (upstream default).
    pub fn long_form() -> Self {
        Self {
            max_new_tokens: 1200,
            temperature: 0.6,
            top_p: 0.95,
            top_k: 0,
            repetition_penalty: 1.1,
            seed: 42,
            greedy: false,
        }
    }

    /// Chat / assistant replies — lower ceiling for faster synthesis.
    pub fn chat() -> Self {
        Self {
            max_new_tokens: 384,
            ..Self::long_form()
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
                snac: crate::device::default_snac_exec(device),
            },
            BackboneLoadOptions::for_tts(device),
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
            config: GenerationConfig::chat(),
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
                snac: crate::device::default_snac_exec(device),
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

    /// Synthesize multiple utterances reusing the loaded backbone + SNAC session.
    #[cfg(feature = "llama")]
    pub fn synthesize_batch(&self, items: &[(&str, Option<&str>)]) -> Result<Vec<SynthesisResult>> {
        items
            .iter()
            .map(|(text, voice)| self.synthesize(text, *voice))
            .collect()
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

        normalize_pcm_peak(&mut pcm);
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
        let timing = std::env::var("ORPHEUS_PHASE_TIMING").ok().as_deref() == Some("1");
        let t_lm = std::time::Instant::now();
        let codes = self
            .backbone
            .generate_codes_from_prompt(prompt_ids, &self.config)
            .context("LM code generation")?;
        let lm_ms = t_lm.elapsed().as_secs_f64() * 1000.0;

        let t_snac = std::time::Instant::now();
        let mut samples = self.decode_all_codes(&codes)?;
        let snac_ms = t_snac.elapsed().as_secs_f64() * 1000.0;
        if timing {
            eprintln!(
                "[phase-timing] LM(prefill+decode)={lm_ms:.0} ms ({} codes, {:.1} ms/code)  SNAC={snac_ms:.0} ms",
                codes.len(),
                if codes.is_empty() {
                    0.0
                } else {
                    lm_ms / codes.len() as f64
                },
            );
        }
        normalize_pcm_peak(&mut samples);
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

const PCM_NORM_TARGET_PEAK: f32 = 0.95;
const PCM_NORM_MIN_PEAK: f32 = 0.25;

/// Boost quiet SNAC output toward full-scale (Whisper / playback friendly).
///
/// Default on when peak < 0.25; disable with `ORPHEUS_NORMALIZE_PCM=0`.
pub fn normalize_pcm_peak(samples: &mut [f32]) {
    if matches!(
        std::env::var("ORPHEUS_NORMALIZE_PCM").ok().as_deref(),
        Some("0") | Some("false") | Some("FALSE")
    ) {
        return;
    }
    let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 1e-6 && peak < PCM_NORM_MIN_PEAK {
        let gain = PCM_NORM_TARGET_PEAK / peak;
        for s in samples.iter_mut() {
            *s = (*s * gain).clamp(-1.0, 1.0);
        }
    }
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

    #[test]
    fn chat_caps_tokens_below_long_form() {
        assert!(
            GenerationConfig::chat().max_new_tokens < GenerationConfig::long_form().max_new_tokens
        );
    }
}
