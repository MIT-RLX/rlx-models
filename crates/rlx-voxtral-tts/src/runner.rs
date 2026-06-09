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

//! End-to-end TTS runner — native Rust only.

use crate::acoustic::AcousticTransformer;
use crate::backbone::NativeTtsEngine;
use crate::bench::VoxtralTtsBenchReport;
use crate::codec::decoder::CodecDecoder;
use crate::config::VoxtralTtsConfig;
use crate::generation::GenerationConfig;
use crate::load::{PREFIX_CODEC, VoxtralTtsWeightStore};
use crate::options::{VoxtralTtsOptions, VoxtralTtsRunnerBuilder};
use crate::prompt_tokens::load_prompt_tokens;
use crate::speech_tokenizer::SpeechTokenizer;
use crate::tokens::PRESET_VOICES;
use crate::voice::VoiceEmbedding;
use crate::voice_clone::{VoiceCloneSupport, encode_reference_wav, voice_clone_support};
use anyhow::{Context, Result, bail};
use rlx_runtime::Device;
use std::path::Path;

pub struct VoxtralTtsRunner {
    cfg: VoxtralTtsConfig,
    store: VoxtralTtsWeightStore,
    options: VoxtralTtsOptions,
    codec: CodecDecoder,
    #[allow(dead_code)]
    acoustic: AcousticTransformer,
    native: Option<NativeTtsEngine>,
}

impl VoxtralTtsRunner {
    pub fn builder() -> VoxtralTtsRunnerBuilder {
        VoxtralTtsRunnerBuilder::default()
    }

    pub fn open(model_dir: &Path) -> Result<Self> {
        Self::open_with_options(model_dir, VoxtralTtsOptions::default())
    }

    pub fn open_with_options(model_dir: &Path, options: VoxtralTtsOptions) -> Result<Self> {
        options.validate()?;
        let store = VoxtralTtsWeightStore::open(model_dir)?;
        let cfg = VoxtralTtsConfig::from_model_dir(store.model_dir())?;
        let codec_tensors = store.tensor_snapshot(PREFIX_CODEC)?;
        let codec =
            CodecDecoder::from_tensors(PREFIX_CODEC, &codec_tensors, &cfg.audio_config.codec_args)?;
        let acoustic_tensors = store.tensor_snapshot(crate::load::PREFIX_ACOUSTIC)?;
        let acoustic = AcousticTransformer::from_tensors(
            crate::load::PREFIX_ACOUSTIC,
            &acoustic_tensors,
            &cfg.audio_config.audio_model_args.acoustic_transformer_args,
            cfg.audio_config.audio_model_args.n_acoustic_codebook,
            cfg.audio_config.audio_model_args.semantic_codebook_size,
        )?;
        Ok(Self {
            cfg,
            store,
            options,
            codec,
            acoustic,
            native: None,
        })
    }

    pub fn config(&self) -> &VoxtralTtsConfig {
        &self.cfg
    }

    pub fn device(&self) -> Device {
        self.options.device
    }

    pub fn options(&self) -> &VoxtralTtsOptions {
        &self.options
    }

    pub fn model_dir(&self) -> &Path {
        self.store.model_dir()
    }

    pub fn decode_codes_to_pcm(&self, codes: &[u32], n_frames: usize) -> Result<Vec<f32>> {
        self.codec.decode_codes(codes, n_frames)
    }

    pub fn decode_codes_file(&self, codes_path: &Path, out_wav: &Path) -> Result<()> {
        let (codes, n_frames) = parse_codes_file(codes_path)?;
        let pcm = self.decode_codes_to_pcm(&codes, n_frames)?;
        write_wav_mono(
            out_wav,
            &pcm,
            self.cfg.audio_config.codec_args.sampling_rate as u32,
        )
    }

    pub fn voice_clone_support(&self) -> VoiceCloneSupport {
        voice_clone_support(&self.store)
    }

    /// Native path with an explicit voice embedding (preset or cloned).
    pub fn synthesize_native_with_voice(
        &mut self,
        prompt_tokens: &[u32],
        voice_emb: &VoiceEmbedding,
        out_wav: &Path,
        gen_cfg: &GenerationConfig,
    ) -> Result<()> {
        if self.native.is_none() {
            self.native = Some(NativeTtsEngine::open(
                &self.store,
                &self.cfg,
                &self.options,
            )?);
        }
        let engine = self.native.as_mut().unwrap();
        let pcm = engine.synthesize(prompt_tokens, voice_emb, gen_cfg)?;
        write_wav_mono(
            out_wav,
            &pcm,
            self.cfg.audio_config.codec_args.sampling_rate as u32,
        )
    }

    /// Encode reference audio then synthesize (requires injected encoder weights).
    pub fn synthesize_cloned_from_wav(
        &mut self,
        prompt_tokens: &[u32],
        reference_wav: &Path,
        out_wav: &Path,
        gen_cfg: &GenerationConfig,
    ) -> Result<()> {
        let voice_emb = encode_reference_wav(&self.store, &self.cfg, reference_wav, "cloned")?;
        self.synthesize_native_with_voice(prompt_tokens, &voice_emb, out_wav, gen_cfg)
    }

    /// Encode reference audio, tokenize for its frame count, then synthesize.
    pub fn synthesize_cloned_with_text(
        &mut self,
        text: &str,
        reference_wav: &Path,
        out_wav: &Path,
        gen_cfg: &GenerationConfig,
    ) -> Result<()> {
        let voice_emb = encode_reference_wav(&self.store, &self.cfg, reference_wav, "cloned")?;
        let tok = SpeechTokenizer::from_model_dir(self.model_dir())?;
        let prompt_tokens = tok.encode_speech_with_n_audio(text, voice_emb.n_tokens as u32)?;
        self.synthesize_native_with_voice(&prompt_tokens, &voice_emb, out_wav, gen_cfg)
    }

    /// Native path: compiled LM + acoustic (or eager fallbacks) + codec decode on CPU.
    pub fn synthesize_native(
        &mut self,
        prompt_tokens: &[u32],
        voice: &str,
        out_wav: &Path,
        gen_cfg: &GenerationConfig,
    ) -> Result<()> {
        let hidden = self.cfg.text_config.hidden_size;
        let voice_emb = resolve_preset_voice(self.model_dir(), voice, hidden)?;
        self.synthesize_native_with_voice(prompt_tokens, &voice_emb, out_wav, gen_cfg)
    }

    pub fn synthesize_native_from_token_file(
        &mut self,
        prompt_tokens_path: &Path,
        voice: &str,
        out_wav: &Path,
        gen_cfg: &GenerationConfig,
    ) -> Result<()> {
        let tokens = load_prompt_tokens(prompt_tokens_path)?;
        self.synthesize_native(&tokens, voice, out_wav, gen_cfg)
    }

    pub fn synthesize_cloned_from_token_file(
        &mut self,
        prompt_tokens_path: &Path,
        reference_wav: &Path,
        out_wav: &Path,
        gen_cfg: &GenerationConfig,
    ) -> Result<()> {
        let tokens = load_prompt_tokens(prompt_tokens_path)?;
        self.synthesize_cloned_from_wav(&tokens, reference_wav, out_wav, gen_cfg)
    }

    pub fn synthesize_native_with_embedding_file(
        &mut self,
        prompt_tokens_path: &Path,
        voice_embedding: &Path,
        out_wav: &Path,
        gen_cfg: &GenerationConfig,
    ) -> Result<()> {
        let tokens = load_prompt_tokens(prompt_tokens_path)?;
        let hidden = self.cfg.text_config.hidden_size;
        let voice_emb = VoiceEmbedding::load_f32(voice_embedding, "custom", hidden)?;
        self.synthesize_native_with_voice(&tokens, &voice_emb, out_wav, gen_cfg)
    }

    pub fn encode_reference_to_file(
        &self,
        reference_wav: &Path,
        out_f32: &Path,
        voice_name: &str,
    ) -> Result<VoiceEmbedding> {
        crate::voice_clone::encode_reference_wav_to_file(
            &self.store,
            &self.cfg,
            reference_wav,
            out_f32,
            voice_name,
        )
    }

    /// Timed native synthesis on a fixed prompt (same weights for all option sets).
    pub fn bench_native_profiled(
        &mut self,
        prompt_tokens: &[u32],
        voice: &str,
        gen_cfg: &GenerationConfig,
        options: &VoxtralTtsOptions,
    ) -> Result<VoxtralTtsBenchReport> {
        options.validate()?;
        let hidden = self.cfg.text_config.hidden_size;
        let voice_emb = resolve_preset_voice(self.model_dir(), voice, hidden)?;
        let mut engine = NativeTtsEngine::open(&self.store, &self.cfg, options)?;
        let (_, report) = engine.synthesize_profiled(prompt_tokens, &voice_emb, gen_cfg)?;
        Ok(report)
    }

    pub fn synthesize_native_codes(
        &mut self,
        prompt_tokens: &[u32],
        voice: &str,
        gen_cfg: &GenerationConfig,
    ) -> Result<Vec<u32>> {
        let hidden = self.cfg.text_config.hidden_size;
        let voice_emb = resolve_preset_voice(self.model_dir(), voice, hidden)?;
        if self.native.is_none() {
            self.native = Some(NativeTtsEngine::open(
                &self.store,
                &self.cfg,
                &self.options,
            )?);
        }
        let engine = self.native.as_mut().unwrap();
        engine.synthesize_codes(prompt_tokens, &voice_emb, gen_cfg)
    }
}

fn resolve_preset_voice(model_dir: &Path, voice: &str, hidden: usize) -> Result<VoiceEmbedding> {
    if PRESET_VOICES.contains(&voice) {
        return VoiceEmbedding::load(model_dir, voice, hidden);
    }
    bail!(
        "unknown preset voice {voice:?}; expected one of {} (or use --reference-wav / --voice-embedding)",
        PRESET_VOICES.join(", ")
    )
}

pub fn parse_codes_file(path: &Path) -> Result<(Vec<u32>, usize)> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut lines = text.lines();
    let n_frames: usize = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("empty codes file"))?
        .parse()
        .context("parse frame count")?;
    let body = lines.next().unwrap_or_default();
    let codes: Vec<u32> = body
        .split_whitespace()
        .map(|s| s.parse().context("parse code"))
        .collect::<Result<_>>()?;
    Ok((codes, n_frames))
}

pub fn write_wav_mono(path: &Path, pcm: &[f32], sample_rate: u32) -> Result<()> {
    let mut bytes = Vec::with_capacity(44 + pcm.len() * 2);
    bytes.extend_from_slice(b"RIFF");
    let data_bytes = (pcm.len() * 2 + 36) as u32;
    bytes.extend_from_slice(&data_bytes.to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&((pcm.len() * 2) as u32).to_le_bytes());
    for &s in pcm {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}
