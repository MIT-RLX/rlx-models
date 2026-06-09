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

//! Reference-audio voice cloning via trained codec encoder weights.

use crate::backbone::embed::EmbeddingTables;
use crate::codec::encoder::{
    CodecEncoder, has_encoder_tensors, has_encoder_weights, load_mono_wav,
};
use crate::codec::encoder_seed::seed_encoder_from_decoder;
use crate::config::VoxtralTtsConfig;
use crate::load::{PREFIX_CODEC, VoxtralTtsWeightStore};
use crate::voice::VoiceEmbedding;
use anyhow::{Context, Result, ensure};
use std::path::Path;

const DEFAULT_REF_SECONDS: f32 = 5.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceCloneSupport {
    PresetEmbeddingsOnly,
    ReferenceAudio,
}

pub fn voice_clone_support(store: &VoxtralTtsWeightStore) -> VoiceCloneSupport {
    let keys = store.keys();
    if has_encoder_weights(keys, PREFIX_CODEC) || keys.iter().any(|k| k.contains("decoder_blocks"))
    {
        VoiceCloneSupport::ReferenceAudio
    } else {
        VoiceCloneSupport::PresetEmbeddingsOnly
    }
}

pub fn max_reference_seconds() -> f32 {
    std::env::var("RLX_VOXTRAL_TTS_REF_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|v| *v > 0.0)
        .unwrap_or(DEFAULT_REF_SECONDS)
}

pub fn encode_reference_wav(
    store: &VoxtralTtsWeightStore,
    cfg: &VoxtralTtsConfig,
    reference_wav: &Path,
    voice_name: &str,
) -> Result<VoiceEmbedding> {
    ensure!(
        voice_clone_support(store) == VoiceCloneSupport::ReferenceAudio,
        "reference-audio cloning requires codec decoder weights in consolidated.safetensors.\n\
         The public checkpoint omits a trained encoder; RLX seeds one from the decoder at runtime."
    );
    let rate = cfg.audio_config.codec_args.sampling_rate as u32;
    let mut pcm = load_mono_wav(reference_wav, rate)?;
    let max_samples = (max_reference_seconds() * rate as f32) as usize;
    if pcm.len() > max_samples {
        pcm.truncate(max_samples);
    }
    ensure!(
        pcm.len() >= rate as usize / 2,
        "reference wav too short (need at least ~0.5s at {rate} Hz)"
    );

    let mut codec_tensors = store.tensor_snapshot(PREFIX_CODEC)?;
    let codec_args = &cfg.audio_config.codec_args;
    if !has_encoder_tensors(PREFIX_CODEC, &codec_tensors) {
        seed_encoder_from_decoder(&mut codec_tensors, codec_args)
            .context("seed codec encoder from decoder weights (public checkpoints omit encoder)")?;
    }
    let embed_tensors = store.tensor_snapshot_for_embed()?;
    let encoder = match CodecEncoder::from_tensors(PREFIX_CODEC, &codec_tensors, codec_args) {
        Ok(enc) => enc,
        Err(first) => {
            seed_encoder_from_decoder(&mut codec_tensors, codec_args)
                .context("re-seed codec encoder from decoder after load failure")?;
            CodecEncoder::from_tensors(PREFIX_CODEC, &codec_tensors, codec_args)
                .with_context(|| format!("build encoder after decoder seed: {first}"))
        }?,
    };
    let embed = EmbeddingTables::from_tensors(
        &embed_tensors,
        &cfg.text_config,
        &cfg.audio_config.audio_model_args,
    )?;
    encoder.encode_pcm_to_voice_embedding(&pcm, &embed, voice_name)
}

pub fn encode_reference_wav_to_file(
    store: &VoxtralTtsWeightStore,
    cfg: &VoxtralTtsConfig,
    reference_wav: &Path,
    out_f32: &Path,
    voice_name: &str,
) -> Result<VoiceEmbedding> {
    let emb = encode_reference_wav(store, cfg, reference_wav, voice_name)?;
    emb.save_f32(out_f32)?;
    Ok(emb)
}

pub fn clone_from_reference_audio(
    model_dir: &Path,
    reference_wav: &Path,
) -> Result<VoiceEmbedding> {
    let store = VoxtralTtsWeightStore::open(model_dir)?;
    let cfg = VoxtralTtsConfig::from_model_dir(store.model_dir())?;
    encode_reference_wav(&store, &cfg, reference_wav, "cloned")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_only_without_encoder_keys() {
        let keys = ["audio_tokenizer.decoder_blocks.0".to_string()]
            .into_iter()
            .collect();
        assert!(!has_encoder_weights(&keys, PREFIX_CODEC));
    }
}
