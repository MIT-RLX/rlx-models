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

//! Frozen teacher hidden states for Phase 2 LoRA distillation.

use anyhow::{Result, ensure};
use rlx_voxtral_tts::VoxtralTtsWeightStore;
use rlx_voxtral_tts::backbone::embed::EmbeddingTables;
use rlx_voxtral_tts::backbone::lm::MinistralLm;
use rlx_voxtral_tts::codec::encoder::{CodecEncoder, load_mono_wav};
use rlx_voxtral_tts::config::VoxtralTtsConfig;
use rlx_voxtral_tts::load::PREFIX_CODEC;
use rlx_voxtral_tts::speech_tokenizer::SpeechTokenizer;
use rlx_voxtral_tts::voice::VoiceEmbedding;
use rlx_voxtral_tts::voice_clone::{
    VoiceCloneSupport, encode_reference_wav, max_reference_seconds, voice_clone_support,
};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use crate::config::env_flag;
use crate::weights::{codec_has_encoder, merge_codec_encoder_overlay};

#[derive(Clone)]
pub struct DistillBatch {
    pub seq: usize,
    pub inputs: Vec<f32>,
    pub targets: Vec<f32>,
}

/// Reuses backbone / embed / codec weights and voice encodings across steps.
pub struct TeacherCache {
    store: VoxtralTtsWeightStore,
    cfg: VoxtralTtsConfig,
    tokenizer: SpeechTokenizer,
    embed: EmbeddingTables,
    teacher: MinistralLm,
    codec_encoder: Option<CodecEncoder>,
    voice_by_wav: HashMap<PathBuf, VoiceEmbedding>,
    batch_cache: HashMap<(usize, u64), DistillBatch>,
    clone_support: VoiceCloneSupport,
}

impl TeacherCache {
    pub fn open(
        store: &VoxtralTtsWeightStore,
        cfg: &VoxtralTtsConfig,
        encoder_weights: Option<&Path>,
    ) -> Result<Self> {
        let tokenizer = SpeechTokenizer::from_model_dir(store.model_dir())?;
        let embed_tensors = store.tensor_snapshot_for_embed()?;
        let embed = EmbeddingTables::from_tensors(
            &embed_tensors,
            &cfg.text_config,
            &cfg.audio_config.audio_model_args,
        )?;
        let backbone = store.tensor_snapshot_for_backbone()?;
        let teacher = MinistralLm::from_tensors(&backbone, &cfg.text_config)?;
        let clone_support = voice_clone_support(store);

        let codec_encoder = if let Some(path) = encoder_weights {
            let mut codec_tensors = store.tensor_snapshot(PREFIX_CODEC)?;
            merge_codec_encoder_overlay(&mut codec_tensors, path)?;
            ensure!(
                codec_has_encoder(&codec_tensors),
                "encoder overlay did not provide codec encoder weights — train Phase 1 first or pass --encoder-weights"
            );
            Some(CodecEncoder::from_tensors(
                PREFIX_CODEC,
                &codec_tensors,
                &cfg.audio_config.codec_args,
            )?)
        } else {
            None
        };

        Ok(Self {
            store: store.clone(),
            cfg: cfg.clone(),
            tokenizer,
            embed,
            teacher,
            codec_encoder,
            voice_by_wav: HashMap::new(),
            batch_cache: HashMap::new(),
            clone_support,
        })
    }

    pub fn build_batch(
        &mut self,
        text: &str,
        voice_name: &str,
        reference_wav: Option<&Path>,
        max_seq: usize,
        wav_idx: usize,
    ) -> Result<DistillBatch> {
        let key = (wav_idx, hash_text(text));
        if env_flag("PRECOMPUTE_DISTILL") {
            if let Some(batch) = self.batch_cache.get(&key) {
                return Ok(batch.clone());
            }
        }

        let batch = self.build_batch_inner(text, voice_name, reference_wav, max_seq)?;
        if env_flag("PRECOMPUTE_DISTILL") {
            self.batch_cache.insert(key, batch.clone());
        }
        Ok(batch)
    }

    /// Warm all training-step prompts once before the loop (I/O + teacher CPU forward).
    pub fn prewarm(
        &mut self,
        wavs: &[PathBuf],
        texts: &[String],
        voice_name: &str,
        max_seq: usize,
        steps_per_epoch: usize,
        epochs: usize,
    ) -> Result<usize> {
        let mut built = 0usize;
        let total_steps = epochs.saturating_mul(steps_per_epoch);
        for step in 0..total_steps {
            let idx = step % wavs.len();
            let transcript = texts.get(idx).map(String::as_str);
            let text = crate::distill_text::distill_text_for_sample(step, idx, transcript);
            let key = (idx, hash_text(&text));
            if self.batch_cache.contains_key(&key) {
                continue;
            }
            let batch = self.build_batch_inner(&text, voice_name, Some(&wavs[idx]), max_seq)?;
            self.batch_cache.insert(key, batch);
            built += 1;
        }
        Ok(built)
    }

    pub fn cached_batches(&self) -> usize {
        self.batch_cache.len()
    }

    fn build_batch_inner(
        &mut self,
        text: &str,
        voice_name: &str,
        reference_wav: Option<&Path>,
        max_seq: usize,
    ) -> Result<DistillBatch> {
        let token_ids = self.tokenizer.encode_speech(text, voice_name)?;
        let seq = token_ids.len().min(max_seq);
        ensure!(seq >= 4, "distill prompt too short after truncation");

        let mut embeds = self.embed.embed_tokens(&token_ids[..seq]);
        let voice = if let Some(wav) = reference_wav {
            self.voice_for_wav(wav, voice_name)?
        } else {
            synthetic_voice(&self.embed, voice_name)?
        };
        let voice_rows: Vec<&[f32]> = (0..voice.n_tokens).map(|i| voice.row(i)).collect();
        self.embed
            .inject_voice(&mut embeds, &token_ids[..seq], &voice_rows);

        self.teacher.reset_cache();
        let hidden = self.teacher.forward_pre_norm(embeds.view())?;
        let targets = hidden.iter().copied().collect::<Vec<_>>();
        let inputs = embeds.iter().copied().collect::<Vec<_>>();

        Ok(DistillBatch {
            seq,
            inputs,
            targets,
        })
    }

    fn voice_for_wav(&mut self, wav: &Path, voice_name: &str) -> Result<VoiceEmbedding> {
        if let Some(v) = self.voice_by_wav.get(wav) {
            return Ok(v.clone());
        }
        let voice = if self.codec_encoder.is_some() {
            self.encode_reference_with_overlay(wav, voice_name)?
        } else if self.clone_support == VoiceCloneSupport::ReferenceAudio {
            encode_reference_wav(&self.store, &self.cfg, wav, voice_name)?
        } else {
            synthetic_voice(&self.embed, voice_name)?
        };
        self.voice_by_wav.insert(wav.to_path_buf(), voice.clone());
        Ok(voice)
    }

    fn encode_reference_with_overlay(
        &mut self,
        reference_wav: &Path,
        voice_name: &str,
    ) -> Result<VoiceEmbedding> {
        let rate = self.cfg.audio_config.codec_args.sampling_rate as u32;
        let mut pcm = load_mono_wav(reference_wav, rate)?;
        let max_samples = (max_reference_seconds() * rate as f32) as usize;
        if pcm.len() > max_samples {
            pcm.truncate(max_samples);
        }
        ensure!(
            pcm.len() >= rate as usize / 2,
            "reference wav too short (need at least ~0.5s at {rate} Hz)"
        );
        let encoder = self
            .codec_encoder
            .as_ref()
            .expect("encoder overlay configured");
        encoder.encode_pcm_to_voice_embedding(&pcm, &self.embed, voice_name)
    }
}

fn hash_text(text: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    h.finish()
}

pub fn build_distill_batch(
    store: &VoxtralTtsWeightStore,
    cfg: &VoxtralTtsConfig,
    text: &str,
    voice_name: &str,
    reference_wav: Option<&Path>,
    max_seq: usize,
    encoder_weights: Option<&Path>,
) -> Result<DistillBatch> {
    let mut cache = TeacherCache::open(store, cfg, encoder_weights)?;
    cache.build_batch(text, voice_name, reference_wav, max_seq, 0)
}

fn synthetic_voice(embed: &EmbeddingTables, voice_name: &str) -> Result<VoiceEmbedding> {
    let hidden = embed.hidden_size();
    let n = 8usize;
    let mut data = vec![0f32; n * hidden];
    let seed = voice_name
        .bytes()
        .fold(0u32, |a, b| a.wrapping_mul(31).wrapping_add(b as u32));
    for (i, v) in data.iter_mut().enumerate() {
        *v = ((seed.wrapping_add(i as u32) as f32) * 1e-4).sin() * 0.01;
    }
    Ok(VoiceEmbedding {
        name: voice_name.to_string(),
        hidden,
        n_tokens: n,
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn codec_has_encoder_detects_input_proj() {
        let mut m = HashMap::new();
        m.insert(
            "audio_tokenizer.input_proj.conv.weight".into(),
            (vec![0.0; 4], vec![4]),
        );
        assert!(codec_has_encoder(&m));
    }
}
