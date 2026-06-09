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

//! Native autoregressive TTS (Ministral + flow-matching + codec).

use crate::acoustic_engine::AcousticHead;
use crate::backbone::embed::EmbeddingTables;
use crate::backbone::engine::BackboneLm;
use crate::bench::VoxtralTtsBenchReport;
use crate::codec::decoder::CodecDecoder;
use crate::config::VoxtralTtsConfig;
use crate::generation::GenerationConfig;
use crate::load::VoxtralTtsWeightStore;
use crate::options::VoxtralTtsOptions;
use crate::speech_tokenizer::SpeechTokenizer;
use crate::tokens::END_AUDIO;
use crate::voice::VoiceEmbedding;
use anyhow::{Result, bail, ensure};
use rlx_runtime::Device;
use std::time::Instant;

pub struct NativeTtsEngine {
    cfg: VoxtralTtsConfig,
    lm: BackboneLm,
    embed: EmbeddingTables,
    acoustic: AcousticHead,
    codec: CodecDecoder,
    euler_steps_per_frame: usize,
    device: Device,
    eager_lm: bool,
    eager_acoustic: bool,
}

impl NativeTtsEngine {
    pub fn open(
        store: &VoxtralTtsWeightStore,
        cfg: &VoxtralTtsConfig,
        options: &VoxtralTtsOptions,
    ) -> Result<Self> {
        let embed_tensors = store.tensor_snapshot_for_embed()?;
        let lora = crate::lora::load_lora_bank(store, &cfg.text_config)?;
        let backbone_tensors = if options.eager_lm || options.device == Device::Cpu {
            Some(store.tensor_snapshot_for_backbone()?)
        } else {
            None
        };
        let lm = BackboneLm::open(
            store,
            &cfg.text_config,
            backbone_tensors.as_ref(),
            options.device,
            options.eager_lm,
            lora.as_ref(),
        )?;
        let embed = EmbeddingTables::from_tensors(
            &embed_tensors,
            &cfg.text_config,
            &cfg.audio_config.audio_model_args,
        )?;
        let codec_tensors = store.tensor_snapshot(crate::load::PREFIX_CODEC)?;
        let codec = CodecDecoder::from_tensors(
            crate::load::PREFIX_CODEC,
            &codec_tensors,
            &cfg.audio_config.codec_args,
        )?;
        let acoustic_tensors = store.tensor_snapshot(crate::load::PREFIX_ACOUSTIC)?;
        let acoustic = AcousticHead::open(
            store,
            crate::load::PREFIX_ACOUSTIC,
            &acoustic_tensors,
            &cfg.audio_config.audio_model_args,
            options.device,
            options.eager_acoustic,
        )?;
        let euler_steps_per_frame = cfg
            .audio_config
            .audio_model_args
            .acoustic_transformer_args
            .n_decoding_steps
            .unwrap_or(crate::tokens::DEFAULT_EULER_STEPS);
        Ok(Self {
            cfg: cfg.clone(),
            lm,
            embed,
            acoustic,
            codec,
            euler_steps_per_frame,
            device: options.device,
            eager_lm: options.eager_lm,
            eager_acoustic: options.eager_acoustic,
        })
    }

    /// Greedy audio-frame loop (matches vLLM-Omni forced audio-token sampling).
    pub fn synthesize(
        &mut self,
        token_ids: &[u32],
        voice: &VoiceEmbedding,
        gen_cfg: &GenerationConfig,
    ) -> Result<Vec<f32>> {
        self.synthesize_profiled(token_ids, voice, gen_cfg)
            .map(|(pcm, _)| pcm)
    }

    /// Same as [`Self::synthesize`] with per-stage timings (for benchmarks).
    pub fn synthesize_profiled(
        &mut self,
        token_ids: &[u32],
        voice: &VoiceEmbedding,
        gen_cfg: &GenerationConfig,
    ) -> Result<(Vec<f32>, VoxtralTtsBenchReport)> {
        ensure!(
            voice.hidden == self.cfg.text_config.hidden_size,
            "voice hidden {} != {}",
            voice.hidden,
            self.cfg.text_config.hidden_size
        );
        let mut report = VoxtralTtsBenchReport {
            device: self.device,
            eager_lm: self.eager_lm,
            eager_acoustic: self.eager_acoustic,
            prompt_tokens: token_ids.len(),
            euler_steps_per_frame: self.euler_steps_per_frame,
            embed_ms: 0.0,
            lm_prefill_ms: 0.0,
            lm_decode_ms: 0.0,
            acoustic_ms: 0.0,
            codec_ms: 0.0,
            synthesis_ms: 0.0,
            audio_frames: 0,
            pcm_samples: 0,
            acoustic_velocity_runs: 0,
        };

        self.lm.reset_cache();
        let voice_rows = voice_rows_for_prompt(token_ids, voice)?;

        let t_embed = Instant::now();
        let mut embeds = self.embed.embed_tokens(token_ids);
        self.embed.inject_voice(&mut embeds, token_ids, &voice_rows);
        report.embed_ms = t_embed.elapsed().as_secs_f64() * 1000.0;

        let t_prefill = Instant::now();
        let mut hidden = self.lm.forward(embeds.view())?;
        report.lm_prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

        let mut all_codes: Vec<u32> = Vec::new();
        let mut velocity_runs: u64 = 0;

        for frame_idx in 0..gen_cfg.max_frames {
            let h = self.lm.last_hidden(&hidden);
            let t_ac = Instant::now();
            let frame = self.acoustic.predict_frame(
                h.view(),
                gen_cfg.cfg_alpha,
                gen_cfg.seed,
                frame_idx,
            )?;
            report.acoustic_ms += t_ac.elapsed().as_secs_f64() * 1000.0;
            velocity_runs += (self.euler_steps_per_frame as u64) * 2;
            if frame[0] == END_AUDIO {
                break;
            }
            all_codes.extend_from_slice(&frame);

            let next = self.embed.embed_audio_frame(&frame);
            let next_2d = next.insert_axis(ndarray::Axis(0));
            let t_dec = Instant::now();
            hidden = self.lm.forward(next_2d.view())?;
            report.lm_decode_ms += t_dec.elapsed().as_secs_f64() * 1000.0;
        }

        if all_codes.is_empty() {
            bail!("native TTS produced no audio frames");
        }
        let n_frames = all_codes.len() / 37;
        report.audio_frames = n_frames;
        report.acoustic_velocity_runs = velocity_runs;

        let t_codec = Instant::now();
        let pcm = self.codec.decode_codes(&all_codes, n_frames)?;
        report.codec_ms = t_codec.elapsed().as_secs_f64() * 1000.0;
        report.pcm_samples = pcm.len();
        report.synthesis_ms =
            report.lm_prefill_ms + report.lm_decode_ms + report.acoustic_ms + report.codec_ms;

        Ok((pcm, report))
    }

    /// Return discrete codes without codec decode (parity vs vLLM stage 0).
    pub fn synthesize_codes(
        &mut self,
        token_ids: &[u32],
        voice: &VoiceEmbedding,
        gen_cfg: &GenerationConfig,
    ) -> Result<Vec<u32>> {
        ensure!(
            voice.hidden == self.cfg.text_config.hidden_size,
            "voice hidden {} != {}",
            voice.hidden,
            self.cfg.text_config.hidden_size
        );
        self.lm.reset_cache();
        let voice_rows = voice_rows_for_prompt(token_ids, voice)?;
        let mut embeds = self.embed.embed_tokens(token_ids);
        self.embed.inject_voice(&mut embeds, token_ids, &voice_rows);

        let mut hidden = self.lm.forward(embeds.view())?;
        let mut all_codes: Vec<u32> = Vec::new();

        for frame_idx in 0..gen_cfg.max_frames {
            let h = self.lm.last_hidden(&hidden);
            let frame = self.acoustic.predict_frame(
                h.view(),
                gen_cfg.cfg_alpha,
                gen_cfg.seed,
                frame_idx,
            )?;
            if frame[0] == END_AUDIO {
                break;
            }
            all_codes.extend_from_slice(&frame);
            let next = self.embed.embed_audio_frame(&frame);
            let next_2d = next.insert_axis(ndarray::Axis(0));
            hidden = self.lm.forward(next_2d.view())?;
        }
        Ok(all_codes)
    }
}

fn voice_rows_for_prompt<'a>(
    token_ids: &[u32],
    voice: &'a VoiceEmbedding,
) -> Result<Vec<&'a [f32]>> {
    let n_audio_slots = SpeechTokenizer::count_audio_slots(token_ids);
    ensure!(
        n_audio_slots == voice.n_tokens,
        "prompt has {n_audio_slots} audio slots but voice embedding has {} rows; \
         re-tokenize with SpeechTokenizer::encode_speech_with_n_audio(text, voice.n_tokens)",
        voice.n_tokens
    );
    Ok((0..voice.n_tokens).map(|i| voice.row(i)).collect())
}
