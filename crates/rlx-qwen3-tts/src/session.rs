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

//! Warmed megakernel session — reuse across utterances to skip per-request compile warmup.

use crate::config::{GenerationConfig, Qwen3TtsConfig};
use crate::load::Qwen3TtsWeightStore;
use crate::megakernel::Qwen3TtsMegakernel;
use crate::progress::Progress;
use crate::prompt::{CustomVoicePrompt, build_custom_voice_prompt, load_text_tokenizer};
use crate::speech_tokenizer::{St12HzDecoder, open_speech_decoder_for_frames};
use crate::synthesize::SynthesisResult;
use crate::text_embed::TextEmbedder;
use anyhow::Result;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;

/// Holds warmed talker + CP graphs and optional speech decoder (pay warmup once).
pub struct Qwen3TtsSession {
    cfg: Qwen3TtsConfig,
    store: Qwen3TtsWeightStore,
    device: Device,
    mk: Qwen3TtsMegakernel,
    tokenizer: Tokenizer,
    text_embedder: TextEmbedder,
    speech_dec: Option<St12HzDecoder>,
    warmed_max_frames: usize,
}

impl Qwen3TtsSession {
    pub fn open(model_dir: &Path, device: Device) -> Result<Self> {
        let store = Qwen3TtsWeightStore::open(model_dir)?;
        let cfg = Qwen3TtsConfig::from_model_dir(store.model_dir())?;
        let mk = Qwen3TtsMegakernel::open(&store, cfg.talker(), cfg.code_predictor(), device)?;
        let tokenizer = load_text_tokenizer(model_dir)?;
        let text_embedder = TextEmbedder::open(&store)?;
        Ok(Self {
            cfg,
            store,
            device,
            mk,
            tokenizer,
            text_embedder,
            speech_dec: None,
            warmed_max_frames: 0,
        })
    }

    pub fn open_default_dir(device: Device) -> Result<Self> {
        let model_dir = std::env::var("RLX_QWEN3_TTS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-CustomVoice"));
        Self::open(&model_dir, device)
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn config(&self) -> &Qwen3TtsConfig {
        &self.cfg
    }

    pub fn model_dir(&self) -> &Path {
        self.store.model_dir()
    }

    pub fn megakernel(&self) -> &Qwen3TtsMegakernel {
        &self.mk
    }

    pub fn megakernel_mut(&mut self) -> &mut Qwen3TtsMegakernel {
        &mut self.mk
    }

    /// Compile-only warmup with synthetic embeds (no real prompt). Prefer first synthesis warmup.
    pub fn precompile(&mut self, max_frames: usize, progress: Option<&Progress>) -> Result<()> {
        if max_frames <= self.warmed_max_frames {
            return Ok(());
        }
        let hidden = self.cfg.talker().hidden_size;
        let mut embeds = ndarray::Array2::<f32>::zeros((8, hidden));
        for (i, v) in embeds.iter_mut().enumerate() {
            *v = ((i % 17) as f32) * 1e-5;
        }
        self.mk.warmup(embeds.view(), max_frames, progress)?;
        self.mk.finish_compile_warmup();
        self.warmed_max_frames = max_frames;
        Ok(())
    }

    fn ensure_speech_decoder(&mut self, device: Device, n_codec_frames: usize) -> Result<()> {
        let budget = n_codec_frames.max(32);
        if let Some(dec) = self.speech_dec.as_mut() {
            dec.ensure_warmup(device, budget)?;
            return Ok(());
        }
        self.speech_dec = Some(open_speech_decoder_for_frames(
            self.model_dir(),
            device,
            budget,
        )?);
        Ok(())
    }

    pub fn build_prompt(
        &self,
        text: &str,
        speaker: &str,
        language: &str,
    ) -> Result<CustomVoicePrompt> {
        build_custom_voice_prompt(
            &self.cfg,
            &self.store,
            &self.text_embedder,
            &self.tokenizer,
            text,
            speaker,
            language,
        )
    }

    /// Greedy CustomVoice synthesis using warmed graphs.
    pub fn synthesize_custom_voice(
        &mut self,
        text: &str,
        speaker: &str,
        language: &str,
        gen_cfg: &GenerationConfig,
        skip_speech_decode: bool,
    ) -> Result<SynthesisResult> {
        let timing = crate::synth_opts::synth_timing_enabled();
        let t_total = Instant::now();
        let t0 = Instant::now();
        let prompt = self.build_prompt(text, speaker, language)?;
        if timing {
            eprintln!(
                "[qwen3-tts timing] prompt: {:.2}s",
                t0.elapsed().as_secs_f64()
            );
        }

        let max_frames = gen_cfg.max_new_tokens.max(2);
        let frame_budget = crate::synth_opts::codec_frame_budget(text, max_frames, 0);
        let warmup_prog = Progress::new("warmup", 2);
        let t_warm = Instant::now();
        if max_frames > self.warmed_max_frames {
            self.mk
                .warmup(prompt.embeds.view(), max_frames, Some(&warmup_prog))?;
            self.warmed_max_frames = max_frames;
        } else {
            // New prompt with same frame horizon — do not reuse warmup hidden from a prior utterance.
            self.mk.invalidate_warmup_hidden();
            self.mk.ensure_talk_prefill_compiled(prompt.embeds.view())?;
        }
        if !skip_speech_decode {
            warmup_prog.set(1, &format!("speech decoder ({frame_budget} frames)"));
            self.ensure_speech_decoder(self.device, frame_budget)?;
        }
        warmup_prog.finish("session ready");
        if timing {
            eprintln!(
                "[qwen3-tts timing] session warmup (horizon={max_frames}): {:.2}s",
                t_warm.elapsed().as_secs_f64()
            );
        }

        let frame_prog = Progress::new("synthesis", max_frames);
        let t_synth = Instant::now();
        let t_frames = Instant::now();
        let talker_cfg = self.cfg.talker();
        let (codec_frames, ar_timings) = self.mk.synthesize_codec_ar(
            prompt.embeds.view(),
            talker_cfg,
            max_frames,
            gen_cfg.min_new_tokens.max(1),
            gen_cfg.repetition_penalty,
            &prompt.tts_pad_embed,
            Some(&frame_prog),
        )?;
        frame_prog.finish(&format!("{} codec frames", codec_frames.len()));
        if timing {
            eprintln!(
                "[qwen3-tts timing] codec AR ({} frames): {:.2}s (prefill {:.2}s, talker {:.2}s, CP {:.2}s)",
                codec_frames.len(),
                t_frames.elapsed().as_secs_f64(),
                ar_timings.prefill_secs,
                ar_timings.talker_secs,
                ar_timings.cp_secs
            );
        }

        let pcm = if skip_speech_decode {
            Vec::new()
        } else {
            let t_dec = Instant::now();
            let pcm = self
                .speech_dec
                .as_mut()
                .expect("speech decoder")
                .decode(&codec_frames, self.device)?;
            if timing {
                eprintln!(
                    "[qwen3-tts timing] speech decode: {:.2}s",
                    t_dec.elapsed().as_secs_f64()
                );
            }
            pcm
        };
        if timing {
            let synth_secs = t_synth.elapsed().as_secs_f64();
            let audio_secs = pcm.len() as f64 / crate::tokens::SAMPLE_RATE_HZ as f64;
            if audio_secs > 0.0 {
                eprintln!(
                    "[qwen3-tts timing] audio duration: {:.2}s ({} samples)",
                    audio_secs,
                    pcm.len()
                );
                eprintln!(
                    "[qwen3-tts timing] synthesis rtf: {:.2} ({:.2}s synth / {:.2}s audio; target rtf≤1.0)",
                    synth_secs / audio_secs,
                    synth_secs,
                    audio_secs
                );
            }
            eprintln!(
                "[qwen3-tts timing] total: {:.2}s (device={:?})",
                t_total.elapsed().as_secs_f64(),
                self.device
            );
        }
        Ok(SynthesisResult {
            codec_frames,
            pcm,
            sample_rate: crate::tokens::SAMPLE_RATE_HZ,
        })
    }
}
