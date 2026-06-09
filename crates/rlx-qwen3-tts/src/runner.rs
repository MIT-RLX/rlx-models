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

//! Qwen3-TTS runner — native talker bench + greedy CustomVoice synthesis.

use crate::bench::Qwen3TtsBenchReport;
use crate::config::GenerationConfig;
use crate::config::Qwen3TtsConfig;
use crate::load::Qwen3TtsWeightStore;
use crate::options::{Qwen3TtsOptions, Qwen3TtsRunnerBuilder};
use crate::session::Qwen3TtsSession;
use crate::synthesize::synthesize_custom_voice_greedy;
use crate::talker::TalkerEngine;
use anyhow::{Result, bail};
use ndarray::Array2;
use rlx_runtime::Device;
use std::path::Path;
use std::time::Instant;

pub struct Qwen3TtsRunner {
    cfg: Qwen3TtsConfig,
    store: Qwen3TtsWeightStore,
    options: Qwen3TtsOptions,
}

impl Qwen3TtsRunner {
    pub fn builder() -> Qwen3TtsRunnerBuilder {
        Qwen3TtsRunnerBuilder::default()
    }

    pub fn open_with_options(model_dir: &Path, options: Qwen3TtsOptions) -> Result<Self> {
        options.validate()?;
        let store = Qwen3TtsWeightStore::open(model_dir)?;
        let cfg = Qwen3TtsConfig::from_model_dir(store.model_dir())?;
        Ok(Self {
            cfg,
            store,
            options,
        })
    }

    pub fn config(&self) -> &Qwen3TtsConfig {
        &self.cfg
    }

    pub fn model_dir(&self) -> &Path {
        self.store.model_dir()
    }

    pub fn device(&self) -> Device {
        self.options.device
    }

    /// Micro-benchmark: synthetic talker prefill + greedy decode (same weights, no HF prompt).
    pub fn bench_talker_synthetic(
        &self,
        prefill_seq: usize,
        decode_steps: usize,
    ) -> Result<Qwen3TtsBenchReport> {
        if self.options.eager_talker {
            bail!("eager talker not implemented yet — use compiled (default)");
        }
        let mut talker = TalkerEngine::open(&self.store, self.cfg.talker(), self.options.device)?;
        let hidden = self.cfg.talker().hidden_size;
        let mut prefill = Array2::<f32>::zeros((prefill_seq.max(1), hidden));
        for (i, v) in prefill.iter_mut().enumerate() {
            *v = ((i % 97) as f32) * 1e-4;
        }
        talker.warmup(prefill_seq.max(1))?;

        let t0 = Instant::now();
        talker.reset_kv();
        talker.prefill(prefill.view())?;
        let talker_prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let mut frames = 0usize;
        let mut talker_decode_ms = 0.0;
        if decode_steps > 0 {
            let t_dec = Instant::now();
            for step in 0..decode_steps {
                let mut emb = vec![0f32; hidden];
                emb[0] = (step as f32) * 1e-3;
                let (_h, tok) = talker.decode_step(ndarray::ArrayView1::from(&emb))?;
                if tok == talker.codec_eos() {
                    break;
                }
                frames += 1;
            }
            talker_decode_ms = t_dec.elapsed().as_secs_f64() * 1000.0;
        }

        Ok(Qwen3TtsBenchReport {
            device: self.options.device,
            eager_talker: talker.is_eager(),
            talker_prefill_ms,
            talker_decode_ms,
            code_predictor_ms: 0.0,
            vocoder_ms: 0.0,
            synthesis_ms: talker_prefill_ms + talker_decode_ms,
            codec_frames: frames,
            pcm_samples: 0,
            talker_decode_steps: frames,
        })
    }

    /// Open a warmed session (reuse across utterances to amortize compile warmup).
    pub fn open_session(&self) -> Result<Qwen3TtsSession> {
        Qwen3TtsSession::open(self.model_dir(), self.options.device)
    }

    pub fn synthesize_custom_voice(
        &self,
        text: &str,
        speaker: &str,
        language: &str,
        out_wav: &Path,
    ) -> Result<()> {
        let mut gen_cfg = GenerationConfig::greedy_for_model_dir(self.model_dir())?;
        let budget = crate::synth_opts::codec_frame_budget(
            text,
            gen_cfg.max_new_tokens,
            self.options.max_frames,
        );
        gen_cfg.max_new_tokens = budget;
        if self.options.max_frames == 0 {
            eprintln!(
                "[qwen3-tts] codec frames: auto budget {budget} (stops at talker EOS; --max-frames to cap)"
            );
        }
        let result = synthesize_custom_voice_greedy(
            self.model_dir(),
            &self.cfg,
            &self.store,
            self.options.device,
            text,
            speaker,
            language,
            &gen_cfg,
            false,
        )?;
        write_wav_mono(out_wav, &result.pcm, result.sample_rate)?;
        let peak = result.pcm.iter().map(|s| s.abs()).fold(0f32, f32::max);
        let rms = if result.pcm.is_empty() {
            0.0
        } else {
            (result.pcm.iter().map(|s| s * s).sum::<f32>() / result.pcm.len() as f32).sqrt()
        };
        let dur = result.pcm.len() as f64 / result.sample_rate as f64;
        println!(
            "audio: {dur:.2}s @ {} Hz, {} samples, peak={peak:.3}, rms={rms:.4}",
            result.sample_rate,
            result.pcm.len()
        );
        if peak < 0.01 {
            eprintln!(
                "[qwen3-tts] warning: output is nearly silent (peak={peak:.6}). \
                 For finetuned JFK use `just qwen3-tts-jfk-hf-demo`. \
                 On MLX, speech decode defaults to CPU eager; set RLX_QWEN3_TTS_SPEECH_COMPILED=1 only for experiments."
            );
        } else if peak < 0.25 && crate::synth_opts::wav_peak_normalize_enabled() {
            eprintln!("[qwen3-tts] note: quiet PCM (peak={peak:.3}) — WAV scaled to ~0.95 peak");
        } else if peak < 0.25 {
            eprintln!(
                "[qwen3-tts] note: quiet PCM (peak={peak:.3}); set RLX_QWEN3_TTS_WAV_NORMALIZE=1 to boost level"
            );
        }
        Ok(())
    }
}

pub fn write_wav_mono(path: &Path, pcm: &[f32], sample_rate: u32) -> Result<()> {
    let mut pcm = pcm.to_vec();
    if crate::synth_opts::wav_peak_normalize_enabled() {
        let peak = pcm.iter().map(|s| s.abs()).fold(0f32, f32::max);
        if peak > 1e-6 && peak < 0.5 {
            let scale = 0.95 / peak;
            for s in &mut pcm {
                *s = (*s * scale).clamp(-1.0, 1.0);
            }
        }
    }
    let mut bytes = Vec::with_capacity(44 + pcm.len() * 2);
    bytes.extend_from_slice(b"RIFF");
    let data_bytes = (pcm.len() * 2 + 36) as u32;
    bytes.extend_from_slice(&data_bytes.to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&((pcm.len() * 2) as u32).to_le_bytes());
    for &s in &pcm {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes)?;
    Ok(())
}
