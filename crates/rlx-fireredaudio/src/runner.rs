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

//! FireRedAudio end-to-end runner — understand (ASR / QA) + generation scaffolding.

use crate::audio::{AudioGeometry, MelSpectrogram, pcm_to_log_mel};
use crate::config::FireRedAudioConfig;
use crate::embed::{argmax_token, count_audio_placeholders, fuse_inputs_embeds};
use crate::encoder::build_encoder_built;
use crate::hf_config::load_qwen35_backbone;
use crate::load::{WeightStore, resolve_model_dir};
use crate::prompt::{
    DEFAULT_ASR_PROMPT, EditType, build_asr_prompt, build_edit_prompt, build_tts_prompt,
    build_understand_prompt, build_voice_design_prompt, split_thinking,
};
use crate::tokenizer::FireRedTokenizer;
use anyhow::{Context, Result, bail, ensure};
use rlx_core::flow_util::compile_built;
use rlx_qwen35::{
    MultimodalPrefill, Qwen35Runner, Qwen35RunnerBuilder, SampleOpts, text_section_pos,
};
use rlx_runtime::{CompiledGraph, Device};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone, Default)]
pub struct FireRedRunnerBuilder {
    weights: Option<PathBuf>,
    config_path: Option<PathBuf>,
    config: Option<FireRedAudioConfig>,
    device: Option<Device>,
    max_new_tokens: usize,
    max_seq: usize,
}

impl FireRedRunnerBuilder {
    pub fn weights(mut self, p: impl Into<PathBuf>) -> Self {
        self.weights = Some(p.into());
        self
    }
    pub fn config_path(mut self, p: impl Into<PathBuf>) -> Self {
        self.config_path = Some(p.into());
        self
    }
    pub fn config(mut self, c: FireRedAudioConfig) -> Self {
        self.config = Some(c);
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }
    pub fn max_new_tokens(mut self, n: usize) -> Self {
        self.max_new_tokens = n;
        self
    }
    pub fn max_seq(mut self, n: usize) -> Self {
        self.max_seq = n;
        self
    }

    pub fn build(self) -> Result<FireRedRunner> {
        let weights_path = self
            .weights
            .ok_or_else(|| anyhow::anyhow!("weights path required"))?;
        let model_dir = resolve_model_dir(&weights_path)?;
        // Nested HF layout: …/FireRedAudio/{config,tokenizer,shards}
        let model_dir = if model_dir.join("config.json").is_file() {
            model_dir
        } else if model_dir.join("FireRedAudio/config.json").is_file() {
            model_dir.join("FireRedAudio")
        } else {
            model_dir
        };

        let cfg = match self.config {
            Some(c) => c,
            None => {
                let p = self
                    .config_path
                    .clone()
                    .unwrap_or_else(|| model_dir.join("config.json"));
                FireRedAudioConfig::from_file(&p)?
            }
        };
        cfg.validate()?;

        let device = self.device.unwrap_or(Device::Cpu);
        let max_new_tokens = if self.max_new_tokens == 0 {
            300
        } else {
            self.max_new_tokens
        };
        let max_seq = if self.max_seq == 0 {
            4096
        } else {
            self.max_seq
        };

        let tokenizer = FireRedTokenizer::from_model_dir(&model_dir)?;
        let store = WeightStore::open(&model_dir)?;

        eprintln!(
            "[fireredaudio] loading Qwen3.5 backbone ({} layers, hidden={})…",
            cfg.backbone.num_hidden_layers, cfg.backbone.hidden_size
        );
        let (qcfg, qweights) = load_qwen35_backbone(&model_dir, &cfg)?;
        let token_embd = Arc::clone(&qweights.token_embd);
        let runner = Qwen35RunnerBuilder::default()
            .inline_weights(qcfg, qweights)
            .device(device)
            .batch(1)
            .max_seq(max_seq)
            .prefill_seq(max_seq.min(2048))
            .hidden_prefill(true)
            .force_host_embed(true)
            .fast_greedy_lm_head(true)
            .skip_auto_mmproj(true)
            .skip_warm(true)
            .build()
            .context("build Qwen3.5 backbone runner")?;

        Ok(FireRedRunner {
            cfg,
            device,
            max_new_tokens,
            store,
            tokenizer,
            token_embd,
            backbone: RefCell::new(runner),
            encoder_cache: RefCell::new(HashMap::new()),
        })
    }
}

pub struct FireRedRunner {
    cfg: FireRedAudioConfig,
    device: Device,
    max_new_tokens: usize,
    store: WeightStore,
    tokenizer: FireRedTokenizer,
    token_embd: Arc<[f32]>,
    backbone: RefCell<Qwen35Runner>,
    encoder_cache: RefCell<HashMap<(usize, usize), CompiledGraph>>,
}

impl FireRedRunner {
    pub fn builder() -> FireRedRunnerBuilder {
        FireRedRunnerBuilder::default()
    }

    pub fn config(&self) -> &FireRedAudioConfig {
        &self.cfg
    }

    pub fn model_dir(&self) -> &Path {
        self.store.model_dir()
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Mel → audio encoder → `[n_tok * hidden]` row-major.
    pub fn encode_audio(&self, mel: &MelSpectrogram) -> Result<(Vec<f32>, usize)> {
        ensure!(
            mel.n_mels == self.cfg.audio_encoder.num_mel_bins,
            "mel bins {} != configured {}",
            mel.n_mels,
            self.cfg.audio_encoder.num_mel_bins
        );
        let geom = AudioGeometry::new(&self.cfg.audio_encoder, mel.n_frames)?;
        let padded = pad_mel(&self.cfg, mel, &geom);

        let key = (geom.num_chunks, geom.max_chunk_len);
        let mut caches = self.encoder_cache.borrow_mut();
        if let std::collections::hash_map::Entry::Vacant(e) = caches.entry(key) {
            let mut wm = self.store.load_audio_weights()?;
            let built = build_encoder_built(&self.cfg.audio_encoder, &mut wm, &geom)?;
            let params = built.params().clone();
            let mut compiled = compile_built(built, self.device)?;
            for (n, d) in &params {
                compiled.set_param(n, d);
            }
            e.insert(compiled);
        }
        let compiled = caches.get_mut(&key).expect("encoder cached");
        let out = compiled
            .run(&[("mel", padded.as_slice())])
            .into_iter()
            .next()
            .context("audio encoder output")?;
        Ok((out, geom.num_audio_tokens))
    }

    /// Expand ChatML (one `<|AUDIO|>` per clip) to encoder token count, tokenize.
    pub fn tokenize_understand_prompt(
        &self,
        chatml: &str,
        n_audio_tokens: usize,
    ) -> Result<Vec<u32>> {
        let tok = self.cfg.tokens.audio_special_token;
        let count = chatml.matches(tok).count();
        ensure!(
            count >= 1,
            "understand prompt must contain at least one {tok}"
        );
        // Replace each single placeholder with n_audio_tokens / count copies
        // (usually one audio → all tokens for that clip).
        ensure!(
            n_audio_tokens.is_multiple_of(count) || count == 1,
            "cannot split {n_audio_tokens} audio tokens across {count} placeholders"
        );
        let per = if count == 1 {
            n_audio_tokens
        } else {
            n_audio_tokens / count
        };
        let expanded = chatml.replacen(tok, &tok.repeat(per), count);
        self.tokenizer.encode(&expanded)
    }

    /// Prefill fused embeds + greedy decode until EOS / budget.
    pub fn generate_text(&self, prompt_ids: &[u32], audio_embeds: &[f32]) -> Result<Vec<u32>> {
        let n_ph = count_audio_placeholders(&self.cfg, prompt_ids);
        let h = self.cfg.backbone.hidden_size;
        let n_vecs = audio_embeds.len() / h;
        ensure!(
            n_ph == n_vecs,
            "prompt has {n_ph} <|AUDIO|> slots, encoder produced {n_vecs} vectors"
        );

        let hidden = fuse_inputs_embeds(&self.cfg, &self.token_embd, prompt_ids, audio_embeds)?;
        let seq = prompt_ids.len();
        let prefill = MultimodalPrefill {
            hidden,
            mrope_sections: (0..seq).map(text_section_pos).collect(),
            last_token_idx: seq.saturating_sub(1),
            seq: prompt_ids.to_vec(),
        };

        let mut backbone = self.backbone.borrow_mut();
        let seed = backbone.prefill_from_assembled(prefill)?;
        let eos = self.tokenizer.eos_id();
        let next = argmax_token(&seed.trunk_logits);
        let mut generated = Vec::with_capacity(self.max_new_tokens);
        generated.push(next);
        if next == eos || generated.len() >= self.max_new_tokens {
            return Ok(generated);
        }
        let rest = backbone.generate_continue(
            &[next],
            self.max_new_tokens - 1,
            SampleOpts::greedy(),
            |t| t != eos,
        )?;
        generated.extend(rest);
        // Drop trailing EOS if present.
        if generated.last().copied() == Some(eos) {
            generated.pop();
        }
        Ok(generated)
    }

    pub fn understand_pcm(
        &self,
        pcm_16k: &[f32],
        question: &str,
        enable_thinking: bool,
    ) -> Result<UnderstandResult> {
        let mel = pcm_to_log_mel(pcm_16k, self.cfg.audio_encoder.num_mel_bins)?;
        let (audio_embeds, n_tok) = self.encode_audio(&mel)?;
        let chatml = build_understand_prompt(
            question,
            1,
            self.cfg.tokens.audio_special_token,
            enable_thinking,
        )?;
        let ids = self.tokenize_understand_prompt(&chatml, n_tok)?;
        let out_ids = self.generate_text(&ids, &audio_embeds)?;
        let text = self.tokenizer.decode(&out_ids)?;
        let (reasoning, answer) = split_thinking(&text);
        Ok(UnderstandResult {
            answer: answer.to_string(),
            reasoning: reasoning.map(str::to_string),
            token_ids: out_ids,
        })
    }

    pub fn asr_pcm(&self, pcm_16k: &[f32]) -> Result<UnderstandResult> {
        let mel = pcm_to_log_mel(pcm_16k, self.cfg.audio_encoder.num_mel_bins)?;
        let (audio_embeds, n_tok) = self.encode_audio(&mel)?;
        let chatml = build_asr_prompt(1, self.cfg.tokens.audio_special_token)?;
        let ids = self.tokenize_understand_prompt(&chatml, n_tok)?;
        let out_ids = self.generate_text(&ids, &audio_embeds)?;
        let text = self.tokenizer.decode(&out_ids)?;
        Ok(UnderstandResult {
            answer: text.trim().to_string(),
            reasoning: None,
            token_ids: out_ids,
        })
    }

    pub fn asr_wav(&self, wav: &Path) -> Result<UnderstandResult> {
        let pcm = rlx_whisper::load_wav_mono_f32(wav)?;
        self.asr_pcm(&pcm)
    }

    pub fn understand_wav(
        &self,
        wav: &Path,
        question: &str,
        enable_thinking: bool,
    ) -> Result<UnderstandResult> {
        let pcm = rlx_whisper::load_wav_mono_f32(wav)?;
        self.understand_pcm(&pcm, question, enable_thinking)
    }

    /// Generation tasks need RedAE decoder + DiT hybrid AR — not yet compiled.
    pub fn tts(
        &self,
        _prompt_text: &str,
        _prompt_wav: &Path,
        _target_text: &str,
        _language: &str,
    ) -> Result<GenerationResult> {
        let _ = (
            build_tts_prompt,
            build_edit_prompt,
            build_voice_design_prompt,
            EditType::Semantic,
            DEFAULT_ASR_PROMPT,
        );
        bail!(
            "tts / edit / voice_design generation path is wired at the API layer but the \
             RedAE+DiT hybrid AR graphs are not compiled yet; ASR/understand run end-to-end"
        )
    }
}

#[derive(Debug, Clone)]
pub struct UnderstandResult {
    pub answer: String,
    pub reasoning: Option<String>,
    pub token_ids: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub pcm_24k: Vec<f32>,
    pub text: Option<String>,
}

fn pad_mel(cfg: &FireRedAudioConfig, mel: &MelSpectrogram, geom: &AudioGeometry) -> Vec<f32> {
    let n_mels = cfg.audio_encoder.num_mel_bins;
    let target_t = geom.num_chunks * geom.max_chunk_len;
    let mut out = vec![0f32; n_mels * target_t];
    let copy_t = mel.n_frames.min(target_t);
    for m in 0..n_mels {
        let src = m * mel.n_frames;
        let dst = m * target_t;
        out[dst..dst + copy_t].copy_from_slice(&mel.data[src..src + copy_t]);
    }
    out
}
