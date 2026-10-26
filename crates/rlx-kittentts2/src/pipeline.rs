// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Text → 24 kHz audio: chunk, prompt, sample codec tokens, decode, join.
//!
//! Mirrors `KittenTTS2.generate`, for a built-in or cloned voice: the text is cleaned
//! ([`crate::text::normalize_text`]), split on sentences into chunks the model
//! handles reliably, and each chunk is generated with the voice's reference
//! prefix and speaker vector, decoded by S3Gen against the same voice, and the
//! pieces joined with one consistent pause.

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rlx_runtime::Device;
use tokenizers::Tokenizer;

use crate::config::Kitten2Config;
use crate::lm::{SpeechLm, StopReason};
use crate::reference::ReferenceEncoder;
use crate::s3gen::S3Gen;
use crate::sampling::{Rng, SamplerOpts};
use crate::text;
use crate::voices::{Voice, VoiceLibrary};

/// The codec runs at 25 Hz.
pub const TOKENS_PER_SECOND: f32 = 25.0;
/// Upstream's measured delivery rate, used to size a chunk's token budget.
const CHARS_PER_SECOND: f32 = 20.0;
const CHUNK_BUDGET_SLACK: f32 = 1.8;
const MIN_CHUNK_BUDGET: usize = 200;

/// Hugging Face repository of the S3Gen weights.
pub const S3GEN_REPO: &str = "ResembleAI/chatterbox-turbo";
pub const S3GEN_FILE: &str = "s3gen_meanflow.safetensors";

/// Per-request settings; `None` defers to the repository's config.
#[derive(Debug, Clone)]
pub struct GenerateOpts {
    pub voice: Option<String>,
    pub preset: Option<String>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    /// Per-chunk token budget; 25 tokens ≈ one second.
    pub max_new_tokens: usize,
    pub seed: u64,
    /// Include the reference clip's own tokens in the prompt.
    pub use_reference_prompt: Option<bool>,
    /// Force the `{emo: 1}` control on or off (default: on when the text
    /// carries an expression tag).
    pub use_emotion: Option<bool>,
    pub greedy: bool,
    /// Speak written forms out ("$12.50" → "twelve dollars fifty cents").
    /// English grammars: turn off for other languages, as upstream advises.
    pub normalize: bool,
    pub chunk_chars: Option<usize>,
    pub chunk_min_chars: Option<usize>,
    pub chunk_gap_s: f32,
}

impl Default for GenerateOpts {
    fn default() -> Self {
        Self {
            voice: None,
            preset: None,
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            max_new_tokens: 1000,
            seed: 0,
            use_reference_prompt: None,
            use_emotion: None,
            greedy: false,
            normalize: true,
            chunk_chars: None,
            chunk_min_chars: None,
            chunk_gap_s: text::CHUNK_GAP_S,
        }
    }
}

/// What one chunk produced, for logging.
#[derive(Debug, Clone)]
pub struct ChunkReport {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated: usize,
    pub audio_tokens: usize,
    pub stop: StopReason,
    /// The chunk's 0-based S3 codec ids, as handed to S3Gen.
    pub audio: Vec<u32>,
}

/// A loaded KittenTTS 2.
pub struct KittenTts2 {
    cfg: Kitten2Config,
    tokenizer: Tokenizer,
    lm: SpeechLm,
    s3gen: S3Gen,
    voices: VoiceLibrary,
    repo: PathBuf,
    s3gen_path: PathBuf,
    device: Device,
    /// Built on the first [`Self::clone_voice`] (or student conditioning).
    reference: Option<ReferenceEncoder>,
    /// Student-decoder prompts, per reference clip.
    student_cond: std::collections::HashMap<PathBuf, crate::voices::S3Conditioning>,
}

/// Locate `s3gen_meanflow.safetensors`: `RLX_KITTENTTS2_S3GEN`, then a
/// `chatterbox-turbo` directory beside the model, then the HF hub cache.
pub fn resolve_s3gen(repo: &Path) -> Result<PathBuf> {
    if let Ok(p) = rlx_ir::env::var_result("RLX_KITTENTTS2_S3GEN") {
        return Ok(PathBuf::from(p));
    }
    let sibling = repo
        .parent()
        .map(|p| p.join("chatterbox-turbo").join(S3GEN_FILE));
    if let Some(p) = sibling.filter(|p| p.is_file()) {
        return Ok(p);
    }
    let home = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface")));
    if let Some(snap) = home.map(|h| h.join("hub/models--ResembleAI--chatterbox-turbo/snapshots"))
        && let Ok(rd) = std::fs::read_dir(&snap)
    {
        for e in rd.flatten() {
            let p = e.path().join(S3GEN_FILE);
            if p.is_file() {
                return Ok(p);
            }
        }
    }
    bail!(
        "{S3GEN_FILE} not found: set RLX_KITTENTTS2_S3GEN, or `hf download {S3GEN_REPO} {S3GEN_FILE} \
         --local-dir {}`",
        repo.parent()
            .unwrap_or(repo)
            .join("chatterbox-turbo")
            .display()
    )
}

/// Which flow turns codec tokens into a mel.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Decoder {
    /// chatterbox-turbo's meanflow flow, two steps (best quality).
    #[default]
    Default,
    /// A distilled student from the repository's `decoders` (e.g.
    /// `student_w4`: 4-bit, one step, 37 MB; lower fidelity).
    Named(String),
}

impl std::str::FromStr for Decoder {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(if s == "default" {
            Self::Default
        } else {
            Self::Named(s.to_string())
        })
    }
}

impl KittenTts2 {
    /// `repo` is a local copy of `KittenML/kitten-tts-2`.
    pub fn open(repo: &Path, s3gen: &Path, device: Device) -> Result<Self> {
        Self::open_with_decoder(repo, s3gen, device, Decoder::Default)
    }

    /// [`Self::open`] with a choice of flow decoder.
    pub fn open_with_decoder(
        repo: &Path,
        s3gen: &Path,
        device: Device,
        decoder: Decoder,
    ) -> Result<Self> {
        let cfg = Kitten2Config::from_dir(repo)?;
        let lm_dir = repo.join(&cfg.lm_dir);
        let ternary = repo.join(
            cfg.lm_packed
                .as_deref()
                .context("config declares no lm_packed checkpoint")?,
        );
        let tokenizer = Tokenizer::from_file(lm_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let s3_voices = repo.join(cfg.s3gen_voices_path().context(
            "config declares no S3Gen voice conditioning (cpp.decoders.default.voices)",
        )?);
        let voices = VoiceLibrary::open(&repo.join(&cfg.voices), &s3_voices)?;
        Ok(Self {
            lm: SpeechLm::open(&lm_dir, &ternary, device)?,
            s3gen: match &decoder {
                Decoder::Default => S3Gen::open(s3gen, device)?,
                Decoder::Named(name) => {
                    let rel = cfg.decoders.get(name).with_context(|| {
                        format!(
                            "unknown decoder {name:?}; available: default, {:?}",
                            cfg.decoders.keys().collect::<Vec<_>>()
                        )
                    })?;
                    S3Gen::open_student(&repo.join(rel), s3gen, device)?
                }
            },
            student_cond: Default::default(),
            tokenizer,
            voices,
            cfg,
            repo: repo.to_path_buf(),
            s3gen_path: s3gen.to_path_buf(),
            device,
            reference: None,
        })
    }

    /// Transcribe a reference clip with Whisper (`whisper_dir` holds
    /// `model.safetensors`, `config.json`, `tokenizer.json`). Upstream uses
    /// large-v3; any Whisper checkpoint `rlx-whisper` loads works.
    #[cfg(feature = "transcribe")]
    pub fn transcribe_reference(
        &self,
        wav: &Path,
        whisper_dir: &Path,
        language: &str,
    ) -> Result<String> {
        let (pcm, sr) = rlx_core::asr_bench::read_wav_mono(wav)
            .with_context(|| format!("read reference {}", wav.display()))?;
        let pcm = rlx_core::audio::resample_sinc(&pcm, sr as usize, rlx_whisper::SAMPLE_RATE);
        let mut asr = rlx_whisper::WhisperRunner::builder()
            .weights(whisper_dir.join("model.safetensors"))
            .config_path(whisper_dir.join("config.json"))
            .tokenizer_path(whisper_dir.join("tokenizer.json"))
            .device(self.device)
            .language(language)
            .build()
            .with_context(|| format!("load Whisper from {}", whisper_dir.display()))?;
        let text = asr.transcribe_greedy(&pcm)?.trim().to_string();
        if text.is_empty() {
            bail!("no speech found in the reference clip {}", wav.display());
        }
        Ok(text)
    }

    /// A [`Voice`] cloned from a reference clip (5–30 s of one speaker) and
    /// what it says. Loads the reference encoders on first use.
    pub fn clone_voice(&mut self, wav: &Path, transcript: &str) -> Result<Voice> {
        self.reference_encoder()?.voice_from_wav(wav, transcript)
    }

    fn reference_encoder(&mut self) -> Result<&ReferenceEncoder> {
        if self.reference.is_none() {
            let speaker = self.repo.join(
                self.cfg
                    .speaker_embedding
                    .as_deref()
                    .unwrap_or("speaker/model.safetensors"),
            );
            self.reference = Some(ReferenceEncoder::open(
                &self.s3gen_path,
                &speaker,
                self.device,
            )?);
        }
        Ok(self.reference.as_ref().unwrap())
    }

    /// What the active flow decoder is conditioned on for `voice`: the
    /// voice's own S3Gen prompt, or — for the student — a 4 s / 112-token
    /// cut of its reference clip, derived once and cached.
    pub fn decoder_conditioning(&mut self, voice: &Voice) -> Result<crate::voices::S3Conditioning> {
        if !self.s3gen.is_student() {
            return Ok(voice.s3.clone());
        }
        let wav = voice
            .reference_wav
            .clone()
            .context("the student decoder needs the voice's reference clip")?;
        if let Some(c) = self.student_cond.get(&wav) {
            return Ok(c.clone());
        }
        let (pcm, sr) = rlx_core::asr_bench::read_wav_mono(&wav)
            .with_context(|| format!("read reference {}", wav.display()))?;
        let c = self.reference_encoder()?.decoder_conditioning(
            &pcm,
            sr,
            crate::s3gen::student::STUDENT_REF_SECONDS,
            Some(crate::s3gen::student::STUDENT_MAX_PROMPT_TOKENS),
        )?;
        self.student_cond.insert(wav, c.clone());
        Ok(c)
    }

    pub fn config(&self) -> &Kitten2Config {
        &self.cfg
    }

    pub fn voices(&self) -> &VoiceLibrary {
        &self.voices
    }

    pub fn sample_rate(&self) -> u32 {
        self.cfg.sample_rate
    }

    pub fn lm_mut(&mut self) -> &mut SpeechLm {
        &mut self.lm
    }

    pub fn s3gen(&self) -> &S3Gen {
        &self.s3gen
    }

    pub fn encode_text(&self, s: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(s, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
            .get_ids()
            .to_vec())
    }

    /// Reference prefix for `voice` (transcript + its codec tokens).
    pub fn reference_prefix(&self, voice: &Voice) -> Result<Vec<u32>> {
        let ids = self.encode_text(voice.transcript.trim())?;
        Ok(text::build_reference_prefix(
            &self.cfg.token_map,
            &ids,
            &voice.reference_tokens,
        ))
    }

    /// Synthesize `text` with a built-in voice (`opts.voice`, else the
    /// repository default). Returns the joined waveform and per-chunk reports.
    pub fn generate(
        &mut self,
        input: &str,
        opts: &GenerateOpts,
        on_chunk: impl FnMut(&ChunkReport),
    ) -> Result<(Vec<f32>, Vec<ChunkReport>)> {
        let name = opts
            .voice
            .as_deref()
            .unwrap_or(&self.cfg.default_voice)
            .to_string();
        let voice = self.voices.load(&name)?;
        self.generate_with_voice(input, &voice, opts, on_chunk)
    }

    /// Synthesize `text` with any [`Voice`] — built-in or from
    /// [`Self::clone_voice`]. `opts.voice` is ignored.
    pub fn generate_with_voice(
        &mut self,
        input: &str,
        voice: &Voice,
        opts: &GenerateOpts,
        mut on_chunk: impl FnMut(&ChunkReport),
    ) -> Result<(Vec<f32>, Vec<ChunkReport>)> {
        let spoken = text::normalize_text(input, opts.normalize);
        if spoken.trim().is_empty() {
            bail!("no text to speak");
        }
        let g = &self.cfg.generation;
        let chunks = text::split_for_synthesis(
            &spoken,
            opts.chunk_chars.unwrap_or(g.chunk_chars),
            opts.chunk_min_chars.unwrap_or(g.chunk_min_chars),
        );
        let preset_name = opts.preset.as_deref().unwrap_or(&self.cfg.default_preset);
        let mut sampler =
            SamplerOpts::from_preset(self.cfg.preset(preset_name)?, g.repetition_window);
        sampler.temperature = opts.temperature.unwrap_or(sampler.temperature);
        sampler.top_k = opts.top_k.unwrap_or(sampler.top_k);
        sampler.top_p = opts.top_p.unwrap_or(sampler.top_p);
        sampler.min_p = opts.min_p.unwrap_or(sampler.min_p);
        sampler.greedy = opts.greedy;

        let use_ref = opts.use_reference_prompt.unwrap_or(g.use_reference_prompt);
        let prefix = if use_ref {
            Some(self.reference_prefix(voice)?)
        } else {
            None
        };
        let emotion = match (
            &g.emotion_control,
            opts.use_emotion
                .unwrap_or_else(|| text::has_expression_tag(input)),
        ) {
            (Some(ctrl), true) => Some(self.encode_text(ctrl)?),
            _ => None,
        };

        let tm = self.cfg.token_map.clone();
        let decoder_cond = self.decoder_conditioning(voice)?;
        let mut rng = Rng::new(opts.seed);
        let flow_rng = RefCell::new(Rng::new(opts.seed ^ 0x5EED_F10E));
        let mut wavs = Vec::new();
        let mut reports = Vec::new();
        let single = chunks.len() == 1;
        for chunk in &chunks {
            let budget = if single {
                opts.max_new_tokens
            } else {
                let expected = chunk.chars().count() as f32 / CHARS_PER_SECOND
                    * TOKENS_PER_SECOND
                    * CHUNK_BUDGET_SLACK;
                (expected as usize)
                    .min(opts.max_new_tokens)
                    .max(MIN_CHUNK_BUDGET)
            };
            let timing = rlx_ir::env::flag("RLX_KITTENTTS2_TIMING");
            let t_lm = std::time::Instant::now();
            let target = self.encode_text(chunk)?;
            let prompt =
                text::build_generation_prompt(&tm, &target, prefix.as_deref(), emotion.as_deref());
            let (generated, stop) = self.lm.generate(
                &prompt,
                &voice.speaker_embedding,
                &tm,
                &sampler,
                budget,
                &mut rng,
                |_| {},
            )?;
            let audio: Vec<u32> = generated
                .iter()
                .filter(|&&t| tm.is_audio(t))
                .map(|&t| t - tm.audio_id_base)
                .collect();
            let report = ChunkReport {
                text: chunk.clone(),
                prompt_tokens: prompt.len(),
                generated: generated.len(),
                audio_tokens: audio.len(),
                stop,
                audio: audio.clone(),
            };
            on_chunk(&report);
            reports.push(report);
            let t_s3 = std::time::Instant::now();
            if let Some(w) = self.s3gen.synthesize(&audio, &decoder_cond, &flow_rng)? {
                wavs.push(w);
            }
            if timing {
                eprintln!(
                    "[kittentts2] timing: lm {:.2}s ({} tokens), s3gen {:.2}s",
                    (t_s3 - t_lm).as_secs_f32(),
                    generated.len(),
                    t_s3.elapsed().as_secs_f32()
                );
            }
        }
        let wav = text::join_chunks(&wavs, self.cfg.sample_rate, opts.chunk_gap_s)
            .context("the model produced no audio tokens — try different text or another seed")?;
        Ok((wav, reports))
    }
}

/// Write mono f32 PCM as a 16-bit wav.
pub fn write_wav(path: &Path, pcm: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create {}", path.display()))?;
    for &s in pcm {
        w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16)?;
    }
    w.finalize()?;
    Ok(())
}
