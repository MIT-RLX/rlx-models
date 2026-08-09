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

//! High-level voice-clone API.
//!
//! Two-step workflow:
//!   1. `extract_reference(wav)` — encode a reference WAV into a
//!      reusable [`SpeakerReference`] (1024-d ECAPA x-vector).
//!   2. `generate(&reference, text)` or `generate_to_wav(...)` —
//!      synthesize speech in that voice.
//!
//! A reference can be saved to / loaded from JSON for reuse across
//! processes — useful when you want to clone the same voice many
//! times without re-encoding the source audio (~50 ms saved per
//! generation).
//!
//! Example:
//! ```no_run
//! use rlx_qwen3_tts::VoiceClone;
//! use rlx_runtime::Device;
//! # fn run() -> anyhow::Result<()> {
//! let mut tts = VoiceClone::open(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base", Device::Metal)?;
//! let reference = tts.extract_reference("speaker.wav")?;
//! reference.save_json("speaker.ref.json")?;
//! tts.generate_to_wav(&reference, "Hello, world.", "out.wav")?;
//! # Ok(()) }
//! ```

type FrameCallback<'a> = Option<&'a mut dyn FnMut(usize, &[u32])>;

use crate::config::Qwen3TtsConfig;
use crate::load::Qwen3TtsWeightStore;
use crate::megakernel::Qwen3TtsMegakernel;
use crate::prompt::load_text_tokenizer;
use crate::speaker_encoder;
use crate::speech_tokenizer::{St12HzDecoder, open_speech_decoder_for_frames};
use crate::stream::{
    ChunkEmitter, PcmChunk, StreamConfig, StreamControl, StreamEvent, StreamMode, StreamStats,
};
use crate::text_embed::TextEmbedder;
use crate::voice_clone::build_x_vector_prompt;
use anyhow::{Context, Result, bail};
use rlx_runtime::Device;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;

/// Reusable speaker representation extracted from a reference WAV.
///
/// Currently holds a 1024-d ECAPA-TDNN x-vector. JSON-serializable so it can
/// be baked once and reused — e.g. ship a `jfk.ref.json` with your app and
/// generate speech without bundling the original WAV.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpeakerReference {
    /// Schema version of this file (bump on breaking changes).
    pub version: u32,
    /// ECAPA-TDNN x-vector (1024-d for the Qwen3-TTS Base model).
    pub x_vector: Vec<f32>,
    /// Source WAV path, if known. Informational only.
    pub source_wav: Option<PathBuf>,
    /// Free-form note (speaker name, recording context, etc.).
    pub note: Option<String>,
}

impl SpeakerReference {
    const VERSION: u32 = 1;

    /// Dimensionality of the x-vector.
    pub fn dim(&self) -> usize {
        self.x_vector.len()
    }

    /// L2 norm of the x-vector — sanity check that extraction succeeded.
    pub fn norm(&self) -> f32 {
        self.x_vector.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Cosine similarity to another reference. > 0.7 typically indicates
    /// "same speaker" by Voxceleb baselines; > 0.9 is essentially same
    /// recording session.
    pub fn cosine(&self, other: &SpeakerReference) -> f32 {
        let n = self.x_vector.len().min(other.x_vector.len());
        let mut dot = 0f32;
        let mut na = 0f32;
        let mut nb = 0f32;
        for i in 0..n {
            dot += self.x_vector[i] * other.x_vector[i];
            na += self.x_vector[i] * self.x_vector[i];
            nb += other.x_vector[i] * other.x_vector[i];
        }
        if na <= 0.0 || nb <= 0.0 {
            return 0.0;
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    /// Persist as JSON for reuse across processes / shipping with an app.
    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let json = serde_json::to_string_pretty(self).context("serialize reference")?;
        std::fs::write(path.as_ref(), json)
            .with_context(|| format!("write {}", path.as_ref().display()))?;
        Ok(())
    }

    /// Load a previously saved reference.
    pub fn load_json(path: impl AsRef<Path>) -> Result<Self> {
        let json = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("read {}", path.as_ref().display()))?;
        let r: Self = serde_json::from_str(&json).context("parse reference")?;
        if r.version > Self::VERSION {
            bail!(
                "reference file version {} not understood (max supported: {})",
                r.version,
                Self::VERSION
            );
        }
        Ok(r)
    }
}

/// High-level voice-clone TTS — opens the model once, then clone a voice
/// (`extract_reference`) and synthesize text in that voice (`generate`).
///
/// Holds the talker + code predictor + speech decoder all warm and ready.
/// Defaults to sampled decoding (top-k = 50, temperature = 0.9), which is
/// required for the talker to produce intelligible speech.
pub struct VoiceClone {
    cfg: Qwen3TtsConfig,
    store: Qwen3TtsWeightStore,
    text_embedder: TextEmbedder,
    tokenizer: Tokenizer,
    mk: Qwen3TtsMegakernel,
    decoder: St12HzDecoder,
    model_dir: PathBuf,
    device: Device,
    max_frames: usize,
}

impl VoiceClone {
    /// Open the Qwen3-TTS Base model from `model_dir` on the given device.
    ///
    /// This is the slow step (~1 second on Apple M3 Pro Metal). Reuse the
    /// returned `VoiceClone` for as many clones / generations as you need.
    ///
    /// Also flips on top-k sampling defaults (`RLX_QWEN3_TTS_SAMPLE=1`,
    /// `TEMP=0.9`, `TOP_K=50`) unless the caller has already set them. The
    /// model is trained for sampling and produces unintelligible audio with
    /// greedy decoding.
    pub fn open(model_dir: impl Into<PathBuf>, device: Device) -> Result<Self> {
        Self::open_with_max_frames(model_dir, device, 256)
    }

    /// As [`Self::open`] but allows setting a different max-frame cap (sets
    /// the upper bound on how long any single generated utterance can run
    /// — at 12 Hz codec rate, 256 frames ≈ 21 s of audio).
    pub fn open_with_max_frames(
        model_dir: impl Into<PathBuf>,
        device: Device,
        max_frames: usize,
    ) -> Result<Self> {
        // Sampling defaults — required for the model to produce intelligible
        // audio. Caller can override by setting these before calling open().
        for (k, v) in [
            ("RLX_QWEN3_TTS_SAMPLE", "1"),
            ("RLX_QWEN3_TTS_TEMP", "0.9"),
            ("RLX_QWEN3_TTS_TOP_K", "50"),
            ("RLX_QWEN3_TTS_WAV_NORMALIZE", "1"),
        ] {
            if std::env::var(k).is_err() {
                // SAFETY: at process startup, no other threads running.
                unsafe {
                    std::env::set_var(k, v);
                }
            }
        }
        let model_dir = model_dir.into();
        let cfg = Qwen3TtsConfig::from_model_dir(&model_dir)?;
        let store = Qwen3TtsWeightStore::open(&model_dir)?;
        let text_embedder = TextEmbedder::open(&store)?;
        let tokenizer = load_text_tokenizer(&model_dir)?;
        let mk = Qwen3TtsMegakernel::open(&store, cfg.talker(), cfg.code_predictor(), device)?;
        let mut decoder = open_speech_decoder_for_frames(&model_dir, device, max_frames)?;
        // Pre-warm decoder for the chunk-size horizons that Progressive streaming
        // commonly produces, so the first partial-decode in live mode doesn't pay
        // per-shape compile cost. The full-horizon warmup above covers the final
        // batched decode path.
        for &n in &[4, 8, 16, 32, 64] {
            if n <= max_frames {
                let _ = decoder.warmup(device, Some(n));
            }
        }
        Ok(Self {
            cfg,
            store,
            text_embedder,
            tokenizer,
            mk,
            decoder,
            model_dir,
            device,
            max_frames,
        })
    }

    /// Extract a [`SpeakerReference`] from a 24 kHz mono WAV. Fast (~50 ms).
    pub fn extract_reference(&self, wav: impl AsRef<Path>) -> Result<SpeakerReference> {
        let wav = wav.as_ref();
        let x = speaker_encoder::encode_reference_wav(&self.model_dir, &self.store, wav)?;
        Ok(SpeakerReference {
            version: SpeakerReference::VERSION,
            x_vector: x,
            source_wav: Some(wav.to_path_buf()),
            note: None,
        })
    }

    /// Generate speech in the reference voice and return raw 24 kHz mono f32 PCM.
    pub fn generate(
        &mut self,
        reference: &SpeakerReference,
        target_text: &str,
    ) -> Result<Vec<f32>> {
        self.synthesize_utterance(reference, target_text, None)
    }

    /// Shared synthesis core for [`Self::generate`] and streaming paths.
    fn synthesize_utterance(
        &mut self,
        reference: &SpeakerReference,
        target_text: &str,
        mut on_frame: FrameCallback<'_>,
    ) -> Result<Vec<f32>> {
        let prompt = build_x_vector_prompt(
            &self.cfg,
            &self.store,
            &self.text_embedder,
            &self.tokenizer,
            target_text,
            &reference.x_vector,
        )?;
        self.mk.invalidate_warmup_hidden();
        self.mk
            .warmup(prompt.embeds.view(), self.max_frames, None)?;
        let (frames, _timings) = if let Some(cb) = on_frame.as_mut() {
            self.mk.synthesize_codec_ar_streaming(
                prompt.embeds.view(),
                self.cfg.talker(),
                self.max_frames,
                4,
                1.0,
                &prompt.tts_pad_embed,
                None,
                cb,
            )?
        } else {
            self.mk.synthesize_codec_ar(
                prompt.embeds.view(),
                self.cfg.talker(),
                self.max_frames,
                4,
                1.0,
                &prompt.tts_pad_embed,
                None,
            )?
        };
        let pcm = self.decoder.decode(&frames, self.device)?;
        self.mk.invalidate_warmup_hidden();
        Ok(pcm)
    }

    /// Chunk a fully synthesized PCM buffer for streaming callbacks.
    fn emit_stream_pcm(
        pcm: &[f32],
        chunk_samples: usize,
        start: Instant,
        on_event: &mut dyn FnMut(StreamEvent) -> StreamControl,
    ) -> Result<StreamStats> {
        let mut emitter = ChunkEmitter::new(chunk_samples, start);
        let _ = emitter.drain(pcm, 0, true, on_event);
        let mut stats = emitter.finalize(pcm.len() / 1920 + 1);
        stats.samples_emitted = pcm.len();
        stats.audio_secs = pcm.len() as f64 / 24_000.0;
        Ok(stats)
    }

    /// Generate speech in the reference voice and write to a WAV file.
    pub fn generate_to_wav(
        &mut self,
        reference: &SpeakerReference,
        target_text: &str,
        out_wav: impl AsRef<Path>,
    ) -> Result<()> {
        let pcm = self.generate(reference, target_text)?;
        crate::runner::write_wav_mono(out_wav.as_ref(), &pcm, 24_000)?;
        Ok(())
    }

    /// Generate speech using talker speculative decoding with a **learned
    /// draft model** (`cfg(feature = "speculative-decode")`).
    ///
    /// The `draft` is a separately-loaded N-layer Qwen3-shaped sidecar
    /// (see [`crate::talker::learned_draft::LearnedDraft::open`]). It must
    /// share the verifier talker's `hidden_size`, head config, and RoPE
    /// params — the v1 constraint — so that `codec_emb_t` can flow into
    /// the draft directly and the draft's hidden states feed back through
    /// the verifier's `codec_head`.
    #[cfg(feature = "speculative-decode")]
    pub fn generate_speculative_learned_draft(
        &mut self,
        reference: &SpeakerReference,
        target_text: &str,
        draft: &mut crate::talker::learned_draft::LearnedDraft,
        draft_len: usize,
    ) -> Result<(Vec<f32>, crate::talker::speculative::SpecRunStats)> {
        use crate::megakernel_speculative::SpecConfig;
        let prompt = build_x_vector_prompt(
            &self.cfg,
            &self.store,
            &self.text_embedder,
            &self.tokenizer,
            target_text,
            &reference.x_vector,
        )?;
        self.mk.invalidate_warmup_hidden();
        self.mk
            .warmup(prompt.embeds.view(), self.max_frames, None)?;
        let mut stub = crate::talker::speculative::TrivialDraft;
        let cfg = SpecConfig::new(&mut stub, 1.0, &prompt.tts_pad_embed, 4, self.max_frames)
            .with_draft_len(draft_len)
            .with_learned_draft(draft);
        let run = self.mk.synthesize_codec_ar_speculative(
            prompt.embeds.view(),
            self.cfg.talker(),
            cfg,
        )?;
        let pcm = self.decoder.decode(&run.codec_frames, self.device)?;
        Ok((pcm, run.stats))
    }

    /// Generate speech using talker speculative decoding with
    /// **self-speculative early-exit drafting** — the talker's own first
    /// `early_exit_layers` transformer layers act as the draft model.
    /// No separate training; no extra weights; predictions are inherently
    /// correlated with the verifier's. (`cfg(feature = "speculative-decode")`)
    #[cfg(feature = "speculative-decode")]
    pub fn generate_speculative_early_exit(
        &mut self,
        reference: &SpeakerReference,
        target_text: &str,
        early_exit_layers: usize,
        draft_len: usize,
    ) -> Result<(Vec<f32>, crate::talker::speculative::SpecRunStats)> {
        use crate::megakernel_speculative::SpecConfig;
        let prompt = build_x_vector_prompt(
            &self.cfg,
            &self.store,
            &self.text_embedder,
            &self.tokenizer,
            target_text,
            &reference.x_vector,
        )?;
        self.mk.invalidate_warmup_hidden();
        self.mk
            .warmup(prompt.embeds.view(), self.max_frames, None)?;
        // The DraftModel dispatch is bypassed by `early_exit_layers`, but
        // the field is non-optional — pass a stub TrivialDraft.
        let mut stub = crate::talker::speculative::TrivialDraft;
        let cfg = SpecConfig::new(&mut stub, 1.0, &prompt.tts_pad_embed, 4, self.max_frames)
            .with_draft_len(draft_len)
            .with_early_exit_layers(early_exit_layers);
        let run = self.mk.synthesize_codec_ar_speculative(
            prompt.embeds.view(),
            self.cfg.talker(),
            cfg,
        )?;
        let pcm = self.decoder.decode(&run.codec_frames, self.device)?;
        Ok((pcm, run.stats))
    }

    /// Generate speech in the reference voice using talker speculative
    /// decoding (EXPERIMENTAL — `cfg(feature = "speculative-decode")`).
    ///
    /// Same output contract as [`Self::generate`], plus the speculative
    /// run statistics ([`crate::talker::speculative::SpecRunStats`]) so the
    /// caller can see how often the draft's proposals matched the big
    /// talker's argmax. See [`crate::megakernel_speculative`] for the
    /// algorithmic details and the first-cut trivial-draft approximation.
    ///
    /// Requires the eager talker backend (no GPU rollback yet) and the
    /// non-fused CP path — returns an error otherwise.
    #[cfg(feature = "speculative-decode")]
    pub fn generate_speculative<D: crate::talker::speculative::DraftModel>(
        &mut self,
        reference: &SpeakerReference,
        target_text: &str,
        draft: &mut D,
        draft_len: usize,
    ) -> Result<(Vec<f32>, crate::talker::speculative::SpecRunStats)> {
        use crate::megakernel_speculative::SpecConfig;
        let prompt = build_x_vector_prompt(
            &self.cfg,
            &self.store,
            &self.text_embedder,
            &self.tokenizer,
            target_text,
            &reference.x_vector,
        )?;
        self.mk.invalidate_warmup_hidden();
        self.mk
            .warmup(prompt.embeds.view(), self.max_frames, None)?;
        let cfg = SpecConfig::new(
            draft,
            1.0, // repetition penalty (matches `generate`)
            &prompt.tts_pad_embed,
            4, // min_frames before EOS
            self.max_frames,
        )
        .with_draft_len(draft_len);
        let run = self.mk.synthesize_codec_ar_speculative(
            prompt.embeds.view(),
            self.cfg.talker(),
            cfg,
        )?;
        let pcm = self.decoder.decode(&run.codec_frames, self.device)?;
        Ok((pcm, run.stats))
    }

    /// Talker config for the loaded model (hidden size, vocab, codec EOS, …).
    pub fn talker_config(&self) -> &crate::config::TalkerConfig {
        self.cfg.talker()
    }

    /// Device that this model is running on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Model directory this instance was opened from.
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// Streaming generation with talker speculative decoding
    /// (`cfg(feature = "speculative-decode")`).
    ///
    /// Same streaming modes as [`Self::generate_stream`] — Progressive
    /// interleaves partial decode with speculative AR for lower time-to-first-
    /// audio while the talker verify batch amortises decode cost.
    #[cfg(feature = "speculative-decode")]
    pub fn generate_speculative_stream<D: crate::talker::speculative::DraftModel, F>(
        &mut self,
        reference: &SpeakerReference,
        target_text: &str,
        draft: &mut D,
        draft_len: usize,
        config: StreamConfig,
        on_event: F,
    ) -> Result<(StreamStats, crate::talker::speculative::SpecRunStats)>
    where
        F: FnMut(StreamEvent) -> StreamControl,
    {
        use crate::megakernel_speculative::SpecConfig;
        let prompt = build_x_vector_prompt(
            &self.cfg,
            &self.store,
            &self.text_embedder,
            &self.tokenizer,
            target_text,
            &reference.x_vector,
        )?;
        self.mk.invalidate_warmup_hidden();
        self.mk
            .warmup(prompt.embeds.view(), self.max_frames, None)?;
        let mut spec_cfg = SpecConfig::new(draft, 1.0, &prompt.tts_pad_embed, 4, self.max_frames)
            .with_draft_len(draft_len);
        let start = Instant::now();
        let mut cb = on_event;
        let (stream_stats, spec_stats) = match config.mode {
            StreamMode::Batched => self.run_spec_batched(
                prompt.embeds.view(),
                config.chunk_samples,
                &mut spec_cfg,
                &mut cb,
                false,
                start,
            )?,
            StreamMode::PerFrame => self.run_spec_batched(
                prompt.embeds.view(),
                config.chunk_samples,
                &mut spec_cfg,
                &mut cb,
                true,
                start,
            )?,
            StreamMode::Progressive { frames_per_chunk } => self.run_spec_progressive_parallel(
                prompt.embeds.view(),
                config.chunk_samples,
                frames_per_chunk.max(1),
                &mut spec_cfg,
                &mut cb,
                start,
            )?,
        };
        Ok((stream_stats, spec_stats))
    }

    /// Speculative streaming with self-speculative early-exit drafting.
    #[cfg(feature = "speculative-decode")]
    pub fn generate_speculative_stream_early_exit<F>(
        &mut self,
        reference: &SpeakerReference,
        target_text: &str,
        early_exit_layers: usize,
        draft_len: usize,
        config: StreamConfig,
        on_event: F,
    ) -> Result<(StreamStats, crate::talker::speculative::SpecRunStats)>
    where
        F: FnMut(StreamEvent) -> StreamControl,
    {
        use crate::megakernel_speculative::SpecConfig;
        let prompt = build_x_vector_prompt(
            &self.cfg,
            &self.store,
            &self.text_embedder,
            &self.tokenizer,
            target_text,
            &reference.x_vector,
        )?;
        self.mk.invalidate_warmup_hidden();
        self.mk
            .warmup(prompt.embeds.view(), self.max_frames, None)?;
        let mut stub = crate::talker::speculative::TrivialDraft;
        let mut spec_cfg =
            SpecConfig::new(&mut stub, 1.0, &prompt.tts_pad_embed, 4, self.max_frames)
                .with_draft_len(draft_len)
                .with_early_exit_layers(early_exit_layers);
        let start = Instant::now();
        let mut cb = on_event;
        let (stream_stats, spec_stats) = match config.mode {
            StreamMode::Batched => self.run_spec_batched(
                prompt.embeds.view(),
                config.chunk_samples,
                &mut spec_cfg,
                &mut cb,
                false,
                start,
            )?,
            StreamMode::PerFrame => self.run_spec_batched(
                prompt.embeds.view(),
                config.chunk_samples,
                &mut spec_cfg,
                &mut cb,
                true,
                start,
            )?,
            StreamMode::Progressive { frames_per_chunk } => self.run_spec_progressive_parallel(
                prompt.embeds.view(),
                config.chunk_samples,
                frames_per_chunk.max(1),
                &mut spec_cfg,
                &mut cb,
                start,
            )?,
        };
        Ok((stream_stats, spec_stats))
    }

    /// Speculative streaming with a learned draft sidecar.
    #[cfg(feature = "speculative-decode")]
    pub fn generate_speculative_stream_learned_draft<F>(
        &mut self,
        reference: &SpeakerReference,
        target_text: &str,
        draft: &mut crate::talker::learned_draft::LearnedDraft,
        draft_len: usize,
        config: StreamConfig,
        on_event: F,
    ) -> Result<(StreamStats, crate::talker::speculative::SpecRunStats)>
    where
        F: FnMut(StreamEvent) -> StreamControl,
    {
        use crate::megakernel_speculative::SpecConfig;
        let prompt = build_x_vector_prompt(
            &self.cfg,
            &self.store,
            &self.text_embedder,
            &self.tokenizer,
            target_text,
            &reference.x_vector,
        )?;
        self.mk.invalidate_warmup_hidden();
        self.mk
            .warmup(prompt.embeds.view(), self.max_frames, None)?;
        let mut stub = crate::talker::speculative::TrivialDraft;
        let mut spec_cfg =
            SpecConfig::new(&mut stub, 1.0, &prompt.tts_pad_embed, 4, self.max_frames)
                .with_draft_len(draft_len)
                .with_learned_draft(draft);
        let start = Instant::now();
        let mut cb = on_event;
        let (stream_stats, spec_stats) = match config.mode {
            StreamMode::Batched => self.run_spec_batched(
                prompt.embeds.view(),
                config.chunk_samples,
                &mut spec_cfg,
                &mut cb,
                false,
                start,
            )?,
            StreamMode::PerFrame => self.run_spec_batched(
                prompt.embeds.view(),
                config.chunk_samples,
                &mut spec_cfg,
                &mut cb,
                true,
                start,
            )?,
            StreamMode::Progressive { frames_per_chunk } => self.run_spec_progressive_parallel(
                prompt.embeds.view(),
                config.chunk_samples,
                frames_per_chunk.max(1),
                &mut spec_cfg,
                &mut cb,
                start,
            )?,
        };
        Ok((stream_stats, spec_stats))
    }

    /// Streaming generation. The closure receives [`StreamEvent`]s as the
    /// pipeline runs and returns [`StreamControl`] to keep going or stop early.
    ///
    /// See [`StreamConfig`] for the three modes (Batched / PerFrame /
    /// Progressive) and their latency / precision / CPU-work trade-offs.
    pub fn generate_stream<F>(
        &mut self,
        reference: &SpeakerReference,
        target_text: &str,
        config: StreamConfig,
        on_event: F,
    ) -> Result<StreamStats>
    where
        F: FnMut(StreamEvent) -> StreamControl,
    {
        let start = Instant::now();
        let mut cb = on_event;
        match config.mode {
            StreamMode::PerFrame => {
                self.run_per_frame(reference, target_text, config.chunk_samples, &mut cb, start)
            }
            StreamMode::Batched => {
                let pcm = self.generate(reference, target_text)?;
                Self::emit_stream_pcm(&pcm, config.chunk_samples, start, &mut cb)
            }
            StreamMode::Progressive { frames_per_chunk } => self.run_progressive_parallel(
                reference,
                target_text,
                config.chunk_samples,
                frames_per_chunk.max(1),
                &mut cb,
                start,
            ),
        }
    }

    /// Materialize all chunks into a `Vec<PcmChunk>`. Convenience wrapper around
    /// [`Self::generate_stream`].
    pub fn generate_chunks(
        &mut self,
        reference: &SpeakerReference,
        target_text: &str,
        config: StreamConfig,
    ) -> Result<(Vec<PcmChunk>, StreamStats)> {
        let mut chunks = Vec::new();
        let stats = self.generate_stream(reference, target_text, config, |evt| {
            if let StreamEvent::Pcm(c) = evt {
                chunks.push(c);
            }
            StreamControl::Continue
        })?;
        Ok((chunks, stats))
    }

    fn run_per_frame(
        &mut self,
        reference: &SpeakerReference,
        target_text: &str,
        chunk_samples: usize,
        on_event: &mut dyn FnMut(StreamEvent) -> StreamControl,
        start: Instant,
    ) -> Result<StreamStats> {
        let max_frames = self.max_frames;
        let mut stop_signal = false;
        let mut on_frame = |idx: usize, _frame: &[u32]| {
            if !stop_signal {
                match on_event(StreamEvent::FrameProduced {
                    frame_index: idx,
                    max_frames,
                }) {
                    StreamControl::Continue => {}
                    StreamControl::Stop => stop_signal = true,
                }
            }
        };
        let pcm = self.synthesize_utterance(reference, target_text, Some(&mut on_frame))?;
        Self::emit_stream_pcm(&pcm, chunk_samples, start, on_event)
    }

    /// Live mode with AR and decoder running in PARALLEL via a scoped worker
    /// thread. Wall time = max(AR_total, decode_total) instead of their sum,
    /// pushing live RTF close to 1.0.
    ///
    /// With `cfg(feature = "incremental-decode")`, the worker uses the
    /// [`StreamingDecoder`](crate::speech_tokenizer::decode_streaming::StreamingDecoder)
    /// wrapper which caps per-chunk decode work to the decoder's receptive
    /// field for utterances longer than the pre-transformer sliding window.
    fn run_progressive_parallel(
        &mut self,
        reference: &SpeakerReference,
        target_text: &str,
        chunk_samples: usize,
        frames_per_chunk: usize,
        on_event: &mut dyn FnMut(StreamEvent) -> StreamControl,
        start: Instant,
    ) -> Result<StreamStats> {
        use std::sync::mpsc;

        let prompt = build_x_vector_prompt(
            &self.cfg,
            &self.store,
            &self.text_embedder,
            &self.tokenizer,
            target_text,
            &reference.x_vector,
        )?;
        self.mk.invalidate_warmup_hidden();
        self.mk
            .warmup(prompt.embeds.view(), self.max_frames, None)?;

        let max_frames = self.max_frames;
        let device = self.device;
        let decode_device = crate::gpu_pipeline::progressive_speech_decode_device(device);
        let mk = &mut self.mk;
        let decoder = &mut self.decoder;
        let model_dir_path: PathBuf = self.model_dir.clone();
        let pad_embed: Vec<f32> = prompt.tts_pad_embed.clone();
        let talker_cfg = self.cfg.talker().clone();

        let mut emitter = ChunkEmitter::new(chunk_samples, start);
        // Assigned only on the `incremental-decode` drain paths below.
        #[allow(unused_mut)]
        let mut consumed_samples = 0usize;
        let mut stop_early = false;
        let mut streamed_pcm = Vec::new();
        // Incremental mode only: holds samples that didn't fill a complete
        // chunk on the previous drain. Prepended to the next response so
        // partial-chunk PCM is preserved instead of silently dropped.
        #[cfg(feature = "incremental-decode")]
        let mut incremental_pending: Vec<f32> = Vec::new();

        // Request/response shapes for the worker. With `incremental-decode`
        // the response carries only the NEW samples; otherwise it carries the
        // full PCM and the main thread maintains a `consumed_samples` cursor.
        struct DecReq {
            frames: Vec<Vec<u32>>,
            is_final: bool,
        }
        #[cfg(not(feature = "incremental-decode"))]
        struct DecResp {
            pcm: Vec<f32>,
            is_final: bool,
        }
        #[cfg(feature = "incremental-decode")]
        struct DecResp {
            new_pcm: Vec<f32>,
            is_final: bool,
        }

        let (frames_emitted_total, codec_frames) =
            std::thread::scope(|s| -> Result<(usize, Vec<Vec<u32>>)> {
                let (req_tx, req_rx) = mpsc::sync_channel::<DecReq>(1);
                let (resp_tx, resp_rx) = mpsc::channel::<Result<DecResp>>();

                // Decode worker. Two implementations depending on whether the
                // `incremental-decode` feature is enabled.
                #[cfg(not(feature = "incremental-decode"))]
                let worker = {
                    let model_dir_for_worker = model_dir_path.clone();
                    s.spawn(move || {
                        let mut worker_decoder = match crate::speech_tokenizer::St12HzDecoder::open(
                            &model_dir_for_worker,
                        ) {
                            Ok(d) => d,
                            Err(e) => {
                                let _ = resp_tx.send(Err(e));
                                return;
                            }
                        };
                        if let Err(e) = worker_decoder.warmup(decode_device, Some(max_frames)) {
                            let _ = resp_tx.send(Err(e));
                            return;
                        }
                        while let Ok(req) = req_rx.recv() {
                            let pcm = match worker_decoder.decode(&req.frames, decode_device) {
                                Ok(pcm) => pcm,
                                Err(e) => {
                                    let _ = resp_tx.send(Err(e));
                                    return;
                                }
                            };
                            if resp_tx
                                .send(Ok(DecResp {
                                    pcm,
                                    is_final: req.is_final,
                                }))
                                .is_err()
                            {
                                return;
                            }
                            if req.is_final {
                                return;
                            }
                        }
                    })
                };
                #[cfg(feature = "incremental-decode")]
                let worker = {
                    // The incremental worker doesn't use the borrowed `decoder` —
                    // it builds a separate StreamingDecoder owning its own state.
                    let model_dir_for_worker = model_dir_path.clone();
                    s.spawn(move || {
                        let mut sd =
                            match crate::speech_tokenizer::decode_streaming::StreamingDecoder::open(
                                &model_dir_for_worker,
                                decode_device,
                                0,
                            ) {
                                Ok(sd) => sd,
                                Err(e) => {
                                    let _ = resp_tx.send(Err(e));
                                    return;
                                }
                            };
                        let mut frames_seen = 0usize;
                        while let Ok(req) = req_rx.recv() {
                            let total = req.frames.len();
                            if total < frames_seen {
                                let _ = resp_tx.send(Err(anyhow::anyhow!("frames shrank")));
                                return;
                            }
                            let new = req.frames[frames_seen..total].to_vec();
                            frames_seen = total;
                            let new_pcm = match sd.decode_chunk(&new, decode_device) {
                                Ok(p) => p,
                                Err(e) => {
                                    let _ = resp_tx.send(Err(e));
                                    return;
                                }
                            };
                            if resp_tx
                                .send(Ok(DecResp {
                                    new_pcm,
                                    is_final: req.is_final,
                                }))
                                .is_err()
                            {
                                return;
                            }
                            if req.is_final {
                                return;
                            }
                        }
                    })
                };
                let _ = decoder;

                let mut state = mk.begin_codec_ar(prompt.embeds.view(), &talker_cfg, max_frames)?;
                let mut last_request_count = 0usize;

                // Drive AR; whenever K new frames are ready and worker is idle,
                // send a decode request. After every AR step also drain any
                // completed decode responses and emit chunks.
                while !state.is_done() {
                    let new_frame =
                        mk.codec_ar_step(&mut state, &talker_cfg, 4, 1.0, &pad_embed)?;

                    if let Some(idx) = new_frame {
                        match on_event(StreamEvent::FrameProduced {
                            frame_index: idx,
                            max_frames,
                        }) {
                            StreamControl::Continue => {}
                            StreamControl::Stop => {
                                stop_early = true;
                                break;
                            }
                        }
                    }

                    // Drain any completed decode responses.
                    while let Ok(resp) = resp_rx.try_recv() {
                        let resp = resp?;
                        #[cfg(not(feature = "incremental-decode"))]
                        {
                            consumed_samples = drain_progressive_pcm(
                                &resp.pcm,
                                consumed_samples,
                                false,
                                &mut emitter,
                                on_event,
                                &mut streamed_pcm,
                            );
                        }
                        #[cfg(feature = "incremental-decode")]
                        {
                            let _ = consumed_samples;
                            incremental_pending.extend_from_slice(&resp.new_pcm);
                            let consumed = drain_progressive_pcm(
                                &incremental_pending,
                                0,
                                false,
                                &mut emitter,
                                on_event,
                                &mut streamed_pcm,
                            );
                            incremental_pending.drain(..consumed);
                        }
                        if emitter.stopped {
                            stop_early = true;
                            break;
                        }
                    }
                    if stop_early {
                        break;
                    }

                    // Try to enqueue a new decode request if we have enough new
                    // frames AND the worker is idle (sync_channel(1) blocks on send
                    // when busy, so use try_send).
                    let count = state.codec_frames.len();
                    if count > 0 && count - last_request_count >= frames_per_chunk {
                        let req = DecReq {
                            frames: state.codec_frames.clone(),
                            is_final: false,
                        };
                        if req_tx.try_send(req).is_ok() {
                            last_request_count = count;
                        }
                        // If try_send failed, worker is busy — try again next step.
                    }
                }

                // AR done — always run a terminal decode so the trailing partial
                // chunk is emitted (`ChunkEmitter::drain` needs `is_terminal=true`).
                let final_count = state.codec_frames.len();
                let frames = state.codec_frames.clone();
                if !stop_early && final_count > 0 {
                    let _ = req_tx.send(DecReq {
                        frames: frames.clone(),
                        is_final: true,
                    });
                }
                drop(req_tx);

                // Drain any remaining responses.
                for resp in resp_rx.iter() {
                    let resp = resp?;
                    let is_terminal = resp.is_final;
                    #[cfg(not(feature = "incremental-decode"))]
                    {
                        consumed_samples = drain_progressive_pcm(
                            &resp.pcm,
                            consumed_samples,
                            is_terminal,
                            &mut emitter,
                            on_event,
                            &mut streamed_pcm,
                        );
                    }
                    #[cfg(feature = "incremental-decode")]
                    {
                        let _ = consumed_samples;
                        incremental_pending.extend_from_slice(&resp.new_pcm);
                        let consumed = drain_progressive_pcm(
                            &incremental_pending,
                            0,
                            is_terminal,
                            &mut emitter,
                            on_event,
                            &mut streamed_pcm,
                        );
                        incremental_pending.drain(..consumed);
                    }
                }

                worker
                    .join()
                    .map_err(|_| anyhow::anyhow!("decode worker panicked"))?;
                Ok((final_count, frames))
            })?;

        self.mk.invalidate_warmup_hidden();
        if !stop_early && !codec_frames.is_empty() {
            let reference_pcm = if decode_device == device {
                self.decoder.decode(&codec_frames, device)?
            } else {
                let mut ref_decoder =
                    crate::speech_tokenizer::St12HzDecoder::open(&self.model_dir)?;
                ref_decoder.warmup(decode_device, Some(max_frames))?;
                ref_decoder.decode(&codec_frames, decode_device)?
            };
            ensure_progressive_pcm_matches(&streamed_pcm, &reference_pcm)?;
        }

        let _ = consumed_samples;
        let mut stats = emitter.finalize(frames_emitted_total);
        stats.samples_emitted = streamed_pcm.len();
        stats.audio_secs = streamed_pcm.len() as f64 / 24_000.0;
        if stop_early {
            stats.stopped_early = true;
        }
        Ok(stats)
    }

    /// Progressive streaming: synthesize with [`Self::generate`] (golden PCM), then
    /// slice into live chunks. Quality matches non-streaming exactly; chunk cadence
    /// follows `chunk_samples` for duplex / speaker sinks.
    #[cfg(feature = "speculative-decode")]
    fn run_spec_batched<D: crate::talker::speculative::DraftModel>(
        &mut self,
        prefill_embeds: ndarray::ArrayView2<f32>,
        chunk_samples: usize,
        spec_cfg: &mut crate::megakernel_speculative::SpecConfig<'_, D>,
        on_event: &mut dyn FnMut(StreamEvent) -> StreamControl,
        per_frame_events: bool,
        start: Instant,
    ) -> Result<(StreamStats, crate::talker::speculative::SpecRunStats)> {
        let talker_cfg = self.cfg.talker();
        let max_frames = self.max_frames;
        let mut stop_signal = false;
        let mut spec_state =
            self.mk
                .begin_codec_ar_speculative(prefill_embeds, talker_cfg, spec_cfg)?;
        while !spec_state.is_done() {
            let outcome =
                self.mk
                    .codec_ar_speculative_step(&mut spec_state, talker_cfg, spec_cfg)?;
            if per_frame_events && !stop_signal {
                for idx in outcome.new_frame_indices {
                    match on_event(StreamEvent::FrameProduced {
                        frame_index: idx,
                        max_frames,
                    }) {
                        StreamControl::Continue => {}
                        StreamControl::Stop => stop_signal = true,
                    }
                }
            }
            if outcome.done || stop_signal {
                break;
            }
        }
        let (frames, spec_stats, _, _, _) = spec_state.finish();
        let pcm = self.decoder.decode(&frames, self.device)?;
        let mut emitter = ChunkEmitter::new(chunk_samples, start);
        let _ = emitter.drain(&pcm, 0, true, on_event);
        let mut stats = emitter.finalize(frames.len());
        if stop_signal {
            stats.stopped_early = true;
        }
        Ok((stats, spec_stats))
    }

    #[cfg(feature = "speculative-decode")]
    fn run_spec_progressive_parallel<D: crate::talker::speculative::DraftModel>(
        &mut self,
        prefill_embeds: ndarray::ArrayView2<f32>,
        chunk_samples: usize,
        frames_per_chunk: usize,
        spec_cfg: &mut crate::megakernel_speculative::SpecConfig<'_, D>,
        on_event: &mut dyn FnMut(StreamEvent) -> StreamControl,
        start: Instant,
    ) -> Result<(StreamStats, crate::talker::speculative::SpecRunStats)> {
        use std::sync::mpsc;

        let talker_cfg = self.cfg.talker();
        let max_frames = self.max_frames;
        let device = self.device;
        let mk = &mut self.mk;
        let decoder = &mut self.decoder;

        let mut emitter = ChunkEmitter::new(chunk_samples, start);
        let mut consumed_samples = 0usize;
        let mut stop_early = false;

        struct DecReq {
            frames: Vec<Vec<u32>>,
            is_final: bool,
        }
        struct DecResp {
            pcm: Vec<f32>,
            is_final: bool,
        }

        let (frames_emitted_total, spec_stats) = std::thread::scope(
            |s| -> Result<(usize, crate::talker::speculative::SpecRunStats)> {
                let (req_tx, req_rx) = mpsc::sync_channel::<DecReq>(1);
                let (resp_tx, resp_rx) = mpsc::channel::<Result<DecResp>>();

                let worker = s.spawn(move || {
                    while let Ok(req) = req_rx.recv() {
                        let pcm = match decoder.decode(&req.frames, device) {
                            Ok(pcm) => pcm,
                            Err(e) => {
                                let _ = resp_tx.send(Err(e));
                                return;
                            }
                        };
                        if resp_tx
                            .send(Ok(DecResp {
                                pcm,
                                is_final: req.is_final,
                            }))
                            .is_err()
                        {
                            return;
                        }
                        if req.is_final {
                            return;
                        }
                    }
                });

                let mut spec_state =
                    mk.begin_codec_ar_speculative(prefill_embeds, talker_cfg, spec_cfg)?;
                let mut last_request_count = 0usize;

                while !spec_state.is_done() && !stop_early {
                    let outcome =
                        mk.codec_ar_speculative_step(&mut spec_state, talker_cfg, spec_cfg)?;
                    for idx in outcome.new_frame_indices {
                        match on_event(StreamEvent::FrameProduced {
                            frame_index: idx,
                            max_frames,
                        }) {
                            StreamControl::Continue => {}
                            StreamControl::Stop => {
                                stop_early = true;
                                break;
                            }
                        }
                    }

                    while let Ok(resp) = resp_rx.try_recv() {
                        let resp = resp?;
                        let new_consumed =
                            emitter.drain(&resp.pcm, consumed_samples, false, on_event);
                        consumed_samples = new_consumed;
                        if emitter.stopped {
                            stop_early = true;
                            break;
                        }
                    }
                    if stop_early {
                        break;
                    }

                    let count = spec_state.codec_frames.len();
                    if count > 0 && count - last_request_count >= frames_per_chunk {
                        let req = DecReq {
                            frames: spec_state.codec_frames.clone(),
                            is_final: false,
                        };
                        if req_tx.try_send(req).is_ok() {
                            last_request_count = count;
                        }
                    }

                    if outcome.done {
                        break;
                    }
                }

                let final_count = spec_state.codec_frames.len();
                if !stop_early && (final_count > last_request_count || consumed_samples == 0) {
                    let _ = req_tx.send(DecReq {
                        frames: spec_state.codec_frames.clone(),
                        is_final: true,
                    });
                }
                drop(req_tx);

                for resp in resp_rx.iter() {
                    let resp = resp?;
                    let is_terminal = resp.is_final;
                    let new_consumed =
                        emitter.drain(&resp.pcm, consumed_samples, is_terminal, on_event);
                    consumed_samples = new_consumed;
                }

                worker
                    .join()
                    .map_err(|_| anyhow::anyhow!("decode worker panicked"))?;
                let (_, spec_stats, _, _, _) = spec_state.finish();
                Ok((final_count, spec_stats))
            },
        )?;

        let _ = consumed_samples;
        let mut stats = emitter.finalize(frames_emitted_total);
        if stop_early {
            stats.stopped_early = true;
        }
        Ok((stats, spec_stats))
    }
}

fn drain_progressive_pcm(
    pcm: &[f32],
    consumed: usize,
    is_terminal: bool,
    emitter: &mut crate::stream::ChunkEmitter,
    on_event: &mut dyn FnMut(crate::stream::StreamEvent) -> crate::stream::StreamControl,
    streamed_pcm: &mut Vec<f32>,
) -> usize {
    let mut tap_emit = |evt: crate::stream::StreamEvent| -> crate::stream::StreamControl {
        if let crate::stream::StreamEvent::Pcm(ref chunk) = evt {
            streamed_pcm.extend_from_slice(&chunk.samples);
        }
        on_event(evt)
    };
    let tap: &mut dyn FnMut(crate::stream::StreamEvent) -> crate::stream::StreamControl =
        &mut tap_emit;
    emitter.drain(pcm, consumed, is_terminal, tap)
}

/// Progressive partial-decode must be sample-identical to one-shot decode of the
/// same codec frames (causal Mimi).
fn ensure_progressive_pcm_matches(streamed: &[f32], reference: &[f32]) -> Result<()> {
    if streamed.len() > reference.len() {
        anyhow::bail!(
            "progressive stream over-emitted: {} > {} samples",
            streamed.len(),
            reference.len()
        );
    }
    for (i, (&a, &b)) in streamed.iter().zip(reference.iter()).enumerate() {
        if (a - b).abs() > 1e-4 {
            anyhow::bail!("progressive prefix mismatch at sample {i}: streamed={a} reference={b}");
        }
    }
    if streamed.len() < reference.len() {
        anyhow::bail!(
            "progressive stream truncated: {} < {} samples",
            streamed.len(),
            reference.len()
        );
    }
    Ok(())
}
