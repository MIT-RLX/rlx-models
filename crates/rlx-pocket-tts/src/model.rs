// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! Top-level Pocket TTS entry point.
//!
//! ```no_run
//! use rlx_pocket_tts::TtsModel;
//! let model = TtsModel::open("tts.safetensors", "tokenizer.model")?;
//! let voice = model.load_voice("voice.safetensors")?;
//! let audio = model.generate("Hello, world.", &voice, Default::default())?;
//! audio.write_wav("out.wav")?;
//! # Ok::<_, anyhow::Error>(())
//! ```

use std::path::Path;

use anyhow::{Context, Result};
use ndarray::Array2;

use crate::audio::write_wav_mono;
use crate::config::PocketTtsConfig;
use crate::flow_lm::{FlowLm, make_rng, sample_latent};
use crate::mimi::MimiDecoder;
use crate::tokenizer::{
    MAX_TOKENS_PER_CHUNK, PocketTokenizer, prepare_text_prompt, split_into_chunks,
};
use crate::voice::Voice;
use crate::weights::WeightFile;
use crate::{FRAME_RATE, SAMPLE_RATE};

#[derive(Debug, Clone)]
pub struct GenerationOptions {
    /// Maximum latent frames to generate per chunk. `0` ⇒ unbounded
    /// (capped at `MAX_DEFAULT_FRAMES`). Latent rate is 12.5 Hz so 250 ≈ 20 s.
    pub max_frames: usize,

    /// RNG seed for the flow head's noise initialization. Set to a fixed value
    /// for reproducible runs.
    pub seed: u64,

    /// Optional voice-conditioning BOS prefix override.
    pub insert_bos_before_voice: bool,

    /// Override `frames_after_eos`. `None` ⇒ derived per chunk from word count
    /// (matches pocket_tts: 3 if ≤ 4 words else 1, then + 2).
    pub frames_after_eos: Option<usize>,

    /// Skip pocket_tts's text-prep (capitalize, append period, pad with spaces
    /// for short inputs). Leave at `false` for parity with upstream.
    pub skip_text_prep: bool,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_frames: 0,
            seed: 0xC0FFEE,
            insert_bos_before_voice: true,
            frames_after_eos: None,
            skip_text_prep: false,
        }
    }
}

/// Default cap on max latent frames per chunk (~ 20 s of audio).
pub const MAX_DEFAULT_FRAMES: usize = 250;

pub struct TtsModel {
    pub cfg: PocketTtsConfig,
    pub flow_lm: FlowLm,
    pub mimi: MimiDecoder,
    pub tokenizer: PocketTokenizer,
}

impl TtsModel {
    /// Load the full TTS model from a safetensors weights file and a
    /// SentencePiece tokenizer file (the english default config is used).
    pub fn open(weights: impl AsRef<Path>, tokenizer: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_config(weights, tokenizer, PocketTtsConfig::english())
    }

    pub fn open_with_config(
        weights: impl AsRef<Path>,
        tokenizer: impl AsRef<Path>,
        cfg: PocketTtsConfig,
    ) -> Result<Self> {
        let wf = WeightFile::open(weights.as_ref())
            .with_context(|| format!("open weights {}", weights.as_ref().display()))?;
        let flow_lm = FlowLm::load(&wf, cfg.clone()).context("load FlowLM")?;
        let mimi = MimiDecoder::load(&wf, cfg.clone()).context("load Mimi decoder")?;
        let tokenizer = PocketTokenizer::open(tokenizer.as_ref())
            .with_context(|| format!("open tokenizer {}", tokenizer.as_ref().display()))?;
        Ok(Self {
            cfg,
            flow_lm,
            mimi,
            tokenizer,
        })
    }

    pub fn load_voice(&self, path: impl AsRef<Path>) -> Result<Voice> {
        Voice::open(path)
    }

    /// Generate audio for `text` using the given `voice`. Returns the concatenated
    /// 24 kHz mono waveform.
    pub fn generate(&self, text: &str, voice: &Voice, opts: GenerationOptions) -> Result<Audio> {
        // Mirror pocket_tts's text-prep pipeline before chunking: this is what
        // gates the model away from a too-early EOS on short prompts (the
        // 8-space prefix is the magic). Also derive a chunk-level
        // `frames_after_eos` like generate_audio_stream does.
        let chunks = if opts.skip_text_prep {
            split_into_chunks(&self.tokenizer, text, MAX_TOKENS_PER_CHUNK)?
        } else {
            split_into_chunks(&self.tokenizer, text, MAX_TOKENS_PER_CHUNK)?
                .into_iter()
                .map(|chunk| prepare_text_prompt(&chunk, true).0)
                .collect()
        };
        let mut all_samples: Vec<f32> = Vec::new();

        // Match pocket_tts's `get_state_for_audio_prompt` + `generate_audio_stream`:
        // build the voice-conditioned KV cache ONCE, then deep-copy it per chunk
        // so each chunk starts from the voice-only prefix (without bleed-over
        // from the prior chunk's text + latents).
        let voice_kv = {
            let mut kv = self.flow_lm.transformer.make_cache();
            let voice_prefix =
                build_voice_prefix(&self.flow_lm, voice, opts.insert_bos_before_voice);
            if voice_prefix.shape()[0] > 0 {
                let _ = self.flow_lm.transformer.forward(voice_prefix, &mut kv);
            }
            kv
        };

        for chunk in chunks {
            // Tokenize.
            let token_ids = self.tokenizer.encode(&chunk)?;
            if token_ids.is_empty() {
                continue;
            }
            let token_count = token_ids.len();
            // Per-chunk frames_after_eos guess: `guess + 2` as in generate_audio_stream.
            let frames_after_eos = opts.frames_after_eos.unwrap_or_else(|| {
                let (_, guess) = prepare_text_prompt(&chunk, false);
                guess + 2
            });
            let cap = if opts.max_frames == 0 {
                MAX_DEFAULT_FRAMES
            } else {
                opts.max_frames
            };
            let max_frames =
                cap.min(((token_count as f32 / 3.0 + 2.0) * FRAME_RATE).ceil() as usize);

            // Fresh per-chunk cache (clone of the voice-conditioned snapshot).
            let mut kv = voice_kv.clone();

            // Push the chunk's text embedding through the backbone before
            // entering the AR latent loop.
            let text_emb = self.flow_lm.embed_tokens(&token_ids);
            let _ = self.flow_lm.transformer.forward(text_emb, &mut kv);

            // Auto-regressive latent decoding.
            let mut rng = make_rng(opts.seed.wrapping_add(all_samples.len() as u64));
            let mut latents: Vec<Array2<f32>> = Vec::with_capacity(max_frames);
            let mut eos_seen: Option<usize> = None;
            for step in 0..max_frames {
                // Compute the last-position backbone output. For step 0 the input
                // is a single NaN row → replaced by `bos_emb` inside `project_latent`.
                let prev_latent: Array2<f32> = match latents.last() {
                    Some(prev) => prev.clone(),
                    None => Array2::<f32>::from_elem((1, self.flow_lm.ldim()), f32::NAN),
                };
                let step_input = self.flow_lm.project_latent(&prev_latent);
                let backbone_out = self.flow_lm.transformer.forward(step_input, &mut kv);
                let t_out = backbone_out.shape()[0];
                let d = self.flow_lm.d_model();
                let mut last_2d = Array2::<f32>::zeros((1, d));
                for j in 0..d {
                    last_2d[[0, j]] = backbone_out[[t_out - 1, j]];
                }
                let normed = self.flow_lm.out_norm(last_2d);

                // EOS gate.
                let eos_logit = self.flow_lm.eos_logit(&normed);
                let is_eos = eos_logit > self.cfg.eos_threshold;

                // Flow head sample.
                let latent = sample_latent(&self.flow_lm.flow_net, &normed, &self.cfg, &mut rng);
                latents.push(latent);

                if is_eos && eos_seen.is_none() {
                    eos_seen = Some(step);
                }
                if let Some(es) = eos_seen {
                    // Match pocket_tts: break BEFORE accepting the latent that
                    // would push us past `eos_step + frames_after_eos`.
                    if step >= es + frames_after_eos {
                        latents.pop();
                        break;
                    }
                }
            }

            // Collect [T_lat, ldim] latent matrix, de-normalize, Mimi-decode.
            if latents.is_empty() {
                continue;
            }
            let t_lat = latents.len();
            let ldim = self.flow_lm.ldim();
            let mut latent_matrix = Array2::<f32>::zeros((t_lat, ldim));
            for (i, l) in latents.iter().enumerate() {
                for j in 0..ldim {
                    latent_matrix[[i, j]] = l[[0, j]];
                }
            }
            let denormed = self.flow_lm.denormalize_latent(&latent_matrix);
            let mut audio = self.mimi.decode_latents(&denormed);
            all_samples.append(&mut audio);
        }

        Ok(Audio {
            samples: all_samples,
            sample_rate: SAMPLE_RATE,
        })
    }
}

/// Generated audio.
pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl Audio {
    pub fn write_wav(&self, path: impl AsRef<Path>) -> Result<()> {
        write_wav_mono(path, &self.samples)
    }

    pub fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }
}

/// Concatenate `[bos_before_voice?, voice_conditioning]` into a single
/// `[T, d_model]` slab. Used to warm the KV cache before any text/latents.
fn build_voice_prefix(flow_lm: &FlowLm, voice: &Voice, use_bos: bool) -> Array2<f32> {
    let d = flow_lm.d_model();
    debug_assert_eq!(voice.embed_dim(), d);
    let bos = if use_bos {
        flow_lm.bos_before_voice()
    } else {
        None
    };
    let bos_t = bos.map(|b| b.shape()[0]).unwrap_or(0);
    let voice_t = voice.num_frames();
    let total = bos_t + voice_t;
    let mut out = Array2::<f32>::zeros((total, d));
    let mut row = 0;
    if let Some(b) = bos {
        for i in 0..bos_t {
            for j in 0..d {
                out[[row + i, j]] = b[[i, j]];
            }
        }
        row += bos_t;
    }
    for i in 0..voice_t {
        for j in 0..d {
            out[[row + i, j]] = voice.conditioning[[i, j]];
        }
    }
    out
}
