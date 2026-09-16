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

//! Reference audio + transcript → [`VoicePrompt`] (`Encoder.forward`).
//!
//! Two models run over the same signal at two rates: the aligner at 16 kHz to
//! decide which 50 Hz frame each text token owns, and the codec encoder at
//! 24 kHz to produce the latent sitting on that frame. They agree because 320
//! samples at 16 kHz and 480 at 24 kHz are both exactly 20 ms.
//!
//! Building a prompt is the expensive half of TADA and its result does not
//! depend on what you go on to say, so the output is meant to be cached — see
//! [`VoicePrompt::save`].

use crate::align::align_tokens;
use crate::aligner::Aligner;
use crate::codec::CodecEncoder;
use crate::config::{ALIGNER_SAMPLE_RATE, AlignerConfig, EncoderConfig, FRAME_RATE, SAMPLE_RATE};
use crate::model::find_backbone;
use crate::prompt::VoicePrompt;
use crate::resample::resample;
use crate::rng::Normal;
use crate::text::normalize_text;
use crate::tokenizer::TadaTokenizer;
use crate::weights::TensorStore;
use anyhow::{Context, Result, bail};
use ndarray::Array2;
use rlx_core::voice_clone::{
    BAND_LIMIT_CUTOFF_HZ, ReferenceLimits, ReferencePrep, high_band_energy_ratio,
    looks_band_limited, preprocess_reference, validate_reference,
};
use rlx_runtime::Device;
use std::path::Path;
use std::sync::Arc;

/// Settings for prompt construction.
#[derive(Debug, Clone)]
pub struct PromptOptions {
    /// Standard deviation of the noise added to the encoder's latents.
    ///
    /// The codec encoder is a stochastic bottleneck and upstream keeps sampling
    /// on at inference (`EncoderConfig.std = 0.5`), so the LM sees prompts that
    /// look like the ones it trained against. Set to `0.0` for a deterministic
    /// prompt — a cleaner artifact to diff, but off-distribution.
    pub latent_noise_std: f32,
    /// Seed for that noise.
    pub seed: u64,
    /// Mean subtracted from the gathered latents before the model sees them.
    pub acoustic_mean: f32,
    /// Divisor applied after the mean subtraction.
    pub acoustic_std: f32,
    /// Clean the reference clip before encoding it — drop DC offset, trim edge
    /// silence, cap a hot peak (see [`rlx_core::voice_clone`]).
    ///
    /// On by default because real recordings need it and the parity fixtures
    /// are unaffected (they are already clean, and the per-stage suites feed
    /// latents directly rather than going through here). Turn it off to
    /// reproduce a byte-exact prompt from an untouched clip.
    pub preprocess: bool,
    /// Reject clips that are too short, too long, or effectively silent.
    /// `None` skips the check.
    pub limits: Option<ReferenceLimits>,
}

impl Default for PromptOptions {
    fn default() -> Self {
        Self {
            latent_noise_std: 0.5,
            seed: 0x7ada_0000_0000_0002,
            acoustic_mean: 0.0,
            acoustic_std: 1.5,
            preprocess: true,
            limits: Some(ReferenceLimits::default()),
        }
    }
}

/// The aligner plus the codec encoder.
pub struct PromptBuilder {
    device: Device,
    aligner: Aligner,
    encoder: CodecEncoder,
    tokenizer: TadaTokenizer,
}

impl PromptBuilder {
    /// Load from the checkpoint layout described in [`crate::model`].
    ///
    /// `language` selects a per-language aligner (`aligner-de`, `aligner-ja`, …
    /// in `HumeAI/tada-codec`); `None` uses the English `aligner/`. The choice
    /// matters even though generation is language-agnostic: the alignment is
    /// baked into the prompt, so a prompt built with the wrong aligner is
    /// wrong for every utterance made from it.
    pub fn open(root: &Path, device: Device, language: Option<&str>) -> Result<Self> {
        let codec = root.join("tada-codec");
        let aligner_dir = match language {
            Some(lang) => codec.join(format!("aligner-{lang}")),
            None => codec.join("aligner"),
        };
        if !aligner_dir.join("model.safetensors").is_file() {
            bail!(
                "no aligner at {} — download `HumeAI/tada-codec` into {}",
                aligner_dir.display(),
                codec.display()
            );
        }
        let aligner_store = Arc::new(TensorStore::open(&aligner_dir.join("model.safetensors"))?);
        let aligner = Aligner::load(aligner_store, AlignerConfig::default())
            .context("load aligner weights")?;

        let enc_path = codec.join("encoder/model.safetensors");
        let enc_store = Arc::new(
            TensorStore::open(&enc_path).with_context(|| format!("open {}", enc_path.display()))?,
        );
        let encoder = CodecEncoder::load(enc_store, EncoderConfig::default())
            .context("load codec encoder weights")?;

        // The tokenizer must be the same one the backbone uses; resolving it
        // through the backbone's root keeps the two from drifting apart.
        let _ = find_backbone(root)?;
        let tokenizer = TadaTokenizer::load(&root.join("tokenizer"))?;

        Ok(Self {
            device,
            aligner,
            encoder,
            tokenizer,
        })
    }

    /// Build one prompt from several clips of the same speaker.
    ///
    /// TADA conditions on a single contiguous reference, so more material means
    /// concatenating it: each clip is cleaned and resampled on its own, the
    /// transcripts are joined in the same order, and the result goes through
    /// [`PromptBuilder::build`] as if it had been recorded in one take. More
    /// reference frames is exactly what the aligner wants, and a speaker rarely
    /// has one clip that covers their whole range.
    ///
    /// Each clip is validated against `opts.limits` individually; the
    /// concatenation is not, since exceeding the single-clip duration ceiling is
    /// the entire point.
    pub fn build_multi(
        &self,
        clips: &[(&[f32], usize, &str)],
        opts: &PromptOptions,
    ) -> Result<VoicePrompt> {
        match clips {
            [] => bail!("no reference clips supplied"),
            [(pcm, sr, text)] => return self.build(pcm, *sr, text, opts),
            _ => {}
        }
        let prep = ReferencePrep::default();
        let mut pcm = Vec::new();
        let mut texts = Vec::with_capacity(clips.len());
        for (i, (clip, sr, text)) in clips.iter().enumerate() {
            if text.trim().is_empty() {
                bail!("reference clip {i} has an empty transcript");
            }
            let cleaned = if opts.preprocess {
                preprocess_reference(clip, *sr as u32, &prep)
            } else {
                clip.to_vec()
            };
            if let Some(limits) = opts.limits {
                validate_reference(&cleaned, *sr as u32, &limits)
                    .map_err(|e| anyhow::anyhow!("reference clip {i}: {e}"))?;
            }
            pcm.extend_from_slice(&resample(&cleaned, *sr, SAMPLE_RATE));
            texts.push(text.trim());
        }
        // Already cleaned and length-checked per clip.
        let opts = PromptOptions {
            preprocess: false,
            limits: None,
            ..opts.clone()
        };
        self.build(&pcm, SAMPLE_RATE, &texts.join(" "), &opts)
    }

    /// Build a prompt from mono `pcm` at `sample_rate` plus its `transcript`.
    pub fn build(
        &self,
        pcm: &[f32],
        sample_rate: usize,
        transcript: &str,
        opts: &PromptOptions,
    ) -> Result<VoicePrompt> {
        if pcm.is_empty() {
            bail!("reference audio is empty");
        }
        let cleaned;
        let pcm = if opts.preprocess {
            cleaned = preprocess_reference(pcm, sample_rate as u32, &ReferencePrep::default());
            &cleaned[..]
        } else {
            pcm
        };
        if let Some(limits) = opts.limits {
            validate_reference(pcm, sample_rate as u32, &limits)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        if looks_band_limited(pcm, sample_rate as u32) {
            eprintln!(
                "[tada] warning: the reference is band-limited — only {:.2}% of its energy \
                 sits above {:.0} Hz. The codec is full-band 24 kHz, so it will encode a \
                 muffled voice and the clone will inherit that. Measured on an archival \
                 excerpt: ~0.82 speaker cosine, against ~0.94 for a studio clip.",
                high_band_energy_ratio(pcm, sample_rate as u32, BAND_LIMIT_CUTOFF_HZ) * 100.0,
                BAND_LIMIT_CUTOFF_HZ
            );
        }

        let text = normalize_text(transcript);
        let token_ids = self.tokenizer.encode(&text)?;
        if token_ids.is_empty() {
            bail!("reference transcript `{transcript}` tokenized to nothing");
        }

        let pcm24 = resample(pcm, sample_rate, SAMPLE_RATE);
        let pcm16 = resample(&pcm24, SAMPLE_RATE, ALIGNER_SAMPLE_RATE);

        // Upstream measures the frame budget from the *original* duration, not
        // from whatever the conv stacks happen to emit.
        let audio_frames =
            (pcm24.len() as f64 / SAMPLE_RATE as f64 * FRAME_RATE as f64).ceil() as usize;
        if audio_frames <= token_ids.len() {
            bail!(
                "reference audio is {audio_frames} frames but the transcript has {} tokens — \
                 each token needs its own frame",
                token_ids.len()
            );
        }

        let (logits, ctc_frames) = self
            .aligner
            .logits(self.device, &pcm16)
            .context("run the forced aligner")?;
        let vocab = self.aligner.vocab_size();
        // The DP runs over every CTC frame; the mask is sized by the audio.
        let frames_for_mask = audio_frames.max(ctc_frames);
        let alignment = align_tokens(&logits, vocab, &token_ids, frames_for_mask)
            .context("align transcript to reference audio")?;

        if alignment.looks_mismatched() {
            eprintln!(
                "[tada] warning: the transcript may not match the reference audio \
                 (mean aligned token logprob {:.2}). The aligner places every token \
                 somewhere regardless, so a wrong transcript yields a plausible-looking \
                 alignment; check the text against what the clip actually says.",
                alignment.mean_token_logprob
            );
        }

        let latents = self
            .encoder
            .encode(self.device, &pcm24, &alignment.token_mask)
            .context("encode reference audio")?;

        let mut token_values = Array2::<f32>::zeros((token_ids.len(), latents.ncols()));
        let mut rng = Normal::new(opts.seed);
        for (i, &pos) in alignment.token_positions.iter().enumerate() {
            // Positions are 1-based; upstream gathers at `(pos - 1).clamp(0)`.
            let row = (pos.saturating_sub(1) as usize).min(latents.nrows() - 1);
            for c in 0..latents.ncols() {
                let mut v = latents[[row, c]];
                if opts.latent_noise_std > 0.0 {
                    v += rng.sample() * opts.latent_noise_std;
                }
                token_values[[i, c]] = (v - opts.acoustic_mean) / opts.acoustic_std;
            }
        }

        let prompt = VoicePrompt {
            text,
            token_ids,
            token_positions: alignment.token_positions,
            token_values,
            alignment_score: alignment.mean_token_logprob,
            audio_samples: pcm24.len(),
            sample_rate: SAMPLE_RATE,
        };
        prompt.validate()?;
        Ok(prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_match_the_upstream_bottleneck() {
        let o = PromptOptions::default();
        assert_eq!(o.latent_noise_std, 0.5);
        assert_eq!(o.acoustic_std, 1.5);
        assert_eq!(o.acoustic_mean, 0.0);
    }
}

#[cfg(test)]
mod prep_tests {
    use super::*;

    #[test]
    fn defaults_clean_and_validate() {
        let o = PromptOptions::default();
        assert!(o.preprocess, "real recordings need the cleanup");
        assert!(o.limits.is_some(), "a silent reference should be rejected");
    }

    #[test]
    fn a_hot_reference_is_brought_under_full_scale() {
        // 3 s of loud tone: preprocessing must cap it without rejecting it.
        let sr = SAMPLE_RATE;
        let pcm: Vec<f32> = (0..sr * 3)
            .map(|i| 1.4 * (i as f32 * 220.0 * std::f32::consts::TAU / sr as f32).sin())
            .collect();
        let cleaned = preprocess_reference(&pcm, sr as u32, &ReferencePrep::default());
        let peak = cleaned.iter().fold(0f32, |m, x| m.max(x.abs()));
        assert!(peak <= 0.951, "peak {peak}");
        assert!(
            validate_reference(&cleaned, sr as u32, &ReferenceLimits::default()).is_ok(),
            "a merely-hot clip must still be usable"
        );
    }
}
