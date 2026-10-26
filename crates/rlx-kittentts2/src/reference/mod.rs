// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Voice cloning: a reference clip + its transcript → a [`Voice`], the same
//! conditioning a built-in voice ships precomputed.
//!
//! | output | path (upstream) |
//! |---|---|
//! | LM reference tokens | whole clip @ 16 kHz → [`tokenizer`] (`S3Codec.encode_reference`) |
//! | speaker vector (512) | whole clip @ 16 kHz → [`pyannote`] → unit norm (`SpeakerEncoder`) |
//! | S3Gen prompt mel | first 10 s @ 24 kHz (tiled to ≥ 6 s) → [`dsp::s3gen_mel`] |
//! | S3Gen prompt tokens | that clip → 16 kHz → [`tokenizer`], cut to `mel / 2` |
//! | CAM++ x-vector (192) | that clip → 16 kHz → [`dsp::kaldi_fbank`] → CAM++ (`rlx-funasr`) |
//!
//! Resampling uses `rlx_core::audio::resample_sinc`, which is torchaudio's
//! kernel; upstream loads the clip itself with librosa (soxr), so the
//! decimated inputs differ at the 1e-4 level and a token or two of the
//! reference can flip — the networks are checked against the reference on
//! identical inputs.

pub mod dsp;
pub mod pyannote;
pub mod tokenizer;

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use rlx_ir::Shape;
use rlx_ir::hir::{HirModule, HirMut};
use rlx_runtime::{CompiledGraph, Device};

use crate::s3gen::nn::{Ctx, F32, S3GenWeights};
use crate::voices::{S3Conditioning, Voice};

pub const S3_SR: usize = 16_000;
pub const S3GEN_SR: usize = 24_000;
/// S3Gen conditions on at most this much reference audio …
pub const DECODER_COND_SECONDS: f32 = 10.0;
/// … and needs at least this much (short clips are tiled, never padded).
pub const DECODER_MIN_SECONDS: f32 = 6.0;
/// Mel frames the S3 tokenizer encodes at once (30 s); longer clips go
/// through overlapping windows.
pub const TOKENIZER_WINDOW_FRAMES: usize = 3000;
/// Window overlap (4 s); half of it is dropped on each side of a seam.
pub const TOKENIZER_OVERLAP_FRAMES: usize = 400;
const TOKENS_PER_SECOND: usize = 25;

struct Slot {
    len: usize,
    graph: CompiledGraph,
}

/// Encoders for a new reference clip.
pub struct ReferenceEncoder {
    s3: Arc<S3GenWeights>,
    spk: Arc<S3GenWeights>,
    campplus: rlx_funasr::speaker::CamPlus,
    device: Device,
    tokenizer_slot: RefCell<Option<Slot>>,
    pyannote_slot: RefCell<Option<Slot>>,
}

fn compile(
    device: Device,
    hir: HirModule,
    params: Vec<(String, Vec<f32>)>,
    what: &str,
) -> Result<CompiledGraph> {
    let graph =
        rlx_ir::Graph::from_hir(hir).map_err(|e| anyhow::anyhow!("{what}: HIR lowering: {e}"))?;
    let mut g = rlx_runtime::Session::new(device).compile(graph);
    for (n, d) in &params {
        g.set_param(n, d);
    }
    g.finalize_params();
    Ok(g)
}

fn run_slot(
    slot: &RefCell<Option<Slot>>,
    len: usize,
    build: impl FnOnce() -> Result<CompiledGraph>,
    inputs: &[(&str, &[f32])],
) -> Result<Vec<f32>> {
    let mut s = slot.borrow_mut();
    if s.as_ref().is_none_or(|s| s.len != len) {
        *s = None;
        *s = Some(Slot {
            len,
            graph: build()?,
        });
    }
    s.as_mut()
        .unwrap()
        .graph
        .run(inputs)
        .into_iter()
        .next()
        .context("graph produced no output")
}

/// `merge_tokenized_segments(overlap=4, token_rate=25)`: keep each window's
/// middle, dropping `(4 // 2) · 25` tokens at every internal seam.
pub fn merge_segments(segments: &[Vec<u32>]) -> Vec<u32> {
    let drop = (TOKENIZER_OVERLAP_FRAMES / 100 / 2) * TOKENS_PER_SECOND;
    let last = segments.len().saturating_sub(1);
    let mut out = Vec::new();
    for (i, s) in segments.iter().enumerate() {
        let l = if i == 0 { 0 } else { drop };
        // Python's `tokens[l:-drop]` is empty once `drop ≥ len`, never negative.
        let r = if i == last {
            s.len()
        } else {
            s.len().saturating_sub(drop)
        };
        if l < r {
            out.extend_from_slice(&s[l..r]);
        }
    }
    out
}

impl ReferenceEncoder {
    /// `s3gen` is `s3gen_meanflow.safetensors` (S3 tokenizer + CAM++ live
    /// there); `speaker` is the repository's `speaker/model.safetensors`.
    pub fn open(s3gen: &Path, speaker: &Path, device: Device) -> Result<Self> {
        let s3 = Arc::new(S3GenWeights::open_prefixes(
            s3gen,
            &["tokenizer.", "speaker_encoder."],
        )?);
        let mut cam: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        for key in s3.keys_with_prefix("speaker_encoder.") {
            let short = key.trim_start_matches("speaker_encoder.").to_string();
            cam.insert(short, (s3.get(&key)?.to_vec(), s3.shape(&key)?.to_vec()));
        }
        let campplus = rlx_funasr::speaker::CamPlus::from_parts(
            rlx_funasr::config::CamPlusConfig::default(),
            rlx_core::weight_map::WeightMap::from_tensors(cam),
            device,
        );
        Ok(Self {
            s3,
            spk: Arc::new(pyannote::open_weights(speaker)?),
            campplus,
            device,
            tokenizer_slot: RefCell::new(None),
            pyannote_slot: RefCell::new(None),
        })
    }

    /// S3 tokenizer encoder output `[T, 1280]` for a 128-bin log-mel `[128, F]`.
    pub fn tokenizer_hidden(&self, mel: &[f32], frames: usize) -> Result<Vec<f32>> {
        ensure!(mel.len() == tokenizer::N_MELS * frames, "log-mel shape");
        let w = self.s3.clone();
        let device = self.device;
        run_slot(
            &self.tokenizer_slot,
            frames,
            move || {
                let mut hir = HirModule::new("s3_tokenizer");
                let mut g = HirMut::new(&mut hir);
                let mut ctx = Ctx::new(&mut g);
                let mi = ctx
                    .g
                    .input("mel", Shape::new(&[1, tokenizer::N_MELS, frames, 1], F32));
                let h = tokenizer::build(&mut ctx, &w, mi)?;
                let params = ctx.into_params();
                hir.set_outputs(vec![h]);
                compile(device, hir, params, "s3 tokenizer")
            },
            &[("mel", mel)],
        )
    }

    /// FSQ codes of tokenizer encoder output `[T, 1280]`.
    pub fn quantize(&self, hidden: &[f32]) -> Result<Vec<u32>> {
        tokenizer::quantize(&self.s3, hidden)
    }

    /// 16 kHz audio → S3 codes (25 Hz).
    ///
    /// Clips over 30 s follow `S3TokenizerV2._quantize_mixed_batch`: the
    /// log-mel (normalized over the whole clip) is cut into 30 s windows every
    /// 26 s, each encoded on its own, and the codes merged by dropping 1 s of
    /// tokens on each side of every seam (`merge_tokenized_segments`).
    /// Upstream zero-pads the last window and masks the padding; encoding it
    /// at its own length is the same computation for every real frame.
    pub fn tokenize(&self, wav16: &[f32]) -> Result<Vec<u32>> {
        let (frames, mel) = dsp::tokenizer_log_mel(wav16);
        ensure!(frames >= 4, "reference too short to tokenize");
        if frames <= TOKENIZER_WINDOW_FRAMES {
            let hidden = self.tokenizer_hidden(&mel, frames)?;
            return self.quantize(&hidden);
        }
        let stride = TOKENIZER_WINDOW_FRAMES - TOKENIZER_OVERLAP_FRAMES;
        let mut segments = Vec::new();
        let mut start = 0;
        while start < frames {
            let len = TOKENIZER_WINDOW_FRAMES.min(frames - start);
            let mut win = Vec::with_capacity(tokenizer::N_MELS * len);
            for m in 0..tokenizer::N_MELS {
                win.extend_from_slice(&mel[m * frames + start..m * frames + start + len]);
            }
            let hidden = self.tokenizer_hidden(&win, len)?;
            segments.push(self.quantize(&hidden)?);
            start += stride;
        }
        Ok(merge_segments(&segments))
    }

    /// pyannote x-vector of 16 kHz audio (raw, not normalized).
    pub fn speaker_raw(&self, wav16: &[f32]) -> Result<Vec<f32>> {
        let n = wav16.len();
        let w = self.spk.clone();
        let device = self.device;
        run_slot(
            &self.pyannote_slot,
            n,
            move || {
                let mut hir = HirModule::new("pyannote_xvector");
                let mut g = HirMut::new(&mut hir);
                let mut ctx = Ctx::new(&mut g);
                let wi = ctx.g.input("wav", Shape::new(&[1, 1, n, 1], F32));
                let e = pyannote::build(&mut ctx, &w, wi)?;
                let params = ctx.into_params();
                hir.set_outputs(vec![e]);
                compile(device, hir, params, "pyannote x-vector")
            },
            &[("wav", wav16)],
        )
    }

    /// CAM++ x-vector of 16 kHz audio.
    pub fn xvector(&self, wav16: &[f32]) -> Result<Vec<f32>> {
        let (frames, mut fb) = dsp::kaldi_fbank(wav16);
        ensure!(frames > 0, "reference too short for CAM++");
        dsp::mean_normalize(&mut fb, frames, 80);
        self.campplus.run_embedding(&fb, frames)
    }

    /// The S3Gen conditioning of a 24 kHz clip (≤ 10 s, ≥ 6 s).
    pub fn s3_conditioning(&self, wav24: &[f32]) -> Result<S3Conditioning> {
        let (prompt_frames, prompt_feat) = dsp::s3gen_mel(wav24);
        let wav16 = rlx_core::audio::resample_sinc(wav24, S3GEN_SR, S3_SR);
        let mut prompt_token = self.tokenize(&wav16)?;
        // `embed_ref`: the mel must be twice the token count; cut the tokens.
        if prompt_frames != 2 * prompt_token.len() {
            prompt_token.truncate(prompt_frames / 2);
        }
        Ok(S3Conditioning {
            prompt_token,
            prompt_feat,
            prompt_frames,
            xvector: self.xvector(&wav16)?,
        })
    }

    /// The S3Gen prompt from a clip at `sr`, cut the way a decoder expects:
    /// at least 6 s (short clips tiled), at most `seconds`, then at most
    /// `max_tokens` prompt tokens with the mel cut to twice that.
    pub fn decoder_conditioning(
        &self,
        wav: &[f32],
        sr: u32,
        seconds: f32,
        max_tokens: Option<usize>,
    ) -> Result<S3Conditioning> {
        ensure!(!wav.is_empty(), "reference audio is empty");
        let mut wav24 = rlx_core::audio::resample_sinc(wav, sr as usize, S3GEN_SR);
        let need = (DECODER_MIN_SECONDS * S3GEN_SR as f32) as usize;
        if wav24.len() < need {
            let reps = need.div_ceil(wav24.len());
            wav24 = wav24.repeat(reps);
        }
        wav24.truncate((seconds * S3GEN_SR as f32) as usize);
        let mut c = self.s3_conditioning(&wav24)?;
        if let Some(max) = max_tokens {
            // `S3Codec._truncate_prompt`: tokens and mel cut together.
            let n = c.prompt_token.len().min(c.prompt_frames / 2).min(max);
            c.prompt_token.truncate(n);
            c.prompt_feat.truncate(2 * n * 80);
            c.prompt_frames = 2 * n;
        }
        Ok(c)
    }

    /// A [`Voice`] from mono audio at `sr` and what it says.
    pub fn voice(&self, name: &str, wav: &[f32], sr: u32, transcript: &str) -> Result<Voice> {
        if transcript.trim().is_empty() {
            bail!("a reference transcript is required (it is part of the prompt)");
        }
        ensure!(!wav.is_empty(), "reference audio is empty");
        let wav16 = rlx_core::audio::resample_sinc(wav, sr as usize, S3_SR);
        let reference_tokens = self.tokenize(&wav16)?;
        let raw = self.speaker_raw(&wav16)?;
        let norm = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
        let speaker_embedding = if norm > 0.0 {
            raw.iter().map(|v| v / norm).collect()
        } else {
            raw
        };

        Ok(Voice {
            name: name.to_string(),
            transcript: transcript.trim().to_string(),
            reference_tokens,
            speaker_embedding,
            s3: self.decoder_conditioning(wav, sr, DECODER_COND_SECONDS, None)?,
            reference_wav: None,
        })
    }

    /// [`Self::voice`] from a wav file.
    pub fn voice_from_wav(&self, path: &Path, transcript: &str) -> Result<Voice> {
        let (pcm, sr) = rlx_core::asr_bench::read_wav_mono(path)
            .with_context(|| format!("read reference {}", path.display()))?;
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "reference".into());
        let mut v = self.voice(&name, &pcm, sr, transcript)?;
        v.reference_wav = Some(path.to_path_buf());
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::merge_segments;

    #[test]
    fn segments_merge_like_upstream() {
        let seg = |a: u32, n: u32| (a..a + n).collect::<Vec<u32>>();
        // three windows: keep [..−50], [50..−50], [50..]
        let m = merge_segments(&[seg(0, 750), seg(1000, 750), seg(2000, 120)]);
        assert_eq!(m.len(), 700 + 650 + 70);
        assert_eq!(m[699], 699);
        assert_eq!(m[700], 1050);
        assert_eq!(*m.last().unwrap(), 2119);
        // a short middle window contributes nothing, like `tokens[50:-50]`
        assert_eq!(
            merge_segments(&[seg(0, 60), seg(100, 80), seg(200, 70)]).len(),
            10 + 20
        );
    }
}
