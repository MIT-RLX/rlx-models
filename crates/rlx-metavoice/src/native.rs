// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! MetaVoice native runtime: first-stage GPT + second-stage fine books + EnCodec.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use rlx_encodec::EncodecCodec;
use rlx_runtime::Device;
use safetensors::SafeTensors;

use crate::config::{DEFAULT_LOCAL_DIR, FirstStageArgs, SAMPLE_RATE, SecondStageArgs};
use crate::first_stage::{self, FirstStage};
use crate::second_stage::{SecondStage, TEXT_OFFSET as SECOND_TEXT_OFFSET};
use crate::speaker::SpeakerEncoder;
use crate::tokenize::MetaTokenizer;

pub const DEFAULT_ENCODEC_PATH: &str = "weights/tts/encodec24/model.safetensors";
pub const DEFAULT_REFERENCE: &str = "weights/tts/metavoice/bria_16k.wav";

/// Fox-sentence content words used by Whisper harnesses (≥5/6 target).
pub const FOX_WORDS: [&str; 6] = ["quick", "brown", "fox", "jumps", "lazy", "dog"];

#[derive(Debug, Clone, Copy)]
pub struct InferOpts {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub guidance_scale: f32,
    pub seed: u64,
    /// Greedy argmax (default). Stochastic top-p needs an explicit `--sample`.
    pub greedy: bool,
}

impl Default for InferOpts {
    fn default() -> Self {
        Self {
            // Official demos use 864; shorter budgets (e.g. 448) truncate mid-phrase.
            max_new_tokens: 864,
            temperature: 1.0,
            top_p: 0.95,
            guidance_scale: 3.0,
            seed: 1337,
            // Deterministic: top-p seed 1337 without a speaker emb was the
            // "To quick brain fox straps…" 67% Whisper failure mode.
            greedy: true,
        }
    }
}

/// Collapse whitespace / strip — matches MetaVoice `normalize_text` (ASCII path).
pub fn normalize_text(text: &str) -> String {
    let t = text
        .replace(['\t', '\n', '\r', '*'], " ")
        .trim()
        .to_string();
    let mut out = String::with_capacity(t.len());
    let mut prev_space = false;
    for ch in t.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    out
}

/// Trim near-silence and peak-normalize so Whisper sees a consistent level.
pub fn postprocess_pcm(pcm: &[f32], sample_rate: u32) -> Vec<f32> {
    if pcm.is_empty() {
        return Vec::new();
    }
    let peak = pcm.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    if peak < 1e-6 {
        return pcm.to_vec();
    }
    // ~40 dB below peak counts as silence for trim.
    let thr = peak * 0.01;
    let mut lo = 0usize;
    while lo < pcm.len() && pcm[lo].abs() < thr {
        lo += 1;
    }
    let mut hi = pcm.len();
    while hi > lo && pcm[hi - 1].abs() < thr {
        hi -= 1;
    }
    // Keep a little padding so Whisper does not clip the first/last phone.
    let pad = (sample_rate as usize / 50).max(1); // 20 ms
    lo = lo.saturating_sub(pad);
    hi = (hi + pad).min(pcm.len());
    if lo >= hi {
        return pcm.to_vec();
    }
    let slice = &pcm[lo..hi];
    let peak2 = slice.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-6);
    let gain = 0.95 / peak2;
    slice.iter().map(|&x| (x * gain).clamp(-1.0, 1.0)).collect()
}

pub struct MetaVoice {
    #[allow(dead_code)]
    dir: PathBuf,
    device: Device,
    tok: MetaTokenizer,
    first_args: FirstStageArgs,
    second_args: SecondStageArgs,
    first: FirstStage,
    second: SecondStage,
    speaker: SpeakerEncoder,
    encodec: EncodecCodec,
}

impl MetaVoice {
    pub fn open(dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        Self::open_with_encodec(dir, DEFAULT_ENCODEC_PATH, device)
    }

    pub fn open_with_encodec(
        dir: impl AsRef<Path>,
        encodec: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let tok = MetaTokenizer::load(dir.join("tokenizer_metavoice.json"))
            .context("load MetaVoice tokenizer")?;
        let first_args = load_first_args(&dir)?;
        let second_args = load_second_args(&dir)?;
        let first_weights = load_st(&dir.join("first_stage.safetensors"))?;
        let first = FirstStage::from_weights(&first_args, &first_weights)
            .context("build first-stage GPT")?;
        let second_weights = load_st(&dir.join("second_stage.safetensors"))?;
        let second =
            SecondStage::from_weights(&second_args, &second_weights, first_args.speaker_emb_size)
                .context("build second-stage")?;
        let speaker_weights = load_st(&dir.join("speaker_encoder.safetensors"))?;
        let speaker =
            SpeakerEncoder::from_weights(&speaker_weights).context("build speaker encoder")?;
        let encodec = EncodecCodec::from_safetensors_path(encodec.as_ref(), device)
            .with_context(|| format!("open EnCodec at {}", encodec.as_ref().display()))?;
        Ok(Self {
            dir,
            device,
            tok,
            first_args,
            second_args,
            first,
            second,
            speaker,
            encodec,
        })
    }

    pub fn open_default(device: Device) -> Result<Self> {
        Self::open(DEFAULT_LOCAL_DIR, device)
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn first_args(&self) -> &FirstStageArgs {
        &self.first_args
    }

    pub fn second_args(&self) -> &SecondStageArgs {
        &self.second_args
    }

    pub fn tokenizer(&self) -> &MetaTokenizer {
        &self.tok
    }

    pub fn weight_counts(&self) -> (usize, usize) {
        (self.first_args.n_layer, self.first_args.speaker_emb_size)
    }

    /// Speaker embedding from a reference wav (16 kHz preferred).
    pub fn embed_reference(&self, path: &Path) -> Result<Vec<f32>> {
        self.speaker.embed_wav_path(path)
    }

    /// Resolve reference path: explicit → default Bria → error (zero emb kills intelligibility).
    pub fn resolve_speaker_emb(&self, reference_wav: Option<&Path>) -> Result<Vec<f32>> {
        if let Some(p) = reference_wav {
            return self.embed_reference(p);
        }
        let default = self.dir.join("bria_16k.wav");
        if default.is_file() {
            return self.embed_reference(&default);
        }
        let bundled = PathBuf::from(DEFAULT_REFERENCE);
        if bundled.is_file() {
            return self.embed_reference(&bundled);
        }
        Err(anyhow!(
            "speaker reference required (zero emb collapses intelligibility); \
             pass --reference WAV or place bria_16k.wav under {} / {}",
            self.dir.display(),
            DEFAULT_REFERENCE
        ))
    }

    /// BPE encode then add first-stage text offset (+ append EOT).
    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_with_offset(text, self.tok.offset)
    }

    /// BPE encode with second-stage text offset (+ append EOT).
    pub fn encode_text_second(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_with_offset(text, SECOND_TEXT_OFFSET)
    }

    fn encode_with_offset(&self, text: &str, off: u32) -> Result<Vec<u32>> {
        let mut ids = self.tok.encode(text)?;
        for id in &mut ids {
            *id += off;
        }
        ids.push(self.tok.eot_id() + off);
        Ok(ids)
    }

    fn ensure_speaker_emb(spk_emb: &[f32]) -> Result<()> {
        let energy: f32 = spk_emb.iter().map(|x| x * x).sum();
        if energy < 1e-8 {
            return Err(anyhow!(
                "speaker embedding is ~zero; provide a reference wav (e.g. bria_16k.wav)"
            ));
        }
        Ok(())
    }

    /// First-stage AR only → interleaved tokens (CPU eager).
    pub fn generate_tokens(
        &self,
        text: &str,
        spk_emb: &[f32],
        opts: &InferOpts,
    ) -> Result<Vec<u32>> {
        let text = normalize_text(text);
        if text.is_empty() {
            return Err(anyhow!("empty text"));
        }
        Self::ensure_speaker_emb(spk_emb)?;
        let prompt = self.encode_text(&text)?;
        let temp = if opts.greedy { 0.0 } else { opts.temperature };
        self.first.generate(
            &prompt,
            spk_emb,
            opts.max_new_tokens,
            opts.guidance_scale,
            temp,
            opts.top_p,
            opts.seed,
        )
    }

    /// First-stage → second-stage → 8 EnCodec codebooks.
    pub fn tokens_to_codes(
        &self,
        text: &str,
        tokens: &[u32],
        spk_emb: &[f32],
    ) -> Result<Vec<Vec<u32>>> {
        let text = normalize_text(text);
        let (c0, c1) = first_stage::extract_codebooks(tokens);
        if c0.is_empty() {
            return Err(anyhow!(
                "first stage produced no audio codes ({} tokens, eos={})",
                tokens.len(),
                tokens.contains(&first_stage::EOS_AUDIO)
            ));
        }
        let text_ids = self.encode_text_second(&text)?;
        let fine = self
            .second
            .predict_fine(&text_ids, &c0, &c1, spk_emb)
            .context("second-stage predict")?;
        let mut codes = vec![c0, c1];
        codes.extend(fine);
        anyhow::ensure!(
            codes.len() == 8,
            "expected 8 codebooks, got {}",
            codes.len()
        );
        Ok(codes)
    }

    /// EnCodec decode of precomputed 8 codebooks on this session's device.
    /// Output is silence-trimmed and peak-normalized for ASR / listening.
    pub fn decode_codes(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        let pcm = self.encodec.decode(codes).context("EnCodec decode")?;
        Ok(postprocess_pcm(&pcm, SAMPLE_RATE))
    }

    /// Full pipeline: first-stage tokens → fine books → EnCodec PCM (+ postprocess).
    pub fn decode_tokens(&self, text: &str, tokens: &[u32], spk_emb: &[f32]) -> Result<Vec<f32>> {
        let text = normalize_text(text);
        let codes = self.tokens_to_codes(&text, tokens, spk_emb)?;
        self.decode_codes(&codes)
    }

    /// Synthesize `text`. Uses `reference_wav` (or bundled Bria) for speaker LSTM.
    pub fn synthesize(
        &self,
        text: &str,
        reference_wav: Option<&Path>,
        opts: &InferOpts,
    ) -> Result<Vec<f32>> {
        let text = normalize_text(text);
        let spk = self.resolve_speaker_emb(reference_wav)?;
        let tokens = self.generate_tokens(&text, &spk, opts)?;
        self.decode_tokens(&text, &tokens, &spk)
    }

    pub fn write_wav(&self, audio: &[f32], path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec)
            .with_context(|| format!("create {}", path.display()))?;
        for &s in audio {
            w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        }
        w.finalize()?;
        Ok(())
    }
}

fn load_st(path: &Path) -> Result<HashMap<String, Vec<f32>>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st =
        SafeTensors::deserialize(&bytes).with_context(|| format!("parse {}", path.display()))?;
    let mut out = HashMap::new();
    for name in st.names() {
        let t = st.tensor(name)?;
        anyhow::ensure!(
            matches!(t.dtype(), safetensors::Dtype::F32),
            "unsupported dtype {:?} for {name}",
            t.dtype()
        );
        let vals: Vec<f32> = t
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        out.insert(name.to_string(), vals);
    }
    Ok(out)
}

fn load_first_args(dir: &Path) -> Result<FirstStageArgs> {
    let p = dir.join("first_stage_args.json");
    if !p.is_file() {
        return Ok(FirstStageArgs::default());
    }
    #[derive(serde::Deserialize)]
    struct Wrap {
        model_args: FirstStageArgs,
        #[serde(default)]
        meta: Option<MetaBits>,
    }
    #[derive(serde::Deserialize)]
    struct MetaBits {
        #[serde(default)]
        speaker_emb_size: Option<usize>,
    }
    let w: Wrap = serde_json::from_str(&std::fs::read_to_string(&p)?)?;
    let mut args = w.model_args;
    if let Some(Some(s)) = w.meta.map(|m| m.speaker_emb_size) {
        args.speaker_emb_size = s;
    }
    Ok(args)
}

fn load_second_args(dir: &Path) -> Result<SecondStageArgs> {
    let p = dir.join("second_stage_args.json");
    if !p.is_file() {
        return Ok(SecondStageArgs::default());
    }
    #[derive(serde::Deserialize)]
    struct Wrap {
        model_args: SecondStageArgs,
    }
    Ok(serde_json::from_str::<Wrap>(&std::fs::read_to_string(&p)?)?.model_args)
}
