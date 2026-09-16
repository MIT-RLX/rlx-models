// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Streaming ASR runner for microsoft/VibeVoice-ASR-Streaming-* (safetensors).
// Mirrors `VibeVoiceASRForConditionalGeneration.streaming_generate`.
//
// Default file mode is `encode_then_split` (one VAE pass, then feature chunks) —
// faster and usually more accurate than per-chunk encode. Use
// `EncodeMode::SplitThenEncode` for true mic streaming.

use anyhow::{Result, ensure};
use rlx_runtime::Device;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

use crate::audio::AudioData;
use crate::config::{
    COMPRESS_RATIO, STREAMING_PROMPT_PREFIX, TARGET_SR, TOK_SPEECH_END, TOK_SPEECH_START,
    VibeAsrConfig,
};
use crate::load_streaming::StreamingWeightStore;
use crate::stream_lm::StreamLm;
use crate::tokenizer::VibeTokenizer;
use crate::vae::{VaeEncoderGraph, pad_to_multiple};
use crate::weights::VaeEncoderWeights;

/// Default max new tokens per audio chunk (HF demo default).
pub const DEFAULT_MAX_NEW_PER_CHUNK: usize = 256;

/// How to turn a full waveform into per-chunk speech features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncodeMode {
    /// Encode the whole clip once, then slice feature rows (HF
    /// `encode_then_split`). Best for file transcription.
    #[default]
    EncodeThenSplit,
    /// Encode each (chunk+lookahead) audio segment independently (HF
    /// `split_then_encode`). Needed for live mic streaming.
    SplitThenEncode,
}

pub struct StreamingAsr {
    cfg: VibeAsrConfig,
    acoustic: VaeEncoderWeights,
    semantic: VaeEncoderWeights,
    lm: StreamLm,
    tok: VibeTokenizer,
    device: Device,
    encode_mode: EncodeMode,
    /// Compiled VAE graphs keyed by padded input length (interior mutability
    /// so `&self` encode can reuse them across chunks).
    vae_cache: RefCell<HashMap<usize, (VaeEncoderGraph, VaeEncoderGraph)>>,
}

/// One emitted streaming chunk.
#[derive(Debug, Clone)]
pub struct StreamChunk {
    pub index: usize,
    pub total: usize,
    pub text: String,
}

impl StreamingAsr {
    pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
        let cfg = VibeAsrConfig::from_model_dir(model_dir)?;
        let store = StreamingWeightStore::open(model_dir)?;
        let (acoustic, semantic) = store.load_vae_pair()?;
        let lm = StreamLm::load(store, &cfg.lm, device)?;
        let tok = VibeTokenizer::from_file(&model_dir.join("tokenizer.json"))?;
        let encode_mode = match std::env::var("RLX_VIBEVOICE_ASR_ENCODE")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "split" | "split_then_encode" => EncodeMode::SplitThenEncode,
            _ => EncodeMode::EncodeThenSplit,
        };
        Ok(Self {
            cfg,
            acoustic,
            semantic,
            lm,
            tok,
            device,
            encode_mode,
            vae_cache: RefCell::new(HashMap::new()),
        })
    }

    pub fn config(&self) -> &VibeAsrConfig {
        &self.cfg
    }

    pub fn encode_mode(&self) -> EncodeMode {
        self.encode_mode
    }

    pub fn set_encode_mode(&mut self, mode: EncodeMode) {
        self.encode_mode = mode;
    }

    fn timing_enabled() -> bool {
        std::env::var("RLX_VIBEVOICE_ASR_TIMING").is_ok()
    }

    /// Encode a PCM segment → speech features `[n_frames, hidden]` (acoustic +
    /// semantic sum). Reuses compiled VAE graphs for the same padded length.
    pub fn encode_speech(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        ensure!(!pcm.is_empty(), "empty pcm");
        let padded = pad_to_multiple(pcm, COMPRESS_RATIO);
        let act = self.cfg.vae_ffn;
        let key = padded.len();
        let mut cache = self.vae_cache.borrow_mut();
        if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(key) {
            let a = VaeEncoderGraph::compile_for_streaming(self.device, &self.acoustic, key, act)?;
            let s = VaeEncoderGraph::compile_for_streaming(self.device, &self.semantic, key, act)?;
            e.insert((a, s));
        }
        let (aenc, senc) = cache.get_mut(&key).expect("vae cached");
        let af = aenc.run(&padded)?;
        let sf = senc.run(&padded)?;
        let hidden = self.lm.hidden();
        let na = af.len() / hidden;
        let ns = sf.len() / hidden;
        let n = na.min(ns);
        ensure!(n > 0, "no speech frames from encoders");
        let mut out = vec![0f32; n * hidden];
        for i in 0..n * hidden {
            out[i] = af[i] + sf[i];
        }
        Ok(out)
    }

    /// Build the streaming system prompt (optional hotwords / context_info).
    pub fn build_stream_prompt(&self, context_info: Option<&str>) -> Result<Vec<i64>> {
        let text = match context_info.map(str::trim).filter(|s| !s.is_empty()) {
            Some(info) => format!("{STREAMING_PROMPT_PREFIX} and extra info: {info}\n"),
            None => format!("{STREAMING_PROMPT_PREFIX}\n"),
        };
        Ok(self.tok.encode_plain(&text))
    }

    /// Split audio into (chunk + lookahead) sample segments.
    pub fn split_segments(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let sched = &self.cfg.streaming;
        let chunk = sched.chunk_samples();
        let look = sched.lookahead_samples();
        let target = chunk + look;
        let mut out = Vec::new();
        let mut start = 0usize;
        while start < samples.len() {
            let end = (start + target).min(samples.len());
            if end > start {
                let mut seg = samples[start..end].to_vec();
                if seg.len() < target {
                    seg.resize(target, 0.0);
                }
                out.push(seg);
            }
            let next = start + chunk;
            if next >= samples.len() {
                break;
            }
            start = next;
        }
        if out.is_empty() && !samples.is_empty() {
            let mut seg = samples.to_vec();
            seg.resize(target.max(COMPRESS_RATIO), 0.0);
            out.push(seg);
        }
        out
    }

    /// Slice a full feature matrix into (chunk + lookahead) feature chunks.
    fn split_feature_chunks(&self, feats: &[f32], n_frames: usize) -> Vec<Vec<f32>> {
        let hidden = self.lm.hidden();
        let chunk_f = self.cfg.streaming.chunk_frames;
        let look_f = self.cfg.streaming.lookahead_frames;
        let target_f = chunk_f + look_f;
        let mut out = Vec::new();
        let mut start = 0usize;
        while start < n_frames {
            let end = (start + target_f).min(n_frames);
            if end > start {
                let mut chunk = feats[start * hidden..end * hidden].to_vec();
                let got = end - start;
                if got < target_f {
                    chunk.resize(target_f * hidden, 0.0);
                }
                out.push(chunk);
            }
            let next = start + chunk_f;
            if next >= n_frames {
                break;
            }
            start = next;
        }
        if out.is_empty() && n_frames > 0 {
            let mut chunk = feats[..n_frames * hidden].to_vec();
            chunk.resize(target_f.max(1) * hidden, 0.0);
            out.push(chunk);
        }
        out
    }

    fn clean_chunk_text(&self, ids: &[i64]) -> String {
        let mut text = self.tok.decode(ids, true);
        for st in [
            "<|text_chunk_end|>",
            "<|object_ref_start|>",
            "<|object_ref_end|>",
            "<|box_start|>",
            "<|speech_start|>",
            "<|speech_end|>",
            "<|speech_pad|>",
            "<|im_start|>",
            "<|im_end|>",
        ] {
            text = text.replace(st, "");
        }
        text.trim().to_string()
    }

    /// Run streaming transcription; yields one [`StreamChunk`] per audio chunk.
    pub fn streaming_generate(
        &mut self,
        mono: &[f32],
        src_rate: usize,
        context_info: Option<&str>,
        max_new_per_chunk: usize,
    ) -> Result<Vec<StreamChunk>> {
        let timing = Self::timing_enabled();
        let t0 = std::time::Instant::now();
        let normalize = self.cfg.streaming.normalize_audio;
        let audio = AudioData::from_mono(mono, src_rate, normalize);
        ensure!(!audio.samples.is_empty(), "empty audio");

        let feature_chunks: Vec<Vec<f32>> = match self.encode_mode {
            EncodeMode::EncodeThenSplit => {
                let t_enc = std::time::Instant::now();
                let feats = self.encode_speech(&audio.samples)?;
                let n_frames = feats.len() / self.lm.hidden();
                if timing {
                    eprintln!(
                        "[vibeasr-timing] encode_then_split {:.3}s ({} frames)",
                        t_enc.elapsed().as_secs_f64(),
                        n_frames
                    );
                }
                self.split_feature_chunks(&feats, n_frames)
            }
            EncodeMode::SplitThenEncode => {
                let segs = self.split_segments(&audio.samples);
                let t_enc = std::time::Instant::now();
                let mut out = Vec::with_capacity(segs.len());
                for seg in &segs {
                    out.push(self.encode_speech(seg)?);
                }
                if timing {
                    eprintln!(
                        "[vibeasr-timing] split_then_encode {:.3}s ({} segs)",
                        t_enc.elapsed().as_secs_f64(),
                        segs.len()
                    );
                }
                out
            }
        };
        let total = feature_chunks.len();
        ensure!(total > 0, "no segments");

        let prompt_ids = self.build_stream_prompt(context_info)?;
        let mut prompt_embeds = Vec::with_capacity(prompt_ids.len() * self.lm.hidden());
        for &id in &prompt_ids {
            prompt_embeds.extend_from_slice(self.lm.embed_row(id)?);
        }
        // ~26 speech frames/chunk + specials + max_new; power-of-two ladder.
        let max_total = (prompt_ids.len() + total * (2 + 64 + max_new_per_chunk) + 64)
            .next_power_of_two()
            .max(256) as u64;
        self.lm.prepare_decode_ladder(max_total);

        let t_pf = std::time::Instant::now();
        let (_logits0, mut kv) = self.lm.prefill_embeds(&prompt_embeds, prompt_ids.len())?;
        if timing {
            eprintln!(
                "[vibeasr-timing] prompt prefill {:.3}s ({} tok)",
                t_pf.elapsed().as_secs_f64(),
                prompt_ids.len()
            );
        }

        let sp_start = self.lm.embed_row(TOK_SPEECH_START)?.to_vec();
        let sp_end = self.lm.embed_row(TOK_SPEECH_END)?.to_vec();

        let mut chunks = Vec::with_capacity(total);
        for (idx, feats) in feature_chunks.iter().enumerate() {
            let t_chunk = std::time::Instant::now();
            let n_frames = feats.len() / self.lm.hidden();
            let mut audio_embeds = Vec::with_capacity((n_frames + 2) * self.lm.hidden());
            audio_embeds.extend_from_slice(&sp_start);
            audio_embeds.extend_from_slice(feats);
            audio_embeds.extend_from_slice(&sp_end);

            let logits =
                self.lm
                    .continue_embeds(&audio_embeds, n_frames + 2, &mut kv, max_total)?;
            let ids =
                self.lm
                    .generate_until_chunk_end(&mut kv, &logits, max_new_per_chunk, max_total)?;
            let text = self.clean_chunk_text(&ids);
            if timing {
                eprintln!(
                    "[vibeasr-timing] chunk {}/{} {:.3}s ({} frames, {} new tok) {:?}",
                    idx + 1,
                    total,
                    t_chunk.elapsed().as_secs_f64(),
                    n_frames,
                    ids.len(),
                    &text[..text.len().min(48)]
                );
            }
            chunks.push(StreamChunk {
                index: idx,
                total,
                text,
            });
        }
        if timing {
            let audio_s = audio.samples.len() as f64 / TARGET_SR as f64;
            let wall = t0.elapsed().as_secs_f64();
            eprintln!(
                "[vibeasr-timing] total {:.3}s audio={audio_s:.2}s RTF={:.3}",
                wall,
                wall / audio_s.max(1e-6)
            );
        }
        Ok(chunks)
    }

    /// Join streaming chunks into a single transcript string.
    pub fn transcribe(
        &mut self,
        mono: &[f32],
        src_rate: usize,
        context_info: Option<&str>,
        max_new_per_chunk: usize,
    ) -> Result<String> {
        let chunks = self.streaming_generate(mono, src_rate, context_info, max_new_per_chunk)?;
        Ok(chunks
            .into_iter()
            .map(|c| c.text)
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" "))
    }
}

/// Sample-count helpers exposed for unit tests.
pub fn chunk_sample_counts(cfg: &VibeAsrConfig) -> (usize, usize) {
    (
        cfg.streaming.chunk_samples(),
        cfg.streaming.lookahead_samples(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_respects_chunk_and_lookahead() {
        let cfg = VibeAsrConfig::streaming_7b();
        let (chunk, look) = chunk_sample_counts(&cfg);
        let target = chunk + look;
        let samples = vec![0.1f32; chunk * 2 + look / 2];
        let mut segs = Vec::new();
        let mut start = 0usize;
        while start < samples.len() {
            let end = (start + target).min(samples.len());
            if end > start {
                let mut seg = samples[start..end].to_vec();
                if seg.len() < target {
                    seg.resize(target, 0.0);
                }
                segs.push(seg);
            }
            let next = start + chunk;
            if next >= samples.len() {
                break;
            }
            start = next;
        }
        assert!(!segs.is_empty());
        for s in &segs {
            assert_eq!(s.len(), target);
        }
    }

    #[test]
    fn encode_mode_default_is_file_fast_path() {
        assert_eq!(EncodeMode::default(), EncodeMode::EncodeThenSplit);
    }
}
