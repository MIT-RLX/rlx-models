// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//! Staged Moonshine execution: encode PCM → bucketed decoder → host LM head.

use crate::config::MoonshineConfig;
use crate::weight_source::CloningWeightSource;
use crate::weights::MoonshineWeightPrefix;
use anyhow::{Result, anyhow, bail};
use rayon::prelude::*;
use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Loaded Moonshine model with lazily compiled encoder/decoder graphs.
pub struct MoonshineModel {
    cfg: MoonshineConfig,
    device: Device,
    weights: WeightMap,
    pfx: MoonshineWeightPrefix,
    /// Host `embed_tokens` / tied `proj_out` `[vocab, d]`.
    embed_table: Vec<f32>,
    encoders: HashMap<usize, CompiledGraph>,
    /// Decoder graphs keyed by `bucket * 1_000_000 + enc_seq`.
    decoders: HashMap<usize, CompiledGraph>,
    embed_scratch: Vec<f32>,
    hidden_row_scratch: Vec<f32>,
    embed_cache: Option<EmbedCache>,
    #[cfg(feature = "tokenizer")]
    tokenizer: Option<tokenizers::Tokenizer>,
    #[cfg(feature = "tokenizer")]
    tokenizer_path: Option<PathBuf>,
}

/// Decoder sequence-length buckets (Florence-style full-prefix graphs).
pub(crate) const DECODER_BUCKETS: [usize; 11] = [4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256];

/// Buckets pre-compiled before timed ASR (longer buckets compile on first use).
const PRECOMPILE_DECODER_CAP: usize = 128;

#[derive(Debug, Clone, Copy)]
struct EmbedCache {
    cap: usize,
    enc_seq: usize,
    filled_cur: usize,
}

impl MoonshineModel {
    pub fn config(&self) -> &MoonshineConfig {
        &self.cfg
    }
    pub fn device(&self) -> Device {
        self.device
    }

    /// Load from a safetensors checkpoint directory or file.
    pub fn load(weights_path: &Path, cfg: MoonshineConfig, device: Device) -> Result<Self> {
        rlx_core::validate_standard_device("moonshine", device)?;
        let mut weights = if weights_path.is_dir() {
            WeightMap::from_safetensors_dir(weights_path)?
        } else {
            WeightMap::from_file(
                weights_path
                    .to_str()
                    .ok_or_else(|| anyhow!("non-UTF8 weights path"))?,
            )?
        };
        let pfx = MoonshineWeightPrefix::detect(&weights);
        let embed_key = pfx.dec_embed_tokens();
        let (embed_table, shape) = if weights.has(&embed_key) {
            weights.take(&embed_key)?
        } else if let Some(ref pk) = pfx.proj_out {
            weights.take(pk)?
        } else {
            bail!("moonshine: missing embed_tokens / proj_out weights");
        };
        if shape.len() != 2 || shape[0] != cfg.vocab_size || shape[1] != cfg.hidden_size {
            // Allow slight mismatch for synth / truncated embeds.
            if shape.len() != 2 || shape[1] != cfg.hidden_size {
                bail!(
                    "moonshine: embed shape {shape:?} incompatible with vocab={} d={}",
                    cfg.vocab_size,
                    cfg.hidden_size
                );
            }
        }
        // Drop tied proj_out if still present (embedding is host-side).
        if let Some(ref pk) = pfx.proj_out {
            let _ = weights.take(pk);
        }

        #[cfg(feature = "tokenizer")]
        let (tokenizer, tokenizer_path) = {
            let dir = if weights_path.is_dir() {
                weights_path.to_path_buf()
            } else {
                weights_path
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .to_path_buf()
            };
            let tp = dir.join("tokenizer.json");
            if tp.is_file() {
                let tk = tokenizers::Tokenizer::from_file(&tp)
                    .map_err(|e| anyhow!("load tokenizer {}: {e}", tp.display()))?;
                (Some(tk), Some(tp))
            } else {
                (None, None)
            }
        };

        Ok(Self {
            cfg,
            device,
            weights,
            pfx,
            embed_table,
            encoders: HashMap::new(),
            decoders: HashMap::new(),
            embed_scratch: Vec::new(),
            hidden_row_scratch: Vec::new(),
            embed_cache: None,
            #[cfg(feature = "tokenizer")]
            tokenizer,
            #[cfg(feature = "tokenizer")]
            tokenizer_path,
        })
    }

    /// Construct from an in-memory [`WeightMap`] (synthetic / tests).
    pub fn from_weight_map(
        weights: WeightMap,
        cfg: MoonshineConfig,
        device: Device,
    ) -> Result<Self> {
        rlx_core::validate_standard_device("moonshine", device)?;
        let mut weights = weights;
        let pfx = MoonshineWeightPrefix::detect(&weights);
        let embed_key = pfx.dec_embed_tokens();
        let (embed_table, _) = if weights.has(&embed_key) {
            weights.take(&embed_key)?
        } else if let Some(ref pk) = pfx.proj_out {
            weights.take(pk)?
        } else {
            bail!("moonshine: missing embed_tokens");
        };
        if let Some(ref pk) = pfx.proj_out {
            let _ = weights.take(pk);
        }
        Ok(Self {
            cfg,
            device,
            weights,
            pfx,
            embed_table,
            encoders: HashMap::new(),
            decoders: HashMap::new(),
            embed_scratch: Vec::new(),
            hidden_row_scratch: Vec::new(),
            embed_cache: None,
            #[cfg(feature = "tokenizer")]
            tokenizer: None,
            #[cfg(feature = "tokenizer")]
            tokenizer_path: None,
        })
    }

    /// Encode raw PCM (`f32` mono @ 16 kHz) → flat encoder hidden `[T * d]`.
    pub fn encode_pcm(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        let audio_len = pcm.len();
        let enc_seq = MoonshineConfig::feat_extract_output_length(audio_len);
        if enc_seq == 0 {
            bail!("moonshine: PCM length {audio_len} too short (need ≥895 samples @ 16 kHz)");
        }
        if !self.encoders.contains_key(&audio_len) {
            self.ensure_encoder(audio_len)?;
        }
        let g = self.encoders.get_mut(&audio_len).unwrap();
        g.run(&[("pcm", pcm)])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("encoder graph produced no output"))
    }

    /// Next-token logits for the last real position of `token_ids` (bucketed pad).
    pub fn decode_logits(
        &mut self,
        token_ids: &[u32],
        encoder_hidden: &[f32],
        enc_seq: usize,
        cap: usize,
    ) -> Result<Vec<f32>> {
        self.decode_hidden_row(token_ids, encoder_hidden, enc_seq, cap)?;
        Ok(self.lm_head(&self.hidden_row_scratch))
    }

    /// Greedy next token (tied LM head argmax, no `[vocab]` logits alloc).
    pub(crate) fn decode_next_token(
        &mut self,
        token_ids: &[u32],
        encoder_hidden: &[f32],
        enc_seq: usize,
        cap: usize,
    ) -> Result<u32> {
        self.decode_hidden_row(token_ids, encoder_hidden, enc_seq, cap)?;
        Ok(self.lm_head_argmax(&self.hidden_row_scratch))
    }

    fn decode_hidden_row(
        &mut self,
        token_ids: &[u32],
        encoder_hidden: &[f32],
        enc_seq: usize,
        cap: usize,
    ) -> Result<()> {
        let d = self.cfg.hidden_size;
        let cur = token_ids.len();
        debug_assert!(cur >= 1 && cur <= cap);
        let cap = bucket_len(cur).min(cap).max(cur);
        let key = cap * 1_000_000 + enc_seq;
        if !self.decoders.contains_key(&key) {
            self.ensure_decoder(cap, enc_seq)?;
        }
        self.fill_decoder_embeds(token_ids, enc_seq, cap, cur, d);
        let hidden = {
            let g = self.decoders.get_mut(&key).unwrap();
            g.run(&[
                ("decoder_inputs_embeds", self.embed_scratch.as_slice()),
                ("encoder_hidden", encoder_hidden),
            ])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("decoder graph produced no output"))?
        };
        let row_start = (cur - 1) * d;
        self.hidden_row_scratch.resize(d, 0.0);
        self.hidden_row_scratch
            .copy_from_slice(&hidden[row_start..row_start + d]);
        Ok(())
    }

    fn fill_decoder_embeds(
        &mut self,
        token_ids: &[u32],
        enc_seq: usize,
        cap: usize,
        cur: usize,
        d: usize,
    ) {
        let pad = self.cfg.pad_token_id as usize;
        self.embed_scratch.resize(cap * d, 0.0);
        let incremental = self
            .embed_cache
            .is_some_and(|c| c.cap == cap && c.enc_seq == enc_seq && c.filled_cur + 1 == cur);
        if incremental {
            let i = cur - 1;
            let tok = token_ids[i] as usize;
            let src = tok * d;
            let row = &self.embed_table[src..src + d];
            self.embed_scratch[i * d..(i + 1) * d].copy_from_slice(row);
        } else {
            for i in 0..cap {
                let tok = if i < cur { token_ids[i] as usize } else { pad };
                let src = tok * d;
                let row = &self.embed_table[src..src + d];
                self.embed_scratch[i * d..(i + 1) * d].copy_from_slice(row);
            }
        }
        self.embed_cache = Some(EmbedCache {
            cap,
            enc_seq,
            filled_cur: cur,
        });
    }

    pub(crate) fn reset_decode_state(&mut self) {
        self.embed_cache = None;
    }

    /// Pre-compile encoder graphs for upcoming VAD chunk PCM lengths.
    pub fn prepare_encoders(&mut self, pcm_lengths: &[usize]) -> Result<()> {
        for &len in pcm_lengths {
            if len >= 895 && MoonshineConfig::feat_extract_output_length(len) > 0 {
                self.ensure_encoder(len)?;
            }
        }
        Ok(())
    }

    /// Pre-compile decoder bucket graphs for the given encoder sequence lengths.
    pub fn prepare_decoders(&mut self, enc_seqs: &[usize], max_dec_len: usize) -> Result<()> {
        let max_dec = max_dec_len.min(self.cfg.max_position_embeddings).max(1);
        for &enc_seq in enc_seqs {
            if enc_seq == 0 {
                continue;
            }
            for cap in DECODER_BUCKETS {
                if cap <= max_dec {
                    self.ensure_decoder(cap, enc_seq)?;
                }
            }
        }
        Ok(())
    }

    /// Compile all encoder/decoder graphs needed for a chunked transcription pass.
    pub fn prepare_for_transcription(&mut self, pcm_lengths: &[usize]) -> Result<()> {
        self.prepare_encoders(pcm_lengths)?;
        let enc_seqs: Vec<usize> = pcm_lengths
            .iter()
            .filter_map(|&len| {
                let t = MoonshineConfig::feat_extract_output_length(len);
                if t > 0 { Some(t) } else { None }
            })
            .collect();
        let precompile_cap = self
            .cfg
            .max_position_embeddings
            .clamp(1, PRECOMPILE_DECODER_CAP);
        self.prepare_decoders(&enc_seqs, precompile_cap)
    }

    fn ensure_encoder(&mut self, audio_len: usize) -> Result<()> {
        if self.encoders.contains_key(&audio_len) {
            return Ok(());
        }
        let mut src = CloningWeightSource(&self.weights);
        let built = crate::flow::build_encoder_built(&self.cfg, &mut src, &self.pfx, 1, audio_len)?;
        self.encoders
            .insert(audio_len, compile_built(built, self.device)?);
        Ok(())
    }

    fn ensure_decoder(&mut self, cap: usize, enc_seq: usize) -> Result<()> {
        let key = cap * 1_000_000 + enc_seq;
        if self.decoders.contains_key(&key) {
            return Ok(());
        }
        let mut src = CloningWeightSource(&self.weights);
        let built = crate::flow::build_decoder_hidden_built(
            &self.cfg, &mut src, &self.pfx, 1, cap, enc_seq,
        )?;
        self.decoders
            .insert(key, compile_built(built, self.device)?);
        Ok(())
    }

    fn lm_head_argmax(&self, hidden_row: &[f32]) -> u32 {
        let d = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size.min(self.embed_table.len() / d);
        self.embed_table[..vocab * d]
            .par_chunks(d)
            .enumerate()
            .map(|(v, row)| {
                let mut acc = 0f32;
                for j in 0..d {
                    acc += hidden_row[j] * row[j];
                }
                (v as u32, acc)
            })
            .reduce(
                || (0u32, f32::NEG_INFINITY),
                |a, b| if b.1 > a.1 { b } else { a },
            )
            .0
    }

    fn lm_head(&self, hidden_row: &[f32]) -> Vec<f32> {
        let d = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size.min(self.embed_table.len() / d);
        let table = &self.embed_table;
        let mut logits = vec![0f32; vocab];
        logits.par_iter_mut().enumerate().for_each(|(v, out)| {
            let row = &table[v * d..v * d + d];
            let mut acc = 0f32;
            for j in 0..d {
                acc += hidden_row[j] * row[j];
            }
            *out = acc;
        });
        logits
    }

    /// Greedy transcribe: encode → decode until EOS / max length → detokenize.
    pub fn transcribe(&mut self, pcm: &[f32]) -> Result<String> {
        let enc = self.encode_pcm(pcm)?;
        let enc_seq = enc.len() / self.cfg.hidden_size;
        let ids = self.generate_greedy(&enc, enc_seq)?;
        self.decode_tokens(&ids)
    }

    fn decode_tokens(&self, ids: &[u32]) -> Result<String> {
        // Skip decoder_start; drop trailing eos.
        let mut slice = ids;
        if let Some(&first) = slice.first()
            && first == self.cfg.decoder_start_token_id
        {
            slice = &slice[1..];
        }
        if let Some(&last) = slice.last()
            && last == self.cfg.eos_token_id
        {
            slice = &slice[..slice.len() - 1];
        }
        #[cfg(feature = "tokenizer")]
        {
            if let Some(ref tk) = self.tokenizer {
                return tk
                    .decode(slice, true)
                    .map_err(|e| anyhow!("tokenizer decode: {e}"));
            }
            if let Some(ref p) = self.tokenizer_path {
                bail!("tokenizer failed to load from {}", p.display());
            }
            bail!(
                "moonshine: no tokenizer.json beside weights — place HuggingFace tokenizer.json next to the checkpoint"
            );
        }
        #[cfg(not(feature = "tokenizer"))]
        {
            let _ = slice;
            bail!("moonshine: rebuild with `--features tokenizer` to detokenize")
        }
    }
}

fn bucket_len(n: usize) -> usize {
    for b in DECODER_BUCKETS {
        if b >= n {
            return b;
        }
    }
    n.next_power_of_two()
}

/// Builder-style runner wrapping [`MoonshineModel`].
pub struct MoonshineRunner {
    model: MoonshineModel,
}

impl MoonshineRunner {
    pub fn builder() -> MoonshineRunnerBuilder {
        MoonshineRunnerBuilder::default()
    }

    pub fn config(&self) -> &MoonshineConfig {
        self.model.config()
    }

    pub fn encode_pcm(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        self.model.encode_pcm(pcm)
    }

    pub fn transcribe(&mut self, pcm: &[f32]) -> Result<String> {
        self.model.transcribe(pcm)
    }

    pub fn prepare_for_transcription(&mut self, pcm_lengths: &[usize]) -> Result<()> {
        self.model.prepare_for_transcription(pcm_lengths)
    }

    pub fn model_mut(&mut self) -> &mut MoonshineModel {
        &mut self.model
    }
}

#[derive(Debug, Default)]
pub struct MoonshineRunnerBuilder {
    weights: Option<PathBuf>,
    config: Option<MoonshineConfig>,
    device: Option<Device>,
}

impl MoonshineRunnerBuilder {
    pub fn weights(mut self, path: impl Into<PathBuf>) -> Self {
        self.weights = Some(path.into());
        self
    }

    pub fn config(mut self, cfg: MoonshineConfig) -> Self {
        self.config = Some(cfg);
        self
    }

    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }

    pub fn build(self) -> Result<MoonshineRunner> {
        let device = self.device.unwrap_or(Device::Cpu);
        let path = self
            .weights
            .ok_or_else(|| anyhow!("moonshine: .weights(path) required"))?;
        let cfg = if let Some(c) = self.config {
            c
        } else if path.is_dir() {
            MoonshineConfig::from_dir(&path).unwrap_or_else(|_| MoonshineConfig::tiny())
        } else {
            path.parent()
                .and_then(|d| MoonshineConfig::from_dir(d).ok())
                .unwrap_or_else(MoonshineConfig::tiny)
        };
        let model = MoonshineModel::load(&path, cfg, device)?;
        Ok(MoonshineRunner { model })
    }
}
