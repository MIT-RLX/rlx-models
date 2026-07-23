// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! Streaming Conformer encoder — chunk 64 / lookahead 16, 28 layers.
//!
//! Until the folded encoder graph is wired from `model.gguf`,
//! [`Encoder::forward_stub`] supplies shaped outputs for the pipeline.

use crate::spec::{
    AED_WINDOW_FRAMES, CHUNK_FRAMES, DECODER_DIM, DECODER_HEADS, DECODER_HEAD_DIM, LOOKAHEAD_FRAMES,
    MEL_BINS, SUBSAMPLE, VOCAB,
};
use anyhow::{bail, Result};
use std::path::Path;

pub const ENCODER_LAYERS: usize = 28;
pub const CNN_KERNEL: usize = 7;
pub const ATT_K_CACHE: [usize; 4] = [4, 16, DECODER_HEADS, DECODER_HEAD_DIM];
pub const ATT_V_CACHE: [usize; 4] = [ENCODER_LAYERS, 64, DECODER_HEADS, 16];
pub const CNN_CACHE: [usize; 4] = [ENCODER_LAYERS, DECODER_DIM, 1, CNN_KERNEL];

pub struct EncoderOutputs {
    pub wp_logprob: Vec<f32>,
    pub encoder_cache: Vec<f32>,
    pub n_frames: usize,
}

pub struct Encoder {
    pub n_layers: usize,
    pub chunk: usize,
    pub lookahead: usize,
}

impl Default for Encoder {
    fn default() -> Self {
        Self {
            n_layers: ENCODER_LAYERS,
            chunk: CHUNK_FRAMES,
            lookahead: LOOKAHEAD_FRAMES,
        }
    }
}

impl Encoder {
    pub fn load(_weights: &Path) -> Result<Self> {
        Ok(Self::default())
    }

    /// Deterministic stub until a native weight pack is loaded.
    pub fn forward_stub(&self, mel: &[Vec<f32>]) -> Result<EncoderOutputs> {
        if mel.is_empty() {
            bail!("empty mel");
        }
        if mel[0].len() != MEL_BINS {
            bail!("expected {MEL_BINS} mel bins");
        }
        let n_frames = (mel.len() / SUBSAMPLE).max(1);
        let mut wp_logprob = vec![0f32; n_frames * VOCAB];
        for t in 0..n_frames {
            let row = &mut wp_logprob[t * VOCAB..(t + 1) * VOCAB];
            let v = -(VOCAB as f32).ln();
            row.fill(v);
            row[0] = 0.0;
        }
        let mut encoder_cache = vec![0f32; AED_WINDOW_FRAMES * DECODER_DIM];
        for i in 0..encoder_cache.len() {
            encoder_cache[i] = 0.02 * ((i as f32) * 0.001).sin();
        }
        for (t, frame) in mel.iter().take(AED_WINDOW_FRAMES).enumerate() {
            let e: f32 = frame.iter().copied().sum::<f32>() / MEL_BINS as f32;
            for d in 0..DECODER_DIM.min(80) {
                encoder_cache[t * DECODER_DIM + d] += 0.01 * e;
            }
        }
        Ok(EncoderOutputs {
            wp_logprob,
            encoder_cache,
            n_frames,
        })
    }

    pub fn cache_from_bin(path: &Path) -> Result<Vec<f32>> {
        let bytes = std::fs::read(path)?;
        if bytes.len() % 4 != 0 {
            bail!("encoder_cache bin size not multiple of 4");
        }
        let mut v = vec![0f32; bytes.len() / 4];
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            v[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        if v.len() < AED_WINDOW_FRAMES * DECODER_DIM {
            v.resize(AED_WINDOW_FRAMES * DECODER_DIM, 0.0);
        }
        v.truncate(AED_WINDOW_FRAMES * DECODER_DIM);
        Ok(v)
    }
}
