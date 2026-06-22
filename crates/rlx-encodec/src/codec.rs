// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// High-level EnCodec codec: device + weights + per-length compiled-graph caches.
// Conv stacks run on the chosen rlx backend; the LSTM bottleneck + euclidean RVQ
// run on the host.

use crate::config::EncodecConfig;
use crate::graph::{DecodePostLstmGraph, DecodePreLstmGraph, PostLstmGraph, PreLstmGraph};
use crate::model::EncodecWeights;
use crate::{eager, lstm};
use anyhow::{Result, ensure};
use rlx_core::{AudioCodec, CodecInfo, RvqCodes};
use rlx_runtime::Device;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

pub struct EncodecCodec {
    weights: EncodecWeights,
    device: Device,
    enc_pre: RefCell<HashMap<usize, PreLstmGraph>>,
    enc_post: RefCell<HashMap<usize, PostLstmGraph>>,
    dec_pre: RefCell<HashMap<usize, DecodePreLstmGraph>>,
    dec_post: RefCell<HashMap<usize, DecodePostLstmGraph>>,
}

impl EncodecCodec {
    pub fn new(weights: EncodecWeights, device: Device) -> Self {
        Self {
            weights,
            device,
            enc_pre: RefCell::new(HashMap::new()),
            enc_post: RefCell::new(HashMap::new()),
            dec_pre: RefCell::new(HashMap::new()),
            dec_post: RefCell::new(HashMap::new()),
        }
    }

    pub fn from_safetensors_path(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())?;
        let weights = EncodecWeights::from_safetensors(&bytes, EncodecConfig::encodec_24khz())?;
        ensure!(
            weights.decoder.is_some(),
            "checkpoint has no decoder weights"
        );
        Ok(Self::new(weights, device))
    }

    pub fn config(&self) -> &EncodecConfig {
        &self.weights.config
    }

    /// Encode mono PCM → codes `[n_q][T]` (quantizer-major).
    pub fn encode(&self, pcm: &[f32], num_quantizers: usize) -> Result<Vec<Vec<u32>>> {
        let enc = &self.weights.encoder;
        let lstm_dim = self.weights.config.lstm_dim();

        let mut pre_cache = self.enc_pre.borrow_mut();
        let pre = match pre_cache.get_mut(&pcm.len()) {
            Some(g) => g,
            None => {
                let g = PreLstmGraph::compile_for(self.device, enc, pcm.len())?;
                pre_cache.entry(pcm.len()).or_insert(g)
            }
        };
        let pre_out = pre.run(pcm)?;
        let t = pre.out_t;
        let post_in = lstm::forward(&enc.lstm, &pre_out, lstm_dim, t);

        let mut post_cache = self.enc_post.borrow_mut();
        let post = match post_cache.get_mut(&t) {
            Some(g) => g,
            None => {
                let g = PostLstmGraph::compile_for(self.device, enc, lstm_dim, t)?;
                post_cache.entry(t).or_insert(g)
            }
        };
        let latent = post.run(&post_in)?;
        Ok(eager::rvq_encode(
            &self.weights.codebooks,
            &latent,
            self.weights.config.hidden_size,
            t,
            num_quantizers,
        ))
    }

    /// Decode codes `[n_q][T]` → mono PCM.
    pub fn decode(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        let dec = self.weights.decoder.as_ref().expect("decoder weights");
        let lstm_dim = self.weights.config.lstm_dim();
        let z_q = eager::rvq_decode(
            &self.weights.codebooks,
            codes,
            self.weights.config.hidden_size,
        );
        let t = codes.first().map(|c| c.len()).unwrap_or(0);

        let mut pre_cache = self.dec_pre.borrow_mut();
        let pre = match pre_cache.get_mut(&t) {
            Some(g) => g,
            None => {
                let g = DecodePreLstmGraph::compile_for(self.device, dec, t)?;
                pre_cache.entry(t).or_insert(g)
            }
        };
        let pre_out = pre.run(&z_q)?;
        let post_in = lstm::forward(&dec.lstm, &pre_out, lstm_dim, t);

        let mut post_cache = self.dec_post.borrow_mut();
        let post = match post_cache.get_mut(&t) {
            Some(g) => g,
            None => {
                let g = DecodePostLstmGraph::compile_for(self.device, dec, lstm_dim, t)?;
                post_cache.entry(t).or_insert(g)
            }
        };
        post.run(&post_in)
    }
}

impl AudioCodec for EncodecCodec {
    fn info(&self) -> CodecInfo {
        let cfg = &self.weights.config;
        CodecInfo {
            sample_rate: cfg.sampling_rate,
            frame_rate: cfg.sampling_rate as f32 / cfg.hop_length() as f32,
            hop_length: cfg.hop_length(),
            channels: cfg.audio_channels,
            max_quantizers: self.weights.codebooks.len(),
            codebook_size: cfg.codebook_size,
        }
    }

    fn device(&self) -> Device {
        self.device
    }

    fn encode_pcm(&self, pcm: &[f32], num_quantizers: Option<usize>) -> Result<RvqCodes> {
        let nq = num_quantizers
            .unwrap_or(self.weights.codebooks.len())
            .min(self.weights.codebooks.len());
        let codes = self.encode(pcm, nq)?;
        Ok(RvqCodes::from_quantizer_major(&codes))
    }

    fn decode_codes(&self, codes: &RvqCodes) -> Result<Vec<f32>> {
        self.decode(&codes.to_quantizer_major())
    }
}
