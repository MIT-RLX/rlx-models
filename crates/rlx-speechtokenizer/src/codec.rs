// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// High-level SpeechTokenizer codec implementing the unified AudioCodec trait.

use crate::config::SpeechTokenizerConfig;
use crate::model::{LstmLayerW, StWeights};
use crate::{eager, graph, lstm};
use anyhow::{Result, ensure};
use rlx_core::{AudioCodec, CodecInfo, RvqCodes};
use rlx_runtime::Device;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

fn layers(ls: &[LstmLayerW]) -> Vec<lstm::LayerW<'_>> {
    ls.iter()
        .map(|l| lstm::LayerW {
            w_ih: &l.w_ih,
            w_hh: &l.w_hh,
            b_ih: &l.b_ih,
            b_hh: &l.b_hh,
        })
        .collect()
}

pub struct SpeechTokenizerCodec {
    w: StWeights,
    device: Device,
    enc_pre: RefCell<HashMap<usize, graph::StGraph>>,
    enc_post: RefCell<HashMap<usize, graph::StGraph>>,
    dec_pre: RefCell<HashMap<usize, graph::StGraph>>,
    dec_post: RefCell<HashMap<usize, graph::StGraph>>,
}

impl SpeechTokenizerCodec {
    pub fn new(w: StWeights, device: Device) -> Self {
        Self {
            w,
            device,
            enc_pre: RefCell::new(HashMap::new()),
            enc_post: RefCell::new(HashMap::new()),
            dec_pre: RefCell::new(HashMap::new()),
            dec_post: RefCell::new(HashMap::new()),
        }
    }

    pub fn from_safetensors_path(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        let w = StWeights::from_safetensors(
            &std::fs::read(path.as_ref())?,
            SpeechTokenizerConfig::default_16khz(),
        )?;
        ensure!(!w.codebooks.is_empty(), "no codebooks loaded");
        Ok(Self::new(w, device))
    }

    pub fn config(&self) -> &SpeechTokenizerConfig {
        &self.w.config
    }

    pub fn encode(&self, pcm: &[f32], num_quantizers: usize) -> Result<Vec<Vec<u32>>> {
        let dim = self.w.config.dimension;
        let mut pre_c = self.enc_pre.borrow_mut();
        let pre = match pre_c.get_mut(&pcm.len()) {
            Some(g) => g,
            None => {
                let (g, p, oc, ot) = graph::build_enc_pre(&self.w.encoder, pcm.len())?;
                pre_c.entry(pcm.len()).or_insert(graph::StGraph::new(
                    self.device,
                    g,
                    p,
                    oc,
                    ot,
                    "pcm",
                ))
            }
        };
        let pre_out = pre.run(pcm)?;
        let t = pre.out_t;
        let fwd = layers(&self.w.encoder.lstm.fwd);
        let rev = layers(&self.w.encoder.lstm.rev);
        let post_in = lstm::bilstm(&fwd, &rev, &pre_out, dim, t);

        let mut post_c = self.enc_post.borrow_mut();
        let post = match post_c.get_mut(&t) {
            Some(g) => g,
            None => {
                let (g, p, oc) = graph::build_enc_post(&self.w.encoder, 2 * dim, t)?;
                post_c
                    .entry(t)
                    .or_insert(graph::StGraph::new(self.device, g, p, oc, t, "z"))
            }
        };
        let latent = post.run(&post_in)?;
        Ok(eager::rvq_encode(
            &self.w.codebooks,
            &latent,
            dim,
            t,
            num_quantizers,
        ))
    }

    pub fn decode(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        let dim = self.w.config.dimension;
        let z_q = eager::rvq_decode(&self.w.codebooks, codes, dim);
        let t = codes.first().map(|c| c.len()).unwrap_or(0);

        let mut pre_c = self.dec_pre.borrow_mut();
        let pre = match pre_c.get_mut(&t) {
            Some(g) => g,
            None => {
                let (g, p, oc) = graph::build_dec_pre(&self.w.decoder, t)?;
                pre_c
                    .entry(t)
                    .or_insert(graph::StGraph::new(self.device, g, p, oc, t, "zq"))
            }
        };
        let pre_out = pre.run(&z_q)?;
        let dl = layers(&self.w.decoder.lstm);
        let post_in = lstm::lstm(&dl, &pre_out, dim, t);

        let mut post_c = self.dec_post.borrow_mut();
        let post = match post_c.get_mut(&t) {
            Some(g) => g,
            None => {
                let (g, p, out_t) = graph::build_dec_post(&self.w.decoder, dim, t)?;
                post_c
                    .entry(t)
                    .or_insert(graph::StGraph::new(self.device, g, p, 1, out_t, "z"))
            }
        };
        post.run(&post_in)
    }
}

impl AudioCodec for SpeechTokenizerCodec {
    fn info(&self) -> CodecInfo {
        let cfg = &self.w.config;
        CodecInfo {
            sample_rate: cfg.sampling_rate,
            frame_rate: cfg.sampling_rate as f32 / cfg.hop_length() as f32,
            hop_length: cfg.hop_length(),
            channels: cfg.audio_channels,
            max_quantizers: self.w.codebooks.len(),
            codebook_size: cfg.codebook_size,
        }
    }

    fn device(&self) -> Device {
        self.device
    }

    fn encode_pcm(&self, pcm: &[f32], num_quantizers: Option<usize>) -> Result<RvqCodes> {
        let nq = num_quantizers
            .unwrap_or(self.w.codebooks.len())
            .min(self.w.codebooks.len());
        Ok(RvqCodes::from_quantizer_major(&self.encode(pcm, nq)?))
    }

    fn decode_codes(&self, codes: &RvqCodes) -> Result<Vec<f32>> {
        self.decode(&codes.to_quantizer_major())
    }
}
