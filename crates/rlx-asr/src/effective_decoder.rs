// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! Native step-1 / history-free token AED using fit effective maps.
//!
//! ```text
//! h = embed[token] · Ah_tok
//! logprob = log_softmax(h · W_outᵀ + b_out)
//! ```
//! Optional encoder-conditioned path (when `We`/`Wc`/`b_h` are present):
//! `h = emb@We + mean(encoder_cache)@Wc + b_h`.

use crate::spec::{DECODER_DIM, VOCAB};
use crate::weights::read_f32_bin;
use anyhow::{bail, Context, Result};
use std::path::Path;

fn matvec_rowmajor(rows: usize, cols: usize, w: &[f32], x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    for r in 0..rows {
        let mut s = 0.0f32;
        let row = &w[r * cols..(r + 1) * cols];
        for c in 0..cols {
            s += row[c] * x[c];
        }
        out[r] = s;
    }
}

fn log_softmax_inplace(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v = (*v * inv).ln();
    }
}

/// Effective first-token decoder (linear embed→hidden→logits).
pub struct EffectiveStep1 {
    pub embed: Vec<f32>,
    pub ah_tok: Vec<f32>,
    pub w_out: Vec<f32>,
    pub b_out: Vec<f32>,
    pub ak: Option<Vec<f32>>,
    pub av: Option<Vec<f32>>,
    pub we: Option<Vec<f32>>,
    pub wc: Option<Vec<f32>>,
    pub b_h: Option<Vec<f32>>,
}

impl EffectiveStep1 {
    /// Load from `weights/asr/decoder/` bins, or from sibling `model.gguf` when present.
    pub fn load(dir: &Path) -> Result<Self> {
        if let Some(root) = dir.parent() {
            if let Some(gguf) = crate::gguf_io::resolve_gguf_path(root) {
                if let Ok(g) = crate::gguf_io::AsrGguf::open(&gguf) {
                    if g.has("decoder.embed") {
                        return g.load_effective_step1();
                    }
                }
            }
        }
        Self::load_bins(dir)
    }

    /// Load from loose `decoder/*.bin` files.
    pub fn load_bins(dir: &Path) -> Result<Self> {
        let embed = read_f32_bin(&dir.join("embed.bin")).context("embed.bin")?;
        let ah = read_f32_bin(&dir.join("effective_Ah_tok.bin"))
            .or_else(|_| read_f32_bin(&dir.join("effective_Ah.bin")))
            .context("effective_Ah_tok.bin")?;
        let w_out = read_f32_bin(&dir.join("W_out.bin")).context("W_out.bin")?;
        let b_out = read_f32_bin(&dir.join("b_out.bin")).context("b_out.bin")?;
        if embed.len() != VOCAB * DECODER_DIM {
            bail!("embed len {} want {}", embed.len(), VOCAB * DECODER_DIM);
        }
        if ah.len() != DECODER_DIM * DECODER_DIM {
            bail!("Ah len {} want {}", ah.len(), DECODER_DIM * DECODER_DIM);
        }
        if w_out.len() != VOCAB * DECODER_DIM {
            bail!("W_out len {}", w_out.len());
        }
        if b_out.len() != VOCAB {
            bail!("b_out len {}", b_out.len());
        }
        Ok(Self {
            embed,
            ah_tok: ah,
            w_out,
            b_out,
            ak: read_f32_bin(&dir.join("effective_Ak.bin")).ok(),
            av: read_f32_bin(&dir.join("effective_Av.bin")).ok(),
            we: read_f32_bin(&dir.join("effective_We.bin")).ok(),
            wc: read_f32_bin(&dir.join("effective_Wc.bin")).ok(),
            b_h: read_f32_bin(&dir.join("effective_bh.bin")).ok(),
        })
    }

    fn embed_row(&self, token: u32) -> &[f32] {
        let t = token as usize;
        &self.embed[t * DECODER_DIM..(t + 1) * DECODER_DIM]
    }

    fn x_mat(x: &[f32], m: &[f32], out_dim: usize, y: &mut [f32]) {
        let in_dim = x.len();
        debug_assert_eq!(m.len(), in_dim * out_dim);
        for c in 0..out_dim {
            let mut s = 0.0f32;
            for k in 0..in_dim {
                s += x[k] * m[k * out_dim + c];
            }
            y[c] = s;
        }
    }

    fn logits_from_h(&self, h: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0f32; VOCAB];
        matvec_rowmajor(VOCAB, DECODER_DIM, &self.w_out, h, &mut logits);
        for (l, b) in logits.iter_mut().zip(self.b_out.iter()) {
            *l += *b;
        }
        log_softmax_inplace(&mut logits);
        logits
    }

    pub fn logprob(&self, token: u32) -> Result<Vec<f32>> {
        if token as usize >= VOCAB {
            bail!("token {token} out of range");
        }
        let mut h = vec![0.0f32; DECODER_DIM];
        Self::x_mat(self.embed_row(token), &self.ah_tok, DECODER_DIM, &mut h);
        Ok(self.logits_from_h(&h))
    }

    /// Encoder-conditioned path: `h = emb@We + mean(enc)@Wc + b_h`.
    pub fn logprob_with_encoder(&self, token: u32, encoder_cache: &[f32]) -> Result<Vec<f32>> {
        let (Some(we), Some(wc), Some(bh)) = (&self.we, &self.wc, &self.b_h) else {
            return self.logprob(token);
        };
        if token as usize >= VOCAB {
            bail!("token {token} out of range");
        }
        if encoder_cache.len() < DECODER_DIM {
            bail!("encoder_cache too short");
        }
        let t_frames = encoder_cache.len() / DECODER_DIM;
        let mut enc_mean = vec![0.0f32; DECODER_DIM];
        for t in 0..t_frames {
            let row = &encoder_cache[t * DECODER_DIM..(t + 1) * DECODER_DIM];
            for d in 0..DECODER_DIM {
                enc_mean[d] += row[d];
            }
        }
        let inv = 1.0 / t_frames as f32;
        for v in &mut enc_mean {
            *v *= inv;
        }
        let mut h_tok = vec![0.0f32; DECODER_DIM];
        let mut h_enc = vec![0.0f32; DECODER_DIM];
        Self::x_mat(self.embed_row(token), we, DECODER_DIM, &mut h_tok);
        Self::x_mat(&enc_mean, wc, DECODER_DIM, &mut h_enc);
        let mut h = vec![0.0f32; DECODER_DIM];
        for i in 0..DECODER_DIM {
            h[i] = h_tok[i] + h_enc[i] + bh[i];
        }
        Ok(self.logits_from_h(&h))
    }

    pub fn self_k(&self, token: u32) -> Result<Option<Vec<f32>>> {
        let Some(ak) = self.ak.as_ref() else {
            return Ok(None);
        };
        let mut k = vec![0.0f32; DECODER_DIM];
        Self::x_mat(self.embed_row(token), ak, DECODER_DIM, &mut k);
        Ok(Some(k))
    }

    pub fn self_v(&self, token: u32) -> Result<Option<Vec<f32>>> {
        let Some(av) = self.av.as_ref() else {
            return Ok(None);
        };
        let mut v = vec![0.0f32; DECODER_DIM];
        Self::x_mat(self.embed_row(token), av, DECODER_DIM, &mut v);
        Ok(Some(v))
    }
}
