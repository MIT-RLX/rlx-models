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

//! Mamba1 language-model wrapper: token embedding → N residual
//! `RMSNorm + Mamba1Block` layers → final RMSNorm → lm_head (tied or
//! untied). Algorithmically mirrors `burn_mamba::mamba1::Mamba1Network`.

use crate::block::Mamba1Block;
use crate::cache::Mamba1Caches;
use crate::config::Mamba1NetworkConfig;
use anyhow::{Result, ensure};
use rlx_cpu::blas;

/// One pre-norm Mamba layer: `x = x + Block(RMSNorm(x))`.
#[derive(Debug, Clone)]
pub struct Mamba1Layer {
    pub norm_gamma: Vec<f32>, // [d_model]
    pub block: Mamba1Block,
}

#[derive(Debug, Clone)]
pub struct Mamba1Network {
    pub cfg: Mamba1NetworkConfig,
    /// `[padded_vocab, d_model]`.
    pub embedding: Vec<f32>,
    pub layers: Vec<Mamba1Layer>,
    /// `[d_model]`.
    pub norm_f_gamma: Vec<f32>,
    /// `[d_model, padded_vocab]`. If `None`, lm_head is tied to the
    /// embedding (computed as `x @ embedding^T`).
    pub lm_head: Option<Vec<f32>>,
}

impl Mamba1Network {
    /// Convenience: build a randomly-initialized network for benches/tests.
    pub fn random_for_bench(cfg: Mamba1NetworkConfig, seed: u64) -> Self {
        let m = cfg.mamba_block.d_model;
        let pv = cfg.padded_vocab_size();
        let mut rng_seed = seed;
        let next = |s: &mut u64, scale: f32, len: usize| -> Vec<f32> {
            // mini SplitMix64 inline
            (0..len)
                .map(|_| {
                    *s = s.wrapping_add(0x9E3779B97F4A7C15);
                    let mut z = *s;
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                    z ^= z >> 31;
                    let u = ((z >> 40) as f32) / ((1u32 << 24) as f32);
                    (u * 2.0 - 1.0) * scale
                })
                .collect()
        };

        let embedding = next(&mut rng_seed, 1.0 / (m as f32).sqrt(), pv * m);
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            layers.push(Mamba1Layer {
                norm_gamma: vec![1.0; m],
                block: Mamba1Block::random_for_bench(
                    cfg.mamba_block.clone(),
                    seed.wrapping_add(0x1000 + i as u64),
                ),
            });
        }
        let norm_f_gamma = vec![1.0; m];
        let lm_head = if cfg.tied_lm_head {
            None
        } else {
            // [d_model, padded_vocab]
            Some(next(&mut rng_seed, 1.0 / (m as f32).sqrt(), m * pv))
        };
        Self {
            cfg,
            embedding,
            layers,
            norm_f_gamma,
            lm_head,
        }
    }

    /// `tokens` is `[batch, seq]`. Output is `[batch, seq, padded_vocab]`.
    pub fn forward(&self, tokens: &[u32], batch: usize, seq: usize) -> Result<Vec<f32>> {
        ensure!(tokens.len() == batch * seq, "tokens shape");
        let m = self.cfg.mamba_block.d_model;
        let pv = self.cfg.padded_vocab_size();

        // Embedding gather → x [batch*seq, d_model]
        let bs = batch * seq;
        let mut x = vec![0.0; bs * m];
        for i in 0..bs {
            let tid = tokens[i] as usize;
            debug_assert!(tid < pv, "token id {tid} >= vocab {pv}");
            let src = &self.embedding[tid * m..(tid + 1) * m];
            x[i * m..(i + 1) * m].copy_from_slice(src);
        }

        // Residual stack
        for layer in &self.layers {
            let normed = rms_norm(&x, m, &layer.norm_gamma, 1e-5);
            let block_out = layer.block.forward(&normed, batch, seq)?;
            for i in 0..x.len() {
                x[i] += block_out[i];
            }
        }
        let x = rms_norm(&x, m, &self.norm_f_gamma, 1e-5);

        // lm_head
        let mut logits = vec![0.0; bs * pv];
        let bias = vec![0.0; pv];
        match &self.lm_head {
            Some(w) => {
                blas::sgemm(&x, w, &mut logits, bs, m, pv);
                let _ = &bias;
            }
            None => {
                // Tied: logits = x @ embedding^T. embedding is [pv, m].
                blas::sgemm_bt(&x, &self.embedding, &mut logits, bs, m, pv, 1.0);
            }
        }
        Ok(logits)
    }

    /// Decode one token per batch row. `tokens` is `[batch]`.
    /// Output is `[batch, padded_vocab]`.
    pub fn step(
        &self,
        tokens: &[u32],
        batch: usize,
        caches: &mut Mamba1Caches,
    ) -> Result<Vec<f32>> {
        ensure!(tokens.len() == batch, "tokens shape");
        ensure!(caches.caches.len() == self.cfg.n_layer, "cache n_layer");
        let m = self.cfg.mamba_block.d_model;
        let pv = self.cfg.padded_vocab_size();

        let mut x = vec![0.0; batch * m];
        for i in 0..batch {
            let tid = tokens[i] as usize;
            let src = &self.embedding[tid * m..(tid + 1) * m];
            x[i * m..(i + 1) * m].copy_from_slice(src);
        }
        for (layer, cache) in self.layers.iter().zip(caches.caches.iter_mut()) {
            let normed = rms_norm(&x, m, &layer.norm_gamma, 1e-5);
            let block_out = layer.block.step(&normed, batch, cache)?;
            for i in 0..x.len() {
                x[i] += block_out[i];
            }
        }
        let x = rms_norm(&x, m, &self.norm_f_gamma, 1e-5);

        let mut logits = vec![0.0; batch * pv];
        let bias = vec![0.0; pv];
        match &self.lm_head {
            Some(w) => {
                blas::sgemm(&x, w, &mut logits, batch, m, pv);
                let _ = &bias;
            }
            None => blas::sgemm_bt(&x, &self.embedding, &mut logits, batch, m, pv, 1.0),
        }
        Ok(logits)
    }
}

fn rms_norm(x: &[f32], dim: usize, gamma: &[f32], eps: f32) -> Vec<f32> {
    let rows = x.len() / dim;
    let mut out = vec![0.0; x.len()];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for c in 0..dim {
            out[r * dim + c] = row[c] * inv * gamma[c];
        }
    }
    out
}
