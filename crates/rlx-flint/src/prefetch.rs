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

//! Guessing the next layer's experts, so the disk starts on them early.
//!
//! On a machine where the checkpoint does not fit in RAM, decode waits on the
//! disk for the experts it has not touched yet, and the disk sits idle while
//! the attention graphs and expert math run. Which experts layer `l + 1` fires
//! is known only after its router — but the router reads an RMS-normalized
//! residual stream that one layer barely moves. So as soon as layer `l` has
//! picked its experts, its router input `x_l = rms(h) · g_l` is rescaled to
//! `x_l / g_l · g_{l+1}` and run through layer `l + 1`'s router weights, and
//! the top 6 per row are handed to the OS as read-ahead
//! ([`crate::checkpoint::Checkpoint::advise_read`]).
//!
//! Measured over a 64-token generation (3 680 layer steps), this guess holds
//! 79.9% of the experts layer `l + 1` actually fires (88.0% for a top 8,
//! 93.9% for a top 12). Each wrong guess costs a read the disk did not need,
//! which is why only `top_k` are requested. The guess never changes the
//! output: it only warms the page cache that the real reads go through.
//!
//! Off by default ([`crate::RunnerOptions::prefetch_experts`]): an A/B on a
//! heavily shared machine could not separate its effect from load noise.

use anyhow::Result;

use crate::checkpoint::Checkpoint;
use crate::config::FlintConfig;
use crate::experts::ExpertWeights;
use crate::moe::EXPERT_PROJS;

/// Router weights and norm gain of one sparse layer, as f32.
struct LayerRouter {
    /// `[num_experts, hidden]`.
    gate: Vec<f32>,
    /// `post_attention_layernorm.weight`, the gain the router input carries.
    gain: Vec<f32>,
}

/// The read-ahead state across one step's layers.
pub struct ExpertPrefetcher {
    layers: Vec<Option<LayerRouter>>,
    /// The guess made for a layer: `(layer, experts)`.
    pending: Option<(usize, Vec<usize>)>,
    /// Experts requested ahead of their layer.
    pub issued: u64,
    /// Of those, how many the layer then fired.
    pub used: u64,
}

impl ExpertPrefetcher {
    /// Load every sparse layer's router weights (~2 MB each as f32).
    pub fn new(ckpt: &Checkpoint, cfg: &FlintConfig) -> Result<Self> {
        let layers = (0..cfg.num_hidden_layers)
            .map(|l| -> Result<Option<LayerRouter>> {
                if cfg.layer(l).dense {
                    return Ok(None);
                }
                Ok(Some(LayerRouter {
                    gate: ckpt
                        .load_f32(&format!("model.layers.{l}.mlp.gate.weight"))?
                        .0,
                    gain: ckpt
                        .load_f32(&format!("model.layers.{l}.post_attention_layernorm.weight"))?
                        .0,
                }))
            })
            .collect::<Result<_>>()?;
        Ok(ExpertPrefetcher {
            layers,
            pending: None,
            issued: 0,
            used: 0,
        })
    }

    /// Layer `layer` fired `ids` (`[rows, top_k]`, f32-encoded): score the
    /// guess made for it, if any.
    pub fn observe(&mut self, layer: usize, ids: &[f32]) {
        if let Some((l, guess)) = self.pending.take()
            && l == layer
        {
            self.used += guess
                .iter()
                .filter(|&&e| ids.iter().any(|&i| i as usize == e))
                .count() as u64;
        }
    }

    /// Layer `layer` routed `x` (`[rows, hidden]`, the first `rows` real):
    /// guess the next layer's experts and start reading them.
    pub fn after_route(
        &mut self,
        src: ExpertWeights<'_>,
        cfg: &FlintConfig,
        layer: usize,
        x: &[f32],
        rows: usize,
    ) {
        let next = layer + 1;
        let (Some(Some(cur)), Some(Some(nxt))) = (self.layers.get(layer), self.layers.get(next))
        else {
            return;
        };
        let (h, k) = (cfg.hidden_size, cfg.num_experts_per_tok);
        // `x / g_l · g_{l+1}`; a zero gain carries nothing to rescale.
        let ratio: Vec<f32> = cur
            .gain
            .iter()
            .zip(&nxt.gain)
            .map(|(&a, &b)| if a == 0.0 { 0.0 } else { b / a })
            .collect();
        let mut guess: Vec<usize> = Vec::new();
        let mut v = vec![0f32; h];
        for row in x.chunks_exact(h).take(rows) {
            for ((o, &xi), &r) in v.iter_mut().zip(row).zip(&ratio) {
                *o = xi * r;
            }
            // σ is monotonic, so ranking the logits ranks the scores.
            let mut scored: Vec<(f32, usize)> = nxt
                .gate
                .chunks_exact(h)
                .enumerate()
                .map(|(e, w)| (crate::experts::dot_f32(&v, w), e))
                .collect();
            let top = k.min(scored.len());
            scored.select_nth_unstable_by(top.saturating_sub(1), |a, b| b.0.total_cmp(&a.0));
            for &(_, e) in &scored[..top] {
                if !guess.contains(&e) {
                    guess.push(e);
                }
            }
        }
        for &e in &guess {
            for proj in EXPERT_PROJS {
                src.advise(next, e, proj);
            }
        }
        self.issued += guess.len() as u64;
        self.pending = Some((next, guess));
    }
}
