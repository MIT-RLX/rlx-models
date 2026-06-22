//! Per-step DepFormer wrapper for Kyutai TTS.
//!
//! Standard DepFormer (Moshi 7B) shares all weights across codebook slices.
//! Kyutai TTS sets `depformer_weights_per_step = true` with a
//! `depformer_weights_per_step_schedule` of length `n_q` that maps each
//! codebook slot to a *head index*. The published 1.6B schedule collapses
//! 32 codebooks to 11 heads (codebooks 0..8 each own a head; 8..16 share head 8;
//! 16..24 share head 9; 24..32 share head 10) — see
//! [`crate::config::DEPFORMER_WEIGHTS_PER_STEP_SCHEDULE`].
//!
//! This module provides the bookkeeping side: given a codebook index, pick the
//! correct weights / projections / norms from per-head tables. The actual
//! mini-transformer layer is reused from [`crate::nn`].

use crate::config::DepFormerConfig;
use crate::low_rank_embedding::LowRankEmbedding;
use crate::nn::{Embedding, linear, rms_norm, swiglu_mlp};
use anyhow::{Result, bail};
use ndarray::{Array1, Array2};

/// Per-head weight bundle for a single DepFormer "head".
///
/// One head services one or more codebooks (per the schedule). Each head owns:
/// - input/output projections (`d_model → depformer.dim`, `depformer.dim → card`)
/// - one or more transformer layers (here we expose the per-head FFN tables;
///   attention weights are typically shared across heads in Kyutai TTS).
#[derive(Debug, Clone)]
pub struct DepFormerHead {
    /// Project the temporal hidden state into the depth-transformer dim.
    /// `[depformer.dim, d_model_temporal]`
    pub in_proj: Array2<f32>,
    /// Output logits projection: `[card, depformer.dim]`.
    pub out_proj: Array2<f32>,
    /// FFN weights for the (single) transformer layer assigned to this head.
    /// `linear_in: [2·hidden, depformer.dim]`, `linear_out: [depformer.dim, hidden]`.
    pub ffn_in: Array2<f32>,
    pub ffn_out: Array2<f32>,
    /// RMSNorm scale before the FFN. `[depformer.dim]`.
    pub norm_alpha: Array1<f32>,
}

/// Per-step DepFormer.
///
/// Codebook → head mapping is `schedule[codebook] = head_id`. Codebook input
/// embeddings are factorised (`low_rank_embeddings`) and are one per codebook
/// (not per head).
#[derive(Debug, Clone)]
pub struct DepFormer {
    pub cfg: DepFormerConfig,
    /// Per-codebook low-rank embeddings (length `n_q`).
    pub embeddings: Vec<LowRankEmbedding>,
    /// Per-codebook standard embeddings (used when `low_rank_embeddings = 0`).
    /// Mutually exclusive with [`Self::embeddings`].
    pub dense_embeddings: Vec<Embedding>,
    /// Per-head weights (length = unique heads in the schedule).
    pub heads: Vec<DepFormerHead>,
    /// Cached schedule (mirror of [`DepFormerConfig::weights_per_step_schedule`]).
    pub schedule: Vec<usize>,
}

impl DepFormer {
    /// Resolve the head id for a given codebook slot.
    pub fn head_for(&self, codebook: usize) -> Result<usize> {
        if codebook >= self.schedule.len() {
            bail!(
                "codebook {codebook} out of range (schedule len {})",
                self.schedule.len()
            );
        }
        let head = self.schedule[codebook];
        if head >= self.heads.len() {
            bail!(
                "schedule[{codebook}] = head {head} but only {} heads loaded",
                self.heads.len()
            );
        }
        Ok(head)
    }

    /// Number of distinct heads in this configuration.
    pub fn num_heads_unique(&self) -> usize {
        self.heads.len()
    }

    /// Embed one audio token from a specific codebook (low-rank if available,
    /// dense otherwise).
    pub fn embed(&self, codebook: usize, token: u32) -> Result<Array1<f32>> {
        if codebook < self.embeddings.len() {
            return Ok(self.embeddings[codebook].forward_one(token));
        }
        if codebook < self.dense_embeddings.len() {
            return Ok(self.dense_embeddings[codebook].forward_one(token));
        }
        bail!("no embedding loaded for codebook {codebook}")
    }

    /// One depth step: project the temporal hidden into the depth dim, run the
    /// per-head FFN, project to logits, return `[card]`.
    ///
    /// This is the minimal pipeline shape; the full DepFormer also runs causal
    /// self-attention over previously generated codebooks. The attention path
    /// is intentionally not encoded here so the caller (eventual generation
    /// loop) can compose it with the shared attention weights.
    pub fn forward_step(
        &self,
        codebook: usize,
        hidden_temporal: &Array1<f32>,
    ) -> Result<Array1<f32>> {
        let head_id = self.head_for(codebook)?;
        let head = &self.heads[head_id];
        let h_view = hidden_temporal.view().insert_axis(ndarray::Axis(0));
        let h_in = linear(h_view, &head.in_proj); // [1, depformer.dim]
        let h_norm = rms_norm(h_in.view(), &head.norm_alpha);
        let mlp = swiglu_mlp(h_norm.view(), &head.ffn_in, &head.ffn_out);
        let post = &h_in + &mlp;
        let logits = linear(post.view(), &head.out_proj);
        Ok(logits.row(0).to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DepFormerConfig, PositionalEmbedding};
    use ndarray::Array;

    fn dep_config_minimal(n_q: usize) -> DepFormerConfig {
        let schedule = (0..n_q).map(|i| i % 3).collect::<Vec<_>>();
        DepFormerConfig {
            dim: 4,
            num_heads: 1,
            num_layers: 1,
            dim_feedforward: 8,
            multi_linear: true,
            positional_embedding: PositionalEmbedding::None,
            weights_per_step: true,
            weights_per_step_schedule: schedule,
            low_rank_embeddings: 2,
        }
    }

    fn dummy_head(d_temporal: usize, d_dep: usize, hidden: usize, card: usize) -> DepFormerHead {
        DepFormerHead {
            in_proj: Array::ones((d_dep, d_temporal)),
            out_proj: Array::ones((card, d_dep)),
            ffn_in: Array::ones((2 * hidden, d_dep)),
            ffn_out: Array::ones((d_dep, hidden)),
            norm_alpha: Array::ones(d_dep),
        }
    }

    fn dummy_low_rank(card: usize, rank: usize, dim: usize) -> LowRankEmbedding {
        let a = Array::ones((card, rank));
        let b = Array::ones((rank, dim));
        LowRankEmbedding::new(a, b)
    }

    fn dep_fixture(n_q: usize) -> DepFormer {
        let cfg = dep_config_minimal(n_q);
        let card = 4;
        let heads = (0..3)
            .map(|_| dummy_head(6, cfg.dim, cfg.dim_feedforward / 2, card))
            .collect();
        let embeddings = (0..n_q)
            .map(|_| dummy_low_rank(card, cfg.low_rank_embeddings, cfg.dim))
            .collect();
        DepFormer {
            cfg: cfg.clone(),
            embeddings,
            dense_embeddings: vec![],
            heads,
            schedule: cfg.weights_per_step_schedule.clone(),
        }
    }

    #[test]
    fn head_for_resolves_via_schedule() {
        let df = dep_fixture(7);
        for cb in 0..7 {
            assert_eq!(df.head_for(cb).unwrap(), cb % 3);
        }
    }

    #[test]
    fn head_for_rejects_out_of_range_codebook() {
        let df = dep_fixture(4);
        assert!(df.head_for(99).is_err());
    }

    #[test]
    fn forward_step_returns_card_logits() {
        let df = dep_fixture(5);
        let hidden = Array1::<f32>::from_vec(vec![0.1; 6]);
        let logits = df.forward_step(0, &hidden).unwrap();
        assert_eq!(logits.len(), 4);
    }

    #[test]
    fn embed_uses_low_rank_when_available() {
        let df = dep_fixture(3);
        let e = df.embed(0, 1).unwrap();
        // ones × ones with rank=2, dim=4 → all entries = rank = 2.
        for v in e.iter() {
            assert!((v - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn num_heads_unique_matches_loaded_heads() {
        let df = dep_fixture(8);
        assert_eq!(df.num_heads_unique(), 3);
    }
}
