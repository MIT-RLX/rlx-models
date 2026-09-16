//! Reading a transported residual out as vocabulary tokens.
//!
//! This is the last step of the lens: take a residual at layer `l`, transport
//! it into the target-layer basis with `J_l`, and decode it with the model's
//! *own* unembedding. Using the model's head is the point — the transported
//! vector is constructed to live in the basis that head expects, which is
//! exactly what a plain logit lens gets wrong.

use anyhow::{Context as _, Result, ensure};
use rlx_runtime::{CompiledGraph, Device, Session};

use crate::fit::Jacobian;
use crate::model::{LensModel, UnembedGraph};

/// One decoded vocabulary entry.
#[derive(Debug, Clone, PartialEq)]
pub struct TopToken {
    pub token_id: u32,
    pub logit: f32,
    /// Rank over the full vocabulary, 0 = top.
    pub rank: usize,
}

/// A compiled unembedding, reusable across layers and positions.
pub struct Readout {
    unembed: CompiledGraph,
    residual_input: String,
    rows: usize,
    vocab: usize,
    d_model: usize,
}

impl Readout {
    pub fn new(model: &dyn LensModel, rows: usize, device: Device) -> Result<Self> {
        let UnembedGraph {
            graph,
            params,
            residual_input,
            rows,
            vocab,
        } = model.unembed(rows).map_err(anyhow::Error::from)?;
        let mut unembed = Session::new(device).compile(graph);
        for (name, data) in &params {
            unembed.set_param(name, data);
        }
        Ok(Self {
            unembed,
            residual_input,
            rows,
            vocab,
            d_model: model.d_model(),
        })
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Logits for `rows` residual rows, `[rows, vocab]`.
    pub fn logits(&mut self, residual: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            residual.len() == self.rows * self.d_model,
            "residual is {} elements, need {} ([{}, {}])",
            residual.len(),
            self.rows * self.d_model,
            self.rows,
            self.d_model
        );
        self.unembed
            .run(&[(self.residual_input.as_str(), residual)])
            .into_iter()
            .next()
            .context("unembed produced no output")
    }

    /// Transport `residual` through `jacobian`, then decode.
    ///
    /// Pass `None` to skip the transport — that is the plain logit lens, and is
    /// the baseline worth comparing against.
    pub fn read(
        &mut self,
        residual: &[f32],
        jacobian: Option<&Jacobian>,
        top_k: usize,
    ) -> Result<Vec<Vec<TopToken>>> {
        let transported = match jacobian {
            Some(j) => {
                ensure!(
                    j.d_model == self.d_model,
                    "Jacobian is {}-wide, model is {}-wide",
                    j.d_model,
                    self.d_model
                );
                j.transport(residual)
            }
            None => residual.to_vec(),
        };
        let logits = self.logits(&transported)?;
        Ok((0..self.rows)
            .map(|r| top_tokens(&logits[r * self.vocab..(r + 1) * self.vocab], top_k))
            .collect())
    }
}

/// The `top_k` highest-scoring vocabulary entries, best first.
pub fn top_tokens(logits: &[f32], top_k: usize) -> Vec<TopToken> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    // Descending by logit; NaN sorts last so a broken readout is visible rather
    // than silently reordering.
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.into_iter()
        .take(top_k)
        .enumerate()
        .map(|(rank, token_id)| TopToken {
            token_id: token_id as u32,
            logit: logits[token_id],
            rank,
        })
        .collect()
}

/// Full-vocabulary rank of `token_id`, 0 = top.
pub fn rank_of(logits: &[f32], token_id: u32) -> usize {
    let target = logits[token_id as usize];
    logits.iter().filter(|&&l| l > target).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_tokens_are_ordered_best_first() {
        let logits = [0.1, 5.0, -2.0, 3.0];
        let top = top_tokens(&logits, 3);
        assert_eq!(
            top.iter().map(|t| t.token_id).collect::<Vec<_>>(),
            vec![1, 3, 0]
        );
        assert_eq!(
            top.iter().map(|t| t.rank).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(top[0].logit, 5.0);
    }

    #[test]
    fn top_k_larger_than_the_vocab_is_clamped() {
        assert_eq!(top_tokens(&[1.0, 2.0], 10).len(), 2);
    }

    #[test]
    fn rank_counts_strictly_greater_logits() {
        let logits = [0.1, 5.0, -2.0, 3.0];
        assert_eq!(rank_of(&logits, 1), 0);
        assert_eq!(rank_of(&logits, 3), 1);
        assert_eq!(rank_of(&logits, 0), 2);
        assert_eq!(rank_of(&logits, 2), 3);
    }
}
