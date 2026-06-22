//! Factorized codebook embedding (`depformer_low_rank_embeddings = 128`).
//!
//! The Kyutai TTS DepFormer represents each codebook's `[card=2048, dim=1024]`
//! embedding matrix `E` as two factors `E ≈ A · B`, where:
//!
//! - `A` has shape `[card, rank=128]`
//! - `B` has shape `[rank=128, dim=1024]`
//!
//! For a single audio token `id`, the embedding is `A[id, :] · B` — one row
//! pickup + one small matmul, ~16× smaller than dense storage.

use crate::nn::linear;
use ndarray::{Array1, Array2, Axis};

/// Per-codebook low-rank embedding.
#[derive(Debug, Clone)]
pub struct LowRankEmbedding {
    /// `A`: `[card, rank]` — token-indexed row.
    pub a: Array2<f32>,
    /// `B`: `[rank, dim]` — projects rank-vector up to model dim.
    pub b: Array2<f32>,
}

impl LowRankEmbedding {
    /// Construct from factors. Panics on rank mismatch.
    pub fn new(a: Array2<f32>, b: Array2<f32>) -> Self {
        assert_eq!(
            a.ncols(),
            b.nrows(),
            "rank mismatch: A.cols ({}) != B.rows ({})",
            a.ncols(),
            b.nrows()
        );
        Self { a, b }
    }

    /// Vocabulary size (`card`).
    pub fn vocab_size(&self) -> usize {
        self.a.nrows()
    }

    /// Low-rank dimensionality (`depformer_low_rank_embeddings`).
    pub fn rank(&self) -> usize {
        self.a.ncols()
    }

    /// Output embedding dim.
    pub fn dim(&self) -> usize {
        self.b.ncols()
    }

    /// Look up one token id → `[dim]`.
    pub fn forward_one(&self, id: u32) -> Array1<f32> {
        let row = self.a.row(id as usize).to_owned();
        // [rank] · [rank, dim] = [dim]
        row.dot(&self.b)
    }

    /// Look up a batch of token ids → `[t, dim]`.
    pub fn forward(&self, ids: &[u32]) -> Array2<f32> {
        let rank = self.rank();
        let mut a_rows = Array2::<f32>::zeros((ids.len(), rank));
        for (i, &id) in ids.iter().enumerate() {
            let row = self.a.row(id as usize);
            for j in 0..rank {
                a_rows[[i, j]] = row[j];
            }
        }
        linear(a_rows.view(), &self.b.t().to_owned())
    }

    /// Materialise the full embedding table `E = A · B` for parity checks.
    /// Allocates `[vocab × dim]` floats — only use in tests / debug paths.
    pub fn materialise(&self) -> Array2<f32> {
        self.a.dot(&self.b)
    }

    /// Average L2 norm of the materialised embedding rows (debug stat).
    pub fn avg_norm(&self) -> f32 {
        let mat = self.materialise();
        let v = mat.vocab_size();
        let mut sum = 0.0;
        for r in mat.axis_iter(Axis(0)) {
            sum += r.iter().map(|x| x * x).sum::<f32>().sqrt();
        }
        sum / v as f32
    }
}

// Convenience: allow [`Array2`] to expose `.vocab_size()` for the avg_norm helper.
trait VocabSize {
    fn vocab_size(&self) -> usize;
}
impl VocabSize for Array2<f32> {
    fn vocab_size(&self) -> usize {
        self.nrows()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn forward_one_matches_materialised_lookup() {
        let a = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]; // [3, 2]
        let b = array![[0.5, -1.0, 2.0], [1.0, 0.0, 3.0]]; // [2, 3]
        let lre = LowRankEmbedding::new(a.clone(), b.clone());
        assert_eq!(lre.vocab_size(), 3);
        assert_eq!(lre.rank(), 2);
        assert_eq!(lre.dim(), 3);

        let dense = lre.materialise();
        for id in 0..3 {
            let lo = lre.forward_one(id as u32);
            for k in 0..3 {
                assert!(
                    (lo[k] - dense[[id, k]]).abs() < 1e-6,
                    "id={id} k={k}: low_rank={} dense={}",
                    lo[k],
                    dense[[id, k]]
                );
            }
        }
    }

    #[test]
    fn batched_forward_matches_per_token_forward_one() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let b = array![[1.0, 0.0], [0.0, 1.0]];
        let lre = LowRankEmbedding::new(a, b);
        let ids = vec![0u32, 2, 1];
        let batched = lre.forward(&ids);
        for (i, &id) in ids.iter().enumerate() {
            let single = lre.forward_one(id);
            for k in 0..lre.dim() {
                assert!((batched[[i, k]] - single[k]).abs() < 1e-6);
            }
        }
    }

    #[test]
    #[should_panic(expected = "rank mismatch")]
    fn rank_mismatch_panics() {
        let a = Array2::<f32>::zeros((4, 3));
        let b = Array2::<f32>::zeros((2, 5));
        let _ = LowRankEmbedding::new(a, b);
    }
}
