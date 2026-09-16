//! The Jacobian estimator's index arithmetic.
//!
//! `J_l = E[∂h_final/∂h_l]` is estimated one *block of output dimensions* at a
//! time. The prompt is replicated `dim_batch` times along the batch axis, and
//! batch element `b` carries a one-hot cotangent at output dimension
//! `dim_start + b` — so a single VJP yields `dim_batch` rows of `J_l`, and
//! `ceil(d_model / dim_batch)` of them cover the matrix.
//!
//! The one-hot is set at **every valid target position at once**. The gradient
//! at source position `p` is therefore `Σ_{p' ≥ p} ∂h_final[p']/∂h_l[p]` — a sum
//! over current-and-future targets — which is then averaged over source
//! positions. This is the reduction the paper uses; a strict per-position
//! estimator gives a slightly different `J_l` and also works as a lens.

use anyhow::{Result, ensure};

/// Positions before this index are excluded from the average: early positions
/// act as attention sinks and have atypical residual statistics.
pub const SKIP_FIRST_N_POSITIONS: usize = 16;

/// Shape of the residual stream a lens tap reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualShape {
    pub batch: usize,
    pub seq: usize,
    pub d_model: usize,
}

impl ResidualShape {
    pub fn new(batch: usize, seq: usize, d_model: usize) -> Self {
        Self {
            batch,
            seq,
            d_model,
        }
    }

    pub fn elements(&self) -> usize {
        self.batch * self.seq * self.d_model
    }
}

/// Sequence positions included in the Jacobian average.
///
/// Drops the first `skip_first` (attention sinks) and the last (no next-token
/// target), matching the reference implementation.
pub fn valid_positions(seq: usize, skip_first: usize) -> Result<Vec<usize>> {
    ensure!(
        seq > skip_first + 1,
        "prompt too short: seq_len={seq}, need > {} tokens",
        skip_first + 1
    );
    Ok((skip_first..seq - 1).collect())
}

/// Write the one-hot cotangent for the pass covering output dimensions
/// `dim_start .. dim_start + n_dims`.
///
/// Batch element `b` gets `1.0` at dimension `dim_start + b`, at every position
/// in `positions`. Batch elements at or beyond `n_dims` stay zero — they only
/// occur in the final ragged pass, where their gradients are discarded.
pub fn fill_onehot_cotangent(
    buf: &mut [f32],
    shape: ResidualShape,
    dim_start: usize,
    n_dims: usize,
    positions: &[usize],
) -> Result<()> {
    ensure!(
        buf.len() == shape.elements(),
        "cotangent buffer is {} elements, need {}",
        buf.len(),
        shape.elements()
    );
    ensure!(
        n_dims <= shape.batch,
        "n_dims={n_dims} exceeds batch={} — one dimension per batch element",
        shape.batch
    );
    ensure!(
        dim_start + n_dims <= shape.d_model,
        "dims {dim_start}..{} exceed d_model={}",
        dim_start + n_dims,
        shape.d_model
    );

    buf.fill(0.0);
    for b in 0..n_dims {
        let dim = dim_start + b;
        for &pos in positions {
            buf[b * shape.seq * shape.d_model + pos * shape.d_model + dim] = 1.0;
        }
    }
    Ok(())
}

/// Reduce one VJP result into rows `dim_start .. dim_start + n_dims` of `J`.
///
/// `grad` is `∂h_final/∂h_l` for this pass, shaped `[batch, seq, d_model]`; row
/// `dim_start + b` of `J` is `grad[b]` averaged over the valid source positions.
/// `jacobian` is row-major `[d_model, d_model]` with `J[i][j] =
/// ∂h_final_i/∂h_l_j`, so a readout is `J·h` (i.e. `h @ Jᵀ`).
pub fn write_rows(
    grad: &[f32],
    shape: ResidualShape,
    dim_start: usize,
    n_dims: usize,
    positions: &[usize],
    jacobian: &mut [f32],
) -> Result<()> {
    ensure!(
        grad.len() == shape.elements(),
        "gradient is {} elements, need {}",
        grad.len(),
        shape.elements()
    );
    ensure!(
        jacobian.len() == shape.d_model * shape.d_model,
        "jacobian is {} elements, need {}",
        jacobian.len(),
        shape.d_model * shape.d_model
    );
    ensure!(!positions.is_empty(), "no valid positions to average over");

    let scale = 1.0 / positions.len() as f32;
    for b in 0..n_dims {
        let row = &mut jacobian[(dim_start + b) * shape.d_model..][..shape.d_model];
        row.fill(0.0);
        for &pos in positions {
            let src = &grad[b * shape.seq * shape.d_model + pos * shape.d_model..][..shape.d_model];
            for (dst, s) in row.iter_mut().zip(src) {
                *dst += *s;
            }
        }
        for v in row.iter_mut() {
            *v *= scale;
        }
    }
    Ok(())
}

/// `‖J‖_F / √d` — the per-prompt magnitude diagnostic. Heavy-tailed outliers
/// across a corpus show up here before they show up in the fitted lens.
pub fn scaled_frobenius_norm(jacobian: &[f32], d_model: usize) -> f32 {
    let sum_sq: f64 = jacobian.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    (sum_sq.sqrt() / (d_model as f64).sqrt()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_positions_drops_sinks_and_last() {
        assert_eq!(valid_positions(6, 2).unwrap(), vec![2, 3, 4]);
        // seq must exceed skip_first + 1, else nothing is left to average.
        assert!(valid_positions(3, 2).is_err());
        assert!(valid_positions(4, 2).is_ok());
    }

    #[test]
    fn cotangent_is_one_hot_per_batch_element() {
        let shape = ResidualShape::new(2, 3, 4);
        let mut buf = vec![9.0; shape.elements()];
        fill_onehot_cotangent(&mut buf, shape, 1, 2, &[0, 2]).unwrap();

        // Exactly one 1.0 per (batch, position) pair, at dim_start + b.
        assert_eq!(buf.iter().filter(|v| **v == 1.0).count(), 4);
        assert_eq!(buf.iter().filter(|v| **v == 0.0).count(), 20);
        let at = |b: usize, p: usize, d: usize| buf[b * 12 + p * 4 + d];
        assert_eq!(at(0, 0, 1), 1.0);
        assert_eq!(at(0, 2, 1), 1.0);
        assert_eq!(at(1, 0, 2), 1.0);
        assert_eq!(at(1, 2, 2), 1.0);
        // Position 1 was not requested.
        assert_eq!(at(0, 1, 1), 0.0);
    }

    #[test]
    fn ragged_final_pass_leaves_spare_batch_elements_zero() {
        let shape = ResidualShape::new(4, 2, 3);
        let mut buf = vec![0.0; shape.elements()];
        // d_model = 3, dim_start = 2 ⇒ only one dimension left to cover.
        fill_onehot_cotangent(&mut buf, shape, 2, 1, &[0]).unwrap();
        assert_eq!(buf.iter().filter(|v| **v == 1.0).count(), 1);
        assert_eq!(buf[2], 1.0);
    }

    #[test]
    fn cotangent_rejects_overrun() {
        let shape = ResidualShape::new(2, 3, 4);
        let mut buf = vec![0.0; shape.elements()];
        // 3 dimensions into a batch of 2.
        assert!(fill_onehot_cotangent(&mut buf, shape, 0, 3, &[0]).is_err());
        // dims 3..5 out of d_model = 4.
        assert!(fill_onehot_cotangent(&mut buf, shape, 3, 2, &[0]).is_err());
    }

    #[test]
    fn write_rows_averages_over_positions_only() {
        let shape = ResidualShape::new(2, 3, 2);
        // grad[b, pos, :] = [b + pos, 10·(b + pos)]
        let mut grad = vec![0.0f32; shape.elements()];
        for b in 0..2 {
            for p in 0..3 {
                let v = (b + p) as f32;
                grad[b * 6 + p * 2] = v;
                grad[b * 6 + p * 2 + 1] = 10.0 * v;
            }
        }
        let mut j = vec![-1.0f32; 4];
        // Average over positions 0 and 2 only — position 1 must not contribute.
        write_rows(&grad, shape, 0, 2, &[0, 2], &mut j).unwrap();

        // b=0: mean of (0, 2) = 1 ⇒ row [1, 10]; b=1: mean of (1, 3) = 2 ⇒ [2, 20].
        assert_eq!(j, vec![1.0, 10.0, 2.0, 20.0]);
    }

    #[test]
    fn write_rows_overwrites_rather_than_accumulates() {
        let shape = ResidualShape::new(1, 2, 2);
        let grad = vec![1.0, 2.0, 1.0, 2.0];
        let mut j = vec![100.0; 4];
        write_rows(&grad, shape, 0, 1, &[0], &mut j).unwrap();
        // Row 0 replaced, row 1 untouched — each row is written exactly once
        // per prompt, and cross-prompt accumulation happens a level up.
        assert_eq!(&j[..2], &[1.0, 2.0]);
        assert_eq!(&j[2..], &[100.0, 100.0]);
    }

    #[test]
    fn scaled_norm_is_frobenius_over_sqrt_d() {
        // 2x2 of all 3s: ‖J‖_F = sqrt(4·9) = 6, /√2 ≈ 4.2426
        let j = vec![3.0f32; 4];
        assert!((scaled_frobenius_norm(&j, 2) - 4.2426_f32).abs() < 1e-4);
    }
}
