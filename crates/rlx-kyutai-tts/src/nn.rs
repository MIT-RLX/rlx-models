//! Native eager nn primitives used by the Kyutai TTS modules.
//!
//! All ops are pure-Rust ndarray; nothing depends on candle / moshi / GPU
//! runtimes. Intentionally small — only what the Kyutai-TTS-specific layers
//! ([`crate::cross_attention`], [`crate::conditioner`], [`crate::depformer`],
//! [`crate::low_rank_embedding`]) consume.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Epsilon used by all RMSNorm variants in this crate (matches Kyutai TTS
/// `norm: "rms_norm_f32"`).
pub const RMS_EPS: f32 = 1e-8;

#[derive(Debug, Clone)]
pub struct Embedding {
    pub weight: Array2<f32>, // [vocab, dim]
}

impl Embedding {
    pub fn forward_one(&self, id: u32) -> Array1<f32> {
        self.weight.row(id as usize).to_owned()
    }

    pub fn forward(&self, ids: &[u32]) -> Array2<f32> {
        let d = self.weight.ncols();
        let mut out = Array2::<f32>::zeros((ids.len(), d));
        for (i, &id) in ids.iter().enumerate() {
            let row = self.weight.row(id as usize);
            for j in 0..d {
                out[[i, j]] = row[j];
            }
        }
        out
    }
}

/// `x @ w^T` — bias-free linear projection matching the Kyutai checkpoints.
pub fn linear(x: ArrayView2<f32>, w: &Array2<f32>) -> Array2<f32> {
    x.dot(&w.t())
}

/// `y_i = x_i / sqrt(mean(x^2) + eps) * alpha_i` — RMSNorm with per-channel scale.
pub fn rms_norm(x: ArrayView2<f32>, alpha: &Array1<f32>) -> Array2<f32> {
    let (t, c) = x.dim();
    let mut out = Array2::<f32>::zeros((t, c));
    let inv_c = 1.0 / c as f32;
    for ti in 0..t {
        let row = x.row(ti);
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() * inv_c;
        let scale = 1.0 / (mean_sq + RMS_EPS).sqrt();
        for ci in 0..c {
            out[[ti, ci]] = row[ci] * scale * alpha[ci];
        }
    }
    out
}

/// Affine layer norm over the last dim: `(x - mean) / std * weight + bias`.
pub fn layer_norm(x: ArrayView2<f32>, weight: &Array1<f32>, bias: &Array1<f32>) -> Array2<f32> {
    let (t, c) = x.dim();
    let mut out = Array2::<f32>::zeros((t, c));
    let inv_c = 1.0 / c as f32;
    for ti in 0..t {
        let row = x.row(ti);
        let mean = row.iter().sum::<f32>() * inv_c;
        let var = row
            .iter()
            .map(|v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            * inv_c;
        let inv_std = 1.0 / (var + RMS_EPS).sqrt();
        for ci in 0..c {
            out[[ti, ci]] = (row[ci] - mean) * inv_std * weight[ci] + bias[ci];
        }
    }
    out
}

/// SiLU / Swish gate.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU MLP: `out = (silu(x·Wgate) * x·Wup) · Wdown`.
///
/// `linear_in` packs `[Wgate; Wup]` so the result has shape `[t, 2·hidden]`;
/// the first half is gated by SiLU and elementwise-multiplied with the second.
pub fn swiglu_mlp(
    x: ArrayView2<f32>,
    linear_in: &Array2<f32>,
    linear_out: &Array2<f32>,
) -> Array2<f32> {
    let gate_up = linear(x, linear_in);
    let (t, two_h) = gate_up.dim();
    let h = two_h / 2;
    let mut gated = Array2::<f32>::zeros((t, h));
    for ti in 0..t {
        for hi in 0..h {
            let gate = silu(gate_up[[ti, hi]]);
            let up = gate_up[[ti, h + hi]];
            gated[[ti, hi]] = gate * up;
        }
    }
    linear(gated.view(), linear_out)
}

/// Row-wise softmax (last axis).
pub fn softmax_last_dim(logits: &Array2<f32>) -> Array2<f32> {
    let (t, v) = logits.dim();
    let mut out = Array2::<f32>::zeros((t, v));
    for ti in 0..t {
        let row = logits.row(ti);
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for vi in 0..v {
            let e = (row[vi] - max).exp();
            out[[ti, vi]] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for vi in 0..v {
            out[[ti, vi]] *= inv;
        }
    }
    out
}

/// In-place row-wise softmax (used by attention kernels).
pub fn softmax_inplace(row: &mut [f32]) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}

/// Sinusoidal positional embedding table — matches Moshi `create_sin_embedding`.
///
/// Layout: `[cos(phase_0..), sin(phase_0..)]` with
/// `phase_i = pos / max_period^(i / (half - 1))`.
pub fn sin_pos_embed(len: usize, dim: usize, max_period: f32) -> Array2<f32> {
    assert!(
        dim.is_multiple_of(2),
        "sin pos-emb dim must be even, got {dim}"
    );
    let half = dim / 2;
    let mut out = Array2::<f32>::zeros((len, dim));
    let denom = (half as f32 - 1.0).max(1.0);
    for pos in 0..len {
        for i in 0..half {
            let phase = pos as f32 / max_period.powf(i as f32 / denom);
            out[[pos, i]] = phase.cos();
            out[[pos, i + half]] = phase.sin();
        }
    }
    out
}

/// Interleaved RoPE (Moshi `apply_rope` with `interleave=True`) on one Q/K head vector.
pub fn apply_rope_interleaved(q: &mut [f32], k: &mut [f32], pos: usize, max_period: f32) {
    let d = q.len();
    assert_eq!(d, k.len());
    assert!(d.is_multiple_of(2));
    for i in 0..(d / 2) {
        let freq = max_period.powf(-2.0 * i as f32 / d as f32);
        let f = pos as f32 * freq;
        let rotr = f.cos();
        let roti = f.sin();
        let qr = q[2 * i];
        let qi = q[2 * i + 1];
        q[2 * i] = qr * rotr - qi * roti;
        q[2 * i + 1] = qr * roti + qi * rotr;
        let kr = k[2 * i];
        let ki = k[2 * i + 1];
        k[2 * i] = kr * rotr - ki * roti;
        k[2 * i + 1] = kr * roti + ki * rotr;
    }
}

/// Row sum: `y = x.sum(axis=0)`.
pub fn sum_rows(x: ArrayView2<f32>) -> Array1<f32> {
    x.sum_axis(ndarray::Axis(0))
}

/// RoPE table generator (matches Kyutai TTS `max_period = 10_000`).
///
/// Returns `(cos, sin)` of shape `[len(positions), head_dim]`.
/// Each row is laid out so the first half holds the sin/cos of the low-frequency
/// pair and the second half mirrors them for the half-rotation form used by
/// [`apply_rope_vec`].
pub fn rope_tables(
    head_dim: usize,
    max_period: usize,
    positions: &[usize],
) -> (Array2<f32>, Array2<f32>) {
    let half = head_dim / 2;
    let t = positions.len();
    let mut inv_freq = Vec::with_capacity(half);
    for i in 0..half {
        inv_freq.push(1.0 / (max_period as f32).powf(i as f32 / half as f32));
    }
    let mut cos = Array2::<f32>::zeros((t, head_dim));
    let mut sin = Array2::<f32>::zeros((t, head_dim));
    for (ti, &pos) in positions.iter().enumerate() {
        for hi in 0..half {
            let f = pos as f32 * inv_freq[hi];
            let c = f.cos();
            let s = f.sin();
            cos[[ti, hi]] = c;
            cos[[ti, hi + half]] = c;
            sin[[ti, hi]] = s;
            sin[[ti, hi + half]] = s;
        }
    }
    (cos, sin)
}

/// Apply RoPE to a single (Q, K) head-vector pair, in-place.
///
/// `cos` / `sin` are one row of the table from [`rope_tables`] — length `head_dim`.
pub fn apply_rope_vec(q: &mut [f32], k: &mut [f32], cos: ArrayView1<f32>, sin: ArrayView1<f32>) {
    let half = cos.len() / 2;
    for hi in 0..half {
        let qc = q[hi];
        let qs = q[hi + half];
        q[hi] = qc * cos[hi] - qs * sin[hi];
        q[hi + half] = qs * cos[hi + half] + qc * sin[hi + half];
        let kc = k[hi];
        let ks = k[hi + half];
        k[hi] = kc * cos[hi] - ks * sin[hi];
        k[hi + half] = ks * cos[hi + half] + kc * sin[hi + half];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn linear_matches_manual_dot() {
        let x = array![[1.0, 2.0, 3.0]];
        let w = array![[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]];
        let y = linear(x.view(), &w);
        // row0: [1,2,3] · [1,0,-1] = -2, [1,2,3] · [2,1,0] = 4
        assert_eq!(y, array![[-2.0, 4.0]]);
    }

    #[test]
    fn rms_norm_preserves_unit_input() {
        // alpha = 1, unit-norm input → output ≈ input.
        let x = array![[1.0, 0.0, 0.0]];
        let alpha = array![1.0, 1.0, 1.0];
        let y = rms_norm(x.view(), &alpha);
        // sqrt(mean([1,0,0]^2)) = sqrt(1/3) ⇒ scale = sqrt(3); y[0,0] ≈ sqrt(3)
        let s = (3.0_f32).sqrt();
        assert!((y[[0, 0]] - s).abs() < 1e-3, "got {}", y[[0, 0]]);
    }

    #[test]
    fn silu_matches_known_values() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!((silu(1.0) - 0.7310586).abs() < 1e-4);
        assert!((silu(-10.0)).abs() < 1e-3); // SiLU(-10) ≈ 0
    }

    #[test]
    fn softmax_sums_to_one() {
        let x = array![[1.0, 2.0, 3.0]];
        let p = softmax_last_dim(&x);
        let s: f32 = p.row(0).iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
        // Monotonic in input.
        assert!(p[[0, 0]] < p[[0, 1]] && p[[0, 1]] < p[[0, 2]]);
    }

    #[test]
    fn sin_pos_embed_matches_moshi_layout() {
        let pe = sin_pos_embed(1, 8, 10_000.0);
        assert_eq!(pe.dim(), (1, 8));
        // pos=0 → cos=1 in first half, sin=0 in second half.
        for i in 0..4 {
            assert!((pe[[0, i]] - 1.0).abs() < 1e-6, "cos@0");
            assert!(pe[[0, 4 + i]].abs() < 1e-6, "sin@0");
        }
    }

    #[test]
    fn apply_rope_interleaved_at_zero_is_identity() {
        let mut q = vec![1.0, 2.0, 3.0, 4.0];
        let mut k = vec![5.0, 6.0, 7.0, 8.0];
        let q0 = q.clone();
        let k0 = k.clone();
        apply_rope_interleaved(&mut q, &mut k, 0, 10_000.0);
        assert_eq!(q, q0);
        assert_eq!(k, k0);
    }

    #[test]
    fn embedding_forward_pulls_rows() {
        let emb = Embedding {
            weight: array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        };
        assert_eq!(emb.forward_one(1), array![3.0, 4.0]);
        let batched = emb.forward(&[2, 0]);
        assert_eq!(batched, array![[5.0, 6.0], [1.0, 2.0]]);
    }
}
