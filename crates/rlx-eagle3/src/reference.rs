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

//! Pure-Rust scalar reference operators for the EAGLE3 draft model.
//!
//! Slow (no SIMD, no BLAS, no autotuned blocking) but correct: enough
//! to drive `Eagle3Speculator::propose` end-to-end without standing
//! up the full HIR compile pipeline. Real inference will eventually
//! route through [`crate::draft::Eagle3DraftBuilder`] (HIR graph) —
//! this module's purpose is to be a numerical oracle while that
//! lands, and to make integration tests possible without weights.
//!
//! All inputs/outputs are `f32` flat row-major vectors. No batching.

/// In-place RMSNorm: `x = x * rsqrt(mean(x^2) + eps) * gamma`.
/// `x` and `gamma` must have the same length.
pub fn rms_norm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), gamma.len(), "rms_norm shape mismatch");
    let n = x.len();
    let mean_sq: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / (n as f64);
    let scale = (mean_sq + eps as f64).sqrt().recip();
    x.iter()
        .zip(gamma)
        .map(|(&v, &g)| ((v as f64) * scale) as f32 * g)
        .collect()
}

/// Row-major matmul: `out[i] = sum_j A[i, j] * x[j]` for a single vector
/// input. `a` has shape `[rows, cols]` row-major, `x` has length `cols`.
pub fn matvec(a: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    assert_eq!(
        a.len(),
        rows * cols,
        "matvec: a len {} != rows * cols = {} * {}",
        a.len(),
        rows,
        cols
    );
    assert_eq!(x.len(), cols, "matvec: x len {} != cols {}", x.len(), cols);
    let mut out = vec![0.0f32; rows];
    for i in 0..rows {
        let mut acc: f64 = 0.0;
        let row = &a[i * cols..(i + 1) * cols];
        for j in 0..cols {
            acc += (row[j] as f64) * (x[j] as f64);
        }
        out[i] = acc as f32;
    }
    out
}

/// SiLU activation: `x * sigmoid(x)`.
pub fn silu_in_place(x: &mut [f32]) {
    for v in x.iter_mut() {
        let s = (*v as f64).exp() / (1.0 + (*v as f64).exp());
        *v = ((*v as f64) * s) as f32;
    }
}

/// Element-wise multiply in place.
pub fn mul_in_place(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (av, bv) in a.iter_mut().zip(b) {
        *av *= *bv;
    }
}

/// Element-wise add of `b` into `a`.
pub fn add_in_place(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (av, bv) in a.iter_mut().zip(b) {
        *av += *bv;
    }
}

/// Apply RoPE in place to `qk`, interpreting it as
/// `[n_heads, head_dim]`. `cos` and `sin` have length `head_dim / 2`
/// and correspond to the *current* token position.
///
/// Uses the **split-halves** convention (HF Llama / vllm-speculators
/// `apply_rotary_pos_emb`): the first `head_dim/2` of each head
/// rotates with the second half via
///
/// ```text
/// new_x1 = x1 * cos - x2 * sin
/// new_x2 = x2 * cos + x1 * sin   (=  rotate_half(x) · sin + x · cos)
/// ```
///
/// This matches `Op::Rope` in `rlx-cpu/src/executor.rs:743` (and
/// thus the HIR-compiled draft graph). Earlier revisions of this
/// scalar used interleaved-pairs and silently disagreed with HIR
/// past_seq>0 — caught by `propose_e2e::hir_propose_matches_scalar_on_cpu`.
pub fn rope_in_place(qk: &mut [f32], n_heads: usize, head_dim: usize, cos: &[f32], sin: &[f32]) {
    assert_eq!(
        qk.len(),
        n_heads * head_dim,
        "rope: qk len {} != n_heads * head_dim = {} * {}",
        qk.len(),
        n_heads,
        head_dim
    );
    assert_eq!(cos.len(), head_dim / 2, "rope: cos len");
    assert_eq!(sin.len(), head_dim / 2, "rope: sin len");
    let half = head_dim / 2;
    for h in 0..n_heads {
        let base = h * head_dim;
        for k in 0..half {
            let x1 = qk[base + k];
            let x2 = qk[base + half + k];
            let c = cos[k];
            let s = sin[k];
            qk[base + k] = x1 * c - x2 * s;
            qk[base + half + k] = x2 * c + x1 * s;
        }
    }
}

/// Argmax over a slice (returns 0 on empty).
pub fn argmax(x: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i as u32;
        }
    }
    best
}

/// Softmax in place over the full slice.
pub fn softmax_in_place(x: &mut [f32]) {
    let mut max_v = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max_v {
            max_v = v;
        }
    }
    if !max_v.is_finite() {
        for v in x.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let mut sum = 0.0f64;
    for v in x.iter_mut() {
        let e = ((*v - max_v) as f64).exp();
        *v = e as f32;
        sum += e;
    }
    if sum > 0.0 {
        let inv = (1.0 / sum) as f32;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

/// Single-token GQA attention step. Given:
///
/// - `q`: query for the current token, shape `[n_heads, head_dim]`.
/// - `past_k`: all past keys including the current step, shape
///   `[seq, n_kv_heads, head_dim]`.
/// - `past_v`: all past values, same shape.
///
/// Returns `[n_heads, head_dim]` — the GQA attention output, with
/// no mask (the implicit causal mask is satisfied because `past_k`
/// only contains entries for positions ≤ current).
pub fn gqa_attention(
    q: &[f32],
    past_k: &[f32],
    past_v: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), n_heads * head_dim);
    assert_eq!(past_k.len(), seq * n_kv_heads * head_dim);
    assert_eq!(past_v.len(), seq * n_kv_heads * head_dim);
    assert!(n_kv_heads > 0);
    assert_eq!(n_heads % n_kv_heads, 0, "gqa head count not divisible");
    let group = n_heads / n_kv_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; n_heads * head_dim];

    let mut scores = vec![0.0f32; seq];
    for h in 0..n_heads {
        let kv_h = h / group;
        let q_off = h * head_dim;
        // scores[t] = q[h] · k[t, kv_h]
        for (t, score) in scores.iter_mut().enumerate() {
            let k_off = (t * n_kv_heads + kv_h) * head_dim;
            let mut acc: f64 = 0.0;
            for d in 0..head_dim {
                acc += (q[q_off + d] as f64) * (past_k[k_off + d] as f64);
            }
            *score = (acc as f32) * scale;
        }
        softmax_in_place(&mut scores);
        // out[h, d] = sum_t scores[t] * v[t, kv_h, d]
        let out_off = h * head_dim;
        for d in 0..head_dim {
            let mut acc: f64 = 0.0;
            for (t, &score) in scores.iter().enumerate() {
                let v_off = (t * n_kv_heads + kv_h) * head_dim + d;
                acc += (score as f64) * (past_v[v_off] as f64);
            }
            out[out_off + d] = acc as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_unit_variance_passes_through_gamma() {
        // For x with mean(x^2) = 1, rms_norm normalizes to (1 / sqrt(1+eps))
        // ≈ 1, then multiplies by gamma. With gamma=2, output ≈ 2x.
        let x = vec![1.0, -1.0, 1.0, -1.0];
        let gamma = vec![2.0; 4];
        let out = rms_norm(&x, &gamma, 0.0);
        for v in out {
            assert!((v.abs() - 2.0).abs() < 1e-5, "expected |2|, got {v}");
        }
    }

    #[test]
    fn matvec_identity_passes_through() {
        let n = 4;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        let x = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(matvec(&a, n, n, &x), x);
    }

    #[test]
    fn softmax_uniform_over_neg_inf_logits_outside_d2t() {
        // Models the d2t scatter case: 4 finite + 2 -inf entries.
        let mut x = vec![0.0, 0.0, 0.0, 0.0, f32::NEG_INFINITY, f32::NEG_INFINITY];
        softmax_in_place(&mut x);
        for v in &x[..4] {
            assert!((v - 0.25).abs() < 1e-5);
        }
        assert_eq!(x[4], 0.0);
        assert_eq!(x[5], 0.0);
    }

    #[test]
    fn rope_pos_zero_is_identity() {
        // At position 0, cos=1, sin=0 ⇒ rotation by 0 angle.
        let n_heads = 2;
        let head_dim = 4;
        let mut qk = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let cos = vec![1.0, 1.0];
        let sin = vec![0.0, 0.0];
        let expected = qk.clone();
        rope_in_place(&mut qk, n_heads, head_dim, &cos, &sin);
        for (a, b) in qk.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn rope_pos_pi_over_2_split_halves() {
        // Split-halves convention with cos=0, sin=1:
        //   new_x1 = x1*0 - x2*1 = -x2
        //   new_x2 = x2*0 + x1*1 = +x1
        // For head_dim=4, x = [x1a, x1b, x2a, x2b] (first half +
        // second half), expect [-x2a, -x2b, x1a, x1b].
        let n_heads = 1;
        let head_dim = 4;
        let mut qk = vec![3.0, 5.0, 7.0, 11.0];
        let cos = vec![0.0, 0.0];
        let sin = vec![1.0, 1.0];
        rope_in_place(&mut qk, n_heads, head_dim, &cos, &sin);
        assert!((qk[0] - -7.0).abs() < 1e-5);
        assert!((qk[1] - -11.0).abs() < 1e-5);
        assert!((qk[2] - 3.0).abs() < 1e-5);
        assert!((qk[3] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn argmax_finds_largest() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[]), 0);
        assert_eq!(argmax(&[f32::NEG_INFINITY, 0.0]), 1);
    }

    #[test]
    fn silu_zero_is_zero_one_passes_through_sigmoid() {
        let mut x = vec![0.0f32, 1.0, -1.0];
        silu_in_place(&mut x);
        assert!(x[0].abs() < 1e-6);
        // 1 * sigmoid(1) ≈ 0.7311
        assert!((x[1] - 0.7311).abs() < 1e-3);
        // -1 * sigmoid(-1) ≈ -0.2689
        assert!((x[2] - -0.2689).abs() < 1e-3);
    }

    #[test]
    fn gqa_attention_single_token_returns_v() {
        // Single past entry → softmax over [score] gives [1.0] →
        // output equals v.
        let n_heads = 2;
        let n_kv_heads = 2;
        let head_dim = 2;
        let q = vec![0.0, 1.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 1.0, 0.0]; // shape [1, 2, 2]
        let v = vec![5.0, 6.0, 7.0, 8.0]; // shape [1, 2, 2]
        let out = gqa_attention(&q, &k, &v, n_heads, n_kv_heads, head_dim, 1);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - [5.0, 6.0, 7.0, 8.0][i]).abs() < 1e-4,
                "gqa[{}] = {}",
                i,
                v
            );
        }
    }

    #[test]
    fn gqa_attention_groups_share_kv_heads() {
        // 4 query heads, 2 kv heads, group size = 2.
        // Both query heads in group 0 should consult kv_head 0.
        // Construct so that group 0's q dot product with kv 0's k is
        // large (1.0); group 1's q with kv 1's k is large too.
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 1;
        let seq = 1;
        let q = vec![1.0, 1.0, 1.0, 1.0]; // [4, 1]
        let k = vec![1.0, -10.0]; // [1, 2, 1] — kv0 = 1, kv1 = -10
        let v = vec![5.0, 99.0]; // [1, 2, 1]
        let out = gqa_attention(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq);
        // Heads 0,1 → kv0 → v[0] = 5
        // Heads 2,3 → kv1 → v[1] = 99
        assert!((out[0] - 5.0).abs() < 1e-3);
        assert!((out[1] - 5.0).abs() < 1e-3);
        assert!((out[2] - 99.0).abs() < 1e-3);
        assert!((out[3] - 99.0).abs() < 1e-3);
    }
}
