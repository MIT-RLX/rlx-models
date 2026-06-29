//! Multi-head cross-attention for speaker conditioning.
//!
//! Kyutai TTS injects voice identity into every backbone transformer layer
//! via a cross-attention block whose Q comes from the temporal hidden state
//! and K/V come from the fuser's `cross` context (typically the `speaker_wavs`
//! tensor conditioner output).
//!
//! Architectural notes (from `config.json`):
//!
//! - `cross_attention = true`
//! - `fuser.cross_attention_pos_emb = true`, `_scale = 1.0` — sinusoidal
//!   positional embedding added to K/V before projection.
//! - The K/V context is fixed for the duration of one generation, so we cache
//!   the projected `k_proj`, `v_proj` once in [`CrossAttention::prepare_kv`]
//!   and reuse them on every step.

use crate::nn::{linear, sin_pos_embed, softmax_inplace};
use crate::util::split_qkv;
use anyhow::{Result, bail};
use ndarray::{Array1, Array2};

/// Projected cross-attention K/V cache.
#[derive(Debug, Clone)]
pub struct CrossKvCache {
    pub k: Array2<f32>, // [t, num_heads * head_dim]
    pub v: Array2<f32>, // [t, num_heads * head_dim]
    pub num_heads: usize,
    pub head_dim: usize,
}

/// Multi-head cross-attention.
#[derive(Debug, Clone)]
pub struct CrossAttention {
    pub d_model: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    /// `[d_model, d_model]` query projection.
    pub w_q: Array2<f32>,
    /// `[d_model, d_kv]` where `d_kv` matches the cross context dim.
    pub w_k: Array2<f32>,
    pub w_v: Array2<f32>,
    /// `[d_model, d_model]` output projection.
    pub w_o: Array2<f32>,
    /// When true, add a sinusoidal positional embedding to the cross context
    /// before K/V projection (matches `fuser.cross_attention_pos_emb`).
    pub pos_emb: bool,
    /// Scale applied to the positional embedding (matches
    /// `fuser.cross_attention_pos_emb_scale`).
    pub pos_emb_scale: f32,
    /// `max_period` for the sinusoidal table.
    pub pos_max_period: f32,
}

impl CrossAttention {
    /// Build from the fused `in_proj_weight [3·d, d]` used in Kyutai checkpoints.
    pub fn from_fused_in_proj(
        d_model: usize,
        num_heads: usize,
        in_proj: Array2<f32>,
        out_proj: Array2<f32>,
        pos_emb: bool,
        pos_emb_scale: f32,
        pos_max_period: f32,
    ) -> Result<Self> {
        let head_dim = d_model / num_heads;
        let (w_q, w_k, w_v) = split_qkv(&in_proj, d_model)?;
        Ok(Self {
            d_model,
            num_heads,
            head_dim,
            w_q,
            w_k,
            w_v,
            w_o: out_proj,
            pos_emb,
            pos_emb_scale,
            pos_max_period,
        })
    }

    /// Project the cross context to a reusable K/V cache. Call once per voice.
    pub fn prepare_kv(&self, ctx: &Array2<f32>) -> Result<CrossKvCache> {
        let t = ctx.nrows();
        let d_kv = ctx.ncols();
        if d_kv != self.w_k.ncols() {
            bail!(
                "cross context dim {} != w_k cols {}",
                d_kv,
                self.w_k.ncols()
            );
        }
        let ctx = if self.pos_emb && t > 0 {
            let pe = sin_pos_embed(t, d_kv, self.pos_max_period);
            let mut out = ctx.clone();
            for ti in 0..t {
                for di in 0..d_kv {
                    out[[ti, di]] += self.pos_emb_scale * pe[[ti, di]];
                }
            }
            out
        } else {
            ctx.clone()
        };
        let k = linear(ctx.view(), &self.w_k); // [t, d_model]
        let v = linear(ctx.view(), &self.w_v);
        Ok(CrossKvCache {
            k,
            v,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
        })
    }

    /// One-step cross-attention. `hidden` is `[d_model]`, returns `[d_model]`.
    pub fn forward_step(&self, hidden: &Array1<f32>, kv: &CrossKvCache) -> Result<Array1<f32>> {
        let h_view = hidden.view().insert_axis(ndarray::Axis(0));
        let q = linear(h_view, &self.w_q); // [1, d_model]
        let nh = self.num_heads;
        let dh = self.head_dim;
        let scale = (dh as f32).powf(-0.5);
        let t_kv = kv.k.nrows();

        let mut out = Array1::<f32>::zeros(self.d_model);
        // Per-head scaled dot-product attention against fixed K/V.
        for head in 0..nh {
            let h_off = head * dh;
            // Score every kv row.
            let mut scores = vec![0.0f32; t_kv];
            for ti in 0..t_kv {
                let mut s = 0.0f32;
                for di in 0..dh {
                    s += q[[0, h_off + di]] * kv.k[[ti, h_off + di]];
                }
                scores[ti] = s * scale;
            }
            softmax_inplace(&mut scores);
            // Weighted V.
            for ti in 0..t_kv {
                let w = scores[ti];
                for di in 0..dh {
                    out[h_off + di] += w * kv.v[[ti, h_off + di]];
                }
            }
        }

        let out_v = out.view().insert_axis(ndarray::Axis(0));
        let projected = linear(out_v, &self.w_o);
        Ok(projected.row(0).to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    fn random_mat(rows: usize, cols: usize, seed: u32) -> Array2<f32> {
        // Deterministic LCG so tests are stable; we don't want a rand dep here.
        let mut state = seed as u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as i32) as f32 * 1e-9
        };
        Array2::from_shape_fn((rows, cols), |_| next())
    }

    fn cross_fixture() -> CrossAttention {
        // 2 heads × 4 head_dim = 8 d_model; cross context is also 8-D.
        let d = 8;
        CrossAttention {
            d_model: d,
            num_heads: 2,
            head_dim: 4,
            w_q: random_mat(d, d, 1),
            w_k: random_mat(d, d, 2),
            w_v: random_mat(d, d, 3),
            w_o: random_mat(d, d, 4),
            pos_emb: false,
            pos_emb_scale: 1.0,
            pos_max_period: 10_000.0,
        }
    }

    #[test]
    fn forward_step_shape_is_d_model() {
        let xa = cross_fixture();
        let ctx = random_mat(5, 8, 5); // [t_kv=5, d=8]
        let kv = xa.prepare_kv(&ctx).unwrap();
        assert_eq!(kv.k.dim(), (5, 8));
        assert_eq!(kv.v.dim(), (5, 8));
        let h = Array1::<f32>::from_vec(vec![0.1; 8]);
        let out = xa.forward_step(&h, &kv).unwrap();
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn single_kv_row_is_just_weighted_v() {
        // With one kv row, attention softmax collapses to 1 — output ≈ W_o · (W_v · ctx).
        let mut xa = cross_fixture();
        xa.pos_emb = false;
        let ctx = random_mat(1, 8, 6);
        let kv = xa.prepare_kv(&ctx).unwrap();
        let h = Array1::<f32>::from_vec(vec![0.1; 8]);
        let out = xa.forward_step(&h, &kv).unwrap();
        // Deterministic value: W_o · v_row. Compare to manual.
        let expected = linear(kv.v.view(), &xa.w_o);
        for i in 0..8 {
            assert!(
                (out[i] - expected[[0, i]]).abs() < 1e-3,
                "i={i}: out={} expected={}",
                out[i],
                expected[[0, i]]
            );
        }
    }

    #[test]
    fn prepare_kv_rejects_dim_mismatch() {
        let xa = cross_fixture();
        let bad_ctx = random_mat(3, 4, 7); // 4 != 8
        assert!(xa.prepare_kv(&bad_ctx).is_err());
    }

    #[test]
    fn pos_emb_changes_kv_when_enabled() {
        let mut xa = cross_fixture();
        let ctx = random_mat(3, 8, 8);
        xa.pos_emb = false;
        let kv0 = xa.prepare_kv(&ctx).unwrap();
        xa.pos_emb = true;
        let kv1 = xa.prepare_kv(&ctx).unwrap();
        // KV at pos>0 should differ once positional bias is added.
        let mut differs = false;
        for ti in 1..3 {
            for di in 0..8 {
                if (kv0.k[[ti, di]] - kv1.k[[ti, di]]).abs() > 1e-5 {
                    differs = true;
                }
            }
        }
        assert!(
            differs,
            "positional embedding should mutate K rows past pos 0"
        );
    }
}
