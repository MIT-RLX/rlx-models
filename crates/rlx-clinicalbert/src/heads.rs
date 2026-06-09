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

//! Optional ClinicalBERT heads — pooler + Masked-Language-Model.
//!
//! These are pure-Rust post-processing on the encoder's `hidden_states`.
//! The encoder still runs on the user's chosen RLX backend; only the small
//! head matmul + activations execute on CPU after the encoder is done.
//!
//! Enabled with the `pooler` and `mlm` Cargo features.

#![cfg(any(feature = "pooler", feature = "mlm"))]

use anyhow::{Context, Result, bail};
use rlx_core::config::BertConfig;
use rlx_core::weight_map::WeightMap;

/// Pre-trained `[CLS]` pooler: `Dense(H → H) + tanh`.
///
/// Consumes `{prefix}pooler.dense.{weight, bias}`. The HF Linear stores
/// weights as `[out, in]`; we transpose to `[in, out]` once at load time so
/// the forward path is `cls @ W + b` with row-major contiguous reads.
#[cfg(feature = "pooler")]
pub struct PoolerHead {
    weight_t: Vec<f32>, // [H, H]  (transposed from HF [H_out, H_in])
    bias: Vec<f32>,     // [H]
    hidden_size: usize,
}

#[cfg(feature = "pooler")]
impl PoolerHead {
    pub fn load(cfg: &BertConfig, weights: &mut WeightMap) -> Result<Self> {
        let h = cfg.hidden_size;
        let prefix = if weights.has("bert.pooler.dense.weight") {
            "bert."
        } else {
            ""
        };
        let (w, w_shape) = weights
            .take(&format!("{prefix}pooler.dense.weight"))
            .with_context(|| format!("loading {prefix}pooler.dense.weight"))?;
        let (b, _) = weights
            .take(&format!("{prefix}pooler.dense.bias"))
            .with_context(|| format!("loading {prefix}pooler.dense.bias"))?;
        if w_shape != vec![h, h] {
            bail!(
                "rlx-clinicalbert: pooler.dense.weight has shape {w_shape:?}, expected [{h}, {h}]"
            );
        }
        let weight_t = transpose(&w, w_shape[0], w_shape[1]);
        Ok(Self {
            weight_t,
            bias: b,
            hidden_size: h,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Apply pooler to encoder `hidden_states [batch, seq, H]`.
    /// Returns `pooler_output [batch, H]` — `tanh(W · h_cls + b)`.
    ///
    /// One fused `sgemm_bias_epilogue` call: matmul + bias add + tanh in a
    /// single sweep over the output buffer.
    pub fn apply(&self, hidden: &[f32], batch: usize, seq: usize) -> Result<Vec<f32>> {
        let h = self.hidden_size;
        if hidden.len() != batch * seq * h {
            bail!(
                "rlx-clinicalbert::PoolerHead: expected hidden of {} floats, got {}",
                batch * seq * h,
                hidden.len()
            );
        }
        // Pack the [CLS] rows for each batch element into a contiguous [B, H]
        // matrix so we can do a single GEMM. For B=1 this is just a copy of
        // the first `H` floats.
        let mut cls = vec![0f32; batch * h];
        for bi in 0..batch {
            let src = bi * seq * h;
            cls[bi * h..(bi + 1) * h].copy_from_slice(&hidden[src..src + h]);
        }
        let mut out = vec![0f32; batch * h];
        // Fused: out = tanh(cls @ weight_t + bias)
        rlx_cpu::blas::sgemm_bias_epilogue(
            &cls,
            &self.weight_t,
            &self.bias,
            &mut out,
            batch,
            h,
            h,
            |v| v.tanh(),
        );
        Ok(out)
    }
}

/// Masked-Language-Model head.
///
/// Layout (HF `BertLMPredictionHead`):
///   `hidden [B,S,H]
///      → Dense(H→H, exact-GeLU) → LayerNorm
///      → Linear(H→V) with weight tied to bert.embeddings.word_embeddings
///      → logits [B,S,V]`
///
/// Consumes `cls.predictions.{transform.dense.{w,b}, transform.LayerNorm.{w,b}, bias}`
/// and a copy of `{prefix}embeddings.word_embeddings.weight` (held internally —
/// HF ties this matrix as the decoder).
#[cfg(feature = "mlm")]
pub struct MlmHead {
    transform_w_t: Vec<f32>, // [H, H]
    transform_b: Vec<f32>,   // [H]
    ln_w: Vec<f32>,          // [H]
    ln_b: Vec<f32>,          // [H]
    /// Tied decoder weight — clone of `bert.embeddings.word_embeddings.weight`.
    /// Stored as `[H, V]` (transposed from HF `[V, H]`) for `x @ W` matmul.
    decoder_w_t: Vec<f32>, // [H, V]
    decoder_b: Vec<f32>,     // [V]  (cls.predictions.bias)
    hidden_size: usize,
    vocab_size: usize,
    eps: f32,
}

#[cfg(feature = "mlm")]
impl MlmHead {
    /// Loads MLM head weights. Must be called BEFORE `BertFlow::build` so the
    /// embedding matrix is still present (we clone, BertFlow then consumes).
    pub fn load(cfg: &BertConfig, weights: &mut WeightMap) -> Result<Self> {
        let h = cfg.hidden_size;
        let v = cfg.vocab_size;
        let prefix = if weights.has("bert.embeddings.word_embeddings.weight") {
            "bert."
        } else {
            ""
        };

        // 1. transform.dense + LayerNorm + bias  (consume; not needed by encoder).
        let (tw, tw_shape) = weights
            .take("cls.predictions.transform.dense.weight")
            .context("loading cls.predictions.transform.dense.weight")?;
        let (tb, _) = weights
            .take("cls.predictions.transform.dense.bias")
            .context("loading cls.predictions.transform.dense.bias")?;
        let (lnw, _) = weights
            .take("cls.predictions.transform.LayerNorm.weight")
            .context("loading cls.predictions.transform.LayerNorm.weight")?;
        let (lnb, _) = weights
            .take("cls.predictions.transform.LayerNorm.bias")
            .context("loading cls.predictions.transform.LayerNorm.bias")?;
        let decoder_b = if weights.has("cls.predictions.bias") {
            weights
                .take("cls.predictions.bias")
                .context("loading cls.predictions.bias")?
                .0
        } else if weights.has("cls.predictions.decoder.bias") {
            weights
                .take("cls.predictions.decoder.bias")
                .context("loading cls.predictions.decoder.bias")?
                .0
        } else {
            bail!("rlx-clinicalbert: MLM bias missing (cls.predictions.bias / .decoder.bias)");
        };
        if decoder_b.len() != v {
            bail!(
                "rlx-clinicalbert: MLM bias length {} != vocab_size {v}",
                decoder_b.len()
            );
        }

        // 2. Decoder weight — either explicit `cls.predictions.decoder.weight`,
        //    or (more commonly) tied with the input embedding matrix. We clone
        //    via `get` so the encoder builder still gets a fresh take().
        let decoder_w_raw: Vec<f32>;
        let decoder_w_shape: Vec<usize>;
        if weights.has("cls.predictions.decoder.weight") {
            let (w, s) = weights
                .take("cls.predictions.decoder.weight")
                .context("loading cls.predictions.decoder.weight")?;
            decoder_w_raw = w;
            decoder_w_shape = s;
        } else {
            let key = format!("{prefix}embeddings.word_embeddings.weight");
            let (data, shape) = weights
                .get(&key)
                .ok_or_else(|| anyhow::anyhow!("rlx-clinicalbert: tied MLM decoder needs {key}"))?;
            decoder_w_raw = data.to_vec();
            decoder_w_shape = shape.to_vec();
        }
        if decoder_w_shape != vec![v, h] {
            bail!(
                "rlx-clinicalbert: MLM decoder weight has shape {decoder_w_shape:?}, expected [{v}, {h}]"
            );
        }

        let transform_w_t = transpose(&tw, tw_shape[0], tw_shape[1]);
        let decoder_w_t = transpose(&decoder_w_raw, decoder_w_shape[0], decoder_w_shape[1]);

        Ok(Self {
            transform_w_t,
            transform_b: tb,
            ln_w: lnw,
            ln_b: lnb,
            decoder_w_t,
            decoder_b,
            hidden_size: h,
            vocab_size: v,
            eps: cfg.layer_norm_eps as f32,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Apply MLM head into a caller-provided logits buffer. Avoids the
    /// `Vec::new` + zero-fill for the 3.6MB output buffer on repeated calls.
    /// `logits.len()` must equal `batch * seq * vocab_size`.
    pub fn apply_into(
        &self,
        hidden: &[f32],
        batch: usize,
        seq: usize,
        logits: &mut [f32],
    ) -> Result<()> {
        let h = self.hidden_size;
        let v = self.vocab_size;
        let bs = batch * seq;
        if hidden.len() != bs * h {
            bail!(
                "rlx-clinicalbert::MlmHead: expected hidden of {} floats, got {}",
                bs * h,
                hidden.len()
            );
        }
        if logits.len() != bs * v {
            bail!(
                "rlx-clinicalbert::MlmHead: logits buffer must be {} floats, got {}",
                bs * v,
                logits.len()
            );
        }
        let mut x = vec![0f32; bs * h];
        rlx_cpu::blas::sgemm_bias_epilogue(
            hidden,
            &self.transform_w_t,
            &self.transform_b,
            &mut x,
            bs,
            h,
            h,
            gelu_exact,
        );
        for row in 0..bs {
            let off = row * h;
            let slice = &mut x[off..off + h];
            let mean: f32 = slice.iter().copied().sum::<f32>() / (h as f32);
            let mut var = 0f32;
            for &val in slice.iter() {
                let d = val - mean;
                var += d * d;
            }
            var /= h as f32;
            let inv_std = (var + self.eps).sqrt().recip();
            for j in 0..h {
                slice[j] = (slice[j] - mean) * inv_std * self.ln_w[j] + self.ln_b[j];
            }
        }
        rlx_cpu::blas::sgemm_bias(&x, &self.decoder_w_t, &self.decoder_b, logits, bs, h, v);
        Ok(())
    }

    /// Apply MLM head: `hidden [B,S,H] → logits [B,S,V]`.
    ///
    /// Fused BLAS path:
    ///  * Transform layer: `sgemm_bias_epilogue` does matmul + bias + GeLU
    ///    in one pass.
    ///  * Decoder: `sgemm_bias` does matmul + bias in one pass.
    ///
    /// LayerNorm is `O(B·S·H)` elementwise and runs inline between them.
    pub fn apply(&self, hidden: &[f32], batch: usize, seq: usize) -> Result<Vec<f32>> {
        let h = self.hidden_size;
        let v = self.vocab_size;
        let bs = batch * seq;
        if hidden.len() != bs * h {
            bail!(
                "rlx-clinicalbert::MlmHead: expected hidden of {} floats, got {}",
                bs * h,
                hidden.len()
            );
        }

        // 1. Fused Dense + bias + exact GeLU.
        let mut x = vec![0f32; bs * h];
        rlx_cpu::blas::sgemm_bias_epilogue(
            hidden,
            &self.transform_w_t,
            &self.transform_b,
            &mut x,
            bs,
            h,
            h,
            gelu_exact,
        );
        // 2. LayerNorm per row.
        for row in 0..bs {
            let off = row * h;
            let slice = &mut x[off..off + h];
            let mean: f32 = slice.iter().copied().sum::<f32>() / (h as f32);
            let mut var = 0f32;
            for &val in slice.iter() {
                let d = val - mean;
                var += d * d;
            }
            var /= h as f32;
            let inv_std = (var + self.eps).sqrt().recip();
            for j in 0..h {
                slice[j] = (slice[j] - mean) * inv_std * self.ln_w[j] + self.ln_b[j];
            }
        }
        // 3. Fused Decoder + bias.
        let mut logits = vec![0f32; bs * v];
        rlx_cpu::blas::sgemm_bias(
            &x,
            &self.decoder_w_t,
            &self.decoder_b,
            &mut logits,
            bs,
            h,
            v,
        );
        Ok(logits)
    }

    /// Allocate a properly-sized logits buffer for repeated `apply_into` calls.
    pub fn allocate_logits_buffer(&self, batch: usize, seq: usize) -> Vec<f32> {
        vec![0f32; batch * seq * self.vocab_size]
    }
}

fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Exact GeLU using an Abramowitz & Stegun erf polynomial (max error ~1.5e-7,
/// well within FP32 precision). Matches HF's default `gelu` (PyTorch
/// `approximate='none'`).
#[allow(dead_code)]
fn gelu_exact(x: f32) -> f32 {
    0.5 * x * (1.0 + erf_approx(x / std::f32::consts::SQRT_2))
}

#[allow(dead_code, clippy::excessive_precision)]
fn erf_approx(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26 — max abs error ≈ 1.5e-7.
    let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0
        - ((((1.061405429_f32 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-(ax * ax)).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_libm_ish_values() {
        // Spot-check erf against textbook reference values.
        assert!((erf_approx(0.0)).abs() < 1e-7);
        assert!((erf_approx(1.0) - 0.842_700_8).abs() < 1e-6);
        assert!((erf_approx(-1.0) + 0.842_700_8).abs() < 1e-6);
        assert!((erf_approx(2.0) - 0.995_322_3).abs() < 1e-6);
    }
}
