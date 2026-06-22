//! Native streaming temporal transformer for Kyutai TTS.
//!
//! Helium-style decoder-only stack:
//!
//! ```text
//! h = x
//! for layer in layers:
//!     h += self_attn(rms_norm1(h))            # RoPE, causal, KV cache
//!     if cross_attention:
//!         h += cross_attn(rms_norm_cx(h), kv) # Q from h, K/V from speaker context
//!     h += swiglu_mlp(rms_norm2(h))
//! ```
//!
//! Differences vs. the rlx-moshi backbone:
//!
//! - **Cross-attention per layer** when [`TransformerConfig::cross_attention`]
//!   is set (Kyutai TTS `cross_attention = true`).
//! - **Hidden FFN width** is derived from `hidden_scale` (= `dim_feedforward`)
//!   rather than the Moshi 11/4 SwiGLU heuristic, since Kyutai sets
//!   `hidden_scale = 4.125` explicitly.
//!
//! Streaming: one step at a time. KV cache is appended on every step; ring-buffer
//! when `len > context`.

use crate::config::PositionalEmbedding;
use crate::cross_attention::{CrossAttention, CrossKvCache};
use crate::nn::{apply_rope_vec, linear, rms_norm, rope_tables, swiglu_mlp};
use anyhow::Result;
use ndarray::{Array1, Array2, Array3};

/// Static config for one transformer stack (backbone or DepFormer inner block).
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    /// SwiGLU FFN hidden width (packed `[2·hidden, d_model]` in/out projections).
    pub dim_feedforward: usize,
    pub causal: bool,
    /// Max context length for the KV cache ring buffer.
    pub context: usize,
    /// RoPE `max_period` (10 000 in Kyutai TTS).
    pub max_period: usize,
    pub positional_embedding: PositionalEmbedding,
    /// When true, every layer has a cross-attention block.
    pub cross_attention: bool,
}

#[derive(Debug, Clone)]
pub struct AttnWeights {
    /// Packed QKV input projection: `[3·d_model, d_model]`.
    pub in_proj: Array2<f32>,
    /// Output projection: `[d_model, d_model]`.
    pub out_proj: Array2<f32>,
    pub num_heads: usize,
    pub head_dim: usize,
}

/// One transformer layer's static weights.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub norm1_alpha: Array1<f32>,
    pub norm2_alpha: Array1<f32>,
    pub attn: AttnWeights,
    pub gate_in: Array2<f32>,
    pub gate_out: Array2<f32>,
    pub cross_attn: Option<CrossAttention>,
    pub norm_cross_alpha: Option<Array1<f32>>,
}

#[derive(Debug, Clone)]
struct KvCache {
    k: Array3<f32>, // [heads, context, head_dim]
    v: Array3<f32>,
    len: usize,
    context: usize,
}

impl KvCache {
    fn new(num_heads: usize, head_dim: usize, context: usize) -> Self {
        Self {
            k: Array3::<f32>::zeros((num_heads, context, head_dim)),
            v: Array3::<f32>::zeros((num_heads, context, head_dim)),
            len: 0,
            context,
        }
    }

    fn reset(&mut self) {
        self.len = 0;
    }

    fn append(&mut self, k_new: &Array3<f32>, v_new: &Array3<f32>) {
        let t = k_new.dim().1;
        for step in 0..t {
            let slot = self.len % self.context;
            for h in 0..self.k.dim().0 {
                for d in 0..self.k.dim().2 {
                    self.k[[h, slot, d]] = k_new[[h, step, d]];
                    self.v[[h, slot, d]] = v_new[[h, step, d]];
                }
            }
            self.len += 1;
        }
    }

    fn effective_len(&self) -> usize {
        self.len.min(self.context)
    }
}

fn self_attention(
    w: &AttnWeights,
    kv: &mut KvCache,
    x: &Array2<f32>,
    rope_cos: Option<&Array2<f32>>,
    rope_sin: Option<&Array2<f32>>,
    causal: bool,
) -> Array2<f32> {
    let (t, d_model) = x.dim();
    let h = w.num_heads;
    let hd = w.head_dim;
    let qkv = linear(x.view(), &w.in_proj);
    let mut q = Array3::<f32>::zeros((h, t, hd));
    let mut k = Array3::<f32>::zeros((h, t, hd));
    let mut v = Array3::<f32>::zeros((h, t, hd));
    for ti in 0..t {
        for hi in 0..h {
            for di in 0..hd {
                let base = hi * hd + di;
                q[[hi, ti, di]] = qkv[[ti, base]];
                k[[hi, ti, di]] = qkv[[ti, h * hd + base]];
                v[[hi, ti, di]] = qkv[[ti, 2 * h * hd + base]];
            }
        }
    }
    if let (Some(cos), Some(sin)) = (rope_cos, rope_sin) {
        for ti in 0..t {
            for hi in 0..h {
                let mut qv = q.slice_mut(ndarray::s![hi, ti, ..]).to_vec();
                let mut kv_ = k.slice_mut(ndarray::s![hi, ti, ..]).to_vec();
                apply_rope_vec(&mut qv, &mut kv_, cos.row(ti), sin.row(ti));
                for di in 0..hd {
                    q[[hi, ti, di]] = qv[di];
                    k[[hi, ti, di]] = kv_[di];
                }
            }
        }
    }
    kv.append(&k, &v);
    let k_len = kv.effective_len();
    let mut out = Array2::<f32>::zeros((t, d_model));
    let scale = 1.0 / (hd as f32).sqrt();
    for ti in 0..t {
        let q_pos = kv.len - t + ti;
        for hi in 0..h {
            let mut weights = Vec::with_capacity(k_len);
            let mut keep = Vec::with_capacity(k_len);
            for ki in 0..k_len {
                let slot = if kv.len <= kv.context {
                    ki
                } else {
                    (kv.len - k_len + ki) % kv.context
                };
                let k_pos = if kv.len <= kv.context {
                    ki
                } else {
                    kv.len - k_len + ki
                };
                if causal && k_pos > q_pos {
                    continue;
                }
                let mut dot = 0.0f32;
                for di in 0..hd {
                    dot += q[[hi, ti, di]] * kv.k[[hi, slot, di]];
                }
                weights.push((dot * scale).exp());
                keep.push(slot);
            }
            let sum: f32 = weights.iter().sum();
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for (wi, &slot) in keep.iter().enumerate() {
                let w_norm = weights[wi] * inv;
                for di in 0..hd {
                    out[[ti, hi * hd + di]] += w_norm * kv.v[[hi, slot, di]];
                }
            }
        }
    }
    linear(out.view(), &w.out_proj)
}

#[derive(Debug, Clone)]
struct Layer {
    w: LayerWeights,
    kv: KvCache,
}

impl Layer {
    fn forward(
        &mut self,
        x: &Array2<f32>,
        rope_cos: Option<&Array2<f32>>,
        rope_sin: Option<&Array2<f32>>,
        causal: bool,
        cross_kv: Option<&CrossKvCache>,
    ) -> Result<Array2<f32>> {
        let n1 = rms_norm(x.view(), &self.w.norm1_alpha);
        let attn_out = self_attention(&self.w.attn, &mut self.kv, &n1, rope_cos, rope_sin, causal);
        let mut h = x + attn_out;
        if let (Some(xa), Some(nx_a), Some(kv)) = (
            self.w.cross_attn.as_ref(),
            self.w.norm_cross_alpha.as_ref(),
            cross_kv,
        ) {
            let n_cx = rms_norm(h.view(), nx_a);
            let mut cross_out = Array2::<f32>::zeros(h.dim());
            for ti in 0..n_cx.nrows() {
                let row = n_cx.row(ti).to_owned();
                let y = xa.forward_step(&row, kv)?;
                for di in 0..y.len() {
                    cross_out[[ti, di]] = y[di];
                }
            }
            h = &h + &cross_out;
        }
        let n2 = rms_norm(h.view(), &self.w.norm2_alpha);
        let mlp_out = swiglu_mlp(n2.view(), &self.w.gate_in, &self.w.gate_out);
        h = &h + &mlp_out;
        Ok(h)
    }

    fn reset(&mut self) {
        self.kv.reset();
    }
}

/// Streaming transformer stack.
#[derive(Debug)]
pub struct StreamingTransformer {
    cfg: TransformerConfig,
    layers: Vec<Layer>,
    seq_len: usize,
}

impl StreamingTransformer {
    /// Construct from already-loaded per-layer weights. Allocates the KV cache.
    pub fn new(cfg: TransformerConfig, weights: Vec<LayerWeights>) -> Result<Self> {
        anyhow::ensure!(
            weights.len() == cfg.num_layers,
            "expected {} layers of weights, got {}",
            cfg.num_layers,
            weights.len()
        );
        let head_dim = cfg.d_model / cfg.num_heads;
        let layers = weights
            .into_iter()
            .map(|w| Layer {
                w,
                kv: KvCache::new(cfg.num_heads, head_dim, cfg.context.max(1)),
            })
            .collect();
        Ok(Self {
            cfg,
            layers,
            seq_len: 0,
        })
    }

    /// Clear KV cache and step counter.
    pub fn reset_state(&mut self) {
        self.seq_len = 0;
        for l in &mut self.layers {
            l.reset();
        }
    }

    pub fn cfg(&self) -> &TransformerConfig {
        &self.cfg
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// One forward step over `x: [t, d_model]`. Returns `[t, d_model]`.
    pub fn forward(
        &mut self,
        x: &Array2<f32>,
        cross_kv: Option<&CrossKvCache>,
    ) -> Result<Array2<f32>> {
        let t = x.dim().0;
        let positions: Vec<usize> = (self.seq_len..self.seq_len + t).collect();
        let (rope_cos, rope_sin) = match self.cfg.positional_embedding {
            PositionalEmbedding::Rope => {
                let head_dim = self.cfg.d_model / self.cfg.num_heads;
                let (cos, sin) = rope_tables(head_dim, self.cfg.max_period, &positions);
                (Some(cos), Some(sin))
            }
            PositionalEmbedding::None | PositionalEmbedding::Sin => (None, None),
        };
        let mut h = x.clone();
        for layer in &mut self.layers {
            h = layer.forward(
                &h,
                rope_cos.as_ref(),
                rope_sin.as_ref(),
                self.cfg.causal,
                cross_kv,
            )?;
        }
        self.seq_len += t;
        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PositionalEmbedding;
    use ndarray::Array;

    fn ones_layer(d_model: usize, num_heads: usize, ff: usize, with_cross: bool) -> LayerWeights {
        let hd = d_model / num_heads;
        let zero_cross = if with_cross {
            Some(CrossAttention {
                d_model,
                num_heads,
                head_dim: hd,
                w_q: Array::zeros((d_model, d_model)),
                w_k: Array::zeros((d_model, d_model)),
                w_v: Array::zeros((d_model, d_model)),
                w_o: Array::zeros((d_model, d_model)),
                pos_emb: false,
                pos_emb_scale: 1.0,
                pos_max_period: 10_000.0,
            })
        } else {
            None
        };
        LayerWeights {
            norm1_alpha: Array::ones(d_model),
            norm2_alpha: Array::ones(d_model),
            attn: AttnWeights {
                in_proj: Array::zeros((3 * d_model, d_model)),
                out_proj: Array::zeros((d_model, d_model)),
                num_heads,
                head_dim: hd,
            },
            gate_in: Array::zeros((2 * (ff / 2), d_model)),
            gate_out: Array::zeros((d_model, ff / 2)),
            cross_attn: zero_cross,
            norm_cross_alpha: if with_cross {
                Some(Array::ones(d_model))
            } else {
                None
            },
        }
    }

    fn zero_cfg(
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        ff: usize,
        cross: bool,
    ) -> TransformerConfig {
        TransformerConfig {
            d_model,
            num_heads,
            num_layers,
            dim_feedforward: ff,
            causal: true,
            context: 32,
            max_period: 10_000,
            positional_embedding: PositionalEmbedding::Rope,
            cross_attention: cross,
        }
    }

    #[test]
    fn forward_shape_matches_input() {
        let cfg = zero_cfg(8, 2, 2, 16, false);
        let layers = vec![ones_layer(8, 2, 16, false), ones_layer(8, 2, 16, false)];
        let mut t = StreamingTransformer::new(cfg, layers).unwrap();
        let x = Array::ones((1, 8));
        let y = t.forward(&x, None).unwrap();
        assert_eq!(y.dim(), (1, 8));
        assert_eq!(t.seq_len(), 1);
    }

    #[test]
    fn zero_weights_propagate_input_through_residual() {
        // With all zero projections, every layer is identity (x + 0 + 0 = x).
        let cfg = zero_cfg(8, 2, 3, 16, false);
        let layers = (0..3).map(|_| ones_layer(8, 2, 16, false)).collect();
        let mut t = StreamingTransformer::new(cfg, layers).unwrap();
        let x = Array::from_shape_fn((1, 8), |(_, j)| j as f32);
        let y = t.forward(&x, None).unwrap();
        for j in 0..8 {
            assert!(
                (y[[0, j]] - x[[0, j]]).abs() < 1e-3,
                "j={j} y={} x={}",
                y[[0, j]],
                x[[0, j]]
            );
        }
    }

    #[test]
    fn reset_state_clears_seq_len() {
        let cfg = zero_cfg(4, 2, 1, 8, false);
        let layers = vec![ones_layer(4, 2, 8, false)];
        let mut t = StreamingTransformer::new(cfg, layers).unwrap();
        let x = Array::ones((1, 4));
        t.forward(&x, None).unwrap();
        t.forward(&x, None).unwrap();
        assert_eq!(t.seq_len(), 2);
        t.reset_state();
        assert_eq!(t.seq_len(), 0);
    }

    #[test]
    fn cross_attention_no_op_when_weights_are_zero() {
        let cfg = zero_cfg(8, 2, 1, 16, true);
        let layers = vec![ones_layer(8, 2, 16, true)];
        let mut t = StreamingTransformer::new(cfg, layers).unwrap();
        let ctx = Array::ones((3, 8));
        let kv = t.layers[0]
            .w
            .cross_attn
            .as_ref()
            .unwrap()
            .prepare_kv(&ctx)
            .unwrap();
        let x = Array::from_shape_fn((1, 8), |(_, j)| j as f32);
        let y = t.forward(&x, Some(&kv)).unwrap();
        // Zero w_o on cross-attn → identity.
        for j in 0..8 {
            assert!((y[[0, j]] - x[[0, j]]).abs() < 1e-3);
        }
    }

    #[test]
    fn layer_count_mismatch_errors() {
        let cfg = zero_cfg(4, 1, 2, 8, false);
        let layers = vec![ones_layer(4, 1, 8, false)]; // only 1 vs cfg.num_layers=2
        assert!(StreamingTransformer::new(cfg, layers).is_err());
    }
}
