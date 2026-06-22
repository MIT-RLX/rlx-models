// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! `StreamingTransformer` — the 6-layer FlowLM backbone (also re-used for the
//! Mimi codec's 2-layer projected transformer, with layer-scale enabled and a
//! sliding-window context).

use anyhow::Result;
use ndarray::{Array1, Array2, Array3};

use crate::config::TransformerConfig;
use crate::ops::{apply_rope, layernorm, linear, rope_inv_freq};
use crate::weights::WeightFile;

#[derive(Debug, Clone)]
struct AttnLayer {
    in_proj_w: Array2<f32>,  // [3 * d, d]
    out_proj_w: Array2<f32>, // [d, d]
    norm1_w: Array1<f32>,
    norm1_b: Array1<f32>,
    norm2_w: Array1<f32>,
    norm2_b: Array1<f32>,
    linear1_w: Array2<f32>, // [ffn, d]
    linear2_w: Array2<f32>, // [d, ffn]
    layer_scale_1: Option<Array1<f32>>,
    layer_scale_2: Option<Array1<f32>>,
}

#[derive(Debug, Clone)]
pub struct StreamingTransformer {
    layers: Vec<AttnLayer>,
    inv_freq: Array1<f32>,
    num_heads: usize,
    head_dim: usize,
    d_model: usize,
    #[allow(dead_code)]
    ffn: usize,
    eps: f32,
    context: Option<usize>,
}

/// Per-layer KV cache (one entry per transformer layer). Shape: `[T_cached, H, D]`.
#[derive(Debug, Clone, Default)]
pub struct LayerKvCache {
    pub k: Array3<f32>,
    pub v: Array3<f32>,
}

/// One KV cache per layer + the absolute position offset.
#[derive(Debug, Clone, Default)]
pub struct KvCache {
    pub layers: Vec<LayerKvCache>,
    pub offset: usize,
}

impl KvCache {
    pub fn new(num_layers: usize, num_heads: usize, head_dim: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerKvCache {
                k: Array3::<f32>::zeros((0, num_heads, head_dim)),
                v: Array3::<f32>::zeros((0, num_heads, head_dim)),
            })
            .collect();
        Self { layers, offset: 0 }
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.k = Array3::<f32>::zeros((0, layer.k.shape()[1], layer.k.shape()[2]));
            layer.v = Array3::<f32>::zeros((0, layer.v.shape()[1], layer.v.shape()[2]));
        }
        self.offset = 0;
    }
}

impl StreamingTransformer {
    /// Load weights with the layout `<prefix>.layers.{i}.{...}`. The `<prefix>`
    /// is e.g. `flow_lm.transformer` or `mimi.decoder_transformer.transformer`.
    pub fn load(wf: &WeightFile, prefix: &str, cfg: &TransformerConfig, eps: f32) -> Result<Self> {
        let head_dim = cfg.d_model / cfg.num_heads;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let lp = format!("{prefix}.layers.{i}");
            let in_proj_w = wf.get_2d(&format!("{lp}.self_attn.in_proj.weight"))?;
            let out_proj_w = wf.get_2d(&format!("{lp}.self_attn.out_proj.weight"))?;
            let norm1_w = wf.get_1d(&format!("{lp}.norm1.weight"))?;
            let norm1_b = wf.get_1d(&format!("{lp}.norm1.bias"))?;
            let norm2_w = wf.get_1d(&format!("{lp}.norm2.weight"))?;
            let norm2_b = wf.get_1d(&format!("{lp}.norm2.bias"))?;
            let linear1_w = wf.get_2d(&format!("{lp}.linear1.weight"))?;
            let linear2_w = wf.get_2d(&format!("{lp}.linear2.weight"))?;
            let layer_scale_1 = wf.opt_1d(&format!("{lp}.layer_scale_1.scale"))?;
            let layer_scale_2 = wf.opt_1d(&format!("{lp}.layer_scale_2.scale"))?;
            layers.push(AttnLayer {
                in_proj_w,
                out_proj_w,
                norm1_w,
                norm1_b,
                norm2_w,
                norm2_b,
                linear1_w,
                linear2_w,
                layer_scale_1,
                layer_scale_2,
            });
        }
        Ok(Self {
            layers,
            inv_freq: rope_inv_freq(head_dim, cfg.max_period),
            num_heads: cfg.num_heads,
            head_dim,
            d_model: cfg.d_model,
            ffn: cfg.dim_feedforward,
            eps,
            context: cfg.context,
        })
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn make_cache(&self) -> KvCache {
        KvCache::new(self.layers.len(), self.num_heads, self.head_dim)
    }

    /// Forward `T` new tokens through every layer, updating the cache in place.
    /// `x: [T, d_model]` → `[T, d_model]`. RoPE positions are
    /// `[cache.offset .. cache.offset + T]` for both Q and K-new.
    pub fn forward(&self, mut x: Array2<f32>, cache: &mut KvCache) -> Array2<f32> {
        let t = x.shape()[0];
        if t == 0 {
            return x;
        }
        let positions_new: Vec<i64> = (cache.offset..cache.offset + t).map(|p| p as i64).collect();

        for (li, layer) in self.layers.iter().enumerate() {
            // ── self-attn block ──
            let x_in = x.clone();
            let normed = layernorm(
                x.view(),
                Some(layer.norm1_w.view()),
                Some(layer.norm1_b.view()),
                self.eps,
            );

            // QKV projection: [T, 3d] → split into Q/K/V each [T, H, D].
            let qkv = linear(normed.view(), layer.in_proj_w.view(), None);
            let mut q = Array3::<f32>::zeros((t, self.num_heads, self.head_dim));
            let mut k_new = Array3::<f32>::zeros((t, self.num_heads, self.head_dim));
            let mut v_new = Array3::<f32>::zeros((t, self.num_heads, self.head_dim));
            for ti in 0..t {
                for hi in 0..self.num_heads {
                    for di in 0..self.head_dim {
                        let base = hi * self.head_dim + di;
                        q[[ti, hi, di]] = qkv[[ti, base]];
                        k_new[[ti, hi, di]] = qkv[[ti, self.d_model + base]];
                        v_new[[ti, hi, di]] = qkv[[ti, 2 * self.d_model + base]];
                    }
                }
            }
            apply_rope(&mut q, &positions_new, self.inv_freq.view());
            apply_rope(&mut k_new, &positions_new, self.inv_freq.view());

            // Append new K/V to the layer's cache.
            let mut k_cache = cache.layers[li].k.clone();
            let mut v_cache = cache.layers[li].v.clone();
            let cached_t = k_cache.shape()[0];
            let new_t = cached_t + t;
            let mut k_full = Array3::<f32>::zeros((new_t, self.num_heads, self.head_dim));
            let mut v_full = Array3::<f32>::zeros((new_t, self.num_heads, self.head_dim));
            for ti in 0..cached_t {
                for hi in 0..self.num_heads {
                    for di in 0..self.head_dim {
                        k_full[[ti, hi, di]] = k_cache[[ti, hi, di]];
                        v_full[[ti, hi, di]] = v_cache[[ti, hi, di]];
                    }
                }
            }
            for ti in 0..t {
                for hi in 0..self.num_heads {
                    for di in 0..self.head_dim {
                        k_full[[cached_t + ti, hi, di]] = k_new[[ti, hi, di]];
                        v_full[[cached_t + ti, hi, di]] = v_new[[ti, hi, di]];
                    }
                }
            }
            cache.layers[li].k = k_full.clone();
            cache.layers[li].v = v_full.clone();
            k_cache = k_full;
            v_cache = v_full;

            // ── attention: q [T, H, D] × k/v [T_kv, H, D] ──
            let scale = 1.0 / (self.head_dim as f32).sqrt();
            let t_kv = k_cache.shape()[0];
            // Per-(t, h) attention output.
            let mut attn_out = Array3::<f32>::zeros((t, self.num_heads, self.head_dim));
            // Causal mask + optional sliding window. pos_q for new token ti is
            // `cache.offset + ti`; pos_k for cached token is its absolute index 0..T_kv.
            for ti in 0..t {
                let q_pos = (cache.offset + ti) as i64;
                for hi in 0..self.num_heads {
                    // Compute scores for this (ti, hi) across all kv positions.
                    let mut scores = vec![f32::NEG_INFINITY; t_kv];
                    let mut max_score = f32::NEG_INFINITY;
                    for kj in 0..t_kv {
                        let k_pos = kj as i64;
                        let delta = q_pos - k_pos;
                        if delta < 0 {
                            continue;
                        }
                        if let Some(ctx) = self.context {
                            if (delta as usize) >= ctx {
                                continue;
                            }
                        }
                        let mut s = 0.0;
                        for di in 0..self.head_dim {
                            s += q[[ti, hi, di]] * k_cache[[kj, hi, di]];
                        }
                        s *= scale;
                        scores[kj] = s;
                        if s > max_score {
                            max_score = s;
                        }
                    }
                    if max_score == f32::NEG_INFINITY {
                        continue;
                    }
                    // Softmax + weighted sum.
                    let mut denom = 0.0;
                    for s in scores.iter_mut() {
                        if *s == f32::NEG_INFINITY {
                            *s = 0.0;
                        } else {
                            *s = (*s - max_score).exp();
                            denom += *s;
                        }
                    }
                    let inv_denom = 1.0 / denom;
                    for kj in 0..t_kv {
                        let w = scores[kj] * inv_denom;
                        if w == 0.0 {
                            continue;
                        }
                        for di in 0..self.head_dim {
                            attn_out[[ti, hi, di]] += w * v_cache[[kj, hi, di]];
                        }
                    }
                }
            }

            // Re-pack [T, H, D] → [T, d_model].
            let mut attn_flat = Array2::<f32>::zeros((t, self.d_model));
            for ti in 0..t {
                for hi in 0..self.num_heads {
                    for di in 0..self.head_dim {
                        attn_flat[[ti, hi * self.head_dim + di]] = attn_out[[ti, hi, di]];
                    }
                }
            }
            let mut update = linear(attn_flat.view(), layer.out_proj_w.view(), None);
            if let Some(ls) = &layer.layer_scale_1 {
                apply_layer_scale(&mut update, ls.view());
            }
            // Residual.
            for ti in 0..t {
                for ci in 0..self.d_model {
                    x[[ti, ci]] = x_in[[ti, ci]] + update[[ti, ci]];
                }
            }

            // ── FFN block ──
            let x_in2 = x.clone();
            let normed2 = layernorm(
                x.view(),
                Some(layer.norm2_w.view()),
                Some(layer.norm2_b.view()),
                self.eps,
            );
            let mut h = linear(normed2.view(), layer.linear1_w.view(), None);
            for v in h.iter_mut() {
                *v = crate::ops::gelu_scalar(*v);
            }
            let mut update2 = linear(h.view(), layer.linear2_w.view(), None);
            if let Some(ls) = &layer.layer_scale_2 {
                apply_layer_scale(&mut update2, ls.view());
            }
            for ti in 0..t {
                for ci in 0..self.d_model {
                    x[[ti, ci]] = x_in2[[ti, ci]] + update2[[ti, ci]];
                }
            }
        }

        cache.offset += t;
        x
    }
}

fn apply_layer_scale(x: &mut Array2<f32>, scale: ndarray::ArrayView1<f32>) {
    let (t, c) = x.dim();
    debug_assert_eq!(c, scale.len());
    for ti in 0..t {
        for ci in 0..c {
            x[[ti, ci]] *= scale[ci];
        }
    }
}
