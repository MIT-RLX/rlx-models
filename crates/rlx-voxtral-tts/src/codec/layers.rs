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

//! Shared codec conv + transformer blocks (encoder and decoder).

use crate::config::CodecArgs;
use crate::math::{conv_transpose1d, conv1d, linear2, rms_norm, swiglu};
use anyhow::{Context, Result, ensure};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};
use std::collections::HashMap;

pub(crate) const CODEC_NORM_EPS: f32 = 1e-2;

pub(crate) enum CodecConvBlock {
    Forward {
        weight: Array3<f32>,
        stride: usize,
        pad_left: usize,
    },
    Transpose {
        weight: Array3<f32>,
        stride: usize,
        trim_left: usize,
        trim_right: usize,
    },
}

pub(crate) struct CodecTransformer {
    pub window: usize,
    pub layers: Vec<CodecLayer>,
}

pub(crate) struct CodecLayer {
    pub wq: Array2<f32>,
    pub wk: Array2<f32>,
    pub wv: Array2<f32>,
    pub wo: Array2<f32>,
    pub q_norm: Array1<f32>,
    pub k_norm: Array1<f32>,
    pub attn_norm: Array1<f32>,
    pub ffn_norm: Array1<f32>,
    pub w1: Array2<f32>,
    pub w2: Array2<f32>,
    pub w3: Array2<f32>,
    pub attn_scale: Array1<f32>,
    pub ffn_scale: Array1<f32>,
    pub alibi_slopes: Array1<f32>,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl CodecLayer {
    pub(crate) fn forward(&self, x: ArrayView2<f32>, window: usize) -> Result<Array2<f32>> {
        let (t, d) = x.dim();
        let h = rms_norm(x, self.attn_norm.view(), CODEC_NORM_EPS);
        let q = linear2(h.view(), self.wq.view(), None);
        let k = linear2(h.view(), self.wk.view(), None);
        let v = linear2(h.view(), self.wv.view(), None);
        let qn = rms_norm(q.view(), self.q_norm.view(), self.head_dim as f32 * 1e-6);
        let kn = rms_norm(k.view(), self.k_norm.view(), self.head_dim as f32 * 1e-6);
        let attn = alibi_attention(
            qn.view(),
            kn.view(),
            v.view(),
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            window,
            self.alibi_slopes.view(),
        )?;
        let mut attn_out = linear2(attn.view(), self.wo.view(), None);
        for i in 0..t {
            for j in 0..d {
                attn_out[[i, j]] = x[[i, j]] + self.attn_scale[j] * attn_out[[i, j]];
            }
        }
        let h2 = rms_norm(attn_out.view(), self.ffn_norm.view(), CODEC_NORM_EPS);
        let w1 = linear2(h2.view(), self.w1.view(), None);
        let w3 = linear2(h2.view(), self.w3.view(), None);
        let ff = swiglu(w1.view(), w3.view(), &self.w2);
        let mut out = attn_out;
        for i in 0..t {
            for j in 0..d {
                out[[i, j]] += self.ffn_scale[j] * ff[[i, j]];
            }
        }
        Ok(out)
    }
}

pub(crate) fn run_transformer(x: &Array2<f32>, tr: &CodecTransformer) -> Result<Array2<f32>> {
    let (d, t) = x.dim();
    let mut h = Array2::<f32>::zeros((t, d));
    for ti in 0..t {
        for di in 0..d {
            h[[ti, di]] = x[[di, ti]];
        }
    }
    for layer in &tr.layers {
        h = layer.forward(h.view(), tr.window)?;
    }
    let mut out = Array2::<f32>::zeros((d, t));
    for ti in 0..t {
        for di in 0..d {
            out[[di, ti]] = h[[ti, di]];
        }
    }
    Ok(out)
}

pub(crate) fn run_conv(x: &Array2<f32>, conv: &CodecConvBlock) -> Array2<f32> {
    match conv {
        CodecConvBlock::Forward {
            weight,
            stride,
            pad_left,
        } => conv1d(x.view(), weight.view(), *stride, *pad_left),
        CodecConvBlock::Transpose {
            weight,
            stride,
            trim_left,
            trim_right,
        } => conv_transpose1d(x.view(), weight.view(), *stride, *trim_left, *trim_right),
    }
}

pub(crate) fn compute_semantic_embedding(sum: &Array2<f32>, usage: &Array1<f32>) -> Array2<f32> {
    let (v, d) = sum.dim();
    let mut out = Array2::<f32>::zeros((v, d));
    for i in 0..v {
        let denom = usage[i].max(1e-5);
        for j in 0..d {
            out[[i, j]] = sum[[i, j]] / denom;
        }
    }
    out
}

pub(crate) fn rescale_fsq(code: u32, levels: usize) -> f32 {
    let q = code as f32;
    (q * 2.0 / (levels as f32 - 1.0)) - 1.0
}

pub(crate) fn take2d(
    map: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    key: &str,
) -> Result<Array2<f32>> {
    let (data, shape) = map
        .get(key)
        .with_context(|| format!("missing tensor {key}"))?;
    ensure!(shape.len() == 2, "{key}: expected rank 2");
    Array2::from_shape_vec((shape[0], shape[1]), data.clone()).with_context(|| key.to_string())
}

pub(crate) fn take1d(
    map: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    key: &str,
) -> Result<Array1<f32>> {
    let (data, shape) = map
        .get(key)
        .with_context(|| format!("missing tensor {key}"))?;
    ensure!(shape.len() == 1, "{key}: expected rank 1");
    Array1::from_shape_vec(shape[0], data.clone()).with_context(|| key.to_string())
}

pub(crate) fn take_conv(
    map: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
) -> Result<Array3<f32>> {
    reconstruct_conv(map, prefix, false)
}

pub(crate) fn take_conv_transpose(
    map: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
) -> Result<Array3<f32>> {
    reconstruct_conv(map, prefix, true)
}

fn reconstruct_conv(
    map: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
    _transpose: bool,
) -> Result<Array3<f32>> {
    let g_key = format!("{prefix}.conv.parametrizations.weight.original0");
    let v_key = format!("{prefix}.conv.parametrizations.weight.original1");
    if let (Some((g, _gs)), Some((v, vs))) = (map.get(&g_key), map.get(&v_key)) {
        let shape = [vs[0], vs[1], vs[2]];
        let mut w = vec![0f32; v.len()];
        let out_ch = vs[0];
        let fan_in = vs[1] * vs[2];
        for oc in 0..out_ch {
            let mut norm_sq = 0f32;
            for i in 0..fan_in {
                let idx = oc * fan_in + i;
                norm_sq += v[idx] * v[idx];
            }
            let scale = g[oc] / (norm_sq.sqrt() + 1e-12);
            for i in 0..fan_in {
                let idx = oc * fan_in + i;
                w[idx] = v[idx] * scale;
            }
        }
        return Array3::from_shape_vec((shape[0], shape[1], shape[2]), w)
            .with_context(|| prefix.to_string());
    }
    let w_key = format!("{prefix}.conv.weight");
    let (data, shape) = map
        .get(&w_key)
        .with_context(|| format!("missing conv weight {prefix}"))?;
    ensure!(shape.len() == 3, "{w_key}: expected rank 3");
    Array3::from_shape_vec((shape[0], shape[1], shape[2]), data.clone())
        .with_context(|| w_key.clone())
}

pub(crate) fn load_codec_layer(
    map: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
    cfg: &CodecArgs,
) -> Result<CodecLayer> {
    let tp = |s: &str| format!("{prefix}.{s}");
    Ok(CodecLayer {
        wq: take2d(map, &tp("attention.wq.weight"))?,
        wk: take2d(map, &tp("attention.wk.weight"))?,
        wv: take2d(map, &tp("attention.wv.weight"))?,
        wo: take2d(map, &tp("attention.wo.weight"))?,
        q_norm: take1d(map, &tp("attention.q_norm.weight"))?,
        k_norm: take1d(map, &tp("attention.k_norm.weight"))?,
        attn_norm: take1d(map, &tp("attention_norm.weight"))?,
        ffn_norm: take1d(map, &tp("ffn_norm.weight"))?,
        w1: {
            let w = take2d(map, &tp("feed_forward.w1.weight"))?;
            ensure!(
                w.dim() == (cfg.hidden_dim, cfg.dim),
                "{prefix} feed_forward.w1: expected [{}, {}], got {:?}",
                cfg.hidden_dim,
                cfg.dim,
                w.dim()
            );
            w
        },
        w2: {
            let w = take2d(map, &tp("feed_forward.w2.weight"))?;
            ensure!(
                w.dim() == (cfg.dim, cfg.hidden_dim),
                "{prefix} feed_forward.w2: expected [{}, {}], got {:?}",
                cfg.dim,
                cfg.hidden_dim,
                w.dim()
            );
            w
        },
        w3: {
            let w = take2d(map, &tp("feed_forward.w3.weight"))?;
            ensure!(
                w.dim() == (cfg.hidden_dim, cfg.dim),
                "{prefix} feed_forward.w3: expected [{}, {}], got {:?}",
                cfg.hidden_dim,
                cfg.dim,
                w.dim()
            );
            w
        },
        attn_scale: take1d(map, &tp("attention_scale"))?,
        ffn_scale: take1d(map, &tp("ffn_scale"))?,
        alibi_slopes: alibi_slopes(cfg.n_heads),
        n_heads: cfg.n_heads,
        n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim,
    })
}

fn alibi_slopes(n_heads: usize) -> Array1<f32> {
    fn pow2(n: usize) -> Vec<f32> {
        let r = 2f32.powf(-8.0 / n as f32);
        (0..n).map(|i| r.powi(i as i32)).collect()
    }
    let slopes = if n_heads.is_power_of_two() {
        pow2(n_heads)
    } else {
        let m = 1usize << (n_heads.ilog2());
        let mut s = pow2(m);
        let rest: Vec<f32> = pow2(2 * m)
            .into_iter()
            .step_by(2)
            .take(n_heads - m)
            .collect();
        s.extend(rest);
        s
    };
    Array1::from_vec(slopes)
}

fn alibi_attention(
    q: ArrayView2<f32>,
    k: ArrayView2<f32>,
    v: ArrayView2<f32>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: usize,
    slopes: ArrayView1<f32>,
) -> Result<Array2<f32>> {
    let (t, _) = q.dim();
    let repeats = n_heads / n_kv_heads;
    let mut out = Array2::<f32>::zeros((t, n_heads * head_dim));
    for hi in 0..n_heads {
        let kv_h = hi / repeats;
        for qi in 0..t {
            let mut weights = vec![0f32; t];
            let mut max_w = f32::NEG_INFINITY;
            for ki in 0..t {
                if qi < ki {
                    continue;
                }
                if qi.saturating_sub(ki) > window {
                    continue;
                }
                let mut dot = 0f32;
                for di in 0..head_dim {
                    dot += q[[qi, hi * head_dim + di]] * k[[ki, kv_h * head_dim + di]];
                }
                dot /= (head_dim as f32).sqrt();
                dot += slopes[hi] * (ki as f32 - qi as f32);
                weights[ki] = dot;
                max_w = max_w.max(dot);
            }
            let mut sum = 0f32;
            for ki in 0..t {
                if weights[ki].is_finite() && weights[ki] > f32::NEG_INFINITY / 2.0 {
                    weights[ki] = (weights[ki] - max_w).exp();
                    sum += weights[ki];
                } else {
                    weights[ki] = 0.0;
                }
            }
            if sum > 0.0 {
                for w in weights.iter_mut() {
                    *w /= sum;
                }
            }
            for di in 0..head_dim {
                let mut acc = 0f32;
                for ki in 0..t {
                    acc += weights[ki] * v[[ki, kv_h * head_dim + di]];
                }
                out[[qi, hi * head_dim + di]] = acc;
            }
        }
    }
    Ok(out)
}
