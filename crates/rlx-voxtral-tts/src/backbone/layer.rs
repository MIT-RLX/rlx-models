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

//! Ministral decoder layer (GQA + RoPE + SwiGLU).

use crate::config::TextConfig;
use crate::math::{rms_norm, silu};
use anyhow::{Context, Result, ensure};
use ndarray::{Array2, ArrayView2};
use std::collections::HashMap;

pub struct DecoderLayer {
    wq: Array2<f32>,
    wk: Array2<f32>,
    wv: Array2<f32>,
    wo: Array2<f32>,
    lora: Option<crate::lora::LayerLora>,
    lora_scale: f32,
    attn_norm: Array1Like,
    ffn_norm: Array1Like,
    w1: Array2<f32>,
    w2: Array2<f32>,
    w3: Array2<f32>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

type Array1Like = ndarray::Array1<f32>;

pub struct LayerKv {
    pub k: Array2<f32>,
    pub v: Array2<f32>,
}

impl DecoderLayer {
    pub fn load(
        map: &HashMap<String, (Vec<f32>, Vec<usize>)>,
        prefix: &str,
        cfg: &TextConfig,
    ) -> Result<Self> {
        let tp = |s: &str| format!("{prefix}.{s}");
        Ok(Self {
            wq: take2d(map, &tp("attention.wq.weight"))?,
            wk: take2d(map, &tp("attention.wk.weight"))?,
            wv: take2d(map, &tp("attention.wv.weight"))?,
            wo: take2d(map, &tp("attention.wo.weight"))?,
            lora: None,
            lora_scale: 1.0,
            attn_norm: take1d(map, &tp("attention_norm.weight"))?,
            ffn_norm: take1d(map, &tp("ffn_norm.weight"))?,
            w1: take2d(map, &tp("feed_forward.w1.weight"))?,
            w2: take2d(map, &tp("feed_forward.w2.weight"))?,
            w3: take2d(map, &tp("feed_forward.w3.weight"))?,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        })
    }

    pub fn set_lora(&mut self, lora: crate::lora::LayerLora, scale: f32) {
        self.lora = Some(lora);
        self.lora_scale = scale;
    }

    pub fn forward(
        &self,
        x: ArrayView2<f32>,
        cos: &[f32],
        sin: &[f32],
        start_pos: usize,
        kv: &mut LayerKv,
    ) -> Result<Array2<f32>> {
        let eps = 1e-5f32;
        let h = rms_norm(x, self.attn_norm.view(), eps);
        let scale = self.lora_scale;
        let lora = self.lora.as_ref();
        let mut q =
            crate::lora::apply_lora_linear(&h, &self.wq, lora.and_then(|l| l.wq.as_ref()), scale);
        let mut k =
            crate::lora::apply_lora_linear(&h, &self.wk, lora.and_then(|l| l.wk.as_ref()), scale);
        let v =
            crate::lora::apply_lora_linear(&h, &self.wv, lora.and_then(|l| l.wv.as_ref()), scale);
        super::rope::apply_rope_qk(
            &mut q,
            &mut k,
            cos,
            sin,
            start_pos,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
        );

        if kv.k.nrows() == 0 {
            kv.k = k;
            kv.v = v;
        } else {
            let (t_new, d) = k.dim();
            let (t_old, _) = kv.k.dim();
            let mut k_cat = Array2::<f32>::zeros((t_old + t_new, d));
            let mut v_cat = Array2::<f32>::zeros((t_old + t_new, d));
            for i in 0..t_old {
                for j in 0..d {
                    k_cat[[i, j]] = kv.k[[i, j]];
                    v_cat[[i, j]] = kv.v[[i, j]];
                }
            }
            for i in 0..t_new {
                for j in 0..d {
                    k_cat[[t_old + i, j]] = k[[i, j]];
                    v_cat[[t_old + i, j]] = v[[i, j]];
                }
            }
            kv.k = k_cat;
            kv.v = v_cat;
        }

        let attn = gqa_attention(
            q.view(),
            kv.k.view(),
            kv.v.view(),
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
        );
        let attn_out = crate::lora::apply_lora_linear(
            &attn,
            &self.wo,
            lora.and_then(|l| l.wo.as_ref()),
            scale,
        );
        let mut out = x.to_owned() + attn_out;
        let h2 = rms_norm(out.view(), self.ffn_norm.view(), eps);
        let w1 =
            crate::lora::apply_lora_linear(&h2, &self.w1, lora.and_then(|l| l.w1.as_ref()), scale);
        let w3 =
            crate::lora::apply_lora_linear(&h2, &self.w3, lora.and_then(|l| l.w3.as_ref()), scale);
        let swiglu = &silu(w1.view()) * w3;
        let ff = crate::lora::apply_lora_linear(
            &swiglu,
            &self.w2,
            lora.and_then(|l| l.w2.as_ref()),
            scale,
        );
        out = out + ff;
        Ok(out)
    }
}

fn gqa_attention(
    q: ArrayView2<f32>,
    k: ArrayView2<f32>,
    v: ArrayView2<f32>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Array2<f32> {
    let (t_q, _) = q.dim();
    let t_k = k.dim().0;
    let repeats = n_heads / n_kv_heads;
    let mut out = Array2::<f32>::zeros((t_q, n_heads * head_dim));
    for qi in 0..t_q {
        for hi in 0..n_heads {
            let kv_h = hi / repeats;
            let mut max_w = f32::NEG_INFINITY;
            let mut weights = vec![0f32; t_k];
            for ki in 0..t_k {
                if ki > qi + (t_k - t_q) {
                    continue;
                }
                let mut dot = 0f32;
                for di in 0..head_dim {
                    dot += q[[qi, hi * head_dim + di]] * k[[ki, kv_h * head_dim + di]];
                }
                dot /= (head_dim as f32).sqrt();
                weights[ki] = dot;
                max_w = max_w.max(dot);
            }
            let mut sum = 0f32;
            for w in weights.iter_mut() {
                *w = (*w - max_w).exp();
                sum += *w;
            }
            for w in weights.iter_mut() {
                *w /= sum.max(1e-12);
            }
            for di in 0..head_dim {
                let mut acc = 0f32;
                for ki in 0..t_k {
                    acc += weights[ki] * v[[ki, kv_h * head_dim + di]];
                }
                out[[qi, hi * head_dim + di]] = acc;
            }
        }
    }
    out
}

fn take2d(map: &HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str) -> Result<Array2<f32>> {
    let (data, shape) = map.get(key).with_context(|| format!("missing {key}"))?;
    ensure!(shape.len() == 2);
    Array2::from_shape_vec((shape[0], shape[1]), data.clone()).with_context(|| key.to_string())
}

fn take1d(map: &HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str) -> Result<Array1Like> {
    let (data, shape) = map.get(key).with_context(|| format!("missing {key}"))?;
    ensure!(shape.len() == 1);
    Array1Like::from_shape_vec(shape[0], data.clone()).with_context(|| key.to_string())
}
