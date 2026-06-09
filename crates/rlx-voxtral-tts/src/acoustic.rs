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

//! Flow-matching acoustic transformer (vLLM-Omni `FlowMatchingAudioTransformer`).

use crate::acoustic_compiled::CompiledAcousticStack;
use crate::config::AcousticTransformerArgs;
use crate::math::{linear2, rms_norm, silu};
use crate::tokens::{AUDIO_TOKEN_OFFSET, EMPTY_AUDIO, END_AUDIO};
use anyhow::{Context, Result, ensure};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::HashMap;

pub const FM_SEQ: usize = 3;

pub struct AcousticTransformer {
    args: AcousticTransformerArgs,
    n_acoustic_codebook: usize,
    semantic_vocab: usize,
    semantic_limit: usize,
    semantic_out: Array2<f32>,
    acoustic_out: Array2<f32>,
    input_proj: Array2<f32>,
    time_proj: Array2<f32>,
    llm_proj: Array2<f32>,
    time_inv_freq: Array1<f32>,
    layers: Vec<AcousticLayer>,
    norm: Array1<f32>,
    n_steps: usize,
}

struct AcousticLayer {
    wq: Array2<f32>,
    wk: Array2<f32>,
    wv: Array2<f32>,
    wo: Array2<f32>,
    attn_norm: Array1<f32>,
    ffn_norm: Array1<f32>,
    w1: Array2<f32>,
    w2: Array2<f32>,
    w3: Array2<f32>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl AcousticTransformer {
    pub fn from_tensors(
        prefix: &str,
        tensors: &HashMap<String, (Vec<f32>, Vec<usize>)>,
        args: &AcousticTransformerArgs,
        n_acoustic_codebook: usize,
        semantic_codebook_size: usize,
    ) -> Result<Self> {
        let tp = |s: &str| format!("{prefix}{s}");
        let semantic_out = take2d(tensors, &tp("semantic_codebook_output.weight"))?;
        Ok(Self {
            args: args.clone(),
            n_acoustic_codebook,
            semantic_vocab: semantic_out.dim().0,
            semantic_limit: 2 + semantic_codebook_size,
            semantic_out,
            input_proj: take2d(tensors, &tp("input_projection.weight"))?,
            time_proj: take2d(tensors, &tp("time_projection.weight"))?,
            llm_proj: take2d(tensors, &tp("llm_projection.weight"))?,
            acoustic_out: take2d(tensors, &tp("acoustic_codebook_output.weight"))?,
            time_inv_freq: time_inv_freq(args.dim, args.time_theta()),
            norm: take1d(tensors, &tp("norm.weight"))?,
            n_steps: args
                .n_decoding_steps
                .unwrap_or(crate::tokens::DEFAULT_EULER_STEPS),
            layers: (0..args.n_layers)
                .map(|i| load_layer(tensors, &tp(&format!("layers.{i}")), args))
                .collect::<Result<_>>()?,
        })
    }

    /// Predict `[37]` vLLM-layout codes (semantic raw, acoustic +2 offset).
    pub fn predict_frame(
        &self,
        llm_hidden: ArrayView1<f32>,
        cfg_alpha: f32,
        seed: u64,
        frame_index: usize,
    ) -> Result<Vec<u32>> {
        let h = llm_hidden.to_owned().insert_axis(ndarray::Axis(0));
        let mut sem = linear2(h.view(), self.semantic_out.view(), None)
            .row(0)
            .to_owned();
        mask_semantic_logits(&mut sem, self.semantic_vocab, self.semantic_limit);
        let semantic_id = argmax1(sem.view());
        if semantic_id == END_AUDIO as usize {
            let mut codes = vec![0u32; 37];
            codes[0] = END_AUDIO;
            for slot in codes.iter_mut().skip(1) {
                *slot = AUDIO_TOKEN_OFFSET;
            }
            return Ok(codes);
        }

        let mut x0 = vec![0f32; self.n_acoustic_codebook];
        crate::rng::fill_standard_normal(&mut x0, crate::rng::frame_seed(seed, frame_index));
        let mut sampled = Array2::<f32>::zeros((1, self.n_acoustic_codebook));
        for (i, v) in x0.iter().enumerate() {
            sampled[[0, i]] = *v * self.args.sigma as f32;
        }
        let timesteps: Vec<f32> = (0..=self.n_steps)
            .map(|i| i as f32 / self.n_steps as f32)
            .collect();
        for i in 0..self.n_steps {
            let t = timesteps[i];
            let dt = timesteps[i + 1] - timesteps[i];
            let t_emb = time_embed(t, self.time_inv_freq.view());
            let v_cond = self.predict_velocity(sampled.view(), h.view(), t_emb.view())?;
            let v_uncond = self.predict_velocity(
                sampled.view(),
                Array2::<f32>::zeros((1, self.args.input_dim)).view(),
                t_emb.view(),
            )?;
            let v = cfg_alpha * &v_cond + (1.0 - cfg_alpha) * &v_uncond;
            sampled = &sampled + &(&v * dt);
        }
        let mut codes = vec![0u32; 37];
        codes[0] = semantic_id as u32;
        let levels = self.args.acoustic_levels();
        for ai in 0..self.n_acoustic_codebook {
            let v = sampled[[0, ai]].clamp(-1.0, 1.0);
            let scaled = ((v + 1.0) / 2.0) * (levels as f32 - 1.0);
            codes[1 + ai] = scaled.round() as u32 + AUDIO_TOKEN_OFFSET;
        }
        Ok(codes)
    }

    pub fn predict_frame_compiled(
        &self,
        llm_hidden: ArrayView1<f32>,
        cfg_alpha: f32,
        seed: u64,
        frame_index: usize,
        stack: &mut CompiledAcousticStack,
    ) -> Result<Vec<u32>> {
        let h = llm_hidden.to_owned().insert_axis(ndarray::Axis(0));
        let mut sem = linear2(h.view(), self.semantic_out.view(), None)
            .row(0)
            .to_owned();
        mask_semantic_logits(&mut sem, self.semantic_vocab, self.semantic_limit);
        let semantic_id = argmax1(sem.view());
        if semantic_id == END_AUDIO as usize {
            let mut codes = vec![0u32; 37];
            codes[0] = END_AUDIO;
            for slot in codes.iter_mut().skip(1) {
                *slot = AUDIO_TOKEN_OFFSET;
            }
            return Ok(codes);
        }

        let mut x0 = vec![0f32; self.n_acoustic_codebook];
        crate::rng::fill_standard_normal(&mut x0, crate::rng::frame_seed(seed, frame_index));
        let mut sampled = Array2::<f32>::zeros((1, self.n_acoustic_codebook));
        for (i, v) in x0.iter().enumerate() {
            sampled[[0, i]] = *v * self.args.sigma as f32;
        }
        let timesteps: Vec<f32> = (0..=self.n_steps)
            .map(|i| i as f32 / self.n_steps as f32)
            .collect();
        for i in 0..self.n_steps {
            let t = timesteps[i];
            let dt = timesteps[i + 1] - timesteps[i];
            let t_emb = time_embed(t, self.time_inv_freq.view());
            let v_cond =
                self.predict_velocity_compiled(sampled.view(), h.view(), t_emb.view(), stack)?;
            let v_uncond = self.predict_velocity_compiled(
                sampled.view(),
                Array2::<f32>::zeros((1, self.args.input_dim)).view(),
                t_emb.view(),
                stack,
            )?;
            let v = cfg_alpha * &v_cond + (1.0 - cfg_alpha) * &v_uncond;
            sampled = &sampled + &(&v * dt);
        }
        let mut codes = vec![0u32; 37];
        codes[0] = semantic_id as u32;
        let levels = self.args.acoustic_levels();
        for ai in 0..self.n_acoustic_codebook {
            let v = sampled[[0, ai]].clamp(-1.0, 1.0);
            let scaled = ((v + 1.0) / 2.0) * (levels as f32 - 1.0);
            codes[1 + ai] = scaled.round() as u32 + AUDIO_TOKEN_OFFSET;
        }
        Ok(codes)
    }

    fn predict_velocity_compiled(
        &self,
        x_t: ArrayView2<f32>,
        llm: ArrayView2<f32>,
        t_emb: ArrayView2<f32>,
        stack: &mut CompiledAcousticStack,
    ) -> Result<Array2<f32>> {
        let h = build_fm_tokens(
            x_t,
            llm,
            t_emb,
            &self.input_proj,
            &self.time_proj,
            &self.llm_proj,
        )?;
        let tokens = flatten_fm_tokens(&h);
        ensure!(
            tokens.len() == stack.input_len(),
            "FM token flat len {} != stack {}",
            tokens.len(),
            stack.input_len()
        );
        let vel = stack.forward(&tokens)?;
        ensure!(
            vel.len() == self.n_acoustic_codebook,
            "velocity len {} != n_acoustic_codebook {}",
            vel.len(),
            self.n_acoustic_codebook
        );
        Ok(Array2::from_shape_vec((1, self.n_acoustic_codebook), vel)?)
    }

    fn predict_velocity(
        &self,
        x_t: ArrayView2<f32>,
        llm: ArrayView2<f32>,
        t_emb: ArrayView2<f32>,
    ) -> Result<Array2<f32>> {
        let mut h = build_fm_tokens(
            x_t,
            llm,
            t_emb,
            &self.input_proj,
            &self.time_proj,
            &self.llm_proj,
        )?;
        for layer in &self.layers {
            h = layer.forward(h.view(), self.args.norm_eps as f32)?;
        }
        h = rms_norm(h.view(), self.norm.view(), self.args.norm_eps as f32);
        let v = linear2(
            h.row(0).insert_axis(ndarray::Axis(0)),
            self.acoustic_out.view(),
            None,
        );
        Ok(v)
    }
}

impl AcousticTransformerArgs {
    pub fn acoustic_levels(&self) -> usize {
        21
    }
}

impl AcousticLayer {
    fn forward(&self, x: ArrayView2<f32>, norm_eps: f32) -> Result<Array2<f32>> {
        let h = rms_norm(x, self.attn_norm.view(), norm_eps);
        let q = linear2(h.view(), self.wq.view(), None);
        let k = linear2(h.view(), self.wk.view(), None);
        let v = linear2(h.view(), self.wv.view(), None);
        let attn = mha(
            q.view(),
            k.view(),
            v.view(),
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
        );
        let attn_out = linear2(attn.view(), self.wo.view(), None);
        let mut out = x.to_owned() + attn_out;
        let h2 = rms_norm(out.view(), self.ffn_norm.view(), norm_eps);
        let w1 = linear2(h2.view(), self.w1.view(), None);
        let w3 = linear2(h2.view(), self.w3.view(), None);
        let swiglu = &silu(w1.view()) * w3;
        let ff = linear2(swiglu.view(), self.w2.view(), None);
        out = out + ff;
        Ok(out)
    }
}

fn flatten_fm_tokens(tokens: &Array2<f32>) -> Vec<f32> {
    tokens.iter().copied().collect()
}

fn build_fm_tokens(
    x_t: ArrayView2<f32>,
    llm: ArrayView2<f32>,
    t_emb: ArrayView2<f32>,
    input_proj: &Array2<f32>,
    time_proj: &Array2<f32>,
    llm_proj: &Array2<f32>,
) -> Result<Array2<f32>> {
    let x_proj = linear2(x_t, input_proj.view(), None);
    let t_raw = linear2(t_emb, time_proj.view(), None);
    let l_proj = linear2(llm, llm_proj.view(), None);
    ensure!(
        x_proj.dim().0 == 1 && t_raw.dim().0 == 1 && l_proj.dim().0 == 1,
        "FM projections expect batch=1"
    );
    let dim = x_proj.dim().1;
    ensure!(
        t_raw.dim().1 == dim && l_proj.dim().1 == dim,
        "projection dims mismatch"
    );
    let mut out = Array2::<f32>::zeros((FM_SEQ, dim));
    for j in 0..dim {
        out[[0, j]] = x_proj[[0, j]];
        out[[1, j]] = t_raw[[0, j]];
        out[[2, j]] = l_proj[[0, j]];
    }
    Ok(out)
}

fn mask_semantic_logits(sem: &mut Array1<f32>, vocab: usize, valid_end: usize) {
    if (EMPTY_AUDIO as usize) < sem.len() {
        sem[EMPTY_AUDIO as usize] = f32::NEG_INFINITY;
    }
    let end = valid_end.min(vocab).min(sem.len());
    for v in sem.iter_mut().take(vocab).skip(end) {
        *v = f32::NEG_INFINITY;
    }
}

fn mha(
    q: ArrayView2<f32>,
    k: ArrayView2<f32>,
    v: ArrayView2<f32>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Array2<f32> {
    let (t, _) = q.dim();
    let repeats = n_heads / n_kv_heads;
    let mut out = Array2::<f32>::zeros((t, n_heads * head_dim));
    for hi in 0..n_heads {
        let kv_h = hi / repeats;
        for qi in 0..t {
            let mut max_w = f32::NEG_INFINITY;
            let mut weights = vec![0f32; t];
            for ki in 0..t {
                let mut dot = 0f32;
                for di in 0..head_dim {
                    dot += q[[qi, hi * head_dim + di]] * k[[ki, kv_h * head_dim + di]];
                }
                dot /= (head_dim as f32).sqrt();
                weights[ki] = dot;
                max_w = max_w.max(dot);
            }
            let mut sum = 0f32;
            for ki in 0..t {
                weights[ki] = (weights[ki] - max_w).exp();
                sum += weights[ki];
            }
            for w in weights.iter_mut() {
                *w /= sum.max(1e-12);
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
    out
}

fn time_inv_freq(dim: usize, theta: f64) -> Array1<f32> {
    let half = dim / 2;
    Array1::from_iter((0..half).map(|i| (-(i as f64) * theta.ln() / half as f64).exp() as f32))
}

fn time_embed(t: f32, inv_freq: ArrayView1<f32>) -> Array2<f32> {
    let half = inv_freq.len();
    let mut emb = vec![0f32; half * 2];
    for i in 0..half {
        let ang = t * inv_freq[i];
        emb[i] = ang.cos();
        emb[i + half] = ang.sin();
    }
    Array2::from_shape_vec((1, half * 2), emb).unwrap()
}

fn argmax1(x: ArrayView1<f32>) -> usize {
    x.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn take2d(map: &HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str) -> Result<Array2<f32>> {
    let (data, shape) = map.get(key).with_context(|| format!("missing {key}"))?;
    ensure!(shape.len() == 2, "{key}: rank 2 expected");
    Array2::from_shape_vec((shape[0], shape[1]), data.clone()).with_context(|| key.to_string())
}

fn take1d(map: &HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str) -> Result<Array1<f32>> {
    let (data, shape) = map.get(key).with_context(|| format!("missing {key}"))?;
    ensure!(shape.len() == 1, "{key}: rank 1 expected");
    Array1::from_shape_vec(shape[0], data.clone()).with_context(|| key.to_string())
}

fn load_layer(
    map: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
    args: &AcousticTransformerArgs,
) -> Result<AcousticLayer> {
    let tp = |s: &str| format!("{prefix}.{s}");
    Ok(AcousticLayer {
        wq: take2d(map, &tp("attention.wq.weight"))?,
        wk: take2d(map, &tp("attention.wk.weight"))?,
        wv: take2d(map, &tp("attention.wv.weight"))?,
        wo: take2d(map, &tp("attention.wo.weight"))?,
        attn_norm: take1d(map, &tp("attention_norm.weight"))?,
        ffn_norm: take1d(map, &tp("ffn_norm.weight"))?,
        w1: take2d(map, &tp("feed_forward.w1.weight"))?,
        w2: take2d(map, &tp("feed_forward.w2.weight"))?,
        w3: take2d(map, &tp("feed_forward.w3.weight"))?,
        n_heads: args.n_heads,
        n_kv_heads: args.n_kv_heads,
        head_dim: args.head_dim,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fm_token_layout_is_three_by_dim() {
        let dim = 3;
        let x = Array2::<f32>::ones((1, dim));
        let llm = Array2::<f32>::ones((1, dim));
        let t = Array2::<f32>::ones((1, dim));
        let input_proj = Array2::<f32>::eye(dim);
        let time_proj = Array2::<f32>::eye(dim);
        let llm_proj = Array2::<f32>::eye(dim);
        let out = build_fm_tokens(
            x.view(),
            llm.view(),
            t.view(),
            &input_proj,
            &time_proj,
            &llm_proj,
        )
        .unwrap();
        assert_eq!(out.dim(), (FM_SEQ, dim));
    }
}
