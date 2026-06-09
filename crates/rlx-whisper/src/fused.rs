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

//! Pre-fused encoder/decoder weights (QKV projection + tied logit embed), inspired by
//! [fast-whisper-burn](https://github.com/AdrianEddy/fast-whisper-burn).

use crate::config::WhisperConfig;
use crate::weights::WhisperWeightPrefix;
use anyhow::{Context, Result};
use std::collections::HashMap;

type WeightEntry = (String, Vec<f32>, Vec<usize>);
type OptionalBias = Option<WeightEntry>;
type OptionalBiasList = Vec<OptionalBias>;

/// Fused encoder self-attention QKV per layer (`[d_model, 3*d_model]`).
#[derive(Debug, Clone)]
pub struct FusedEncoderWeights {
    pub layer_qkv_w: Vec<WeightEntry>,
    pub layer_qkv_b: OptionalBiasList,
}

impl FusedEncoderWeights {
    pub fn from_checkpoint(
        tensors: &HashMap<String, (Vec<f32>, Vec<usize>)>,
        cfg: &WhisperConfig,
        pfx: &WhisperWeightPrefix,
    ) -> Result<Self> {
        let d = cfg.d_model;
        let mut layer_qkv_w = Vec::with_capacity(cfg.encoder_layers);
        let mut layer_qkv_b = Vec::with_capacity(cfg.encoder_layers);

        for i in 0..cfg.encoder_layers {
            let qw = format_enc_layer(pfx, i, "self_attn.q_proj.weight");
            let kw = format_enc_layer(pfx, i, "self_attn.k_proj.weight");
            let vw = format_enc_layer(pfx, i, "self_attn.v_proj.weight");
            let qb = format_enc_layer(pfx, i, "self_attn.q_proj.bias");
            let kb = format_enc_layer(pfx, i, "self_attn.k_proj.bias");
            let vb = format_enc_layer(pfx, i, "self_attn.v_proj.bias");

            let (qw_d, qw_s) = tensors.get(&qw).with_context(|| qw.clone())?;
            let (kw_d, kw_s) = tensors.get(&kw).with_context(|| kw.clone())?;
            let (vw_d, vw_s) = tensors.get(&vw).with_context(|| vw.clone())?;
            ensure_mat(qw_s, d, d)?;
            ensure_mat(kw_s, d, d)?;
            ensure_mat(vw_s, d, d)?;

            let w_key = format!("fused.enc.{i}.self_attn.qkv.weight");
            layer_qkv_w.push((
                w_key,
                concat_qkv_weights(qw_d, kw_d, vw_d, d),
                vec![d, 3 * d],
            ));

            let b_key = format!("fused.enc.{i}.self_attn.qkv.bias");
            let bias = match (tensors.get(&qb), tensors.get(&kb), tensors.get(&vb)) {
                (Some((qb_d, _)), Some((kb_d, _)), Some((vb_d, _))) => {
                    Some((b_key, concat_qkv_bias(qb_d, kb_d, vb_d, d), vec![3 * d]))
                }
                _ => None,
            };
            layer_qkv_b.push(bias);
        }
        Ok(Self {
            layer_qkv_w,
            layer_qkv_b,
        })
    }

    pub fn merge_into_tensors(&self, tensors: &mut HashMap<String, (Vec<f32>, Vec<usize>)>) {
        for (k, data, shape) in &self.layer_qkv_w {
            tensors.insert(k.clone(), (data.clone(), shape.clone()));
        }
        for (k, data, shape) in self.layer_qkv_b.iter().flatten() {
            tensors.insert(k.clone(), (data.clone(), shape.clone()));
        }
    }

    pub fn qkv_w_key(&self, layer: usize) -> &str {
        &self.layer_qkv_w[layer].0
    }

    pub fn qkv_b_key(&self, layer: usize) -> Option<&str> {
        self.layer_qkv_b[layer].as_ref().map(|(k, _, _)| k.as_str())
    }
}

/// Fused decoder self-attn QKV per layer (`[d_model, 3*d_model]`).
#[derive(Debug, Clone)]
pub struct FusedDecoderWeights {
    pub layer_qkv_w: Vec<WeightEntry>,
    pub layer_qkv_b: OptionalBiasList,
}

impl FusedDecoderWeights {
    /// Build fused QKV `[d_model, 3*d_model]` per decoder layer.
    pub fn from_checkpoint(
        tensors: &HashMap<String, (Vec<f32>, Vec<usize>)>,
        cfg: &WhisperConfig,
        pfx: &WhisperWeightPrefix,
    ) -> Result<Self> {
        let d = cfg.d_model;
        let mut layer_qkv_w = Vec::with_capacity(cfg.decoder_layers);
        let mut layer_qkv_b = Vec::with_capacity(cfg.decoder_layers);

        for i in 0..cfg.decoder_layers {
            let qw = format_layer(pfx, i, "self_attn.q_proj.weight");
            let kw = format_layer(pfx, i, "self_attn.k_proj.weight");
            let vw = format_layer(pfx, i, "self_attn.v_proj.weight");
            let qb = format_layer(pfx, i, "self_attn.q_proj.bias");
            let kb = format_layer(pfx, i, "self_attn.k_proj.bias");
            let vb = format_layer(pfx, i, "self_attn.v_proj.bias");

            let (qw_d, qw_s) = tensors.get(&qw).with_context(|| qw.clone())?;
            let (kw_d, kw_s) = tensors.get(&kw).with_context(|| kw.clone())?;
            let (vw_d, vw_s) = tensors.get(&vw).with_context(|| vw.clone())?;
            ensure_mat(qw_s, d, d)?;
            ensure_mat(kw_s, d, d)?;
            ensure_mat(vw_s, d, d)?;

            let w_key = format!("fused.dec.{i}.self_attn.qkv.weight");
            let w_data = concat_qkv_weights(qw_d, kw_d, vw_d, d);
            layer_qkv_w.push((w_key, w_data, vec![d, 3 * d]));

            let b_key = format!("fused.dec.{i}.self_attn.qkv.bias");
            let bias = match (tensors.get(&qb), tensors.get(&kb), tensors.get(&vb)) {
                (Some((qb_d, _)), Some((kb_d, _)), Some((vb_d, _))) => {
                    let b_data = concat_qkv_bias(qb_d, kb_d, vb_d, d);
                    Some((b_key, b_data, vec![3 * d]))
                }
                _ => None,
            };
            layer_qkv_b.push(bias);
        }

        Ok(Self {
            layer_qkv_w,
            layer_qkv_b,
        })
    }

    pub fn merge_into_tensors(&self, tensors: &mut HashMap<String, (Vec<f32>, Vec<usize>)>) {
        for (k, data, shape) in &self.layer_qkv_w {
            tensors.insert(k.clone(), (data.clone(), shape.clone()));
        }
        for (k, data, shape) in self.layer_qkv_b.iter().flatten() {
            tensors.insert(k.clone(), (data.clone(), shape.clone()));
        }
    }

    pub fn merge_into_params(&self, params: &mut HashMap<String, Vec<f32>>) {
        for (k, data, _) in &self.layer_qkv_w {
            params.insert(k.clone(), data.clone());
        }
        for (k, data, _) in self.layer_qkv_b.iter().flatten() {
            params.insert(k.clone(), data.clone());
        }
    }

    pub fn qkv_w_key(&self, layer: usize) -> &str {
        &self.layer_qkv_w[layer].0
    }

    pub fn qkv_b_key(&self, layer: usize) -> Option<&str> {
        self.layer_qkv_b[layer].as_ref().map(|(k, _, _)| k.as_str())
    }
}

fn format_layer(pfx: &WhisperWeightPrefix, i: usize, suffix: &str) -> String {
    pfx.dec_layer(i, suffix)
}

fn format_enc_layer(pfx: &WhisperWeightPrefix, i: usize, suffix: &str) -> String {
    pfx.enc_layer(i, suffix)
}

fn ensure_mat(shape: &[usize], rows: usize, cols: usize) -> Result<()> {
    anyhow::ensure!(
        shape == [rows, cols],
        "expected mat [{rows}, {cols}], got {shape:?}"
    );
    Ok(())
}

/// HF `[out, in]` → fused `mm(x, w)` layout `[in, out]` per Q/K/V block (see `linear_fused_qkv`).
fn concat_qkv_weights(q: &[f32], k: &[f32], v: &[f32], d: usize) -> Vec<f32> {
    let mut out = vec![0f32; d * 3 * d];
    for i in 0..d {
        let base = i * 3 * d;
        for j in 0..d {
            out[base + j] = q[j * d + i];
            out[base + d + j] = k[j * d + i];
            out[base + 2 * d + j] = v[j * d + i];
        }
    }
    out
}

fn concat_qkv_bias(q: &[f32], k: &[f32], v: &[f32], d: usize) -> Vec<f32> {
    let mut out = vec![0f32; 3 * d];
    out[..d].copy_from_slice(q);
    out[d..2 * d].copy_from_slice(k);
    out[2 * d..].copy_from_slice(v);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mm_row(x: &[f32], w: &[f32], in_f: usize, out_f: usize) -> Vec<f32> {
        let mut y = vec![0f32; out_f];
        for o in 0..out_f {
            for i in 0..in_f {
                y[o] += x[i] * w[i * out_f + o];
            }
        }
        y
    }

    fn mm_row_fused_qkv(x: &[f32], fused: &[f32], d: usize, col_off: usize) -> Vec<f32> {
        let mut y = vec![0f32; d];
        for o in 0..d {
            for i in 0..d {
                y[o] += x[i] * fused[i * 3 * d + col_off + o];
            }
        }
        y
    }

    fn hf_linear_transposed(w: &[f32], d: usize) -> Vec<f32> {
        let mut out = vec![0f32; d * d];
        for o in 0..d {
            for i in 0..d {
                out[i * d + o] = w[o * d + i];
            }
        }
        out
    }

    #[test]
    fn concat_qkv_matches_hf_linear_layout() {
        let d = 16usize;
        let q: Vec<f32> = (0..d * d).map(|i| (i as f32 * 0.013).sin()).collect();
        let k: Vec<f32> = (0..d * d).map(|i| (i as f32 * 0.017).cos()).collect();
        let v: Vec<f32> = (0..d * d).map(|i| (i as f32 * 0.019).sin()).collect();
        let x: Vec<f32> = (0..d).map(|i| (i as f32 + 1.0) * 0.05).collect();

        let fused = concat_qkv_weights(&q, &k, &v, d);
        for (proj, col_off) in [(&q, 0usize), (&k, d), (&v, 2 * d)] {
            let w = hf_linear_transposed(proj, d);
            let expected = mm_row(&x, &w, d, d);
            let got = mm_row_fused_qkv(&x, &fused, d, col_off);
            let mx = expected
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(mx < 1e-6, "proj max_abs={mx}");
        }
    }
}
