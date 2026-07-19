// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Eager MetaVoice second-stage (6×384 non-causal) → EnCodec codebooks 2–7.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use ndarray::{Array1, Array2, Array3, s};

use crate::config::SecondStageArgs;

/// Pad / end-of-audio marker in second-stage hierarchies (same as official HF).
pub const PAD: u32 = 1024;
/// Text token offset for second-stage BPE (from `second_stage.pt` meta).
pub const TEXT_OFFSET: u32 = 1025;

pub struct SecondStage {
    args: SecondStageArgs,
    wte0: Array2<f32>,
    wte1: Array2<f32>,
    wpe: Array2<f32>,
    spk_proj: Array2<f32>,
    ln_f: Array1<f32>,
    lm_heads: Vec<Array2<f32>>,
    layers: Vec<Layer>,
    speaker_emb_size: usize,
}

struct Layer {
    ln1: Array1<f32>,
    c_attn: Array2<f32>,
    c_proj: Array2<f32>,
    ln2: Array1<f32>,
    c_fc: Array2<f32>,
    c_proj_mlp: Array2<f32>,
}

impl SecondStage {
    pub fn from_weights(
        args: &SecondStageArgs,
        w: &HashMap<String, Vec<f32>>,
        speaker_emb_size: usize,
    ) -> Result<Self> {
        let c = args.n_embd;
        let v0 = *args
            .vocab_sizes
            .first()
            .ok_or_else(|| anyhow!("empty vocab_sizes"))?;
        let v1 = *args
            .vocab_sizes
            .get(1)
            .ok_or_else(|| anyhow!("need 2 vocab_sizes"))?;
        let targets = if args.target_vocab_sizes.is_empty() {
            vec![1025; 6]
        } else {
            args.target_vocab_sizes.clone()
        };
        let mut layers = Vec::with_capacity(args.n_layer);
        for i in 0..args.n_layer {
            let p = format!("transformer.h.{i}");
            layers.push(Layer {
                ln1: arr1(w, &format!("{p}.ln_1.weight"), c)?,
                c_attn: arr2(w, &format!("{p}.attn.c_attn.weight"), 3 * c, c)?,
                c_proj: arr2(w, &format!("{p}.attn.c_proj.weight"), c, c)?,
                ln2: arr1(w, &format!("{p}.ln_2.weight"), c)?,
                c_fc: arr2(w, &format!("{p}.mlp.c_fc.weight"), 4 * c, c)?,
                c_proj_mlp: arr2(w, &format!("{p}.mlp.c_proj.weight"), c, 4 * c)?,
            });
        }
        let mut lm_heads = Vec::with_capacity(targets.len());
        for (i, &vs) in targets.iter().enumerate() {
            lm_heads.push(arr2(w, &format!("lm_heads.{i}.weight"), vs, c)?);
        }
        Ok(Self {
            args: args.clone(),
            wte0: arr2(w, "transformer.wtes.0.weight", v0, c)?,
            wte1: arr2(w, "transformer.wtes.1.weight", v1, c)?,
            wpe: arr2(w, "transformer.wpe.weight", args.block_size, c)?,
            spk_proj: arr2(w, "speaker_cond_pos.weight", c, speaker_emb_size)?,
            ln_f: arr1(w, "transformer.ln_f.weight", c)?,
            lm_heads,
            layers,
            speaker_emb_size,
        })
    }

    /// Predict fine EnCodec books (2–7) from coarse `c0`/`c1` + text (BPE+offset).
    pub fn predict_fine(
        &self,
        text_ids: &[u32],
        c0: &[u32],
        c1: &[u32],
        spk_emb: &[f32],
    ) -> Result<Vec<Vec<u32>>> {
        if c0.is_empty() || c0.len() != c1.len() {
            return Err(anyhow!(
                "coarse books empty/mismatched: {} vs {}",
                c0.len(),
                c1.len()
            ));
        }
        if spk_emb.len() != self.speaker_emb_size {
            return Err(anyhow!(
                "spk_emb len {} != {}",
                spk_emb.len(),
                self.speaker_emb_size
            ));
        }
        let bs = self.args.block_size;
        let mut h0 = Vec::with_capacity(bs);
        h0.extend_from_slice(text_ids);
        h0.extend_from_slice(c0);
        h0.push(PAD);
        let mut h1 = Vec::with_capacity(bs);
        h1.extend(std::iter::repeat(PAD).take(text_ids.len()));
        h1.extend_from_slice(c1);
        h1.push(PAD);
        anyhow::ensure!(h0.len() == h1.len(), "hierarchy length mismatch");
        if h0.len() > bs {
            h0.truncate(bs);
            h1.truncate(bs);
        } else {
            h0.resize(bs, PAD);
            h1.resize(bs, PAD);
        }

        let x = self.forward(&h0, &h1, spk_emb)?;
        let t_audio = c0.len();
        let mut fine = Vec::with_capacity(self.lm_heads.len());
        for head in &self.lm_heads {
            let mut book = Vec::with_capacity(t_audio);
            // Align predicted codes with the coarse span: after text, before
            // the trailing pad that ends the real content window.
            let start = text_ids.len();
            for ti in start..start + t_audio {
                if ti >= bs {
                    break;
                }
                let h = x.slice(s![ti, ..]);
                let logits = head.dot(&h);
                let mut id = argmax(logits.as_slice().unwrap());
                // Vocab is 1025 (includes pad=1024); EnCodec embeds only 0..1023.
                if id >= PAD {
                    id = 0;
                }
                book.push(id);
            }
            book.truncate(t_audio);
            fine.push(book);
        }
        Ok(fine)
    }

    fn forward(&self, h0: &[u32], h1: &[u32], spk_emb: &[f32]) -> Result<Array2<f32>> {
        let t = h0.len();
        let c = self.args.n_embd;
        let n_head = self.args.n_head;
        let hd = c / n_head;
        let eps = 1e-5f32;

        let spk = Array1::from_vec(spk_emb.to_vec());
        let spk_c = self.spk_proj.dot(&spk);

        let mut x = Array2::<f32>::zeros((t, c));
        for i in 0..t {
            let e0 = self.wte0.row(h0[i] as usize);
            let e1 = self.wte1.row(h1[i] as usize);
            let pe = self.wpe.row(i);
            for j in 0..c {
                x[[i, j]] = e0[j] + e1[j] + pe[j] + spk_c[j];
            }
        }

        for layer in &self.layers {
            let n1 = layer_norm2(&x, &layer.ln1, eps);
            let y = n1.dot(&layer.c_attn.t()); // [T, 3C]
            let mut q = Array3::<f32>::zeros((n_head, t, hd));
            let mut k = Array3::<f32>::zeros((n_head, t, hd));
            let mut v = Array3::<f32>::zeros((n_head, t, hd));
            for ti in 0..t {
                for h in 0..n_head {
                    for d in 0..hd {
                        let o = h * hd + d;
                        q[[h, ti, d]] = y[[ti, o]];
                        k[[h, ti, d]] = y[[ti, c + o]];
                        v[[h, ti, d]] = y[[ti, 2 * c + o]];
                    }
                }
            }
            let attn = mha_noncausal(&q, &k, &v, hd);
            let attn_proj = attn.dot(&layer.c_proj.t());
            x = &x + &attn_proj;

            let n2 = layer_norm2(&x, &layer.ln2, eps);
            let hidden = n2.dot(&layer.c_fc.t());
            let mut gelued = Array2::<f32>::zeros(hidden.raw_dim());
            for ti in 0..t {
                for hi in 0..hidden.ncols() {
                    gelued[[ti, hi]] = gelu(hidden[[ti, hi]]);
                }
            }
            let mlp = gelued.dot(&layer.c_proj_mlp.t());
            x = &x + &mlp;
        }

        Ok(layer_norm2(&x, &self.ln_f, eps))
    }
}

fn mha_noncausal(q: &Array3<f32>, k: &Array3<f32>, v: &Array3<f32>, hd: usize) -> Array2<f32> {
    let (n_head, t, _) = q.dim();
    let c = n_head * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = Array2::<f32>::zeros((t, c));
    for h in 0..n_head {
        for qi in 0..t {
            let mut scores = vec![0.0f32; t];
            let mut mx = f32::NEG_INFINITY;
            for kj in 0..t {
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q[[h, qi, d]] * k[[h, kj, d]];
                }
                scores[kj] = dot * scale;
                if scores[kj] > mx {
                    mx = scores[kj];
                }
            }
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - mx).exp();
                sum += *s;
            }
            let inv = 1.0 / sum.max(1e-20);
            for d in 0..hd {
                let mut acc = 0.0f32;
                for kj in 0..t {
                    acc += scores[kj] * inv * v[[h, kj, d]];
                }
                out[[qi, h * hd + d]] = acc;
            }
        }
    }
    out
}

fn layer_norm2(x: &Array2<f32>, weight: &Array1<f32>, eps: f32) -> Array2<f32> {
    let (t, c) = x.dim();
    let mut out = Array2::zeros((t, c));
    let inv_c = 1.0 / c as f32;
    for ti in 0..t {
        let mut mean = 0.0f32;
        for ci in 0..c {
            mean += x[[ti, ci]];
        }
        mean *= inv_c;
        let mut var = 0.0f32;
        for ci in 0..c {
            let d = x[[ti, ci]] - mean;
            var += d * d;
        }
        var *= inv_c;
        let inv = 1.0 / (var + eps).sqrt();
        for ci in 0..c {
            out[[ti, ci]] = (x[[ti, ci]] - mean) * inv * weight[ci];
        }
    }
    out
}

fn gelu(x: f32) -> f32 {
    // Exact-ish GELU via erf approximation (matches torch within ~1e-6).
    0.5 * x * (1.0 + erf_approx(x / std::f32::consts::SQRT_2))
}

fn erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-ax * ax).exp();
    sign * y
}

fn arr1(w: &HashMap<String, Vec<f32>>, name: &str, n: usize) -> Result<Array1<f32>> {
    let v = w.get(name).with_context(|| format!("missing {name}"))?;
    anyhow::ensure!(v.len() == n, "{name}: len {} != {n}", v.len());
    Ok(Array1::from_vec(v.clone()))
}

fn arr2(
    w: &HashMap<String, Vec<f32>>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Array2<f32>> {
    let v = w.get(name).with_context(|| format!("missing {name}"))?;
    let rows = if rows == 0 { v.len() / cols } else { rows };
    anyhow::ensure!(
        v.len() == rows * cols,
        "{name}: len {} != {rows}×{cols}",
        v.len()
    );
    Ok(Array2::from_shape_vec((rows, cols), v.clone())?)
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}
