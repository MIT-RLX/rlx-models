// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Eager first-stage MetaVoice GPT (24×2048, absolute pos, SwiGLU, CFG).

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use ndarray::{Array1, Array2, Array3, Array4, Axis, s};

use crate::config::FirstStageArgs;

pub const EOS_AUDIO: u32 = 2048;

pub struct FirstStage {
    args: FirstStageArgs,
    wte: Array2<f32>,
    wpe: Array2<f32>,
    spk_proj: Array2<f32>,
    ln_f: Array1<f32>,
    lm_head: Array2<f32>,
    layers: Vec<Layer>,
}

struct Layer {
    ln1: Array1<f32>,
    c_attn: Array2<f32>,
    c_proj: Array2<f32>,
    ln2: Array1<f32>,
    w1: Array2<f32>,
    w3: Array2<f32>,
    w2: Array2<f32>,
}

/// Per-layer KV: [2, n_head, T, hd]
struct Kv {
    k: Array4<f32>,
    v: Array4<f32>,
}

impl FirstStage {
    pub fn from_weights(args: &FirstStageArgs, w: &HashMap<String, Vec<f32>>) -> Result<Self> {
        let c = args.n_embd;
        let v = *args
            .vocab_sizes
            .first()
            .ok_or_else(|| anyhow!("empty vocab_sizes"))?;
        let mut layers = Vec::with_capacity(args.n_layer);
        for i in 0..args.n_layer {
            let p = format!("transformer.h.{i}");
            let w1 = arr2(w, &format!("{p}.mlp.swiglu.w1.weight"), 0, c)?;
            let h = w1.nrows();
            layers.push(Layer {
                ln1: arr1(w, &format!("{p}.ln_1.weight"), c)?,
                c_attn: arr2(w, &format!("{p}.attn.c_attn.weight"), 3 * c, c)?,
                c_proj: arr2(w, &format!("{p}.attn.c_proj.weight"), c, c)?,
                ln2: arr1(w, &format!("{p}.ln_2.weight"), c)?,
                w1,
                w3: arr2(w, &format!("{p}.mlp.swiglu.w3.weight"), h, c)?,
                w2: arr2(w, &format!("{p}.mlp.c_proj.weight"), c, h)?,
            });
        }
        Ok(Self {
            args: args.clone(),
            wte: arr2(w, "transformer.wtes.0.weight", v, c)?,
            wpe: arr2(w, "transformer.wpe.weight", args.block_size, c)?,
            spk_proj: arr2(w, "speaker_cond_pos.weight", c, args.speaker_emb_size)?,
            ln_f: arr1(w, "transformer.ln_f.weight", c)?,
            lm_head: arr2(w, "lm_heads.0.weight", v, c)?,
            layers,
        })
    }

    pub fn generate_greedy(
        &self,
        prompt: &[u32],
        spk_emb: &[f32],
        max_new: usize,
        guidance_scale: f32,
    ) -> Result<Vec<u32>> {
        self.generate(
            prompt,
            spk_emb,
            max_new,
            guidance_scale,
            /*temperature*/ 0.0,
            /*top_p*/ 1.0,
            /*seed*/ 0,
        )
    }

    /// Autoregressive generate. `temperature <= 0` → greedy argmax.
    pub fn generate(
        &self,
        prompt: &[u32],
        spk_emb: &[f32],
        max_new: usize,
        guidance_scale: f32,
        temperature: f32,
        top_p: f32,
        seed: u64,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(anyhow!("empty prompt"));
        }
        if spk_emb.len() != self.args.speaker_emb_size {
            return Err(anyhow!(
                "spk_emb len {} != {}",
                spk_emb.len(),
                self.args.speaker_emb_size
            ));
        }
        let mut tokens = prompt.to_vec();
        let mut kv: Vec<Option<Kv>> = (0..self.layers.len()).map(|_| None).collect();
        let mut rng = XorShift64::new(seed.max(1));

        for step in 0..max_new {
            let start = if step == 0 { 0 } else { tokens.len() - 1 };
            let ctxt = &tokens[start..];
            let mut logits = self.forward_step(ctxt, start, spk_emb, guidance_scale, &mut kv)?;
            let next = if temperature <= 0.0 {
                argmax(&logits)
            } else {
                for l in &mut logits {
                    *l /= temperature;
                }
                sample_top_p(&mut logits, top_p, &mut rng)
            };
            tokens.push(next);
            if step == 0 || (step + 1) % 16 == 0 || next == EOS_AUDIO || step + 1 == max_new {
                eprintln!(
                    "[metavoice] first-stage AR {}/{} (CPU eager; EnCodec uses --device)",
                    step + 1,
                    max_new
                );
            }
            if next == EOS_AUDIO || tokens.len() >= self.args.block_size {
                break;
            }
        }
        Ok(tokens)
    }

    fn forward_step(
        &self,
        toks: &[u32],
        pos0: usize,
        spk_emb: &[f32],
        guidance: f32,
        kv: &mut [Option<Kv>],
    ) -> Result<Vec<f32>> {
        let t = toks.len();
        let c = self.args.n_embd;
        let n_head = self.args.n_head;
        let hd = c / n_head;
        let eps = self.args.rmsnorm_eps;

        let spk = Array1::from_vec(spk_emb.to_vec());
        let spk_c = self.spk_proj.dot(&spk);
        let mut x = Array3::<f32>::zeros((2, t, c));
        for (i, &id) in toks.iter().enumerate() {
            let emb = self.wte.row(id as usize);
            let pe = self.wpe.row(pos0 + i);
            for row in 0..2 {
                for j in 0..c {
                    let sj = if row == 0 { spk_c[j] } else { 0.0 };
                    x[[row, i, j]] = emb[j] + pe[j] + sj;
                }
            }
        }

        for (li, layer) in self.layers.iter().enumerate() {
            let n1 = rms_norm3(&x, &layer.ln1, eps);
            let mut q = Array4::<f32>::zeros((2, n_head, t, hd));
            let mut k_new = Array4::<f32>::zeros((2, n_head, t, hd));
            let mut v_new = Array4::<f32>::zeros((2, n_head, t, hd));
            for row in 0..2 {
                let y = n1.index_axis(Axis(0), row).dot(&layer.c_attn.t()); // [T, 3C]
                for ti in 0..t {
                    for h in 0..n_head {
                        for d in 0..hd {
                            let o = h * hd + d;
                            q[[row, h, ti, d]] = y[[ti, o]];
                            k_new[[row, h, ti, d]] = y[[ti, c + o]];
                            v_new[[row, h, ti, d]] = y[[ti, 2 * c + o]];
                        }
                    }
                }
            }
            let (k_all, v_all) = match &mut kv[li] {
                None => {
                    let owned = Kv {
                        k: k_new.clone(),
                        v: v_new.clone(),
                    };
                    kv[li] = Some(owned);
                    let e = kv[li].as_ref().unwrap();
                    (e.k.clone(), e.v.clone())
                }
                Some(prev) => {
                    let k = concat_time(&prev.k, &k_new);
                    let v = concat_time(&prev.v, &v_new);
                    prev.k = k.clone();
                    prev.v = v.clone();
                    (k, v)
                }
            };
            let attn = mha(&q, k_all.view(), v_all.view(), hd);
            let mut attn_proj = Array3::<f32>::zeros((2, t, c));
            for row in 0..2 {
                let y = attn.index_axis(Axis(0), row).dot(&layer.c_proj.t());
                attn_proj.index_axis_mut(Axis(0), row).assign(&y);
            }
            x = &x + &attn_proj;

            let n2 = rms_norm3(&x, &layer.ln2, eps);
            let mut mlp = Array3::<f32>::zeros((2, t, c));
            for row in 0..2 {
                let xin = n2.index_axis(Axis(0), row);
                let gate = xin.dot(&layer.w1.t());
                let up = xin.dot(&layer.w3.t());
                let mut gated = Array2::<f32>::zeros(gate.raw_dim());
                for ti in 0..t {
                    for hi in 0..gate.ncols() {
                        gated[[ti, hi]] = silu(gate[[ti, hi]]) * up[[ti, hi]];
                    }
                }
                mlp.index_axis_mut(Axis(0), row)
                    .assign(&gated.dot(&layer.w2.t()));
            }
            x = &x + &mlp;
        }

        let xf = rms_norm3(&x, &self.ln_f, eps);
        let h0 = xf.slice(s![0, t - 1, ..]).to_owned();
        let h1 = xf.slice(s![1, t - 1, ..]).to_owned();
        let logits0: Array1<f32> = self.lm_head.dot(&h0);
        let logits1: Array1<f32> = self.lm_head.dot(&h1);
        let mut out = vec![0.0f32; logits0.len()];
        for i in 0..out.len() {
            out[i] = guidance * logits0[i] + (1.0 - guidance) * logits1[i];
        }
        Ok(out)
    }
}

fn concat_time(prev: &Array4<f32>, new: &Array4<f32>) -> Array4<f32> {
    let (b, h, t0, d) = prev.dim();
    let t1 = new.dim().2;
    let mut out = Array4::<f32>::zeros((b, h, t0 + t1, d));
    out.slice_mut(s![.., .., 0..t0, ..]).assign(prev);
    out.slice_mut(s![.., .., t0.., ..]).assign(new);
    out
}

/// q: [2,H,Tq,D], k/v: [2,H,Tk,D] → out [2,Tq,C]
fn mha(
    q: &Array4<f32>,
    k: ndarray::ArrayView4<'_, f32>,
    v: ndarray::ArrayView4<'_, f32>,
    hd: usize,
) -> Array3<f32> {
    let (b, n_head, tq, _) = q.dim();
    let tk = k.dim().2;
    let c = n_head * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = Array3::<f32>::zeros((b, tq, c));
    for bi in 0..b {
        for h in 0..n_head {
            for qi in 0..tq {
                // causal: key positions 0..=(tk - tq + qi)
                let k_end = tk - tq + qi + 1;
                let mut scores = vec![0.0f32; k_end];
                let mut mx = f32::NEG_INFINITY;
                for kj in 0..k_end {
                    let mut dot = 0.0f32;
                    for d in 0..hd {
                        dot += q[[bi, h, qi, d]] * k[[bi, h, kj, d]];
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
                let inv = 1.0 / sum;
                for d in 0..hd {
                    let mut acc = 0.0f32;
                    for kj in 0..k_end {
                        acc += scores[kj] * inv * v[[bi, h, kj, d]];
                    }
                    out[[bi, qi, h * hd + d]] = acc;
                }
            }
        }
    }
    out
}

fn rms_norm3(x: &Array3<f32>, alpha: &Array1<f32>, eps: f32) -> Array3<f32> {
    let (b, t, c) = x.dim();
    let mut out = Array3::zeros((b, t, c));
    let inv_c = 1.0 / c as f32;
    for bi in 0..b {
        for ti in 0..t {
            let mut ms = 0.0f32;
            for ci in 0..c {
                let v = x[[bi, ti, ci]];
                ms += v * v;
            }
            let scale = 1.0 / (ms * inv_c + eps).sqrt();
            for ci in 0..c {
                out[[bi, ti, ci]] = x[[bi, ti, ci]] * scale * alpha[ci];
            }
        }
    }
    out
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
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

struct XorShift64(u64);
impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 as f64 / u64::MAX as f64) as f32
    }
}

/// Softmax + nucleus sample. Mutates `logits` in place for a scratch workspace.
fn sample_top_p(logits: &mut [f32], top_p: f32, rng: &mut XorShift64) -> u32 {
    let n = logits.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());

    let mx = logits[order[0]];
    let mut sum = 0.0f32;
    for &i in &order {
        logits[i] = (logits[i] - mx).exp();
        sum += logits[i];
    }
    let inv = 1.0 / sum.max(1e-20);
    for &i in &order {
        logits[i] *= inv;
    }

    let cutoff = top_p.clamp(0.0, 1.0);
    let mut cum = 0.0f32;
    let mut last = 0usize;
    for (k, &i) in order.iter().enumerate() {
        cum += logits[i];
        last = k;
        if cum >= cutoff {
            break;
        }
    }
    let target = rng.next_f32() * cum.min(1.0);
    let mut running = 0.0f32;
    for &i in order.iter().take(last + 1) {
        running += logits[i];
        if running >= target {
            return i as u32;
        }
    }
    order[last] as u32
}

/// Unpack interleaved first-stage tokens → EnCodec codebook 0/1.
pub fn extract_codebooks(tokens: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut c0 = Vec::new();
    let mut c1 = Vec::new();
    for &t in tokens {
        if t < 1024 {
            c0.push(t);
        } else if t < 2048 {
            c1.push(t - 1024);
        }
    }
    let n = c0.len().min(c1.len());
    c0.truncate(n);
    c1.truncate(n);
    (c0, c1)
}
