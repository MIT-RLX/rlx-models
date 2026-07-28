// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! PrefixConditioner (Zyphra/Zonos `conditioning.py`).

use anyhow::{Context, Result, bail};

use crate::ops::{layer_norm, linear};
use crate::weights::WeightMap;

/// ISO / espeak language codes from Zyphra Zonos (index = language_id).
pub const LANGUAGE_CODES: &[&str] = &[
    "af",
    "am",
    "an",
    "ar",
    "as",
    "az",
    "ba",
    "bg",
    "bn",
    "bpy",
    "bs",
    "ca",
    "cmn",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en-029",
    "en-gb",
    "en-gb-scotland",
    "en-gb-x-gbclan",
    "en-gb-x-gbcwmd",
    "en-gb-x-rp",
    "en-us",
    "eo",
    "es",
    "es-419",
    "et",
    "eu",
    "fa",
    "fa-latn",
    "fi",
    "fr-be",
    "fr-ch",
    "fr-fr",
    "ga",
    "gd",
    "gn",
    "grc",
    "gu",
    "hak",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "hyw",
    "ia",
    "id",
    "is",
    "it",
    "ja",
    "jbo",
    "ka",
    "kk",
    "kl",
    "kn",
    "ko",
    "kok",
    "ku",
    "ky",
    "la",
    "lfn",
    "lt",
    "lv",
    "mi",
    "mk",
    "ml",
    "mr",
    "ms",
    "mt",
    "my",
    "nb",
    "nci",
    "ne",
    "nl",
    "om",
    "or",
    "pa",
    "pap",
    "pl",
    "pt",
    "pt-br",
    "py",
    "quc",
    "ro",
    "ru",
    "ru-lv",
    "sd",
    "shn",
    "si",
    "sk",
    "sl",
    "sq",
    "sr",
    "sv",
    "sw",
    "ta",
    "te",
    "tn",
    "tr",
    "tt",
    "ur",
    "uz",
    "vi",
    "vi-vn-x-central",
    "vi-vn-x-south",
    "yue",
];

pub fn language_id(code: &str) -> Result<i64> {
    LANGUAGE_CODES
        .iter()
        .position(|&c| c.eq_ignore_ascii_case(code))
        .map(|i| i as i64)
        .ok_or_else(|| anyhow::anyhow!("unsupported language {code}"))
}

/// Default emotion vector from Zonos `make_cond_dict` (pre-normalization).
pub const DEFAULT_EMOTION: [f32; 8] = [
    0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077,
];

#[derive(Debug, Clone)]
pub struct CondOpts {
    pub language: String,
    pub speaking_rate: f32,
    pub fmax: f32,
    pub pitch_std: f32,
    pub emotion: [f32; 8],
    /// Optional speaker embedding `[128]` — `None` → learned uncond.
    pub speaker: Option<Vec<f32>>,
}

impl Default for CondOpts {
    fn default() -> Self {
        Self {
            language: "en-us".into(),
            speaking_rate: 15.0,
            fmax: 22_050.0,
            pitch_std: 20.0,
            emotion: DEFAULT_EMOTION,
            speaker: None,
        }
    }
}

pub struct PrefixConditioner {
    dim: usize,
    eps: f32,
}

impl PrefixConditioner {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self { dim, eps }
    }

    /// Build prefix `[P+6, d_model]` for one sample.
    ///
    /// `phoneme_ids`: `[BOS … EOS]` (already tokenized).
    pub fn forward(
        &self,
        w: &WeightMap,
        phoneme_ids: &[i64],
        opts: &CondOpts,
        unconditional: bool,
    ) -> Result<Vec<f32>> {
        let d = self.dim;
        let mut chunks: Vec<Vec<f32>> = Vec::with_capacity(7);

        // 0) espeak — always required (no uncond vector)
        let emb = w.get("prefix_conditioner.conditioners.0.phoneme_embedder.weight")?;
        let vocab = w.shape("prefix_conditioner.conditioners.0.phoneme_embedder.weight")?[0];
        let mut ph = vec![0.0f32; phoneme_ids.len() * d];
        for (t, &id) in phoneme_ids.iter().enumerate() {
            let id = id as usize;
            if id >= vocab {
                bail!("phoneme id {id} out of range {vocab}");
            }
            ph[t * d..(t + 1) * d].copy_from_slice(&emb[id * d..(id + 1) * d]);
        }
        chunks.push(ph);

        // 1) speaker
        if unconditional || opts.speaker.is_none() {
            chunks.push(
                w.get("prefix_conditioner.conditioners.1.uncond_vector")?
                    .to_vec(),
            );
        } else if let Some(spk) = opts.speaker.as_ref() {
            anyhow::ensure!(spk.len() == 128, "speaker emb must be 128-d");
            let yw = w.get("prefix_conditioner.conditioners.1.project.weight")?;
            let yb = w.get("prefix_conditioner.conditioners.1.project.bias")?;
            let mut projected = linear(spk, yw, Some(yb), 1, d, 128);
            chunks.push(std::mem::take(&mut projected));
        }

        // 2–5) Fourier: emotion, fmax, pitch_std, speaking_rate
        if unconditional {
            for i in 2..=5 {
                chunks.push(
                    w.get(&format!(
                        "prefix_conditioner.conditioners.{i}.uncond_vector"
                    ))?
                    .to_vec(),
                );
            }
        } else {
            let mut emo = opts.emotion;
            let s: f32 = emo.iter().sum();
            if s > 0.0 {
                for v in &mut emo {
                    *v /= s;
                }
            }
            chunks.push(fourier(w, 2, &emo, 8, 0.0, 1.0, d)?);
            chunks.push(fourier(w, 3, &[opts.fmax], 1, 0.0, 24_000.0, d)?);
            chunks.push(fourier(w, 4, &[opts.pitch_std], 1, 0.0, 400.0, d)?);
            chunks.push(fourier(w, 5, &[opts.speaking_rate], 1, 0.0, 40.0, d)?);
        }

        // 6) language_id
        if unconditional {
            chunks.push(
                w.get("prefix_conditioner.conditioners.6.uncond_vector")?
                    .to_vec(),
            );
        } else {
            let lid = language_id(&opts.language)?;
            // IntegerConditioner: index = x - min_val, min_val = -1
            let idx = (lid - (-1)) as usize;
            let emb = w.get("prefix_conditioner.conditioners.6.int_embedder.weight")?;
            let vocab = w.shape("prefix_conditioner.conditioners.6.int_embedder.weight")?[0];
            if idx >= vocab {
                bail!("language embed idx {idx} >= {vocab}");
            }
            chunks.push(emb[idx * d..(idx + 1) * d].to_vec());
        }

        // Concat along sequence, then project + LayerNorm.
        let seq: usize = chunks.iter().map(|c| c.len() / d).sum();
        let mut cat = Vec::with_capacity(seq * d);
        for c in &chunks {
            anyhow::ensure!(c.len() % d == 0, "bad conditioner chunk len");
            cat.extend_from_slice(c);
        }
        let pw = w.get("prefix_conditioner.project.weight")?;
        let pb = w.get("prefix_conditioner.project.bias")?;
        let projected = linear(&cat, pw, Some(pb), seq, d, d);
        let nw = w.get("prefix_conditioner.norm.weight")?;
        let nb = w.get("prefix_conditioner.norm.bias")?;
        Ok(layer_norm(&projected, nw, nb, seq, d, self.eps))
    }
}

fn fourier(
    w: &WeightMap,
    idx: usize,
    x: &[f32],
    input_dim: usize,
    min_val: f32,
    max_val: f32,
    output_dim: usize,
) -> Result<Vec<f32>> {
    anyhow::ensure!(x.len() == input_dim, "fourier input_dim");
    let weight = w
        .get(&format!("prefix_conditioner.conditioners.{idx}.weight"))
        .with_context(|| format!("fourier weight {idx}"))?;
    // weight: [output_dim/2, input_dim]
    let half = output_dim / 2;
    debug_assert_eq!(weight.len(), half * input_dim);
    let mut xn = vec![0.0f32; input_dim];
    let span = (max_val - min_val).max(1e-8);
    for i in 0..input_dim {
        xn[i] = (x[i] - min_val) / span;
    }
    // f = 2π * x @ W^T  → [half]
    let mut f = vec![0.0f32; half];
    for o in 0..half {
        let mut acc = 0.0f32;
        let wr = &weight[o * input_dim..(o + 1) * input_dim];
        for i in 0..input_dim {
            acc += xn[i] * wr[i];
        }
        f[o] = 2.0 * std::f32::consts::PI * acc;
    }
    let mut out = vec![0.0f32; output_dim];
    for o in 0..half {
        out[o] = f[o].cos();
        out[half + o] = f[o].sin();
    }
    Ok(out)
}
