use crate::config::LmConfig;
use crate::depformer::DepFormer;
use crate::nn::{Embedding, linear, rms_norm};
use crate::transformer::StreamingTransformer;
use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Moshi temporal LM (Helium) + optional DepFormer depth decoder.
pub struct LmModel {
    cfg: LmConfig,
    text_emb: Embedding,
    audio_embs: Vec<Embedding>,
    text_linear: Array2<f32>,
    out_norm_alpha: Array1<f32>,
    transformer: StreamingTransformer,
    depformer: Option<DepFormer>,
}

impl LmModel {
    pub fn open(cfg: LmConfig, weights: HashMap<String, (Vec<f32>, Vec<usize>)>) -> Result<Self> {
        let text_emb = Embedding {
            weight: take_mat(&weights, "text_emb.weight")?,
        };
        let mut audio_embs = Vec::with_capacity(cfg.audio_codebooks);
        for i in 0..cfg.audio_codebooks {
            audio_embs.push(Embedding {
                weight: take_mat(&weights, &format!("emb.{i}.weight"))?,
            });
        }
        let text_linear = take_mat(&weights, "text_linear.weight")?;
        let out_norm_alpha = take_vec1(&weights, "out_norm.alpha")?;
        let transformer = StreamingTransformer::build(&cfg.transformer, &weights)?;
        let depformer = match &cfg.depformer {
            None => None,
            Some(df) => Some(DepFormer::build(
                df,
                cfg.text_in_vocab_size,
                cfg.audio_vocab_size,
                cfg.transformer.d_model,
                &weights,
            )?),
        };
        Ok(Self {
            cfg,
            text_emb,
            audio_embs,
            text_linear,
            out_norm_alpha,
            transformer,
            depformer,
        })
    }

    pub fn config(&self) -> &LmConfig {
        &self.cfg
    }

    pub fn reset_state(&mut self) {
        self.transformer.reset_state();
    }

    pub fn text_start_token(&self) -> u32 {
        self.cfg.text_in_vocab_size as u32 - 1
    }

    pub fn audio_pad_token(&self) -> u32 {
        self.cfg.audio_vocab_size as u32 - 1
    }

    /// Single streaming step: sum embeddings → temporal transformer → text logits + hidden.
    pub fn forward_step(
        &mut self,
        text_token: Option<u32>,
        audio_tokens: &[Option<u32>],
    ) -> Result<(Array1<f32>, Array1<f32>)> {
        let d = self.cfg.transformer.d_model;
        let mut emb = vec![0.0f32; d];
        if let Some(tt) = text_token {
            let e = self.text_emb.forward_one(tt);
            for (i, v) in e.iter().enumerate() {
                emb[i] += v;
            }
        }
        for (cb, tok) in audio_tokens.iter().zip(self.audio_embs.iter()) {
            if let Some(t) = cb {
                let e = tok.forward_one(*t);
                for (i, v) in e.iter().enumerate() {
                    emb[i] += v;
                }
            }
        }
        let x = Array2::from_shape_vec((1, d), emb)?;
        let h = self.transformer.forward(&x);
        let normed = rms_norm(h.view(), &self.out_norm_alpha);
        let logits = linear(normed.view(), &self.text_linear);
        Ok((logits.row(0).to_owned(), h.row(0).to_owned()))
    }

    pub fn depformer_sample(
        &mut self,
        hidden: &Array1<f32>,
        text_token: Option<u32>,
        forced: &[Option<u32>],
        lp: &mut crate::sampling::LogitsProcessor,
    ) -> Result<Option<Vec<u32>>> {
        match self.depformer.as_mut() {
            None => Ok(None),
            Some(df) => Ok(Some(df.sample(hidden, text_token, forced, lp)?)),
        }
    }
}

fn take_mat(weights: &HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str) -> Result<Array2<f32>> {
    let (data, shape) = weights
        .get(key)
        .with_context(|| format!("missing weight {key}"))?;
    Ok(Array2::from_shape_vec((shape[0], shape[1]), data.clone())?)
}

fn take_vec1(weights: &HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str) -> Result<Array1<f32>> {
    let (data, shape) = weights
        .get(key)
        .with_context(|| format!("missing weight {key}"))?;
    let _n: usize = shape.iter().product();
    Ok(Array1::from_vec(data.clone()))
}
