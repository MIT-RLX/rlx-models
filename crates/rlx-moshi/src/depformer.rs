use crate::config::DepFormerConfig;
use crate::nn::{Embedding, linear};
use crate::sampling::LogitsProcessor;
use crate::transformer::StreamingTransformer;
use anyhow::{Context, Result, ensure};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

#[derive(Debug)]
struct DepSlice {
    linear_in: Array2<f32>,
    linear_out: Array2<f32>,
    emb: Embedding,
    transformer: StreamingTransformer,
}

/// Depth decoder — one mini-transformer per audio codebook slice.
pub struct DepFormer {
    slices: Vec<DepSlice>,
}

impl DepFormer {
    pub fn build(
        cfg: &DepFormerConfig,
        text_vocab_size: usize,
        audio_vocab_size: usize,
        _main_dim: usize,
        weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    ) -> Result<Self> {
        let mut slices = Vec::with_capacity(cfg.num_slices);
        for si in 0..cfg.num_slices {
            let prefix = format!("depformer.{si}.");
            let in_vs = if si == 0 {
                text_vocab_size
            } else {
                audio_vocab_size
            };
            let emb = Embedding {
                weight: take_mat(weights, &format!("{prefix}emb.weight"))?,
            };
            ensure!(emb.weight.dim().0 >= in_vs, "depformer.{si}.emb too small");
            let linear_in = take_mat(weights, &format!("{prefix}linear_in.weight"))?;
            let linear_out = take_mat(weights, &format!("{prefix}linear_out.weight"))?;
            let transformer =
                StreamingTransformer::build_prefixed(&cfg.transformer, &prefix, weights)?;
            slices.push(DepSlice {
                linear_in,
                linear_out,
                emb,
                transformer,
            });
        }
        Ok(Self { slices })
    }

    pub fn sample(
        &mut self,
        temporal_hidden: &Array1<f32>,
        text_token: Option<u32>,
        forced: &[Option<u32>],
        lp: &mut LogitsProcessor,
    ) -> Result<Vec<u32>> {
        let mut tokens = Vec::with_capacity(self.slices.len());
        let mut last_token = text_token;
        let xs = Array2::from_shape_vec((1, temporal_hidden.len()), temporal_hidden.to_vec())?;
        for si in 0..self.slices.len() {
            if si == 0 {
                self.slices[si].transformer.reset_state();
            } else {
                let prev = self.slices[si - 1].transformer.clone_kv_snapshot();
                self.slices[si].transformer.restore_kv_snapshot(&prev);
            }
            let slice = &mut self.slices[si];
            let mut h = linear(xs.view(), &slice.linear_in);
            if let Some(tok) = last_token {
                let emb = slice.emb.forward_one(tok);
                for di in 0..emb.len() {
                    h[[0, di]] += emb[di];
                }
            }
            let h = slice.transformer.forward(&h);
            let logits = linear(h.view(), &slice.linear_out);
            let row = logits.row(0);
            let token = lp.sample(row)?;
            tokens.push(token);
            let next = forced.get(si).copied().flatten().unwrap_or(token);
            last_token = Some(next);
        }
        Ok(tokens)
    }
}

fn take_mat(weights: &HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str) -> Result<Array2<f32>> {
    let (data, shape) = weights
        .get(key)
        .with_context(|| format!("missing weight {key}"))?;
    Ok(Array2::from_shape_vec((shape[0], shape[1]), data.clone())?)
}
