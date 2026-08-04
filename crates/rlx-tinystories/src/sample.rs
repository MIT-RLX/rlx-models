//! Autoregressive text generation from a trained model.

use rlx_tensor::{DType, Device};

use crate::config::GptConfig;
use crate::model;
use crate::rng::Rng;
use crate::tokenizer;

/// Sampling knobs.
#[derive(Clone, Copy, Debug)]
pub struct GenOptions {
    pub max_new_tokens: usize,
    /// Softmax temperature (`<= 0` ⇒ greedy/argmax).
    pub temperature: f32,
    /// Keep only the top-`k` logits before sampling (`0` ⇒ no cap).
    pub top_k: usize,
    pub seed: u64,
}

impl Default for GenOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 400,
            temperature: 0.8,
            top_k: 40,
            seed: 1,
        }
    }
}

/// Bind trained `params` into a fresh batch-1 forward graph and generate text
/// continuing `prompt` (byte-level). Returns the full text (prompt + sample).
pub fn generate(
    cfg: &GptConfig,
    params: &[(String, Vec<f32>)],
    prompt: &str,
    dev: Device,
    opts: &GenOptions,
    bpe: Option<&crate::bpe::Bpe>,
) -> String {
    let t = cfg.block_size;
    let v = cfg.vocab;

    // Inference model: batch = 1, logits output (no loss).
    let mut icfg = *cfg;
    icfg.batch = 1;
    let mut model = model::build(&icfg, 1, false, DType::F32);
    for (name, data) in params {
        model = model.with_param(name.clone(), data.clone());
    }

    // Seed with the prompt tokens (or a newline if empty, to give a start).
    let encode = |s: &str| match bpe {
        Some(b) => b.encode(s.as_bytes()),
        None => tokenizer::encode(s),
    };
    let mut ids: Vec<u32> = if prompt.is_empty() {
        encode("\n")
    } else {
        encode(prompt)
    };

    let mut rng = Rng::new(opts.seed);

    for _ in 0..opts.max_new_tokens {
        let ctx_len = ids.len().min(t);
        let window = &ids[ids.len() - ctx_len..];

        // Token ids for the [T]-wide window (fed as f32; the model casts to i64
        // and gathers). Positions past the context stay id 0 — the causal mask
        // means the read position (ctx_len-1) never attends to them.
        let mut tok = vec![0f32; t];
        for (p, &id) in window.iter().enumerate() {
            tok[p] = id as f32;
        }
        let feed: &[(&str, &[f32])] = &[("tok_ids", &tok)];
        let logits = model.run_on(dev, feed).remove(0); // [T*V]

        let row = (ctx_len - 1) * v;
        let next = sample_row(&logits[row..row + v], opts, &mut rng);
        ids.push(next as u32);
    }

    match bpe {
        Some(b) => b.decode(&ids),
        None => tokenizer::decode(&ids),
    }
}

/// Pick a token from one row of logits with temperature + top-k.
fn sample_row(logits: &[f32], opts: &GenOptions, rng: &mut Rng) -> usize {
    // Greedy when temperature is off or k == 1.
    if opts.temperature <= 0.0 || opts.top_k == 1 {
        return argmax(logits);
    }
    let inv_t = 1.0 / opts.temperature;

    // Rank indices by logit, keep the top-k.
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    let k = if opts.top_k == 0 {
        logits.len()
    } else {
        opts.top_k.min(logits.len())
    };
    idx.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);

    // Softmax over the kept logits (temperature-scaled, max-shifted).
    let max = idx.iter().map(|&i| logits[i]).fold(f32::MIN, f32::max);
    let mut probs: Vec<f32> = idx
        .iter()
        .map(|&i| ((logits[i] - max) * inv_t).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    // Multinomial draw.
    let r = rng.uniform() as f32;
    let mut acc = 0.0;
    for (j, &p) in probs.iter().enumerate() {
        acc += p;
        if r <= acc {
            return idx[j];
        }
    }
    idx[k - 1]
}

fn argmax(xs: &[f32]) -> usize {
    xs.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
