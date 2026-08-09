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

//! llama.cpp reference logits via `llama-cpp-2` (optional `parity-llama` feature).

use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;
use std::path::Path;

/// Last-token logits top-K from llama.cpp for raw token ids (no tokenizer).
pub fn top_k_logits(path: &Path, prompt_ids: &[u32], top_k: usize) -> Result<Vec<(u32, f32)>> {
    let logits = last_token_logits(path, prompt_ids)?;
    let mut pairs: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(top_k);
    Ok(pairs)
}

/// Tokenize `text` with the GGUF's embedded tokenizer (BOS prepended) so
/// parity tests can feed a *valid* prompt to both llama.cpp and rlx.
pub fn tokenize(path: &Path, text: &str) -> Result<Vec<u32>> {
    let backend = LlamaBackend::init().context("LlamaBackend::init")?;
    let model_params = LlamaModelParams::default().with_n_gpu_layers(0);
    let model = LlamaModel::load_from_file(&backend, path, &model_params)
        .with_context(|| format!("load GGUF {}", path.display()))?;
    let toks = model
        .str_to_token(text, llama_cpp_2::model::AddBos::Always)
        .context("str_to_token")?;
    Ok(toks.into_iter().map(|t| t.0 as u32).collect())
}

/// Full last-token logit vector (length = model vocab).
pub fn last_token_logits(path: &Path, prompt_ids: &[u32]) -> Result<Vec<f32>> {
    anyhow::ensure!(!prompt_ids.is_empty(), "prompt_ids must be non-empty");

    let backend = LlamaBackend::init().context("LlamaBackend::init")?;
    let model_params = LlamaModelParams::default().with_n_gpu_layers(0);
    let model = LlamaModel::load_from_file(&backend, path, &model_params)
        .with_context(|| format!("load GGUF {}", path.display()))?;

    let n_ctx = NonZeroU32::new(4096).expect("4096 fits in NonZeroU32");
    let ctx_params = LlamaContextParams::default().with_n_ctx(Some(n_ctx));
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .context("new llama context")?;

    let mut batch = LlamaBatch::new(prompt_ids.len().max(8), 1);
    let last_index = prompt_ids.len() as i32 - 1;
    for (i, &tok) in prompt_ids.iter().enumerate() {
        batch.add(
            LlamaToken(tok as i32),
            i as i32,
            &[0],
            i as i32 == last_index,
        )?;
    }
    ctx.decode(&mut batch).context("llama decode prompt")?;

    let n_vocab = model.n_vocab();
    eprintln!("# llama.cpp n_vocab={n_vocab}");
    let logits = ctx.get_logits();
    assert_eq!(logits.len(), n_vocab as usize, "logits len vs n_vocab");
    Ok(logits.to_vec())
}

/// Post-`output_norm` hidden state for the last prompt token (`n_embd` dims).
pub fn last_token_hidden(path: &Path, prompt_ids: &[u32]) -> Result<Vec<f32>> {
    anyhow::ensure!(!prompt_ids.is_empty(), "prompt_ids must be non-empty");

    let backend = LlamaBackend::init().context("LlamaBackend::init")?;
    let model_params = LlamaModelParams::default().with_n_gpu_layers(0);
    let model = LlamaModel::load_from_file(&backend, path, &model_params)
        .with_context(|| format!("load GGUF {}", path.display()))?;

    let n_ctx = NonZeroU32::new(4096).expect("4096 fits in NonZeroU32");
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(n_ctx))
        .with_embeddings(true);
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .context("new llama context")?;

    let mut batch = LlamaBatch::new(prompt_ids.len().max(8), 1);
    let last_index = prompt_ids.len() as i32 - 1;
    for (i, &tok) in prompt_ids.iter().enumerate() {
        batch.add(
            LlamaToken(tok as i32),
            i as i32,
            &[0],
            i as i32 == last_index,
        )?;
    }
    ctx.decode(&mut batch).context("llama decode prompt")?;

    let n_embd = model.n_embd() as usize;
    let hidden = ctx
        .embeddings_ith(last_index)
        .context("llama embeddings_ith(last token)")?;
    assert_eq!(hidden.len(), n_embd, "hidden len vs n_embd");
    Ok(hidden.to_vec())
}

/// Greedy token continuation (same batch/decode loop as `rlx-neutts` llama-cpp backbone).
pub fn greedy_generation_ids(
    path: &Path,
    prompt_ids: &[u32],
    max_new_tokens: u32,
    n_ctx: u32,
) -> Result<Vec<u32>> {
    use llama_cpp_2::sampling::LlamaSampler;

    anyhow::ensure!(!prompt_ids.is_empty(), "prompt_ids must be non-empty");
    let n_ctx_nz = NonZeroU32::new(n_ctx.max(prompt_ids.len() as u32 + max_new_tokens))
        .context("n_ctx must be non-zero")?;

    let backend = LlamaBackend::init().context("LlamaBackend::init")?;
    let model_params = LlamaModelParams::default().with_n_gpu_layers(0);
    let model = LlamaModel::load_from_file(&backend, path, &model_params)
        .with_context(|| format!("load GGUF {}", path.display()))?;

    let ctx_params = LlamaContextParams::default().with_n_ctx(Some(n_ctx_nz));
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .context("new llama context")?;

    let mut batch = LlamaBatch::new(prompt_ids.len().max(1), 1);
    let last_idx = prompt_ids.len() as i32 - 1;
    for (i, &tok) in prompt_ids.iter().enumerate() {
        batch.add(LlamaToken(tok as i32), i as i32, &[0], i as i32 == last_idx)?;
    }
    ctx.decode(&mut batch).context("llama decode prompt")?;

    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    let mut out: Vec<u32> = Vec::with_capacity(max_new_tokens as usize);

    for n_cur in (prompt_ids.len() as i32..).take(max_new_tokens as usize) {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        out.push(token.0 as u32);
        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        ctx.decode(&mut batch).context("llama decode step")?;
    }
    Ok(out)
}

/// Logit for a single token id from the last prompt position.
pub fn token_logit(path: &Path, prompt_ids: &[u32], token: u32) -> Result<f32> {
    let logits = last_token_logits(path, prompt_ids)?;
    logits
        .get(token as usize)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("token {token} out of vocab range {}", logits.len()))
}
