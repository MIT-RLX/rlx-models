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

//! llama.cpp reference backbone (`llama-cpp-2`) — parity / cross-check only.
//!
//! Enable with the `parity-llama-cpp` feature. Production inference uses
//! [`super::rlx::BackboneModel`] ([`rlx_llama32::Llama32Runner`]).

use std::path::Path;

use anyhow::{Context, Result};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel, params::LlamaModelParams},
    sampling::LlamaSampler,
    token::LlamaToken,
};

use crate::tokens::STOP_TOKEN;

/// Reference backbone for parity tests (llama-cpp-2).
pub struct LlamaCppBackbone {
    _backend: LlamaBackend,
    model: LlamaModel,
    n_ctx: u32,
    pub seed: Option<u32>,
}

impl LlamaCppBackbone {
    fn neutts_sampler(seed: u32) -> LlamaSampler {
        LlamaSampler::chain_simple([
            LlamaSampler::top_k(50),
            LlamaSampler::top_p(0.9, 1),
            LlamaSampler::temp(1.0),
            LlamaSampler::dist(seed),
        ])
    }

    pub fn load(path: &Path, n_ctx: u32) -> Result<Self> {
        let backend = LlamaBackend::init().context("Failed to initialise llama.cpp backend")?;
        // CPU-only for deterministic parity with RLX (matches rlx-qwen35 llama_reference).
        let model_params = LlamaModelParams::default().with_n_gpu_layers(0);
        let model = LlamaModel::load_from_file(&backend, path, &model_params)
            .with_context(|| format!("Cannot load GGUF model: {}", path.display()))?;
        Ok(Self {
            _backend: backend,
            model,
            n_ctx,
            seed: None,
        })
    }

    /// Greedy token IDs from a pre-encoded prompt (GGUF vocab — matches RLX encode path).
    pub fn generate_greedy_ids(&self, prompt_ids: &[u32], max_new_tokens: u32) -> Result<Vec<u32>> {
        let n_ctx = std::num::NonZeroU32::new(self.n_ctx).context("n_ctx must be non-zero")?;
        let ctx_params = LlamaContextParams::default().with_n_ctx(Some(n_ctx));
        let mut ctx = self
            .model
            .new_context(&self._backend, ctx_params)
            .context("Failed to create llama.cpp context")?;

        if prompt_ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut batch = LlamaBatch::new(prompt_ids.len().max(1), 1);
        let last_idx = prompt_ids.len() as i32 - 1;
        for (i, &tok) in prompt_ids.iter().enumerate() {
            batch.add(LlamaToken(tok as i32), i as i32, &[0], i as i32 == last_idx)?;
        }
        ctx.decode(&mut batch).context("Prompt decode failed")?;

        let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
        let mut n_cur = prompt_ids.len() as i32;
        let mut out: Vec<u32> = Vec::with_capacity(max_new_tokens as usize);

        for _ in 0..max_new_tokens {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if self.model.is_eog_token(token) {
                break;
            }
            out.push(token.0 as u32);
            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            ctx.decode(&mut batch).context("Decode step failed")?;
            n_cur += 1;
        }
        Ok(out)
    }

    /// Greedy generation for parity vs [`super::rlx::BackboneModel::generate_greedy`].
    pub fn generate_greedy(&self, prompt: &str, max_new_tokens: u32) -> Result<String> {
        let mut output = String::new();
        self.generate_streaming_greedy(prompt, max_new_tokens, |piece| {
            output.push_str(piece);
            Ok(())
        })?;
        Ok(output)
    }

    pub fn generate(&self, prompt: &str, max_new_tokens: u32) -> Result<String> {
        let mut output = String::new();
        self.generate_streaming(prompt, max_new_tokens, |piece| {
            output.push_str(piece);
            Ok(())
        })?;
        Ok(output)
    }

    pub fn generate_streaming<F>(
        &self,
        prompt: &str,
        max_new_tokens: u32,
        mut on_piece: F,
    ) -> Result<()>
    where
        F: FnMut(&str) -> Result<()>,
    {
        let n_ctx = std::num::NonZeroU32::new(self.n_ctx).context("n_ctx must be non-zero")?;
        let ctx_params = LlamaContextParams::default().with_n_ctx(Some(n_ctx));
        let mut ctx = self
            .model
            .new_context(&self._backend, ctx_params)
            .context("Failed to create llama.cpp context")?;

        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .context("Tokenisation failed")?;

        eprintln!(
            "[backbone/llama-cpp] prompt tokens: {} / n_ctx={}",
            tokens.len(),
            self.n_ctx
        );
        if tokens.len() as u32 > self.n_ctx {
            anyhow::bail!(
                "Prompt too long: {} tokens exceeds n_ctx={}",
                tokens.len(),
                self.n_ctx
            );
        }
        if tokens.is_empty() {
            return Ok(());
        }

        let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
        let last_idx = tokens.len() as i32 - 1;
        for (i, &tok) in tokens.iter().enumerate() {
            batch.add(tok, i as i32, &[0], i as i32 == last_idx)?;
        }
        ctx.decode(&mut batch).context("Prompt decode failed")?;

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let seed = self.seed.unwrap_or_else(rand::random);
        let mut sampler = Self::neutts_sampler(seed);

        let mut n_cur = tokens.len() as i32;
        let max_cur = n_cur + max_new_tokens as i32;

        loop {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                break;
            }

            let piece = self
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .map_err(|e| anyhow::anyhow!("token decode error: {e}"))?;

            if let Some(pos) = piece.find(STOP_TOKEN) {
                let before = &piece[..pos];
                if !before.is_empty() {
                    on_piece(before)?;
                }
                break;
            }

            on_piece(&piece)?;

            if n_cur >= max_cur {
                break;
            }

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            ctx.decode(&mut batch).context("Decode step failed")?;
            n_cur += 1;
        }

        Ok(())
    }

    fn generate_streaming_greedy<F>(
        &self,
        prompt: &str,
        max_new_tokens: u32,
        mut on_piece: F,
    ) -> Result<()>
    where
        F: FnMut(&str) -> Result<()>,
    {
        let n_ctx = std::num::NonZeroU32::new(self.n_ctx).context("n_ctx must be non-zero")?;
        let ctx_params = LlamaContextParams::default().with_n_ctx(Some(n_ctx));
        let mut ctx = self
            .model
            .new_context(&self._backend, ctx_params)
            .context("Failed to create llama.cpp context")?;

        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .context("Tokenisation failed")?;
        if tokens.is_empty() {
            return Ok(());
        }

        let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
        let last_idx = tokens.len() as i32 - 1;
        for (i, &tok) in tokens.iter().enumerate() {
            batch.add(tok, i as i32, &[0], i as i32 == last_idx)?;
        }
        ctx.decode(&mut batch).context("Prompt decode failed")?;

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);

        let mut n_cur = tokens.len() as i32;
        let max_cur = n_cur + max_new_tokens as i32;

        loop {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if self.model.is_eog_token(token) {
                break;
            }
            let piece = self
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .map_err(|e| anyhow::anyhow!("token decode error: {e}"))?;
            if let Some(pos) = piece.find(STOP_TOKEN) {
                let before = &piece[..pos];
                if !before.is_empty() {
                    on_piece(before)?;
                }
                break;
            }
            on_piece(&piece)?;
            if n_cur >= max_cur {
                break;
            }
            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            ctx.decode(&mut batch).context("Decode step failed")?;
            n_cur += 1;
        }
        Ok(())
    }
}
