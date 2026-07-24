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

//! NeuTTS GGUF backbone via [`rlx_llama32::Llama32Runner`] (rlx-models).
//!
//! Tokenisation uses the GGUF embedded vocab ([`rlx_qwen35::encode_prompt_from_gguf`]
//! / [`rlx_qwen35::decode_ids_from_gguf`], same path as `rlx-llama32` CLI).
//! Sampling matches the original NeuTTS defaults: top-k=50, top-p=0.9, temp=1.0.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use rlx_core::validate_standard_device;
use rlx_llama_base::LlamaBaseConfig;
use rlx_llama32::{Llama32Runner, Llama32RunnerBuilder};
use rlx_qwen3::{SampleOpts, sample_token};
use rlx_qwen35::{decode_ids_from_gguf, encode_prompt_from_gguf};
use rlx_runtime::Device;

use crate::tokens::STOP_TOKEN;

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// Default context window (must match Python's `max_context = 2048`).
pub const DEFAULT_N_CTX: u32 = 2048;

/// NeuTTS backbone — RLX Llama-3.2 runner over a llama-tagged GGUF.
pub struct BackboneModel {
    runner: Mutex<Llama32Runner>,
    weights: PathBuf,
    n_ctx: u32,
    pub seed: Option<u32>,
    /// When true, greedy parity uses incremental prefill+decode (llama.cpp-shaped).
    #[allow(dead_code)]
    greedy_parity: bool,
    _arch: String,
}

impl BackboneModel {
    pub fn load(path: &Path, n_ctx: u32) -> Result<Self> {
        Self::load_on(path, n_ctx, Device::Cpu)
    }

    /// Load the GGUF backbone on a specific execution device.
    ///
    /// Packed K-quant (`Op::DequantMatMul`) is preferred on every device —
    /// Metal / MLX / wgpu match CPU for Q4_K / Q6_K. NeuTTS Nano/Air stay
    /// small either way; greedy-parity (F32) remains available via
    /// [`Self::load_greedy_parity_on`].
    pub fn load_on(path: &Path, n_ctx: u32, device: Device) -> Result<Self> {
        Self::load_inner(
            path, n_ctx, /*packed*/ true, device, /*greedy_parity*/ false,
        )
    }

    /// F32 dequant + incremental greedy (tail parity vs llama-cpp Q4).
    pub fn load_greedy_parity(path: &Path, n_ctx: u32) -> Result<Self> {
        Self::load_greedy_parity_on(path, n_ctx, Device::Cpu)
    }

    /// Greedy parity load on a specific execution device.
    pub fn load_greedy_parity_on(path: &Path, n_ctx: u32, device: Device) -> Result<Self> {
        Self::load_inner(path, n_ctx, false, device, true)
    }

    fn load_inner(
        path: &Path,
        n_ctx: u32,
        packed_weights: bool,
        device: Device,
        greedy_parity: bool,
    ) -> Result<Self> {
        validate_standard_device("neutts", device)?;
        let base = LlamaBaseConfig::from_gguf_path(path)
            .with_context(|| format!("parse GGUF {:?}", path))?;
        // NeuTTS Nano/Air GGUFs are llama-tagged (same layout as LLaMA 3.2 / Bonsai).
        if base.arch != "llama" {
            bail!(
                "rlx-neutts: expected `general.architecture = llama` in {}; got `{}`. \
                 Point at a NeuTTS / Llama-shaped GGUF.",
                path.display(),
                base.arch
            );
        }

        let runner = Llama32RunnerBuilder::default()
            .weights(path)
            .max_seq(n_ctx as usize)
            .device(device)
            .packed_weights(packed_weights)
            .sample(SampleOpts::greedy())
            .build()
            .context("build Llama32Runner for NeuTTS backbone")?;

        eprintln!(
            "[backbone/rlx-llama32] loaded {} (hidden={}, layers={})",
            path.display(),
            base.hidden_size,
            base.num_hidden_layers
        );

        Ok(Self {
            runner: Mutex::new(runner),
            weights: path.to_path_buf(),
            n_ctx,
            seed: None,
            greedy_parity,
            _arch: base.arch,
        })
    }

    fn sample_opts(&self) -> SampleOpts {
        let seed = self.seed.map(u64::from).unwrap_or_else(rand::random);
        SampleOpts::temperature(1.0, seed)
            .with_top_k(50)
            .with_top_p(0.9)
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
        let prompt_ids = encode_prompt_from_gguf(&self.weights, prompt)
            .with_context(|| format!("tokenize prompt for {}", self.weights.display()))?;

        eprintln!(
            "[backbone/rlx-llama32] prompt token count: {} / n_ctx={}",
            prompt_ids.len(),
            self.n_ctx
        );
        if prompt_ids.len() as u32 > self.n_ctx {
            bail!(
                "Prompt too long: {} tokens exceeds n_ctx={}",
                prompt_ids.len(),
                self.n_ctx
            );
        }
        if prompt_ids.is_empty() {
            return Ok(());
        }

        let mut ids = prompt_ids;
        let sample = self.sample_opts();
        let mut runner = self
            .runner
            .lock()
            .map_err(|e| anyhow::anyhow!("backbone runner lock poisoned: {e}"))?;

        for _ in 0..max_new_tokens {
            let logits = runner
                .predict_logits(&ids)
                .context("RLX backbone predict_logits failed")?;
            let next = sample_token(&logits, sample) as u32;

            let piece = decode_ids_from_gguf(&self.weights, std::slice::from_ref(&next), true)
                .with_context(|| format!("decode token {next}"))?;

            if piece.is_empty() {
                ids.push(next);
                continue;
            }

            if let Some(pos) = piece.find(STOP_TOKEN) {
                let before = &piece[..pos];
                if !before.is_empty() {
                    on_piece(before)?;
                }
                break;
            }

            on_piece(&piece)?;
            ids.push(next);
        }

        Ok(())
    }

    /// Greedy token IDs for parity tests (same GGUF vocab as production).
    pub fn generate_greedy_ids(&self, prompt: &str, max_new_tokens: u32) -> Result<Vec<u32>> {
        let prompt_ids = encode_prompt_from_gguf(&self.weights, prompt)?;
        self.generate_greedy_ids_from_prompt(&prompt_ids, max_new_tokens)
    }

    /// Greedy continuation for parity tests.
    ///
    /// [`load_greedy_parity`] uses KV-cached [`Llama32Runner::generate`] (F32 weights,
    /// MSVC uses oneshot decode in `step_cached`). Production [`load`] uses packed Q4.
    /// Debug: `NEUTTS_GREEDY_INCREMENTAL=1` or `NEUTTS_GREEDY_PREDICT_LOGITS=1`.
    pub fn generate_greedy_ids_from_prompt(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: u32,
    ) -> Result<Vec<u32>> {
        let mut runner = self.runner.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
        let n = max_new_tokens as usize;
        if env_truthy("NEUTTS_GREEDY_PREDICT_LOGITS") {
            let opts = SampleOpts::greedy();
            let mut history = prompt_ids.to_vec();
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                let logits = runner
                    .predict_logits(&history)
                    .context("greedy parity predict_logits")?;
                let next = sample_token(&logits, opts) as u32;
                out.push(next);
                history.push(next);
            }
            return Ok(out);
        }
        runner.generate(prompt_ids, n, |_| {})
    }

    /// Greedy text generation for parity tests (deterministic vs llama.cpp reference).
    pub fn generate_greedy(&self, prompt: &str, max_new_tokens: u32) -> Result<String> {
        let new_ids = self.generate_greedy_ids(prompt, max_new_tokens)?;
        let mut out = String::new();
        for &tok in &new_ids {
            let piece = decode_ids_from_gguf(&self.weights, std::slice::from_ref(&tok), true)?;
            if piece.find(STOP_TOKEN).is_some() {
                break;
            }
            out.push_str(&piece);
        }
        Ok(out)
    }
}
