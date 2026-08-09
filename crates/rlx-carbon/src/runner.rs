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

//! [`CarbonRunner`] — the Carbon DNA LM surface, wrapping the shared
//! [`rlx_llama32::Llama32Runner`] backbone plus (with the `tokenizer` feature)
//! the native `HybridDnaTokenizer`.

use anyhow::{Context, Result, anyhow};
use rlx_llama32::{Llama32Config, Llama32Runner, SampleOpts};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

/// Builder for [`CarbonRunner`]. Mirrors the relevant knobs of
/// [`rlx_llama32::Llama32RunnerBuilder`].
#[derive(Debug, Clone, Default)]
pub struct CarbonRunnerBuilder {
    weights: Option<PathBuf>,
    device: Option<Device>,
    max_seq: Option<usize>,
    sample: Option<SampleOpts>,
    packed_weights: Option<bool>,
}

impl CarbonRunnerBuilder {
    /// Path to a Carbon model directory (containing `config.json`,
    /// `model.safetensors`, `tokenizer.json`, `dna_config.json`), or directly
    /// to the `model.safetensors` / `.gguf` file inside it.
    pub fn weights(mut self, path: impl Into<PathBuf>) -> Self {
        self.weights = Some(path.into());
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }
    pub fn max_seq(mut self, n: usize) -> Self {
        self.max_seq = Some(n);
        self
    }
    pub fn sample(mut self, opts: SampleOpts) -> Self {
        self.sample = Some(opts);
        self
    }
    pub fn packed_weights(mut self, on: bool) -> Self {
        self.packed_weights = Some(on);
        self
    }

    pub fn build(self) -> Result<CarbonRunner> {
        let weights = self
            .weights
            .ok_or_else(|| anyhow!("weights path required (call .weights(...))"))?;
        let device = self.device.unwrap_or(Device::Cpu);
        let max_seq = self.max_seq.unwrap_or(512);

        // Resolve the model directory (holds config.json + tokenizer files).
        // `weights` may be the directory itself or a file inside it. The inner
        // Llama32Runner resolves safetensors config.json from the weights
        // *parent*, which is wrong when `weights` is the directory — so pin the
        // config source to `<dir>/config.json` explicitly (harmless for GGUF,
        // which carries its config in-band and ignores an unused JsonFile only
        // when it is safetensors).
        let dir = if weights.is_dir() {
            weights.clone()
        } else {
            weights
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| std::path::PathBuf::from("."))
        };

        // The shared safetensors loader accepts a directory only for *sharded*
        // checkpoints (via `model.safetensors.index.json`); a lone
        // `model.safetensors` must be passed as the file. Resolve accordingly.
        let weights_arg = resolve_weights_arg(&weights, &dir)?;

        let mut b = Llama32Runner::builder()
            .weights(weights_arg)
            .device(device)
            .max_seq(max_seq)
            .stream(false);
        let cfg_json = dir.join("config.json");
        if cfg_json.is_file() {
            b = b.config(rlx_llama32::Llama32ConfigSource::JsonFile(cfg_json));
        }
        if let Some(s) = self.sample {
            b = b.sample(s);
        }
        if let Some(p) = self.packed_weights {
            b = b.packed_weights(p);
        }
        let inner = b
            .build()
            .context("rlx-carbon: building Llama32Runner backbone")?;
        let cfg = inner.config().clone();

        #[cfg(feature = "tokenizer")]
        let tokenizer = crate::HybridDnaTokenizer::from_dir(&dir)
            .with_context(|| format!("rlx-carbon: loading tokenizer from {}", dir.display()))?;

        Ok(CarbonRunner {
            inner,
            cfg,
            #[cfg(feature = "tokenizer")]
            tokenizer,
        })
    }
}

/// Carbon DNA language-model runner.
pub struct CarbonRunner {
    inner: Llama32Runner,
    cfg: Llama32Config,
    #[cfg(feature = "tokenizer")]
    tokenizer: crate::HybridDnaTokenizer,
}

impl CarbonRunner {
    pub fn builder() -> CarbonRunnerBuilder {
        CarbonRunnerBuilder::default()
    }

    /// Transformers-style one-liner: load a Carbon model directory on `device`.
    pub fn from_pretrained(dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        Self::builder()
            .weights(dir.as_ref().to_path_buf())
            .device(device)
            .build()
    }

    /// Underlying Llama config (dims, RoPE, GQA, vocab).
    pub fn config(&self) -> &Llama32Config {
        &self.cfg
    }

    /// Borrow the backbone runner for advanced prefill/decode control.
    pub fn inner_mut(&mut self) -> &mut Llama32Runner {
        &mut self.inner
    }

    /// Swap sampling (greedy / temperature / top-p) without rebuilding.
    pub fn set_sample(&mut self, opts: SampleOpts) {
        self.inner.set_sample(opts);
    }

    /// KV-cached generation from raw token ids; `on_token` sees each new id.
    pub fn generate_ids(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        self.inner.generate(prompt_ids, n_new, on_token)
    }

    /// KV-cached generation from ids that stops early when `keep_going` returns
    /// `false` (e.g. on the end-of-sequence id).
    pub fn generate_ids_until(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        keep_going: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        self.inner.generate_until(prompt_ids, n_new, keep_going)
    }

    /// Last-position logits after a single prefill.
    pub fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        self.inner.predict_logits(prompt_ids)
    }
}

#[cfg(feature = "tokenizer")]
impl CarbonRunner {
    /// Access the hybrid DNA tokenizer.
    pub fn tokenizer(&self) -> &crate::HybridDnaTokenizer {
        &self.tokenizer
    }

    /// Encode text/DNA to token ids (respects `dna_config.auto_dna_tags`).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text)
    }

    /// Decode token ids back to text/DNA.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer.decode(ids, skip_special_tokens)
    }

    /// Streaming completion from prompt ids: generates up to `max_new_tokens`,
    /// stopping on the end-of-sequence id (which is not emitted). `on_text` is
    /// called with each freshly-decoded text fragment (the running continuation
    /// re-decoded and diffed). Returns the generated ids (eos excluded).
    ///
    /// Implemented here so the base-tokenizer borrow (`&self.tokenizer`) and the
    /// backbone borrow (`&mut self.inner`) stay field-disjoint.
    pub fn generate_ids_streaming(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        skip_special_tokens: bool,
        mut on_text: impl FnMut(&str),
    ) -> Result<Vec<u32>> {
        let eos = self.tokenizer.eos_id();
        let tok = &self.tokenizer;
        let mut generated: Vec<u32> = Vec::new();
        let mut last_len = 0usize;
        self.inner.generate_until(prompt_ids, max_new_tokens, |t| {
            if t == eos {
                return false;
            }
            generated.push(t);
            if let Ok(text) = tok.decode(&generated, skip_special_tokens) {
                if text.len() >= last_len {
                    on_text(&text[last_len..]);
                } else {
                    on_text(&text);
                }
                last_len = text.len();
            }
            true
        })?;
        Ok(generated)
    }

    /// One-shot DNA/text completion. When `as_dna` is `Some(true)` (or `None`
    /// and the config's `auto_dna_tags` is set) and the prompt has no `<dna>`
    /// tag, the prompt is treated as an **open** DNA region (`<dna>…`, no
    /// closing tag) so the model *continues* the nucleotide sequence — a closed
    /// `</dna>` would signal completion and yield an immediate end-of-sequence.
    /// Prompts that already contain `<dna>` tags are honored verbatim.
    /// Generation stops on the end-of-sequence id or after `max_new_tokens`.
    pub fn complete(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        as_dna: Option<bool>,
    ) -> Result<CarbonCompletion> {
        let has_tag = prompt.contains("<dna>");
        let treat_dna = !has_tag && as_dna.unwrap_or(self.tokenizer.dna_config().auto_dna_tags);
        let owned;
        let prompt_text = if treat_dna {
            owned = format!("<dna>{prompt}");
            owned.as_str()
        } else {
            prompt
        };
        let prompt_ids = self.tokenizer.encode_opt(prompt_text, Some(false))?;
        let eos = self.tokenizer.eos_id();
        let generated = self
            .inner
            .generate_until(&prompt_ids, max_new_tokens, |tok| tok != eos)?;
        // Drop a trailing eos so it isn't rendered.
        let trimmed: Vec<u32> = generated
            .iter()
            .copied()
            .take_while(|&t| t != eos)
            .collect();
        let text = self.tokenizer.decode(&trimmed, true)?;
        Ok(CarbonCompletion {
            prompt_ids,
            generated,
            text,
        })
    }
}

/// Resolve the weights argument to hand the backbone loader: the directory
/// itself for sharded checkpoints, else the concrete `model.safetensors` /
/// `*.safetensors` / `*.gguf` file inside it. A non-directory `weights` is
/// returned unchanged.
fn resolve_weights_arg(weights: &Path, dir: &Path) -> Result<PathBuf> {
    if !weights.is_dir() {
        return Ok(weights.to_path_buf());
    }
    // Sharded HF checkpoint — the mmap loader accepts the directory.
    if dir.join("model.safetensors.index.json").is_file() {
        return Ok(dir.to_path_buf());
    }
    let single = dir.join("model.safetensors");
    if single.is_file() {
        return Ok(single);
    }
    // Fall back to a lone .safetensors or .gguf file in the directory.
    let (mut st, mut gguf) = (None, None);
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            match p.extension().and_then(|s| s.to_str()) {
                Some("safetensors") if st.is_none() => st = Some(p),
                Some("gguf") if gguf.is_none() => gguf = Some(p),
                _ => {}
            }
        }
    }
    st.or(gguf)
        .ok_or_else(|| anyhow!("no model.safetensors / *.gguf found in {}", dir.display()))
}

/// Result of [`CarbonRunner::complete`].
#[cfg(feature = "tokenizer")]
#[derive(Debug, Clone)]
pub struct CarbonCompletion {
    /// Encoded prompt ids.
    pub prompt_ids: Vec<u32>,
    /// Newly generated ids (may include a trailing eos).
    pub generated: Vec<u32>,
    /// Decoded continuation (eos trimmed, DNA special tokens skipped).
    pub text: String,
}
