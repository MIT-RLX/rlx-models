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

//! Bonsai runner — dispatches by GGUF architecture.
//!
//! "Bonsai" spans two unrelated model lineages that share only the name:
//!
//! * **Bonsai small-reasoning family (1.7B / 2B / 4B / 8B)** — ships as
//!   `general.architecture = llama`, a standard Llama-shaped decoder
//!   with hyperparameters tuned for small-context reasoning. Runs
//!   through [`rlx_llama32::Llama32Runner`] (see [`BonsaiRunner`]).
//! * **[`prism-ml/Bonsai-27B`](https://huggingface.co/prism-ml/Bonsai-27B-gguf)**
//!   — a Qwen3.6-27B derivative (`general.architecture = qwen35`) with
//!   hybrid gated-DeltaNet + full attention and custom 1-bit `Q1_0`
//!   weights (~1.125 bpw). Runs through [`rlx_qwen35::Qwen35Runner`].
//! * **[`prism-ml/Ternary-Bonsai-27B`](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf)**
//!   — same hybrid arch with ternary `Q2_0` weights (~2.125 bpw).
//!
//! [`cli_run`] sniffs `general.architecture` from the `--weights` GGUF
//! header (cheap — no tensor-data slurp, even on the multi-GB packed
//! files) and routes to the matching runner. [`detect_arch`] exposes the
//! same classification programmatically. [`BonsaiRunner`] itself is the
//! Llama-shaped small-family runner; it rejects a `qwen35` GGUF with a
//! pointer to the qwen35 path.

use anyhow::{Context, Result, anyhow, bail};
use rlx_gguf::{GgufFile, MetaValue};
use rlx_llama_base::LlamaBaseConfig;
use std::path::{Path, PathBuf};

pub use rlx_llama32::{Llama32ConfigSource, Llama32Runner, Llama32RunnerBuilder};

pub const PLAN_MILESTONE: &str = "M4";
pub const FAMILY: &str = "Bonsai";

/// Which Bonsai lineage a GGUF belongs to — see the crate docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BonsaiArch {
    /// Small-reasoning family (1.7B–8B), `general.architecture = llama`.
    /// Handled by [`rlx_llama32::Llama32Runner`].
    LlamaSmall,
    /// `prism-ml/Bonsai-27B` / Ternary-Bonsai-27B: Qwen3.6-derived hybrid,
    /// `qwen35` / `qwen36` tags. Handled by [`rlx_qwen35::Qwen35Runner`].
    Qwen35Hybrid,
}

/// Classify a GGUF `general.architecture` tag. Returns `None` for tags
/// that aren't a known Bonsai lineage.
pub fn bonsai_arch_from_tag(arch: &str) -> Option<BonsaiArch> {
    match arch {
        "llama" => Some(BonsaiArch::LlamaSmall),
        "qwen35" | "qwen35moe" | "qwen36" | "qwen36moe" => Some(BonsaiArch::Qwen35Hybrid),
        _ => None,
    }
}

/// Read `general.architecture` from a GGUF header (metadata only — no
/// tensor-data slurp) and classify it as a Bonsai lineage. Errors on a
/// tag that is neither `llama` nor a `qwen35`-family variant.
pub fn detect_arch(weights: &Path) -> Result<BonsaiArch> {
    let raw = GgufFile::header_from_path(weights)
        .with_context(|| format!("rlx-bonsai: reading GGUF header {weights:?}"))?;
    let arch = raw
        .metadata
        .get("general.architecture")
        .and_then(MetaValue::as_str)
        .ok_or_else(|| anyhow!("rlx-bonsai: {weights:?} missing general.architecture"))?;
    bonsai_arch_from_tag(arch).ok_or_else(|| {
        anyhow!(
            "rlx-bonsai: unexpected general.architecture=`{arch}` at {weights:?} — expected \
             `llama` (Bonsai 1.7B–8B) or `qwen35` (prism-ml/Bonsai-27B)"
        )
    })
}

/// Per-family runner for the Llama-shaped small Bonsai (1.7B–8B). Wraps
/// [`Llama32Runner`] and validates the GGUF is `llama`-arch so a
/// misrouted file — including the Qwen3.6-hybrid Bonsai-27B — fails
/// loudly at `build()` instead of silently producing garbage.
///
/// For `prism-ml/Bonsai-27B` (`qwen35` arch) use
/// [`rlx_qwen35::Qwen35Runner`] directly, or drive the CLI via
/// [`cli_run`] / `rlx-run bonsai`, which auto-dispatches.
pub struct BonsaiRunner {
    inner: Llama32Runner,
    config: LlamaBaseConfig,
}

impl BonsaiRunner {
    pub fn builder() -> BonsaiRunnerBuilder {
        BonsaiRunnerBuilder::default()
    }

    /// Borrow the parsed LLaMA-base config (dims, RoPE, GQA, etc.).
    pub fn config(&self) -> &LlamaBaseConfig {
        &self.config
    }

    /// Read-only access to the underlying Llama-3.2 runner for advanced
    /// callers that need to drive prefill/decode directly.
    pub fn inner(&self) -> &Llama32Runner {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut Llama32Runner {
        &mut self.inner
    }

    /// Packed-decode generation. Mirrors
    /// [`Llama32Runner::generate_packed`] so callers don't need to
    /// reach through `.inner_mut()` for the common case.
    pub fn generate_packed(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        self.inner.generate_packed(prompt_ids, n_new, on_token)
    }
}

/// Builder. Same surface as [`Llama32RunnerBuilder`] but builds a
/// [`BonsaiRunner`] and enforces the `llama` arch tag.
#[derive(Debug, Clone, Default)]
pub struct BonsaiRunnerBuilder {
    weights: Option<PathBuf>,
    inner: Llama32RunnerBuilder,
}

impl BonsaiRunnerBuilder {
    pub fn weights(mut self, path: impl Into<PathBuf>) -> Self {
        let p: PathBuf = path.into();
        self.weights = Some(p.clone());
        self.inner = self.inner.weights(p);
        self
    }

    pub fn max_seq(mut self, n: usize) -> Self {
        self.inner = self.inner.max_seq(n);
        self
    }

    pub fn packed_weights(mut self, on: bool) -> Self {
        self.inner = self.inner.packed_weights(on);
        self
    }

    /// Build the runner. Returns `Err` when the GGUF isn't
    /// `general.architecture = llama` — including the Qwen3.6-hybrid
    /// Bonsai-27B, which is routed to `rlx-qwen35` instead.
    pub fn build(self) -> Result<BonsaiRunner> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weights path required (call .weights(...))"))?
            .clone();
        // Cheap header sniff first (avoids slurping a multi-GB file just
        // to reject it) — and gives a targeted hint for Bonsai-27B.
        match detect_arch(&weights)? {
            BonsaiArch::LlamaSmall => {}
            BonsaiArch::Qwen35Hybrid => bail!(
                "rlx-bonsai: {weights:?} is a Qwen3.6-hybrid Bonsai-27B \
                 (general.architecture = qwen35); expected the Llama-shaped Bonsai 1.7B–8B here. \
                 Run it through the qwen35 runner instead — `rlx-run qwen35 --weights …`, or \
                 `rlx-run bonsai …` which auto-dispatches."
            ),
        }
        let config = LlamaBaseConfig::from_gguf_path(&weights)
            .with_context(|| format!("rlx-bonsai: parse {weights:?}"))?;
        let inner = self
            .inner
            .build()
            .context("rlx-bonsai: building underlying Llama32Runner")?;
        Ok(BonsaiRunner { inner, config })
    }
}

/// Extract the value following `--weights` (both `--weights PATH` and
/// `--weights=PATH` forms).
fn weights_arg(args: &[String]) -> Option<&str> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--weights" {
            return it.next().map(String::as_str);
        }
        if let Some(rest) = a.strip_prefix("--weights=") {
            return Some(rest);
        }
    }
    None
}

/// CLI entry point. Sniffs the `--weights` GGUF architecture and routes:
/// `llama` → [`rlx_llama32::cli::run`] (Bonsai 1.7B–8B), `qwen35` →
/// [`rlx_qwen35::cli::run`] (`prism-ml/Bonsai-27B`). With no `--weights`
/// (e.g. `--help`) it falls back to the Llama runner so `--help` works.
pub fn cli_run(args: &[String]) -> Result<()> {
    match weights_arg(args) {
        Some(path) => match detect_arch(Path::new(path))? {
            BonsaiArch::LlamaSmall => rlx_llama32::cli::run(args),
            BonsaiArch::Qwen35Hybrid => rlx_qwen35::cli::run(args),
        },
        None => rlx_llama32::cli::run(args),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_tag_classification() {
        assert_eq!(bonsai_arch_from_tag("llama"), Some(BonsaiArch::LlamaSmall));
        assert_eq!(
            bonsai_arch_from_tag("qwen35"),
            Some(BonsaiArch::Qwen35Hybrid)
        );
        assert_eq!(
            bonsai_arch_from_tag("qwen36moe"),
            Some(BonsaiArch::Qwen35Hybrid)
        );
        assert_eq!(bonsai_arch_from_tag("qwen2"), None);
        assert_eq!(bonsai_arch_from_tag("gemma3"), None);
    }

    #[test]
    fn weights_arg_forms() {
        let a: Vec<String> = ["--prompt", "hi", "--weights", "m.gguf", "--packed"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(weights_arg(&a), Some("m.gguf"));
        let b: Vec<String> = ["--weights=x.gguf"].iter().map(|s| s.to_string()).collect();
        assert_eq!(weights_arg(&b), Some("x.gguf"));
        let c: Vec<String> = ["--help"].iter().map(|s| s.to_string()).collect();
        assert_eq!(weights_arg(&c), None);
    }
}
