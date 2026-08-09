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

//! Carbon — [HuggingFaceBio](https://huggingface.co/HuggingFaceBio) decoder-only
//! autoregressive **DNA** language models (500M / 3B / 8B).
//!
//! Carbon is a stock `LlamaForCausalLM` (GQA + RoPE θ=500000 + SwiGLU +
//! RMSNorm, tied embeddings) with a large hybrid vocabulary, so the transformer
//! backbone reuses [`rlx_llama32::Llama32Runner`] verbatim. The distinctive
//! part is the tokenizer: a **Qwen3 byte-level BPE** for text combined with an
//! **algorithmic DNA 6-mer** vocabulary — nucleotide runs wrapped in
//! `<dna>…</dna>` are split into non-overlapping 6-mers (each ≈ 6 bp) and mapped
//! to a fixed id table above the BPE range. This crate ports that
//! `HybridDNATokenizer` natively (see `HybridDnaTokenizer`).
//!
//! ```no_run
//! # #[cfg(feature = "tokenizer")]
//! # fn demo() -> anyhow::Result<()> {
//! use rlx_carbon::CarbonRunner;
//! use rlx_runtime::Device;
//!
//! let mut carbon = CarbonRunner::from_pretrained("/path/to/Carbon-500M", Device::Cpu)?;
//! // Feed a raw nucleotide sequence as a DNA region and continue it.
//! let out = carbon.complete("ATCGATCGATCGATCG", 64, Some(true))?;
//! println!("{}", out.text);
//! # Ok(()) }
//! ```

pub mod cli;
pub mod dna_config;
pub mod runner;
#[cfg(feature = "tokenizer")]
pub mod tokenizer;

pub use dna_config::{DnaConfig, DnaRegion, parse_dna_region, split_by_dna_tags};
#[cfg(feature = "tokenizer")]
pub use runner::CarbonCompletion;
pub use runner::{CarbonRunner, CarbonRunnerBuilder};
#[cfg(feature = "tokenizer")]
pub use tokenizer::HybridDnaTokenizer;

/// Re-export the sampling knobs so callers need not depend on rlx-llama32.
pub use rlx_llama32::SampleOpts;

/// Human-readable family label.
pub const FAMILY: &str = "Carbon";

/// Hugging Face model id for the 500M draft checkpoint.
pub const HF_MODEL_ID_500M: &str = "HuggingFaceBio/Carbon-500M";
