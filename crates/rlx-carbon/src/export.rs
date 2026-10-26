// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! Export a Carbon checkpoint directory as an `.rlxp` package rlx runs
//! alone: the Llama backbone as the Qwen3 / Llama exporter builds it (its
//! `generate` entry, token ids in and out), and a program answering
//! `generate` with Carbon's hybrid tokenizer around that entry — text
//! through the package's Qwen3 BPE, nucleotides through the 6-mer table
//! ([`crate::hybrid`], [`crate::dna_config`], built from this crate's own
//! files), a DNA prompt opened as [`crate::CarbonRunner::complete`] opens it.
//!
//! `Model.generate_text("ATGGCGACC…", {max_tokens, as_dna})` returns the
//! continuation as `CarbonRunner::complete` decodes it.

use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::program;
pub use rlx_core::export::{ExportOptions, ExportReport};
use serde_json::json;

use crate::dna_config::DnaConfig;

/// Export the Carbon directory `dir` (`config.json`, `model.safetensors`,
/// `tokenizer.json` — Qwen3's, which the release leaves to its base —
/// `dna_config.json`, `generation_config.json`) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let tok = crate::HybridDnaTokenizer::from_dir(dir)
        .with_context(|| format!("{}: Carbon's tokenizer", dir.display()))?;
    let dna: DnaConfig = tok.dna_config().clone();
    let eos = tok.eos_id();
    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))?;
    rlx_qwen3::export::export_rlxp_with(dir, out, opts, &mut |pkg| {
        let tokenizer = pkg.model["entries"]["generate"]["tokenizer"].clone();
        anyhow::ensure!(
            tokenizer.is_string(),
            "the backbone's package has no generate entry with a tokenizer"
        );
        let config = json!({
            "tokenizer": tokenizer,
            "dna": serde_json::to_value(&dna)?,
            "eos": eos,
        });
        pkg.weights_as_stored();
        program::entry(
            pkg,
            "complete",
            module.clone(),
            &["generate"],
            "text-generation",
            config,
        );
        pkg.metadata("dna_start_id", json!(dna.dna_start_id));
        Ok(())
    })
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]),
/// taken by name (`--family carbon`): its `config.json` says `llama`.
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("carbon", "text-generation", export_rlxp));
}
